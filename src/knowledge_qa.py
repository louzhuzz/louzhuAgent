import re
from dataclasses import dataclass
from typing import Callable

from ark_embeddings import ArkEmbeddings
from chroma_knowledge_base import PersistentKnowledgeBase


@dataclass
class KnowledgeQARequest:
    """描述一次知识点问答请求。"""

    question: str
    max_notes: int = 3
    max_chunks: int = 4


@dataclass
class NoteCandidate:
    """保存候选知识点文件及其打分结果。"""

    file_name: str
    score: float


class KnowledgeQAService:
    """主项目里的“少量候选文件 + embedding 检索 + ChromaDB 持久化”问答服务。"""

    def __init__(
        self,
        list_notes: Callable[[], list[str]],
        read_note: Callable[[str], str],
        invoke_text: Callable[[str, float], str],
        render_prompt: Callable[[str, str], str],
        api_key: str,
        base_url: str,
        embedding_model: str,
        chroma_persist_directory: str,
    ) -> None:
        """保存工具函数、模型调用函数，并初始化本地持久化知识库。"""
        self.list_notes = list_notes
        self.read_note = read_note
        self.invoke_text = invoke_text
        self.render_prompt = render_prompt
        self.embeddings = ArkEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model,
        )
        self.knowledge_base = PersistentKnowledgeBase(
            read_note=read_note,
            embeddings=self.embeddings,
            persist_directory=chroma_persist_directory,
            embedding_model=embedding_model,
        )

    def _extract_query_terms(self, text: str) -> set[str]:
        """提取问题中的英文词、中文短语和中文二元片段。"""
        english_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)

        chinese_terms: set[str] = set()
        for block in chinese_blocks:
            cleaned = block.strip()
            if len(cleaned) >= 2:
                chinese_terms.add(cleaned)
            for index in range(len(cleaned) - 1):
                chinese_terms.add(cleaned[index : index + 2])

        return {term for term in english_terms | chinese_terms if term}

    def _score_note_name(self, question: str, file_name: str) -> float:
        """根据问题关键词和文件名重叠程度给候选文件打分。"""
        query_terms = self._extract_query_terms(question)
        if not query_terms:
            return 0.0

        lowered_name = file_name.lower()
        hit_count = sum(1 for term in query_terms if term in lowered_name)
        return hit_count / len(query_terms)

    def _validate_request(self, request: KnowledgeQARequest) -> KnowledgeQARequest:
        """校验问答请求。"""
        request.question = request.question.strip()
        if not request.question:
            raise ValueError("问题不能为空。")
        if request.max_notes <= 0:
            raise ValueError("max_notes 必须大于 0。")
        if request.max_notes > 5:
            raise ValueError("为了控制上下文和 token，当前最多只处理 5 篇知识点。")
        if request.max_chunks <= 0:
            raise ValueError("max_chunks 必须大于 0。")
        if request.max_chunks > 8:
            raise ValueError("为了控制上下文和 token，当前最多只返回 8 个片段。")
        return request

    def select_notes(self, request: KnowledgeQARequest) -> list[NoteCandidate]:
        """从全部知识点文件中挑出少量最相关的候选文件。"""
        note_names = self.list_notes()
        if not note_names:
            raise ValueError("当前没有可用的知识点文件。")

        candidates = [
            NoteCandidate(
                file_name=file_name,
                score=self._score_note_name(request.question, file_name),
            )
            for file_name in note_names
        ]
        candidates.sort(key=lambda item: (item.score, item.file_name), reverse=True)
        return candidates[: request.max_notes]

    def retrieve_context(
        self,
        request: KnowledgeQARequest,
        candidates: list[NoteCandidate],
    ) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
        """只在候选文件范围内执行持久化检索，并返回命中片段与缓存状态。

        如果当前候选文件里没有找到相关片段，这里返回空上下文，由上层决定如何兜底。
        """
        results, index_statuses = self.knowledge_base.search(
            query=request.question,
            file_names=[candidate.file_name for candidate in candidates],
            per_file_k=max(2, request.max_chunks),
            final_k=request.max_chunks,
        )

        parts: list[str] = []
        retrieved_chunks: list[dict[str, object]] = []
        for index, item in enumerate(results, start=1):
            document = item["document"]
            distance = item["distance"]
            file_name = document.metadata.get("file_name", "unknown")
            parts.append(
                f"[资料 {index}] 文件: {file_name} | distance={distance:.4f}\n"
                f"{document.page_content}"
            )
            retrieved_chunks.append(
                {
                    "file_name": file_name,
                    "chunk_index": document.metadata.get("chunk_index"),
                    "chunk_start": document.metadata.get("chunk_start"),
                    "chunk_end": document.metadata.get("chunk_end"),
                    "distance": distance,
                }
            )
        return "\n\n".join(parts), retrieved_chunks, index_statuses

    def answer(self, request: KnowledgeQARequest) -> dict[str, object]:
        """执行一次“少量候选文件 + embedding 缓存 + ChromaDB 检索”的问答。"""
        validated_request = self._validate_request(request)
        candidates = self.select_notes(validated_request)
        context, retrieved_chunks, index_statuses = self.retrieve_context(validated_request, candidates)
        if not retrieved_chunks:
            return {
                "selected_notes": [candidate.file_name for candidate in candidates],
                "retrieved_chunks": [],
                "index_statuses": index_statuses,
                "context": "",
                "answer": "当前候选知识点中没有检索到足够相关的资料，请尝试把问题问得更具体，或先用 /read 查看相关笔记。",
            }
        prompt = self.render_prompt(validated_request.question, context)
        answer = self.invoke_text(prompt, 0.2)
        return {
            "selected_notes": [candidate.file_name for candidate in candidates],
            "retrieved_chunks": retrieved_chunks,
            "index_statuses": index_statuses,
            "context": context,
            "answer": answer,
        }
