import re
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ark_embeddings import ArkEmbeddings


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
    """主项目里的“少量候选文件 + embedding 检索”知识点问答服务。"""

    def __init__(
        self,
        list_notes: Callable[[], list[str]],
        read_note: Callable[[str], str],
        invoke_text: Callable[[str, float], str],
        render_prompt: Callable[[str, str], str],
        api_key: str,
        base_url: str,
        embedding_model: str,
    ) -> None:
        """保存工具函数、模型调用函数和 Prompt 渲染函数。"""
        self.list_notes = list_notes
        self.read_note = read_note
        self.invoke_text = invoke_text
        self.render_prompt = render_prompt
        self.embeddings = ArkEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=embedding_model,
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

    def _split_note_to_documents(self, file_name: str, content: str) -> list[Document]:
        """把单篇笔记切成少量 chunk，便于候选范围内做 embedding 检索。"""
        chunk_size = 500
        chunk_overlap = 80
        chunks: list[Document] = []
        start = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "file_name": file_name,
                            "chunk_start": start,
                            "chunk_end": end,
                        },
                    )
                )
            if end >= len(content):
                break
            start = max(end - chunk_overlap, start + 1)

        return chunks

    def build_candidate_documents(self, candidates: list[NoteCandidate]) -> list[Document]:
        """只对少量候选知识点文件构造文档片段。"""
        documents: list[Document] = []
        for candidate in candidates:
            note_content = self.read_note(candidate.file_name)
            documents.extend(self._split_note_to_documents(candidate.file_name, note_content))
        return documents

    def retrieve_context(
        self,
        request: KnowledgeQARequest,
        candidates: list[NoteCandidate],
    ) -> tuple[str, list[Document]]:
        """只对候选文件做 embedding 和向量检索，并返回命中的资料片段。"""
        documents = self.build_candidate_documents(candidates)
        if not documents:
            raise ValueError("候选知识点文件为空，无法执行问答。")

        vector_store = InMemoryVectorStore(self.embeddings)
        vector_store.add_documents(documents)
        results = vector_store.similarity_search_with_score(
            request.question,
            k=request.max_chunks,
        )

        retrieved_docs = [document for document, _ in results]
        parts: list[str] = []
        for index, (document, score) in enumerate(results, start=1):
            file_name = document.metadata.get("file_name", "unknown")
            parts.append(
                f"[资料 {index}] 文件: {file_name} | score={score:.4f}\n"
                f"{document.page_content}"
            )
        return "\n\n".join(parts), retrieved_docs

    def answer(self, request: KnowledgeQARequest) -> dict[str, object]:
        """执行一次“少量候选文件 + embedding 检索”的知识点问答。"""
        validated_request = self._validate_request(request)
        candidates = self.select_notes(validated_request)
        context, retrieved_docs = self.retrieve_context(validated_request, candidates)
        prompt = self.render_prompt(validated_request.question, context)
        answer = self.invoke_text(prompt, 0.2)
        return {
            "selected_notes": [candidate.file_name for candidate in candidates],
            "retrieved_chunks": [
                {
                    "file_name": document.metadata.get("file_name", "unknown"),
                    "chunk_start": document.metadata.get("chunk_start"),
                    "chunk_end": document.metadata.get("chunk_end"),
                }
                for document in retrieved_docs
            ],
            "context": context,
            "answer": answer,
        }
