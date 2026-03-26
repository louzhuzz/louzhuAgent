import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ark_embeddings import ArkEmbeddings


class PersistentKnowledgeBase:
    """负责知识点向量的本地缓存与 ChromaDB 持久化存储。"""

    def __init__(
        self,
        read_note: Callable[[str], str],
        embeddings: ArkEmbeddings,
        persist_directory: str,
        embedding_model: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        """初始化持久化向量库、缓存清单路径和切分参数。"""
        self.read_note = read_note
        self.embeddings = embeddings
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = self._build_collection_name(embedding_model)
        self.manifest_path = self.persist_directory / f"{self.collection_name}_manifest.json"
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )
        self.manifest = self._load_manifest()

    def _build_collection_name(self, embedding_model: str) -> str:
        """根据 embedding 模型名生成稳定且可读的 collection 名称。"""
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", embedding_model).strip("-").lower()
        cleaned = cleaned[:48] or "default"
        return f"knowledge-qa-{cleaned}"

    def _load_manifest(self) -> dict[str, Any]:
        """读取本地缓存清单；不存在时创建一个空结构。"""
        if not self.manifest_path.exists():
            return {
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "files": {},
            }

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != self.embedding_model:
            # 这里理论上不会命中，因为不同模型会映射成不同 collection。
            # 仍然保留这层保护，避免后面改命名规则时错误复用旧缓存。
            return {
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "files": {},
            }

        return manifest

    def _save_manifest(self) -> None:
        """把当前缓存清单落盘，便于后续运行直接复用。"""
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _hash_text(self, text: str) -> str:
        """基于文件内容生成稳定 hash，用于判断是否需要重新 embedding。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_document_id(self, file_name: str, chunk_index: int, file_hash: str) -> str:
        """为每个 chunk 构造稳定唯一的文档 ID。"""
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", file_name).strip("-").lower()
        return f"{safe_name}-{file_hash[:12]}-{chunk_index}"

    def _split_note_to_documents(self, file_name: str, content: str) -> list[Document]:
        """把一篇知识点文件切成多个可检索 chunk。"""
        documents: list[Document] = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end].strip()
            if chunk_text:
                documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "file_name": file_name,
                            "chunk_index": chunk_index,
                            "chunk_start": start,
                            "chunk_end": end,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(content):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return documents

    def ensure_note_indexed(self, file_name: str) -> dict[str, Any]:
        """确保单篇知识点文件已经写入本地 ChromaDB，并尽量复用旧向量。"""
        content = self.read_note(file_name)
        file_hash = self._hash_text(content)
        existing = self.manifest["files"].get(file_name)

        if (
            existing
            and existing.get("file_hash") == file_hash
            and existing.get("chunk_size") == self.chunk_size
            and existing.get("chunk_overlap") == self.chunk_overlap
        ):
            return {
                "file_name": file_name,
                "status": "cache_hit",
                "chunk_count": existing.get("chunk_count", 0),
            }

        # 文件内容变了，先删除这篇文件在旧索引里的所有 chunk，避免旧向量残留。
        if existing and existing.get("doc_ids"):
            self.vector_store.delete(ids=existing["doc_ids"])

        documents = self._split_note_to_documents(file_name, content)
        doc_ids = [
            self._make_document_id(file_name, index, file_hash)
            for index, _ in enumerate(documents)
        ]

        if documents:
            self.vector_store.add_documents(documents=documents, ids=doc_ids)

        self.manifest["files"][file_name] = {
            "file_hash": file_hash,
            "doc_ids": doc_ids,
            "chunk_count": len(documents),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        self._save_manifest()

        return {
            "file_name": file_name,
            "status": "reindexed",
            "chunk_count": len(documents),
        }

    def search(
        self,
        query: str,
        file_names: list[str],
        per_file_k: int = 3,
        final_k: int = 4,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """只在指定候选文件范围内检索，并返回命中的 chunk 与索引状态。"""
        if not file_names:
            raise ValueError("检索前至少要提供一个候选知识点文件。")

        index_statuses = [self.ensure_note_indexed(file_name) for file_name in file_names]

        merged_results: list[dict[str, Any]] = []
        for file_name in file_names:
            # 这里按单文件过滤检索，避免虽然库里已经持久化了很多文件，
            # 但本次查询仍然被限制在“小范围候选文件”内。
            results = self.vector_store.similarity_search_with_score(
                query,
                k=per_file_k,
                filter={"file_name": file_name},
            )
            for document, distance in results:
                merged_results.append(
                    {
                        "document": document,
                        "distance": float(distance),
                    }
                )

        merged_results.sort(key=lambda item: item["distance"])
        return merged_results[:final_k], index_statuses
