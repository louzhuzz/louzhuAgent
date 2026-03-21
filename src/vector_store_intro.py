from dataclasses import dataclass

from langchain_core.documents import Document

from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from embedding_intro import cosine_similarity, embed_text
from text_splitting_intro import split_documents


@dataclass
class VectorRecord: # 一个向量记录，包含原始文档和对应的向量表示
    document: Document
    vector: list[float]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []

    def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self.records.append(
                VectorRecord(
                    document=document,
                    vector=embed_text(document.page_content),
                )
            )

    def similarity_search(self, query: str, top_k: int = 3) -> list[tuple[float, Document]]:
        query_vector = embed_text(query)
        scored: list[tuple[float, Document]] = []

        for record in self.records:
            score = cosine_similarity(query_vector, record.vector)
            scored.append((score, record.document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def main() -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    store = InMemoryVectorStore() # 创建一个内存中的向量库
    store.add_documents(chunks)

    query = "什么是输出解析器"
    results = store.similarity_search(query, top_k=3)

    print(f"向量库记录数: {len(store.records)}")
    print(f"查询: {query}")

    for rank, (score, document) in enumerate(results, start=1):
        preview = document.page_content[:100].replace("\n", " ")
        print(f"\nTop {rank}")
        print(f"score: {score:.4f}")
        print(f"metadata: {document.metadata}")
        print(f"preview: {preview}...")


if __name__ == "__main__":
    main()

