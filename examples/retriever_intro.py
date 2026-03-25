from langchain_core.documents import Document

from _bootstrap import setup_example_path

setup_example_path()

from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from text_splitting_intro import split_documents
from vector_store_intro import InMemoryVectorStore


class SimpleRetriever:
    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self.vector_store.similarity_search(query, top_k=self.top_k)
        return [document for _, document in results]


def main() -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    store = InMemoryVectorStore()
    store.add_documents(chunks)

    retriever = SimpleRetriever(store, top_k=3)
    query = "什么是输出解析器"
    results = retriever.get_relevant_documents(query)

    print(f"查询: {query}")
    print(f"返回文档数: {len(results)}")

    for rank, document in enumerate(results, start=1):
        preview = document.page_content[:100].replace("\n", " ")
        print(f"\nTop {rank}")
        print(f"metadata: {document.metadata}")
        print(f"preview: {preview}...")


if __name__ == "__main__":
    main()
