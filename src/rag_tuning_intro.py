from dataclasses import dataclass

from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from retriever_intro import SimpleRetriever
from text_splitting_intro import split_documents
from vector_store_intro import InMemoryVectorStore


@dataclass
class TuningCase:
    name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def run_case(question: str, case: TuningCase) -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(
        documents,
        chunk_size=case.chunk_size,
        chunk_overlap=case.chunk_overlap,
    )

    store = InMemoryVectorStore()
    store.add_documents(chunks)
    retriever = SimpleRetriever(store, top_k=case.top_k)
    results = retriever.get_relevant_documents(question)

    print(f"\n=== {case.name} ===")
    print(
        f"参数: chunk_size={case.chunk_size}, "
        f"chunk_overlap={case.chunk_overlap}, top_k={case.top_k}"
    )
    print(f"chunk 总数: {len(chunks)}")

    for rank, document in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        preview = document.page_content[:80].replace("\n", " ")
        print(
            f"Top {rank}: {source} | chunk_index={chunk_index} | {preview}..."
        )


def main() -> None:
    question = "什么是输出解析器，它在 LangChain 学习路径里有什么作用？"
    print(f"问题: {question}")

    cases = [
        TuningCase("基线参数", chunk_size=300, chunk_overlap=50, top_k=3),
        TuningCase("更小 chunk", chunk_size=180, chunk_overlap=30, top_k=3),
        TuningCase("更大 overlap", chunk_size=300, chunk_overlap=120, top_k=3),
        TuningCase("更大 top_k", chunk_size=300, chunk_overlap=50, top_k=5),
    ]

    for case in cases:
        run_case(question, case)


if __name__ == "__main__":
    main()

