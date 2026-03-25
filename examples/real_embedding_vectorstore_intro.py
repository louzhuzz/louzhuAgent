from _bootstrap import setup_example_path
from langchain_core.vectorstores import InMemoryVectorStore

setup_example_path()

from ark_embeddings import ArkEmbeddings
from config import load_settings, require_embedding_model
from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from text_splitting_intro import split_documents


def main() -> None:
    settings = load_settings()
    embedding_model = require_embedding_model(settings)

    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    embeddings = ArkEmbeddings(
        model=embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    query = "什么是输出解析器，它在 LangChain 学习路径里有什么作用？"
    results = vector_store.similarity_search_with_score(query, k=3)

    print(f"真实 embedding 模型: {embedding_model}")
    print(f"入库 chunk 数: {len(chunks)}")
    print(f"查询: {query}")

    for rank, (document, score) in enumerate(results, start=1):
        preview = document.page_content[:100].replace("\n", " ")
        print(f"\nTop {rank}")
        print(f"score: {score:.4f}")
        print(f"metadata: {document.metadata}")
        print(f"preview: {preview}...")

if __name__ == "__main__":
    main()
