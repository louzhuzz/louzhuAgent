import sys

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import load_settings
from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from output_parsers import parse_text_output
from retriever_intro import SimpleRetriever
from text_splitting_intro import split_documents
from vector_store_intro import InMemoryVectorStore


def build_context(documents: list[Document]) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        content = document.page_content.strip()
        parts.append(
            f"[片段 {index}] 来源: {source} | chunk_index: {chunk_index}\n{content}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(question: str, context: str) -> str:
    return f"""
你是一个基于本地知识库回答问题的学习助理。

请严格基于下面提供的参考资料回答问题。
如果参考资料不足以支持完整回答，请明确说明“参考资料不足”。

参考资料：
{context}

用户问题：
{question}

回答要求：
1. 使用中文简体
2. 优先基于参考资料作答
3. 尽量分点
4. 每一个核心结论后，尽量用 [片段X] 的形式标注依据
5. 如果某个结论没有足够依据，不要强行下结论
""".strip()


def print_sources(documents: list[Document]) -> None:
    print("\n来源清单:")
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        preview = document.page_content[:80].replace("\n", " ")
        print(f"[片段 {index}] {source} | chunk_index={chunk_index} | {preview}...")


def main() -> None:
    question = "什么是输出解析器，它在 LangChain 学习路径里有什么作用？"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip() or question

    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    store = InMemoryVectorStore()
    store.add_documents(chunks)
    retriever = SimpleRetriever(store, top_k=3)
    relevant_docs = retriever.get_relevant_documents(question)

    context = build_context(relevant_docs)

    settings = load_settings()
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.3,
    )

    prompt = build_rag_prompt(question, context)
    response = model.invoke(prompt)
    answer = parse_text_output(response.content)

    print(f"问题: {question}")
    print_sources(relevant_docs)
    print(f"\n带引用的 RAG 回答:\n{answer}")


if __name__ == "__main__":
    main()

