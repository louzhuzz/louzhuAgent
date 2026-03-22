import sys

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import load_settings
from output_parsers import parse_text_output
from real_retriever_helpers import RealRetriever, build_real_vector_store


def build_context(documents: list[Document]) -> str:
    """把检索到的文档片段拼成可直接放进 Prompt 的上下文字符串。"""
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        content = document.page_content.strip()
        parts.append(f"[片段 {index}] 来源: {source}\n{content}")
    return "\n\n".join(parts)


def build_rag_prompt(question: str, context: str) -> str:
    """构造 RAG v1 的回答提示词。"""
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
4. 如有需要，可在回答中提到来源片段编号
""".strip()


def main() -> None:
    """运行真实 embedding 版的 RAG v1 示例。"""
    question = "什么是输出解析器，它在 LangChain 学习路径里有什么作用？"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip() or question

    settings = load_settings()
    vector_store, embedding_model, chunks = build_real_vector_store(settings)
    retriever = RealRetriever(vector_store, top_k=3)
    relevant_docs = retriever.get_relevant_documents(question)
    context = build_context(relevant_docs)

    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.3,
    )

    prompt = build_rag_prompt(question, context)
    response = model.invoke(prompt)
    answer = parse_text_output(response.content)

    print(f"真实 embedding 模型: {embedding_model}")
    print(f"入库 chunk 数: {len(chunks)}")
    print(f"问题: {question}")
    print("\n检索到的参考片段:")
    print(context)
    print(f"\nRAG 回答:\n{answer}")

if __name__ == "__main__":
    main()
