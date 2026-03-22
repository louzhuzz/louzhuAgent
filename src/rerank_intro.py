import re
import sys
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import load_settings
from output_parsers import parse_text_output
from real_retriever_helpers import build_real_vector_store


@dataclass
class RerankResult:
    """保存单个文档片段在重排前后的评分信息。"""

    document: Document
    vector_score: float
    overlap_score: float
    final_score: float


def extract_query_terms(text: str) -> set[str]:
    """提取查询中的英文词和中文关键词，用于做一个轻量级的重排信号。"""
    lowered = text.lower()
    english_terms = set(re.findall(r"[a-z0-9]+", lowered))
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)

    chinese_terms: set[str] = set()
    for block in chinese_blocks:
        cleaned = block.strip()
        if len(cleaned) >= 2:
            chinese_terms.add(cleaned)
        # 为了提升中文短语的可匹配性，再额外补一层二元切分。
        for index in range(len(cleaned) - 1):
            chinese_terms.add(cleaned[index : index + 2])

    return {term for term in english_terms | chinese_terms if term}


def compute_overlap_score(query: str, content: str) -> float:
    """计算查询关键词和文档内容的重叠比例，作为重排的第二信号。"""
    query_terms = extract_query_terms(query)
    if not query_terms:
        return 0.0

    lowered_content = content.lower()
    hit_count = sum(1 for term in query_terms if term in lowered_content)
    return hit_count / len(query_terms)


def rerank_documents(
    query: str,
    raw_results: list[tuple[Document, float]],
    top_k: int = 3,
    vector_weight: float = 0.7,
    overlap_weight: float = 0.3,
) -> list[RerankResult]:
    """结合向量分数和关键词重叠分数，对初始召回结果做二次排序。"""
    reranked: list[RerankResult] = []

    for document, vector_score in raw_results:
        overlap_score = compute_overlap_score(query, document.page_content)
        final_score = vector_score * vector_weight + overlap_score * overlap_weight
        reranked.append(
            RerankResult(
                document=document,
                vector_score=vector_score,
                overlap_score=overlap_score,
                final_score=final_score,
            )
        )

    reranked.sort(key=lambda item: item.final_score, reverse=True)
    return reranked[:top_k]


def build_context(results: list[RerankResult]) -> str:
    """把重排后的片段拼成最终发给聊天模型的 RAG 上下文。"""
    parts: list[str] = []
    for index, result in enumerate(results, start=1):
        source = result.document.metadata.get("source", "unknown")
        content = result.document.page_content.strip()
        parts.append(
            f"[片段 {index}] 来源: {source} | "
            f"vector_score={result.vector_score:.4f} | "
            f"overlap_score={result.overlap_score:.4f}\n"
            f"{content}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(question: str, context: str) -> str:
    """构造带重排上下文的 RAG 提示词。"""
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
4. 回答时可以引用 [片段X]
""".strip()


def print_raw_results(results: list[tuple[Document, float]]) -> None:
    """打印原始向量检索结果，方便和重排结果做对比。"""
    print("\n原始向量检索结果:")
    for index, (document, score) in enumerate(results, start=1):
        preview = document.page_content[:80].replace("\n", " ")
        source = document.metadata.get("source", "unknown")
        print(f"Top {index} | score={score:.4f} | source={source} | {preview}...")


def print_reranked_results(results: list[RerankResult]) -> None:
    """打印重排后的结果和最终分数。"""
    print("\n重排后的结果:")
    for index, result in enumerate(results, start=1):
        preview = result.document.page_content[:80].replace("\n", " ")
        source = result.document.metadata.get("source", "unknown")
        print(
            f"Top {index} | final={result.final_score:.4f} | "
            f"vector={result.vector_score:.4f} | overlap={result.overlap_score:.4f} | "
            f"source={source} | {preview}..."
        )


def main() -> None:
    """运行一个带轻量级重排逻辑的 RAG 示例。"""
    question = "什么是输出解析器，它在 LangChain 学习路径里有什么作用？"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip() or question

    settings = load_settings()
    vector_store, embedding_model, chunks = build_real_vector_store(settings)

    # 先多召回一些候选片段，再做二次排序。
    raw_results = vector_store.similarity_search_with_score(question, k=6)
    reranked_results = rerank_documents(question, raw_results, top_k=3)
    context = build_context(reranked_results)

    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.3,
    )
    response = model.invoke(build_rag_prompt(question, context))
    answer = parse_text_output(response.content)

    print(f"真实 embedding 模型: {embedding_model}")
    print(f"入库 chunk 数: {len(chunks)}")
    print(f"问题: {question}")
    print_raw_results(raw_results)
    print_reranked_results(reranked_results)
    print(f"\nRerank 后的 RAG 回答:\n{answer}")


if __name__ == "__main__":
    main()
