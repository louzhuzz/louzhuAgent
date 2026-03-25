import hashlib
import math
import re

from _bootstrap import setup_example_path

setup_example_path()

from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from text_splitting_intro import split_documents


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE)] # 使用正则表达式提取单词，并转换为小写


def embed_text(text: str, dimension: int = 32) -> list[float]:
    if dimension <= 0:
        raise ValueError("dimension 必须大于 0。")

    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest() # 计算 token 的 MD5 哈希值
        index = int(digest, 16) % dimension # 将哈希值映射到向量维度范围内
        vector[index] += 1.0 # 简单地使用 token 出现的频率作为向量值

    norm = math.sqrt(sum(value * value for value in vector)) # 计算向量的 L2 范数
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("向量维度必须一致。")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def rank_chunks_by_query(query: str, chunk_texts: list[str], top_k: int = 3) -> list[tuple[int, float, str]]: # 返回 (chunk_index, score, chunk_text) 的列表
    query_vector = embed_text(query)
    scored: list[tuple[int, float, str]] = []

    for index, chunk_text in enumerate(chunk_texts):
        score = cosine_similarity(query_vector, embed_text(chunk_text)) # 计算查询向量与 chunk 向量之间的余弦相似度
        scored.append((index, score, chunk_text))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def main() -> None:
    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    query = "什么是 LangChain 的输出解析器"
    top_results = rank_chunks_by_query(query, [chunk.page_content for chunk in chunks], top_k=3)

    print(f"查询: {query}")
    print(f"chunk 总数: {len(chunks)}")

    for rank, (index, score, chunk_text) in enumerate(top_results, start=1):
        preview = chunk_text[:100].replace("\n", " ")
        print(f"\nTop {rank}")
        print(f"chunk_index: {index}")
        print(f"score: {score:.4f}")
        print(f"preview: {preview}...")


if __name__ == "__main__":
    main()
