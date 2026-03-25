from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from _bootstrap import setup_example_path

setup_example_path()

from ark_embeddings import ArkEmbeddings
from config import Settings, require_embedding_model
from document_loading_intro import KNOWLEDGE_DIR, load_markdown_documents
from text_splitting_intro import split_documents


class RealRetriever:
    """基于真实 embedding 和 LangChain 向量库的最小检索器。"""

    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 3):
        """保存向量库实例和默认返回条数。"""
        self.vector_store = vector_store
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        """根据查询语句返回最相关的文档片段列表。"""
        return self.vector_store.similarity_search(query, k=self.top_k)


def build_real_vector_store(
    settings: Settings, # settings 参数包含了构建 ArkEmbeddings 所需的配置信息，如 API key、base URL 和模型名称等。
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> tuple[InMemoryVectorStore, str, list[Document]]: # 返回值第一项是构建好的向量库实例，第二项是实际使用的 embedding 模型名称，第三项是入库的 chunk 列表
    """构建基于真实 embedding 的向量库，并返回入库后的 chunk 列表。"""
    embedding_model = require_embedding_model(settings) # 从 settings 中获取 embedding 模型名称，如果没有配置则抛出异常提示用户配置真实 embedding 模型或接入点。

    documents = load_markdown_documents(KNOWLEDGE_DIR)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embeddings = ArkEmbeddings(
        model=embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    vector_store = InMemoryVectorStore(embeddings) 
    # InMemoryVectorStore 接收一个 embedding 实例作为参数，构建时会使用该 embedding 实例来计算文档的向量表示。这里我们传入了 ArkEmbeddings 的实例，它会根据配置调用真实的 embedding 模型进行向量计算。
    vector_store.add_documents(chunks) # 将切分好的文档片段列表添加到向量库中，向量库会自动计算每个文档片段的向量表示并存储起来，以便后续的相似度搜索使用。
    return vector_store, embedding_model, chunks
