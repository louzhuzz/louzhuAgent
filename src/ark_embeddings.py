from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkBadRequestError


class ArkEmbeddings(Embeddings):
    """基于火山方舟 Ark Runtime SDK 的 LangChain Embeddings 适配层。"""

    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 128):
        """保存 embedding 模型配置，并初始化 Ark 客户端。"""
        self.model = model
        self.client = Ark(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size

    def _use_multimodal_api(self) -> bool:
        """根据模型名判断是否优先走多模态 embedding 接口。"""
        model_name = self.model.lower()
        return "embedding-vision" in model_name or "seed-1.6-embedding" in model_name

    def _batched(self, texts: list[str]) -> list[list[str]]:
        """把文本列表按批次切分，避免超过接口限制。"""
        batch_size = 1 if self._use_multimodal_api() else self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")

        return [
            texts[index : index + batch_size]
            for index in range(0, len(texts), batch_size)
        ]

    def _create_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """调用普通文本 embedding 接口。"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def _create_multimodal_embeddings(self, texts: list[str]) -> list[list[float]]:
        """调用多模态 embedding 接口，并把纯文本包装成 text 输入。"""
        vectors: list[list[float]] = []
        for text in texts:
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}],
                encoding_format="float",
            )
            vectors.append(response.data.embedding)
        return vectors

    def _should_retry_with_other_api(self, exc: ArkBadRequestError) -> bool:
        """判断当前报错是否值得切换到另一套接口重试。"""
        return "does not support this api" in str(exc).lower()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """根据模型能力选择接口，并在必要时自动回退。"""
        prefer_multimodal = self._use_multimodal_api()
        first_call = (
            self._create_multimodal_embeddings
            if prefer_multimodal
            else self._create_text_embeddings
        )
        second_call = (
            self._create_text_embeddings
            if prefer_multimodal
            else self._create_multimodal_embeddings
        )

        try:
            return first_call(texts)
        except ArkBadRequestError as exc:
            if self._should_retry_with_other_api(exc):
                return second_call(texts)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把文档文本转成向量。"""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in self._batched(texts):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """把查询文本转成向量。"""
        return self._embed_batch([text])[0]
