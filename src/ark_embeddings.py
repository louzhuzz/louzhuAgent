from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkBadRequestError


class ArkEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 128):
        self.model = model
        self.client = Ark(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size

    def _use_multimodal_api(self) -> bool:
        model_name = self.model.lower()
        return "embedding-vision" in model_name or "seed-1.6-embedding" in model_name

    def _batched(self, texts: list[str]) -> list[list[str]]:
        # multimodal_embeddings.create 的 input 表示单个样本内的多模态内容，
        # 对当前“文搜文”场景来说，一次请求应只发送一段文本，避免把多条文本融合成一个向量。
        batch_size = 1 if self._use_multimodal_api() else self.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        return [
            texts[index : index + batch_size]
            for index in range(0, len(texts), batch_size)
        ]

    def _create_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def _create_multimodal_embeddings(self, texts: list[str]) -> list[list[float]]:
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
        message = str(exc).lower()
        return "does not support this api" in message

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
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
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in self._batched(texts):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
