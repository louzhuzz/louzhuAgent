from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._exceptions import ArkBadRequestError

# 相当于重写了 LangChain 的 Embeddings 类，使其内部调用 Ark Runtime SDK 来获取真实的 embedding 向量。这个类实现了 embed_documents 和 embed_query 两个方法，分别用于批量文本和单条文本的向量化。它还包含了一些智能选择接口和错误处理的逻辑，以适应不同模型的能力和兼容性需求。
class ArkEmbeddings(Embeddings): 
    '''基于 Ark Runtime SDK 的 LangChain Embeddings 实现，支持智能选择文本接口或多模态接口，并在遇到不支持的错误时自动切换。'''
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 128):
        self.model = model
        self.client = Ark(api_key=api_key, base_url=base_url) # 注意: client是
        self.batch_size = batch_size

    def _use_multimodal_api(self) -> bool: # 根据模型名称判断是否优先使用多模态接口，目前以包含 "embedding-vision" 或 "seed-1.6-embedding" 的模型为例
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
            input=texts, # embeddings.create 的 input 是字符串列表，表示一批文本输入
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def _create_multimodal_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts: # multimodal_embeddings.create 的 input 是一个列表，列表内每个元素表示一个样本的多模态输入。对于当前场景，每个样本只有一段文本，因此列表内每个元素都是一个包含文本的字典。
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}], # 这里把文本包装成一个字典，表示这是一个文本类型的输入
                encoding_format="float",
            )
            vectors.append(response.data.embedding)
        return vectors

    def _should_retry_with_other_api(self, exc: ArkBadRequestError) -> bool:
        message = str(exc).lower()
        return "does not support this api" in message

    def _embed_batch(self, texts: list[str]) -> list[list[float]]: # 根据模型能力智能选择接口，并在遇到不支持的错误时自动切换
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
        ) # first_call 时 如果prefer_multimodal为True 则先调用多模态接口，遇到不支持的错误时再调用文本接口；如果 prefer_multimodal 为 False 则先调用文本接口，遇到不支持的错误时再调用多模态接口。
        # 为什么要有second_call? 因为不同模型可能支持不同的接口，某些模型虽然名字里包含 "embedding-vision" 或 "seed-1.6-embedding"，但实际上可能不支持多模态接口，这时就需要在捕获到不支持的错误后切换到文本接口。同样地，如果优先调用文本接口但遇到不支持的错误，也可以切换到多模态接口尝试。这样设计可以提高兼容性，适应不同模型的实际能力。

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
            vectors.extend(self._embed_batch(batch)) # 将每个批次的结果累积到 vectors 列表中，最后返回
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0] # embed_query 只处理单条文本，因此直接调用 _embed_batch 并取第一个结果返回即可
