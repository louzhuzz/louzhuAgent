# 第二十四步：真实 Embedding 与更真实向量检索

## 1. 本节目标

这一节的目标是把你前面一直在用的：

- 教学型本地 embedding
- 手写内存向量库

升级成：

- 火山方舟官方 Ark SDK embedding 调用
- LangChain 原生内存向量库

本节新增的示例文件是：

- [src/real_embedding_vectorstore_intro.py](/mnt/d/AIcodes/Agent/src/real_embedding_vectorstore_intro.py)

## 2. 是什么

这一节做的升级包含两部分：

### 真实 embedding

我们不再用哈希桶近似向量化，而是用：

- `volcenginesdkarkruntime.Ark`

通过火山方舟官方 SDK 调用真实 embedding 模型。

### 更真实的向量检索

我们不再自己维护手写 `InMemoryVectorStore`，而是使用 LangChain 原生：

- `langchain_core.vectorstores.InMemoryVectorStore`

## 3. 为什么

你前面已经观察到了一个真实问题：

- 问“输出解析器”
- 结果却没稳定命中 [知识点/13-输出解析器.md](/mnt/d/AIcodes/Agent/知识点/13-输出解析器.md)

这不是单纯调 `chunk_size` 或 `top_k` 就一定能解决的。

很多时候核心问题是：

**文本表示能力太弱。**

所以这一节的核心意义是：

- 把“教学原理版”升级到“更接近真实工程版”

## 4. 怎么做

### 第一步：新增 embedding 模型配置

在 [`.env.example`](/mnt/d/AIcodes/Agent/.env.example) 中增加：

```env
ARK_EMBEDDING_MODEL=your_embedding_model_or_endpoint_id_here
```

这通常需要你在火山方舟准备一个可用的 embedding 模型或接入点。

### 第二步：在配置层中显式支持 embedding model

我们在 [src/config.py](/mnt/d/AIcodes/Agent/src/config.py) 中增加了：

- `embedding_model`
- `require_embedding_model(...)`

这样做的价值是：

- 聊天模型和 embedding 模型可以分开配置
- 错误提示更明确

### 第三步：用火山方舟官方 SDK 接真实向量模型

核心代码是：

```python
response = client.embeddings.create(
    model="文本 embedding 模型",
    input="需要向量化的文本内容",
    encoding_format="float"
)
```

不过为了继续对接 LangChain 的 `InMemoryVectorStore`，我们额外封装了一个：

- [src/ark_embeddings.py](/mnt/d/AIcodes/Agent/src/ark_embeddings.py)

它实现了 LangChain 的 `Embeddings` 接口，把官方 Ark SDK 调用适配成：

- `embed_documents(...)`
- `embed_query(...)`

如果你使用的是：

- `Doubao-embedding-vision`
- `Seed-1.6-Embedding`

这类多模态 embedding 模型，那么它们支持文本输入，但要走：

- `client.multimodal_embeddings.create(...)`

而不是普通的：

- `client.embeddings.create(...)`

当前代码已经自动根据模型名切换接口。

### 第四步：用 LangChain 原生 InMemoryVectorStore

核心代码是：

```python
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)
```

这说明：

- embedding 负责生成向量
- vector store 负责存和查

而且这次是真正通过 embedding 对文档做入库。

## 5. 关键代码

### 代码 1：embedding 模型配置

文件：[src/config.py](/mnt/d/AIcodes/Agent/src/config.py)

```python
embedding_model = os.getenv("ARK_EMBEDDING_MODEL", "").strip() or None
```

这段代码的作用：

- 让 embedding 模型和聊天模型配置解耦

### 代码 2：强制校验 embedding 配置

文件：[src/config.py](/mnt/d/AIcodes/Agent/src/config.py)

```python
def require_embedding_model(settings: Settings) -> str:
    if not settings.embedding_model:
        raise ValueError("缺少 ARK_EMBEDDING_MODEL，请先在 .env 中配置真实 embedding 模型或接入点。")
```

这一步的意义是：

- 避免你跑到一半才发现没配 embedding

### 代码 3：Ark SDK 适配成 LangChain Embeddings

文件：[src/ark_embeddings.py](/mnt/d/AIcodes/Agent/src/ark_embeddings.py)

```python
class ArkEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = Ark(api_key=api_key, base_url=base_url)
```

这段代码的意义是：

- 下层调用走火山方舟官方 SDK
- 上层接口仍然兼容 LangChain 向量库

这是这一课最关键的桥接层。

### 代码 4：批量向量化

文件：[src/ark_embeddings.py](/mnt/d/AIcodes/Agent/src/ark_embeddings.py)

```python
if self._use_multimodal_api():
    response = self.client.multimodal_embeddings.create(
        model=self.model,
        input=[{"type": "text", "text": text} for text in batch],
        encoding_format="float",
    )
else:
    response = self.client.embeddings.create(
        model=self.model,
        input=batch,
        encoding_format="float",
    )
```

这一步就是官方文档里的批量 embedding 调用，差别只是我们把它包进了 LangChain 适配器。

这里还多做了一层“自动分批”，因为真实向量库在入库时，往往会一次提交很多 chunk。

如果你不分批，就可能遇到类似错误：

- `array too long`
- `expected an array with maximum length 256`

### 代码 5：真实向量检索

文件：[src/real_embedding_vectorstore_intro.py](/mnt/d/AIcodes/Agent/src/real_embedding_vectorstore_intro.py)

```python
results = vector_store.similarity_search_with_score(query, k=3)
```

这一步说明：

- 你现在已经进入“真实 embedding + 原生 vector store”的检索链路

## 6. 常见错误

### 错误 1：以为聊天模型和 embedding 模型一定是同一个

很多平台里它们不是一个东西，甚至常常要分开配置。

### 错误 2：把聊天模型接入点当成 embedding 模型来填

这样通常会直接调用失败，或者拿不到正确向量结果。

### 错误 3：用多模态 embedding 模型，却仍然调用普通文本 embedding API

后果：

- 会报 `does not support this api`

这时应该改成：

- `/v3/embeddings/multimodal`

当前代码已经自动处理这个切换。

### 错误 4：只换向量库，不换 embedding

如果 embedding 表示能力本身很弱，只换存储层通常解决不了核心召回问题。

### 错误 5：已经成功生成向量，但检索阶段报 `numpy` 缺失

`langchain_core.vectorstores.InMemoryVectorStore` 在做余弦相似度计算时依赖 `numpy`。

典型报错是：

```text
ImportError: cosine_similarity requires numpy to be installed
```

解决方式：

```bash
pip install -r requirements.txt
```

或者单独安装：

```bash
pip install numpy
```

### 错误 5：拿不到结果就以为是代码问题

也有可能是：

- `ARK_EMBEDDING_MODEL` 配错
- 平台没有该模型权限
- 你填的是聊天模型接入点，不是 embedding 模型
- 你填的是多模态 embedding 模型，但代码没有走多模态接口

### 错误 6：一次批量提交太多文本

后果：

- 会触发平台的 `input array too long`
- 需要在 embedding 层做自动分批

## 7. 常见面试问题

### 问题 1：为什么真实 embedding 比教学型哈希向量化更有价值？

回答要点：

- 它更能表达语义相似性
- 能更真实反映检索质量
- 是生产级 RAG 的基础能力

深入追问：

- 为什么参数调优不能完全替代高质量 embedding？

### 问题 2：为什么 embedding 模型要单独配置？

回答要点：

- 聊天生成和向量表示是两类不同能力
- 平台常常分开提供
- 这有利于灵活替换和成本控制

深入追问：

- 真实系统里如何做聊天模型和 embedding 模型的组合选型？

### 问题 3：为什么这一步要做 Ark SDK 到 LangChain Embeddings 的适配层？

回答要点：

- 下层保持官方 SDK 调用方式
- 上层继续复用 LangChain 向量库和检索生态
- 降低后续切换和组合成本

深入追问：

- 为什么桥接层是做工程解耦的关键？

### 问题 4：为什么这一步选择 LangChain 原生 InMemoryVectorStore？

回答要点：

- 它比手写版更接近真实接口习惯
- 但仍然足够轻量，适合教学
- 后面迁移到 Chroma / Milvus 时思路更顺

深入追问：

- InMemoryVectorStore 和真正外部向量数据库差别在哪里？

### 问题 5：为什么这一步仍然不等于“完整生产级 RAG”？

回答要点：

- 还缺持久化
- 还缺更成熟的索引和过滤能力
- 还缺评测、监控、更新策略

深入追问：

- 下一步如果继续工程化，最应该补什么？

## 8. 本节验收

先在 [`.env.example`](/mnt/d/AIcodes/Agent/.env.example) 对应的真实 `.env` 中配置：

```env
ARK_EMBEDDING_MODEL=你的真实 embedding 模型或接入点
```

然后运行：

```bash
python src/real_embedding_vectorstore_intro.py
```

如果你能看到：

- 真实 embedding 模型名
- 入库 chunk 数
- Top 3 检索结果及分数

就说明这一节已经跑通了。

## 9. 这一节和后面有什么关系

这一步意味着你已经从：

- 教学型 embedding
- 教学型 vector store

走到了：

- 更真实的 embedding 检索链

这会直接影响后面这些内容的质量：

- Retriever
- RAG v1 / v2
- 参数调优
- 检索评测

## 10. 官方参考

这一节实现方式对齐的是火山方舟官方文档的 embedding 调用思路：

- 快速入门：https://www.volcengine.com/docs/82379/1399008?lang=zh
- 文本向量化：https://www.volcengine.com/docs/82379/1583857?lang=zh

## 11. 安装说明修正

如果你在安装火山方舟 Python SDK 时遇到：

```bash
pip install volcenginesdkarkruntime
```

报错找不到包，这通常不是你的环境问题，而是：

- `volcenginesdkarkruntime` 更像是导入路径
- 官方推荐的安装包名是：

```bash
pip install "volcengine-python-sdk[ark]"
```

如果镜像源里没有同步，可以改用官方 PyPI：

```bash
pip install -i https://pypi.org/simple "volcengine-python-sdk[ark]"
```
