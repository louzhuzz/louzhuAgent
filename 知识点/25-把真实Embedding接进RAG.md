# 第二十五步：把真实 Embedding 接进 RAG

## 1. 本节目标

这一节的目标是把上一节已经跑通的：

- 真实 embedding
- LangChain 原生内存向量库

真正接回到完整的 RAG 问答链路里。

本节完成后，你的两个示例都会升级：

- [src/rag_v1_intro.py](/mnt/d/AIcodes/Agent/src/rag_v1_intro.py)
- [src/rag_v2_citations_intro.py](/mnt/d/AIcodes/Agent/src/rag_v2_citations_intro.py)

它们不再使用前面教学版的本地哈希向量检索，而是切到真实 embedding 检索。

## 2. 是什么

这一节本质上做的是一件事：

**把“真实检索链”接回“完整回答链”。**

你前面已经分别学过：

- 文档加载
- 文本切分
- embedding
- 向量存储
- Retriever
- RAG 回答

但前面的 RAG v1 / v2 示例，检索层还是教学型版本。现在这一节把它升级成真实版本，让“检索”和“回答”都走同一条更接近真实工程的链路。

## 3. 为什么

如果你只单独跑“真实向量检索”示例，你看到的是：

- Top 1 命中了哪个 chunk
- 得分是多少

但这还不等于一个完整可用的 RAG。

真正的 RAG 问答闭环应该是：

1. 用户提问
2. 用真实 embedding 检索相关 chunk
3. 把检索结果拼进 Prompt
4. 让聊天模型基于这些资料回答
5. 在需要时给出引用依据

所以这一节的意义是：

- 把“检索层”接回“生成层”
- 让你看到真实工程里的完整数据流

## 4. 怎么做

### 第一步：抽一个可复用的真实向量库构建函数

为了避免 `rag_v1_intro.py` 和 `rag_v2_citations_intro.py` 重复写同一套入库逻辑，我们新增了：

- [src/real_retriever_helpers.py](/mnt/d/AIcodes/Agent/src/real_retriever_helpers.py)

这个文件负责：

- 读取本地知识点文档
- 做文本切分
- 创建 `ArkEmbeddings`
- 创建 `InMemoryVectorStore`
- 把 chunk 入库

### 第二步：封装一个真实检索器

我们在同一个文件里新增了：

- `RealRetriever`

它的职责很简单：

- 接收一个真实向量库
- 提供 `get_relevant_documents(query)` 方法

这样做的好处是：

- `RAG v1 / v2` 可以继续保持“Retriever -> build_context -> build_prompt”这种清晰结构
- 不会把向量库实现细节直接散落到业务文件里

### 第三步：升级 `rag_v1_intro.py`

原来它是：

- 教学型向量库
- 教学型 Retriever

现在它会：

- 调 `build_real_vector_store(settings)`
- 用 `RealRetriever` 检索
- 打印真实 embedding 模型名和入库 chunk 数

### 第四步：升级 `rag_v2_citations_intro.py`

RAG v2 的目标是“带引用回答”，所以它也必须先建立在真实检索之上。

现在它同样会：

- 使用真实 embedding 检索
- 打印来源清单
- 让模型尽量用 `[片段X]` 标注依据

## 5. 关键代码

### 代码 1：构建真实向量库

文件：[src/real_retriever_helpers.py](/mnt/d/AIcodes/Agent/src/real_retriever_helpers.py)

```python
def build_real_vector_store(
    settings: Settings,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> tuple[InMemoryVectorStore, str, list[Document]]:
    embedding_model = require_embedding_model(settings)

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
    vector_store.add_documents(chunks)
    return vector_store, embedding_model, chunks
```

这段代码的数据流要看清楚：

1. `settings` 里取出 `ARK_EMBEDDING_MODEL`
2. 读取本地 Markdown 文档
3. 切成 chunk
4. `ArkEmbeddings` 负责把文本变成真实向量
5. `InMemoryVectorStore` 负责存和查
6. 返回：
   - 已构建好的向量库
   - 当前使用的真实 embedding 模型名
   - 所有入库 chunk

为什么返回 `chunks`？

因为业务层有时需要打印：

- 入库 chunk 数

这能帮助你判断：

- 文档是否真的被切分了
- 向量库是否真的完成了入库

### 代码 2：真实检索器

文件：[src/real_retriever_helpers.py](/mnt/d/AIcodes/Agent/src/real_retriever_helpers.py)

```python
class RealRetriever:
    """基于真实 embedding 和 LangChain 向量库的最小检索器。"""

    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.top_k)
```

这段代码很简单，但你要理解它的工程意义：

- 业务代码只关心“给我相关文档”
- 不关心底层是本地哈希向量库，还是 Ark + LangChain 向量库

这就是“抽象层”的价值。

### 代码 3：RAG v1 的升级点

文件：[src/rag_v1_intro.py](/mnt/d/AIcodes/Agent/src/rag_v1_intro.py)

```python
settings = load_settings()
vector_store, embedding_model, chunks = build_real_vector_store(settings)
retriever = RealRetriever(vector_store, top_k=3)
relevant_docs = retriever.get_relevant_documents(question)
```

这段代码说明：

- `rag_v1_intro.py` 不再自己负责向量入库
- 它只负责：
  - 读取用户问题
  - 拿检索结果
  - 拼 Prompt
  - 调聊天模型生成答案

这就是一个更干净的职责拆分。

### 代码 4：RAG v2 的升级点

文件：[src/rag_v2_citations_intro.py](/mnt/d/AIcodes/Agent/src/rag_v2_citations_intro.py)

```python
print(f"真实 embedding 模型: {embedding_model}")
print(f"入库 chunk 数: {len(chunks)}")
print_sources(relevant_docs)
```

这部分虽然只是打印，但很重要。

它能帮助你在调试时明确区分：

- 是不是已经切到真实 embedding
- 入库数据够不够
- 当前回答依据的是哪些来源片段

## 6. 常见错误

### 错误 1：以为 RAG 升级只需要替换 embedding

不够。

你至少还要确认：

- 检索层是不是也切到了真实向量库
- RAG 脚本是不是确实在用新的 Retriever

### 错误 2：忘了配置 `ARK_EMBEDDING_MODEL`

这会导致真实向量库根本建不起来。

你需要确认 [`.env`](/mnt/d/AIcodes/Agent/.env) 里已经配置了：

```env
ARK_EMBEDDING_MODEL=你的真实 embedding 接入点或模型
```

### 错误 3：真实检索已经命中，但回答仍然不理想

这时问题往往不在 embedding，而在下面这些环节：

- chunk 切分质量
- top_k 设置
- Prompt 要求不够明确
- 引用格式约束不够强

### 错误 4：RAG v2 有来源清单，但回答里不一定严格标满所有引用

这很正常。

因为当前还是“提示模型尽量标注引用”，不是程序强制对每一句做证据绑定。

也就是说：

- 当前版本是“弱约束引用”
- 不是“程序级强约束 citation system”

## 7. 常见面试问题

### 问题 1：为什么要把真实 embedding 单独抽成辅助模块？

回答要点：

- 避免多个 RAG 脚本重复写入库逻辑
- 降低耦合
- 让业务层专注于“问答流程”，而不是“索引构建细节”

深入追问：

- 如果后面要把 `InMemoryVectorStore` 换成 `FAISS / Chroma / Milvus`，你希望改哪些层，哪些层保持不变？

### 问题 2：为什么 `Retriever` 这一层仍然值得保留？

回答要点：

- 因为业务代码要的是“相关文档”，不是“向量库细节”
- `Retriever` 是从“存储接口”到“业务接口”的一层抽象
- 这层抽象能让你后面加 rerank、过滤、查询改写而不破坏上层代码

### 问题 3：真实 embedding 接进 RAG 后，效果提升主要来自哪里？

回答要点：

- 主要来自更强的语义表示能力
- 相比教学型 embedding，更容易把“问题语义”和“文档语义”对齐
- 命中正确 chunk 的概率更高

深入追问：

- 如果已经命中了正确文档，但命中的 chunk 片段仍然不理想，你会优先调哪些参数？

## 8. 本节验收

你现在可以这样验证：

### 验证 1：运行真实 embedding 版 RAG v1

```bash
python src/rag_v1_intro.py
```

你应该能看到：

- 真实 embedding 模型名
- 入库 chunk 数
- 检索到的参考片段
- 最终 RAG 回答

### 验证 2：运行真实 embedding 版 RAG v2

```bash
python src/rag_v2_citations_intro.py
```

你应该能看到：

- 真实 embedding 模型名
- 来源清单
- 带引用倾向的回答

如果这两个都正常，这一节就算真正打通了：

**真实 embedding 检索 -> RAG 上下文拼接 -> 模型基于资料回答**
