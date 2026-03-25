# 第二十步：Retriever

## 1. 本节目标

这一节要在前面的向量存储之上，再抽出一层：

- Retriever

本节新增的示例文件是：

- [examples/retriever_intro.py](/mnt/d/AIcodes/Agent/examples/retriever_intro.py)

## 2. 是什么

Retriever 可以先简单理解成：

**一个专门负责“给我相关文档”的高层接口。**

它不强调内部怎么存、怎么算，而强调：

- 输入一个查询
- 输出一组相关文档

## 3. 为什么

前一课的 `VectorStore` 已经能做：

- 存文档
- 存向量
- 相似度检索

但它仍然是偏底层的数据层。

而在 RAG 工作流里，业务层通常更关心的是：

- 给我最相关的文档

而不是：

- 你内部怎么存向量
- 你怎么排序

所以 Retriever 的价值是：

**把底层存储和上层调用解耦。**

## 4. 怎么做

### 第一步：保留 `VectorStore` 作为底层

这一节没有推翻前一课的实现。

我们继续复用：

- [examples/vector_store_intro.py](/mnt/d/AIcodes/Agent/examples/vector_store_intro.py)

### 第二步：定义一个更高层的检索类

我们新增：

```python
class SimpleRetriever:
```

它只关心两件事：

- 持有一个向量库
- 知道默认返回多少条结果

### 第三步：封装 `get_relevant_documents`

这一节最关键的方法是：

```python
def get_relevant_documents(self, query: str) -> list[Document]:
```

这意味着从这一层开始，对外暴露的已经不是：

- 分数 + 文档

而是更贴近业务的：

- 文档列表

### 第四步：把查询逻辑藏在内部

Retriever 内部仍然会调用：

- `vector_store.similarity_search(...)`

但调用方不需要关心这些细节。

## 5. 关键代码

### 代码 1：Retriever 定义

文件：[examples/retriever_intro.py](/mnt/d/AIcodes/Agent/examples/retriever_intro.py)

```python
class SimpleRetriever:
    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k
```

这说明 Retriever 的职责不是重新实现向量库，而是：

- 在向量库之上提供更高层的检索入口

### 代码 2：Retriever 对外接口

文件：[examples/retriever_intro.py](/mnt/d/AIcodes/Agent/examples/retriever_intro.py)

```python
def get_relevant_documents(self, query: str) -> list[Document]:
    results = self.vector_store.similarity_search(query, top_k=self.top_k)
    return [document for _, document in results]
```

这段代码就是本节最核心的抽象动作。

它说明：

- VectorStore 返回更底层结果
- Retriever 返回更业务化结果

### 代码 3：调用方变简单

文件：[examples/retriever_intro.py](/mnt/d/AIcodes/Agent/examples/retriever_intro.py)

```python
retriever = SimpleRetriever(store, top_k=3)
results = retriever.get_relevant_documents(query)
```

这两行代码就体现了 Retriever 的价值：

- 调用层更清晰
- 不需要反复关心相似度实现细节

## 6. 常见错误

### 错误 1：以为 Retriever 和 VectorStore 是同一个东西

不是。

它们的职责不同：

- VectorStore：偏底层，负责存和查
- Retriever：偏上层，负责给业务返回相关文档

### 错误 2：Retriever 里又重写一遍向量检索逻辑

这样会让分层失效。

Retriever 更合理的做法是：

- 调用向量存储层
- 做适度包装

### 错误 3：过早把 Retriever 做得很复杂

当前阶段最重要的是理解边界，不是一次把所有过滤、重排、混合检索都塞进去。

## 7. 常见面试问题

### 问题 1：Retriever 的核心职责是什么？

回答要点：

- 提供更高层的文档检索接口
- 屏蔽底层向量存储实现细节
- 为 RAG 流程提供“相关文档列表”

深入追问：

- 为什么 Retriever 不应该直接承担 embedding 和存储职责？

### 问题 2：Retriever 和 VectorStore 的区别是什么？

回答要点：

- VectorStore 负责保存和相似度查询
- Retriever 负责面向业务暴露检索接口
- Retriever 通常建立在 VectorStore 之上

深入追问：

- 如果更换底层向量库，Retriever 层为什么更稳定？

### 问题 3：为什么 Retriever 对外通常返回文档而不是分数？

回答要点：

- 上层业务往往更关心文档内容本身
- 分数可以是内部细节
- 如果需要，也可以在后续扩展中保留分数

深入追问：

- 哪些场景下应该把分数也返回给上层？

### 问题 4：为什么说 Retriever 是 RAG 流程的重要边界？

回答要点：

- 它把“检索实现细节”和“生成层使用方式”隔开
- 后面生成层只需要吃相关文档
- 这有利于系统模块化

深入追问：

- 如果后面接 reranker，应该放在 Retriever 哪一侧？

## 8. 本节验收

运行：

```bash
python examples/retriever_intro.py
```

如果你能看到：

- 查询文本
- 返回文档数
- Top 结果的 metadata 和预览

就说明这一节已经跑通了。

## 9. 这一节和后面有什么关系

这一步是后面这些内容的直接前置：

- RAG 检索链
- 上下文拼接
- 来源引用

因为从这一层开始，生成模块终于可以通过一个更干净的接口拿到“相关文档”了。

