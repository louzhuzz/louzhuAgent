# 第二十一步：RAG v1 检索后拼接上下文回答

## 1. 本节目标

这一节要把你前面完成的这些部分第一次真正串起来：

- 文档加载
- 文本切分
- Embedding
- 向量存储
- Retriever
- 模型回答

本节新增的示例文件是：

- [examples/rag_v1_intro.py](/mnt/d/AIcodes/Agent/examples/rag_v1_intro.py)

## 2. 是什么

RAG v1 的最小闭环可以概括成一句话：

**先检索相关资料，再把资料拼进上下文，让模型基于资料回答。**

它的基本流程是：

1. 用户提问
2. 检索相关文档片段
3. 把检索结果拼成参考上下文
4. 把“问题 + 上下文”一起发给模型
5. 模型基于参考资料生成回答

## 3. 为什么

如果你直接把问题丢给模型，模型可能会：

- 依赖训练记忆
- 进行合理猜测
- 产生幻觉

而加入检索后，模型的回答会更偏向：

- 基于你自己的资料
- 可追溯
- 可解释

这就是 RAG 的核心价值：

**不是让模型更聪明，而是让模型更有依据。**

## 4. 怎么做

### 第一步：先完成检索

这一节直接复用前面的 Retriever：

- [examples/retriever_intro.py](/mnt/d/AIcodes/Agent/examples/retriever_intro.py)

先拿到最相关的 3 个文档片段。

### 第二步：把检索结果拼成上下文

本节新增了：

```python
build_context(documents)
```

它会把多个文档片段整理成一个大字符串，并保留来源信息。

### 第三步：构造 RAG Prompt

我们新增了：

```python
build_rag_prompt(question, context)
```

它负责明确告诉模型：

- 只能基于参考资料回答
- 如果资料不足要明确说出来

### 第四步：调用模型回答

最后再把：

- 用户问题
- 检索上下文

一起送给模型。

这时模型就不再是“裸答”，而是“检索增强生成”。

## 5. 关键代码

### 代码 1：检索相关文档

文件：[examples/rag_v1_intro.py](/mnt/d/AIcodes/Agent/examples/rag_v1_intro.py)

```python
retriever = SimpleRetriever(store, top_k=3)
relevant_docs = retriever.get_relevant_documents(question)
```

这一步负责：

- 把用户问题送进 Retriever
- 拿到最相关文档片段

### 代码 2：构造上下文

文件：[examples/rag_v1_intro.py](/mnt/d/AIcodes/Agent/examples/rag_v1_intro.py)

```python
def build_context(documents: list) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        content = document.page_content.strip()
        parts.append(f"[片段 {index}] 来源: {source}\n{content}")
```

这段代码的作用是：

- 把多个检索结果变成一个统一上下文
- 同时保留来源信息

这是后面做引用回答的重要基础。

### 代码 3：RAG Prompt

文件：[examples/rag_v1_intro.py](/mnt/d/AIcodes/Agent/examples/rag_v1_intro.py)

```python
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
你是一个基于本地知识库回答问题的学习助理。
...
参考资料：
{context}
...
用户问题：
{question}
""".strip()
```

这一步的意义是：

- 把“参考资料”和“用户问题”一起组织进 Prompt
- 给模型明确边界

### 代码 4：模型回答

文件：[examples/rag_v1_intro.py](/mnt/d/AIcodes/Agent/examples/rag_v1_intro.py)

```python
response = model.invoke(prompt)
answer = parse_text_output(response.content)
```

这说明：

- 检索之后，最终仍然是模型负责生成回答
- 只是现在它有了参考依据

## 6. 常见错误

### 错误 1：检索到了文档，但没真正拼进 Prompt

后果：

- 看起来像做了 RAG
- 实际上模型还是裸答

### 错误 2：上下文拼接时不保留来源

后果：

- 回答时无法追溯依据

### 错误 3：不给模型“资料不足时要说明”的约束

后果：

- 资料不够时模型可能会自己脑补

## 7. 常见面试问题

### 问题 1：RAG 的最小闭环是什么？

回答要点：

- 用户问题
- 检索相关资料
- 把资料拼进上下文
- 模型基于上下文生成回答

深入追问：

- 为什么检索和生成必须都存在，才能叫 RAG？

### 问题 2：为什么不能只做检索，不做生成？

回答要点：

- 只检索只能返回片段
- 生成层负责整理、归纳、组织表达
- RAG 的价值在于“检索 + 生成”结合

深入追问：

- 哪些场景只检索就够，哪些场景必须做生成？

### 问题 3：RAG 为什么能降低幻觉？

回答要点：

- 模型回答时有明确参考资料
- 回答更受检索上下文约束
- 不是完全依赖模型内部记忆

深入追问：

- 为什么 RAG 不能彻底消灭幻觉？

### 问题 4：Retriever 和生成模型在 RAG 里的分工是什么？

回答要点：

- Retriever 负责找资料
- 模型负责基于资料生成回答
- 两者职责不同但必须协同

深入追问：

- 如果检索质量差，会如何影响最终回答？

## 8. 本节验收

运行：

```bash
python examples/rag_v1_intro.py
```

或者：

```bash
python examples/rag_v1_intro.py 什么是 Retriever
```

如果你能看到：

- 用户问题
- 检索到的参考片段
- 最终 RAG 回答

就说明这一节已经跑通了。

## 9. 这一节和后面有什么关系

这一步意味着你已经第一次真正跑通了：

**检索增强生成闭环。**

后面你可以继续做的增强包括：

- 更好的切分
- 更真实的 embedding
- 更真实的向量库
- 引用来源
- 分数展示
- RAG 评测

