# 第二十三步：RAG 参数调优与误差来源分析

## 1. 本节目标

这一节不追求新增“花哨功能”，而是训练你一个更重要的能力：

**当 RAG 效果不好时，知道问题到底出在哪一层。**

本节新增的示例文件是：

- [src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py)

## 2. 是什么

RAG 参数调优，指的是系统性观察这些参数如何影响检索结果：

- `chunk_size`
- `chunk_overlap`
- `top_k`

误差来源分析，则是判断问题可能来自：

- 文档切分
- embedding 质量
- 检索结果不足
- 生成阶段约束

## 3. 为什么

你刚才已经遇到了一个非常典型的真实问题：

- 问题是“什么是输出解析器”
- 但没检索到 [知识点/13-输出解析器.md](/mnt/d/AIcodes/Agent/知识点/13-输出解析器.md)
- 最终模型只能回答“参考资料不足”

这不是坏事，反而说明你已经进入了真正的 RAG 调优阶段。

因为做 RAG 最关键的能力之一，不是只会搭流程，而是会诊断：

- 是检索没命中？
- 是 chunk 切得不合适？
- 是 top_k 太小？
- 还是 embedding 太弱？

## 4. 怎么做

### 第一步：固定问题

这一节我们用一个真实失败案例做实验：

```text
什么是输出解析器，它在 LangChain 学习路径里有什么作用？
```

### 第二步：设计多组参数

我们在 [src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py) 中准备了多组实验：

- 基线参数
- 更小 chunk
- 更大 overlap
- 更大 top_k

### 第三步：逐组跑检索

每组实验都会输出：

- 当前参数
- chunk 总数
- Top 命中文档来源

这样你就能直观看到：

- 参数变化是否改变了召回结果

### 第四步：分析误差来源

如果多组参数都没能命中真正相关文档，就要开始怀疑：

- 当前 embedding 质量太弱

这一步非常重要，因为它体现了工程思维：

**不是所有问题都能靠调参数解决。**

## 5. 关键代码

### 代码 1：参数实验结构

文件：[src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py)

```python
@dataclass
class TuningCase:
    name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
```

这段代码的作用：

- 把一组调优参数组织成清晰结构

### 代码 2：单组实验执行

文件：[src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py)

```python
chunks = split_documents(
    documents,
    chunk_size=case.chunk_size,
    chunk_overlap=case.chunk_overlap,
)
```

这一步表示：

- 每次实验都用不同切分参数重新构建 chunk

### 代码 3：切换 top_k

文件：[src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py)

```python
retriever = SimpleRetriever(store, top_k=case.top_k)
```

这一步表示：

- `top_k` 也是调优的重要参数

### 代码 4：输出命中文档来源

文件：[src/rag_tuning_intro.py](/mnt/d/AIcodes/Agent/src/rag_tuning_intro.py)

```python
source = document.metadata.get("source", "unknown")
chunk_index = document.metadata.get("chunk_index", "unknown")
```

如果你不输出这些信息，就很难真正判断：

- 检索命中的到底是不是正确文档

## 6. 常见误差来源

### 1. 切分误差

表现：

- 相关信息被切散
- 单个 chunk 太大或太小

### 2. 召回误差

表现：

- top_k 太小
- 真正相关片段没被召回

### 3. embedding 误差

表现：

- 即使调了参数，仍然经常召回错文档

你当前项目这里非常可能就有这个问题，因为我们用的是：

- 教学型本地 embedding

### 4. 生成误差

表现：

- 检索对了，但模型总结错了
- 或模型过度推断

## 7. 常见错误

### 错误 1：RAG 效果差就只怪模型

很多时候问题根本不在生成层，而在检索层。

### 错误 2：只调一个参数就下结论

RAG 是多参数系统，不能只看一次结果。

### 错误 3：不打印命中文档来源

这样你永远不知道：

- 是检索错了
- 还是回答错了

## 8. 常见面试问题

### 问题 1：RAG 效果差时，应该优先排查什么？

回答要点：

- 先看检索命中是否正确
- 再看上下文是否足够
- 最后再看生成层是否正确利用资料

深入追问：

- 为什么不能一上来只调 Prompt？

### 问题 2：`chunk_size`、`chunk_overlap`、`top_k` 分别影响什么？

回答要点：

- `chunk_size` 影响文本块粒度
- `chunk_overlap` 影响上下文连续性
- `top_k` 影响召回范围

深入追问：

- 什么时候 top_k 增大反而可能伤害回答质量？

### 问题 3：为什么有些问题调参数也救不回来？

回答要点：

- 因为 embedding 本身太弱
- 召回能力受表示能力限制
- 参数只能微调，不能根本替代高质量向量表示

深入追问：

- 这时下一步应该换什么？embedding 模型还是向量库？

### 问题 4：怎么区分“检索错了”和“生成错了”？

回答要点：

- 看命中文档是否正确
- 如果命中文档正确但回答仍然错，多半是生成问题
- 如果命中文档本身就不对，多半是检索问题

深入追问：

- 为什么打印来源清单是诊断 RAG 的关键动作？

## 9. 本节验收

运行：

```bash
python src/rag_tuning_intro.py
```

如果你能看到多组参数对应的：

- chunk 总数
- 命中文档来源
- 命中片段预览

就说明这一节已经跑通了。

## 10. 这一节和后面有什么关系

这一步是后面这些能力的前置基础：

- 更真实的 embedding
- 更真实的向量库
- RAG 评测
- 检索质量分析

因为从这一步开始，你不只是“会搭 RAG”，而是开始“会调 RAG”。

