# 第二十六步：Rerank 与检索结果重排

## 1. 本节目标

这一节的目标是解决一个你已经亲眼看到的问题：

- 文档能命中
- 但 `Top 1` 不一定是最理想的片段

所以这一节要做的事是：

- 先用真实 embedding 召回一批候选片段
- 再用额外信号对这些片段做二次排序
- 最后把重排后的结果送进 RAG 回答

本节新增文件：

- [examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

## 2. 是什么

`Rerank` 的中文通常叫：

- 重排
- 二次排序

它的作用不是替代检索，而是接在检索后面，专门做这件事：

**在已经召回的一小批候选文档里，重新判断谁更应该排前面。**

所以数据流通常是：

1. Retriever 先召回 Top N
2. Reranker 再对这 Top N 做更细的判断
3. 选出更适合送给大模型的 Top K

## 3. 为什么

向量检索很擅长做“粗召回”：

- 能把大方向相关的文档捞出来

但它不一定最擅长做“细排序”：

- 哪一段最像定义
- 哪一段最适合直接回答
- 哪一段只是提到了关键词，但不是核心解释

所以真实 RAG 系统里，经常会有两层：

- 第一层：召回
- 第二层：重排

这一节虽然还没有接入专门的 reranker 模型，但已经先让你理解“二次排序”这件事的工程位置。

## 4. 怎么做

### 第一步：先多召回一些候选片段

我们这次不直接只拿 Top 3，而是先拿：

```python
raw_results = vector_store.similarity_search_with_score(question, k=6)
```

这一步很关键。

因为如果你只召回 3 条，再去重排，空间太小了。

重排的前提是：

- 先把“可能相关”的候选集拿到手

### 第二步：为每个候选片段增加第二种分数

当前这节做的是一个教学版 rerank：

- 第一种分数：向量相似度分数 `vector_score`
- 第二种分数：关键词重叠分数 `overlap_score`

然后把它们加权合成：

```python
final_score = vector_score * 0.7 + overlap_score * 0.3
```

这个版本不是工业级 reranker，但它能帮你看懂：

- 为什么“召回”和“重排”要分开
- 为什么排序可以使用多种信号

### 第三步：按最终分数重新排序

重排后我们拿：

- `final_score` 更高的文档排前面

再把这些结果送进 RAG Prompt。

### 第四步：打印“重排前 / 重排后”结果

这一步非常重要。

因为重排不是只看最终回答，而是要能观察中间过程：

- 原始向量分数是多少
- 关键词重叠分数是多少
- 最终合成分数是多少

这就是为什么脚本里专门打印了两套结果。

## 5. 关键代码

### 代码 1：重排结果数据结构

文件：[examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

```python
@dataclass
class RerankResult:
    document: Document
    vector_score: float
    overlap_score: float
    final_score: float
```

这段代码的作用是把每个候选片段的多种分数放在一起。

你要注意这里为什么不用普通元组。

因为一旦开始做 rerank，一个结果通常就不再只有一个分数，而是至少会有：

- 原始召回分数
- 重排分数
- 最终排序分数

用 `dataclass` 会比 `(doc, a, b, c)` 这种元组更清晰。

### 代码 2：提取查询关键词

文件：[examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

```python
def extract_query_terms(text: str) -> set[str]:
    lowered = text.lower()
    english_terms = set(re.findall(r"[a-z0-9]+", lowered))
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]+", text)

    chinese_terms: set[str] = set()
    for block in chinese_blocks:
        cleaned = block.strip()
        if len(cleaned) >= 2:
            chinese_terms.add(cleaned)
        for index in range(len(cleaned) - 1):
            chinese_terms.add(cleaned[index : index + 2])
```

这一段要重点理解。

我们做的是一个“轻量级教学版”重排器，所以没有引入真正的 reranker 模型，而是先做一个简单但有用的信号：

- 如果查询里出现了“输出解析器”
- 文档片段里也直接出现了“输出解析器”

那么这个片段往往更像你真正想要的答案。

为什么这里还额外做了“中文二元切分”？

因为中文不像英文天然有空格。

如果只拿整句做关键词，匹配会太粗；补一层二元切分，能让“输出”“解析”“解析器”这类局部特征更容易命中。

### 代码 3：关键词重叠分数

文件：[examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

```python
def compute_overlap_score(query: str, content: str) -> float:
    query_terms = extract_query_terms(query)
    if not query_terms:
        return 0.0

    lowered_content = content.lower()
    hit_count = sum(1 for term in query_terms if term in lowered_content)
    return hit_count / len(query_terms)
```

这里的含义是：

- 查询里提取出一组关键词
- 看文档内容命中了多少个
- 命中比例越高，`overlap_score` 越高

这不是语义理解最强的方法，但它能补足一个问题：

- 向量检索已经语义相关
- 但不一定“字面上最贴题”

### 代码 4：合成最终分数

文件：[examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

```python
final_score = vector_score * vector_weight + overlap_score * overlap_weight
```

这句是这节最核心的代码。

它代表：

- 向量检索负责语义相关性
- 关键词重叠负责字面贴题性
- 最终排序由两者共同决定

为什么默认是 `0.7` 和 `0.3`？

因为当前我们希望：

- 语义仍然是主信号
- 关键词只是辅助信号

如果把关键词权重拉太高，就会退化成“谁字面命中更多谁靠前”，这通常会伤害泛化能力。

### 代码 5：重排后的 RAG

文件：[examples/rerank_intro.py](/mnt/d/AIcodes/Agent/examples/rerank_intro.py)

```python
raw_results = vector_store.similarity_search_with_score(question, k=6)
reranked_results = rerank_documents(question, raw_results, top_k=3)
context = build_context(reranked_results)
```

这三行代码要当成一条完整链路来理解：

1. 先召回 6 条候选
2. 再重排成最终 3 条
3. 最后把这 3 条送进 Prompt

这就是最小的：

**召回 -> 重排 -> 生成**

## 6. 常见错误

### 错误 1：把 rerank 理解成“重新检索”

不是。

检索是：

- 去大库里找候选

重排是：

- 对候选集再排序

### 错误 2：召回条数太少

如果你一开始只召回 2 到 3 条，rerank 的发挥空间会很小。

所以通常会先取一个更大的候选集，比如：

- Top 5
- Top 10
- Top 20

再交给 reranker。

### 错误 3：把关键词权重设太高

如果 `overlap_score` 权重过高，系统就会更像“关键词搜索”，而不是语义检索。

### 错误 4：只看最终答案，不看排序过程

这会让你误判问题来源。

做 rerank 调试时，必须看：

- 原始 Top N
- 重排后 Top K
- 每条的多种分数

## 7. 常见面试问题

### 问题 1：为什么 RAG 里需要 rerank？

回答要点：

- Retriever 更擅长粗召回
- Reranker 更适合细排序
- 这样能提升送进大模型上下文的片段质量

深入追问：

- 如果召回已经很准，什么情况下仍然需要 rerank？

### 问题 2：为什么这节要把向量分数和关键词分数组合起来？

回答要点：

- 向量分数反映语义相关性
- 关键词分数补充字面命中信号
- 多信号融合是常见的工程做法

深入追问：

- 如果要把这套教学版 rerank 升级成更真实的系统，你会加入哪些额外信号？

### 问题 3：rerank 一定要用大模型吗？

不一定。

回答要点：

- 可以用规则
- 可以用传统排序模型
- 可以用 cross-encoder
- 也可以用 LLM 做判断

关键不是“必须用什么模型”，而是：

- 有没有把候选片段排得更合理

## 8. 本节验收

你现在可以运行：

```bash
python examples/rerank_intro.py
```

或者：

```bash
python examples/rerank_intro.py 什么是输出解析器，它在 LangChain 学习路径里有什么作用？
```

你应该能看到三段输出：

1. 原始向量检索结果
2. 重排后的结果
3. 基于重排结果生成的最终回答

如果你能清楚看出：

- 为什么原始 `Top 1` 不一定最好
- 为什么重排后顺序发生了变化

这一节就真正学会了。
