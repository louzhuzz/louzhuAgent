# 第十八步：Embedding 与向量化

## 1. 本节目标

这一节要把切分后的 chunk 从：

- 文本

进一步变成：

- 向量

本节新增的示例文件是：

- [examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

## 2. 是什么

Embedding，可以先简单理解成：

**把文本映射成一组数字向量。**

这样程序后面就可以做这些事：

- 比较文本之间的相似度
- 找到和问题最接近的文本块
- 为后续检索做准备

## 3. 为什么

字符串本身不适合直接做“语义相似度”比较。

例如：

- “输出解析器是什么”
- “LangChain 中 parser 的作用”

虽然字面写法不同，但语义接近。

Embedding 的作用就是：

**把文本转成可比较的向量表示。**

不过这一节有一个非常重要的说明：

当前实现是：

**教学型本地 embedding**

它的目标是帮助你理解：

- 文本如何变成向量
- 向量如何算相似度

它不是生产级语义 embedding 模型。

## 4. 怎么做

### 第一步：把文本切成 token

我们先做最简单的分词近似：

- 用正则提取单词
- 统一转小写

### 第二步：把 token 映射到固定维度

本节用了一个教学型做法：

- 对 token 做哈希
- 映射到固定维度桶

这样就能得到一个固定长度向量。

### 第三步：做归一化

如果不归一化，文本越长，向量值通常越大。

这会影响相似度比较。

所以我们对向量做了 L2 归一化。

### 第四步：用余弦相似度比较文本

把：

- 查询文本
- chunk 文本

都变成向量后，就可以计算：

- `cosine_similarity`

得出谁更接近查询。

## 5. 关键代码

### 代码 1：tokenize

文件：[examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

```python
def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE)]
```

这一步是最小 token 化逻辑。

作用是：

- 从文本里提取可比较的词项

### 代码 2：文本转向量

文件：[examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

```python
def embed_text(text: str, dimension: int = 32) -> list[float]:
    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dimension
        vector[index] += 1.0
```

这段代码的本质是：

- 每个 token 被映射到一个固定维度位置
- 这个位置上的值不断累加

这就是最小可运行的“向量化”。

### 代码 3：向量归一化

文件：[examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

```python
norm = math.sqrt(sum(value * value for value in vector))
return [value / norm for value in vector]
```

这一步的意义是：

- 让向量长度被标准化
- 更适合做相似度比较

### 代码 4：余弦相似度

文件：[examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

```python
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))
```

这里因为前面已经做了归一化，所以点积就等价于余弦相似度。

### 代码 5：对 chunk 排序

文件：[examples/embedding_intro.py](/mnt/d/AIcodes/Agent/examples/embedding_intro.py)

```python
score = cosine_similarity(query_vector, embed_text(chunk_text))
scored.sort(key=lambda item: item[1], reverse=True)
```

这一步就是最小版“语义检索前半段”。

## 6. 常见错误

### 错误 1：把教学型 embedding 当成生产级 embedding

不对。

这一节只是帮助你理解原理，不代表真实语义模型效果。

### 错误 2：不做归一化就直接比较

后果：

- 长文本容易因为长度优势拿到更高分

### 错误 3：以为 embedding 之后就已经完成检索系统

还没有。

后面还需要：

- 向量存储
- 检索器
- 上下文拼接

## 7. 常见面试问题

### 问题 1：Embedding 是什么？

回答要点：

- 是把文本映射成向量表示
- 使文本可以做相似度比较
- 是语义检索的重要基础

深入追问：

- 为什么字符串不能直接很好地表达语义相似度？

### 问题 2：为什么做 embedding 后可以检索？

回答要点：

- 因为查询和文档都被映射到同一向量空间
- 可以通过相似度函数比较接近程度
- 从而找到最相关文本块

深入追问：

- 为什么常用余弦相似度，而不是直接比较原字符串？

### 问题 3：为什么这一节先做本地教学型向量化？

回答要点：

- 先理解文本到向量的本质
- 先理解相似度比较的流程
- 后面再接真实 embedding API 更容易理解

深入追问：

- 真实 embedding 模型和本地哈希向量化的本质差别是什么？

### 问题 4：归一化为什么重要？

回答要点：

- 避免文本长度影响相似度
- 让比较更关注方向而不是大小
- 这是余弦相似度常见前置步骤

深入追问：

- 如果不归一化，会在检索中造成什么偏差？

## 8. 本节验收

运行：

```bash
python examples/embedding_intro.py
```

如果你能看到：

- 查询文本
- chunk 总数
- Top 3 相似 chunk 及分数

就说明这一节已经跑通了。

## 9. 这一节和后面有什么关系

这一步是后面这些内容的前置：

- 向量存储
- Retriever
- 检索排序

也就是说，从这一步开始，你已经不只是“读文档”和“切文档”了，而是开始真正进入“语义检索”的入口层。

