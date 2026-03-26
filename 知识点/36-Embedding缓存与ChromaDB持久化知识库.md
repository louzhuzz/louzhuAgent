# 第 36 课：Embedding 缓存与 ChromaDB 持久化知识库

## 1. 本节目标

把主项目里的知识点问答系统，从：

- 每次都临时切块
- 每次都重新 embedding
- 每次都只用临时内存向量库

升级成：

- 文件没变就复用旧向量
- 向量持久化到本地 `ChromaDB`
- 仍然只在少量候选知识点范围内检索

这一课的重点不只是“能用 ChromaDB”，而是：

**让主项目的 RAG 开始具备真实工程里的缓存和持久化能力。**

## 2. 是什么

这一步一共做了两层升级：

### 第一层：Embedding 缓存

缓存的不是最终回答，而是：

- 某篇知识点文件切出来的 chunk
- 这些 chunk 对应的向量
- 这篇文件当前的内容 hash

这样下次运行时，只要文件没变，就不需要重新调用 embedding 模型。

### 第二层：ChromaDB 持久化向量库

之前主项目里用的是：

- `InMemoryVectorStore`

它的问题是：

- 程序一退出，向量就没了
- 每次运行都可能重新建库
- 不适合项目长期迭代

现在改成：

- `Chroma`
- `persist_directory=...`

这样向量会被保存到本地目录里，下次运行仍然能直接复用。

## 3. 为什么

这一课对你当前项目特别重要，原因有三个：

### 1. 直接降低 embedding 成本

你前面已经真实遇到了这个问题：

- embedding token 用得太快

所以现在最值钱的升级不是再加一个新花样，而是：

**让同一篇知识点文件不要被反复 embedding。**

### 2. 让 RAG 从“教学示例”更像“真实工程”

真实项目里不会每次问一个问题就：

- 重新读所有文件
- 重新向量化所有 chunk

现在这一步补上之后，你的主项目就更接近：

- 可长期维护
- 可持续扩展
- 可面试表达

### 3. 为后面日志、评测、指标统计打基础

后面如果你要统计：

- 缓存命中率
- embedding 调用次数下降
- 首次构建与复用构建耗时差异

都必须先有这一层缓存和持久化。

## 4. 怎么做

这一课的实现分成四步。

### 第一步：给配置增加持久化目录

文件：

- [src/config.py](/mnt/d/AIcodes/Agent/src/config.py)
- [.env.example](/mnt/d/AIcodes/Agent/.env.example)

新增配置：

```env
CHROMA_PERSIST_DIRECTORY=data/chroma/knowledge_qa
```

它的作用是告诉主项目：

- 把本地 ChromaDB 数据放到哪里

### 第二步：新增持久化知识库层

文件：

- [src/chroma_knowledge_base.py](/mnt/d/AIcodes/Agent/src/chroma_knowledge_base.py)

这个模块专门负责三件事：

1. 判断知识点文件是否变更
2. 文件没变就复用旧向量
3. 文件变了才重新切块和重新 embedding

### 第三步：把知识点问答系统接到持久化知识库

文件：

- [src/knowledge_qa.py](/mnt/d/AIcodes/Agent/src/knowledge_qa.py)

现在 `KnowledgeQAService` 不再自己临时 new 一个内存向量库，而是：

- 委托 `PersistentKnowledgeBase`

### 第四步：命令行显示索引状态

文件：

- [src/main.py](/mnt/d/AIcodes/Agent/src/main.py)

现在运行 `/qa` 或 `/v1` 的问答能力时，会额外打印：

- `cache_hit`
- `reindexed`

这样你能直接看到本次到底有没有复用旧向量。

## 5. 关键代码

### 代码 1：初始化本地持久化 ChromaDB

文件：

- [src/chroma_knowledge_base.py](/mnt/d/AIcodes/Agent/src/chroma_knowledge_base.py)

```python
self.vector_store = Chroma(
    collection_name=self.collection_name,
    embedding_function=self.embeddings,
    persist_directory=str(self.persist_directory),
)
```

这里最关键的是：

- `collection_name`
- `embedding_function`
- `persist_directory`

它们分别决定：

- 这批数据属于哪个集合
- 用哪个 embedding 模型生成向量
- 向量最终落到本地哪个目录

为什么现在不再用 `InMemoryVectorStore`？

因为它只适合教学或一次性示例，不适合你当前这个要继续迭代的主项目。

### 代码 2：用文件 hash 判断是否需要重新 embedding

```python
content = self.read_note(file_name)
file_hash = self._hash_text(content)
existing = self.manifest["files"].get(file_name)

if (
    existing
    and existing.get("file_hash") == file_hash
    and existing.get("chunk_size") == self.chunk_size
    and existing.get("chunk_overlap") == self.chunk_overlap
):
    return {
        "file_name": file_name,
        "status": "cache_hit",
        "chunk_count": existing.get("chunk_count", 0),
    }
```

这一段是本课最重要的缓存判断逻辑。

它在做的事是：

1. 读取当前知识点文件内容
2. 计算内容 hash
3. 去本地缓存清单里找旧记录
4. 如果：
   - 文件内容没变
   - chunk 参数没变
5. 那就直接判定：
   - `cache_hit`
   - 不重新 embedding

这一步非常关键，因为它直接决定：

- 你后面是不是每次问答都还要重新花 token

### 代码 3：文件变了就先删旧向量，再写新向量

```python
if existing and existing.get("doc_ids"):
    self.vector_store.delete(ids=existing["doc_ids"])
```

为什么这里必须先删？

因为如果文件内容改了，而旧 chunk 对应的向量还留在库里，就会出现：

- 同一篇知识点文件的旧内容和新内容同时存在

这样后面检索就会把脏数据一起召回。

所以这里的工程原则是：

**文件变了，不是简单追加，而是先删旧索引，再写新索引。**

### 代码 4：仍然只在少量候选文件范围内检索

```python
results = self.vector_store.similarity_search_with_score(
    query,
    k=per_file_k,
    filter={"file_name": file_name},
)
```

这一步很重要，因为你前面已经定下了一个现实约束：

- 不做全库无脑 embedding
- 也不做全库无脑检索

所以即使现在已经有了持久化 ChromaDB，主项目仍然坚持：

1. 先选少量候选知识点文件
2. 再只在这些候选文件里检索

这样可以兼顾：

- 召回效果
- 成本控制
- 工程可维护性

### 代码 5：把缓存状态回传给主程序

```python
return {
    "selected_notes": [candidate.file_name for candidate in candidates],
    "retrieved_chunks": retrieved_chunks,
    "index_statuses": index_statuses,
    "context": context,
    "answer": answer,
}
```

这里多出来的：

- `index_statuses`

就是为了让主程序打印：

- 哪篇文件是 `cache_hit`
- 哪篇文件被 `reindexed`

这样你后面做日志和指标统计时，就已经有基础数据了。

## 6. 常见错误

### 错误 1：以为上了 ChromaDB 就自动有缓存

不对。

ChromaDB 负责的是：

- 向量持久化

但“要不要重新 embedding”这件事，还需要你自己判断。

所以这里额外加了：

- manifest 清单
- 文件 hash 校验

### 错误 2：文件变了却不删除旧向量

这会导致：

- 同一篇文件的新旧内容共存
- 检索结果污染

### 错误 3：用了持久化库就开始全量检索

这不符合你当前项目的现实约束。

你这里最重要的策略仍然是：

- 先少量候选筛选
- 再局部检索

### 错误 4：把 `score` 当成一定“越大越好”

不同向量库返回的分数定义可能不同。

这一版里对 Chroma 的结果更稳妥的表述是：

- `distance`

也就是：

- 越小通常越接近

## 7. 常见面试问题

### 问题 1：为什么已经有向量库了，还要自己做 manifest 缓存清单？

回答要点：

- 向量库存的是结果，不负责帮你判断文件有没有变
- 项目仍然需要一层自己的缓存控制逻辑
- manifest 用来记录文件 hash、chunk 参数、doc_ids，便于决定是否复用旧向量

### 问题 2：为什么这里不用全量知识库直接检索？

回答要点：

- 当前项目有真实 token 成本约束
- 先做候选文件筛选，再局部 embedding / 检索，更适合当前阶段
- 这是在效果、成本、复杂度之间做权衡

### 问题 3：ChromaDB 在这个项目里解决了什么问题？

回答要点：

- 解决向量不能持久化的问题
- 为 embedding 缓存提供稳定存储层
- 为后续增量更新、评测、日志和更长期的问答系统升级打基础

## 8. 本节验收

先安装依赖：

```bash
pip install -r requirements.txt
```

然后运行：

```bash
python src/main.py
```

执行两次同样的问题：

```text
/qa 什么是输出解析器？
/qa 什么是输出解析器？
```

如果第一次你看到某些知识点文件是：

- `reindexed`

第二次变成：

- `cache_hit`

说明这节真正跑通了。

再去看本地目录：

```text
data/chroma/knowledge_qa
```

如果里面已经出现持久化数据文件，也说明 ChromaDB 已经落地成功。
