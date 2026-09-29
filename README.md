# rag-agent · 知识库问答 Agent

一个**手写 RAG 核心链路**的知识库问答智能体：文档切分、稠密 + 稀疏混合检索、
RRF 融合、两阶段重排、多轮查询改写、function-calling 工具循环，以及一套自建评测。
LLM / 嵌入 / 向量库均可插拔，兼容任意 OpenAI 协议的服务（OpenAI、DeepSeek、智谱、通义、vLLM）。

> 设计主线：**Naive RAG → Advanced RAG → Agentic RAG 的演进**。每一步都能说清「为什么」，
> 而不是把 LangChain 拼起来。

---

## 亮点（面试可讲）

| 能力 | 做到什么 | 可讲的点 |
| --- | --- | --- |
| 手写核心 | NumPy 余弦向量库 + 手写 BM25（倒排索引） | 讲清检索原理，不黑盒调库 |
| 混合检索 | 向量召回 + BM25 召回 | 语义召回 vs 精确词匹配的互补性 |
| RRF 融合 | 倒数排名融合两路结果 | 为什么 RRF 不依赖分数尺度、为何稳健 |
| 两阶段重排 | 先召回 candidate_k，再精排 top_k | 工业界「先召回后精排」范式 |
| 多轮改写 | 结合历史消解指代，生成独立查询 | 解决「它/这个」类追问检索不到 |
| Agentic RAG | function-calling 让模型自主决定是否检索 | 对比 Naive RAG 的固定流程 |
| 引用溯源 | 答案带 `[n]` 标注，返回来源与分数 | 可解释性、可追溯 |
| 自建评测 | HitRate / Recall / MRR / 忠实度 | **多数简历项目不做评测，这是差异点** |
| 工程化 | CLI + FastAPI + SSE、配置驱动、单测全离线 | 有后端与测试意识 |

---

## 架构

数据流（摄入）：

> 文档(PDF/MD/TXT/DOCX) → 加载 → 递归切分(chunk) → 嵌入(embed) → 双索引(向量库 + BM25) → 持久化

数据流（问答）：

> 问题 → 查询改写 → 混合检索(向量 + BM25) → RRF 融合 → 重排 → 组装带引用上下文 → 生成带 `[n]` 的答案

Agent 模式：

> 问题 → 模型判断 → 调用 `knowledge_search` / `calculator` → 观察结果 → 循环 → 最终答案（步数受 `max_steps` 约束）

---

## 快速开始

### 1. 安装

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; Linux/mac 用 source .venv/bin/activate
pip install -r requirements.txt
# 或 pip install -e .   让它成为可安装包，提供 rag-agent 命令
```

### 2. 零配置离线体验（无需 API key、无需联网）

内置 `mock` provider，可直接跑通全流程：

```bash
python main.py --mock ingest data/docs      # 摄入示例文档
python main.py --mock ask "什么是混合检索"
python main.py --mock chat                   # 交互式多轮
python main.py --mock eval                   # 在自建评测集上打分
```

> mock 的嵌入是词面哈希、LLM 是规则脚本，**只用于验证链路**，指标不代表真实效果。

### 3. 接入真实模型

```bash
cp .env.example .env      # 填入 OPENAI_API_KEY（CLI 会自动加载 .env）
python main.py ingest data/docs
python main.py ask "RRF 是怎么融合两路检索的？" --mode rag
```

默认走 DeepSeek（`config.yaml` 里 `llm.base_url`）；换 OpenAI / 智谱 / 通义只改 `base_url` 与模型名。

---

## 目录结构

```
rag-agent/
├── config.yaml            # 全部可调参数
├── main.py                # 入口：python main.py <命令>
├── src/rag_agent/
│   ├── config.py          # 配置加载（YAML + 环境变量覆盖）
│   ├── types.py           # Document / Chunk 数据模型
│   ├── text_utils.py      # 共享分词器（中英文）
│   ├── llm/               # LLM/嵌入抽象 + OpenAI 兼容 + mock + 本地 + factory
│   ├── ingest/            # loader + splitter + pipeline
│   ├── store/             # numpy_store + bm25 + knowledge_base
│   ├── retrieve/          # retriever（混合+RRF）+ reranker
│   ├── rag/               # chain（检索→生成→引用）+ rewrite（查询改写）
│   ├── agent/             # tools + agent（function-calling 循环）
│   ├── memory.py          # 会话记忆
│   ├── eval/              # metrics + dataset + runner
│   └── api/server.py      # FastAPI + SSE
├── tests/                 # 37 个单测，全部离线
└── data/                  # 示例文档 + 评测集
```

---

## 关键设计

**为什么核心链路手写？** 切分、向量检索、BM25、RRF、重排这些正是 RAG 面试的考察点。
自己写一遍才讲得出「余弦相似度怎么算的」「BM25 的 idf 为什么这么定义」「RRF 为什么稳」。

**混合检索为什么要 RRF？** 向量分数是余弦（约 0~1），BM25 是近似的对数概率（尺度不定），
直接相加会被某一方主导。RRF 只用**排名**：`score = Σ weight · 1/(k + rank)`，k 取 60，天然跨尺度。

**为什么两阶段？** 召回要快且全（candidate_k 取大），精排要准（top_k 取小）。
重排把「LLM 逐条打分」这步单独拆出来，可开关、可换实现（`LLMReranker` / `HeuristicReranker`）。

**Agentic RAG 的区别？** Naive RAG 对每个问题固定检索一次；Agent 由模型自主决定是否检索、
要不要调计算器、信息够了就停。循环用 `max_steps` 兜底，避免无限调用与成本失控。

**评测怎么看？** 检索侧 `HitRate@k / Recall@k / MRR / Precision@k`；生成侧 `引用率 / 关键词覆盖 / 忠实度(词面代理)`，
并额外提供可选的 **LLM-as-judge** 忠实度（需真实模型）。所有指标口径都在 `eval/metrics.py` 里写清。

---

## 使用方式

### CLI

```bash
python main.py ingest <路径>                 # 摄入文件或目录
python main.py ask "问题" --mode rag|agent    # 单轮问答
python main.py ask "问题" --session s1        # 带会话记忆的多轮
python main.py chat                           # 交互式对话（/reset 清记忆，/exit 退出）
python main.py serve                          # 启动 HTTP 服务
python main.py eval --dataset data/eval/qa.jsonl
```
全局参数 `--mock` / `--config <路径>` 写在子命令**前面**。

### HTTP API

```bash
python main.py serve        # http://127.0.0.1:8000
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/stats` | 片段数、模型、工具 |
| POST | `/ingest` | `{"path": "data/docs"}` |
| POST | `/ask` | `{"question": "...", "session_id": "s1", "mode": "rag"}` |
| POST | `/chat/stream` | SSE 流式：`token` 事件 + 末尾 `done`（含引用） |

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"重排序有什么用","session_id":"s1"}'
```

`/ask` 返回示例：

```json
{
  "mode": "rag",
  "query": "什么是混合检索",
  "answer": "…… [1][2]",
  "citations": [
    {"id": 1, "source": "data/docs/vector_search.md", "chunk_id": "…::0003", "index": 3, "score": 0.83}
  ]
}
```

---

## 配置

`config.yaml` 集中管理，任何字段都可用环境变量 `RAG__<SECTION>__<FIELD>` 覆盖（优先级最高）。

嵌入 provider 三选一（`embedding.provider`）：

| provider | 适用 | 说明 |
| --- | --- | --- |
| `openai_compat` | 有 OpenAI 协议服务 | 填 `base_url` + `model`，如智谱 `embedding-3`、通义 `text-embedding-v3` |
| `fastembed` | 想本地且轻量（推荐） | 需 `pip install fastembed`；ONNX、**无需 torch**，默认 `BAAI/bge-small-zh-v1.5` |
| `local` | 想完全本地 | 需 `pip install sentence-transformers`（较重），默认 `BAAI/bge-small-zh-v1.5` |
| `mock` | 离线/测试 | 词面哈希，仅验证链路 |

> 注意：DeepSeek 聊天接口兼容 OpenAI，但**没有 embeddings 端点**——嵌入需另配（上表）。

---

## 测试

```bash
python -m pytest -q        # 37 个用例，全部离线（mock provider）
```

覆盖：provider、切分/加载、向量库/BM25/持久化、RRF/重排、RAG 链/查询改写、
工具/Agent 循环、记忆/CLI/HTTP/SSE、评测指标。

---

## 评测结果（真实运行）

配置：聊天 = DeepSeek `deepseek-chat`；嵌入 = 本地 `BAAI/bge-small-zh-v1.5`（fastembed，512 维）；
语料 = `data/docs`（3 篇）；评测集 = `data/eval/qa.jsonl`（10 题）；`top_k=5`、`rerank=on`。

| 指标 | 值 |
| --- | --- |
| HitRate@5 | 1.000 |
| Recall@5 | 1.000 |
| Precision@5 | 0.340 |
| MRR | 1.000 |
| 忠实度（词面代理） | 0.915 |
| 答案覆盖率 | 0.950 |
| 引用率 | 1.000 |

复现：

```bash
python main.py ingest data/docs && python main.py eval
```

> **诚实说明**：语料仅 3 篇、且每题恰好对应一篇相关文档，检索指标已**饱和**
> （MRR=1.0、命中率 100%），只用于验证链路正确，不足以反映真实检索难度。
> 想要有区分度的简历数字，应**扩大语料**或提高题目难度——见「已知局限」。

---

## 已知局限与后续改进

- **评测语料太小（3 篇）导致检索指标饱和**（MRR=1.0）——扩到几十篇后指标才有区分度；
  mock 指标仅验证链路，真实效果需接真实模型重跑。
- 重摄入同一文件按 chunk id 幂等去重；**文件内容变更会新增 chunk 而不删除旧的**——
  生产应加 `remove_by_source`。
- 向量检索为暴力余弦（`O(n)`）；十万级以上应换 ANN（HNSW/faiss）。
- 忠实度默认是词面代理；接真实模型后可启用 `judge_faithfulness` 做 LLM 评判。

---

## 简历怎么写（可直接改写）

> **知识库问答 Agent（个人项目）** — Python / FastAPI
> - 手写 RAG 核心链路（NumPy 余弦向量库、倒排索引 BM25），实现**混合检索 + RRF 融合 + 两阶段重排**；
>   自建评测 harness（HitRate/Recall/MRR/忠实度），在 N 篇文档、M 题评测集上 HitRate@5 = __%、MRR = __。
>   （填实测值前先扩大语料，避免满分指标经不起追问）
> - 实现多轮**查询改写**消解指代，并结合 **function-calling** 构建 Agentic RAG，
>   让模型自主决定检索与工具调用，`max_steps` 约束成本。
> - 答案**带引用溯源**，FastAPI 提供 SSE 流式接口；配套 **37 个离线单测**与**自建评测 harness**。
