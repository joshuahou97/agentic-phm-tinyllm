# Chroma 向量库 RAG

这是当前项目中较正式的 RAG 版本，主要用于基于论文知识库回答 PHM、轴承故障诊断、RUL 预测等问题，并支持 Agent workflow 演示。

整体流程如下：

```text
PDF 论文 -> 文本切分 chunk -> OpenAI embedding -> Chroma 向量数据库 -> 检索 -> LLM 回答
```

## 安装依赖

```bash
python3 -m pip install chromadb pypdf
```

如果要使用 Langfuse 演示完整 Agent workflow，还需要安装：

```bash
python3 -m pip install langfuse
```

## 配置 API Key

推荐使用环境变量：

```bash
export OPENAI_API_KEY="your_api_key"
```

也可以在本地配置文件中填写：

```bash
rag_vector/config_local.py
```

示例文件是：

```bash
rag_vector/config_local.example.py
```

注意：`config_local.py` 只用于本地保存密钥，不要提交到 Git。

## 构建向量数据库

在项目根目录运行：

```bash
python3 rag_vector/build_vector_db.py --pdf-dir data/pdfs --db-dir data/chroma
```

命令含义：

- `--pdf-dir data/pdfs`：指定 PDF 论文所在目录。
- `--db-dir data/chroma`：指定 Chroma 向量数据库保存位置。

脚本会递归读取 `data/pdfs` 下的 PDF，包括：

```bash
data/pdfs/papers
```

构建索引时还会读取论文元数据：

```bash
data/paper_metadata.json
```

这些元数据会被写入每个 chunk 中，包括论文编号、标题、年份、任务类型、方法和数据集等信息。

## 普通 RAG 问答

运行：

```bash
python3 rag_vector/ask_vector.py "RAG模块如何帮助AI Agent选择故障诊断算法？"
```

跨语言测试示例：

```bash
python3 rag_vector/ask_vector.py "哪些论文比较了CNN、CNN-SVM和DNN在轴承故障检测中的表现？主要结论是什么？"
```

当前问答脚本包含这些检索优化：

1. Query rewrite：将用户问题改写成更适合检索的中英双语 query。
2. Query profiling：提取问题中的技术词、领域词、检索意图和证据需求。
3. Hybrid search：结合向量检索和关键词检索。
4. Metadata weighting：标题、方法、数据集、任务类型和正文中的命中词会被加权。
5. Evidence-aware ranking：优先选择 method、experiment、result、conclusion 等更有证据价值的片段。
6. Candidate retrieval：先取较多候选 chunk，再进行后续筛选。
7. Reranking：使用 LLM 对候选证据重新排序。
8. Context selection：控制最终上下文数量，避免同一篇论文占据过多 chunk。
9. Metadata filtering：支持按任务、方法、年份、标题或文件名过滤。

常用参数：

```bash
python3 rag_vector/ask_vector.py "你的问题" --top-k 5 --candidate-k 12
python3 rag_vector/ask_vector.py "你的问题" --method CNN
python3 rag_vector/ask_vector.py "你的问题" --task "RUL prediction"
python3 rag_vector/ask_vector.py "你的问题" --title "CNN-SVM"
python3 rag_vector/ask_vector.py "你的问题" --year 2019
python3 rag_vector/ask_vector.py "你的问题" --source "Bearing fault detection"
python3 rag_vector/ask_vector.py "你的问题" --max-per-paper 2
python3 rag_vector/ask_vector.py "你的问题" --no-hybrid
python3 rag_vector/ask_vector.py "你的问题" --no-query-rewrite
python3 rag_vector/ask_vector.py "你的问题" --no-rerank
python3 rag_vector/ask_vector.py "你的问题" --show-context
python3 rag_vector/ask_vector.py "你的问题" --keyword-only --retrieve-only --debug --show-context
```

回答末尾的 sources 会显示稳定的上下文编号，例如 `[C01]`，并包含论文页码、章节、摘要片段和本地 PDF 路径。

## 离线检索评估

如果要评估 RAG 检索质量，可以运行：

```bash
python3 rag_vector/eval_rag.py --keyword-only --top-k 5
```

它会输出：

- `Recall@k`：期望命中的论文是否出现在前 k 个检索结果中。
- `MRR`：期望论文排得是否靠前。
- 每个测试问题的 top sources，方便人工检查。

评估题目保存在：

```bash
rag_vector/eval_questions.json
```

这个评估是针对 RAG 系统本身的，不是故障诊断模型训练后的 accuracy 评估。

## Ragas 端到端评估

如果要评估最终 RAG 回答质量，可以运行 Ragas 测评：

```bash
python3 rag_vector/eval_ragas.py --limit 2
```

跑完整评估集：

```bash
python3 rag_vector/eval_ragas.py --limit 0 --output rag_vector/ragas_results_full.json
```

当前脚本复用真实的 RAG 流程，包括 query rewrite、hybrid retrieval、LLM reranking、context selection 和最终回答生成。Ragas 会评估：

- `faithfulness`：回答是否被检索上下文支持。
- `answer_relevancy`：回答是否回应用户问题。
- `llm_context_precision_without_reference`：检索上下文是否对回答有用。
- `nv_context_relevance`：上下文与问题是否相关。

Ragas 评估结果会保存为 JSON，例如：

```bash
rag_vector/ragas_results.json
rag_vector/ragas_results_full.json
```

## Agent Demo

普通 Agent demo 用于展示 RAG 如何支持 PHM workflow 规划，而不仅是问答：

```bash
python3 rag_vector/agent_demo.py "我有一个轴承故障诊断任务，数据量较少并且工况变化明显，应该选择什么流程和方法？" --retrieve-only --keyword-only --debug
```

如果已经配置 API Key，可以运行完整版本：

```bash
python3 rag_vector/agent_demo.py "我有一个轴承故障诊断任务，数据量较少并且工况变化明显，应该选择什么流程和方法？"
```

输出内容包括：

- 任务理解
- 推荐 PHM 流程
- 候选方法和理由
- 实验注意事项
- RAG 证据来源
- 仍然缺少的用户信息

## Langfuse Agent Workflow 演示

中期演讲建议使用这个脚本，它可以在 Langfuse 中展示一次完整 Agent workflow：

```bash
python3 rag_vector/agent_workflow_langfuse.py "我有轴承振动数据，想做故障诊断，数据量较少，工况变化明显，应该怎么设计流程？"
```

Langfuse 凭证可以通过环境变量配置：

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

也可以写入：

```bash
rag_vector/config_local.py
```

示例：

```python
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

Langfuse trace 中包含这些节点：

1. `1-understand-task`：理解用户任务。
2. `2-plan-rag-search`：规划需要检索的子问题。
3. `3-call-rag-retriever`：调用 RAG 检索器。
4. `3a-query-rewrite-*`：为每个检索子任务生成更适合检索的 query。
5. `4-rerank-and-select-evidence`：对候选证据重新排序。
6. `5-context-selection`：选择最终进入 prompt 的上下文。
7. `6-summarize-evidence`：总结论文证据。
8. `7-check-missing-information`：检查还缺少哪些用户信息。
9. `8-generate-workflow-recommendation`：生成最终 workflow 推荐。

完整流程可以理解为：

```text
用户输入
-> 任务理解
-> 检索规划
-> 多 query RAG 检索
-> 证据重排序与选择
-> 证据总结
-> 缺失信息检查
-> workflow 推荐
-> 最终回答和 sources
```

如果只想本地预览，不调用 Langfuse 或 LLM，可以运行：

```bash
python3 rag_vector/agent_workflow_langfuse.py \
"我有轴承振动数据，想做故障诊断，数据量较少，工况变化明显，应该怎么设计流程？" \
--local-preview --debug
```

## PDF 切分与清洗

当前 PDF 处理模块做了这些优化：

- 识别粗粒度章节，例如 abstract、introduction、method、experiment、result、conclusion。
- 将章节信息写入 Chroma metadata。
- 标记表格、图注或公式较多的 chunk。
- 过滤明显的 references / bibliography 区域。
- 对中文文本做更适合的切分，避免完全依赖英文空格。

## 主要文件说明

- `build_vector_db.py`：读取 PDF，切分文本，生成 embedding，并写入 Chroma。
- `ask_vector.py`：普通 RAG 问答入口。
- `agent_demo.py`：不接 Langfuse 的 Agent demo。
- `agent_workflow_langfuse.py`：带 Langfuse tracing 的完整 Agent workflow。
- `eval_rag.py`：离线检索评估脚本。
- `eval_ragas.py`：Ragas 端到端回答质量评估脚本。
- `pdf_utils.py`：PDF 文本抽取、章节识别和 chunk 切分。
- `metadata.py`：读取论文元数据。
- `embedding.py`：调用 OpenAI embedding / chat completion API。
- `eval_questions.json`：离线评估问题集。

## 与旧版 TF-IDF Demo 的区别

旧版 `rag_demo` 主要依赖关键词匹配。

当前 `rag_vector` 使用 embedding 向量和 Chroma 向量数据库，因此更适合：

- 语义检索
- 中英文跨语言问题
- 根据论文证据生成回答
- 支持 Agent workflow 中的多步骤检索和决策
