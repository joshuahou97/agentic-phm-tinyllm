# Agentic PHM TinyLLM

Bachelor thesis project for an intelligent PHM system targeting bearing fault diagnosis, literature-grounded reasoning, and future TinyLLM deployment on embedded/edge devices.

The current repository contains the first working knowledge-retrieval and agent-workflow prototype. It is not yet the final embedded TinyLLM system. At this stage, the RAG module is used as a knowledge-support component for later PHM agent development.

## Current Scope

- Build a literature-grounded PHM knowledge module from PDF papers.
- Retrieve evidence from bearing fault diagnosis / RUL / PHM papers using OpenAI embeddings and Chroma.
- Generate source-aware Chinese answers and workflow recommendations.
- Evaluate retrieval and answer quality with lightweight retrieval metrics and Ragas.
- Provide the foundation for later FCF feature extraction, multimodal bearing datasets, TinyLLM fine-tuning, and embedded validation.

## Repository Structure

```text
rag_vector/
  build_vector_db.py              # Build Chroma vector DB from PDF papers
  ask_vector.py                   # Source-aware RAG question answering
  agent_workflow_langfuse.py      # PHM agent workflow with optional Langfuse tracing
  eval_rag.py                     # Lightweight retrieval evaluation
  eval_ragas.py                   # Ragas answer-quality evaluation
  pdf_utils.py                    # PDF extraction, cleaning, chunking
  metadata.py                     # Paper metadata loading
  embedding.py                    # OpenAI embedding/chat API helpers
  eval_questions.json             # Evaluation questions
  config_local.example.py         # Local key configuration example

data/
  paper_metadata.json             # Metadata for indexed papers

requirements-vector.txt           # Python dependencies
```

PDF files, Chroma databases, local API keys, and generated outputs are intentionally not committed.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m pip install -r requirements-vector.txt
```

Set the OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Alternatively, copy the example local config:

```bash
cp rag_vector/config_local.example.py rag_vector/config_local.py
```

Then fill in your local key. `rag_vector/config_local.py` is ignored by Git.

## Prepare Papers

Put PDF papers under:

```text
data/pdfs/
```

Subfolders are supported, for example:

```text
data/pdfs/papers/
```

The PDF files are not included in this repository because they may be copyrighted and can be large.

## Build Vector Database

Run from the project root:

```bash
python3 rag_vector/build_vector_db.py --pdf-dir data/pdfs --db-dir data/chroma
```

This creates a local Chroma database in `data/chroma/`, which is also ignored by Git.

## Ask Questions

```bash
python3 rag_vector/ask_vector.py "哪些论文比较了CNN、CNN-SVM和DNN在轴承故障检测中的表现？主要结论是什么？"
```

Useful options:

```bash
python3 rag_vector/ask_vector.py "你的问题" --show-context
python3 rag_vector/ask_vector.py "你的问题" --keyword-only --retrieve-only --debug
python3 rag_vector/ask_vector.py "你的问题" --method CNN
python3 rag_vector/ask_vector.py "你的问题" --task "RUL prediction"
```

## Agent Workflow Preview

Local preview without Langfuse or LLM calls:

```bash
python3 rag_vector/agent_workflow_langfuse.py \
"我有轴承振动数据，想做故障诊断，数据量较少，工况变化明显，应该怎么设计流程？" \
--local-preview --debug
```

Full workflow with Langfuse tracing requires Langfuse credentials:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"

python3 rag_vector/agent_workflow_langfuse.py \
"我有轴承振动数据，想做故障诊断，数据量较少，工况变化明显，应该怎么设计流程？"
```

## Evaluation

Lightweight retrieval evaluation:

```bash
python3 rag_vector/eval_rag.py --keyword-only --top-k 5
```

Ragas answer-quality evaluation:

```bash
python3 rag_vector/eval_ragas.py --limit 2
```

## Notes

To run the full vector-based pipeline after cloning, the user needs:

- the PDF papers under `data/pdfs/`
- a valid `OPENAI_API_KEY`
- Python dependencies installed from `requirements-vector.txt`

Without an API key, parts of the system can still be previewed with keyword-only or local-preview modes.
