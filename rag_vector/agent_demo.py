from __future__ import annotations

import argparse
import json
from pathlib import Path

from ask_vector import (
    analyze_query,
    build_prompt,
    normalize_search_query,
    print_sources,
    retrieve,
    select_contexts,
)
from embedding import chat_completion


def infer_workflow(query_profile: dict) -> list[str]:
    terms = " ".join(query_profile["technical_terms"] + query_profile["general_terms"]).lower()
    if "rul" in terms or "remaining useful life" in terms or "寿命" in terms:
        return [
            "数据读取与工况确认",
            "退化特征或健康指标构建",
            "RUL预测模型选择",
            "不确定性或误差评估",
            "结果解释与维护建议",
        ]
    if "simulation" in terms or "model" in terms or "digital" in terms or "twin" in terms:
        return [
            "故障类型与工况定义",
            "仿真/数字孪生模型选择",
            "信号生成或特征提取",
            "诊断模型验证",
            "仿真结果与实测数据对比",
        ]
    return [
        "数据预处理",
        "特征提取或端到端信号输入",
        "故障诊断模型选择",
        "参数调优与训练",
        "准确率、鲁棒性和工况适应性评估",
    ]


def build_agent_prompt(user_request: str, contexts: list[dict], query_profile: dict) -> str:
    evidence_prompt = build_prompt(user_request, contexts, query_profile)
    workflow = infer_workflow(query_profile)
    return (
        "你是一个面向轴承PHM系统的AI Agent原型。"
        "你的任务不是只回答论文内容，而是根据RAG检索到的论文证据，给出可执行的PHM任务方案。\n"
        "请严格基于资料，不要编造论文没有支持的方法效果。\n"
        "请输出以下结构：\n"
        "1. 任务理解\n"
        "2. 推荐PHM流程\n"
        "3. 候选方法与选择理由\n"
        "4. 参数/实验注意事项\n"
        "5. RAG证据来源\n"
        "6. 仍需用户补充的信息\n\n"
        f"默认流程候选：{json.dumps(workflow, ensure_ascii=False)}\n\n"
        f"{evidence_prompt}"
    )


def print_plan_without_llm(user_request: str, contexts: list[dict], query_profile: dict, pdf_dir: Path) -> None:
    print("\nAGENT PLAN (RETRIEVAL ONLY)")
    print("=" * 80)
    print(f"task_intent={query_profile['intent']}")
    print(f"technical_terms={', '.join(query_profile['technical_terms']) or '-'}")
    print("\nRecommended workflow:")
    for i, step in enumerate(infer_workflow(query_profile), start=1):
        print(f"{i}. {step}")
    print("\nCandidate methods from retrieved papers:")
    seen = set()
    for ctx in contexts:
        methods = ctx.get("methods") or "unknown"
        title = ctx.get("title") or ctx.get("source")
        key = (title, methods)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {title}: {methods}")
    print_sources(contexts, pdf_dir=pdf_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small RAG-powered PHM Agent demo.")
    parser.add_argument("request", help="User task request for the agent.")
    parser.add_argument("--db-dir", default="data/chroma", help="Persistent Chroma database directory.")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="PDF root directory used for source paths.")
    parser.add_argument("--collection", default="phm_papers", help="Chroma collection name.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--chat-model", default="gpt-4.1-mini", help="OpenAI chat model.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks used by the agent.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate count before context selection.")
    parser.add_argument("--keyword-only", action="store_true", help="Use local keyword retrieval only.")
    parser.add_argument("--retrieve-only", action="store_true", help="Do not call the chat model.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval diagnostics.")
    args = parser.parse_args()

    query_profile = analyze_query(args.request)
    query = normalize_search_query(args.request)
    filters = {
        "task": "",
        "method": "",
        "year": "",
        "title": "",
        "source": "",
        "contains": "",
    }
    contexts = retrieve(
        query=query,
        db_dir=Path(args.db_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        candidate_k=max(args.candidate_k, args.top_k),
        filters=filters,
        use_hybrid=True,
        keyword_only=args.keyword_only,
        debug=args.debug,
        query_profile=query_profile,
    )
    if not contexts:
        raise SystemExit("No matching context found for the agent request.")

    contexts = select_contexts(contexts[: max(args.top_k * 2, args.top_k)], top_k=args.top_k, max_per_paper=2)
    if args.retrieve_only:
        print_plan_without_llm(args.request, contexts, query_profile, pdf_dir=Path(args.pdf_dir))
        return

    prompt = build_agent_prompt(args.request, contexts, query_profile)
    answer = chat_completion(prompt, model=args.chat_model)
    print("\nAGENT ANSWER")
    print("=" * 80)
    print(answer)
    print_sources(contexts, pdf_dir=Path(args.pdf_dir))


if __name__ == "__main__":
    main()
