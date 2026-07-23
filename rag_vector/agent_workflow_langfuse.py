from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from ask_vector import (
    analyze_query,
    build_prompt,
    context_key,
    normalize_search_query,
    print_sources,
    rerank_contexts,
    retrieve,
    select_contexts,
)
from embedding import chat_completion


SMALL_DATA_RE = re.compile(r"小样本|少量|数据量较少|limited data|small data|imbalanced", re.IGNORECASE)
CHANGING_CONDITION_RE = re.compile(r"工况变化|变工况|不同工况|载荷|转速|working condition|load|speed|domain", re.IGNORECASE)
VIBRATION_RE = re.compile(r"振动|vibration|accelerometer|acceleration", re.IGNORECASE)
LABEL_RE = re.compile(r"标签|label|labeled|unlabeled", re.IGNORECASE)
FAULT_RE = re.compile(r"故障诊断|故障检测|故障分类|fault diagnosis|fault detection|fault classification", re.IGNORECASE)
RUL_RE = re.compile(r"rul|剩余寿命|remaining useful life|prognosis|寿命预测", re.IGNORECASE)


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


def load_local_config() -> None:
    try:
        import config_local
    except ImportError:
        return

    mappings = {
        "LLM_API_KEY": "OPENAI_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "LANGFUSE_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY": "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL": "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST": "LANGFUSE_BASE_URL",
    }
    for local_name, env_name in mappings.items():
        value = getattr(config_local, local_name, None)
        if value and not os.environ.get(env_name):
            os.environ[env_name] = value


def get_langfuse_client():
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise SystemExit(
            "langfuse is not installed. Run:\n"
            "python3 -m pip install langfuse\n"
            "or:\n"
            "python3 -m pip install -r requirements-vector.txt"
        ) from exc

    langfuse = get_client()
    if not langfuse.auth_check():
        raise SystemExit(
            "Langfuse authentication failed. Please set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL."
        )
    return langfuse


def summarize_contexts(contexts: list[dict]) -> list[dict]:
    return [
        {
            "context_id": f"C{i:02d}",
            "paper_id": ctx.get("paper_id"),
            "title": ctx.get("title"),
            "source": ctx.get("source"),
            "page": ctx.get("page"),
            "section": ctx.get("section"),
            "methods": ctx.get("methods"),
            "distance": ctx.get("distance"),
            "keyword_score": ctx.get("keyword_score"),
            "coverage_score": ctx.get("coverage_score"),
            "evidence_score": ctx.get("evidence_score"),
            "retrieval_source": ctx.get("retrieval_source"),
            "excerpt": ctx.get("text", "")[:300],
        }
        for i, ctx in enumerate(contexts, start=1)
    ]


def build_filters(args: argparse.Namespace) -> dict:
    return {
        "task": args.task,
        "method": args.method,
        "year": args.year,
        "title": args.title,
        "source": args.source,
        "contains": args.contains,
    }


def understand_task(user_request: str) -> dict:
    task = "fault diagnosis" if FAULT_RE.search(user_request) else "unknown"
    if RUL_RE.search(user_request):
        task = "RUL prediction" if task == "unknown" else f"{task}; RUL prediction"

    issues = []
    if SMALL_DATA_RE.search(user_request):
        issues.append("small data")
    if CHANGING_CONDITION_RE.search(user_request):
        issues.append("changing working conditions")
    if LABEL_RE.search(user_request):
        issues.append("label availability")

    data_type = "vibration signal" if VIBRATION_RE.search(user_request) else "unknown"
    output_needed = "workflow recommendation"
    if "参数" in user_request or "parameter" in user_request.lower():
        output_needed += "; parameter suggestions"
    if "方法" in user_request or "method" in user_request.lower():
        output_needed += "; method selection"

    return {
        "task": task,
        "data_type": data_type,
        "issues": issues,
        "output_needed": output_needed,
        "raw_request": user_request,
    }


def build_retrieval_plan(task_state: dict) -> list[dict]:
    task = task_state["task"]
    data_type = task_state["data_type"]
    issues = set(task_state["issues"])

    queries = [
        {
            "name": "core-task",
            "query": f"{task} {data_type} PHM bearing methods workflow feature extraction model selection",
            "purpose": "Retrieve papers that match the main PHM task and data type.",
        }
    ]
    if "small data" in issues:
        queries.append(
            {
                "name": "small-data",
                "query": "bearing fault diagnosis small data limited samples data augmentation sample generation imbalanced data",
                "purpose": "Find methods for limited data, imbalance, augmentation, or sample generation.",
            }
        )
    if "changing working conditions" in issues:
        queries.append(
            {
                "name": "working-condition-adaptation",
                "query": "bearing fault diagnosis changing working conditions load speed adaptation transfer learning robustness",
                "purpose": "Find methods that address working condition shifts and generalization.",
            }
        )
    if data_type == "vibration signal":
        queries.append(
            {
                "name": "vibration-feature-extraction",
                "query": "bearing vibration signal preprocessing segmentation frequency-domain features wavelet entropy feature extraction",
                "purpose": "Find preprocessing and feature extraction choices for vibration signals.",
            }
        )
    queries.append(
        {
            "name": "evaluation",
            "query": f"{task} accuracy confusion matrix robustness stability evaluation metrics",
            "purpose": "Find evaluation metrics and validation evidence.",
        }
    )
    return queries


def merge_query_profiles(base_profile: dict, plan_profile: dict) -> dict:
    merged_weighted = dict(base_profile.get("weighted_terms", {}))
    for term, weight in plan_profile.get("weighted_terms", {}).items():
        merged_weighted[term] = max(merged_weighted.get(term, 0.0), weight)
    technical_terms = list(
        dict.fromkeys(base_profile.get("technical_terms", []) + plan_profile.get("technical_terms", []))
    )
    general_terms = list(dict.fromkeys(base_profile.get("general_terms", []) + plan_profile.get("general_terms", [])))
    phrase_terms = list(dict.fromkeys(base_profile.get("phrase_terms", []) + plan_profile.get("phrase_terms", [])))
    evidence_terms = list(dict.fromkeys(base_profile.get("evidence_terms", []) + plan_profile.get("evidence_terms", [])))
    return {
        "intent": base_profile.get("intent", "general"),
        "technical_terms": technical_terms,
        "general_terms": general_terms,
        "phrase_terms": phrase_terms,
        "weighted_terms": merged_weighted,
        "evidence_terms": evidence_terms,
    }


def merge_and_rank_contexts(context_groups: list[dict], query_profile: dict, limit: int) -> list[dict]:
    merged = {}
    for group in context_groups:
        for ctx in group["contexts"]:
            key = context_key(ctx)
            existing = merged.get(key)
            if existing is None:
                ctx = dict(ctx)
                ctx["retrieval_plan_steps"] = [group["name"]]
                ctx["retrieval_plan_query"] = group["query"]
                merged[key] = ctx
            else:
                existing["keyword_score"] = max(existing.get("keyword_score", 0.0), ctx.get("keyword_score", 0.0))
                existing["coverage_score"] = max(existing.get("coverage_score", 0.0), ctx.get("coverage_score", 0.0))
                existing["evidence_score"] = max(existing.get("evidence_score", 0.0), ctx.get("evidence_score", 0.0))
                existing.setdefault("retrieval_plan_steps", []).append(group["name"])

    contexts = list(merged.values())
    for ctx in contexts:
        plan_bonus = min(len(set(ctx.get("retrieval_plan_steps", []))) * 0.05, 0.2)
        ctx["agent_score"] = (
            ctx.get("combined_score", 0.0)
            + 0.20 * ctx.get("coverage_score", 0.0)
            + 0.12 * min(ctx.get("evidence_score", 0.0) / 5.0, 1.0)
            + plan_bonus
        )
    contexts.sort(key=lambda item: item.get("agent_score", 0.0), reverse=True)
    return contexts[:limit]


def summarize_evidence_prompt(task_state: dict, contexts: list[dict], query_profile: dict) -> str:
    evidence_prompt = build_prompt(task_state["raw_request"], contexts, query_profile)
    return (
        "你是PHM Agent的证据整理模块。请只基于RAG资料总结证据，不要给最终方案。\n"
        "输出结构：\n"
        "1. 适合的方法\n"
        "2. 论文中表现较好的方法及证据\n"
        "3. 方法适用条件\n"
        "4. 限制和风险\n"
        "5. 仍缺少的信息\n\n"
        f"结构化任务理解：{json.dumps(task_state, ensure_ascii=False)}\n\n"
        f"{evidence_prompt}"
    )


def recommendation_prompt(task_state: dict, evidence_summary: str, contexts: list[dict], query_profile: dict) -> str:
    workflow = infer_workflow(query_profile)
    source_summary = summarize_contexts(contexts)
    return (
        "你是一个可执行PHM流程设计Agent。请根据任务理解和证据总结，生成workflow recommendation。\n"
        "要求：\n"
        "- 给出清晰步骤，不要只做论文总结。\n"
        "- 每个关键建议都要说明依据，必要时引用[Cxx]来源。\n"
        "- 如果证据不足，要明确写成风险或追问，不要编造。\n"
        "- 参数建议只能给合理范围或需要调优的参数类型，除非资料中有明确数值。\n\n"
        "请输出：\n"
        "1. 任务理解\n"
        "2. 推荐workflow\n"
        "3. 方法选择建议\n"
        "4. 参数和实验设计建议\n"
        "5. 评估指标\n"
        "6. 风险提示\n"
        "7. 需要继续追问用户的问题\n\n"
        f"结构化任务理解：{json.dumps(task_state, ensure_ascii=False)}\n"
        f"默认流程候选：{json.dumps(workflow, ensure_ascii=False)}\n"
        f"证据总结：\n{evidence_summary}\n\n"
        f"来源摘要：{json.dumps(source_summary, ensure_ascii=False)}"
    )


def follow_up_questions(task_state: dict) -> list[str]:
    questions = []
    if task_state["data_type"] == "unknown":
        questions.append("你的数据类型是什么？是振动信号、电流信号，还是已经提取好的特征？")
    if "label availability" not in task_state["issues"]:
        questions.append("数据是否有故障类别标签？标签是否完整、是否存在类别不平衡？")
    if "changing working conditions" in task_state["issues"]:
        questions.append("工况变化具体包括哪些因素？例如转速、载荷、采样位置分别有几种？")
    questions.append("数据来自公开数据集（如 CWRU、PU、XJTU-SY、FEMTO）还是你自己采集？")
    questions.append("你的目标是故障分类、故障位置识别，还是还包括RUL预测？")
    return questions


def run_workflow(args: argparse.Namespace) -> str:
    load_local_config()
    langfuse = get_langfuse_client()

    filters = build_filters(args)

    with langfuse.start_as_current_observation(
        as_type="span",
        name="phm-rag-agent-workflow",
        input={
            "request": args.request,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "filters": filters,
        },
    ) as root:
        root.update(
            name="PHM RAG Agent Workflow",
            metadata={
                "user_id": args.user_id,
                "session_id": args.session_id,
                "tags": ["midterm-demo", "rag", "agent", "phm"],
                "collection": args.collection,
                "db_dir": args.db_dir,
            },
        )

        with langfuse.start_as_current_observation(as_type="span", name="1-understand-task") as span:
            task_state = understand_task(args.request)
            query_profile = analyze_query(args.request)
            span.update(
                input=args.request,
                output={
                    "task_state": task_state,
                    "query_profile": query_profile,
                },
            )

        with langfuse.start_as_current_observation(as_type="span", name="2-plan-rag-search") as span:
            retrieval_plan = build_retrieval_plan(task_state)
            span.update(
                input={"task_state": task_state},
                output={"retrieval_plan": retrieval_plan},
            )

        context_groups = []
        with langfuse.start_as_current_observation(as_type="span", name="3-call-rag-retriever") as retrieval_span:
            for plan_item in retrieval_plan:
                query = normalize_search_query(plan_item["query"])
                plan_query_profile = merge_query_profiles(query_profile, analyze_query(query))
                if not args.no_query_rewrite and not args.keyword_only:
                    with langfuse.start_as_current_observation(
                        as_type="generation",
                        name=f"3a-query-rewrite-{plan_item['name']}",
                        model=args.chat_model,
                        input=plan_item,
                    ) as generation:
                        from ask_vector import rewrite_query

                        query = rewrite_query(plan_item["query"], model=args.chat_model)
                        plan_query_profile = merge_query_profiles(query_profile, analyze_query(query))
                        generation.update(output=query)
                contexts = retrieve(
                    query=normalize_search_query(query),
                    db_dir=Path(args.db_dir),
                    collection_name=args.collection,
                    embedding_model=args.embedding_model,
                    candidate_k=max(args.candidate_k, args.top_k),
                    filters=filters,
                    use_hybrid=not args.no_hybrid,
                    keyword_only=args.keyword_only,
                    debug=args.debug,
                    query_profile=plan_query_profile,
                )
                context_groups.append(
                    {
                        "name": plan_item["name"],
                        "query": query,
                        "query_profile": plan_query_profile,
                        "contexts": contexts,
                    }
                )
            retrieval_span.update(
                input={"retrieval_plan": retrieval_plan, "filters": filters},
                output={
                    "groups": [
                        {
                            "name": group["name"],
                            "query": group["query"],
                            "candidate_count": len(group["contexts"]),
                            "top": summarize_contexts(group["contexts"][: args.trace_contexts]),
                        }
                        for group in context_groups
                    ]
                },
            )

        contexts = merge_and_rank_contexts(context_groups, query_profile, limit=max(args.candidate_k, args.top_k))
        if not contexts:
            root.update(output="No matching context found.")
            raise SystemExit("No matching context found.")

        selection_pool_k = max(args.top_k * 2, args.top_k)
        if not args.no_rerank:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="4-rerank-and-select-evidence",
                model=args.chat_model,
                input={"request": args.request, "candidate_count": len(contexts)},
            ) as generation:
                contexts = rerank_contexts(
                    args.request,
                    contexts,
                    model=args.chat_model,
                    top_k=selection_pool_k,
                    query_profile=query_profile,
                )
                generation.update(output={"reranked_contexts": summarize_contexts(contexts[: args.trace_contexts])})
        else:
            contexts = contexts[:selection_pool_k]

        with langfuse.start_as_current_observation(as_type="span", name="5-context-selection") as span:
            final_contexts = select_contexts(contexts, top_k=args.top_k, max_per_paper=args.max_per_paper)
            span.update(
                input={"top_k": args.top_k, "max_per_paper": args.max_per_paper},
                output={"selected_contexts": summarize_contexts(final_contexts)},
            )

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="6-summarize-evidence",
            model=args.chat_model,
            input={"task_state": task_state, "sources": summarize_contexts(final_contexts)},
        ) as generation:
            evidence_summary = chat_completion(
                summarize_evidence_prompt(task_state, final_contexts, query_profile),
                model=args.chat_model,
            )
            generation.update(output=evidence_summary)

        with langfuse.start_as_current_observation(as_type="span", name="7-check-missing-information") as span:
            questions = follow_up_questions(task_state)
            span.update(input=task_state, output={"follow_up_questions": questions})

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="8-generate-workflow-recommendation",
            model=args.chat_model,
            input={"task_state": task_state, "evidence_summary": evidence_summary, "follow_up_questions": questions},
        ) as generation:
            answer = chat_completion(
                recommendation_prompt(task_state, evidence_summary, final_contexts, query_profile),
                model=args.chat_model,
            )
            generation.update(output=answer)

        root.update(
            output={
                "task_state": task_state,
                "retrieval_plan": retrieval_plan,
                "evidence_summary": evidence_summary,
                "follow_up_questions": questions,
                "answer": answer,
                "sources": summarize_contexts(final_contexts),
            }
        )

    langfuse.flush()

    print("\nAGENT WORKFLOW ANSWER")
    print("=" * 80)
    print(answer)
    print_sources(final_contexts, pdf_dir=Path(args.pdf_dir))
    return answer


def run_local_preview(args: argparse.Namespace) -> None:
    task_state = understand_task(args.request)
    query_profile = analyze_query(args.request)
    retrieval_plan = build_retrieval_plan(task_state)
    filters = build_filters(args)
    context_groups = []
    for plan_item in retrieval_plan:
        plan_query_profile = merge_query_profiles(query_profile, analyze_query(plan_item["query"]))
        contexts = retrieve(
            query=normalize_search_query(plan_item["query"]),
            db_dir=Path(args.db_dir),
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            candidate_k=max(args.candidate_k, args.top_k),
            filters=filters,
            use_hybrid=not args.no_hybrid,
            keyword_only=True,
            debug=args.debug,
            query_profile=plan_query_profile,
        )
        context_groups.append(
            {
                "name": plan_item["name"],
                "query": plan_item["query"],
                "query_profile": plan_query_profile,
                "contexts": contexts,
            }
        )
    contexts = merge_and_rank_contexts(context_groups, query_profile, limit=max(args.candidate_k, args.top_k))
    final_contexts = select_contexts(contexts, top_k=args.top_k, max_per_paper=args.max_per_paper)

    print("\nLOCAL AGENT WORKFLOW PREVIEW")
    print("=" * 80)
    print("1. Task understanding")
    print(json.dumps(task_state, ensure_ascii=False, indent=2))
    print("\n2. RAG retrieval plan")
    print(json.dumps(retrieval_plan, ensure_ascii=False, indent=2))
    print("\n3. Evidence candidates")
    for group in context_groups:
        print(f"- {group['name']}: {len(group['contexts'])} candidates")
    print("\n4. Selected evidence")
    print_sources(final_contexts, pdf_dir=Path(args.pdf_dir))
    print("\n5. Follow-up questions")
    for i, question in enumerate(follow_up_questions(task_state), start=1):
        print(f"{i}. {question}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse-traced PHM RAG Agent workflow.")
    parser.add_argument("request", help="User task request for the agent.")
    parser.add_argument("--db-dir", default="data/chroma", help="Persistent Chroma database directory.")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="PDF root directory used for source paths.")
    parser.add_argument("--collection", default="phm_papers", help="Chroma collection name.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--chat-model", default="gpt-4.1-mini", help="OpenAI chat model.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks used by the agent.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate count before context selection.")
    parser.add_argument("--max-per-paper", type=int, default=2, help="Maximum final chunks from the same paper.")
    parser.add_argument("--task", default="", help="Filter by task metadata.")
    parser.add_argument("--method", default="", help="Filter by method metadata.")
    parser.add_argument("--year", default="", help="Filter by exact year metadata.")
    parser.add_argument("--title", default="", help="Filter by title substring.")
    parser.add_argument("--source", default="", help="Filter by source filename substring.")
    parser.add_argument("--contains", default="", help="Filter by substring in metadata or chunk text.")
    parser.add_argument("--keyword-only", action="store_true", help="Use local keyword retrieval only.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval.")
    parser.add_argument("--no-query-rewrite", action="store_true", help="Disable query rewriting step.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking step.")
    parser.add_argument("--local-preview", action="store_true", help="Preview the workflow locally without Langfuse or LLM calls.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval diagnostics.")
    parser.add_argument("--trace-contexts", type=int, default=8, help="Number of candidates to include in trace metadata.")
    parser.add_argument("--user-id", default="midterm-demo-user", help="Langfuse user id.")
    parser.add_argument("--session-id", default="midterm-demo-session", help="Langfuse session id.")
    args = parser.parse_args()

    if args.local_preview:
        run_local_preview(args)
    else:
        run_workflow(args)


if __name__ == "__main__":
    main()
