from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ask_vector import analyze_query, normalize_search_query, retrieve


def rank_of_expected(contexts: list[dict], expected_sources: list[str]) -> int | None:
    expected = set(expected_sources)
    for index, ctx in enumerate(contexts, start=1):
        if ctx["source"] in expected:
            return index
    return None


def evaluate_question(item: dict, args: argparse.Namespace) -> dict:
    question = item["question"]
    query_profile = analyze_query(question)
    query = normalize_search_query(question)
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
        use_hybrid=not args.no_hybrid,
        keyword_only=args.keyword_only,
        debug=False,
        query_profile=query_profile,
    )
    contexts = contexts[: args.top_k]
    unique_sources = list(dict.fromkeys(ctx["source"] for ctx in contexts))
    rank = rank_of_expected(contexts, item["expected_sources"])
    return {
        "id": item["id"],
        "question": question,
        "hit": rank is not None,
        "rank": rank,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "expected_sources": item["expected_sources"],
        "top_sources": [ctx["source"] for ctx in contexts],
        "top_unique_sources": unique_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for the RAG vector database.")
    parser.add_argument("--questions", default="rag_vector/eval_questions.json", help="Evaluation questions JSON.")
    parser.add_argument("--db-dir", default="data/chroma", help="Persistent Chroma database directory.")
    parser.add_argument("--collection", default="phm_papers", help="Chroma collection name.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--top-k", type=int, default=5, help="Evaluate recall within top-k contexts.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate count before truncating to top-k.")
    parser.add_argument("--keyword-only", action="store_true", help="Evaluate local keyword retrieval only.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    results = [evaluate_question(item, args) for item in questions]
    total = len(results)
    hits = sum(1 for item in results if item["hit"])
    mean_mrr = sum(item["mrr"] for item in results) / total if total else 0.0

    summary = {
        "total": total,
        f"recall@{args.top_k}": hits / total if total else 0.0,
        "mrr": mean_mrr,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("RAG Retrieval Evaluation")
    print("=" * 80)
    print(f"questions={total}")
    print(f"recall@{args.top_k}={summary[f'recall@{args.top_k}']:.3f}")
    print(f"mrr={mean_mrr:.3f}")
    print()
    for result in results:
        status = "HIT" if result["hit"] else "MISS"
        rank = result["rank"] if result["rank"] is not None else "-"
        print(f"[{status}] {result['id']} rank={rank}")
        print(f"  question: {result['question']}")
        print(f"  expected: {', '.join(result['expected_sources'])}")
        print("  top unique papers:")
        for i, source in enumerate(result["top_unique_sources"], start=1):
            print(f"    {i}. {source}")
        print()

    if hits < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
