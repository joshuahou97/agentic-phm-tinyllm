from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate

# Ragas 0.4.x still supports these imports. The collections API requires
# explicit constructor arguments for some metrics, so this path is simpler for
# a small project-local evaluation script.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas\.metrics")
from ragas.metrics import (  # noqa: WPS347
    ContextRelevance,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ask_vector import (
    analyze_query,
    build_prompt,
    normalize_search_query,
    rerank_contexts,
    retrieve,
    rewrite_query,
    select_contexts,
)
from embedding import chat_completion, require_api_key


def configure_api_key() -> None:
    os.environ.setdefault("OPENAI_API_KEY", require_api_key())


def make_filters() -> dict:
    return {
        "task": "",
        "method": "",
        "year": "",
        "title": "",
        "source": "",
        "contains": "",
    }


def run_rag(question: str, args: argparse.Namespace) -> dict:
    query_profile = analyze_query(question)
    search_query = question
    if not args.no_query_rewrite:
        search_query = rewrite_query(question, model=args.chat_model)
    search_query = normalize_search_query(search_query)

    contexts = retrieve(
        query=search_query,
        db_dir=Path(args.db_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        candidate_k=max(args.candidate_k, args.top_k),
        filters=make_filters(),
        use_hybrid=not args.no_hybrid,
        keyword_only=args.keyword_only,
        debug=False,
        query_profile=query_profile,
    )
    if not contexts:
        raise RuntimeError(f"No contexts retrieved for question: {question}")

    selection_pool_k = max(args.top_k * 2, args.top_k)
    if not args.no_rerank:
        contexts = rerank_contexts(
            question,
            contexts,
            model=args.chat_model,
            top_k=selection_pool_k,
            query_profile=query_profile,
        )
    else:
        contexts = contexts[:selection_pool_k]

    contexts = select_contexts(contexts, top_k=args.top_k, max_per_paper=args.max_per_paper)
    prompt = build_prompt(question, contexts, query_profile=query_profile)
    answer = chat_completion(prompt, model=args.chat_model)
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "context_texts": [ctx["text"] for ctx in contexts],
        "sources": [
            {
                "paper_id": ctx.get("paper_id"),
                "title": ctx.get("title"),
                "source": ctx.get("source"),
                "page": ctx.get("page"),
                "chunk": ctx.get("chunk"),
                "section": ctx.get("section"),
            }
            for ctx in contexts
        ],
    }


def load_questions(path: Path, limit: int) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        return questions[:limit]
    return questions


def make_dataset(rag_outputs: list[dict]) -> EvaluationDataset:
    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["context_texts"],
            )
            for item in rag_outputs
        ]
    )


def result_to_dict(result) -> dict:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        rows = frame.to_dict(orient="records")
        numeric_columns = [
            col for col in frame.columns if col not in {"user_input", "response", "retrieved_contexts"}
        ]
        summary = {}
        for col in numeric_columns:
            try:
                summary[col] = float(frame[col].mean())
            except Exception:
                pass
        return {"summary": summary, "rows": rows}
    if hasattr(result, "dict"):
        return result.dict()
    return {"raw": str(result)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on the current PHM RAG pipeline.")
    parser.add_argument("--questions", default="rag_vector/eval_questions.json")
    parser.add_argument("--db-dir", default="data/chroma")
    parser.add_argument("--collection", default="phm_papers")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-per-paper", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2, help="Number of eval questions to run. Use 0 for all.")
    parser.add_argument("--output", default="rag_vector/ragas_results.json")
    parser.add_argument("--keyword-only", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-query-rewrite", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    configure_api_key()
    questions = load_questions(Path(args.questions), args.limit)
    print(f"Generating RAG answers for {len(questions)} questions...")
    rag_outputs = [run_rag(item["question"], args) for item in questions]

    dataset = make_dataset(rag_outputs)
    judge_llm = ChatOpenAI(model=args.judge_model, temperature=0)
    judge_embeddings = OpenAIEmbeddings(model=args.judge_embedding_model)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        ContextRelevance(),
    ]

    print("Running Ragas metrics...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    result_dict = result_to_dict(result)
    payload = {
        "settings": vars(args),
        "rag_outputs": rag_outputs,
        "ragas": result_dict,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRagas Evaluation")
    print("=" * 80)
    for name, value in result_dict.get("summary", {}).items():
        print(f"{name}: {value:.3f}")
    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
