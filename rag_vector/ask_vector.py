from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import chromadb

from embedding import chat_completion, embed_texts


TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")
BOOLEAN_QUERY_RE = re.compile(r"\b(?:AND|OR|NOT)\b|[()\"']", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "the",
    "their",
    "to",
    "using",
    "what",
    "which",
    "with",
}
DOMAIN_PHRASES = [
    "轴承故障诊断",
    "故障诊断",
    "故障检测",
    "故障分类",
    "性能比较",
    "卷积神经网络",
    "深度神经网络",
    "剩余使用寿命",
    "寿命预测",
    "bearing fault diagnosis",
    "bearing fault detection",
    "bearing fault classification",
    "performance comparison",
    "remaining useful life",
    "rul prediction",
]
GENERAL_DOMAIN_TERMS = {
    "phm",
    "bearing",
    "fault",
    "diagnosis",
    "detection",
    "classification",
    "prediction",
    "rul",
    "轴承故障诊断",
    "故障诊断",
    "故障检测",
    "故障分类",
    "寿命预测",
    "bearing fault diagnosis",
    "bearing fault detection",
    "bearing fault classification",
    "rul prediction",
    "remaining useful life",
}
COMPARISON_RE = re.compile(r"(?i)\b(compare|comparison|versus|vs\.?|different|between|performance)\b|比较|对比|性能")
RECOMMEND_RE = re.compile(r"(?i)\b(recommend|select|choose|suggest|which method|best|suitable)\b|推荐|选择|适合")
SUMMARY_RE = re.compile(r"(?i)\b(summarize|summary|conclusion|main idea|overview)\b|总结|概括|主要结论")
EXPLANATION_RE = re.compile(r"(?i)\b(how|why|principle|mechanism|explain)\b|如何|为什么|原理|解释")
LOW_VALUE_CHUNK_RE = re.compile(
    r"(?i)eidesstattliche|acknowledgement|acknowledgment|declaration|supervisor|technische universität|"
    r"table of contents|list of figures|list of tables|bibliography|references"
)
RESULT_SIGNAL_RE = re.compile(
    r"(?i)\b(accuracy|precision|recall|f1|robustness|stability|result|comparison|conclusion|"
    r"outperform|better|worse|experiment|confusion matrix|classification rate)\b|准确率|鲁棒性|稳定性|结果|结论|比较"
)
METHOD_LIKE_RE = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+$|^[a-z]{2,}[0-9]*$|^[a-z]+/[a-z]+$")


def tokenize(text: str) -> list[str]:
    lowered = text.lower().replace("_", "-")
    tokens = [token.lower().replace("_", "-") for token in TOKEN_RE.findall(text)]
    for phrase in DOMAIN_PHRASES:
        if phrase.lower() in lowered:
            tokens.append(phrase.lower())
    return [token for token in tokens if token not in {"and", "or", "not"}]


def normalize_search_query(query: str) -> str:
    query = BOOLEAN_QUERY_RE.sub(" ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def analyze_query(question: str) -> dict:
    normalized = normalize_search_query(question).lower().replace("_", "-")
    tokens = []
    for token in tokenize(normalized):
        if token in STOPWORDS or len(token) <= 1:
            continue
        tokens.append(token)

    phrase_terms = [phrase.lower() for phrase in DOMAIN_PHRASES if phrase.lower() in normalized]
    technical_terms = []
    general_terms = []
    for token in tokens + phrase_terms:
        if token in GENERAL_DOMAIN_TERMS:
            general_terms.append(token)
        elif METHOD_LIKE_RE.match(token) or token.isalnum() and any(char.isdigit() for char in token):
            technical_terms.append(token)
        elif len(token) >= 5:
            technical_terms.append(token)
        else:
            general_terms.append(token)

    # Preserve order while removing duplicates.
    technical_terms = list(dict.fromkeys(technical_terms))
    general_terms = list(dict.fromkeys(general_terms))
    phrase_terms = list(dict.fromkeys(phrase_terms))

    intent = "general"
    if COMPARISON_RE.search(question):
        intent = "comparison"
    elif RECOMMEND_RE.search(question):
        intent = "recommendation"
    elif SUMMARY_RE.search(question):
        intent = "summary"
    elif EXPLANATION_RE.search(question):
        intent = "explanation"

    evidence_terms = {
        "comparison": ["comparison", "accuracy", "result", "table", "outperform", "robustness", "stability"],
        "recommendation": ["accuracy", "result", "limitation", "dataset", "condition", "performance"],
        "summary": ["abstract", "summary", "conclusion", "contribution", "result"],
        "explanation": ["method", "framework", "model", "principle", "algorithm"],
        "general": ["method", "result", "conclusion"],
    }[intent]

    weighted_terms = {}
    for term in general_terms:
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 0.6)
    for term in phrase_terms:
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 1.5)
    for term in technical_terms:
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 2.5)

    return {
        "intent": intent,
        "technical_terms": technical_terms,
        "general_terms": general_terms,
        "phrase_terms": phrase_terms,
        "weighted_terms": weighted_terms,
        "evidence_terms": evidence_terms,
    }


def evidence_score(ctx: dict, query_profile: dict) -> float:
    text = ctx.get("text", "").lower().replace("_", "-")
    section = ctx.get("section", "unknown")
    score = 0.0
    if section in {"result", "experiment", "conclusion"}:
        score += 2.5
    elif section in {"method", "abstract"}:
        score += 1.2
    if RESULT_SIGNAL_RE.search(text):
        score += 2.0
    for term in query_profile["evidence_terms"]:
        if term in text:
            score += 0.8
    if LOW_VALUE_CHUNK_RE.search(text):
        score -= 4.0
    return max(score, 0.0)


def term_coverage(ctx: dict, query_profile: dict) -> float:
    terms = query_profile["technical_terms"] or list(query_profile["weighted_terms"])
    if not terms:
        return 0.0
    searchable = " ".join(
        [
            ctx.get("title", ""),
            ctx.get("source", ""),
            ctx.get("task", ""),
            ctx.get("methods", ""),
            ctx.get("dataset", ""),
            ctx.get("text", ""),
        ]
    ).lower().replace("_", "-")
    matched = sum(1 for term in terms if term in searchable)
    return matched / len(terms)


def context_key(ctx: dict) -> tuple:
    return (ctx.get("source", ""), ctx.get("page", ""), ctx.get("chunk", ""))


def passes_filters(ctx: dict, filters: dict) -> bool:
    metadata_text = " ".join(
        str(ctx.get(key, ""))
        for key in ("source", "title", "authors", "year", "task", "methods", "dataset", "section")
    ).lower()
    if filters["task"] and filters["task"].lower() not in str(ctx.get("task", "")).lower():
        return False
    if filters["method"] and filters["method"].lower() not in str(ctx.get("methods", "")).lower():
        return False
    if filters["year"] and str(filters["year"]) != str(ctx.get("year", "")):
        return False
    if filters["title"] and filters["title"].lower() not in str(ctx.get("title", "")).lower():
        return False
    if filters["source"] and filters["source"].lower() not in str(ctx.get("source", "")).lower():
        return False
    if filters["contains"] and filters["contains"].lower() not in (metadata_text + " " + ctx.get("text", "").lower()):
        return False
    return True


def metadata_to_context(document: str, metadata: dict, distance: float | None = None, source: str = "vector") -> dict:
    return {
        "text": document,
        "source": metadata.get("source", ""),
        "page": metadata.get("page", ""),
        "chunk": metadata.get("chunk", ""),
        "section": metadata.get("section", "unknown"),
        "content_type": metadata.get("content_type", "text"),
        "paper_id": metadata.get("paper_id", ""),
        "title": metadata.get("title") or metadata.get("source", ""),
        "authors": metadata.get("authors", ""),
        "year": metadata.get("year", ""),
        "task": metadata.get("task", ""),
        "methods": metadata.get("methods", ""),
        "dataset": metadata.get("dataset", ""),
        "notes": metadata.get("notes", ""),
        "distance": round(float(distance), 4) if distance is not None else "",
        "retrieval_source": source,
        "vector_score": max(0.0, 1.0 - float(distance)) if distance is not None else 0.0,
        "keyword_score": 0.0,
    }


def vector_retrieve(
    query: str,
    collection,
    embedding_model: str,
    candidate_k: int,
    filters: dict,
) -> list[dict]:
    query_embedding = embed_texts([query], model=embedding_model, batch_size=1)[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        ctx = metadata_to_context(document, metadata, distance=distance, source="vector")
        if passes_filters(ctx, filters):
            contexts.append(ctx)
    return contexts


def keyword_retrieve(query: str, collection, candidate_k: int, filters: dict, query_profile: dict) -> list[dict]:
    weighted_terms = query_profile["weighted_terms"]
    if not weighted_terms:
        return []

    result = collection.get(include=["documents", "metadatas"])
    scored = []
    for document, metadata in zip(result["documents"], result["metadatas"]):
        ctx = metadata_to_context(document, metadata, source="keyword")
        if not passes_filters(ctx, filters):
            continue

        searchable = " ".join(
            [
                ctx["title"],
                ctx["source"],
                ctx["task"],
                ctx["methods"],
                ctx["dataset"],
                ctx["section"],
                document,
            ]
        ).lower().replace("_", "-")
        metadata_text = " ".join(
            [ctx["title"], ctx["source"], ctx["methods"], ctx["task"], ctx["dataset"]]
        ).lower().replace("_", "-")
        document_lower = document.lower().replace("_", "-")
        score = 0.0
        for term, weight in weighted_terms.items():
            if term in document_lower:
                score += 3.0 * weight
            if term in metadata_text:
                score += 2.0 * weight
            elif term in searchable:
                score += 0.8 * weight

        coverage = term_coverage(ctx, query_profile)
        evidence = evidence_score(ctx, query_profile)
        score += 4.0 * coverage
        score += evidence

        if query_profile["intent"] == "comparison" and coverage < 0.45:
            score *= 0.55
        if LOW_VALUE_CHUNK_RE.search(document_lower):
            score *= 0.15
        if score > 0:
            ctx["keyword_score"] = score
            ctx["coverage_score"] = round(coverage, 4)
            ctx["evidence_score"] = round(evidence, 4)
            scored.append(ctx)

    scored.sort(key=lambda item: item["keyword_score"], reverse=True)
    max_score = scored[0]["keyword_score"] if scored else 1.0
    for item in scored:
        item["keyword_score"] = item["keyword_score"] / max_score
    return scored[:candidate_k]


def merge_contexts(
    vector_contexts: list[dict],
    keyword_contexts: list[dict],
    candidate_k: int,
    query_profile: dict,
) -> list[dict]:
    merged = {}
    for ctx in vector_contexts + keyword_contexts:
        key = context_key(ctx)
        existing = merged.get(key)
        if existing is None:
            merged[key] = ctx
            continue
        existing["vector_score"] = max(existing.get("vector_score", 0.0), ctx.get("vector_score", 0.0))
        existing["keyword_score"] = max(existing.get("keyword_score", 0.0), ctx.get("keyword_score", 0.0))
        existing["coverage_score"] = max(existing.get("coverage_score", 0.0), ctx.get("coverage_score", 0.0))
        existing["evidence_score"] = max(existing.get("evidence_score", 0.0), ctx.get("evidence_score", 0.0))
        sources = set(str(existing.get("retrieval_source", "")).split("+"))
        sources.update(str(ctx.get("retrieval_source", "")).split("+"))
        existing["retrieval_source"] = "+".join(sorted(source for source in sources if source))

    contexts = list(merged.values())
    for ctx in contexts:
        ctx["coverage_score"] = ctx.get("coverage_score", term_coverage(ctx, query_profile))
        ctx["evidence_score"] = ctx.get("evidence_score", evidence_score(ctx, query_profile))
        ctx["combined_score"] = (
            0.45 * ctx.get("vector_score", 0.0)
            + 0.30 * ctx.get("keyword_score", 0.0)
            + 0.15 * ctx.get("coverage_score", 0.0)
            + 0.10 * min(ctx.get("evidence_score", 0.0) / 5.0, 1.0)
        )
    contexts.sort(
        key=lambda item: (
            item.get("combined_score", 0.0),
            item.get("coverage_score", 0.0),
            item.get("evidence_score", 0.0),
            item.get("keyword_score", 0.0),
        ),
        reverse=True,
    )
    return contexts[:candidate_k]


def retrieve(
    query: str,
    db_dir: Path,
    collection_name: str,
    embedding_model: str,
    candidate_k: int,
    filters: dict,
    use_hybrid: bool,
    keyword_only: bool = False,
    debug: bool = False,
    query_profile: dict | None = None,
) -> list[dict]:
    query_profile = query_profile or analyze_query(query)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    if debug:
        print("\nDEBUG")
        print("=" * 80)
        print(f"collection_count={collection.count()}")
        print(f"query={query}")
        print(f"filters={filters}")
    vector_contexts = []
    if keyword_only:
        if debug:
            print("keyword_only=True")
    elif use_hybrid and not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY is not set, using keyword retrieval only.")
    else:
        try:
            vector_contexts = vector_retrieve(
                query=query,
                collection=collection,
                embedding_model=embedding_model,
                candidate_k=candidate_k * 2,
                filters=filters,
            )
        except BaseException as exc:
            if not use_hybrid:
                raise
            print(f"\nWARNING: vector retrieval failed, falling back to keyword search: {exc}")
    if not use_hybrid and not keyword_only:
        if debug:
            print(f"vector_contexts={len(vector_contexts)}")
        return vector_contexts[:candidate_k]
    keyword_contexts = keyword_retrieve(
        query=query,
        collection=collection,
        candidate_k=candidate_k * 2,
        filters=filters,
        query_profile=query_profile,
    )
    merged = merge_contexts(vector_contexts, keyword_contexts, candidate_k, query_profile)
    if debug:
        print(f"vector_contexts={len(vector_contexts)}")
        print(f"keyword_contexts={len(keyword_contexts)}")
        print(f"merged_contexts={len(merged)}")
        print(f"query_profile={json.dumps(query_profile, ensure_ascii=False)}")
        for i, ctx in enumerate(merged[:5], start=1):
            print(
                f"candidate[{i}] source={ctx['source']} distance={ctx.get('distance')} "
                f"keyword={ctx.get('keyword_score', 0):.3f} coverage={ctx.get('coverage_score', 0):.3f} "
                f"evidence={ctx.get('evidence_score', 0):.3f} via={ctx.get('retrieval_source')}"
            )
    return merged


def rewrite_query(question: str, model: str) -> str:
    prompt = (
        "Rewrite the user's question into a compact bilingual search query for retrieving "
        "academic papers in the same technical domain.\n"
        "Keep important technical terms, acronyms, method names, datasets, and task names.\n"
        "Include both Chinese and English expressions when useful.\n"
        "Do not add tasks, methods, datasets, or acronyms that are not implied by the user's question.\n"
        "For example, do not add RUL prediction unless the user asks about life prediction, prognosis, or RUL.\n"
        "Do not use Boolean operators such as AND, OR, NOT. Do not use parentheses.\n"
        "Prefer space-separated keywords and short phrases.\n"
        "Return only the rewritten search query, no explanation.\n\n"
        f"User question: {question}"
    )
    return normalize_search_query(chat_completion(prompt, model=model).strip().strip('"'))


def _parse_ranked_indices(text: str) -> list[int]:
    match = re.search(r"\[[\d,\s]+]", text)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [int(value) for value in values if isinstance(value, int)]


def rerank_contexts(question: str, contexts: list[dict], model: str, top_k: int, query_profile: dict) -> list[dict]:
    if len(contexts) <= top_k:
        return contexts

    candidate_text = "\n\n".join(
        (
            f"[{i}] paper_id={ctx.get('paper_id')} title={ctx.get('title')} "
            f"source={ctx['source']} page={ctx['page']} section={ctx.get('section')} "
            f"methods={ctx.get('methods')} distance={ctx.get('distance')} "
            f"keyword_score={ctx.get('keyword_score', 0):.3f} "
            f"coverage_score={ctx.get('coverage_score', 0):.3f} "
            f"evidence_score={ctx.get('evidence_score', 0):.3f}\n"
            f"{ctx['text'][:900]}"
        )
        for i, ctx in enumerate(contexts, start=1)
    )
    prompt = (
        "You are reranking retrieved paper chunks for a RAG system.\n"
        "Select the chunks that most directly help answer the user's question, using the query profile below.\n"
        "Prefer chunks that cover more technical constraints from the query profile.\n"
        "Prefer evidence-bearing chunks: method details, experiments, result tables, metrics, comparisons, conclusions, or definitions.\n"
        "Demote chunks that are cover pages, acknowledgements, tables of contents, references, or generic background.\n"
        "Do not prefer a chunk only because its title is similar; the chunk text itself should support the answer.\n"
        "Return only a JSON array of candidate numbers in best-to-worst order, "
        f"with at most {top_k} numbers. Example: [3, 1, 7]\n\n"
        f"Query profile: {json.dumps(query_profile, ensure_ascii=False)}\n\n"
        f"User question: {question}\n\n"
        f"Candidates:\n{candidate_text}"
    )
    ranked_indices = _parse_ranked_indices(chat_completion(prompt, model=model))

    selected = []
    seen = set()
    for index in ranked_indices:
        if 1 <= index <= len(contexts) and index not in seen:
            selected.append(contexts[index - 1])
            seen.add(index)
        if len(selected) >= top_k:
            break

    for index, ctx in enumerate(contexts, start=1):
        if len(selected) >= top_k:
            break
        if index not in seen:
            selected.append(ctx)
            seen.add(index)

    return selected


def select_contexts(contexts: list[dict], top_k: int, max_per_paper: int) -> list[dict]:
    selected = []
    per_paper = defaultdict(int)
    overflow = []
    for ctx in contexts:
        source = ctx.get("source", "")
        if per_paper[source] < max_per_paper:
            selected.append(ctx)
            per_paper[source] += 1
        else:
            overflow.append(ctx)
        if len(selected) >= top_k:
            return selected

    for ctx in overflow:
        selected.append(ctx)
        if len(selected) >= top_k:
            break
    return selected


def build_prompt(question: str, contexts: list[dict], query_profile: dict) -> str:
    context_text = "\n\n".join(
        (
            f"[C{i:02d}] Paper: {ctx.get('title')}\n"
            f"Paper ID: {ctx.get('paper_id') or 'unknown'}; Authors: {ctx.get('authors') or 'unknown'}; "
            f"Year: {ctx.get('year') or 'unknown'}\n"
            f"Source: {ctx['source']}, page {ctx['page']}, chunk {ctx['chunk']}, "
            f"section: {ctx.get('section')}, content_type: {ctx.get('content_type')}\n"
            f"Task: {ctx.get('task') or 'unknown'}; Methods: {ctx.get('methods') or 'unknown'}; "
            f"Dataset: {ctx.get('dataset') or 'unknown'}\n"
            f"Text: {ctx['text']}"
        )
        for i, ctx in enumerate(contexts, start=1)
    )
    return (
        "你是一个PHM、轴承故障诊断和RUL预测方向的研究助手。\n"
        "请严格基于给定资料回答问题，不要编造资料中没有的信息。\n"
        "如果资料只支持部分结论，请明确说明哪些结论有依据、哪些信息不足。\n"
        "请区分论文明确说明的内容和你根据资料做出的合理推断。\n"
        "不得仅根据论文标题、文件名或摘要中的泛泛表述推断实验结论。\n"
        "只有当资料中出现明确方法描述、实验设置、结果、数值、表格或结论时，才可以给出相应结论。\n"
        "引用格式要求：在每个关键结论后标注来源编号，例如[C01]。这些编号是本次检索来源编号，不是论文参考文献编号。\n"
        "请优先做多来源对比，而不是只总结单个片段。\n"
        "固定输出结构：\n"
        "1. 相关论文\n"
        "2. 方法/模型\n"
        "3. 数据集或实验设置\n"
        "4. 主要结论对比\n"
        "5. 局限与资料不足\n\n"
        f"用户问题分析：{json.dumps(query_profile, ensure_ascii=False)}\n\n"
        f"资料：\n{context_text}\n\n"
        f"问题：{question}\n\n"
        "请用中文回答，结构清晰，避免空泛总结。"
    )


def print_sources(contexts: list[dict], pdf_dir: Path) -> None:
    print("\nSOURCES")
    print("=" * 80)
    grouped = defaultdict(list)
    for i, ctx in enumerate(contexts, start=1):
        grouped[ctx["source"]].append((i, ctx))

    for source, items in grouped.items():
        first = items[0][1]
        title = first.get("title") or source
        paper_id = first.get("paper_id") or "unknown"
        year = first.get("year") or "unknown"
        methods = first.get("methods") or "unknown"
        pdf_path = pdf_dir / source
        print(f"{paper_id} | {title} | year={year}")
        print(f"file={source}")
        print(f"pdf={pdf_path}")
        print(f"methods={methods}")
        for i, ctx in items:
            snippet = ctx["text"][:220].replace("\n", " ")
            print(
                f"  [C{i:02d}] page={ctx['page']} chunk={ctx['chunk']} "
                f"section={ctx.get('section')} distance={ctx.get('distance')} "
                f"keyword={ctx.get('keyword_score', 0):.3f} "
                f"coverage={ctx.get('coverage_score', 0):.3f} "
                f"evidence={ctx.get('evidence_score', 0):.3f} "
                f"combined={ctx.get('combined_score', 0):.3f} "
                f"via={ctx.get('retrieval_source')}"
            )
            print(f"       excerpt={snippet}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against the Chroma paper vector database.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--db-dir", default="data/chroma", help="Persistent Chroma database directory.")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="PDF root directory used for source paths.")
    parser.add_argument("--collection", default="phm_papers", help="Chroma collection name.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--chat-model", default="gpt-4.1-mini", help="OpenAI chat model.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks used in the final answer.")
    parser.add_argument("--candidate-k", type=int, default=20, help="Number of hybrid-search candidates before reranking.")
    parser.add_argument("--max-per-paper", type=int, default=2, help="Maximum final chunks from the same paper before overflow.")
    parser.add_argument("--task", default="", help="Filter by task metadata, e.g. 'RUL prediction'.")
    parser.add_argument("--method", default="", help="Filter by method metadata, e.g. 'CNN' or 'Monte Carlo'.")
    parser.add_argument("--year", default="", help="Filter by exact year metadata.")
    parser.add_argument("--title", default="", help="Filter by title substring.")
    parser.add_argument("--source", default="", help="Filter by source filename substring.")
    parser.add_argument("--contains", default="", help="Filter by substring in metadata or chunk text.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable keyword+vector hybrid retrieval.")
    parser.add_argument("--keyword-only", action="store_true", help="Use local keyword retrieval only; no embedding API call.")
    parser.add_argument("--no-query-rewrite", action="store_true", help="Disable LLM query rewriting.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM reranking.")
    parser.add_argument("--retrieve-only", action="store_true", help="Print retrieved sources/contexts without calling the chat model.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval diagnostics.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context snippets.")
    args = parser.parse_args()

    query_profile = analyze_query(args.question)
    search_query = args.question
    if not args.no_query_rewrite and not args.keyword_only and not args.retrieve_only:
        search_query = rewrite_query(args.question, model=args.chat_model)
        print("\nREWRITTEN QUERY")
        print("=" * 80)
        print(search_query)
    search_query = normalize_search_query(search_query)

    contexts = retrieve(
        query=search_query,
        db_dir=Path(args.db_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        candidate_k=max(args.candidate_k, args.top_k),
        filters={
            "task": args.task,
            "method": args.method,
            "year": args.year,
            "title": args.title,
            "source": args.source,
            "contains": args.contains,
        },
        use_hybrid=not args.no_hybrid,
        keyword_only=args.keyword_only,
        debug=args.debug,
        query_profile=query_profile,
    )
    if not contexts:
        raise SystemExit("No matching context found. Try relaxing filters or rebuilding the vector database.")

    selection_pool_k = max(args.top_k * 2, args.top_k)
    if not args.no_rerank and not args.retrieve_only:
        contexts = rerank_contexts(
            args.question,
            contexts,
            model=args.chat_model,
            top_k=selection_pool_k,
            query_profile=query_profile,
        )
    else:
        contexts = contexts[:selection_pool_k]
    contexts = select_contexts(contexts, top_k=args.top_k, max_per_paper=args.max_per_paper)

    if args.retrieve_only:
        print_sources(contexts, pdf_dir=Path(args.pdf_dir))
        if args.show_context:
            print("\nCONTEXTS")
            print("=" * 80)
            for i, ctx in enumerate(contexts, start=1):
                preview = ctx["text"][:900].replace("\n", " ")
                print(f"[C{i:02d}] {preview}\n")
        return

    prompt = build_prompt(args.question, contexts, query_profile=query_profile)
    answer = chat_completion(prompt, model=args.chat_model)

    print("\nANSWER")
    print("=" * 80)
    print(answer)
    print_sources(contexts, pdf_dir=Path(args.pdf_dir))

    if args.show_context:
        print("\nCONTEXTS")
        print("=" * 80)
        for i, ctx in enumerate(contexts, start=1):
            preview = ctx["text"][:900].replace("\n", " ")
            print(f"[{i}] {preview}\n")


if __name__ == "__main__":
    main()
