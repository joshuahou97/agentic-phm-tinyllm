from __future__ import annotations

import json
import re
from pathlib import Path


def normalize_source(source: str) -> str:
    return source.replace("\\", "/")


def infer_title_from_source(source: str) -> str:
    name = Path(source).stem
    return re.sub(r"\s+", " ", name).strip()


def load_paper_metadata(metadata_path: Path | None) -> dict[str, dict]:
    if metadata_path is None or not metadata_path.exists():
        return {}

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    papers = data.get("papers", data if isinstance(data, list) else [])
    metadata = {}
    for item in papers:
        source = normalize_source(item["file"])
        metadata[source] = {
            "paper_id": item.get("paper_id", ""),
            "title": item.get("title") or infer_title_from_source(source),
            "authors": item.get("authors", ""),
            "year": item.get("year", ""),
            "task": item.get("task", ""),
            "methods": ", ".join(item.get("methods", [])) if isinstance(item.get("methods"), list) else item.get("methods", ""),
            "dataset": item.get("dataset", ""),
            "notes": item.get("notes", ""),
        }
    return metadata


def metadata_for_source(source: str, metadata: dict[str, dict]) -> dict:
    source = normalize_source(source)
    item = metadata.get(source, {})
    return {
        "paper_id": item.get("paper_id", ""),
        "title": item.get("title") or infer_title_from_source(source),
        "authors": item.get("authors", ""),
        "year": item.get("year", ""),
        "task": item.get("task", ""),
        "methods": item.get("methods", ""),
        "dataset": item.get("dataset", ""),
        "notes": item.get("notes", ""),
    }
