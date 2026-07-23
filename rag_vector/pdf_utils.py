from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


SECTION_PATTERNS = [
    ("abstract", r"\babstract\b|摘要"),
    ("introduction", r"\bintroduction\b|引言"),
    ("related work", r"\brelated work\b|literature review|相关工作"),
    ("method", r"\bmethod(?:ology)?\b|\bapproach\b|\bframework\b|\bmodel(?:ing)?\b|方法|模型|框架"),
    ("experiment", r"\bexperiment(?:s|al)?\b|\bevaluation\b|\bcase study\b|实验|验证"),
    ("result", r"\bresult(?:s)?\b|\bdiscussion\b|结果|讨论|分析"),
    ("conclusion", r"\bconclusion(?:s)?\b|\bsummary\b|总结|结论"),
]

REFERENCES_RE = re.compile(r"(?i)(^|\s)(references|bibliography|参考文献)\s*[:：]?\s*")
FIGURE_TABLE_RE = re.compile(r"(?i)\b(fig\.?|figure|table)\s*\d+|图\s*\d+|表\s*\d+")
FORMULA_RE = re.compile(r"=\s*[\w({\[]|∑|Σ|∫|√|≤|≥|≈")


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def strip_references(text: str) -> str:
    match = REFERENCES_RE.search(text)
    if not match:
        return text
    return text[: match.start()].strip()


def detect_section(text: str, current_section: str = "unknown") -> str:
    head = text[:600].lower()
    for section, pattern in SECTION_PATTERNS:
        if re.search(pattern, head, flags=re.IGNORECASE):
            return section
    return current_section


def detect_content_type(text: str) -> str:
    figure_table_count = len(FIGURE_TABLE_RE.findall(text))
    formula_count = len(FORMULA_RE.findall(text))
    if figure_table_count >= 2:
        return "figure/table"
    if formula_count >= 3:
        return "formula-heavy"
    return "text"


def _split_tokens(text: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", text):
        # Keep Chinese characters searchable even when the PDF has no spaces.
        return re.findall(r"[A-Za-z0-9_+\-./]+|[\u4e00-\u9fff]", text)
    return text.split()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = _split_tokens(text)
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def load_pdf_chunks(pdf_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {pdf_dir}")

    chunks = []
    for pdf_path in pdf_paths:
        print(f"Reading {pdf_path.relative_to(pdf_dir)}")
        current_section = "unknown"
        for page in extract_pdf_pages(pdf_path):
            page_text = strip_references(page["text"])
            if not page_text:
                continue
            current_section = detect_section(page_text, current_section)
            page_chunks = chunk_text(page_text, chunk_size, overlap)
            for chunk_index, chunk in enumerate(page_chunks, start=1):
                section = detect_section(chunk, current_section)
                current_section = section
                chunks.append(
                    {
                        "id": f"{len(chunks):08d}",
                        "source": str(pdf_path.relative_to(pdf_dir)),
                        "page": page["page"],
                        "chunk": chunk_index,
                        "section": section,
                        "content_type": detect_content_type(chunk),
                        "text": chunk,
                    }
                )
    return chunks
