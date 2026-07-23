from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from embedding import embed_texts
from metadata import load_paper_metadata, metadata_for_source
from pdf_utils import load_pdf_chunks


def build_vector_db(
    pdf_dir: Path,
    db_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    metadata_path: Path | None,
) -> None:
    chunks = load_pdf_chunks(pdf_dir, chunk_size, overlap)
    print(f"Loaded {len(chunks)} chunks. Creating embeddings with {embedding_model}...")
    paper_metadata = load_paper_metadata(metadata_path)

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = embed_texts(texts, model=embedding_model, batch_size=batch_size)
        collection.add(
            ids=[item["id"] for item in batch],
            documents=texts,
            metadatas=[
                {
                    "source": item["source"],
                    "page": item["page"],
                    "chunk": item["chunk"],
                    "section": item["section"],
                    "content_type": item["content_type"],
                    **metadata_for_source(item["source"], paper_metadata),
                }
                for item in batch
            ],
            embeddings=embeddings,
        )
        print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"Vector DB written to {db_dir}")
    print(f"Collection: {collection_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Chroma vector database from PDF papers.")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Directory containing PDF files.")
    parser.add_argument("--db-dir", default="data/chroma", help="Persistent Chroma database directory.")
    parser.add_argument("--collection", default="phm_papers", help="Chroma collection name.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--chunk-size", type=int, default=260, help="Chunk size measured in whitespace tokens.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap measured in whitespace tokens.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding and Chroma insert batch size.")
    parser.add_argument("--metadata", default="data/paper_metadata.json", help="Optional paper metadata JSON file.")
    args = parser.parse_args()

    build_vector_db(
        pdf_dir=Path(args.pdf_dir),
        db_dir=Path(args.db_dir),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        metadata_path=Path(args.metadata) if args.metadata else None,
    )


if __name__ == "__main__":
    main()
