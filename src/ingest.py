"""Ingestion pipeline for agricultural documents into ChromaDB."""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import chromadb
from chromadb.config import Settings

from src import config
from src.chunking import clean_text, chunk_text
from src.embeddings import EmbeddingModel


def extract_text_from_file(file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text from a .txt or .pdf file, returning (text, metadata) per page or section.
    
    Supports .txt and .pdf files.
    """
    results: List[Tuple[str, Dict[str, Any]]] = []
    base_meta = {
        "filename": file_path.name,
        "title": file_path.stem.replace("_", " ").title(),
        "source": "Agricultural Extension / Research Document",
    }

    if file_path.suffix.lower() == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                meta = dict(base_meta)
                meta["page"] = 1
                results.append((content, meta))
        except Exception as exc:
            print(f"Warning: Failed to read text file {file_path}: {exc}")

    elif file_path.suffix.lower() == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    meta = dict(base_meta)
                    meta["page"] = page_idx
                    results.append((page_text, meta))
        except ImportError:
            print("Warning: pypdf is not installed. PDF extraction skipped.")
        except Exception as exc:
            print(f"Warning: Failed to read PDF file {file_path}: {exc}")

    return results


def ingest_documents(
    raw_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> int:
    """Read documents from raw_dir, chunk, embed, and store idempotently in ChromaDB.
    
    Returns the total number of chunks added.
    """
    raw_path = raw_dir or config.DATA_RAW_DIR
    persist_path = str(persist_dir or config.CHROMA_PERSIST_DIR)
    coll_name = collection_name or config.CHROMA_COLLECTION_NAME

    if not raw_path.exists():
        print(f"Raw data directory does not exist: {raw_path}")
        return 0

    files = [f for f in raw_path.iterdir() if f.suffix.lower() in [".txt", ".pdf"]]
    if not files:
        print(f"No .txt or .pdf files found in {raw_path}")
        return 0

    print(f"Connecting to ChromaDB at: {persist_path}")
    client = chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Fetch existing IDs to ensure idempotency
    existing_count = collection.count()
    existing_ids = set()
    if existing_count > 0:
        # Retrieve all IDs in batches if collection is already populated
        res = collection.get(include=[])
        if res and "ids" in res:
            existing_ids = set(res["ids"])

    embedding_model = EmbeddingModel()
    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    print(f"Processing {len(files)} document(s) from {raw_path}...")
    for file_path in files:
        pages = extract_text_from_file(file_path)
        for page_text, page_meta in pages:
            chunks = chunk_text(
                page_text,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=page_meta,
            )
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_p{page_meta.get('page', 1)}_c{chunk_idx}"
                if chunk_id not in existing_ids:
                    all_ids.append(chunk_id)
                    all_chunks.append(chunk.text)
                    # ChromaDB metadata values must be str, int, float, or bool
                    clean_meta = {
                        k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                        for k, v in chunk.metadata.items()
                    }
                    all_metas.append(clean_meta)

    if not all_ids:
        print(f"All documents already indexed in collection '{coll_name}' ({existing_count} total chunks).")
        return 0

    print(f"Generating embeddings for {len(all_chunks)} new chunks...")
    embeddings = embedding_model.embed_texts(all_chunks)

    # Ingest into ChromaDB in batches of 100
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        end_idx = i + batch_size
        collection.add(
            ids=all_ids[i:end_idx],
            documents=all_chunks[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=all_metas[i:end_idx],
        )

    total_chunks = collection.count()
    print(f"Successfully added {len(all_ids)} new chunk(s). Collection '{coll_name}' now contains {total_chunks} chunks.")
    return len(all_ids)


if __name__ == "__main__":
    print("=== BioShield AI Knowledge Base Ingestion ===")
    ingest_documents()
