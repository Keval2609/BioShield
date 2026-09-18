"""Ingestion pipeline for agricultural documents into ChromaDB."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings

from src import config
from src.chunking import Chunk, clean_text, chunk_page_text
from src.document_catalog import (
    CORE_DOCUMENTS,
    get_document_metadata,
    is_core_document,
)
from src.embeddings import EmbeddingModel

logger = logging.getLogger("bioshield.ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_pages_from_pdf(file_path: Path) -> List[Tuple[int, str]]:
    """Extract non-empty pages from a PDF file with page numbers (1-indexed).
    
    Returns:
        List of (page_number, text) tuples.
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    if file_path.suffix.lower() != ".pdf":
        logger.warning(f"Expected a PDF file, got: {file_path}")
        return []

    results: List[Tuple[int, str]] = []
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        num_pages = len(reader.pages)
        if num_pages == 0:
            logger.warning(f"PDF contains no pages: {file_path}")
            return []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                cleaned = clean_text(page_text)
                if cleaned:
                    results.append((page_idx, page_text))
            except Exception as page_err:
                logger.warning(f"Failed to extract page {page_idx} of {file_path.name}: {page_err}")
                continue

    except Exception as exc:
        logger.error(f"Error opening PDF {file_path}: {exc}")
        return []

    return results


def get_chroma_client(persist_path: str) -> chromadb.ClientAPI:
    """Get or create a PersistentClient with standardized settings."""
    return chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
    )


def ingest_documents(
    resources_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    target_files: Optional[List[str]] = None,
    chunk_size: int = 500,
    overlap: int = 50,
    max_pages: Optional[int] = None,
    embedding_model: Optional[EmbeddingModel] = None,
) -> int:
    """Read core documents, chunk page-by-page, embed, and store idempotently in ChromaDB.
    
    By default, only ingests the 5 CORE_DOCUMENTS.
    Returns the total number of new chunks added.
    """
    res_path = resources_dir or config.CORE_RESOURCES_DIR
    persist_path = str(persist_dir or config.CHROMA_PERSIST_DIR)
    coll_name = collection_name or config.CHROMA_COLLECTION_NAME

    if not res_path.exists():
        logger.warning(f"Resources directory does not exist: {res_path}")
        return 0

    # Determine files to ingest: default to core documents
    if target_files is not None:
        files_to_process = [res_path / fname for fname in target_files if (res_path / fname).exists()]
    else:
        # Strictly ingest the 5 core documents by default
        files_to_process = [res_path / fname for fname in CORE_DOCUMENTS.keys() if (res_path / fname).exists()]

    if not files_to_process:
        logger.warning(f"No matching documents found in {res_path} to ingest.")
        return 0

    logger.info(f"Connecting to ChromaDB at: {persist_path}")
    client = get_chroma_client(persist_path)
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Fetch existing IDs to ensure idempotency
    existing_count = collection.count()
    existing_ids = set()
    if existing_count > 0:
        res = collection.get(include=[])
        if res and "ids" in res:
            existing_ids = set(res["ids"])

    embedder = embedding_model or EmbeddingModel()
    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    logger.info(f"Processing {len(files_to_process)} document(s)...")
    for file_path in files_to_process:
        doc_meta = get_document_metadata(file_path.name)
        doc_id = doc_meta.get("document_id", file_path.stem.lower())
        logger.info(f"Extracting pages for: {file_path.name}")
        pages = extract_pages_from_pdf(file_path)
        if max_pages is not None:
            pages = pages[:max_pages]


        for page_num, page_text in pages:
            chunks = chunk_page_text(
                page_text=page_text,
                page_number=page_num,
                base_metadata=doc_meta,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for c_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{c_idx}"
                if chunk_id not in existing_ids:
                    all_ids.append(chunk_id)
                    all_chunks.append(chunk.text)
                    # Sanitize metadata for ChromaDB (types: str, int, float, bool)
                    sanitized_meta = {
                        k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                        for k, v in chunk.metadata.items()
                    }
                    all_metas.append(sanitized_meta)

    if not all_ids:
        logger.info(
            f"All documents already indexed in collection '{coll_name}' ({existing_count} total chunks). No new chunks to add."
        )
        return 0

    logger.info(f"Generating embeddings for {len(all_chunks)} new chunks...")
    embeddings = embedder.embed_texts(all_chunks)


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
    logger.info(
        f"Successfully added {len(all_ids)} new chunk(s). Collection '{coll_name}' now contains {total_chunks} chunks."
    )
    return len(all_ids)


if __name__ == "__main__":
    print("=== BioShield AI Knowledge Base Ingestion ===")
    ingest_documents()
