#!/usr/bin/env python
"""BioShield AI - Knowledge Base Ingestion Script.

Ingests core natural farming documents into local ChromaDB with
page-aware and section-aware chunking and complete source traceability.

Usage:
    python ingest.py
    python ingest.py --core-only
    python ingest.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config
from src.document_catalog import CORE_DOCUMENTS
from src.ingest import ingest_documents

# Ensure Windows terminal handles UTF-8 characters safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")



def main():
    parser = argparse.ArgumentParser(
        description="BioShield AI Knowledge Base Ingestion Pipeline"
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        default=True,
        help="Ingest only the 5 core natural farming documents (default: True)",
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=config.CORE_RESOURCES_DIR,
        help="Path to directory containing agricultural PDF resources",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=config.CHROMA_PERSIST_DIR,
        help="Path to ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=config.CHROMA_COLLECTION_NAME,
        help="Name of ChromaDB collection",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in words (~400-700 tokens, default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in words (~50-100 tokens, default: 50)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("[BioShield AI] Natural Farming Knowledge Base Ingestion")
    print("=" * 65)

    print(f"Resources Directory : {args.resources_dir}")
    print(f"ChromaDB Persistence: {args.persist_dir}")
    print(f"Collection Name     : {args.collection}")
    print(f"Embedding Model     : {config.EMBEDDING_MODEL}")
    print(f"Core Documents      : {len(CORE_DOCUMENTS)} files")
    for idx, (fname, meta) in enumerate(CORE_DOCUMENTS.items(), start=1):
        status = "FOUND" if (args.resources_dir / fname).exists() else "MISSING"
        print(f"  {idx}. [{status}] {fname} ({meta['document_title']})")
    print("-" * 65)

    num_added = ingest_documents(
        resources_dir=args.resources_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("-" * 65)
    print(f"Ingestion completed. {num_added} new chunks indexed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
