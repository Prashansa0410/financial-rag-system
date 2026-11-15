"""
Document ingestion and sliding-window token chunking for SEC financial filings.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def load_corpus(corpus_dir: Path) -> Dict[str, str]:
    """Loads all raw text SEC filings from the corpus directory."""
    documents = {}
    if not corpus_dir.is_dir():
        return documents
    for file_path in sorted(corpus_dir.glob("*.txt")):
        if file_path.name.startswith("."):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            documents[file_path.name] = f.read()
    return documents


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    doc_name: str = "document.txt"
) -> List[Dict[str, Any]]:
    """
    Splits text into sliding-window token chunks with overlap.
    Tokens are approximated by whitespace boundaries (or subwords).
    """
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    doc_stem = Path(doc_name).stem

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        chunk_id = f"{doc_stem}_c{chunk_size}_{len(chunks)}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "text": chunk_text_str,
            "token_count": len(chunk_words),
            "start_idx": i,
            "end_idx": i + len(chunk_words)
        })
        if i + chunk_size >= len(words):
            break

    return chunks


def load_precomputed_chunks(chunks_file: Path) -> List[Dict[str, Any]]:
    """Loads pre-built chunks from CSV (e.g., chunks_100.csv or chunks_400.csv)."""
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    df = pd.read_csv(chunks_file)
    records = df.to_dict(orient="records")
    # Normalize text / chunk_text key
    for r in records:
        if "text" not in r and "chunk_text" in r:
            r["text"] = str(r["chunk_text"])
        elif "chunk_text" not in r and "text" in r:
            r["chunk_text"] = str(r["text"])
    return records
