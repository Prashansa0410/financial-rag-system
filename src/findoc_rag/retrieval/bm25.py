"""
BM25 lexical keyword search using rank_bm25.BM25Okapi.
"""

from __future__ import annotations
import re
import time
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Lexical keyword search retriever using BM25Okapi."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Builds BM25 index over chunk text."""
        self.chunks = chunks
        self.tokenized_corpus = [simple_tokenize(c.get("text", "")) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches BM25 index for top_k most relevant chunks."""
        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized. Call build_index first.")

        start_time = time.time()
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked_indices = scores.argsort()[::-1]
        latency_ms = (time.time() - start_time) * 1000.0

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            if filter_doc and filter_doc.lower() not in chunk.get("doc_name", "").lower():
                continue
            chunk["score"] = float(scores[idx])
            chunk["bm25_rank"] = rank
            chunk["search_method"] = "bm25"
            chunk["query_latency_ms"] = latency_ms
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
