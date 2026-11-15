"""
Hybrid retrieval combining Dense Vector Search and BM25 using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from .dense import DenseRetriever
from .bm25 import BM25Retriever


class HybridRetriever:
    """Hybrid search combining dense and lexical retrievers via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        pool_size: int = 20,
        filter_doc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Dense and BM25 search, then fuses rankings with RRF:
        RRF_score(d) = 1 / (k + rank_dense) + 1 / (k + rank_bm25)
        """
        start_time = time.time()

        dense_results = self.dense.search(query, top_k=pool_size, filter_doc=filter_doc)
        bm25_results = self.bm25.search(query, top_k=pool_size, filter_doc=filter_doc)

        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for rank, c in enumerate(dense_results, start=1):
            cid = c["chunk_id"]
            dense_ranks[cid] = rank
            chunk_map[cid] = c
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, c in enumerate(bm25_results, start=1):
            cid = c["chunk_id"]
            bm25_ranks[cid] = rank
            chunk_map[cid] = c
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        latency_ms = (time.time() - start_time) * 1000.0

        results = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            chunk = dict(chunk_map[cid])
            chunk["score"] = float(fused_scores[cid])
            chunk["hybrid_rank"] = rank
            chunk["dense_rank"] = dense_ranks.get(cid, None)
            chunk["bm25_rank"] = bm25_ranks.get(cid, None)
            chunk["search_method"] = "hybrid_rrf"
            chunk["query_latency_ms"] = latency_ms
            results.append(chunk)

        return results
