"""
Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
Joint query-passage attention model that re-ranks top candidates.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Joint query-document cross-encoder for precision candidate reranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.model: Optional[CrossEncoder] = None

    def _load_model(self):
        if self.model is None:
            try:
                self.model = CrossEncoder(self.model_name, device=self.device, model_kwargs={"local_files_only": True})
            except Exception:
                try:
                    self.model = CrossEncoder(self.model_name, device=self.device)
                except Exception as e:
                    print(f"[Warning] Could not load CrossEncoder {self.model_name}: {e}")
                    raise

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate chunks, forms (query, chunk_text) pairs,
        scores with cross-encoder, and returns top_k re-ordered chunks.
        """
        if not candidate_chunks:
            return []
        self._load_model()

        start_time = time.time()
        pairs = [[query, c.get("text", "")] for c in candidate_chunks]
        scores = self.model.predict(pairs)
        rerank_latency_ms = (time.time() - start_time) * 1000.0

        scored_candidates = []
        for i, chunk in enumerate(candidate_chunks):
            c = dict(chunk)
            c["pre_rerank_rank"] = i + 1
            c["cross_encoder_score"] = float(scores[i])
            scored_candidates.append(c)

        scored_candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        results = []
        for rank, c in enumerate(scored_candidates[:top_k], start=1):
            c["final_rank"] = rank
            c["rerank_latency_ms"] = rerank_latency_ms
            c["rank_changed"] = (c["pre_rerank_rank"] != rank)
            results.append(c)

        return results
