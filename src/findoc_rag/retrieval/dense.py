"""
Dense semantic search using BAAI/bge-small-en-v1.5 and FAISS IndexFlatIP.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Vector-based dense retriever using BAAI/bge-small-en-v1.5 and FAISS."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        dimension: int = 384
    ):
        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    def _load_model(self):
        if self.model is None:
            try:
                # Try loading with local files first
                self.model = SentenceTransformer(self.model_name, device=self.device, model_kwargs={"local_files_only": True})
            except Exception:
                try:
                    self.model = SentenceTransformer(self.model_name, device=self.device)
                except Exception as e:
                    print(f"[Warning] Could not load SentenceTransformer {self.model_name}: {e}")
                    raise

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 64
    ) -> Tuple[faiss.IndexFlatIP, np.ndarray, float]:
        """Encodes chunks into unit-normalized vectors and builds a FAISS IndexFlatIP."""
        self._load_model()
        self.chunks = chunks
        texts = [c.get("text", "") for c in chunks]

        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        build_time = time.time() - start_time

        self.index = index
        return index, embeddings, build_time

    def save_index(self, index_path: Path, embeddings_path: Optional[Path] = None, embeddings: Optional[np.ndarray] = None):
        """Saves FAISS index and numpy embeddings to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        if embeddings_path is not None and embeddings is not None:
            np.save(str(embeddings_path), embeddings)

    def load_index(self, index_path: Path, chunks: List[Dict[str, Any]]):
        """Loads a pre-built FAISS index from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        self.chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches index for top_k most similar chunks for a given query."""
        if self.index is None:
            raise ValueError("Index is not initialized. Call build_index or load_index first.")
        self._load_model()

        start_time = time.time()
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # Search more if filtering is applied
        fetch_k = top_k * 5 if filter_doc else top_k
        fetch_k = min(fetch_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, fetch_k)
        latency_ms = (time.time() - start_time) * 1000.0

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            if filter_doc and filter_doc.lower() not in chunk.get("doc_name", "").lower():
                continue
            chunk["score"] = float(score)
            chunk["dense_rank"] = rank
            chunk["search_method"] = "dense"
            chunk["query_latency_ms"] = latency_ms
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results
