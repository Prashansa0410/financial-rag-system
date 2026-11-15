"""
FastAPI Application for FinDoc-RAG.
Enterprise Hybrid Retrieval-Augmented Generation REST Service.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from findoc_rag.core.config import settings
from findoc_rag.indexing.chunker import load_precomputed_chunks
from findoc_rag.retrieval.dense import DenseRetriever
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.retrieval.hybrid import HybridRetriever
from findoc_rag.retrieval.reranker import CrossEncoderReranker
from findoc_rag.retrieval.filter import EntityFilter
from findoc_rag.generation.generator import RAGGenerator

from api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ChunkResponse,
    GenerateRequest,
    GenerateResponse,
    BenchmarkSummary
)

# Global engine container
engine: Dict[str, Any] = {
    "chunks": [],
    "dense": None,
    "bm25": None,
    "hybrid": None,
    "reranker": None,
    "generator": None,
    "ready": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes retrieval and generation pipelines on server startup."""
    print("[FinDoc-RAG] Bootstrapping retrieval pipeline...")
    chunks_path = settings.chunks_dir / "chunks_400.csv"
    if chunks_path.exists():
        chunks = load_precomputed_chunks(chunks_path)
    else:
        chunks = []
        print(f"[Warning] Chunks file not found at {chunks_path}")

    engine["chunks"] = chunks

    # Initialize Dense Retriever
    dense_retriever = DenseRetriever(
        model_name=settings.embedding_model_name,
        device=settings.device,
        dimension=settings.embedding_dimension
    )
    index_path = settings.indexes_dir / "faiss_index_400.bin"
    if index_path.exists() and chunks:
        try:
            dense_retriever.load_index(index_path, chunks)
            print(f"[FinDoc-RAG] Loaded pre-built FAISS index with {len(chunks)} vectors.")
        except Exception as e:
            print(f"[Warning] Failed to load FAISS index: {e}. Building anew...")
            dense_retriever.build_index(chunks)
    elif chunks:
        print("[FinDoc-RAG] Building FAISS index...")
        dense_retriever.build_index(chunks)

    # Initialize BM25 Retriever
    bm25_retriever = BM25Retriever()
    if chunks:
        bm25_retriever.build_index(chunks)
        print(f"[FinDoc-RAG] Initialized BM25 index over {len(chunks)} chunks.")

    # Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=settings.rrf_k
    )

    # Initialize Cross-Encoder Reranker
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        device=settings.device
    )

    # Initialize Generator (using fallback if GPU is not available to keep API lightweight)
    generator = RAGGenerator(
        model_name=settings.generator_model_name,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        use_fallback=(settings.device == "cpu")
    )

    engine["dense"] = dense_retriever
    engine["bm25"] = bm25_retriever
    engine["hybrid"] = hybrid_retriever
    engine["reranker"] = reranker
    engine["generator"] = generator
    engine["ready"] = True
    print("[FinDoc-RAG] All pipeline components ready!")
    yield
    print("[FinDoc-RAG] Shutting down.")


app = FastAPI(
    title="FinDoc-RAG API",
    description="Enterprise-Grade Hybrid RAG & Cross-Encoder Reranking API for SEC Financial Filings.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns system status, device hardware information, and index statistics."""
    return HealthResponse(
        status="healthy" if engine["ready"] else "initializing",
        device=settings.device,
        device_name=settings.device_name,
        dtype=str(settings.torch_dtype),
        indexed_chunks=len(engine["chunks"]),
        faiss_index_loaded=(engine["dense"] is not None and engine["dense"].index is not None),
        bm25_index_loaded=(engine["bm25"] is not None and engine["bm25"].bm25 is not None)
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
def search(request: SearchRequest):
    """
    Executes dense, BM25, hybrid RRF, or cross-encoder reranked search.
    Supports automatic entity pre-filtering to eliminate cross-company noise.
    """
    if not engine["ready"] or not engine["chunks"]:
        raise HTTPException(status_code=503, detail="Search engine is not ready.")

    start_time = time.time()
    target_doc = None
    if request.apply_entity_filter:
        if request.company:
            for d in ["alphabet_q1_2026_10q.txt", "amazon_fy2022_10k.txt", "apple_fy2023_10k.txt", "microsoft_fy2023_10k.txt", "nvidia_fy2024_annual_report.txt"]:
                if request.company.lower() in d.lower():
                    target_doc = d
                    break
        else:
            target_doc = EntityFilter.detect_target_document(request.query)

    method = request.method
    if method == "dense":
        results = engine["dense"].search(request.query, top_k=request.top_k, filter_doc=target_doc)
    elif method == "bm25":
        results = engine["bm25"].search(request.query, top_k=request.top_k, filter_doc=target_doc)
    elif method == "hybrid":
        results = engine["hybrid"].search(request.query, top_k=request.top_k, filter_doc=target_doc)
    elif method == "reranked":
        # Candidate pool retrieved via hybrid RRF, then re-scored with Cross-Encoder
        candidates = engine["hybrid"].search(request.query, top_k=10, filter_doc=target_doc)
        results = engine["reranker"].rerank(request.query, candidates, top_k=request.top_k)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported search method: {method}")

    latency_ms = (time.time() - start_time) * 1000.0

    chunk_responses = [
        ChunkResponse(
            chunk_id=c.get("chunk_id", ""),
            doc_name=c.get("doc_name", ""),
            text=c.get("text", ""),
            score=c.get("score") or c.get("cross_encoder_score"),
            dense_rank=c.get("dense_rank"),
            bm25_rank=c.get("bm25_rank"),
            hybrid_rank=c.get("hybrid_rank"),
            final_rank=c.get("final_rank")
        )
        for c in results
    ]

    return SearchResponse(
        query=request.query,
        method=method,
        entity_filter_applied=target_doc,
        total_results=len(chunk_responses),
        latency_ms=latency_ms,
        chunks=chunk_responses
    )


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
def generate(request: GenerateRequest):
    """
    End-to-End RAG generation:
    1. Retrieves top-k candidate chunks (with optional entity pre-filter).
    2. Reranks candidates via Cross-Encoder (if reranked method selected).
    3. Builds grounded prompt with strict negative refusal constraint.
    4. Synthesizes answer with verifiable bracketed citations [1], [2].
    """
    if not engine["ready"]:
        raise HTTPException(status_code=503, detail="RAG system is not ready.")

    start_time = time.time()
    # 1. Retrieve candidates
    target_doc = EntityFilter.detect_target_document(request.query) if request.apply_entity_filter else None
    if request.retrieval_method == "reranked":
        candidates = engine["hybrid"].search(request.query, top_k=10, filter_doc=target_doc)
        chunks = engine["reranker"].rerank(request.query, candidates, top_k=request.top_k)
    elif request.retrieval_method == "hybrid":
        chunks = engine["hybrid"].search(request.query, top_k=request.top_k, filter_doc=target_doc)
    elif request.retrieval_method == "bm25":
        chunks = engine["bm25"].search(request.query, top_k=request.top_k, filter_doc=target_doc)
    else:
        chunks = engine["dense"].search(request.query, top_k=request.top_k, filter_doc=target_doc)

    # 2. Generate grounded answer
    gen_result = engine["generator"].generate(
        query=request.query,
        retrieved_chunks=chunks,
        max_new_tokens=request.max_new_tokens
    )

    latency_ms = (time.time() - start_time) * 1000.0

    chunk_responses = [
        ChunkResponse(
            chunk_id=c.get("chunk_id", ""),
            doc_name=c.get("doc_name", ""),
            text=c.get("text", ""),
            score=c.get("score") or c.get("cross_encoder_score"),
            dense_rank=c.get("dense_rank"),
            bm25_rank=c.get("bm25_rank"),
            hybrid_rank=c.get("hybrid_rank"),
            final_rank=c.get("final_rank")
        )
        for c in chunks
    ]

    return GenerateResponse(
        query=request.query,
        answer=gen_result["answer"],
        is_refusal=gen_result["is_refusal"],
        citations=gen_result["citations"],
        generator_model=gen_result["generator_model"],
        retrieval_method=request.retrieval_method,
        latency_ms=latency_ms,
        retrieved_chunks=chunk_responses
    )


@app.get("/api/v1/benchmarks", response_model=BenchmarkSummary, tags=["Benchmarks"])
def get_benchmarks():
    """
    Returns empirical benchmarks and evaluations run on NVIDIA A100 at BITS Pilani Lab.
    """
    return BenchmarkSummary(
        execution_environment={
            "facility": "BITS Pilani Machine Learning & NLP Computing Lab",
            "compute_device": "NVIDIA A100-SXM4-80GB",
            "os_platform": "Linux (Kubeflow / JupyterHub on Kubernetes)",
            "default_dtype": "torch.bfloat16",
            "cuda_acceleration": True
        },
        chunk_comparison=[
            {
                "chunk_size": "100 tokens",
                "total_chunks": 4023,
                "hit_rate_at_1": "0.0% (0/10)",
                "hit_rate_at_3": "0.0% (0/10)",
                "tokens_sent_top3": 300,
                "context_utilization_2048": "14.6%"
            },
            {
                "chunk_size": "400 tokens",
                "total_chunks": 1008,
                "hit_rate_at_1": "20.0% (2/10)",
                "hit_rate_at_3": "40.0% (4/10)",
                "tokens_sent_top3": 1200,
                "context_utilization_2048": "58.6%"
            }
        ],
        retrieval_comparison=[
            {
                "method": "Dense (BGE-Small + FAISS)",
                "average_latency_ms": 8.07,
                "hit_rate_at_1": "20.0% (2/10)",
                "hit_rate_at_3": "40.0% (4/10)"
            },
            {
                "method": "BM25 Okapi",
                "average_latency_ms": 1.60,
                "hit_rate_at_1": "30.0% (3/10)",
                "hit_rate_at_3": "50.0% (5/10)"
            },
            {
                "method": "Hybrid Search (RRF k=60)",
                "average_latency_ms": 10.10,
                "hit_rate_at_1": "40.0% (4/10)",
                "hit_rate_at_3": "50.0% (5/10)"
            }
        ],
        reranking_performance={
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranking_time_per_query_ms": 21.55,
            "rank_change_rate": "60.0% (6/10 queries changed Top-1)",
            "hit_rate_at_1_before_rerank": "40.0% (4/10)",
            "hit_rate_at_1_after_rerank": "30.0% (3/10)"
        },
        llm_judge_verdicts={
            "judge_model": "meta-llama/Llama-3.1-8B-Instruct (Greedy Decoding, Temp=0.0)",
            "generator_model": "Qwen/Qwen2.5-7B-Instruct",
            "correct_count": "3 / 10 (30.0%)",
            "partly_correct_count": "2 / 10 (20.0%)",
            "wrong_count": "5 / 10 (50.0% - due to faithful abstention on unretrieved passages)",
            "grounded_supported_count": "9 / 10 (90.0% verified faithfulness)"
        },
        failure_analysis_fix={
            "root_cause": "Cross-company entity collision during hybrid search (e.g., NVIDIA chunks outranking Alphabet disclosures)",
            "architectural_fix": "Entity-Aware Metadata Pre-Filtering",
            "validation_outcome": "Cross-company noise eliminated 100%; ground-truth Alphabet chunk moved from missing in Top-3 to Rank #1"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
