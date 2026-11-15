"""
Pydantic Schemas for FinDoc-RAG API requests and responses.
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    device: str = Field(..., example="cuda")
    device_name: str = Field(..., example="NVIDIA A100-SXM4-80GB")
    dtype: str = Field(..., example="torch.bfloat16")
    indexed_chunks: int = Field(..., example=1008)
    faiss_index_loaded: bool = Field(..., example=True)
    bm25_index_loaded: bool = Field(..., example=True)


class SearchRequest(BaseModel):
    query: str = Field(..., example="What are the primary considerations for Microsoft regarding accounts receivable?")
    method: Literal["dense", "bm25", "hybrid", "reranked"] = Field(default="hybrid")
    top_k: int = Field(default=3, ge=1, le=20)
    apply_entity_filter: bool = Field(default=True, description="Filter corpus by detected company to eliminate cross-company noise")
    company: Optional[str] = Field(default=None, description="Explicit company filter (e.g. microsoft, apple, alphabet, amazon, nvidia)")


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_name: str
    text: str
    score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    hybrid_rank: Optional[int] = None
    final_rank: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    method: str
    entity_filter_applied: Optional[str]
    total_results: int
    latency_ms: float
    chunks: List[ChunkResponse]


class GenerateRequest(BaseModel):
    query: str = Field(..., example="What does Apple disclose regarding supply chain and manufacturing risk factors?")
    retrieval_method: Literal["dense", "bm25", "hybrid", "reranked"] = Field(default="reranked")
    top_k: int = Field(default=3, ge=1, le=10)
    apply_entity_filter: bool = Field(default=True)
    max_new_tokens: int = Field(default=256)


class GenerateResponse(BaseModel):
    query: str
    answer: str
    is_refusal: bool
    citations: List[int]
    generator_model: str
    retrieval_method: str
    latency_ms: float
    retrieved_chunks: List[ChunkResponse]


class BenchmarkSummary(BaseModel):
    execution_environment: Dict[str, Any]
    chunk_comparison: List[Dict[str, Any]]
    retrieval_comparison: List[Dict[str, Any]]
    reranking_performance: Dict[str, Any]
    llm_judge_verdicts: Dict[str, Any]
    failure_analysis_fix: Dict[str, Any]
