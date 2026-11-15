"""
FinDoc-RAG CLI & Service Entrypoint.
Provides command-line interface for search, generation, benchmarking, and starting the API server.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
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


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Starts the FastAPI web service."""
    import uvicorn
    print(f"Starting FinDoc-RAG API on http://{host}:{port}...")
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


def run_search(query: str, method: str = "hybrid", top_k: int = 3, filter_entity: bool = True):
    """Executes a retrieval query."""
    chunks_path = settings.chunks_dir / "chunks_400.csv"
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        return

    chunks = load_precomputed_chunks(chunks_path)
    target_doc = EntityFilter.detect_target_document(query) if filter_entity else None

    print(f"\n[Query]: '{query}'")
    print(f"[Method]: {method} | [Entity Filter]: {target_doc or 'None'}")

    dense = DenseRetriever(settings.embedding_model_name, device=settings.device)
    index_path = settings.indexes_dir / "faiss_index_400.bin"
    if index_path.exists():
        dense.load_index(index_path, chunks)
    else:
        dense.build_index(chunks)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    if method == "dense":
        results = dense.search(query, top_k=top_k, filter_doc=target_doc)
    elif method == "bm25":
        results = bm25.search(query, top_k=top_k, filter_doc=target_doc)
    elif method == "hybrid":
        hybrid = HybridRetriever(dense, bm25, rrf_k=settings.rrf_k)
        results = hybrid.search(query, top_k=top_k, filter_doc=target_doc)
    elif method == "reranked":
        hybrid = HybridRetriever(dense, bm25, rrf_k=settings.rrf_k)
        candidates = hybrid.search(query, top_k=10, filter_doc=target_doc)
        reranker = CrossEncoderReranker(settings.reranker_model_name, device=settings.device)
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        print(f"Unsupported method: {method}")
        return

    print(f"\n=== Top {len(results)} Retrieved Results ===")
    for i, r in enumerate(results, start=1):
        score = r.get("score") or r.get("cross_encoder_score") or 0.0
        print(f"[{i}] Rank {i} | Score: {score:.4f} | Document: {r.get('doc_name')} | ID: {r.get('chunk_id')}")
        snippet = r.get("text", "")[:180].replace("\n", " ")
        print(f"    Snippet: {snippet}...\n")


def run_generate(query: str, top_k: int = 3, filter_entity: bool = True):
    """Executes end-to-end grounded generation."""
    chunks_path = settings.chunks_dir / "chunks_400.csv"
    chunks = load_precomputed_chunks(chunks_path)
    target_doc = EntityFilter.detect_target_document(query) if filter_entity else None

    dense = DenseRetriever(settings.embedding_model_name, device=settings.device)
    index_path = settings.indexes_dir / "faiss_index_400.bin"
    if index_path.exists():
        dense.load_index(index_path, chunks)
    else:
        dense.build_index(chunks)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    hybrid = HybridRetriever(dense, bm25, rrf_k=settings.rrf_k)
    candidates = hybrid.search(query, top_k=10, filter_doc=target_doc)
    reranker = CrossEncoderReranker(settings.reranker_model_name, device=settings.device)
    retrieved = reranker.rerank(query, candidates, top_k=top_k)

    has_cloud_key = bool(settings.groq_api_key or settings.openai_api_key)
    generator = RAGGenerator(
        settings.generator_model_name,
        device=settings.device,
        torch_dtype=settings.torch_dtype,
        use_fallback=(settings.device == "cpu" and not has_cloud_key)
    )
    result = generator.generate(query, retrieved)

    print(f"\n[Query]: '{query}'")
    print(f"[Refusal Status]: {'REFUSAL' if result['is_refusal'] else 'GROUNDED ANSWER'}")
    print(f"[Citations]: {result['citations']}")
    print(f"[Latency]: {result['latency_ms']:.2f} ms")
    print("\n--- Synthesized Answer ---")
    print(result["answer"])


def main():
    parser = argparse.ArgumentParser(description="FinDoc-RAG: Production Hybrid RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host binding")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")

    # Search command
    search_parser = subparsers.add_parser("search", help="Execute search query")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "reranked"], default="hybrid")
    search_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks")
    search_parser.add_argument("--no-filter", action="store_false", dest="filter_entity", help="Disable entity pre-filter")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate grounded answer for query")
    gen_parser.add_argument("query", type=str, help="User financial question")
    gen_parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")
    gen_parser.add_argument("--no-filter", action="store_false", dest="filter_entity", help="Disable entity pre-filter")

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args.host, args.port, args.reload)
    elif args.command == "search":
        run_search(args.query, args.method, args.top_k, args.filter_entity)
    elif args.command == "generate":
        run_generate(args.query, args.top_k, args.filter_entity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
