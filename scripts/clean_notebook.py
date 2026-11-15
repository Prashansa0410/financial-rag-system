"""
Cleans and standardizes the executed notebook for public open-source publication.
Removes academic grading references while preserving 100% of the executed cell outputs.
"""

import json
import re
from pathlib import Path

def clean_notebook():
    input_path = Path("assignment2b_bitsExecution/Assignment_2B_Pipeline.ipynb")
    output_path = Path("notebooks/findoc_rag_pipeline.ipynb")

    with open(input_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Clean Header
    nb["cells"][0]["source"] = [
        "# FinDoc-RAG: Production Hybrid RAG & Cross-Encoder Reranking Engine\n",
        "## Domain: Financial SEC Filings (Form 10-K & Form 10-Q)\n",
        "\n",
        "### System Overview & Architecture\n",
        "This notebook demonstrates an end-to-end, production-grade Retrieval-Augmented Generation (RAG) system grounded in SEC financial filings (Alphabet, Amazon, Apple, Microsoft, NVIDIA).\n",
        "\n",
        "**Pipeline Workflow:**\n",
        "1. **Passage Chunking Strategy:** Empirical benchmark comparing 100-token vs. 400-token chunk sizes across hit rates, latency, and context budget.\n",
        "2. **Multi-Modal Retrieval Engine:** Dense vector search (`BAAI/bge-small-en-v1.5` + FAISS), Lexical keyword search (`BM25Okapi`), and Hybrid Rank Fusion (`RRF` with k=60).\n",
        "3. **Cross-Encoder Reranking:** Joint cross-attention reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on candidate pools.\n",
        "4. **Grounded Generation & Guardrails:** Strict negative refusal (\"Not found in the documents.\"), citation enforcement ([1], [2], [3]), and model scaling with `Qwen/Qwen2.5-7B-Instruct`.\n",
        "5. **Automated LLM-as-a-Judge Evaluation:** Cross-family evaluation (`meta-llama/Llama-3.1-8B-Instruct`) auditing factual correctness and grounded faithfulness.\n",
        "6. **Root Cause Failure Analysis & Architectural Fix:** Entity-Aware Metadata Pre-Filtering to eliminate cross-company false positives.\n",
        "\n",
        "*Hardware Environment:* Benchmarked and executed on an **NVIDIA A100-SXM4-80GB GPU** at the **BITS Pilani Machine Learning & NLP Computing Lab**.\n"
    ]

    replacements = [
        (r"Assignment 2B", "FinDoc-RAG Pipeline"),
        (r"Assignment 1B", "Domain Adaptation Baseline"),
        (r"Assignment 1", "Domain Baseline"),
        (r"# Part A — Dense Search and Chunk Size \(5 Marks\)", "# Phase 1 — Dense Search & Chunk Size Strategy"),
        (r"## A1\. Build Dense Search \(2 Marks\)", "## 1.1 Dense Vector Indexing (BAAI/bge-small-en-v1.5 + FAISS IndexFlatIP)"),
        (r"## A2\. Choose a Chunk Size \(3 Marks\)", "## 1.2 Empirical Chunk Size Evaluation (100 vs. 400 Tokens)"),
        (r"Per Assignment Specifications", "Evaluation Metrics & Objectives"),
        (r"The assignment requires:", "Technical Objectives:"),
        (r"\s*\(\d+\s*[Mm]arks?\)", ""),
        (r"# Part B — Keyword and hybrid search \(4 marks\)", "# Phase 2 — Multi-Modal Retrieval Benchmark (Dense, BM25, Hybrid RRF)"),
        (r"Part B1: Dense Search Evaluation", "Phase 2.1: Dense Search Benchmark"),
        (r"Part B2: BM25 Keyword Search Evaluation", "Phase 2.2: Lexical BM25 Search Benchmark"),
        (r"Part B3: Hybrid Search with Reciprocal Rank Fusion \(RRF, \$k=60\$\)", "Phase 2.3: Hybrid Search with Reciprocal Rank Fusion (RRF, k=60)"),
        (r"# Part C — Reranking \(4 marks\)", "# Phase 3 — Cross-Encoder Reranking & Rank Dynamics"),
        (r"Part C1: Retrieve Top 10 with Best Method & Re-Score with Cross-Encoder", "Phase 3.1: Candidate Retrieval & Cross-Encoder Rescoring"),
        (r"Part C2: Compare Rankings Before & After Reranking and Rank Change Rate", "Phase 3.2: Rank Dynamics & Rank Change Rate (RCR)"),
        (r"Part C3: Measure Reranking Latency & Performance", "Phase 3.3: Latency & Accuracy Trade-off Analysis"),
        (r"# Part D — Generate and check answers \(7 marks\)", "# Phase 4 — Grounded Generation & Citation Verification"),
        (r"Part D1\.1: Grounded Answer Generation Prompt & Model Setup", "Phase 4.1: Grounded Generation & Citation Prompting"),
        (r"Part D1\.3: Unanswerable Out-of-Corpus Query Test", "Phase 4.2: Hallucination Guardrail: Out-of-Corpus Negative Refusal Test"),
        (r"Model Transition: From TinyLlama-1\.1B to Qwen2\.5-7B-Instruct", "Phase 4.3: Model Scaling: Transition to Qwen2.5-7B-Instruct"),
        (r"# Part D2 — Check the answers with an LLM judge \(4 marks\)", "# Phase 5 — Automated Evaluation via LLM-as-a-Judge"),
        (r"Part D2\.1: Write and Run the LLM Judge", "Phase 5.1: Dual-Criterion LLM Judge (Correctness & Groundedness)"),
        (r"Part D2\.2: Spot-Check the LLM Judge", "Phase 5.2: Human-in-the-Loop Verification & Spot-Check Audit"),
        (r"Part D2\.3: Root Cause Failure Analysis", "Phase 5.3: Root Cause Failure Analysis"),
        (r"Part D2\.4: Suggest an Architectural Fix", "Phase 5.4: Architectural Solution: Entity-Aware Metadata Pre-Filtering"),
        (r"Group No\. 124", "FinDoc-RAG Core"),
        (r"Assignment Specification", "Production Pipeline Design"),
        (r"As stated in the assignment instructions:", "Evaluation Design Consideration:"),
        (r"The assignment rubric explicitly mandates:", "Evaluation Design Specification:")
    ]

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            for pat, rep in replacements:
                text = re.sub(pat, rep, text)
            lines = text.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Cleaned notebook saved: {output_path} ({len(nb['cells'])} cells)")

if __name__ == "__main__":
    clean_notebook()
