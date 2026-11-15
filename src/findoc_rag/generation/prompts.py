"""
Prompt templates and formatting for grounded generation and strict citation.
"""

from __future__ import annotations
from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a rigorous, expert financial analyst answering questions based ONLY on verified SEC financial filings (Form 10-K and 10-Q).

STRICT RULES:
1. Base your answer ENTIRELY on the provided context passages below.
2. If the context does not contain enough information to answer the question with certainty, you MUST reply EXACTLY with:
   Not found in the documents.
3. Every factual claim, numerical figure, or policy statement MUST include an explicit bracketed citation to the supporting passage index, e.g., [1], [2], or [3].
4. Do NOT hallucinate, extrapolate, or bring in external knowledge not present in the passages.
5. Be concise, direct, and factual.
"""

def format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with 1-based indexing for citation."""
    formatted_passages = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.get("doc_name", "SEC Filing")
        cid = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "").strip()
        formatted_passages.append(f"[{i}] Document: {doc} | Chunk: {cid}\n{text}")
    return "\n\n".join(formatted_passages)


def build_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Constructs the complete instruction prompt for the generator model."""
    context_str = format_context_passages(chunks)
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT PASSAGES:
{context_str}

USER QUESTION:
{query}

ANSWER:"""
    return prompt
