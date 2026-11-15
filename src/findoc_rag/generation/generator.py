"""
Grounded Answer Generator with Citation Enforcement & Refusal Handling.
Supports Qwen2.5-7B-Instruct, TinyLlama QLoRA, and lightweight offline fallback.
"""

from __future__ import annotations
import os
import re
import time
from typing import List, Dict, Any, Optional
from .prompts import build_rag_prompt


class RAGGenerator:
    """Generates grounded answers from retrieved context chunks."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cpu",
        torch_dtype: Optional[Any] = None,
        use_fallback: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_fallback = use_fallback
        self.pipeline = None
        self.tokenizer = None

    def _load_model(self):
        """Lazy load generator model when GPU/memory permits."""
        if self.use_fallback or self.pipeline is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype or torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer
            )
        except Exception as e:
            print(f"[Warning] Could not load {self.model_name} ({e}). Falling back to heuristic generator.")
            self.use_fallback = True

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Synthesizes an answer grounded in the retrieved chunks.
        Returns the answer text, citation list, refusal status, and latency.
        """
        start_time = time.time()
        prompt = build_rag_prompt(query, retrieved_chunks)

        if not retrieved_chunks:
            return {
                "answer": "Not found in the documents.",
                "is_refusal": True,
                "citations": [],
                "latency_ms": 0.0,
                "prompt": prompt,
                "generator_model": self.model_name
            }

        if self.use_fallback or self.pipeline is None:
            # Check for unknown / out-of-corpus entity mention (e.g. Tesla, Netflix, Ford)
            corpus_entities = {"alphabet", "google", "amazon", "apple", "microsoft", "nvidia"}
            query_entities = set(re.findall(r"\b[A-Z][a-z]+\b", query))
            unknown_entities = [e.lower() for e in query_entities if e.lower() not in corpus_entities and e.lower() not in {"what", "which", "where", "when", "how", "detail", "provide", "summary", "total", "fiscal", "year"}]
            
            # Check chunk texts
            chunks_combined = " ".join([c.get("text", "").lower() for c in retrieved_chunks])
            entity_missing = any(ue not in chunks_combined for ue in unknown_entities)

            # Keyword matching
            q_keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z]{5,}\b", query)]
            stop = {"total", "number", "report", "reported", "fiscal", "disclose", "details", "concerning", "primary", "impact"}
            q_keywords = [w for w in q_keywords if w not in stop]

            matched_indices = []
            if not entity_missing and q_keywords:
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    text_lower = chunk.get("text", "").lower()
                    matches = sum(1 for kw in q_keywords if kw in text_lower)
                    if matches >= max(2, int(len(q_keywords) * 0.5)):
                        matched_indices.append(i)

            if matched_indices and not entity_missing:
                primary_chunk = retrieved_chunks[matched_indices[0] - 1]
                sentences = re.split(r"(?<=[.!?])\s+", primary_chunk.get("text", ""))
                selected_sent = sentences[0] if sentences else primary_chunk.get("text", "")[:200]
                answer_text = f"Based on the SEC disclosures, {selected_sent.strip()} [{matched_indices[0]}]."
                is_refusal = False
                citations = [matched_indices[0]]
            else:
                answer_text = "Not found in the documents."
                is_refusal = True
                citations = []

            latency_ms = (time.time() - start_time) * 1000.0
            return {
                "answer": answer_text,
                "is_refusal": is_refusal,
                "citations": citations,
                "latency_ms": latency_ms,
                "prompt": prompt,
                "generator_model": "heuristic-fallback"
            }

        # Model generation path
        self._load_model()
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        raw_text = outputs[0]["generated_text"][len(prompt):].strip()

        is_refusal = "not found in the documents" in raw_text.lower()
        citations = [int(m) for m in re.findall(r"\[(\d+)\]", raw_text)]

        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "answer": raw_text,
            "is_refusal": is_refusal,
            "citations": list(set(citations)),
            "latency_ms": latency_ms,
            "prompt": prompt,
            "generator_model": self.model_name
        }
