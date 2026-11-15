"""
Grounded Answer Generator with Citation Enforcement & Refusal Handling.
Supports:
1. Serverless Cloud LLM (Groq / Together / OpenAI) - $0 cost, zero GPU needed
2. Local HuggingFace Models (Qwen2.5-7B, TinyLlama LoRA) on NVIDIA GPUs
3. Fast deterministic offline fallback when running on CPU without API keys
"""

from __future__ import annotations
import os
import re
import json
import time
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_context_passages


class RAGGenerator:
    """Generates grounded answers from retrieved context chunks across local and cloud backends."""

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

    def _call_cloud_llm(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ) -> Optional[Tuple[str, str]]:
        """Calls Groq Cloud or OpenAI-compatible serverless LLM."""
        from ..core.config import settings

        api_key = settings.groq_api_key or settings.openai_api_key
        if not api_key:
            return None

        if settings.groq_api_key:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            model = settings.groq_model
            provider = "Groq LPU"
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
            model = settings.openai_model
            provider = "OpenAI"

        context_str = format_context_passages(retrieved_chunks)
        user_message = f"CONTEXT PASSAGES:\n{context_str}\n\nUSER QUESTION:\n{query}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "FinDoc-RAG/1.0"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_text = data["choices"][0]["message"]["content"].strip()
                return raw_text, f"{provider} ({model})"
        except Exception as e:
            print(f"[Warning] Cloud LLM call to {provider} failed: {e}. Falling back...")
            return None

    def _load_local_model(self):
        """Lazy load generator model when GPU/memory permits."""
        if self.use_fallback or self.pipeline is not None or self.device != "cuda":
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype or torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer
            )
        except Exception as e:
            print(f"[Warning] Could not load local {self.model_name} ({e}). Falling back.")
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
        Routes to Cloud LLM (if key provided), Local GPU (if CUDA available), or offline fallback.
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

        # 1. Try Serverless Cloud LLM (Groq / OpenAI) if configured
        cloud_result = self._call_cloud_llm(query, retrieved_chunks, max_new_tokens, temperature)
        if cloud_result is not None:
            raw_text, active_model = cloud_result
            is_refusal = "not found in the documents" in raw_text.lower()
            citations = [int(m) for m in re.findall(r"\[(\d+)\]", raw_text)]
            latency_ms = (time.time() - start_time) * 1000.0
            return {
                "answer": raw_text,
                "is_refusal": is_refusal,
                "citations": sorted(list(set(citations))),
                "latency_ms": latency_ms,
                "prompt": prompt,
                "generator_model": active_model
            }

        # 2. Try Local GPU Weights if CUDA is active
        if self.device == "cuda" and not self.use_fallback:
            self._load_local_model()
            if self.pipeline is not None:
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
                    "citations": sorted(list(set(citations))),
                    "latency_ms": latency_ms,
                    "prompt": prompt,
                    "generator_model": self.model_name
                }

        # 3. Deterministic Offline Fallback (For local CPU testing and CI)
        corpus_entities = {"alphabet", "google", "amazon", "apple", "microsoft", "nvidia"}
        query_entities = set(re.findall(r"\b[A-Z][a-z]+\b", query))
        unknown_entities = [e.lower() for e in query_entities if e.lower() not in corpus_entities and e.lower() not in {"what", "which", "where", "when", "how", "detail", "provide", "summary", "total", "fiscal", "year"}]

        chunks_combined = " ".join([c.get("text", "").lower() for c in retrieved_chunks])
        entity_missing = any(ue not in chunks_combined for ue in unknown_entities)

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
            "generator_model": "deterministic-offline-fallback"
        }
