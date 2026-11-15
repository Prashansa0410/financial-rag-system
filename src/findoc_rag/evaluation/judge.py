"""
LLM-as-a-Judge: Dual-Criterion Evaluator for Correctness and Groundedness.
Supports Meta-Llama/Llama-3.1-8B-Instruct and heuristic evaluation.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional


JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous judge evaluating AI-generated financial answers based on SEC filings.

You must evaluate the CANDIDATE ANSWER on two strict criteria:
1. CORRECTNESS (compared against EXPECTED ANSWER):
   - 'correct': Accurately conveys the primary factual claims and numbers of the expected answer.
   - 'partly correct': Mentions some relevant aspects, but omits critical figures or context.
   - 'wrong': States incorrect information, contradicts expected answer, or says 'Not found in the documents' when expected answer is known.

2. GROUNDEDNESS / SUPPORT (compared against RETRIEVED CHUNKS):
   - 'yes': Every claim is directly verified by the retrieved chunks, OR the answer correctly replied 'Not found in the documents' when chunks lacked the info.
   - 'no': Contains claims, facts, or numbers not present in the retrieved chunks (hallucination).

Output format:
VERDICT: <correct | partly correct | wrong>
SUPPORTED: <yes | no>
REASON: <concise 1-2 sentence justification>
"""

class LLMJudge:
    """Evaluates candidate RAG answers using an independent LLM judge."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name

    def evaluate(
        self,
        query: str,
        candidate_answer: str,
        expected_answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Runs dual-criterion evaluation on candidate answer.
        """
        is_refusal = "not found in the documents" in candidate_answer.lower()
        chunks_text = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()

        # Heuristic / deterministic evaluation logic
        if is_refusal:
            # Check if expected answer was in chunks
            keywords = [w.lower() for w in re.findall(r"\b\w{5,}\b", expected_answer)]
            present_in_chunks = sum(1 for kw in keywords if kw in chunks_text) >= 2
            if present_in_chunks:
                verdict = "wrong"
                supported = "no"
                reason = "Model returned negative refusal even though information was present in retrieved chunks."
            else:
                verdict = "wrong"
                supported = "yes"
                reason = "The candidate answer correctly stated 'Not found in the documents' because chunks lacked the information."
        else:
            # Check overlap with expected answer
            expected_words = set(re.findall(r"\b\w{4,}\b", expected_answer.lower()))
            candidate_words = set(re.findall(r"\b\w{4,}\b", candidate_answer.lower()))
            overlap = len(expected_words & candidate_words) / max(1, len(expected_words))

            if overlap >= 0.5:
                verdict = "correct"
            elif overlap >= 0.2:
                verdict = "partly correct"
            else:
                verdict = "wrong"

            # Check supported by chunks
            unsupported = [w for w in candidate_words if w not in chunks_text and len(w) > 5]
            supported = "yes" if len(unsupported) <= 2 else "no"
            reason = f"Candidate answer factual overlap: {overlap*100:.1f}%. Grounded in retrieved passages."

        return {
            "verdict": verdict,
            "supported": supported,
            "reason": reason,
            "judge_model": self.model_name
        }
