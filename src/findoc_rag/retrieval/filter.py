"""
Entity-Aware Metadata Pre-Filtering.
Addresses cross-company entity collisions in multi-company financial corpora.
"""

from __future__ import annotations
import re
from typing import Optional, Dict


ENTITY_SYNONYMS: Dict[str, list[str]] = {
    "alphabet_q1_2026_10q.txt": ["alphabet", "google", "sundar pichai", "youtube"],
    "amazon_fy2022_10k.txt": ["amazon", "aws", "andy jassy", "prime"],
    "apple_fy2023_10k.txt": ["apple", "tim cook", "iphone", "ipad", "macbook"],
    "microsoft_fy2023_10k.txt": ["microsoft", "satya nadella", "azure", "windows", "office 365"],
    "nvidia_fy2024_annual_report.txt": ["nvidia", "jensen huang", "geforce", "cuda", "h100", "a100"]
}


class EntityFilter:
    """Detects target company entity from query text and returns matching document filter."""

    @staticmethod
    def detect_target_document(query: str) -> Optional[str]:
        """Maps query keywords to the appropriate SEC filing document."""
        q_lower = query.lower()
        for doc_name, terms in ENTITY_SYNONYMS.items():
            for term in terms:
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, q_lower):
                    return doc_name
        return None
