"""
Unit and Integration Tests for FinDoc-RAG.
"""

import sys
import unittest
from pathlib import Path

# Add src and root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from findoc_rag.core.config import settings
from findoc_rag.indexing.chunker import chunk_text
from findoc_rag.retrieval.bm25 import BM25Retriever
from findoc_rag.retrieval.filter import EntityFilter
from findoc_rag.generation.generator import RAGGenerator


class TestFinDocRAG(unittest.TestCase):
    def test_config_paths(self):
        """Verify settings resolve to expected directories."""
        self.assertTrue(settings.corpus_dir.is_dir())
        self.assertTrue(settings.chunks_dir.is_dir())
        self.assertTrue(settings.eval_dir.is_dir())

    def test_chunking_logic(self):
        """Verify sliding-window token chunker."""
        sample_text = "word " * 500
        chunks = chunk_text(sample_text, chunk_size=400, overlap=50, doc_name="test.txt")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "test_c400_0")
        self.assertEqual(chunks[0]["token_count"], 400)

    def test_entity_filter(self):
        """Verify company entity extraction for metadata filtering."""
        self.assertEqual(EntityFilter.detect_target_document("What is Alphabet's cybersecurity policy?"), "alphabet_q1_2026_10q.txt")
        self.assertEqual(EntityFilter.detect_target_document("Tell me about Apple's iPhone risks"), "apple_fy2023_10k.txt")
        self.assertEqual(EntityFilter.detect_target_document("Microsoft accounts receivable allowance"), "microsoft_fy2023_10k.txt")
        self.assertIsNone(EntityFilter.detect_target_document("General questions with no company name"))

    def test_bm25_retriever(self):
        """Verify BM25 index creation and scoring."""
        sample_chunks = [
            {"chunk_id": "c1", "doc_name": "d1.txt", "text": "Operating income was 50 billion dollars in fiscal year."},
            {"chunk_id": "c2", "doc_name": "d2.txt", "text": "Accounts receivable and bad debt allowance estimates."},
            {"chunk_id": "c3", "doc_name": "d3.txt", "text": "General risk factors and foreign exchange volatility."},
        ]
        retriever = BM25Retriever()
        retriever.build_index(sample_chunks)
        results = retriever.search("operating income", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_generator_abstention(self):
        """Verify negative refusal when no information is present."""
        generator = RAGGenerator(use_fallback=True)
        res = generator.generate("What was Tesla's vehicle delivery count in 2023?", retrieved_chunks=[])
        self.assertTrue(res["is_refusal"])
        self.assertIn("not found in the documents", res["answer"].lower())


if __name__ == "__main__":
    unittest.main()
