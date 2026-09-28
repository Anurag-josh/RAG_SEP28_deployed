"""
Unit and integration tests for User-Controlled Research Depth, Fallbacks, and Research More features.
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from groq import Groq
from config.config import RESEARCH_DEPTH_LEVELS, DEFAULT_RESEARCH_DEPTH, RESEARCH_DEPTH_CONFIG
from cache.cache_manager import CacheManager
from phase10.web_grounded_rag import WebGroundedRAGPipeline, deduplicate_chunks
from phase8.prompt_builder import PromptBuilder
from phase9.gemini_client import GeminiContentGenerator


class TestResearchDepthSystem(unittest.TestCase):

    def setUp(self):
        self.tmp_db = Path(tempfile.mktemp(suffix=".db"))
        self.cache_manager = CacheManager(db_path=self.tmp_db)
        self.cache_manager.clear_all_cache()
        self.pipeline = WebGroundedRAGPipeline(cache_manager=self.cache_manager)

    def tearDown(self):
        self.pipeline.cache_manager.close()
        if self.tmp_db.exists():
            try:
                os.unlink(self.tmp_db)
            except Exception:
                pass

    def test_configured_constants(self):
        """Verify research depth configured levels, token targets, and default values."""
        self.assertEqual(RESEARCH_DEPTH_LEVELS["quick"], 5)
        self.assertEqual(RESEARCH_DEPTH_LEVELS["standard"], 10)
        self.assertEqual(RESEARCH_DEPTH_LEVELS["deep"], 15)
        self.assertEqual(DEFAULT_RESEARCH_DEPTH, "quick")
        self.assertEqual(RESEARCH_DEPTH_CONFIG["quick"]["max_output_tokens"], 800)
        self.assertEqual(RESEARCH_DEPTH_CONFIG["standard"]["max_output_tokens"], 1500)
        self.assertEqual(RESEARCH_DEPTH_CONFIG["deep"]["max_output_tokens"], 2500)

    def test_quick_depth_sources_and_tokens(self):
        """Test 1: Quick depth maps to 5 target candidate sources and 800 max output tokens."""
        self.pipeline.searcher = MagicMock()
        self.pipeline.searcher.optimize_query.return_value = "What is linear search?"
        self.pipeline.searcher.search.return_value = [
            {"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(10)
        ]
        
        self.pipeline.loader = MagicMock()
        self.pipeline.loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Linear search algorithm content.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]
        
        self.pipeline.gemini_client = MagicMock()
        self.pipeline.gemini_client.generate_response_with_tokens.return_value = ("Linear search is O(N) [1].", {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        res = self.pipeline.run_pipeline("What is linear search?", research_depth="quick")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "quick")
        self.assertEqual(res["sources_analyzed"], 5)
        
        # Verify max_tokens=800 was passed to LLM client
        self.pipeline.gemini_client.generate_response_with_tokens.assert_called_once()
        _, kwargs = self.pipeline.gemini_client.generate_response_with_tokens.call_args
        self.assertEqual(kwargs.get("max_tokens"), 800)

    def test_standard_depth_sources_and_tokens(self):
        """Test 2: Standard depth maps to 10 target sources and 1500 max output tokens."""
        self.pipeline.searcher = MagicMock()
        self.pipeline.searcher.optimize_query.return_value = "What is bubble sort?"
        self.pipeline.searcher.search.return_value = [
            {"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(15)
        ]
        
        self.pipeline.loader = MagicMock()
        self.pipeline.loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Bubble sort content.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]
        
        self.pipeline.gemini_client = MagicMock()
        self.pipeline.gemini_client.generate_response_with_tokens.return_value = ("Bubble sort explanation.", {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        res = self.pipeline.run_pipeline("What is bubble sort?", research_depth="standard")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "standard")
        self.assertEqual(res["sources_analyzed"], 10)
        
        _, kwargs = self.pipeline.gemini_client.generate_response_with_tokens.call_args
        self.assertEqual(kwargs.get("max_tokens"), 1500)

    def test_deep_depth_sources_and_tokens(self):
        """Test 3: Deep depth maps to 15 target sources and 2500 max output tokens."""
        self.pipeline.searcher = MagicMock()
        self.pipeline.searcher.optimize_query.return_value = "What is heap sort?"
        self.pipeline.searcher.search.return_value = [
            {"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(20)
        ]
        
        self.pipeline.loader = MagicMock()
        self.pipeline.loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Deep heap sort analysis.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]
        
        self.pipeline.gemini_client = MagicMock()
        self.pipeline.gemini_client.generate_response_with_tokens.return_value = ("Comprehensive heap sort study.", {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        res = self.pipeline.run_pipeline("What is heap sort?", research_depth="deep")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "deep")
        self.assertEqual(res["sources_analyzed"], 15)

        _, kwargs = self.pipeline.gemini_client.generate_response_with_tokens.call_args
        self.assertEqual(kwargs.get("max_tokens"), 2500)

    def test_cache_sufficiency_hit(self):
        """Test 4: Cached Standard depth is sufficient for Quick request -> CACHE HIT, NO RAG."""
        self.pipeline.searcher = MagicMock()
        self.pipeline.searcher.optimize_query.return_value = "What is radix sort?"
        self.pipeline.searcher.reformulate_conversational_query.side_effect = lambda q, h: q
        self.pipeline.searcher.search.return_value = [
            {"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(12)
        ]
        self.pipeline.loader = MagicMock()
        self.pipeline.loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Radix sort details.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]
        self.pipeline.gemini_client = MagicMock()
        self.pipeline.gemini_client.generate_response_with_tokens.return_value = ("Standard answer for radix sort.", {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        # Run first request at Standard depth
        res1 = self.pipeline.run_pipeline("What is radix sort?", research_depth="standard")
        self.assertEqual(res1["cache_mode"], "RAG_MISS")

        # Run second request at Quick depth -> Should hit cache!
        res2 = self.pipeline.run_pipeline("What is radix sort?", research_depth="quick")
        self.assertTrue(res2["cache_mode"] in ["REDIS_HIT", "MONGODB_HIT", "SEMANTIC_HIT"])

    def test_cache_insufficiency_additional_research(self):
        """Test 5: Cached Quick depth is INSUFFICIENT for Deep request -> ADDITIONAL RESEARCH."""
        self.pipeline.searcher = MagicMock()
        self.pipeline.searcher.optimize_query.return_value = "What is counting sort?"
        self.pipeline.searcher.reformulate_conversational_query.side_effect = lambda q, h: q
        self.pipeline.searcher.search.return_value = [
            {"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(20)
        ]
        self.pipeline.loader = MagicMock()
        self.pipeline.loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Counting sort details.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]
        self.pipeline.gemini_client = MagicMock()
        self.pipeline.gemini_client.generate_response_with_tokens.return_value = ("Counting sort answer.", {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        # Run first request at Quick depth
        res1 = self.pipeline.run_pipeline("What is counting sort?", research_depth="quick")
        self.assertEqual(res1["cache_mode"], "RAG_MISS")

        # Run second request at Deep depth -> Should trigger full research because cache is insufficient
        res2 = self.pipeline.run_pipeline("What is counting sort?", research_depth="deep")
        self.assertEqual(res2["cache_mode"], "RAG_MISS")
        self.assertEqual(res2["research_depth"], "deep")

    def test_chunk_deduplication(self):
        """Test 6: Context deduplication removes duplicate text chunks."""
        chunks = [
            {"text": "Binary search divides the search space in half recursively."},
            {"text": "Binary search divides the search space in half recursively."}, # Exact dup
            {"text": "Binary search algorithm divides the search space in half recursively."}, # Near dup
            {"text": "Linear search checks elements sequentially one by one."} # Unique
        ]
        unique = deduplicate_chunks(chunks)
        self.assertEqual(len(unique), 2)

    def test_prompt_builder_expansion_instructions(self):
        """Test 7: PromptBuilder injects expansion instructions when existing_answer is provided."""
        pb = PromptBuilder()
        retrieved = [{"title": "Doc 1", "url": "https://example.com/1", "text": "New facts about A."}]
        prompt = pb.build_prompt(
            query="Tell me about A",
            retrieved_chunks=retrieved,
            research_depth="deep",
            existing_answer="A is a basic algorithm."
        )
        self.assertIn("INSTRUCTION FOR ANSWER EXPANSION", prompt)
        self.assertIn("A is a basic algorithm.", prompt)
        self.assertIn("Target Synthesis Depth: DEEP", prompt)

    def test_tpd_429_immediate_fallback(self):
        """Test 8: Daily 429 TPD limit triggers fallback model without 5 retries."""
        with patch.object(Groq, '__init__', return_value=None):
            gen = GeminiContentGenerator(api_key="mock_key")
            gen.groq_client = MagicMock()

            # Mock primary model throwing 429 TPD limit
            tpd_error = Exception("429 Too Many Requests: tokens per day (TPD) limit 200000 reached.")
            gen.groq_client.chat.completions.create.side_effect = [
                tpd_error, # primary model fails once
                MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback answer"))], usage=None) # secondary model succeeds
            ]

            answer, _ = gen.generate_response_with_tokens("Test prompt", max_tokens=500)
            self.assertEqual(answer, "Fallback answer")
            # Verify primary model was tried ONLY ONCE (no 5 retries on TPD error)
            self.assertEqual(gen.groq_client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
