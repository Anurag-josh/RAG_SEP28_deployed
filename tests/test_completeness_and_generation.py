"""
Verification test suite for LLM Answer Completeness, Token Limits, and Research Depth Expansion.
Tests exact queries specified in user prompt requirements #17 and #18.
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import RESEARCH_DEPTH_CONFIG
from cache.cache_manager import CacheManager
from phase10.web_grounded_rag import WebGroundedRAGPipeline


class TestLLMAnswerCompleteness(unittest.TestCase):

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

    def _assert_answer_complete(self, answer: str, depth: str):
        """Helper to assert that answer is non-empty and ends naturally without structural truncation."""
        self.assertTrue(answer and len(answer.strip()) > 50, f"Answer for depth {depth} is too short or empty.")
        stripped = answer.strip()

        # Must not end mid-sentence with prepositions / conjunctions / unfinished words
        incomplete_trailing = {
            "the", "a", "an", "and", "or", "but", "with", "between", "for", "of", "to",
            "in", "on", "at", "by", "from", "that", "which", "is", "are", "as", "than"
        }
        words = stripped.split()
        last_word = words[-1].lower().strip(".,;:!?\"'[]()") if words else ""
        self.assertNotIn(
            last_word, incomplete_trailing,
            f"Answer for {depth} ends in an incomplete word/preposition: '{last_word}'"
        )

        # Must not have unmatched code fences
        self.assertEqual(
            stripped.count("```") % 2, 0,
            f"Answer for {depth} contains an unclosed code fence."
        )

        # Must not end with a colon expecting a list/section
        self.assertFalse(
            stripped.endswith(":") or stripped.endswith(": "),
            f"Answer for {depth} ends with a trailing colon."
        )

        # Must not end with an incomplete table row
        lines = stripped.splitlines()
        if lines:
            last_line = lines[-1].strip()
            if last_line.startswith("|"):
                self.assertTrue(
                    last_line.endswith("|"),
                    f"Answer for {depth} ends in an incomplete Markdown table row."
                )

    def test_token_limit_configurations(self):
        """Verify new token budget limits: Quick=800, Standard=1500, Deep=2500."""
        self.assertEqual(RESEARCH_DEPTH_CONFIG["quick"]["max_output_tokens"], 800)
        self.assertEqual(RESEARCH_DEPTH_CONFIG["standard"]["max_output_tokens"], 1500)
        self.assertEqual(RESEARCH_DEPTH_CONFIG["deep"]["max_output_tokens"], 2500)

    def test_query_http_vs_https_quick(self):
        """Test exact query: What is the difference between HTTP and HTTPS? (Quick depth)."""
        res = self.pipeline.run_pipeline("What is the difference between HTTP and HTTPS?", research_depth="quick")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "quick")
        self._assert_answer_complete(res["answer"], "quick")

    def test_query_binary_search_quick(self):
        """Test 1: What is binary search and how does it work? (Quick)."""
        res = self.pipeline.run_pipeline("What is binary search and how does it work?", research_depth="quick")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "quick")
        self._assert_answer_complete(res["answer"], "quick")

    def test_query_rest_api_standard(self):
        """Test 2: Explain REST API architecture and how requests and responses work. (Standard)."""
        res = self.pipeline.run_pipeline("Explain REST API architecture and how requests and responses work.", research_depth="standard")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "standard")
        self._assert_answer_complete(res["answer"], "standard")

    def test_query_machine_learning_deep(self):
        """Test 3: Explain machine learning, its major types, algorithms, applications, advantages, and limitations. (Deep)."""
        res = self.pipeline.run_pipeline("Explain machine learning, its major types, algorithms, applications, advantages, and limitations.", research_depth="deep")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "deep")
        self._assert_answer_complete(res["answer"], "deep")

    def test_query_sql_vs_nosql_deep(self):
        """Test 4: Compare SQL and NoSQL databases in terms of architecture, consistency, scalability, performance, and use cases. (Deep)."""
        res = self.pipeline.run_pipeline("Compare SQL and NoSQL databases in terms of architecture, consistency, scalability, performance, and use cases.", research_depth="deep")
        self.assertTrue(res["success"])
        self.assertEqual(res["research_depth"], "deep")
        self._assert_answer_complete(res["answer"], "deep")

    def test_quick_then_research_more_deep(self):
        """Test 5: Quick -> Research More -> Deep expansion test."""
        chat_id = "chat_completeness_test_101"
        user_id = "test_user_101"

        # 1. Quick request
        res1 = self.pipeline.run_pipeline("What is HTTP vs HTTPS?", user_id=user_id, chat_id=chat_id, research_depth="quick")
        self.assertTrue(res1["success"])
        self._assert_answer_complete(res1["answer"], "quick")

        # 2. Research More -> Deep request with active history
        res2 = self.pipeline.run_pipeline("What is HTTP vs HTTPS?", user_id=user_id, chat_id=chat_id, research_depth="deep")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["research_depth"], "deep")
        self._assert_answer_complete(res2["answer"], "deep")
        
        # Verify deep response is valid, complete, and contains content
        self.assertTrue(len(res2["answer"]) > 200)


if __name__ == "__main__":
    unittest.main()
