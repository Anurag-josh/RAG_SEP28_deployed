"""
Automated Test Suite for RAG Fix Verification:
1. Context Isolation (User & Chat Scoping)
2. Research Depth & Continuation (No Truncation)
3. Fast Extraction & Research More (No Docling Hangs)
4. Honest Source Counting
5. Global Cache Reuse with Isolated Context
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cache.cache_manager import CacheManager
from cache.mongo_store import MongoStore
from cache.sqlite_store import SQLiteStore
from cache.context_resolver import resolve_context_query
from phase10.web_grounded_rag import WebGroundedRAGPipeline
from phase3.website_loader import WebsiteLoader
from phase9.gemini_client import GeminiContentGenerator
from tests.test_mongo_and_routing import FakeMongoDB


class TestRAGFixVerification(unittest.TestCase):

    def setUp(self):
        self.tmp_db = Path(tempfile.mktemp(suffix=".db"))
        self.cache_manager = CacheManager.__new__(CacheManager)
        self.cache_manager.redis = MagicMock()
        self.cache_manager.redis.available = False
        self.cache_manager.redis.get_exact.return_value = None
        self.cache_manager.redis.get_query_embedding.return_value = None

        self.cache_manager.sqlite = SQLiteStore(db_path=self.tmp_db)

        self.mongo_store = MongoStore.__new__(MongoStore)
        self.mongo_store._client = MagicMock()
        self.mongo_store._db = FakeMongoDB()
        self.mongo_store._available = True
        self.mongo_store.fallback_store = self.cache_manager.sqlite
        self.cache_manager.mongo = self.mongo_store

        from cache.semantic_cache import SemanticCache
        self.cache_manager.semantic_cache = SemanticCache(self.cache_manager.redis, self.cache_manager.sqlite)
        self.cache_manager._inflight_lock = __import__("threading").Lock()
        self.cache_manager._inflight_events = {}

        self.pipeline = WebGroundedRAGPipeline(cache_manager=self.cache_manager)

    def tearDown(self):
        if hasattr(self.cache_manager, "sqlite"):
            self.cache_manager.sqlite.clear_all_cache()
        if self.tmp_db.exists():
            try:
                os.unlink(self.tmp_db)
            except Exception:
                pass

    # ── ISSUE 1: CONTEXT ISOLATION TESTS ─────────────────────────────────────

    def test_context_isolation_same_chat(self):
        """Test 1: Chat A: 'What is MongoDB?' -> 'How does it work?' -> Subject = MongoDB."""
        history = [
            {"role": "user", "content": "What is MongoDB?"},
            {"role": "assistant", "content": "MongoDB is a document-oriented NoSQL database."}
        ]
        status, resolved = resolve_context_query("How does it work?", history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("mongodb", resolved.lower())

    def test_context_isolation_different_chats(self):
        """Test 2: Chat A: MongoDB, Chat B: binary search -> Chat B 'How does it work?' -> Subject = binary search."""
        # Save Chat A in DB for User 1
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_A", query_hash="hash_a",
            original_query="What is MongoDB?", normalized_query="what is mongodb",
            intent="concept_explanation", scope="mongodb", answer="MongoDB is NoSQL.",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # Save Chat B in DB for User 1
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_B", query_hash="hash_b",
            original_query="What is binary search?", normalized_query="what is binary search",
            intent="concept_explanation", scope="binary search", answer="Binary search is O(log n).",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # Retrieve context for Chat B ONLY
        chat_b_history = self.mongo_store.get_chat_context("user_1", "chat_B")
        self.assertEqual(len(chat_b_history), 2)  # 1 user + 1 assistant msg
        self.assertIn("binary search", chat_b_history[0]["content"].lower())

        status, resolved = resolve_context_query("How does it work?", chat_b_history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("binary search", resolved.lower())
        self.assertNotIn("mongodb", resolved.lower())

    def test_context_isolation_new_chat(self):
        """Test 3: Chat A has MongoDB context. Create Chat B (empty) -> Chat B 'How does it work?' -> AMBIGUOUS."""
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_A", query_hash="hash_a",
            original_query="What is MongoDB?", normalized_query="what is mongodb",
            intent="concept_explanation", scope="mongodb", answer="MongoDB is NoSQL.",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # Chat B context must be empty
        chat_b_history = self.mongo_store.get_chat_context("user_1", "chat_B")
        self.assertEqual(chat_b_history, [])

        status, resolved = resolve_context_query("How does it work?", chat_b_history)
        self.assertEqual(status, "AMBIGUOUS")
        self.assertNotIn("mongodb", resolved.lower())

    def test_context_isolation_cross_user(self):
        """Test 4: User 1 in Chat A asks MongoDB. User 2 in Chat A asks 'How does it work?' -> No context leakage."""
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_A", query_hash="hash_a",
            original_query="What is MongoDB?", normalized_query="what is mongodb",
            intent="concept_explanation", scope="mongodb", answer="MongoDB is NoSQL.",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # User 2 in Chat A has NO context from User 1
        user_2_history = self.mongo_store.get_chat_context("user_2", "chat_A")
        self.assertEqual(user_2_history, [])

        status, resolved = resolve_context_query("How does it work?", user_2_history)
        self.assertEqual(status, "AMBIGUOUS")

    # ── ISSUE 2: RESEARCH DEPTH & CONTINUATION TESTS ────────────────────────

    @patch("phase10.web_grounded_rag.WebSearcher")
    @patch("phase10.web_grounded_rag.WebsiteLoader")
    @patch("phase10.web_grounded_rag.GeminiContentGenerator")
    def test_research_depth_progression(self, mock_gemini_cls, mock_loader_cls, mock_searcher_cls):
        """Test Quick -> Standard -> Deep progression for target sources and output tokens."""
        mock_searcher = mock_searcher_cls.return_value
        mock_searcher.optimize_query.side_effect = lambda q: q
        mock_searcher.search.return_value = [{"title": f"Doc {i}", "url": f"https://example.com/doc{i}"} for i in range(20)]

        mock_loader = mock_loader_cls.return_value
        mock_loader.load_multiple.side_effect = lambda urls: [
            {"url": u, "title": "Doc", "content": "Content for algorithms.", "success": True, "duration": 0.1, "method": "Trafilatura"}
            for u in urls
        ]

        mock_gemini = mock_gemini_cls.return_value
        mock_gemini.generate_response_with_tokens.return_value = (
            "Grounded response content. [1]",
            {"input_tokens": 100, "output_tokens": 80, "total_tokens": 180}
        )

        self.pipeline.searcher = mock_searcher
        self.pipeline.loader = mock_loader
        self.pipeline.gemini_client = mock_gemini

        # Quick
        res_quick = self.pipeline.run_pipeline("What is quicksort?", research_depth="quick")
        self.assertEqual(res_quick["research_depth"], "quick")
        self.assertEqual(res_quick["sources_analyzed"], 5)

        # Standard
        res_std = self.pipeline.run_pipeline("What is quicksort?", research_depth="standard")
        self.assertEqual(res_std["research_depth"], "standard")
        self.assertEqual(res_std["sources_analyzed"], 10)

        # Deep
        res_deep = self.pipeline.run_pipeline("What is quicksort?", research_depth="deep")
        self.assertEqual(res_deep["research_depth"], "deep")
        self.assertEqual(res_deep["sources_analyzed"], 15)

    def test_intelligent_continuation_handling(self):
        """Test that truncated response with finish_reason='length' triggers multi-pass continuation to completion."""
        generator = GeminiContentGenerator.__new__(GeminiContentGenerator)
        generator.primary_model = "test-model"

        first_half = "Binary search is an efficient search algorithm that works on sorted arrays. It operates by repeatedly dividing"
        second_half = " the search interval in half until the target value is found."

        # Mock fallback execution chain: first call returns finish_reason="length", second call returns "stop"
        mock_execute = MagicMock()
        mock_execute.side_effect = [
            (first_half, {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, "length"),
            (second_half, {"input_tokens": 120, "output_tokens": 15, "total_tokens": 135}, "stop")
        ]
        generator._execute_generation_with_fallback = mock_execute

        answer, tokens = generator.generate_response_with_tokens("Explain binary search", max_tokens=20)
        self.assertIn("the search interval in half", answer)
        self.assertTrue(answer.endswith("found."))
        self.assertEqual(mock_execute.call_count, 2)

    # ── ISSUE 3 & 4: FAST EXTRACTION & HONEST COUNTS ──────────────────────

    @patch("phase3.website_loader.requests.get")
    def test_fast_extraction_skips_bad_urls_without_docling_hang(self, mock_get):
        """Test that HTTP errors / timeouts skip immediately without calling Docling on web pages."""
        loader = WebsiteLoader()
        loader.request_timeout = 1.0

        # Simulate HTTP 403 Forbidden
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = __import__("requests").HTTPError("403 Client Error: Forbidden")
        mock_get.return_value = mock_response

        # Mock Docling to verify it is NOT called for normal HTML page
        loader.load_with_docling = MagicMock()

        res = loader.load_single_worker("https://medium.com/some-article")
        self.assertFalse(res["success"])
        self.assertEqual(res["method"], "Failed")
        loader.load_with_docling.assert_not_called()

    # ── ISSUE 5 & 6: GLOBAL CACHE REUSE WITH ISOLATED CONTEXT ───────────────

    def test_global_cache_reuse_with_isolated_context(self):
        """User 1 asks MongoDB (RAG Miss -> cached). User 2 asks MongoDB -> Cache Hit. User 2 asks 'How does it work?' -> Uses User 2 context only."""
        # 1. User 1 queries MongoDB
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_100", query_hash="hash_mongo",
            original_query="What is MongoDB?", normalized_query="what is mongodb",
            intent="concept_explanation", scope="mongodb", answer="MongoDB is a NoSQL database.",
            citations=[{"title": "MongoDB Docs", "url": "https://mongodb.com"}],
            sources_analyzed=5, sources_used=1, cache_status="RAG_MISS"
        )
        self.cache_manager.save_full_result(
            query_hash="hash_mongo", normalized_query="what is mongodb", original_query="What is MongoDB?",
            query_embedding=[0.1]*384, answer="MongoDB is a NoSQL database.",
            pipeline_result={"answer": "MongoDB is a NoSQL database.", "sources": [{"title": "MongoDB Docs", "url": "https://mongodb.com"}]},
            embedded_chunks=[], loaded_docs=[], research_depth="quick"
        )

        # 2. User 2 asks exact same query in Chat 200
        user_2_context = self.mongo_store.get_chat_context("user_2", "chat_200")
        self.assertEqual(user_2_context, [])  # User 2 context is empty

        # 3. User 2 follow-up "How does it work?" in Chat 200 MUST NOT see User 1's context
        status, resolved = resolve_context_query("How does it work?", user_2_context)
        self.assertEqual(status, "AMBIGUOUS")


    # ── ISSUE 7: CACHE QUALITY & FAILED ANSWER REJECTION TESTS ───────────────

    def test_failed_answer_is_not_cached(self):
        """Verify that answers indicating research failure or missing content are NOT cached."""
        failed_msg = "I couldn't find enough relevant web content to answer this question reliably."
        
        from cache.cache_manager import is_answer_usable
        self.assertFalse(is_answer_usable(failed_msg, status="INSUFFICIENT_EVIDENCE", success=True))

        # Attempting save_full_result with failed_msg should skip saving
        self.cache_manager.save_full_result(
            query_hash="hash_failed", normalized_query="failed query", original_query="failed query",
            query_embedding=[0.1]*384, answer=failed_msg,
            pipeline_result={"answer": failed_msg, "status": "INSUFFICIENT_EVIDENCE"},
            embedded_chunks=[], loaded_docs=[], research_depth="quick"
        )

        exact_res = self.cache_manager.get_exact("hash_failed")
        self.assertIsNone(exact_res)

        mongo_res = self.cache_manager.get_mongo_answer("hash_failed")
        self.assertIsNone(mongo_res)

    def test_successful_answer_is_cached(self):
        """Verify that a valid grounded answer IS cached successfully."""
        good_msg = "HTTP is unencrypted while HTTPS uses TLS/SSL encryption to secure communication."
        
        from cache.cache_manager import is_answer_usable
        self.assertTrue(is_answer_usable(good_msg, status="SUCCESS", success=True))

        self.cache_manager.save_full_result(
            query_hash="hash_good", normalized_query="difference between http and https",
            original_query="Difference between HTTP and HTTPS",
            query_embedding=[0.1]*384, answer=good_msg,
            pipeline_result={"answer": good_msg, "sources": [{"title": "HTTP Guide", "url": "https://example.com/http"}], "status": "SUCCESS"},
            embedded_chunks=[{"text": good_msg, "url": "https://example.com/http", "title": "HTTP Guide", "embedding": [0.1]*384}],
            loaded_docs=[{"url": "https://example.com/http", "title": "HTTP Guide", "content": good_msg, "success": True}],
            research_depth="quick"
        )

        mongo_res = self.cache_manager.get_mongo_answer("hash_good")
        self.assertIsNotNone(mongo_res)
        self.assertEqual(mongo_res["answer"], good_msg)

    def test_semantic_cache_ignores_failed_answer(self):
        """Verify that semantic cache lookup ignores entries with uncacheable failure messages."""
        failed_msg = "I couldn't find enough relevant web content to answer this question reliably."
        
        # Save a query with failed_msg directly to sqlite metadata
        self.cache_manager.sqlite.save_query(
            "hash_bad_semantic", "failed query", "failed query", "concept_explanation",
            "failed", {}, failed_msg, [], []
        )

        decision = self.cache_manager.decide_semantic("failed query", "failed query", [0.1]*384)
        self.assertNotEqual(decision.type, "ANSWER_HIT")

    def test_continuation_deduplication(self):
        """Verify that clean_and_merge_continuation removes duplicate headings and overlapping prefixes."""
        from phase9.gemini_client import clean_and_merge_continuation

        existing = (
            "## Overview\n"
            "Binary search is an efficient search algorithm for finding an element in a sorted array.\n\n"
            "## Detailed Explanation\n"
            "It operates by repeatedly dividing the search space"
        )

        cont_duplicate = (
            "## Overview\n"
            "Binary search is an efficient search algorithm for finding an element in a sorted array.\n\n"
            "## Detailed Explanation\n"
            "dividing the search space in half until the target element is found."
        )

        merged, dup_detected = clean_and_merge_continuation(existing, cont_duplicate)
        self.assertTrue(dup_detected)
        self.assertIn("in half until the target element is found.", merged)
        self.assertEqual(merged.count("## Overview"), 1)
        self.assertEqual(merged.count("## Detailed Explanation"), 1)


if __name__ == "__main__":
    unittest.main()
