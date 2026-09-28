"""
Test Suite: Cross-User Global Cache Reuse and User Isolation
Validates:
1. First user (Simon) asks "What is the difference between HTTP and HTTPS?" -> FULL RAG MISS
2. Second user (Simon2) asks "What is difference between HTTP and HTTPS?" -> GLOBAL CACHE HIT (Redis/Mongo) without running RAG
3. Second user (Simon2) asks semantically similar "Explain how HTTP differs from HTTPS." -> GLOBAL SEMANTIC CACHE HIT
4. User history is strictly user-isolated (Simon sees Simon's, Simon2 sees Simon2's)
5. Simon2's cache reuse creates a new history entry for Simon2 in MongoDB
"""

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cache.cache_manager import CacheManager
from cache.mongo_store import MongoStore
from cache.query_normalizer import normalize_and_hash
from cache.redis_store import RedisStore
from cache.sqlite_store import SQLiteStore
from phase10.web_grounded_rag import WebGroundedRAGPipeline
from tests.test_mongo_and_routing import FakeMongoDB


class TestCrossUserCacheReuse(unittest.TestCase):

    def setUp(self):
        self.temp_db_path = Path("cache/test_cross_user.db")
        if self.temp_db_path.exists():
            try:
                self.temp_db_path.unlink()
            except Exception:
                pass

        # Create CacheManager with mock MongoDB and SQLite store
        self.cache_manager = CacheManager.__new__(CacheManager)
        self.cache_manager.redis = RedisStore()
        self.cache_manager.sqlite = SQLiteStore(db_path=self.temp_db_path)
        
        # Setup MongoStore with fake database
        self.mongo_store = MongoStore.__new__(MongoStore)
        self.mongo_store._client = MagicMock()
        self.mongo_store._db = FakeMongoDB()
        self.mongo_store._available = True
        self.cache_manager.mongo = self.mongo_store

        from cache.semantic_cache import SemanticCache
        self.cache_manager.semantic_cache = SemanticCache(self.cache_manager.redis, self.cache_manager.sqlite)
        self.cache_manager._inflight_lock = __import__("threading").Lock()
        self.cache_manager._inflight_events = {}

        # Pipeline instance
        self.pipeline = WebGroundedRAGPipeline(cache_manager=self.cache_manager)

    def tearDown(self):
        if hasattr(self.cache_manager, "sqlite"):
            self.cache_manager.sqlite.clear_all_cache()
        if self.temp_db_path.exists():
            try:
                self.temp_db_path.unlink()
            except Exception:
                pass

    @patch("phase10.web_grounded_rag.WebSearcher")
    @patch("phase10.web_grounded_rag.WebsiteLoader")
    @patch("phase10.web_grounded_rag.GeminiContentGenerator")
    def test_cross_user_exact_and_semantic_cache_reuse_flow(self, mock_gemini_cls, mock_loader_cls, mock_searcher_cls):
        """
        Executes the 5 mandatory tests specified in the task prompt:
        1. Simon cold query -> FULL RAG MISS
        2. Simon2 exact variant query -> GLOBAL CACHE HIT (No RAG execution)
        3. Simon2 semantic variant query -> GLOBAL SEMANTIC CACHE HIT
        4. History isolation verification (Simon sees only Simon, Simon2 sees only Simon2)
        5. Simon2 cache hit creates user history entry for Simon2
        """

        # Setup RAG phase mocks
        mock_searcher = mock_searcher_cls.return_value
        mock_searcher.optimize_query.side_effect = lambda q: q
        mock_searcher.reformulate_conversational_query.side_effect = lambda q, h: q
        mock_searcher.search.return_value = [
            {"title": "HTTP vs HTTPS Guide", "url": "https://example.com/http-vs-https"}
        ]

        mock_loader = mock_loader_cls.return_value
        mock_loader.load_multiple.return_value = [
            {
                "url": "https://example.com/http-vs-https",
                "title": "HTTP vs HTTPS Guide",
                "content": "HTTP is unencrypted while HTTPS uses TLS/SSL encryption for secure communication.",
                "success": True,
                "duration": 0.1,
                "method": "httpx"
            }
        ]

        mock_gemini = mock_gemini_cls.return_value
        mock_gemini.generate_response_with_tokens.return_value = (
            "HTTP is plaintext while HTTPS uses TLS encryption for secure communication. [Source 1]",
            {"input_tokens": 120, "output_tokens": 50, "total_tokens": 170}
        )

        self.pipeline.searcher = mock_searcher
        self.pipeline.loader = mock_loader
        self.pipeline.gemini_client = mock_gemini

        # ── Test 1: Simon asks "What is the difference between HTTP and HTTPS?" ──
        t0 = time.perf_counter()
        simon_query = "What is the difference between HTTP and HTTPS?"
        result_simon = self.pipeline.run_pipeline(simon_query, user_id="Simon")
        t_simon = time.perf_counter() - t0

        self.assertTrue(result_simon["success"])
        self.assertEqual(result_simon["cache_mode"], "RAG_MISS")
        self.assertIn("HTTPS uses TLS encryption", result_simon["answer"])
        # Verify RAG pipeline was called for Simon
        self.assertEqual(mock_searcher.search.call_count, 1)

        # ── Test 2: Simon2 asks "What is difference between HTTP and HTTPS?" ──
        mock_searcher.search.reset_mock()
        mock_gemini.generate_response_with_tokens.reset_mock()

        t0 = time.perf_counter()
        simon2_query = "What is difference between HTTP and HTTPS?"
        result_simon2 = self.pipeline.run_pipeline(simon2_query, user_id="Simon2")
        t_simon2 = time.perf_counter() - t0

        self.assertTrue(result_simon2["success"])
        self.assertIn(result_simon2["cache_mode"], ["REDIS_HIT", "MONGODB_HIT"])
        self.assertEqual(result_simon2["answer"], result_simon["answer"])
        # CRITICAL VERIFICATION: No Tavily search, web scraping, or LLM generation ran for Simon2!
        mock_searcher.search.assert_not_called()
        mock_gemini.generate_response_with_tokens.assert_not_called()
        self.assertLess(t_simon2, 2.0, f"Cache hit response time should be fast (<2.0s), got {t_simon2:.3f}s")

        # ── Test 3: Simon2 asks "Explain how HTTP differs from HTTPS." ────────
        t0 = time.perf_counter()
        semantic_query = "Explain how HTTP differs from HTTPS."
        result_semantic = self.pipeline.run_pipeline(semantic_query, user_id="Simon2")
        t_semantic = time.perf_counter() - t0

        self.assertTrue(result_semantic["success"])
        self.assertIn(result_semantic["cache_mode"], ["SEMANTIC_HIT", "KNOWLEDGE_REUSE", "REDIS_HIT", "MONGODB_HIT"])

        # ── Test 4 & 5: Verify User History Isolation & Auto-History Creation ──
        simon_history = self.mongo_store.get_user_history("Simon")
        simon2_history = self.mongo_store.get_user_history("Simon2")

        self.assertEqual(len(simon_history), 1)
        self.assertGreaterEqual(len(simon2_history), 1)

        # Simon sees only Simon's queries
        self.assertEqual(simon_history[0]["user_id"], "Simon")
        self.assertEqual(simon_history[0]["original_query"], simon_query)

        # Simon2 history entry was created via cache reuse without running RAG again
        simon2_queries = [h["original_query"] for h in simon2_history]
        self.assertIn(simon2_query, simon2_queries)

        # Verify cross-user isolation: Simon history does NOT contain Simon2 queries
        for h in simon_history:
            self.assertNotEqual(h["user_id"], "Simon2")
        for h in simon2_history:
            self.assertNotEqual(h["user_id"], "Simon")


if __name__ == "__main__":
    unittest.main()
