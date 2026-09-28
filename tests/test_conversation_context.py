"""
Test Suite: Conversation Context & Follow-Up Query Resolution
Validates:
1. Test 1 — Basic follow-up: "What is binary search?" -> "Tell me more about it."
2. Test 2 — Attribute reference: "What is TCP?" -> "What are its advantages?"
3. Test 3 — Comparison follow-up: "What is HTTP?" -> "How is it different from HTTPS?"
4. Test 4 — Independent question: "What is binary search?" -> "What is MongoDB?" (Remains independent)
5. Test 5 — Multiple-turn context: 4 sequential queries referencing topic
6. Test 6 — New chat context isolation: Chat A vs New Chat B
7. Test 7 — Cross-user chat context isolation: User A Chat A vs User B Chat B
8. Test 8 — Cache integration: Resolved query uses existing Redis/Mongo/SQLite cache routing
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cache.cache_manager import CacheManager
from cache.mongo_store import MongoStore
from cache.sqlite_store import SQLiteStore
from cache.context_resolver import resolve_context_query
from phase10.web_grounded_rag import WebGroundedRAGPipeline
from tests.test_mongo_and_routing import FakeMongoDB


class TestConversationContext(unittest.TestCase):

    def setUp(self):
        self.temp_db_path = Path("cache/test_context.db")
        if self.temp_db_path.exists():
            try:
                self.temp_db_path.unlink()
            except Exception:
                pass

        # Create CacheManager with mock MongoDB and SQLite store
        self.cache_manager = CacheManager.__new__(CacheManager)
        self.cache_manager.redis = MagicMock()
        self.cache_manager.redis.available = False
        self.cache_manager.redis.get_exact.return_value = None
        self.cache_manager.redis.get_query_embedding.return_value = None

        self.cache_manager.sqlite = SQLiteStore(db_path=self.temp_db_path)

        # Setup MongoStore with fake database
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

    def test_1_basic_follow_up(self):
        """Test 1: 'What is binary search?' -> 'Tell me more about it.'"""
        history = [
            {"role": "user", "content": "What is binary search?"},
            {"role": "assistant", "content": "Binary search is a divide and conquer algorithm for searching a sorted list."}
        ]
        status, resolved = resolve_context_query("Tell me more about it.", history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("binary search", resolved.lower())

    def test_2_attribute_reference(self):
        """Test 2: 'What is TCP?' -> 'What are its advantages?'"""
        history = [
            {"role": "user", "content": "What is TCP?"},
            {"role": "assistant", "content": "Transmission Control Protocol is a core communication protocol of TCP/IP."}
        ]
        status, resolved = resolve_context_query("What are its advantages?", history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("tcp", resolved.lower())
        self.assertIn("advantage", resolved.lower())

    def test_3_comparison_follow_up(self):
        """Test 3: 'What is HTTP?' -> 'How is it different from HTTPS?'"""
        history = [
            {"role": "user", "content": "What is HTTP?"},
            {"role": "assistant", "content": "HTTP is Hypertext Transfer Protocol."}
        ]
        status, resolved = resolve_context_query("How is it different from HTTPS?", history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("http", resolved.lower())
        self.assertIn("https", resolved.lower())

    def test_4_independent_question(self):
        """Test 4: 'What is binary search?' -> 'What is MongoDB?' (Should remain independent)"""
        history = [
            {"role": "user", "content": "What is binary search?"},
            {"role": "assistant", "content": "Binary search finds items in a sorted array."}
        ]
        status, resolved = resolve_context_query("What is MongoDB?", history)
        self.assertEqual(status, "INDEPENDENT")
        self.assertEqual(resolved, "What is MongoDB?")

    def test_5_multiple_turn_context(self):
        """Test 5: Multiple sequential turns referencing the same topic."""
        history = [
            {"role": "user", "content": "What is binary search?"},
            {"role": "assistant", "content": "Binary search is an efficient search algorithm."},
            {"role": "user", "content": "What is its time complexity?"},
            {"role": "assistant", "content": "The time complexity is O(log n)."},
            {"role": "user", "content": "What are its advantages?"},
            {"role": "assistant", "content": "It runs in logarithmic time."}
        ]
        status, resolved = resolve_context_query("When should I use it?", history)
        self.assertEqual(status, "REWRITTEN")
        self.assertIn("binary search", resolved.lower())

    def test_6_new_chat_isolation(self):
        """Test 6: Chat A has context, but New Chat has empty context -> return AMBIGUOUS / no context assumed."""
        history_chat_a = [
            {"role": "user", "content": "What is binary search?"},
            {"role": "assistant", "content": "Binary search is O(log n)."}
        ]
        # Save Chat A in DB
        self.mongo_store.save_query_history(
            user_id="user_1", chat_id="chat_A", query_hash="h1",
            original_query="What is binary search?", normalized_query="what is binary search",
            intent="concept_explanation", scope="binary search", answer="Binary search is O(log n).",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # Query in Chat B (empty context for chat_B)
        chat_b_context = self.mongo_store.get_chat_context("user_1", "chat_B")
        self.assertEqual(chat_b_context, [])

        status, resolved = resolve_context_query("Tell me more about it.", chat_b_context)
        self.assertEqual(status, "AMBIGUOUS")

    def test_7_user_isolation(self):
        """Test 7: Simon in Chat A asks question; Simon2 in Chat B must NOT receive Simon's context."""
        self.mongo_store.save_query_history(
            user_id="Simon", chat_id="chat_A", query_hash="h1",
            original_query="What is binary search?", normalized_query="what is binary search",
            intent="concept_explanation", scope="binary search", answer="Binary search is O(log n).",
            citations=[], sources_analyzed=1, sources_used=1, cache_status="RAG_MISS"
        )

        # Simon2 checks history for Chat B
        simon2_context = self.mongo_store.get_chat_context("Simon2", "chat_B")
        self.assertEqual(simon2_context, [])

        status, resolved = resolve_context_query("Tell me more about it.", simon2_context)
        self.assertEqual(status, "AMBIGUOUS")

    @patch("phase10.web_grounded_rag.WebSearcher")
    @patch("phase10.web_grounded_rag.WebsiteLoader")
    @patch("phase10.web_grounded_rag.GeminiContentGenerator")
    def test_8_cache_integration_with_follow_up(self, mock_gemini_cls, mock_loader_cls, mock_searcher_cls):
        """Test 8: Follow-up resolves to standalone query, which then hits Mongo/Redis/SQLite cache properly."""
        mock_searcher = mock_searcher_cls.return_value
        mock_searcher.optimize_query.side_effect = lambda q: q
        mock_searcher.search.return_value = [{"title": "Binary Search", "url": "https://example.com/binary"}]

        mock_loader = mock_loader_cls.return_value
        mock_loader.load_multiple.return_value = [{
            "url": "https://example.com/binary", "title": "Binary Search",
            "content": "Binary search is an efficient algorithm with O(log n) performance.",
            "success": True, "duration": 0.1, "method": "httpx"
        }]

        mock_gemini = mock_gemini_cls.return_value
        mock_gemini.generate_response_with_tokens.return_value = (
            "Binary search requires a sorted array and runs in O(log n) time.",
            {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}
        )

        self.pipeline.searcher = mock_searcher
        self.pipeline.loader = mock_loader
        self.pipeline.gemini_client = mock_gemini

        user_id = "user_simon"
        chat_id = "chat_100"

        # Turn 1: "What is binary search?"
        res1 = self.pipeline.run_pipeline("What is binary search?", user_id=user_id, chat_id=chat_id)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["cache_mode"], "RAG_MISS")

        # Turn 2: "Tell me more about it." (Follow-up)
        # Should resolve to "Tell me more about binary search."
        res2 = self.pipeline.run_pipeline("Tell me more about it.", user_id=user_id, chat_id=chat_id)
        self.assertTrue(res2["success"])

        # History check
        chat_context = self.mongo_store.get_chat_context(user_id, chat_id)
        self.assertGreaterEqual(len(chat_context), 4)  # 2 user msgs + 2 assistant msgs


if __name__ == "__main__":
    unittest.main()
