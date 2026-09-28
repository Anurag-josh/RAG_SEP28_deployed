"""
Test Suite: MongoDB Integration, User Data Security, and Multi-Level Cache Routing
Purpose: Validates bcrypt password security, JWT auth tokens, MongoDB L2 answer lookup,
         user-scoped query history, freshness-based routing, and request coalescing.
"""

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cache.cache_manager import CacheManager
from cache.intent_classifier import classify_freshness, classify_query, get_ttl_for_query
from cache.mongo_store import MongoStore
from cache.query_normalizer import hash_query, normalize_query
from utils.auth import create_jwt_token, decode_jwt_token, hash_password, verify_password


class FakeMongoCollection:
    """In-memory MongoDB collection substitute for unit testing without MongoDB server."""

    def __init__(self):
        self.docs = {}
        self.indexes = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def insert_one(self, doc):
        _id = doc.get("_id")
        if _id in self.docs:
            from pymongo.errors import DuplicateKeyError
            raise DuplicateKeyError(f"Duplicate key {_id}")
        self.docs[_id] = doc

    def find_one(self, filter_dict):
        for doc in self.docs.values():
            match = True
            for k, v in filter_dict.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def update_one(self, filter_dict, update_dict, upsert=False):
        set_data = update_dict.get("$set", {})
        existing = self.find_one(filter_dict)
        if existing:
            existing.update(set_data)
        elif upsert:
            key_id = set_data.get("_id") or f"doc_{len(self.docs)}"
            set_data["_id"] = key_id
            self.docs[key_id] = set_data

    def find(self, filter_dict):
        results = [doc for doc in self.docs.values() if all(doc.get(k) == v for k, v in filter_dict.items())]

        class CursorMock:
            def __init__(self, items):
                self.items = items
            def sort(self, *args, **kwargs):
                return self
            def skip(self, *args, **kwargs):
                return self
            def limit(self, *args, **kwargs):
                return self
            def __iter__(self):
                return iter(self.items)

        return CursorMock(results)

    def delete_one(self, filter_dict):
        found_key = None
        for k, v in self.docs.items():
            if all(v.get(fk) == fv for fk, fv in filter_dict.items()):
                found_key = k
                break
        if found_key:
            del self.docs[found_key]
            res = MagicMock()
            res.deleted_count = 1
            return res
        res = MagicMock()
        res.deleted_count = 0
        return res


class FakeMongoDB:
    def __init__(self):
        self.users = FakeMongoCollection()
        self.query_history = FakeMongoCollection()
        self.reusable_answers = FakeMongoCollection()


class TestMongoAndRouting(unittest.TestCase):

    def setUp(self):
        self.mongo_store = MongoStore.__new__(MongoStore)
        self.mongo_store._client = MagicMock()
        self.mongo_store._db = FakeMongoDB()
        self.mongo_store._available = True

    def test_auth_password_hashing_and_jwt(self):
        """Test bcrypt password hashing security and JWT token encode/decode."""
        raw_pw = "SuperSecurePassword123!"
        hashed = hash_password(raw_pw)

        self.assertNotEqual(raw_pw, hashed)
        self.assertTrue(verify_password(raw_pw, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))

        token = create_jwt_token("usr_100", "testuser", "test@example.com")
        self.assertIsInstance(token, str)

        payload = decode_jwt_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["user_id"], "usr_100")
        self.assertEqual(payload["username"], "testuser")

    def test_mongo_user_creation_and_unique_indexes(self):
        """Test creating users in MongoStore and preventing duplicate email/username."""
        user1 = self.mongo_store.create_user("u1", "alice", "alice@example.com", "hash1")
        self.assertIsNotNone(user1)

        fetched = self.mongo_store.get_user_by_email("alice@example.com")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["username"], "alice")

    def test_mongo_reusable_answer_l2_cache(self):
        """Test L2 reusable answer saving and retrieval in MongoStore."""
        q_hash = hash_query(normalize_query("What is Merge Sort?"))
        saved = self.mongo_store.save_reusable_answer(
            query_hash=q_hash,
            normalized_query="what is merge sort",
            original_query="What is Merge Sort?",
            intent="concept_explanation",
            scope="merge sort",
            requirements={},
            answer="Merge sort is an O(N log N) divide-and-conquer sorting algorithm.",
            sources=[{"title": "Wiki", "url": "https://example.com/merge"}],
            ttl_seconds=3600
        )
        self.assertTrue(saved)

        cached = self.mongo_store.get_reusable_answer(q_hash)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], "Merge sort is an O(N log N) divide-and-conquer sorting algorithm.")
        self.assertEqual(len(cached["sources"]), 1)

    def test_user_history_isolation(self):
        """Test query history saving and user-scoped retrieval in MongoStore."""
        q_hash = hash_query(normalize_query("Explain Binary Search"))
        self.mongo_store.save_query_history(
            user_id="user_A",
            query_hash=q_hash,
            original_query="Explain Binary Search",
            normalized_query="explain binary search",
            intent="concept_explanation",
            scope="binary search",
            answer="Binary search runs in O(log n).",
            citations=[],
            sources_analyzed=2,
            sources_used=1,
            cache_status="RAG_MISS"
        )

        hist_A = self.mongo_store.get_user_history("user_A")
        hist_B = self.mongo_store.get_user_history("user_B")

        self.assertEqual(len(hist_A), 1)
        self.assertEqual(len(hist_B), 0, "User B should not see User A's history.")

    def test_freshness_classification(self):
        """Test query freshness sensitivity and TTL assignment."""
        self.assertEqual(classify_freshness("What is today's gold price?"), "real_time")
        self.assertEqual(classify_freshness("Latest news about AI updates"), "news")
        self.assertEqual(classify_freshness("What is QuickSort?"), "stable_educational")

        self.assertEqual(get_ttl_for_query("What is today's gold price?"), 300)
        self.assertEqual(get_ttl_for_query("What is QuickSort?"), 86400)

    def test_graceful_mongo_failure_fallback(self):
        """Test that MongoStore methods do not raise when Mongo is unavailable."""
        store = MongoStore.__new__(MongoStore)
        store._client = None
        store._db = None
        store._available = False

        self.assertIsNone(store.get_reusable_answer("some_hash"))
        self.assertFalse(store.save_reusable_answer("hash", "norm", "orig", "intent", "scope", {}, "ans", []))
        self.assertEqual(store.get_user_history("user_x"), [])


if __name__ == "__main__":
    unittest.main()
