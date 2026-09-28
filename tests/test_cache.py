"""Focused tests for the semantic cache foundation and persistence layer."""

import json
import tempfile
import unittest
from pathlib import Path

from cache.cache_manager import CacheManager
from cache.intent_classifier import classify_query
from cache.query_normalizer import hash_query, normalize_query
from cache.redis_store import RedisStore
from cache.semantic_cache import SemanticCache
from cache.sqlite_store import SQLiteStore


class FakeRedis:
    """Small in-memory Redis substitute for deterministic semantic tests."""

    def __init__(self):
        self.embeddings = {}
        self.semantic_entries = {}
        self.index = set()

    def get_query_embedding(self, query_hash):
        return self.embeddings.get(query_hash)

    def set_query_embedding(self, query_hash, embedding, ttl=0):
        self.embeddings[query_hash] = embedding
        return True

    def get_semantic_entry(self, cache_id):
        return self.semantic_entries.get(cache_id)

    def set_semantic_entry(self, cache_id, metadata, ttl=0):
        self.semantic_entries[cache_id] = metadata
        return True

    def add_to_semantic_index(self, cache_id):
        self.index.add(cache_id)
        return True


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "cache.db"
        self.sqlite = SQLiteStore(self.db_path)
        self.redis = FakeRedis()
        self.semantic = SemanticCache(self.redis, self.sqlite)

        self.source_id = self.sqlite.upsert_source(
            "https://example.com/stack", "Stack", "example.com", "page-hash"
        )
        self.page_id = self.sqlite.upsert_page(
            self.source_id, "A stack is a Last-In, First-Out (LIFO) data structure.", "page-hash"
        )
        self.chunk_id = self.sqlite.upsert_chunk(
            self.page_id, "chunk-hash", "A stack is a Last-In, First-Out (LIFO) data structure.", 0,
            "https://example.com/stack", "Stack"
        )
        self.sqlite.upsert_embedding(
            self.chunk_id, "all-MiniLM-L6-v2", 384, "1.0", [0.1] * 384
        )

        self.normalized = normalize_query("Explain stack")
        self.query_hash = hash_query(self.normalized)
        self.sqlite.save_query(
            self.query_hash, self.normalized, "Explain stack",
            "concept_explanation", "stack", {}, "A stack is a Last-In, First-Out (LIFO) data structure.",
            [self.source_id], [self.chunk_id]
        )
        self.redis.embeddings[self.query_hash] = [1.0, 0.0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalization_and_hash_are_stable(self):
        self.assertEqual(normalize_query("  WHAT IS a Stack? "), "stack")
        self.assertEqual(len(hash_query("stack")), 64)

        # Cross-user normalization test
        q1_norm, q1_hash = normalize_query("What is the difference between HTTP and HTTPS?"), hash_query(normalize_query("What is the difference between HTTP and HTTPS?"))
        q2_norm, q2_hash = normalize_query("What is difference between HTTP and HTTPS?"), hash_query(normalize_query("What is difference between HTTP and HTTPS?"))
        self.assertEqual(q1_norm, "difference between http and https")
        self.assertEqual(q2_norm, "difference between http and https")
        self.assertEqual(q1_hash, q2_hash)

    def test_intent_and_requirements(self):
        result = classify_query("Explain stack in easy language with an example")
        self.assertEqual(result["intent"], "concept_explanation")
        self.assertEqual(result["scope"], "stack")
        self.assertTrue(result["requirements"]["easy_language"])
        self.assertTrue(result["requirements"]["example_required"])

    def test_semantic_answer_hit(self):
        decision = self.semantic.decide("Explain stack", self.normalized, [1.0, 0.0])
        self.assertEqual(decision.type, "ANSWER_HIT")
        self.assertEqual(decision.answer, "A stack is a Last-In, First-Out (LIFO) data structure.")

    def test_requirement_mismatch_requests_knowledge_reuse(self):
        query = normalize_query("Explain stack in easy language")
        decision = self.semantic.decide(query, query, [1.0, 0.0])
        self.assertEqual(decision.type, "KNOWLEDGE_REUSE")
        self.assertEqual(decision.chunk_ids, [self.chunk_id])

    def test_intent_mismatch_is_miss(self):
        query = normalize_query("Write code for stack")
        decision = self.semantic.decide(query, query, [1.0, 0.0])
        self.assertEqual(decision.type, "MISS")

    def test_source_invalidation_is_miss(self):
        self.sqlite.invalidate_source("https://example.com/stack")
        decision = self.semantic.decide("Explain stack", self.normalized, [1.0, 0.0])
        self.assertEqual(decision.type, "MISS")

    def test_page_chunk_and_embedding_round_trip(self):
        manager = object.__new__(CacheManager)
        manager.sqlite = self.sqlite
        manager.redis = self.redis
        manager.semantic_cache = self.semantic

        page = manager.get_page("https://example.com/stack")
        self.assertEqual(page["markdown"], "A stack is a Last-In, First-Out (LIFO) data structure.")
        chunks = manager.get_chunks_for_page(self.page_id)
        self.assertEqual(chunks[0]["chunk_text"], "A stack is a Last-In, First-Out (LIFO) data structure.")
        embedding = manager.get_embedding(self.chunk_id)
        self.assertEqual(len(embedding), 384)

    def test_json_cache_migration(self):
        legacy_path = Path(self.temp_dir.name) / "scraped_pages.json"
        legacy_path.write_text(
            json.dumps({"https://example.com/legacy": "Legacy content"}),
            encoding="utf-8"
        )
        migrated = self.sqlite.migrate_json_cache(legacy_path)
        self.assertEqual(migrated, 1)
        self.assertFalse(legacy_path.exists())
        self.assertTrue(legacy_path.with_suffix(".json.migrated").exists())
        self.assertIsNotNone(self.sqlite.get_source_by_url("https://example.com/legacy"))

    def test_redis_failure_is_graceful(self):
        store = RedisStore()
        self.assertIsNone(store.get_exact("missing"))
        self.assertFalse(store.set_exact("missing", {"answer": "x"}, ttl=1))
        store.close()

    def test_clear_cache_preserves_schema(self):
        self.assertTrue(self.sqlite.clear_all_cache())
        self.assertEqual(self.sqlite.get_all_query_metadata(), [])
        source_id = self.sqlite.upsert_source(
            "https://example.com/new", "New", "example.com", "new-hash"
        )
        self.assertIsNotNone(source_id)


if __name__ == "__main__":
    unittest.main()
