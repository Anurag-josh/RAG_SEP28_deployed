"""Small local benchmark for cache decision latency.

This benchmark uses deterministic vectors and SQLite, so it does not call web or
LLM providers. It measures cache decision overhead only.
"""

import tempfile
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cache.query_normalizer import hash_query, normalize_query
from cache.semantic_cache import SemanticCache
from cache.sqlite_store import SQLiteStore


class BenchmarkRedis:
    def __init__(self):
        self.embeddings = {}

    def get_query_embedding(self, query_hash):
        return self.embeddings.get(query_hash)


def main():
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(Path(directory) / "benchmark.db")
        redis = BenchmarkRedis()
        semantic = SemanticCache(redis, store)
        source_id = store.upsert_source(
            "https://example.com/stack", "Stack", "example.com", "hash"
        )
        page_id = store.upsert_page(source_id, "A stack is LIFO.", "hash")
        chunk_id = store.upsert_chunk(
            page_id, "chunk", "A stack is LIFO.", 0,
            "https://example.com/stack", "Stack"
        )
        query = normalize_query("Explain stack")
        query_hash = hash_query(query)
        store.save_query(
            query_hash, query, "Explain stack", "concept_explanation", "stack", {},
            "A stack is LIFO.", [source_id], [chunk_id]
        )
        redis.embeddings[query_hash] = [1.0, 0.0]

        scenarios = [
            ("Semantic equivalent", "Explain stack"),
            ("New requirement", "Explain stack in easy language"),
            ("Different intent", "Write code for stack"),
        ]
        print("Scenario              | Decision             | Time (ms)")
        print("----------------------|---------------------|----------")
        for label, raw_query in scenarios:
            normalized = normalize_query(raw_query)
            started = time.perf_counter()
            decision = semantic.decide(raw_query, normalized, [1.0, 0.0])
            elapsed = (time.perf_counter() - started) * 1000
            print(f"{label:<21} | {decision.type:<19} | {elapsed:8.3f}")


if __name__ == "__main__":
    main()
