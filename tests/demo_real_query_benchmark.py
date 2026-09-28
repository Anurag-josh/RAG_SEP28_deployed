"""
Real Query Execution & Multi-Level Cache Benchmark Script
Demonstrates cache routing across L1 Redis, L2 MongoDB, L3 SQLite, and RAG Miss.
"""

import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from phase10.web_grounded_rag import WebGroundedRAGPipeline


def run_benchmark():
    print("=" * 75)
    print(" SOURCEIQ MULTI-LEVEL CACHE ROUTING REAL BENCHMARK ")
    print("=" * 75)

    pipeline = WebGroundedRAGPipeline()
    user_id = "demo_user_001"

    queries = [
        ("Query 1 (Cold Start)", "What is Merge Sort?"),
        ("Query 2 (Exact Repeated)", "What is Merge Sort?"),
        ("Query 3 (L2 Mongo Check)", "What is Merge Sort?"),
        ("Query 4 (Semantic Concept)", "Explain the merge sort algorithm"),
        ("Query 5 (Real-time Freshness)", "What is today's gold price?"),
    ]

    results_table = []

    for idx, (label, query) in enumerate(queries, 1):
        if label.startswith("Query 3"):
            # Simulate Redis L1 miss to force L2 MongoDB lookup
            pipeline.cache_manager.redis.clear_all_cache()
            print("\n[Simulating L1 Redis Cache Flush for Query 3...]")

        print(f"\n--- Running [{label}]: '{query}' ---")
        t0 = time.perf_counter()
        res = pipeline.run_pipeline(query, user_id=user_id, chat_history=[])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        cache_mode = res.get("cache_mode", "UNKNOWN")
        sources_count = len(res.get("sources", []))
        answer_len = len(res.get("answer", ""))

        results_table.append({
            "step": f"Q{idx}: {label}",
            "cache_mode": cache_mode,
            "latency_ms": f"{elapsed_ms:.2f} ms",
            "sources": sources_count,
            "ans_len": answer_len
        })

    print("\n" + "=" * 75)
    print(f"{'Step / Scenario':<32} | {'Cache Mode':<18} | {'Measured Latency':<16}")
    print("-" * 75)
    for r in results_table:
        print(f"{r['step']:<32} | {r['cache_mode']:<18} | {r['latency_ms']:<16}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
