"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Semantic cache decision engine — classifies new queries as ANSWER_HIT,
         KNOWLEDGE_REUSE, or MISS based on vector similarity, intent, scope,
         requirements, and cache validity.
         Implements find_similar_queries() as an abstracted cosine scan, designed
         for easy upgrade to Redis Vector Search or ANN indexing.
Dependencies: numpy, cache.redis_store, cache.sqlite_store, cache.intent_classifier, config.config
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config.config import (
    CACHE_EMBEDDING_MODEL,
    CACHE_EMBEDDING_VERSION,
    SEMANTIC_CACHE_TOP_K,
    SEMANTIC_SIMILARITY_THRESHOLD,
    KNOWLEDGE_REUSE_MIN_SIMILARITY,
)
from cache.intent_classifier import (
    classify_query,
    intents_compatible,
    requirements_are_subset,
)
from cache.redis_store import RedisStore
from cache.sqlite_store import SQLiteStore

logger = logging.getLogger("cache.semantic_cache")


# ─── Decision Result ──────────────────────────────────────────────────────────

@dataclass
class CacheDecision:
    """
    Result of the semantic cache decision process.

    Attributes:
        type:          "ANSWER_HIT", "KNOWLEDGE_REUSE", or "MISS"
        answer:        Cached answer text (only for ANSWER_HIT)
        chunk_ids:     Chunk IDs to reuse (for KNOWLEDGE_REUSE)
        source_ids:    Source IDs to reuse (for KNOWLEDGE_REUSE)
        similarity:    Best cosine similarity found (0.0 if MISS)
        matched_hash:  query_hash of the matched cache entry (if any)
        intent:        Classified intent of the new query
        scope:         Extracted scope of the new query
        requirements:  Extracted requirements of the new query
        timings_ms:    Dict of sub-step timings in milliseconds
    """
    type: str = "MISS"
    answer: Optional[str] = None
    chunk_ids: List[int] = field(default_factory=list)
    source_ids: List[int] = field(default_factory=list)
    similarity: float = 0.0
    matched_hash: Optional[str] = None
    intent: str = "concept_explanation"
    scope: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)
    research_depth: str = "quick"
    timings_ms: Dict[str, float] = field(default_factory=dict)


# ─── Cosine Similarity ────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes the cosine similarity between two equal-length float vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        float in [-1.0, 1.0], clamped to [0.0, 1.0] for cache purposes.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


# ─── SemanticCache ────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic cache decision engine.

    Abstracts the similarity search behind find_similar_queries() so the rest
    of the application is agnostic to whether the backend is:
        - Application-side cosine scan (current implementation)
        - Redis Vector Search
        - ANN vector indexing

    Usage:
        decision = semantic_cache.decide(original_query, normalized_query, query_embedding)
        if decision.type == "ANSWER_HIT":   return decision.answer
        if decision.type == "KNOWLEDGE_REUSE": reuse decision.chunk_ids
    """

    def __init__(self, redis_store: RedisStore, sqlite_store: SQLiteStore) -> None:
        """
        Initializes the SemanticCache with shared store instances.

        Args:
            redis_store: Initialized RedisStore.
            sqlite_store: Initialized SQLiteStore.
        """
        self.redis = redis_store
        self.sqlite = sqlite_store
        logger.debug("SemanticCache initialized.")

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(self, original_query: str, normalized_query: str,
               query_embedding: List[float]) -> CacheDecision:
        """
        Main entry point. Classifies the query and returns a CacheDecision.

        Algorithm:
            1. Classify intent, scope, requirements of new query.
            2. Find top-K similar cached queries (cosine scan over SQLite metadata).
            3. For each candidate (descending similarity):
                a. Check similarity threshold
                b. Check intent compatibility
                c. Check scope similarity
                d. Check requirements match
                e. Check cache validity (expiry + model version)
                f. Decide ANSWER_HIT or KNOWLEDGE_REUSE
            4. If no candidate passes → MISS.

        Args:
            original_query: The raw user query string.
            normalized_query: Output of normalize_query().
            query_embedding: Embedding of the normalized query (same model as chunks).

        Returns:
            CacheDecision with type, answer/chunk_ids, and diagnostics.
        """
        decision = CacheDecision()
        timings = {}

        # ── Step 1: Classify new query ────────────────────────────────────────
        t0 = time.perf_counter()
        classification = classify_query(normalized_query)
        new_intent = classification["intent"]
        new_scope = classification["scope"]
        new_requirements = classification["requirements"]
        decision.intent = new_intent
        decision.scope = new_scope
        decision.requirements = new_requirements
        timings["classification_ms"] = (time.perf_counter() - t0) * 1000
        logger.debug(f"[SemanticCache] intent={new_intent} scope={new_scope!r} reqs={new_requirements}")

        # ── Step 2: Find similar cached queries ───────────────────────────────
        t0 = time.perf_counter()
        candidates = self.find_similar_queries(query_embedding, top_k=SEMANTIC_CACHE_TOP_K)
        timings["similarity_search_ms"] = (time.perf_counter() - t0) * 1000
        logger.info(f"[SemanticCache] Found {len(candidates)} semantic candidates")

        # ── Step 3: Validate each candidate ──────────────────────────────────
        best_sim = 0.0
        for candidate in candidates:
            sim = candidate["similarity"]
            if sim > best_sim:
                best_sim = sim
            decision.similarity = best_sim

            # Minimum threshold check
            if sim < KNOWLEDGE_REUSE_MIN_SIMILARITY:
                logger.debug(f"[SemanticCache] similarity {sim:.3f} below minimum — skipping")
                continue

            cache_hash = candidate["query_hash"]
            old_intent = candidate.get("intent", "concept_explanation")
            old_scope = candidate.get("scope", "")
            old_requirements = candidate.get("requirements_json", {})
            old_model = candidate.get("embedding_model", "")
            old_version = candidate.get("embedding_version", "")
            source_ids = candidate.get("source_ids_json", [])
            chunk_ids = candidate.get("chunk_ids_json", [])

            # ── Validity checks ──────────────────────────────────────────────
            # Embedding model / version must match
            if old_model != CACHE_EMBEDDING_MODEL or old_version != CACHE_EMBEDDING_VERSION:
                logger.debug(f"[SemanticCache] Model mismatch ({old_model}≠{CACHE_EMBEDDING_MODEL}) — MISS")
                continue

            if not self.sqlite.are_sources_valid(source_ids):
                logger.debug("[SemanticCache] Source validation failed — skipping candidate")
                continue

            # ── Intent compatibility ─────────────────────────────────────────
            intent_ok = intents_compatible(old_intent, new_intent)
            logger.debug(f"[SemanticCache] intent_ok={intent_ok} ({old_intent}→{new_intent}) sim={sim:.3f}")
            if not intent_ok:
                continue

            # ── Scope comparison ─────────────────────────────────────────────
            scope_ok = self._scopes_compatible(old_scope, new_scope)
            logger.debug(f"[SemanticCache] scope_ok={scope_ok} ({old_scope!r}→{new_scope!r})")
            if not scope_ok:
                continue

            # ── Requirement comparison ────────────────────────────────────────
            reqs_subset = requirements_are_subset(old_requirements, new_requirements)

            # ── Final decision ───────────────────────────────────────────────
            if sim >= SEMANTIC_SIMILARITY_THRESHOLD and reqs_subset:
                # All checks pass — return cached answer directly
                answer = self._get_cached_answer(cache_hash)
                from cache.cache_manager import is_answer_usable
                if answer and is_answer_usable(answer):
                    logger.info(
                        f"[SemanticCache] ANSWER_HIT | sim={sim:.3f} | "
                        f"intent={new_intent} | scope={new_scope!r}"
                    )
                    decision.type = "ANSWER_HIT"
                    decision.answer = answer
                    decision.chunk_ids = chunk_ids
                    decision.source_ids = source_ids
                    decision.matched_hash = cache_hash
                    decision.research_depth = candidate.get("research_depth", "quick")
                    decision.timings_ms = timings
                    return decision
                else:
                    logger.warning(f"[SemanticCache] Candidate answer for hash {cache_hash[:12]} was unusable/failed. Skipping ANSWER_HIT.")
                # Answer not retrievable or unusable — fall through to KNOWLEDGE_REUSE check

            if sim >= KNOWLEDGE_REUSE_MIN_SIMILARITY and intent_ok and scope_ok:
                # Similar enough but requirements differ — reuse cached knowledge
                logger.info(
                    f"[SemanticCache] KNOWLEDGE_REUSE | sim={sim:.3f} | "
                    f"intent={new_intent} | scope={new_scope!r} | reqs_subset={reqs_subset}"
                )
                if chunk_ids:
                    decision.type = "KNOWLEDGE_REUSE"
                    decision.chunk_ids = chunk_ids
                    decision.source_ids = source_ids
                    decision.matched_hash = cache_hash
                    decision.research_depth = candidate.get("research_depth", "quick")
                    decision.timings_ms = timings
                    return decision

        # ── MISS ──────────────────────────────────────────────────────────────
        logger.info(
            f"[SemanticCache] MISS | best_sim={best_sim:.3f} | "
            f"intent={new_intent} | scope={new_scope!r}"
        )
        decision.type = "MISS"
        decision.timings_ms = timings
        return decision

    # ── Similarity Search ─────────────────────────────────────────────────────

    def find_similar_queries(self, query_embedding: List[float],
                             top_k: int = SEMANTIC_CACHE_TOP_K) -> List[Dict[str, Any]]:
        """
        Finds the top-K most similar cached queries using cosine similarity.

        This method is intentionally abstracted so the backend can be swapped:
            - Current: application-side cosine scan over SQLite metadata
            - Future: Redis Vector Search, Faiss, or other ANN index

        Args:
            query_embedding: The embedding of the new query.
            top_k: Maximum number of candidates to return.

        Returns:
            List of dicts sorted by descending similarity, each containing
            the cached query metadata + 'similarity' key.
        """
        # Load all non-expired query metadata from SQLite
        all_metadata = self.sqlite.get_all_query_metadata()
        if not all_metadata:
            return []

        # Try to supplement with Redis for faster access to embeddings
        # (Redis stores the query embedding for each cache_id)
        scored: List[Dict[str, Any]] = []
        for meta in all_metadata:
            cache_hash = meta["query_hash"]

            # Retrieve the cached query embedding (from Redis first, else SQLite)
            cached_embedding = self.redis.get_query_embedding(cache_hash)
            if not cached_embedding:
                cached_embedding = self.sqlite.get_query_embedding(cache_hash)
                if cached_embedding:
                    self.redis.set_query_embedding(cache_hash, cached_embedding)
            if not cached_embedding:
                continue

            if len(cached_embedding) != len(query_embedding):
                continue  # Dimension mismatch

            sim = _cosine_similarity(query_embedding, cached_embedding)
            entry = dict(meta)
            entry["similarity"] = sim
            scored.append(entry)

        # Sort by similarity descending
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    # ── Cache Retrieval Helpers ────────────────────────────────────────────────

    def _get_cached_answer(self, query_hash: str) -> Optional[str]:
        """
        Retrieves the cached answer for a query hash.
        Checks SQLite (source of truth for answers).
        """
        return self.sqlite.get_query_answer(query_hash)

    def _scopes_compatible(self, scope_a: str, scope_b: str) -> bool:
        """
        Determines if two scopes are compatible for cache reuse.
        Uses token overlap: at least 60% of scope_a tokens must appear in scope_b
        or vice versa.

        Args:
            scope_a: Scope from the cached entry.
            scope_b: Scope from the new query.

        Returns:
            bool: True if scopes are compatible.
        """
        if not scope_a or not scope_b:
            return False
        if scope_a == scope_b:
            return True
        tokens_a = set(scope_a.lower().split())
        tokens_b = set(scope_b.lower().split())
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a & tokens_b)
        # At least 60% overlap relative to the smaller set
        threshold = 0.6
        smaller = min(len(tokens_a), len(tokens_b))
        return (overlap / smaller) >= threshold


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SemanticCache — Decision Table Self-Test")
    print("=" * 60)

    # Simulate decisions using the decision table directly
    from cache.intent_classifier import requirements_are_subset, intents_compatible

    tests = [
        {
            "description": "Exact same query (high similarity, same requirements)",
            "sim": 0.98, "intent_a": "concept_explanation", "intent_b": "concept_explanation",
            "scope_a": "stack data structure", "scope_b": "stack data structure",
            "reqs_a": {}, "reqs_b": {},
            "expected": "ANSWER_HIT"
        },
        {
            "description": "Same topic + new requirement (easy language)",
            "sim": 0.94, "intent_a": "concept_explanation", "intent_b": "concept_explanation",
            "scope_a": "stack data structure", "scope_b": "stack data structure",
            "reqs_a": {}, "reqs_b": {"easy_language": True},
            "expected": "KNOWLEDGE_REUSE"
        },
        {
            "description": "Different intent (concept vs code)",
            "sim": 0.91, "intent_a": "concept_explanation", "intent_b": "code_generation",
            "scope_a": "stack", "scope_b": "stack",
            "reqs_a": {}, "reqs_b": {"language": "java"},
            "expected": "MISS"
        },
        {
            "description": "Unrelated topic (low similarity)",
            "sim": 0.40, "intent_a": "concept_explanation", "intent_b": "concept_explanation",
            "scope_a": "stack", "scope_b": "merge sort",
            "expected": "MISS"
        },
    ]

    for t in tests:
        sim = t["sim"]
        intent_ok = intents_compatible(t["intent_a"], t["intent_b"])
        scope_ok = sim >= 0.60 and (t["scope_a"] == t["scope_b"] or
                   len(set(t["scope_a"].split()) & set(t["scope_b"].split())) >= 0.6 * min(
                       len(t["scope_a"].split()), len(t["scope_b"].split())))
        reqs_subset = requirements_are_subset(t.get("reqs_a", {}), t.get("reqs_b", {}))

        if sim >= SEMANTIC_SIMILARITY_THRESHOLD and intent_ok and scope_ok and reqs_subset:
            decision = "ANSWER_HIT"
        elif sim >= KNOWLEDGE_REUSE_MIN_SIMILARITY and intent_ok and scope_ok:
            decision = "KNOWLEDGE_REUSE"
        else:
            decision = "MISS"

        status = "✓" if decision == t["expected"] else "✗"
        print(f"  {status} {t['description']}")
        print(f"    Expected: {t['expected']} | Got: {decision} | sim={sim:.2f}")
        print()
