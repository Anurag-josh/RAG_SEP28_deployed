"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Application-scoped cache manager — single coordinator that unifies
         Redis, SQLite, and Semantic Cache operations behind a clean API.
         Initialize once at application startup; all RAG pipeline components
         call this manager rather than Redis or SQLite directly.
Dependencies: cache.redis_store, cache.sqlite_store, cache.semantic_cache,
              cache.query_normalizer, cache.intent_classifier, config.config
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.config import (
    CACHE_DB_PATH,
    CACHE_EMBEDDING_MODEL,
    CACHE_EMBEDDING_DIMENSION,
    CACHE_EMBEDDING_VERSION,
    EXACT_ANSWER_TTL,
    SEMANTIC_CACHE_TTL,
    PAGE_CACHE_TTL,
)
from cache.redis_store import RedisStore
from cache.sqlite_store import SQLiteStore
from cache.mongo_store import MongoStore
from cache.semantic_cache import SemanticCache, CacheDecision
from cache.query_normalizer import normalize_query, hash_query, hash_content
from cache.intent_classifier import classify_query, get_ttl_for_query

logger = logging.getLogger("cache.cache_manager")

UNCACHEABLE_ANSWER_PHRASES = [
    "couldn't find enough relevant web content",
    "could not find enough relevant web content",
    "no valid web sources were discovered",
    "web page extraction failed",
    "all web extractions failed",
    "temporarily unavailable",
    "generation failed:",
    "ambiguous context reference",
    "could you please clarify what topic",
    "could you please clarify"
]


def is_answer_usable(answer: Optional[str], status: Optional[str] = None, success: bool = True) -> bool:
    """
    Validates whether an answer is a successful and usable answer eligible for global answer caching.
    Rejects failures, insufficient evidence responses, errors, and clarification requests.
    """
    if not success:
        return False
    if status and str(status).upper() in ("INSUFFICIENT_EVIDENCE", "FAILED", "ERROR", "AMBIGUOUS_CONTEXT", "AMBIGUOUS"):
        return False
    if not answer or not isinstance(answer, str) or len(answer.strip()) < 20:
        return False
    ans_lower = answer.lower()
    for phrase in UNCACHEABLE_ANSWER_PHRASES:
        if phrase in ans_lower:
            return False
    return True


class CacheManager:
    """
    Application-scoped coordinator for all multi-level cache operations.

    Architecture:
        - L1 Redis: Fast cache for hot exact answers, semantic metadata, query embeddings
        - L2 MongoDB: Persistent application store for users, reusable answers, and query history
        - L3 SQLite: Persistent storage for pages, chunks, embeddings, and semantic RAG cache
        - SemanticCache: Decision engine (ANSWER_HIT / KNOWLEDGE_REUSE / MISS)
    """

    def __init__(self, db_path: Optional[Path] = None, mongo_uri: Optional[str] = None) -> None:
        """
        Initializes the CacheManager with shared Redis, MongoDB, and SQLite instances.
        Performs JSON cache migration if the legacy scraped_pages.json exists.
        """
        logger.info("Initializing CacheManager with L1 Redis + L2 MongoDB + L3 SQLite...")

        # Initialize stores
        self.sqlite = SQLiteStore(db_path=db_path)
        self.redis = RedisStore()
        self.mongo = MongoStore(uri=mongo_uri, fallback_store=self.sqlite)
        self.semantic_cache = SemanticCache(self.redis, self.sqlite)

        # Stampede protection / request coalescing locks
        self._inflight_lock = threading.Lock()
        self._inflight_events: Dict[str, threading.Event] = {}

        logger.info(
            f"CacheManager ready. Redis={'available' if self.redis.available else 'UNAVAILABLE (degraded mode)'}. "
            f"MongoDB={'available' if self.mongo.available else 'UNAVAILABLE (degraded mode)'}. "
            f"SQLite={self.sqlite.db_path}"
        )

        # Migrate legacy JSON page cache if it exists
        self._migrate_legacy_cache()


    # ── Exact Cache ──────────────────────────────────────────────────────────

    def get_exact(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Returns the exact-match cached pipeline result from Redis.

        Args:
            query_hash: SHA-256 hash of the normalized query.

        Returns:
            Full pipeline result dict (answer, sources, timings, etc.) or None.
        """
        t0 = time.perf_counter()
        result = self.redis.get_exact(query_hash)
        elapsed = (time.perf_counter() - t0) * 1000
        if result:
            ans = result.get("answer", "")
            stat = result.get("status", "SUCCESS")
            succ = result.get("success", True)
            if not is_answer_usable(ans, stat, succ):
                logger.warning(f"[Redis] Invalidating stale/unusable exact answer entry for hash {query_hash[:12]}...")
                self.redis.delete_exact(query_hash)
                return None
            logger.info(f"[Redis] GLOBAL EXACT HIT ({elapsed:.1f} ms) for hash {query_hash[:12]}...")
        else:
            logger.info(f"[Redis] MISS ({elapsed:.1f} ms) for hash {query_hash[:12]}...")
        return result

    def set_exact(self, query_hash: str, payload: Dict[str, Any]) -> None:
        """
        Stores the full pipeline result in Redis for exact-match retrieval.

        Args:
            query_hash: SHA-256 hash of the normalized query.
            payload: Complete pipeline result dict.
        """
        ans = payload.get("answer", "")
        stat = payload.get("status", "SUCCESS")
        succ = payload.get("success", True)
        if not is_answer_usable(ans, stat, succ):
            logger.warning(f"[Cache] set_exact skipped for unusable/insufficient answer (hash={query_hash[:12]}).")
            return

        try:
            self.redis.set_exact(query_hash, payload, ttl=EXACT_ANSWER_TTL)
        except Exception as e:
            logger.warning(f"[Cache] set_exact failed: {e}")

    # ── L2 MongoDB Reusable Answer Cache ─────────────────────────────────────

    def get_mongo_answer(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a persistent reusable answer from MongoDB (L2 Cache).
        If found, backfills L1 Redis so subsequent lookups hit L1 Redis.
        """
        t0 = time.perf_counter()
        res = self.mongo.get_reusable_answer(query_hash)
        elapsed = (time.perf_counter() - t0) * 1000
        if not res:
            logger.info(f"[Mongo] MISS ({elapsed:.1f} ms) for hash {query_hash[:12]}...")
            return None

        ans = res.get("answer", "")
        if not is_answer_usable(ans):
            logger.warning(f"[Mongo] Invalidating stale/unusable answer for hash {query_hash[:12]}...")
            return None

        logger.info(f"[Mongo] GLOBAL REUSABLE ANSWER HIT ({elapsed:.1f} ms) for hash {query_hash[:12]}...")
        payload = {
            "success": True,
            "answer": ans,
            "sources": res.get("sources", []),
            "research_depth": res.get("research_depth", "quick"),
            "sources_analyzed": res.get("sources_analyzed", len(res.get("sources", []))),
            "sources_used": len(res.get("sources", [])),
            "timings": {"mongodb_lookup_ms": elapsed},
            "statistics": {"Cache Tier": "MongoDB (L2 Persistent)"},
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "error": None,
            "cache_mode": "MONGODB_HIT"
        }
        # Backfill L1 Redis exact cache
        self.set_exact(query_hash, payload)
        return payload

    # ── Request Coalescing (Stampede Protection) ─────────────────────────────

    def acquire_coalesce_lock(self, query_hash: str) -> Tuple[bool, Optional[threading.Event]]:
        """
        Attempts to acquire leadership for processing a query hash.
        If another thread is already processing the same query, returns (False, event) to wait on.
        If this thread is the first, returns (True, event).
        """
        with self._inflight_lock:
            if query_hash in self._inflight_events:
                return False, self._inflight_events[query_hash]
            event = threading.Event()
            self._inflight_events[query_hash] = event
            return True, event

    def release_coalesce_lock(self, query_hash: str) -> None:
        """Notifies waiting threads that processing for query_hash has completed."""
        with self._inflight_lock:
            event = self._inflight_events.pop(query_hash, None)
            if event:
                event.set()


    # ── Query Embedding Cache ─────────────────────────────────────────────────

    def get_query_embedding(self, query_hash: str) -> Optional[List[float]]:
        """Returns a cached query embedding from Redis first, or SQLite as fallback."""
        vector = self.redis.get_query_embedding(query_hash)
        if vector is not None:
            return vector
        vector = self.sqlite.get_query_embedding(query_hash)
        if vector is not None:
            self.set_query_embedding(query_hash, vector)
        return vector

    def set_query_embedding(self, query_hash: str, embedding: List[float]) -> None:
        """Stores a query embedding in Redis."""
        try:
            self.redis.set_query_embedding(query_hash, embedding, ttl=SEMANTIC_CACHE_TTL)
        except Exception as e:
            logger.warning(f"[Cache] set_query_embedding failed: {e}")

    # ── Semantic Cache Decision ───────────────────────────────────────────────

    def decide_semantic(self, original_query: str, normalized_query: str,
                        query_embedding: List[float]) -> CacheDecision:
        """
        Delegates to SemanticCache to classify the query.

        Args:
            original_query: Raw user query.
            normalized_query: Output of normalize_query().
            query_embedding: Embedding of the normalized query.

        Returns:
            CacheDecision with type ANSWER_HIT, KNOWLEDGE_REUSE, or MISS.
        """
        return self.semantic_cache.decide(original_query, normalized_query, query_embedding)

    def find_similar_queries(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Returns semantic candidates using the configured cache backend."""
        return self.semantic_cache.find_similar_queries(embedding, top_k)

    def get_semantic_candidate(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """Returns one Redis semantic entry, or None when Redis is unavailable."""
        return self.redis.get_semantic_entry(cache_id)

    def set_semantic_entry(self, cache_id: str, metadata: Dict[str, Any], ttl: int = SEMANTIC_CACHE_TTL) -> bool:
        """Stores and indexes one semantic cache entry."""
        stored = self.redis.set_semantic_entry(cache_id, metadata, ttl=ttl)
        indexed = self.redis.add_to_semantic_index(cache_id)
        return stored and indexed

    # ── Page Cache (replaces website_loader JSON cache) ───────────────────────

    def get_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Returns cached page data for a URL from SQLite, or None if not cached/expired.

        Returns:
            Dict with keys: source_id, page_id, markdown, content_hash, domain, title.
        """
        source = self.sqlite.get_source_by_url(url)
        if not source:
            return None
        page = self.sqlite.get_page_by_source_id(source["id"])
        if not page:
            return None
        return {
            "source_id": source["id"],
            "page_id": page["id"],
            "markdown": page["markdown"],
            "content_hash": page["content_hash"],
            "domain": source["domain"],
            "title": source["title"],
            "url": url,
        }

    def save_page(self, url: str, markdown: str, title: str = "",
                  domain: str = "") -> Optional[Dict[str, int]]:
        """
        Saves a scraped page to SQLite. Returns source_id and page_id, or None on error.

        Args:
            url: Source URL.
            markdown: Scraped/cleaned page content.
            title: Page title.
            domain: Hostname/domain string.

        Returns:
            Dict {"source_id": int, "page_id": int} or None.
        """
        try:
            content_hash = hash_content(markdown)
            source_id = self.sqlite.upsert_source(url, title, domain, content_hash)
            if not source_id:
                return None
            page_id = self.sqlite.upsert_page(source_id, markdown, content_hash)
            if not page_id:
                return None
            return {"source_id": source_id, "page_id": page_id, "content_hash": content_hash}
        except Exception as e:
            logger.error(f"[Cache] save_page failed for {url}: {e}")
            return None

    # ── Chunk Cache ────────────────────────────────────────────────────────────

    def get_chunks_for_page(self, page_id: int) -> List[Dict[str, Any]]:
        """Returns all cached chunks for a page_id, or empty list."""
        try:
            return self.sqlite.get_chunks_by_page_id(page_id)
        except Exception as e:
            logger.error(f"[Cache] get_chunks_for_page failed: {e}")
            return []

    def save_chunks(self, page_id: int, chunks: List[Dict[str, Any]],
                    url: str = "", title: str = "") -> List[int]:
        """
        Saves text chunks derived from a page to SQLite.

        Args:
            page_id: SQLite page row ID.
            chunks: List of chunk dicts (each must have 'text').
            url: Source URL (stored in chunk for convenience).
            title: Source title.

        Returns:
            List of chunk IDs (empty on error).
        """
        chunk_ids = []
        try:
            for idx, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                if not text:
                    continue
                chunk_hash = hash_content(text)
                chunk_id = self.sqlite.upsert_chunk(
                    page_id, chunk_hash, text, idx, url, title
                )
                if chunk_id:
                    chunk_ids.append(chunk_id)
        except Exception as e:
            logger.error(f"[Cache] save_chunks failed: {e}")
        return chunk_ids

    # ── Embedding Cache ────────────────────────────────────────────────────────

    def get_embedding(self, chunk_id: int) -> Optional[List[float]]:
        """Returns the cached embedding for a chunk_id (current model/version), or None."""
        try:
            return self.sqlite.get_embedding(
                chunk_id, CACHE_EMBEDDING_MODEL, CACHE_EMBEDDING_VERSION
            )
        except Exception as e:
            logger.error(f"[Cache] get_embedding failed: {e}")
            return None

    def save_embedding(self, chunk_id: int, vector: List[float]) -> bool:
        """Saves an embedding vector to SQLite for a chunk_id."""
        try:
            return self.sqlite.upsert_embedding(
                chunk_id, CACHE_EMBEDDING_MODEL, CACHE_EMBEDDING_DIMENSION,
                CACHE_EMBEDDING_VERSION, vector
            )
        except Exception as e:
            logger.error(f"[Cache] save_embedding failed: {e}")
            return False

    # ── Knowledge Retrieval (for KNOWLEDGE_REUSE) ──────────────────────────────

    def get_cached_knowledge(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieves cached chunks with their embeddings for KNOWLEDGE_REUSE.
        Also re-validates that each chunk's source domain is still valid.

        Args:
            chunk_ids: List of chunk IDs from the semantic cache decision.

        Returns:
            List of dicts ready for ChromaDB ingestion:
            { id, text, url, title, embedding, metadata }
        """
        if not chunk_ids:
            return []

        chunks = self.sqlite.get_chunks_by_ids(chunk_ids)
        if not chunks:
            return []

        embeddings_map = self.sqlite.get_embeddings_by_chunk_ids(
            [c["id"] for c in chunks],
            CACHE_EMBEDDING_MODEL,
            CACHE_EMBEDDING_VERSION,
        )

        result = []
        for chunk in chunks:
            cid = chunk["id"]
            embedding = embeddings_map.get(cid)
            if not embedding:
                logger.debug(f"[Cache] No embedding for chunk_id={cid} — skipping")
                continue
            result.append({
                "id": f"{chunk.get('url', 'cached')}#chunk-{cid}",
                "text": chunk["chunk_text"],
                "embedding": embedding,
                "url": chunk.get("url", ""),
                "title": chunk.get("title", ""),
                "metadata": {
                    "url": chunk.get("url", ""),
                    "title": chunk.get("title", ""),
                    "chunk_number": chunk.get("chunk_index", 0),
                    "document_id": f"cached_chunk_{cid}",
                },
            })

        logger.info(f"[Cache] Loaded {len(result)} cached chunks for KNOWLEDGE_REUSE")
        return result

    # ── Full Result Persistence ────────────────────────────────────────────────

    def save_full_result(
        self,
        query_hash: str,
        normalized_query: str,
        original_query: str,
        query_embedding: List[float],
        answer: str,
        pipeline_result: Dict[str, Any],
        embedded_chunks: List[Dict[str, Any]],
        loaded_docs: List[Dict[str, Any]],
        research_depth: str = "quick"
    ) -> None:
        """
        Persists all artifacts from a successful Full RAG run to Redis + SQLite.
        Called only after Groq successfully generates an answer.

        Flow:
            1. Classify query (intent, scope, requirements)
            2. Save pages → SQLite
            3. Save chunks → SQLite
            4. Save embeddings → SQLite
            5. Save query metadata + answer → SQLite
            6. Register semantic entry in Redis index
            7. Cache query embedding in Redis
            8. Set exact answer in Redis

        Args:
            query_hash: SHA-256 hash of normalized_query.
            normalized_query: Normalized query text.
            original_query: Raw user query.
            query_embedding: Embedding of normalized_query.
            answer: Generated answer text from Groq.
            pipeline_result: Full pipeline result dict.
            embedded_chunks: List of {text, url, title, embedding, ...} dicts.
            loaded_docs: List of {url, content, title, ...} dicts from Phase 3.
        """
        t0 = time.perf_counter()
        if not is_answer_usable(answer, pipeline_result.get("status"), pipeline_result.get("success", True)):
            logger.warning(f"[Cache] Skipping cache population for unusable/insufficient answer (hash={query_hash[:12]}).")
            return

        logger.info(f"[Cache] Persisting full RAG result for hash {query_hash[:12]}...")

        # 1. Classify query
        try:
            classification = classify_query(normalized_query)
            intent = classification["intent"]
            scope = classification["scope"]
            requirements = classification["requirements"]
        except Exception as e:
            logger.error(f"[Cache] Classification failed during save: {e}")
            intent, scope, requirements = "concept_explanation", "", {}

        # 2. Save pages + build chunk/embedding store
        source_ids: List[int] = []
        all_chunk_ids: List[int] = []

        url_to_source_id: Dict[str, int] = {}

        for doc in loaded_docs:
            if not doc.get("success") or not doc.get("content"):
                continue
            url = doc.get("url", "")
            title = doc.get("title", "")
            from urllib.parse import urlparse
            domain = urlparse(url).hostname or ""
            domain = domain.replace("www.", "")

            page_info = self.save_page(url, doc["content"], title, domain)
            if not page_info:
                continue
            src_id = page_info["source_id"]
            page_id = page_info["page_id"]
            url_to_source_id[url] = src_id
            if src_id not in source_ids:
                source_ids.append(src_id)

            # Save chunks for this page
            page_chunks = [c for c in embedded_chunks if c.get("url") == url or
                           c.get("metadata", {}).get("url") == url]
            if not page_chunks:
                continue

            chunk_ids_for_page = self.save_chunks(page_id, page_chunks, url, title)
            all_chunk_ids.extend(chunk_ids_for_page)

            # Save embeddings for each chunk
            for chunk, chunk_id in zip(page_chunks, chunk_ids_for_page):
                embedding = chunk.get("embedding")
                if embedding and chunk_id:
                    self.save_embedding(chunk_id, embedding)

        # 3. Save query metadata + answer to SQLite
        if all_chunk_ids:
            ok = self.sqlite.save_query(
                query_hash, normalized_query, original_query,
                intent, scope, requirements, answer,
                source_ids, all_chunk_ids,
                query_embedding=query_embedding
            )
            if ok:
                logger.debug(f"[Cache] Query metadata saved (chunks={len(all_chunk_ids)})")

                # 4. Register in Redis semantic index + store embedding
                self.set_query_embedding(query_hash, query_embedding)
                semantic_meta = {
                    "query_hash": query_hash,
                    "intent": intent,
                    "scope": scope,
                    "requirements": requirements,
                    "source_ids": source_ids,
                    "chunk_ids": all_chunk_ids,
                    "research_depth": research_depth,
                    "embedding_model": CACHE_EMBEDDING_MODEL,
                    "embedding_version": CACHE_EMBEDDING_VERSION,
                }
                self.redis.set_semantic_entry(query_hash, semantic_meta)
                self.redis.add_to_semantic_index(query_hash)

        # 5. Set exact answer in Redis (L1)
        exact_payload = {
            "success": True,
            "answer": answer,
            "sources": pipeline_result.get("sources", []),
            "research_depth": research_depth,
            "sources_analyzed": len(loaded_docs),
            "sources_used": len(pipeline_result.get("sources", [])),
            "timings": pipeline_result.get("timings", {}),
            "statistics": pipeline_result.get("statistics", {}),
            "token_usage": pipeline_result.get("token_usage", {}),
            "error": None,
        }
        self.set_exact(query_hash, exact_payload)

        # 6. Save reusable answer in MongoDB (L2)
        ttl = get_ttl_for_query(normalized_query)
        self.mongo.save_reusable_answer(
            query_hash=query_hash,
            normalized_query=normalized_query,
            original_query=original_query,
            intent=intent,
            scope=scope,
            requirements=requirements,
            answer=answer,
            sources=pipeline_result.get("sources", []),
            ttl_seconds=ttl,
            is_cacheable=True,
            research_depth=research_depth,
            sources_analyzed=len(loaded_docs)
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Cache] Persist complete in {elapsed:.1f} ms | "
            f"sources={len(source_ids)} chunks={len(all_chunk_ids)}"
        )

    def save_query_result(
        self,
        query_hash: str,
        normalized_query: str,
        original_query: str,
        intent: str,
        scope: str,
        requirements: Dict[str, Any],
        answer: str,
        source_ids: List[int],
        chunk_ids: List[int],
        embedding: List[float],
        pipeline_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Persists a result that reuses existing cached knowledge."""
        saved = self.sqlite.save_query(
            query_hash, normalized_query, original_query, intent, scope,
            requirements, answer, source_ids, chunk_ids,
            query_embedding=embedding
        )
        if not saved:
            return False

        self.set_query_embedding(query_hash, embedding)
        metadata = {
            "query_hash": query_hash,
            "intent": intent,
            "scope": scope,
            "requirements": requirements,
            "source_ids": source_ids,
            "chunk_ids": chunk_ids,
            "embedding_model": CACHE_EMBEDDING_MODEL,
            "embedding_version": CACHE_EMBEDDING_VERSION,
        }
        self.set_semantic_entry(query_hash, metadata)
        if pipeline_result is not None:
            self.set_exact(query_hash, pipeline_result)
        return True

    # ── Normalize + Hash ──────────────────────────────────────────────────────

    def normalize_and_hash(self, query: str) -> Tuple[str, str]:
        """
        Normalizes a query and returns (normalized_query, query_hash).
        Convenience method so callers don't need to import query_normalizer.
        """
        norm = normalize_query(query)
        h = hash_query(norm)
        return norm, h

    # ── Invalidation ──────────────────────────────────────────────────────────

    def invalidate_query(self, query_hash: str) -> None:
        """Removes a query entry from both Redis and SQLite."""
        self.redis.delete_query(query_hash)
        self.sqlite.invalidate_query(query_hash)
        logger.info(f"[Cache] Invalidated query {query_hash[:12]}...")

    def invalidate_source(self, url: str) -> None:
        """Marks a source URL as expired in SQLite."""
        self.sqlite.invalidate_source(url)
        logger.info(f"[Cache] Invalidated source {url}")

    def get_chat_context(self, user_id: str, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieves recent chat context messages strictly filtered by (user_id, chat_id)."""
        return self.mongo.get_chat_context(user_id, chat_id, limit=limit)

    def clear_all_cache(self) -> bool:
        """
        Clears all cached data from Redis (rag:* keys), MongoDB reusable answers, and SQLite tables.
        Does NOT delete the SQLite schema or the database file.

        Returns:
            True if both Redis and SQLite cleared successfully.
        """
        redis_ok = self.redis.clear_all_cache()
        mongo_ok = self.mongo.clear_reusable_answers()
        sqlite_ok = self.sqlite.clear_all_cache()
        if redis_ok and sqlite_ok:
            logger.info("[Cache] All caches cleared.")
        else:
            logger.warning(f"[Cache] Partial clear — Redis: {redis_ok}, Mongo: {mongo_ok}, SQLite: {sqlite_ok}")
        return redis_ok and sqlite_ok

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Closes all connections gracefully."""
        try:
            self.redis.close()
        except Exception as e:
            logger.warning(f"[Cache] Redis close error: {e}")
        try:
            self.mongo.close()
        except Exception as e:
            logger.warning(f"[Cache] Mongo close error: {e}")
        logger.info("[Cache] CacheManager closed.")

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _migrate_legacy_cache(self) -> None:
        """
        Migrates the legacy scraped_pages.json to SQLite on first startup.
        Silently skips if the file does not exist.
        """
        json_path = CACHE_DB_PATH.parent / "scraped_pages.json"
        if json_path.exists():
            logger.info(f"[Cache] Migrating legacy JSON cache from {json_path}...")
            count = self.sqlite.migrate_json_cache(json_path)
            if count:
                logger.info(f"[Cache] Migrated {count} pages from JSON cache to SQLite.")


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CacheManager — Self-Test")
    print("=" * 50)

    import tempfile, os
    tmp_db = Path(tempfile.mktemp(suffix=".db"))

    cm = CacheManager(db_path=tmp_db)
    print(f"Redis available: {cm.redis.available}")

    # Test normalize + hash
    norm, h = cm.normalize_and_hash("Explain stack data structure?")
    print(f"Normalized: {norm!r}")
    print(f"Hash: {h[:16]}...")

    # Test page save + get
    page_info = cm.save_page(
        "https://example.com/stack", "# Stack\n\nA stack is LIFO.",
        title="Stack Wikipedia", domain="example.com"
    )
    print(f"Page saved: {page_info}")

    got = cm.get_page("https://example.com/stack")
    print(f"Page retrieved: markdown_length={len(got['markdown']) if got else 0}")

    # Test chunk save + get
    if page_info:
        chunks = [{"text": "A stack is LIFO."}]
        chunk_ids = cm.save_chunks(page_info["page_id"], chunks, "https://example.com/stack", "Stack")
        print(f"Chunks saved: {chunk_ids}")

        if chunk_ids:
            em_ok = cm.save_embedding(chunk_ids[0], [0.1] * 384)
            print(f"Embedding saved: {em_ok}")
            em = cm.get_embedding(chunk_ids[0])
            print(f"Embedding retrieved: length={len(em) if em else 0}")

    cm.close()
    os.unlink(tmp_db)
    print("Self-test complete.")
