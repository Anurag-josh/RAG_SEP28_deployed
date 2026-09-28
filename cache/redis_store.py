"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Redis cache store — connection pooling, key-namespaced CRUD for exact
         cache, semantic index, query embeddings, and search results.
         Redis is treated as optional: all methods log warnings on failure and
         return None / False without raising, so the RAG pipeline continues.
Dependencies: redis, json, logging, config.config
"""

import json
import logging
from typing import Any, Dict, List, Optional

try:
    import redis
    from redis.connection import ConnectionPool
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    redis = None  # type: ignore

from config.config import (
    REDIS_URL,
    REDIS_POOL_MAX_CONNECTIONS,
    EXACT_ANSWER_TTL,
    SEMANTIC_CACHE_TTL,
    SEARCH_CACHE_TTL,
)

logger = logging.getLogger("cache.redis_store")

# ─── Key Templates ────────────────────────────────────────────────────────────
# All keys use the rag: namespace prefix.
# Designed so future multi-user support can introduce:
#   rag:global:exact:<hash>
#   rag:user:<uid>:exact:<hash>

_PREFIX = "rag"

def _key_exact(query_hash: str) -> str:
    return f"{_PREFIX}:exact:{query_hash}"

def _key_semantic(cache_id: str) -> str:
    return f"{_PREFIX}:semantic:{cache_id}"

def _key_qembed(query_hash: str) -> str:
    return f"{_PREFIX}:qembed:{query_hash}"

def _key_search(query_hash: str) -> str:
    return f"{_PREFIX}:search:{query_hash}"

def _key_semantic_index() -> str:
    return f"{_PREFIX}:semantic:index"


# ─── RedisStore ───────────────────────────────────────────────────────────────

class RedisStore:
    """
    Manages all Redis operations for the semantic cache system.

    Uses a shared connection pool (created once, reused across all requests).
    If Redis is unavailable or any operation fails, methods return None/False
    and log a warning — they never raise exceptions to the caller.
    """

    def __init__(self) -> None:
        """Initializes the Redis connection pool. Does not raise if Redis is unavailable."""
        self._pool: Optional[Any] = None
        self._available = False

        if not _REDIS_AVAILABLE:
            logger.warning("redis-py package not installed. Redis cache disabled.")
            return

        try:
            self._pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=REDIS_POOL_MAX_CONNECTIONS,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Test connection
            client = redis.Redis(connection_pool=self._pool)
            client.ping()
            self._available = True
            logger.info("Redis connected successfully.")
        except Exception as e:
            logger.warning("Redis connection failed. Cache will run without Redis.")
            self._available = False

    @property
    def available(self) -> bool:
        """Returns True if Redis is reachable."""
        return self._available

    def _client(self) -> Optional[Any]:
        """Returns a Redis client from the pool, or None if unavailable."""
        if not self._available or self._pool is None:
            return None
        try:
            return redis.Redis(connection_pool=self._pool)
        except Exception:
            return None

    # ── Exact Answer Cache ────────────────────────────────────────────────────

    def get_exact(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an exact-match cached answer payload from Redis.

        Key: rag:exact:<query_hash>

        Args:
            query_hash: SHA-256 hash of the normalized query.

        Returns:
            Dict with answer and metadata, or None if not found.
        """
        client = self._client()
        if not client:
            return None
        try:
            raw = client.get(_key_exact(query_hash))
            if raw:
                logger.debug(f"[Redis] Exact cache HIT for hash {query_hash[:12]}...")
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning(f"[Redis] get_exact error: {e}")
            return None

    def set_exact(self, query_hash: str, payload: Dict[str, Any],
                  ttl: int = EXACT_ANSWER_TTL) -> bool:
        """
        Stores an exact-match answer payload in Redis with TTL.

        Key: rag:exact:<query_hash>

        Args:
            query_hash: SHA-256 hash of the normalized query.
            payload: Dict containing the full pipeline result (answer, sources, etc.).
            ttl: Time-to-live in seconds.

        Returns:
            True on success, False on error.
        """
        client = self._client()
        if not client:
            return False
        try:
            client.setex(_key_exact(query_hash), ttl, json.dumps(payload))
            logger.debug(f"[Redis] Exact cache SET for hash {query_hash[:12]}...")
            return True
        except Exception as e:
            logger.warning(f"[Redis] set_exact error: {e}")
            return False

    def delete_exact(self, query_hash: str) -> bool:
        """Deletes an exact cache entry."""
        client = self._client()
        if not client:
            return False
        try:
            client.delete(_key_exact(query_hash))
            return True
        except Exception as e:
            logger.warning(f"[Redis] delete_exact error: {e}")
            return False

    # ── Semantic Cache Entries ────────────────────────────────────────────────

    def get_semantic_entry(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a semantic cache metadata entry from Redis.

        Key: rag:semantic:<cache_id>

        Args:
            cache_id: The query_hash used as the semantic entry identifier.

        Returns:
            Dict with intent, scope, requirements, source_ids, chunk_ids, expires_at, or None.
        """
        client = self._client()
        if not client:
            return None
        try:
            raw = client.get(_key_semantic(cache_id))
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning(f"[Redis] get_semantic_entry error: {e}")
            return None

    def set_semantic_entry(self, cache_id: str, metadata: Dict[str, Any],
                           ttl: int = SEMANTIC_CACHE_TTL) -> bool:
        """
        Stores semantic cache metadata in Redis.

        Key: rag:semantic:<cache_id>

        Metadata should include: intent, scope, requirements, source_ids, chunk_ids, expires_at.
        """
        client = self._client()
        if not client:
            return False
        try:
            client.setex(_key_semantic(cache_id), ttl, json.dumps(metadata))
            return True
        except Exception as e:
            logger.warning(f"[Redis] set_semantic_entry error: {e}")
            return False

    # ── Semantic Index (list of cache_ids for cosine scan) ────────────────────

    def get_semantic_index(self) -> List[str]:
        """
        Returns all semantic cache_ids currently registered in the index.

        Key: rag:semantic:index (Redis HASH field → cache_id)

        Returns:
            List of cache_id strings.
        """
        client = self._client()
        if not client:
            return []
        try:
            result = client.hkeys(_key_semantic_index())
            return result if result else []
        except Exception as e:
            logger.warning(f"[Redis] get_semantic_index error: {e}")
            return []

    def add_to_semantic_index(self, cache_id: str) -> bool:
        """Registers a cache_id in the semantic index HASH."""
        client = self._client()
        if not client:
            return False
        try:
            client.hset(_key_semantic_index(), cache_id, "1")
            return True
        except Exception as e:
            logger.warning(f"[Redis] add_to_semantic_index error: {e}")
            return False

    def remove_from_semantic_index(self, cache_id: str) -> bool:
        """Removes a cache_id from the semantic index HASH."""
        client = self._client()
        if not client:
            return False
        try:
            client.hdel(_key_semantic_index(), cache_id)
            return True
        except Exception as e:
            logger.warning(f"[Redis] remove_from_semantic_index error: {e}")
            return False

    # ── Query Embedding Cache ─────────────────────────────────────────────────

    def get_query_embedding(self, query_hash: str) -> Optional[List[float]]:
        """
        Retrieves a previously cached query embedding vector from Redis.

        Key: rag:qembed:<query_hash>

        Returns:
            List of float embedding values, or None.
        """
        client = self._client()
        if not client:
            return None
        try:
            raw = client.get(_key_qembed(query_hash))
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning(f"[Redis] get_query_embedding error: {e}")
            return None

    def set_query_embedding(self, query_hash: str, embedding: List[float],
                            ttl: int = SEMANTIC_CACHE_TTL) -> bool:
        """
        Stores a query embedding vector in Redis with TTL.

        Key: rag:qembed:<query_hash>
        """
        client = self._client()
        if not client:
            return False
        try:
            client.setex(_key_qembed(query_hash), ttl, json.dumps(embedding))
            return True
        except Exception as e:
            logger.warning(f"[Redis] set_query_embedding error: {e}")
            return False

    # ── Search Results Cache ──────────────────────────────────────────────────

    def get_search_results(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieves cached Tavily search results for a query hash.

        Key: rag:search:<query_hash>
        """
        client = self._client()
        if not client:
            return None
        try:
            raw = client.get(_key_search(query_hash))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"[Redis] get_search_results error: {e}")
            return None

    def set_search_results(self, query_hash: str, results: List[Dict[str, Any]],
                           ttl: int = SEARCH_CACHE_TTL) -> bool:
        """
        Stores Tavily search results in Redis with TTL.

        Key: rag:search:<query_hash>
        """
        client = self._client()
        if not client:
            return False
        try:
            client.setex(_key_search(query_hash), ttl, json.dumps(results))
            return True
        except Exception as e:
            logger.warning(f"[Redis] set_search_results error: {e}")
            return False

    # ── Invalidation ──────────────────────────────────────────────────────────

    def delete_query(self, query_hash: str) -> bool:
        """Deletes all Redis keys associated with a query hash."""
        client = self._client()
        if not client:
            return False
        try:
            client.delete(
                _key_exact(query_hash),
                _key_semantic(query_hash),
                _key_qembed(query_hash),
                _key_search(query_hash),
            )
            self.remove_from_semantic_index(query_hash)
            return True
        except Exception as e:
            logger.warning(f"[Redis] delete_query error: {e}")
            return False

    def clear_all_cache(self) -> bool:
        """
        Deletes all keys matching the rag:* namespace from Redis.
        WARNING: Uses SCAN to avoid blocking the server.
        """
        client = self._client()
        if not client:
            return False
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = client.scan(cursor=cursor, match=f"{_PREFIX}:*", count=100)
                if keys:
                    client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"[Redis] Cleared {deleted} cache keys.")
            return True
        except Exception as e:
            logger.warning(f"[Redis] clear_all_cache error: {e}")
            return False

    def close(self) -> None:
        """Disconnects the connection pool gracefully."""
        if self._pool:
            try:
                self._pool.disconnect()
            except Exception:
                pass


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    store = RedisStore()
    print(f"Redis available: {store.available}")
    if store.available:
        ok = store.set_exact("testhash001", {"answer": "test answer", "sources": []}, ttl=60)
        print(f"set_exact: {ok}")
        result = store.get_exact("testhash001")
        print(f"get_exact: {result}")
        store.delete_exact("testhash001")
        result = store.get_exact("testhash001")
        print(f"After delete: {result}")
        print("Redis self-test complete.")
    else:
        print("Redis not available — skipping connection tests.")
