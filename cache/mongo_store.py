"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: MongoDB persistent store (L2 Cache & Application Database) — handles users,
         user-scoped query history, and reusable global answer metadata.
         MongoDB is treated as resilient: all methods catch errors and log warnings
         so the RAG pipeline continues seamlessly if MongoDB is unavailable.
Dependencies: pymongo, datetime, json, logging, config.config
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False
    pymongo = None  # type: ignore
    ASCENDING = 1
    DESCENDING = -1


from config.config import MONGODB_URI, MONGODB_DATABASE, EXACT_ANSWER_TTL

logger = logging.getLogger("cache.mongo_store")


def _utcnow_iso() -> str:
    """Returns current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(ttl_seconds: int) -> str:
    """Returns UTC expiry timestamp as ISO-8601 string."""
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


def _is_expired(expires_at_iso: Optional[str]) -> bool:
    """Returns True if the ISO timestamp is in the past."""
    if not expires_at_iso:
        return False
    try:
        exp = datetime.fromisoformat(expires_at_iso)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp
    except Exception:
        return False


class MongoStore:
    """
    Manages all MongoDB operations for users, query history, and L2 reusable answers.
    Uses connection pooling and graceful error handling so system degraded mode works.
    """

    fallback_store: Optional[Any] = None

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None, fallback_store: Optional[Any] = None) -> None:
        self.uri = uri or MONGODB_URI
        self.db_name = db_name or MONGODB_DATABASE
        self._client: Optional[Any] = None
        self._db: Optional[Any] = None
        self._available = False
        self.fallback_store = fallback_store

        if not _PYMONGO_AVAILABLE:
            logger.warning("pymongo package not installed. Using local SQLite fallback store for DB operations.")
            return

        try:
            self._client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
                maxPoolSize=20
            )
            # Test connection
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            self._available = True
            logger.info(f"MongoDB connected successfully to '{self.db_name}' at {self.uri}")
            self._init_indexes()
        except Exception as e:
            logger.warning(f"MongoDB connection failed ({e}). Running with local SQLite database store.")
            self._available = False

    def set_fallback_store(self, fallback_store: Any) -> None:
        """Sets or updates the local database fallback store (e.g., SQLiteStore)."""
        self.fallback_store = fallback_store

    @property
    def available(self) -> bool:
        """Returns True if MongoDB server or fallback store is available."""
        return getattr(self, '_available', False) or (getattr(self, 'fallback_store', None) is not None)

    def _init_indexes(self) -> None:
        """Creates required unique and query indexes across collections."""
        if not self._available or self._db is None:
            return
        try:
            # 1. Users Indexes
            self._db.users.create_index([("email", ASCENDING)], unique=True)
            self._db.users.create_index([("username", ASCENDING)], unique=True)

            # 2. Query History Indexes
            self._db.query_history.create_index([("user_id", ASCENDING)])
            self._db.query_history.create_index([("query_hash", ASCENDING)])
            self._db.query_history.create_index([("user_id", ASCENDING), ("query_hash", ASCENDING)], unique=True)
            self._db.query_history.create_index([("user_id", ASCENDING), ("chat_id", ASCENDING)])
            self._db.query_history.create_index([("created_at", DESCENDING)])

            # 3. Reusable Answers Indexes (L2 Cache)
            self._db.reusable_answers.create_index([("query_hash", ASCENDING)], unique=True)
            self._db.reusable_answers.create_index([("expires_at", ASCENDING)])
            logger.debug("MongoDB indexes initialized successfully.")
        except Exception as e:
            logger.warning(f"MongoDB index creation warning: {e}")

    # ── User Management ─────────────────────────────────────────────────────────

    def create_user(self, user_id: str, username: str, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """
        Registers a new user record in MongoDB or SQLite fallback.
        """
        if self._available and self._db is not None:
            now = _utcnow_iso()
            doc = {
                "_id": user_id,
                "user_id": user_id,
                "username": username.lower().strip(),
                "email": email.lower().strip(),
                "password_hash": password_hash,
                "created_at": now,
                "updated_at": now,
                "preferences": {}
            }
            try:
                self._db.users.insert_one(doc)
                logger.info(f"[Mongo] User registered: {username} ({email})")
                return doc
            except pymongo.errors.DuplicateKeyError as e:
                logger.warning(f"[Mongo] Create user duplicate key: {e}")
                return None
            except Exception as e:
                logger.error(f"[Mongo] create_user error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.create_user(user_id, username, email, password_hash)
        return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        if getattr(self, '_available', False) and self._db is not None:
            try:
                user = self._db.users.find_one({"email": email.lower().strip()})
                if user:
                    return user
            except Exception as e:
                logger.warning(f"[Mongo] get_user_by_email error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.get_user_by_email(email)
        return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        if getattr(self, '_available', False) and self._db is not None:
            try:
                user = self._db.users.find_one({"username": username.lower().strip()})
                if user:
                    return user
            except Exception as e:
                logger.warning(f"[Mongo] get_user_by_username error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.get_user_by_username(username)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if getattr(self, '_available', False) and self._db is not None:
            try:
                user = self._db.users.find_one({"_id": user_id})
                if user:
                    return user
            except Exception as e:
                logger.warning(f"[Mongo] get_user_by_id error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.get_user_by_id(user_id)
        return None

    # ── L2 Reusable Answer Cache ─────────────────────────────────────────────

    def get_reusable_answer(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a global reusable answer record from MongoDB (L2 Cache).
        Checks expiry timestamp before returning.
        """
        if not self._available or self._db is None:
            return None
        try:
            doc = self._db.reusable_answers.find_one({"query_hash": query_hash})
            if not doc:
                return None
            if _is_expired(doc.get("expires_at")):
                logger.info(f"[Mongo] Reusable answer for hash {query_hash[:12]} EXPIRED.")
                return None
            logger.debug(f"[Mongo] L2 Reusable answer HIT for hash {query_hash[:12]}")
            return {
                "answer": doc.get("answer"),
                "sources": doc.get("sources", []),
                "intent": doc.get("intent"),
                "scope": doc.get("scope"),
                "requirements": doc.get("requirements", {}),
                "research_depth": doc.get("research_depth", "quick"),
                "sources_analyzed": doc.get("sources_analyzed", len(doc.get("sources", []))),
                "created_at": doc.get("created_at"),
                "expires_at": doc.get("expires_at")
            }
        except Exception as e:
            logger.warning(f"[Mongo] get_reusable_answer error: {e}")
            return None

    def save_reusable_answer(
        self,
        query_hash: str,
        normalized_query: str,
        original_query: str,
        intent: str,
        scope: str,
        requirements: Dict[str, Any],
        answer: str,
        sources: List[Dict[str, Any]],
        ttl_seconds: int = EXACT_ANSWER_TTL,
        is_cacheable: bool = True,
        research_depth: str = "quick",
        sources_analyzed: int = 5
    ) -> bool:
        """
        Saves or updates a global reusable answer record in MongoDB.
        """
        if not self._available or self._db is None or not is_cacheable:
            return False
        if not answer or any(phrase in answer.lower() for phrase in [
            "couldn't find enough relevant web content",
            "could not find enough relevant web content",
            "no valid web sources",
            "web page extraction failed",
            "temporarily unavailable",
            "generation failed:"
        ]):
            logger.warning(f"[Mongo] save_reusable_answer skipped for unsuccessful/insufficient answer (hash={query_hash[:12]}).")
            return False
        now = _utcnow_iso()
        expires_at = _expires_iso(ttl_seconds)
        doc = {
            "query_hash": query_hash,
            "normalized_query": normalized_query,
            "original_query": original_query,
            "intent": intent,
            "scope": scope,
            "requirements": requirements,
            "answer": answer,
            "sources": sources,
            "research_depth": research_depth,
            "sources_analyzed": sources_analyzed,
            "is_cacheable": is_cacheable,
            "created_at": now,
            "expires_at": expires_at
        }
        try:
            self._db.reusable_answers.update_one(
                {"query_hash": query_hash},
                {"$set": doc},
                upsert=True
            )
            logger.debug(f"[Mongo] Reusable answer saved for hash {query_hash[:12]} (depth={research_depth})")
            return True
        except Exception as e:
            logger.warning(f"[Mongo] save_reusable_answer error: {e}")
            return False

    def clear_reusable_answers(self) -> bool:
        """Clears all reusable answer documents from MongoDB."""
        if self._available and self._db is not None:
            try:
                self._db.reusable_answers.delete_many({})
                return True
            except Exception as e:
                logger.warning(f"[Mongo] clear_reusable_answers error: {e}")
                return False
        return True

    # ── User Query History ───────────────────────────────────────────────────

    def save_query_history(
        self,
        user_id: str,
        query_hash: str,
        original_query: str,
        normalized_query: str,
        intent: str,
        scope: str,
        answer: str,
        citations: List[Dict[str, Any]],
        sources_analyzed: int,
        sources_used: int,
        cache_status: str,
        is_cacheable: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
        resolved_query: Optional[str] = None,
        research_depth: str = "quick"
    ) -> bool:
        """
        Persists query history for a specific user into MongoDB or SQLite fallback.
        """
        if not user_id:
            return False

        saved_mongo = False
        if self._available and self._db is not None:
            now = _utcnow_iso()
            history_id = f"hist_{user_id}_{query_hash[:16]}"
            doc = {
                "_id": history_id,
                "user_id": user_id,
                "chat_id": chat_id,
                "query_hash": query_hash,
                "original_query": original_query,
                "resolved_query": resolved_query or original_query,
                "normalized_query": normalized_query,
                "intent": intent,
                "scope": scope,
                "answer": answer,
                "citations": citations,
                "sources_analyzed": sources_analyzed,
                "sources_used": sources_used,
                "research_depth": research_depth,
                "cache_status": cache_status,
                "is_cacheable": is_cacheable,
                "created_at": now,
                "updated_at": now,
                "metadata": metadata or {}
            }
            try:
                filter_cond = {"user_id": user_id, "query_hash": query_hash}
                self._db.query_history.update_one(
                    filter_cond,
                    {"$set": doc},
                    upsert=True
                )
                logger.debug(f"[Mongo] Saved history for user {user_id} (chat_id={chat_id}, status={cache_status})")
                saved_mongo = True
            except Exception as e:
                logger.warning(f"[Mongo] save_query_history error: {e}")

        saved_fallback = False
        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            saved_fallback = fallback.save_query_history(
                user_id, query_hash, original_query, normalized_query, intent, scope,
                answer, citations, sources_analyzed, sources_used, cache_status,
                is_cacheable, metadata, chat_id, resolved_query
            )

        return saved_mongo or saved_fallback

    def get_chat_context(self, user_id: str, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Retrieves recent chat messages strictly scoped by (user_id, chat_id).
        Returns a list of dicts: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        if not user_id or not chat_id or str(chat_id).strip() in ("", "None", "null"):
            return []
        if getattr(self, '_available', False) and self._db is not None:
            try:
                cursor = self._db.query_history.find({"user_id": user_id, "chat_id": chat_id}).sort("created_at", DESCENDING).limit(limit)
                docs = list(cursor)
                docs.reverse()
                messages = []
                for doc in docs:
                    user_msg = doc.get("original_query") or doc.get("prompt") or ""
                    assistant_msg = doc.get("answer") or ""
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
                return messages
            except Exception as e:
                logger.warning(f"[Mongo] get_chat_context error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback and hasattr(fallback, 'get_chat_context'):
            return fallback.get_chat_context(user_id, chat_id, limit)
        return []

    def get_user_history(self, user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieves historical query records for a specific authenticated user.
        Strictly user-scoped to prevent unauthorized cross-user access.
        """
        if not user_id:
            return []
        if getattr(self, '_available', False) and self._db is not None:
            try:
                cursor = self._db.query_history.find({"user_id": user_id}).sort("created_at", DESCENDING).skip(skip).limit(limit)
                results = []
                for doc in cursor:
                    doc["id"] = str(doc["_id"])
                    doc["history_id"] = str(doc["_id"])
                    doc["prompt"] = doc.get("original_query", "")
                    doc["sources"] = doc.get("citations", [])
                    results.append(doc)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"[Mongo] get_user_history error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.get_user_history(user_id, limit, skip)
        return []

    def get_user_history_item(self, user_id: str, history_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single user history record after verifying ownership.
        """
        if not user_id or not history_id:
            return None
        if getattr(self, '_available', False) and self._db is not None:
            try:
                doc = self._db.query_history.find_one({"_id": history_id, "user_id": user_id})
                if doc:
                    doc["id"] = str(doc["_id"])
                    doc["history_id"] = str(doc["_id"])
                    doc["prompt"] = doc.get("original_query", "")
                    doc["sources"] = doc.get("citations", [])
                    return doc
            except Exception as e:
                logger.warning(f"[Mongo] get_user_history_item error: {e}")

        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            return fallback.get_user_history_item(user_id, history_id)
        return None

    def delete_user_history_item(self, user_id: str, history_id: str) -> bool:
        """
        Deletes a single user history record after verifying ownership.
        """
        if not user_id or not history_id:
            return False
        deleted_mongo = False
        if getattr(self, '_available', False) and self._db is not None:
            try:
                res = self._db.query_history.delete_one({"_id": history_id, "user_id": user_id})
                deleted_mongo = res.deleted_count > 0
            except Exception as e:
                logger.warning(f"[Mongo] delete_user_history_item error: {e}")

        deleted_fallback = False
        fallback = getattr(self, 'fallback_store', None)
        if fallback:
            deleted_fallback = fallback.delete_user_history_item(user_id, history_id)

        return deleted_mongo or deleted_fallback

    def close(self) -> None:
        """Closes the MongoDB connection pool."""
        if self._client:
            try:
                self._client.close()
                self._available = False
                logger.info("MongoDB client connection closed.")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")
