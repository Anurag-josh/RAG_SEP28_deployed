"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: SQLite persistent cache store — schema creation, CRUD operations for
         queries, sources, pages, chunks, and embedding vectors.
         Uses WAL mode for concurrency safety, parameterized queries throughout,
         and stores embedding vectors as float32 BLOBs for space efficiency.
Dependencies: sqlite3, numpy, json, datetime, typing, config.config
"""

import json
import time
import sqlite3
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.config import (
    CACHE_DB_PATH,
    CACHE_EMBEDDING_MODEL,
    CACHE_EMBEDDING_DIMENSION,
    CACHE_EMBEDDING_VERSION,
    PAGE_CACHE_TTL,
    SEMANTIC_CACHE_TTL,
    EXACT_ANSWER_TTL,
)

logger = logging.getLogger("cache.sqlite_store")


# ─── Helpers ─────────────────────────────────────────────────────────────────

class _ClosingConnection(sqlite3.Connection):
    """Connection whose context manager also closes the underlying handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()

def _utcnow_iso() -> str:
    """Returns current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(ttl_seconds: int) -> str:
    """Returns UTC expiry time as ISO-8601 string."""
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


def _is_expired(expires_at_iso: Optional[str]) -> bool:
    """Returns True if the expires_at timestamp is in the past."""
    if not expires_at_iso:
        return True
    try:
        exp = datetime.fromisoformat(expires_at_iso)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) > exp
    except Exception:
        return True


def _vec_to_blob(vector: List[float]) -> bytes:
    """Converts a float list to a compact float32 BLOB for SQLite storage."""
    return np.array(vector, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> List[float]:
    """Converts a stored float32 BLOB back to a Python list of floats."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


# ─── SQLiteStore ─────────────────────────────────────────────────────────────

class SQLiteStore:
    """
    Manages all SQLite database operations for the semantic cache system.
    Uses WAL journal mode for concurrent read safety.
    All queries are parameterized. All write failures are logged without raising.
    """

    # ── Initialization ────────────────────────────────────────────────────────

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Opens (or creates) the SQLite database and ensures schema is up to date.

        Args:
            db_path: Optional override for the database file path.
        """
        self.db_path = Path(db_path) if db_path else CACHE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"SQLiteStore initializing at: {self.db_path}")
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Creates a new SQLite connection with WAL mode and row factory."""
        conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10,
            factory=_ClosingConnection
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        """Creates all tables and indexes if they do not already exist."""
        ddl = """
        -- Cached query results
        CREATE TABLE IF NOT EXISTS queries (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash        TEXT    NOT NULL UNIQUE,
            normalized_query  TEXT    NOT NULL,
            original_query    TEXT,
            intent            TEXT,
            scope             TEXT,
            requirements_json TEXT,
            answer            TEXT,
            source_ids_json   TEXT,
            chunk_ids_json    TEXT,
            embedding_model   TEXT    NOT NULL,
            embedding_dim     INTEGER NOT NULL,
            embedding_version TEXT    NOT NULL,
            query_embedding   BLOB,
            created_at        TEXT    NOT NULL,
            expires_at        TEXT    NOT NULL,
            research_depth    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_queries_hash    ON queries(query_hash);
        CREATE INDEX IF NOT EXISTS idx_queries_expires ON queries(expires_at);

        -- Source URL metadata
        CREATE TABLE IF NOT EXISTS sources (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            url           TEXT    NOT NULL UNIQUE,
            title         TEXT,
            domain        TEXT,
            content_hash  TEXT,
            created_at    TEXT    NOT NULL,
            expires_at    TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sources_url     ON sources(url);
        CREATE INDEX IF NOT EXISTS idx_sources_expires ON sources(expires_at);

        -- Scraped page content (Markdown)
        CREATE TABLE IF NOT EXISTS pages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id     INTEGER NOT NULL,
            markdown      TEXT,
            content_hash  TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            FOREIGN KEY(source_id) REFERENCES sources(id)
        );
        CREATE INDEX IF NOT EXISTS idx_pages_source ON pages(source_id);
        CREATE INDEX IF NOT EXISTS idx_pages_hash   ON pages(content_hash);

        -- Text chunks derived from pages
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id     INTEGER NOT NULL,
            chunk_hash  TEXT    NOT NULL UNIQUE,
            chunk_text  TEXT    NOT NULL,
            chunk_index INTEGER,
            url         TEXT,
            title       TEXT,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY(page_id) REFERENCES pages(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);

        -- Embedding vectors stored as float32 BLOBs
        CREATE TABLE IF NOT EXISTS embeddings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id      INTEGER NOT NULL,
            model_name    TEXT    NOT NULL,
            dimension     INTEGER NOT NULL,
            model_version TEXT    NOT NULL,
            vector        BLOB    NOT NULL,
            created_at    TEXT    NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id),
            UNIQUE(chunk_id, model_name, model_version)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);

        -- Users table for local auth fallback
        CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

        -- User query history table for local history fallback
        CREATE TABLE IF NOT EXISTS query_history (
            id               TEXT PRIMARY KEY,
            user_id          TEXT NOT NULL,
            query_hash       TEXT NOT NULL,
            original_query   TEXT NOT NULL,
            normalized_query TEXT,
            intent           TEXT,
            scope            TEXT,
            answer           TEXT,
            citations_json   TEXT,
            sources_analyzed INTEGER,
            sources_used     INTEGER,
            cache_status     TEXT,
            is_cacheable     INTEGER,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            metadata_json    TEXT,
            UNIQUE(user_id, query_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_history_user_id    ON query_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_history_created_at ON query_history(created_at);
        """
        try:
            with self._get_conn() as conn:
                conn.executescript(ddl)
                cols = [row[1] for row in conn.execute("PRAGMA table_info(queries)").fetchall()]
                if "query_embedding" not in cols:
                    conn.execute("ALTER TABLE queries ADD COLUMN query_embedding BLOB")
                if "research_depth" not in cols:
                    conn.execute("ALTER TABLE queries ADD COLUMN research_depth TEXT")
                h_cols = [row[1] for row in conn.execute("PRAGMA table_info(query_history)").fetchall()]
                if "chat_id" not in h_cols:
                    conn.execute("ALTER TABLE query_history ADD COLUMN chat_id TEXT")
                if "resolved_query" not in h_cols:
                    conn.execute("ALTER TABLE query_history ADD COLUMN resolved_query TEXT")
                if "research_depth" not in h_cols:
                    conn.execute("ALTER TABLE query_history ADD COLUMN research_depth TEXT")
            logger.info("SQLite schema initialized successfully.")
        except Exception as e:
            logger.error(f"SQLite schema initialization failed: {e}")
            raise


    # ── Sources ───────────────────────────────────────────────────────────────

    def upsert_source(self, url: str, title: str, domain: str,
                      content_hash: str, ttl: int = PAGE_CACHE_TTL) -> Optional[int]:
        """
        Insert or update a source URL record. Returns the source row ID.

        Args:
            url: Source page URL (unique key).
            title: Page title.
            domain: Extracted hostname/domain.
            content_hash: SHA-256 of page content for change detection.
            ttl: Time-to-live in seconds.

        Returns:
            int: Source row ID, or None on error.
        """
        now = _utcnow_iso()
        expires = _expires_iso(ttl)
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO sources (url, title, domain, content_hash, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        title        = excluded.title,
                        domain       = excluded.domain,
                        content_hash = excluded.content_hash,
                        expires_at   = excluded.expires_at
                """, (url, title, domain, content_hash, now, expires))
                row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
                return row["id"] if row else None
        except Exception as e:
            logger.error(f"SQLite upsert_source failed for {url}: {e}")
            return None

    def get_source_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a source record by URL. Returns None if not found or expired.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM sources WHERE url = ?", (url,)
                ).fetchone()
            if row and not _is_expired(row["expires_at"]):
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"SQLite get_source_by_url failed: {e}")
            return None

    def are_sources_valid(self, source_ids: List[int]) -> bool:
        """Return True only when every source exists and has not expired."""
        if not source_ids:
            return False
        try:
            placeholders = ",".join("?" * len(source_ids))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"SELECT id, expires_at FROM sources WHERE id IN ({placeholders})",
                    source_ids,
                ).fetchall()
            if len(rows) != len(set(source_ids)):
                return False
            return all(not _is_expired(row["expires_at"]) for row in rows)
        except Exception as e:
            logger.error(f"SQLite are_sources_valid failed: {e}")
            return False

    # ── Pages ─────────────────────────────────────────────────────────────────

    def upsert_page(self, source_id: int, markdown: str, content_hash: str) -> Optional[int]:
        """
        Insert or update a page record. Returns the page row ID.
        """
        now = _utcnow_iso()
        try:
            with self._get_conn() as conn:
                # Check if a page with this source_id already exists
                existing = conn.execute(
                    "SELECT id, content_hash FROM pages WHERE source_id = ?", (source_id,)
                ).fetchone()
                if existing:
                    if existing["content_hash"] == content_hash:
                        # Content unchanged — return existing page ID
                        return existing["id"]
                    else:
                        # Content changed — update in place
                        conn.execute("""
                            UPDATE pages SET markdown = ?, content_hash = ?, created_at = ?
                            WHERE id = ?
                        """, (markdown, content_hash, now, existing["id"]))
                        return existing["id"]
                else:
                    cursor = conn.execute("""
                        INSERT INTO pages (source_id, markdown, content_hash, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (source_id, markdown, content_hash, now))
                    return cursor.lastrowid
        except Exception as e:
            logger.error(f"SQLite upsert_page failed: {e}")
            return None

    def get_page_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Returns a page record matching the given content hash, or None.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM pages WHERE content_hash = ?", (content_hash,)
                ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"SQLite get_page_by_content_hash failed: {e}")
            return None

    def get_page_by_source_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        """Returns the latest page record for a given source_id."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM pages WHERE source_id = ? ORDER BY id DESC LIMIT 1",
                    (source_id,)
                ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"SQLite get_page_by_source_id failed: {e}")
            return None

    # ── Chunks ────────────────────────────────────────────────────────────────

    def upsert_chunk(self, page_id: int, chunk_hash: str, chunk_text: str,
                     chunk_index: int, url: str = "", title: str = "") -> Optional[int]:
        """
        Insert or retrieve a chunk record. Returns the chunk row ID.
        chunk_hash is unique — duplicate inserts are ignored.
        """
        now = _utcnow_iso()
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO chunks
                        (page_id, chunk_hash, chunk_text, chunk_index, url, title, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (page_id, chunk_hash, chunk_text, chunk_index, url, title, now))
                row = conn.execute(
                    "SELECT id FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
                ).fetchone()
                return row["id"] if row else None
        except Exception as e:
            logger.error(f"SQLite upsert_chunk failed: {e}")
            return None

    def get_chunks_by_page_id(self, page_id: int) -> List[Dict[str, Any]]:
        """Returns all chunk records for a given page_id."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM chunks WHERE page_id = ? ORDER BY chunk_index ASC",
                    (page_id,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"SQLite get_chunks_by_page_id failed: {e}")
            return []

    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Returns chunk records matching the given list of chunk IDs."""
        if not chunk_ids:
            return []
        try:
            placeholders = ",".join("?" * len(chunk_ids))
            with self._get_conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"SQLite get_chunks_by_ids failed: {e}")
            return []

    def get_chunk_by_hash(self, chunk_hash: str) -> Optional[Dict[str, Any]]:
        """Returns a chunk record by its hash, or None."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
                ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"SQLite get_chunk_by_hash failed: {e}")
            return None

    # ── Embeddings ────────────────────────────────────────────────────────────

    def upsert_embedding(self, chunk_id: int, model_name: str, dimension: int,
                         model_version: str, vector: List[float]) -> bool:
        """
        Stores an embedding vector as a float32 BLOB. Returns True on success.
        Duplicate (chunk_id, model_name, model_version) is silently ignored.
        """
        now = _utcnow_iso()
        blob = _vec_to_blob(vector)
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO embeddings
                        (chunk_id, model_name, dimension, model_version, vector, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, model_name, dimension, model_version, blob, now))
            return True
        except Exception as e:
            logger.error(f"SQLite upsert_embedding failed: {e}")
            return False

    def get_embedding(self, chunk_id: int, model_name: str,
                      model_version: str) -> Optional[List[float]]:
        """
        Returns the embedding vector for a chunk + model combo, or None.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT vector FROM embeddings
                    WHERE chunk_id = ? AND model_name = ? AND model_version = ?
                """, (chunk_id, model_name, model_version)).fetchone()
            return _blob_to_vec(row["vector"]) if row else None
        except Exception as e:
            logger.error(f"SQLite get_embedding failed: {e}")
            return None

    def get_embeddings_by_chunk_ids(self, chunk_ids: List[int],
                                    model_name: str,
                                    model_version: str) -> Dict[int, List[float]]:
        """
        Returns a dict mapping chunk_id → embedding vector for the given chunk IDs.
        """
        if not chunk_ids:
            return {}
        try:
            placeholders = ",".join("?" * len(chunk_ids))
            params = chunk_ids + [model_name, model_version]
            with self._get_conn() as conn:
                rows = conn.execute(f"""
                    SELECT chunk_id, vector FROM embeddings
                    WHERE chunk_id IN ({placeholders})
                      AND model_name = ? AND model_version = ?
                """, params).fetchall()
            return {r["chunk_id"]: _blob_to_vec(r["vector"]) for r in rows}
        except Exception as e:
            logger.error(f"SQLite get_embeddings_by_chunk_ids failed: {e}")
            return {}

    # ── Queries ───────────────────────────────────────────────────────────────

    def save_query(self, query_hash: str, normalized_query: str, original_query: str,
                   intent: str, scope: str, requirements: Dict[str, Any], answer: str,
                   source_ids: List[int], chunk_ids: List[int],
                   query_embedding: Optional[List[float]] = None,
                   ttl: int = SEMANTIC_CACHE_TTL,
                   research_depth: str = "quick") -> bool:
        """
        Saves a completed query result to the queries table. Returns True on success.
        """
        now = _utcnow_iso()
        expires = _expires_iso(ttl)
        q_blob = _vec_to_blob(query_embedding) if query_embedding is not None else None
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO queries
                        (query_hash, normalized_query, original_query, intent, scope,
                         requirements_json, answer, source_ids_json, chunk_ids_json,
                         embedding_model, embedding_dim, embedding_version, query_embedding,
                         created_at, expires_at, research_depth)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_hash) DO UPDATE SET
                        original_query    = excluded.original_query,
                        intent            = excluded.intent,
                        scope             = excluded.scope,
                        requirements_json = excluded.requirements_json,
                        answer            = excluded.answer,
                        source_ids_json   = excluded.source_ids_json,
                        chunk_ids_json    = excluded.chunk_ids_json,
                        query_embedding   = COALESCE(excluded.query_embedding, queries.query_embedding),
                        expires_at        = excluded.expires_at,
                        research_depth    = excluded.research_depth
                """, (
                    query_hash, normalized_query, original_query, intent, scope,
                    json.dumps(requirements), answer,
                    json.dumps(source_ids), json.dumps(chunk_ids),
                    CACHE_EMBEDDING_MODEL, CACHE_EMBEDDING_DIMENSION, CACHE_EMBEDDING_VERSION,
                    q_blob, now, expires, research_depth
                ))
            return True
        except Exception as e:
            logger.error(f"SQLite save_query failed: {e}")
            return False

    def get_query_embedding(self, query_hash: str) -> Optional[List[float]]:
        """
        Retrieves the stored query embedding vector for a query hash from SQLite, or None.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT query_embedding FROM queries WHERE query_hash = ?", (query_hash,)
                ).fetchone()
            if row and row["query_embedding"]:
                return _blob_to_vec(row["query_embedding"])
            return None
        except Exception as e:
            logger.error(f"SQLite get_query_embedding failed: {e}")
            return None

    def get_query(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a non-expired query record by hash. Returns None if not found or expired.
        """
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM queries WHERE query_hash = ?", (query_hash,)
                ).fetchone()
            if not row:
                return None
            record = dict(row)
            if _is_expired(record.get("expires_at")):
                return None
            # Deserialize JSON fields
            record["requirements_json"] = json.loads(record["requirements_json"] or "{}")
            record["source_ids_json"] = json.loads(record["source_ids_json"] or "[]")
            record["chunk_ids_json"] = json.loads(record["chunk_ids_json"] or "[]")
            return record
        except Exception as e:
            logger.error(f"SQLite get_query failed: {e}")
            return None

    def get_all_query_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns all non-expired query records (without answer text, for semantic scanning).
        Used by the semantic cache to find candidate entries.
        """
        now = _utcnow_iso()
        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT query_hash, normalized_query, intent, scope, requirements_json,
                           source_ids_json, chunk_ids_json,
                           embedding_model, embedding_dim, embedding_version, expires_at, research_depth
                    FROM queries
                    WHERE expires_at > ?
                    ORDER BY id DESC
                """, (now,)).fetchall()
            results = []
            for row in rows:
                record = dict(row)
                record["requirements_json"] = json.loads(record["requirements_json"] or "{}")
                record["source_ids_json"] = json.loads(record["source_ids_json"] or "[]")
                record["chunk_ids_json"] = json.loads(record["chunk_ids_json"] or "[]")
                record["research_depth"] = record.get("research_depth") or "quick"
                results.append(record)
            return results
        except Exception as e:
            logger.error(f"SQLite get_all_query_metadata failed: {e}")
            return []

    def get_query_answer(self, query_hash: str) -> Optional[str]:
        """Returns only the answer text for a non-expired query hash."""
        record = self.get_query(query_hash)
        return record["answer"] if record else None

    # ── Invalidation ─────────────────────────────────────────────────────────

    def invalidate_query(self, query_hash: str) -> bool:
        """Deletes a query record by hash."""
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM queries WHERE query_hash = ?", (query_hash,))
            return True
        except Exception as e:
            logger.error(f"SQLite invalidate_query failed: {e}")
            return False

    def invalidate_source(self, url: str) -> bool:
        """Marks a source as expired by setting expires_at to now."""
        now = _utcnow_iso()
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE sources SET expires_at = ? WHERE url = ?", (now, url)
                )
            return True
        except Exception as e:
            logger.error(f"SQLite invalidate_source failed: {e}")
            return False

    def clear_all_cache(self) -> bool:
        """Deletes all data from all cache tables. Preserves schema."""
        try:
            with self._get_conn() as conn:
                conn.executescript("""
                    DELETE FROM embeddings;
                    DELETE FROM chunks;
                    DELETE FROM pages;
                    DELETE FROM sources;
                    DELETE FROM queries;
                    DELETE FROM query_history;
                """)
            logger.info("SQLite cache cleared completely.")
            return True
        except Exception as e:
            logger.error(f"SQLite clear_all_cache failed: {e}")
            return False

    # ── User Management (Local Fallback) ──────────────────────────────────────

    def create_user(self, user_id: str, username: str, email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        now = _utcnow_iso()
        username_clean = username.lower().strip()
        email_clean = email.lower().strip()
        doc = {
            "user_id": user_id,
            "username": username_clean,
            "email": email_clean,
            "password_hash": password_hash,
            "created_at": now,
            "updated_at": now,
            "preferences": {}
        }
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, username, email, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, username_clean, email_clean, password_hash, now, now)
                )
            logger.info(f"[SQLite] User registered: {username_clean} ({email_clean})")
            return doc
        except sqlite3.IntegrityError as e:
            logger.warning(f"[SQLite] Create user duplicate key: {e}")
            return None
        except Exception as e:
            logger.error(f"[SQLite] create_user error: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
                if row:
                    return dict(row)
            return None
        except Exception as e:
            logger.warning(f"[SQLite] get_user_by_email error: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT * FROM users WHERE username = ?", (username.lower().strip(),)).fetchone()
                if row:
                    return dict(row)
            return None
        except Exception as e:
            logger.warning(f"[SQLite] get_user_by_username error: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if row:
                    return dict(row)
            return None
        except Exception as e:
            logger.warning(f"[SQLite] get_user_by_id error: {e}")
            return None

    # ── User Query History (Local Fallback) ───────────────────────────────────

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
        if not user_id:
            return False
        now = _utcnow_iso()
        history_id = f"hist_{user_id}_{query_hash[:16]}"
        citations_json = json.dumps(citations or [])
        metadata_json = json.dumps(metadata or {})
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO query_history (
                        id, user_id, query_hash, original_query, normalized_query, intent, scope,
                        answer, citations_json, sources_analyzed, sources_used, cache_status,
                        is_cacheable, created_at, updated_at, metadata_json, chat_id, resolved_query, research_depth
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, query_hash) DO UPDATE SET
                        answer=excluded.answer,
                        citations_json=excluded.citations_json,
                        sources_analyzed=excluded.sources_analyzed,
                        sources_used=excluded.sources_used,
                        cache_status=excluded.cache_status,
                        updated_at=excluded.updated_at,
                        metadata_json=excluded.metadata_json,
                        chat_id=excluded.chat_id,
                        resolved_query=excluded.resolved_query,
                        research_depth=excluded.research_depth
                    """,
                    (
                        history_id, user_id, query_hash, original_query, normalized_query, intent, scope,
                        answer, citations_json, sources_analyzed, sources_used, cache_status,
                        1 if is_cacheable else 0, now, now, metadata_json, chat_id, resolved_query or original_query, research_depth
                    )
                )
            logger.debug(f"[SQLite] Saved history for user {user_id} (chat_id={chat_id}, status={cache_status})")
            return True
        except Exception as e:
            logger.warning(f"[SQLite] save_query_history error: {e}")
            return False

    def get_chat_context(self, user_id: str, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        if not user_id or not chat_id or str(chat_id).strip() in ("", "None", "null"):
            return []
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT original_query, answer FROM query_history WHERE user_id = ? AND chat_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, chat_id, limit)
                ).fetchall()
                messages = []
                for row in reversed(rows):
                    user_msg = row["original_query"] or ""
                    assistant_msg = row["answer"] or ""
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
                return messages
        except Exception as e:
            logger.warning(f"[SQLite] get_chat_context error: {e}")
            return []

    def get_user_history(self, user_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        if not user_id:
            return []
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM query_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (user_id, limit, skip)
                ).fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    item["_id"] = item["id"]
                    item["id"] = item["id"]
                    item["history_id"] = item["id"]
                    item["prompt"] = item.get("original_query", "")
                    try:
                        item["citations"] = json.loads(item.get("citations_json") or "[]")
                    except Exception:
                        item["citations"] = []
                    item["sources"] = item["citations"]
                    results.append(item)
                return results
        except Exception as e:
            logger.warning(f"[SQLite] get_user_history error: {e}")
            return []

    def get_user_history_item(self, user_id: str, history_id: str) -> Optional[Dict[str, Any]]:
        if not user_id or not history_id:
            return None
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM query_history WHERE id = ? AND user_id = ?",
                    (history_id, user_id)
                ).fetchone()
                if row:
                    item = dict(row)
                    item["_id"] = item["id"]
                    item["id"] = item["id"]
                    item["history_id"] = item["id"]
                    item["prompt"] = item.get("original_query", "")
                    try:
                        item["citations"] = json.loads(item.get("citations_json") or "[]")
                    except Exception:
                        item["citations"] = []
                    item["sources"] = item["citations"]
                    return item
            return None
        except Exception as e:
            logger.warning(f"[SQLite] get_user_history_item error: {e}")
            return None

    def delete_user_history_item(self, user_id: str, history_id: str) -> bool:
        if not user_id or not history_id:
            return False
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM query_history WHERE id = ? AND user_id = ?",
                    (history_id, user_id)
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"[SQLite] delete_user_history_item error: {e}")
            return False

    # ── Migration from JSON cache ─────────────────────────────────────────────

    def migrate_json_cache(self, json_cache_path: Path) -> int:
        """
        Imports entries from the legacy scraped_pages.json into the SQLite pages table.
        After import, renames the JSON file to .migrated to prevent re-import.

        Args:
            json_cache_path: Path to the existing scraped_pages.json file.

        Returns:
            int: Number of entries migrated.
        """
        import hashlib as _hl
        from urllib.parse import urlparse
        if not json_cache_path.exists():
            return 0
        try:
            import json as _json
            with open(json_cache_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            count = 0
            for url, markdown in data.items():
                if not url or not markdown:
                    continue
                parsed = urlparse(url)
                domain = (parsed.hostname or "").replace("www.", "")
                content_hash = _hl.sha256(markdown.encode("utf-8")).hexdigest()
                src_id = self.upsert_source(url, domain, domain, content_hash)
                if src_id:
                    self.upsert_page(src_id, markdown, content_hash)
                    count += 1
            migrated_path = json_cache_path.with_suffix(".json.migrated")
            json_cache_path.rename(migrated_path)
            logger.info(f"Migrated {count} entries from JSON cache → SQLite. Renamed to {migrated_path.name}")
            return count
        except Exception as e:
            logger.error(f"JSON cache migration failed: {e}")
            return 0


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".db")
    store = SQLiteStore(db_path=tmp)
    print("Schema created OK")

    src_id = store.upsert_source("https://example.com/page", "Example Page", "example.com", "abc123")
    print(f"Source upserted, ID={src_id}")

    page_id = store.upsert_page(src_id, "# Hello World\nContent here.", "abc123")
    print(f"Page upserted, ID={page_id}")

    chunk_id = store.upsert_chunk(page_id, "chk001", "Hello World content.", 1, "https://example.com/page", "Example")
    print(f"Chunk upserted, ID={chunk_id}")

    vec = [0.1] * 384
    ok = store.upsert_embedding(chunk_id, "all-MiniLM-L6-v2", 384, "1.0", vec)
    print(f"Embedding stored: {ok}")

    retrieved = store.get_embedding(chunk_id, "all-MiniLM-L6-v2", "1.0")
    print(f"Embedding retrieved, length={len(retrieved)}, first={retrieved[0]:.3f}")

    ok = store.save_query(
        "hash001", "explain stack", "Explain stack", "concept_explanation",
        "stack", {"easy_language": True}, "A stack is LIFO...", [src_id], [chunk_id]
    )
    print(f"Query saved: {ok}")

    record = store.get_query("hash001")
    print(f"Query retrieved: {record['normalized_query']!r} | intent={record['intent']}")

    cleared = store.clear_all_cache()
    print(f"Cache cleared: {cleared}")

    os.unlink(tmp)
    print("Self-test complete.")
