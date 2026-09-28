"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Single clean-reset script for local development and testing.
         Clears MongoDB test collections, SQLite cache tables & history,
         Redis cache keys, persisted ChromaDB temp data, and test artifacts.
"""

import os
import sys
import shutil
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import CHROMA_PERSIST_DIR, CACHE_DB_PATH
from cache.cache_manager import CacheManager


def perform_reset():
    print("=" * 70)
    print(" SOURCEIQ RAG — DEVELOPMENT & TEST DATA RESET SCRIPT ")
    print("=" * 70)
    print("WARNING: This script will clear all local development/test cache,")
    print("query histories, reusable answers, Redis cache keys, and test sessions.")
    print()
    print("Target Resources to Clear:")
    print(" 1. MongoDB: 'query_history', 'reusable_answers', test data")
    print(f" 2. SQLite: '{CACHE_DB_PATH}' queries, embeddings, pages, chunks, history")
    print(" 3. Redis: 'rag:*' cached answer & semantic index keys")
    print(" 4. ChromaDB: '.chroma_temp' directory (if exists)")
    print(" 5. Test Artifacts: 'cache/sessions.json', uploaded temp files")
    print("=" * 70)
    print()

    # Skip prompt if --force or --yes CLI flag passed
    if "--force" not in sys.argv and "--yes" not in sys.argv:
        try:
            confirm = input("Type RESET to continue: ").strip()
        except EOFError:
            confirm = ""
        if confirm != "RESET":
            print("\n[RESET ABORTED] Incorrect confirmation. No data was cleared.")
            sys.exit(1)

    print("\nExecuting clean reset...")

    # Initialize CacheManager
    cache_mgr = CacheManager()

    # 1. MongoDB Reset
    mongo_status = "Skipped / Unavailable"
    if cache_mgr.mongo.available and cache_mgr.mongo._db is not None:
        try:
            cache_mgr.mongo._db.query_history.delete_many({})
            cache_mgr.mongo._db.reusable_answers.delete_many({})
            mongo_status = "Cleared 'query_history' and 'reusable_answers'"
        except Exception as e:
            mongo_status = f"Failed: {e}"

    # 2. SQLite Reset
    sqlite_status = "Failed"
    try:
        ok = cache_mgr.sqlite.clear_all_cache()
        if ok:
            sqlite_status = "Cleared queries, query_embeddings, pages, chunks, embeddings, query_history"
        else:
            sqlite_status = "Partial clear"
    except Exception as e:
        sqlite_status = f"Failed: {e}"

    # 3. Redis Reset
    redis_status = "Skipped / Unavailable"
    if cache_mgr.redis.available:
        try:
            cache_mgr.redis.clear_all_cache()
            redis_status = "Cleared all 'rag:*' namespace keys"
        except Exception as e:
            redis_status = f"Failed: {e}"

    # Close CacheManager connections
    cache_mgr.close()

    # 4. ChromaDB Reset
    chroma_status = "Ephemeral (in-memory) / No persist dir"
    if CHROMA_PERSIST_DIR.exists():
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
            chroma_status = f"Deleted '{CHROMA_PERSIST_DIR.name}' directory"
        except Exception as e:
            chroma_status = f"Failed to delete Chroma directory: {e}"

    # 5. Test Artifacts Reset
    artifact_status = "Cleared sessions.json & temporary uploads"
    try:
        sessions_file = BASE_DIR / "cache" / "sessions.json"
        default_session = [
            {
                "id": "chat_default01",
                "title": "New chat",
                "active": True,
                "messages": []
            }
        ]
        with open(sessions_file, "w", encoding="utf-8") as f:
            json.dump(default_session, f, indent=2)

        uploads_dir = BASE_DIR / "uploads"
        if uploads_dir.exists():
            for item in uploads_dir.iterdir():
                if item.is_file() and not item.name.startswith("."):
                    try:
                        item.unlink()
                    except Exception:
                        pass
    except Exception as e:
        artifact_status = f"Failed: {e}"

    print()
    print("=" * 70)
    print(f"[RESET] MongoDB: {mongo_status}")
    print(f"[RESET] SQLite: {sqlite_status}")
    print(f"[RESET] Redis: {redis_status}")
    print(f"[RESET] ChromaDB: {chroma_status}")
    print(f"[RESET] Test artifacts: {artifact_status}")
    print("[RESET] COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    perform_reset()
