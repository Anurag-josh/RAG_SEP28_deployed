"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Query normalization and SHA-256 hashing for stable cache key generation.
         Normalizes raw user queries to a canonical form before cache lookup.
Dependencies: hashlib, re
"""

import re
import hashlib


# ─── Normalization ───────────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    """
    Normalizes a raw user query into a canonical, stable string for cache key generation.

    Operations:
        1. Strip leading/trailing whitespace & lowercase
        2. Expand common contractions (what's -> what is)
        3. Remove punctuation (?,.,!,;,:, quotes)
        4. Remove conversational preambles (e.g. "can you please", "tell me")
        5. Remove leading question starters ("what is", "what are", "explain", "define", etc.)
        6. Remove standalone articles ("a", "an", "the")
        7. Collapse whitespace into a single space

    Args:
        query (str): Raw user query string.

    Returns:
        str: Canonical normalized query string.
    """
    if not query:
        return ""

    # 1. Strip and lowercase
    q = query.strip().lower()

    # 2. Expand common contractions
    q = re.sub(r"\bwhat's\b", "what is", q)
    q = re.sub(r"\bhow's\b", "how is", q)
    q = re.sub(r"\bwhere's\b", "where is", q)
    q = re.sub(r"\bwho's\b", "who is", q)

    # 3. Remove punctuation (trailing, leading, and inner punctuation)
    q = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", q).strip()
    q = re.sub(r"[?!.,;:]+", " ", q)

    # 4. Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # 5. Remove conversational preambles
    q = re.sub(r"^(can you please|could you please|can you|could you|please|tell me|give me)\s+", "", q)

    # 6. Remove leading intent / question starters
    q = re.sub(
        r"^(what is|what are|explain|describe|define|definition of|tell me about|elaborate on|give an overview of|overview of)\s+",
        "",
        q
    )

    # 7. Remove articles (a, an, the) as standalone words
    q = re.sub(r"\b(a|an|the)\b", " ", q)

    # 8. Collapse multiple spaces again and strip
    norm = re.sub(r"\s+", " ", q).strip()

    # Fallback if normalization stripped everything
    if not norm:
        norm = re.sub(r"\s+", " ", query.strip().lower())

    return norm


def hash_query(normalized_query: str) -> str:
    """
    Produces a stable SHA-256 hex digest for the normalized query string.
    Used as the primary Redis and SQLite cache key.

    Args:
        normalized_query (str): Output of normalize_query().

    Returns:
        str: 64-character hex SHA-256 digest.
    """
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def normalize_and_hash(query: str) -> tuple:
    """
    Convenience function that returns both the normalized query and its hash.

    Args:
        query (str): Raw user query.

    Returns:
        tuple: (normalized_query, query_hash)
    """
    normalized = normalize_query(query)
    return normalized, hash_query(normalized)


def hash_content(content: str) -> str:
    """
    Produces a stable SHA-256 hex digest for arbitrary content (pages, chunks).
    Used for content-change detection to avoid reprocessing identical content.

    Args:
        content (str): Page markdown or chunk text.

    Returns:
        str: 64-character hex SHA-256 digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Explain stack data structure",
        "explain stack data structure?",
        "What is a stack data structure?",
        "  EXPLAIN STACK DATA STRUCTURE!  ",
        "The stack data structure",
        "a stack data structure",
    ]

    print("Query Normalizer — Self-Test")
    print("=" * 55)
    for raw in test_cases:
        norm, h = normalize_and_hash(raw)
        print(f"  Raw   : {raw!r}")
        print(f"  Norm  : {norm!r}")
        print(f"  Hash  : {h[:16]}...")
        print()
