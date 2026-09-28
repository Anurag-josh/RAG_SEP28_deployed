"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Conversation Context Resolver — determines whether a user query is a
         follow-up query dependent on previous chat context, reformulates it into a
         standalone query before cache lookup, or detects ambiguous queries.
Dependencies: groq, typing, logging, config.config
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from groq import Groq

from config.config import GROQ_API_KEY, GROQ_MODEL_NAME

logger = logging.getLogger("cache.context_resolver")

PRONOUN_PATTERN = re.compile(
    r'\b(it|its|that|this|they|them|these|those|above|previous|former|latter)\b',
    re.IGNORECASE
)

FOLLOW_UP_PHRASES = [
    "tell me more", "explain more", "more details", "elaborate", "what else",
    "advantages", "disadvantages", "pros and cons", "time complexity",
    "space complexity", "how does it work", "how is it different", "give an example"
]

GENERIC_ACTION_WORDS = {
    "what", "is", "are", "how", "does", "do", "explain", "define", "describe",
    "tell", "me", "about", "the", "a", "an", "and", "or", "in", "of", "to", "for",
    "it", "its", "this", "that", "they", "them", "these", "those", "work", "works", "working",
    "advantage", "advantages", "disadvantage", "disadvantages", "pro", "pros",
    "con", "cons", "detail", "details", "elaborate", "feature", "features",
    "difference", "differences", "example", "examples", "use", "uses", "used",
    "benefit", "benefits", "more", "much", "many", "application", "applications",
    "function", "functions", "functioning", "code", "implementation", "why",
    "when", "where", "who", "which", "should", "would", "could", "can", "may",
    "might", "must", "shall", "will", "i", "you", "he", "she", "we"
}


def is_query_self_contained(query: str) -> bool:
    """
    Checks if a user query explicitly defines a concrete topic subject within itself,
    making it standalone even if it contains pronouns in follow-up clauses.
    E.g., "What is binary search and how does it work?" -> True
          "How does it work?" -> False
    """
    clean = query.strip()
    if not clean:
        return False

    pronoun_match = PRONOUN_PATTERN.search(clean)
    if not pronoun_match:
        return True

    prefix = clean[:pronoun_match.start()].strip()
    prefix_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', prefix)]
    substantive_prefix = [w for w in prefix_words if w not in GENERIC_ACTION_WORDS]

    return len(substantive_prefix) >= 1


def resolve_context_query(
    query: str,
    chat_history: List[Dict[str, str]],
    groq_api_key: Optional[str] = None
) -> Tuple[str, str]:
    """
    Resolves whether the input query depends on previous conversation history.

    Args:
        query: Raw user query.
        chat_history: Recent conversation messages (list of dicts with role and content/text).
        groq_api_key: Optional Groq API key override.

    Returns:
        Tuple[status, resolved_query] where status is "INDEPENDENT", "REWRITTEN", or "AMBIGUOUS".
    """
    clean_query = query.strip()
    if not clean_query:
        return "INDEPENDENT", clean_query

    # Self-contained queries that define their own subject (e.g. "What is binary search and how does it work?")
    if is_query_self_contained(clean_query):
        logger.info(f"[CONTEXT] Self-contained query with embedded subject - INDEPENDENT")
        return "INDEPENDENT", clean_query

    # If no chat history is present and query is not self-contained
    if not chat_history:
        if PRONOUN_PATTERN.search(clean_query) or any(p in clean_query.lower() for p in ["tell me more", "elaborate"]):
            logger.info(f"[CONTEXT] Query relies on context but history is empty -> AMBIGUOUS")
            return "AMBIGUOUS", clean_query
        logger.info(f"[CONTEXT] Independent query - no reformulation required")
        return "INDEPENDENT", clean_query

    # Format history slice for context window (last 10 messages / ~5 turns)
    history_slice = chat_history[-10:]
    history_text = ""
    user_turns = 0
    for msg in history_slice:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content") or msg.get("text") or ""
        if content:
            history_text += f"{role}: {content}\n"
            if role == "User":
                user_turns += 1

    if not history_text.strip() or user_turns == 0:
        if PRONOUN_PATTERN.search(clean_query) or any(p in clean_query.lower() for p in ["tell me more", "elaborate"]):
            logger.info(f"[CONTEXT] Unable to resolve reference")
            return "AMBIGUOUS", clean_query
        logger.info(f"[CONTEXT] Independent query - no reformulation required")
        return "INDEPENDENT", clean_query

    api_key = groq_api_key or GROQ_API_KEY
    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = (
                "You are an expert conversation context resolver for a technical search/RAG system.\n"
                "Analyze the conversation history and the new user question.\n"
                "Your job is to determine if the new user question depends on the conversation history (e.g. uses pronouns like 'it', 'its', 'this', 'that', 'they', 'them', or follow-up requests like 'tell me more', 'what are its advantages', 'how is it different').\n\n"
                "RULES:\n"
                "1. If the new question is a standalone, independent topic (e.g. User asks about 'MongoDB' after discussing 'binary search', or asks 'What is binary search and how does it work?'), return ONLY: INDEPENDENT\n"
                "2. If the new question relies on previous context AND the previous context makes the reference clear, rewrite the question into a complete, standalone question. Return ONLY: REWRITE: <standalone question>\n"
                "3. If the new question relies on previous context (e.g. 'Tell me more about it', 'What are its advantages') BUT the conversation history is unclear, ambiguous, or lacks the topic referenced, return ONLY: AMBIGUOUS\n\n"
                f"Conversation History:\n{history_text}\n"
                f"New User Question: {clean_query}\n\n"
                "Decision:"
            )

            response = client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()
            if result.startswith("INDEPENDENT"):
                logger.info(f"[CONTEXT] Independent query - no reformulation required")
                return "INDEPENDENT", clean_query
            elif result.startswith("AMBIGUOUS"):
                logger.info(f"[CONTEXT] Unable to resolve reference")
                return "AMBIGUOUS", clean_query
            elif result.startswith("REWRITE:"):
                rewritten = result.split("REWRITE:", 1)[1].strip().strip('"')
                if rewritten:
                    logger.info(f"[CONTEXT] Follow-up detected")
                    logger.info(f"[CONTEXT] Resolved query: '{rewritten}'")
                    return "REWRITTEN", rewritten

        except Exception as e:
            logger.warning(f"[CONTEXT] Groq resolution failed: {e}. Falling back to heuristic resolution.")

    # Heuristic Fallback
    has_pronoun = bool(PRONOUN_PATTERN.search(clean_query))
    has_phrase = any(phrase in clean_query.lower() for phrase in FOLLOW_UP_PHRASES)

    if has_pronoun or has_phrase:
        user_msgs = [msg.get("content") or msg.get("text") or "" for msg in reversed(history_slice) if msg.get("role") == "user"]
        if user_msgs:
            topic = user_msgs[0].strip("?.!")
            found_topic = False
            # Search backwards for known topics or explicit noun-phrases
            for u_msg in user_msgs:
                u_lower = u_msg.lower()
                for known in ["binary search", "tcp", "http", "https", "mongodb", "redis", "sqlite", "bubble sort", "quick sort", "heap sort", "counting sort", "radix sort", "merge sort", "linear search"]:
                    if known in u_lower:
                        topic = known
                        found_topic = True
                        break
                if found_topic:
                    break

            if not found_topic:
                for u_msg in user_msgs:
                    clean_u = u_msg.strip("?.!")
                    if clean_u and not (PRONOUN_PATTERN.search(clean_u) or any(p in clean_u.lower() for p in FOLLOW_UP_PHRASES)):
                        topic = clean_u
                        break

            rewritten = clean_query
            if has_pronoun:
                rewritten = PRONOUN_PATTERN.sub(topic, rewritten)
            else:
                rewritten = f"{clean_query} {topic}"
            logger.info(f"[CONTEXT] Follow-up detected")
            logger.info(f"[CONTEXT] Resolved query: '{rewritten}'")
            return "REWRITTEN", rewritten
        else:
            logger.info(f"[CONTEXT] Unable to resolve reference")
            return "AMBIGUOUS", clean_query

    logger.info(f"[CONTEXT] Independent query - no reformulation required")
    return "INDEPENDENT", clean_query
