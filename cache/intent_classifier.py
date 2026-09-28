"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Lightweight intent, scope, and requirement extractor for semantic cache decisions.
         Uses only keyword pattern matching — NO LLM calls are made.
         Classifies queries into intent labels, extracts normalized topic scope,
         and extracts structured requirement flags used for cache match validation.
Dependencies: re, typing
"""

import re
from typing import Dict, Any, Optional, List


# ─── Intent Patterns ─────────────────────────────────────────────────────────
# Ordered from most-specific to least-specific so first match wins.

_INTENT_PATTERNS: List[tuple] = [
    ("comparison",          [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference between\b", r"\bdifferences\b", r"\bdifference\b", r"\bcontrast\b", r"\bdiffer\b", r"\bdiffers\b", r"\bdiffering\b"]),
    ("debugging",           [r"\bdebug\b", r"\bfix\b", r"\berror\b", r"\bbug\b", r"\bissue\b", r"\bexception\b", r"\btraceback\b"]),
    ("code_generation",     [r"\bcode\b", r"\bprogram\b", r"\bimplement\b", r"\bwrite\b", r"\bsyntax\b", r"\bfunction\b", r"\bclass\b", r"\bscript\b"]),
    ("definition",          [r"\bdefinition of\b", r"\bdefine\b", r"\bmeaning of\b", r"\bwhat does .* mean\b"]),
    ("step_by_step",        [r"\bhow to\b", r"\bsteps\b", r"\bprocedure\b", r"\balgorithm for\b", r"\bprocess of\b"]),
    ("concept_explanation", [r"\bexplain\b", r"\bwhat is\b", r"\bwhat are\b", r"\bdescribe\b", r"\btell me about\b", r"\belaborate\b"]),
    ("example_request",     [r"\bexample\b", r"\bexamples\b", r"\binstance\b", r"\bdemonstrate\b", r"\bshow me\b", r"\billustrate\b"]),
    ("summary",             [r"\bsummarize\b", r"\bsummary\b", r"\boverview\b", r"\bbrief\b", r"\bin short\b", r"\bin brief\b"]),
    ("numerical_problem",   [r"\bcalculate\b", r"\bcompute\b", r"\bformula\b", r"\bnumerical\b", r"\bsolve\b", r"\bproof\b"]),
]

# Default intent when no pattern matches
_DEFAULT_INTENT = "concept_explanation"


# ─── Requirement Patterns ────────────────────────────────────────────────────

_REQUIREMENT_PATTERNS = {
    "easy_language":       [r"\beasy\b", r"\bsimple\b", r"\bsimplified\b", r"\bbeginners?\b", r"\bbasic\b", r"\blayman\b"],
    "example_required":    [r"\bwith example\b", r"\bfor example\b", r"\bwith an example\b", r"\bshow example\b", r"\bexample\b"],
    "step_by_step":        [r"\bstep by step\b", r"\bstep-by-step\b", r"\bsteps\b", r"\bprocedure\b"],
    "detailed":            [r"\bdetailed\b", r"\bin detail\b", r"\bthoroughly\b", r"\bcomprehensive\b", r"\bfull explanation\b"],
    "short_answer":        [r"\bshort\b", r"\bbrief\b", r"\bconcise\b", r"\bsummarize\b", r"\bin short\b"],
    "comparison_required": [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b", r"\bcontrast\b"],
    "numerical":           [r"\bnumerical\b", r"\bwith numbers\b", r"\bwith calculation\b", r"\bwith formula\b"],
    "advantages_disadvantages": [r"\badvantages?\b", r"\bdisadvantages?\b", r"\bpros\b", r"\bcons\b", r"\bbenefits?\b", r"\bdrawbacks?\b"],
}

_PROGRAMMING_LANGUAGES = {
    "java": [r"\bin java\b", r"\bjava code\b", r"\bjava program\b", r"\bjava implementation\b"],
    "python": [r"\bin python\b", r"\bpython code\b", r"\bpython program\b"],
    "c++": [r"\bin c\+\+\b", r"\bcpp\b", r"\bc plus plus\b"],
    "c": [r"\bin c\b", r"\bc code\b", r"\bc program\b"],
    "javascript": [r"\bin javascript\b", r"\bjs code\b", r"\bjavascript\b"],
    "typescript": [r"\bin typescript\b", r"\btypescript\b"],
    "go": [r"\bin golang\b", r"\bin go\b"],
    "rust": [r"\bin rust\b", r"\brust code\b"],
}


# ─── Stop words for scope extraction ────────────────────────────────────────

_INTENT_VERBS = {
    "explain", "what", "is", "are", "define", "definition", "describe", "how",
    "compare", "summarize", "summary", "overview", "brief", "implement", "write",
    "code", "program", "example", "show", "demonstrate", "tell", "elaborate",
    "calculate", "compute", "solve", "fix", "debug", "step", "procedure",
    "of", "me", "about", "give", "provide", "a", "an", "the", "in", "with",
    "for", "to", "by", "and", "or", "easy", "simple", "detailed", "short",
    "basic", "language", "beginner", "from", "between", "than", "versus", "vs"
}


# ─── Public API ─────────────────────────────────────────────────────────────

def classify_intent(query: str) -> str:
    """
    Classifies the query intent using keyword pattern matching.

    Args:
        query (str): Normalized query string (lowercase).

    Returns:
        str: Intent label from the defined intent set.
    """
    q = query.lower()
    for intent_label, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q):
                return intent_label
    return _DEFAULT_INTENT


def extract_scope(query: str) -> str:
    """
    Extracts a normalized topic/scope string from the query by removing
    intent verbs, requirement words, comparison connectors, and stop words.

    Args:
        query (str): Normalized query string (lowercase).

    Returns:
        str: Normalized scope (e.g., "http https", "stack data structure", "merge sort").
    """
    q = query.lower()

    # Remove programming language phrases first (keep the language as part of scope)
    q = re.sub(r"\bin (java|python|c\+\+|c |go|rust|javascript|typescript)\b", "", q)
    q = re.sub(r"\bin golang\b", "", q)

    # Remove requirement phrases
    q = re.sub(r"\bwith (an? )?example\b", "", q)
    q = re.sub(r"\bstep[- ]by[- ]step\b", "", q)
    q = re.sub(r"\bin detail(ed)?\b", "", q)
    q = re.sub(r"\bin (easy|simple|basic) language\b", "", q)
    q = re.sub(r"\bfor beginners?\b", "", q)
    q = re.sub(r"\bwith (advantages? and )?disadvantages?\b", "", q)
    q = re.sub(r"\bwith advantages?\b", "", q)

    # Remove comparison wrappers & connectors
    q = re.sub(r"\b(difference|differences|differs?|differing)\s+(between|from)?\b", "", q)
    q = re.sub(r"\b(between|versus|vs)\b", "", q)

    # Remove leading intent verbs and question words
    q = re.sub(
        r"^(explain|what is|what are|define|definition of|describe|how to|how|"
        r"compare|summarize|give me|tell me about|elaborate on|"
        r"write|implement|code for|program for|show|demonstrate|"
        r"calculate|compute|solve)\s+",
        "",
        q.strip()
    )

    # Remove trailing requirement words
    q = re.sub(r"\b(easy|simple|detailed|short|brief|basic|concise)\s*$", "", q)

    # Tokenize, filter stop words, rejoin
    tokens = q.split()
    filtered = [t for t in tokens if t not in _INTENT_VERBS and len(t) > 1]
    scope = " ".join(filtered).strip()

    # Final cleanup: remove trailing articles / prepositions
    scope = re.sub(r"^(a|an|the|of|for|in|on|from|to)\s+", "", scope)
    scope = re.sub(r"\s+(a|an|the|of|for|in|on|from|to)$", "", scope)

    return scope if scope else query.split()[0] if query else "general"


def extract_requirements(query: str) -> Dict[str, Any]:
    """
    Extracts structured requirement flags from the query.
    These flags are used during cache match validation to decide if a cached
    answer is sufficient or whether KNOWLEDGE_REUSE is required.

    Args:
        query (str): Normalized query string (lowercase).

    Returns:
        Dict[str, Any]: Requirement flags dict.
    """
    q = query.lower()
    requirements: Dict[str, Any] = {}

    # Boolean requirement flags
    for req_key, patterns in _REQUIREMENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                requirements[req_key] = True
                break

    # Programming language detection
    detected_language = None
    for lang, patterns in _PROGRAMMING_LANGUAGES.items():
        for pattern in patterns:
            if re.search(pattern, q):
                detected_language = lang
                break
        if detected_language:
            break
    requirements["language"] = detected_language

    return requirements


from config.config import DYNAMIC_QUERY_TTL, NEWS_QUERY_TTL, STABLE_QUERY_TTL, EXACT_ANSWER_TTL

# Freshness keywords
_REAL_TIME_PATTERNS = [
    r"\btoday'?s\b", r"\bcurrent\b", r"\blive\b", r"\bnow\b", r"\bprice\b",
    r"\bgold price\b", r"\bstock\b", r"\bweather\b", r"\bscore\b", r"\brate\b"
]

_NEWS_PATTERNS = [
    r"\bnews\b", r"\blatest\b", r"\brecent\b", r"\bhappened\b", r"\bthis week\b", r"\bthis month\b"
]


def classify_freshness(query: str) -> str:
    """
    Classifies a query's freshness requirement:
    - 'real_time': requires immediate fresh search (short 5 min TTL)
    - 'news': current events/recent news (1 hour TTL)
    - 'stable_educational': static concept/algorithm (long TTL)
    """
    q = query.lower()
    for pattern in _REAL_TIME_PATTERNS:
        if re.search(pattern, q):
            return "real_time"
    for pattern in _NEWS_PATTERNS:
        if re.search(pattern, q):
            return "news"
    return "stable_educational"


def get_ttl_for_query(query: str) -> int:
    """Returns the appropriate TTL in seconds based on query freshness category."""
    category = classify_freshness(query)
    if category == "real_time":
        return DYNAMIC_QUERY_TTL
    elif category == "news":
        return NEWS_QUERY_TTL
    return EXACT_ANSWER_TTL


def classify_query(query: str) -> Dict[str, Any]:
    """
    Full classification of a query — returns intent, scope, requirements, and freshness in one call.

    Args:
        query (str): Normalized query string (lowercase).

    Returns:
        Dict[str, Any]: {
            "intent": str,
            "scope": str,
            "requirements": Dict[str, Any],
            "freshness": str,
            "ttl": int
        }
    """
    return {
        "intent": classify_intent(query),
        "scope": extract_scope(query),
        "requirements": extract_requirements(query),
        "freshness": classify_freshness(query),
        "ttl": get_ttl_for_query(query),
    }



def intents_compatible(intent_a: str, intent_b: str) -> bool:
    """
    Checks whether two intents are compatible enough for cache reuse.
    Some intents are compatible (e.g., concept_explanation and definition);
    others are not (e.g., concept_explanation and code_generation).

    Args:
        intent_a (str): Intent of cached entry.
        intent_b (str): Intent of new query.

    Returns:
        bool: True if the intents are compatible for cache reuse.
    """
    if intent_a == intent_b:
        return True
    compatible_groups = [
        {"concept_explanation", "definition", "summary"},
        {"example_request", "step_by_step"},
    ]
    for group in compatible_groups:
        if intent_a in group and intent_b in group:
            return True
    return False


def requirements_are_subset(old_reqs: Dict[str, Any], new_reqs: Dict[str, Any]) -> bool:
    """
    Checks whether the new query's requirements are a subset of the old query's requirements.
    If new has MORE requirements than old, we need KNOWLEDGE_REUSE (not ANSWER_HIT).

    Args:
        old_reqs (Dict): Requirements from the cached entry.
        new_reqs (Dict): Requirements from the new query.

    Returns:
        bool: True if new requirements are a subset of (or equal to) old requirements.
    """
    # Language check: if new requires a specific language, old must have same language
    new_lang = new_reqs.get("language")
    old_lang = old_reqs.get("language")
    if new_lang and new_lang != old_lang:
        return False

    # Boolean requirement flags: check that every True flag in new is also True in old
    bool_keys = [k for k in new_reqs if k != "language"]
    for key in bool_keys:
        if new_reqs.get(key) is True and old_reqs.get(key) is not True:
            return False
    return True


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "explain stack data structure",
        "what is a stack data structure",
        "explain stack data structure in easy language with an example",
        "give java code for stack implementation",
        "compare stack and queue data structures",
        "explain merge sort step by step",
        "debug this python code for binary search",
        "summarize quicksort algorithm",
        "calculate time complexity of bubble sort",
    ]

    print("Intent Classifier — Self-Test")
    print("=" * 65)
    for q in tests:
        result = classify_query(q)
        print(f"  Query       : {q!r}")
        print(f"  Intent      : {result['intent']}")
        print(f"  Scope       : {result['scope']!r}")
        print(f"  Requirements: {result['requirements']}")
        print()
