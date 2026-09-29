"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 10 - Integrated stateless web-grounded RAG orchestrator and interactive CLI.
Dependencies: gc, time, typing, sys, config.config, phase1-9 modules
"""

import sys
import gc
import time
import os
import re
import json
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Package configs and utilities
from config.config import (
    validate_config,
    MAX_SEARCH_RESULTS,
    MAX_SOURCES_TO_SCRAPE,
    TOP_K,
    RESEARCH_DEPTH_CONFIG,
    RESEARCH_DEPTH_LEVELS
)
from utils.import_timing import timed_import

CacheManager = timed_import("cache.cache_manager").CacheManager
classify_query = timed_import("cache.intent_classifier").classify_query
normalize_and_hash = timed_import("cache.query_normalizer").normalize_and_hash
is_domain_trusted = timed_import("config.trusted_sources").is_domain_trusted
setup_logger = timed_import("utils.logger").setup_logger
timed_import("utils.helper")
from utils.helper import (
    print_phase_header,
    print_loading,
    print_processing,
    print_success,
    print_failure,
    print_statistics,
    PhaseTimer
)

# Phase Module Imports
WebSearcher = timed_import("phase1.web_search").WebSearcher
ResultFilter = timed_import("phase2.result_filter").ResultFilter
WebsiteLoader = timed_import("phase3.website_loader").WebsiteLoader
DocumentChunker = timed_import("phase4.chunker").DocumentChunker
ChunkEmbedder = timed_import("phase5.embedder").ChunkEmbedder
VectorStoreManager = timed_import("phase6.vector_store").VectorStoreManager
SemanticRetriever = timed_import("phase7.retriever").SemanticRetriever
PromptBuilder = timed_import("phase8.prompt_builder").PromptBuilder
GeminiContentGenerator = timed_import("phase9.gemini_client").GeminiContentGenerator

# Initialize logger
logger = setup_logger("web_grounded_rag_app")

def _log_rag_time(stage: str, started: float) -> float:
    elapsed = time.perf_counter() - started
    logger.info("[RAG-TIME] %s: %.3f sec", stage, elapsed)
    return elapsed


def _time_pipeline_run(method):
    @wraps(method)
    def timed_run(*args, **kwargs):
        started = time.perf_counter()
        logger.info("[RAG-TIME] pipeline start")
        try:
            return method(*args, **kwargs)
        finally:
            _log_rag_time("total", started)
    return timed_run


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightweight context deduplication to remove duplicate or highly overlapping text chunks.
    Ensures unique facts, diverse evidence, and optimized token consumption.
    """
    if not chunks:
        return []

    unique_chunks = []
    seen_word_sets = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        words = set(re.findall(r'\w+', text.lower()))
        if not words:
            continue

        is_dup = False
        for seen in seen_word_sets:
            overlap = len(words & seen)
            smaller_size = min(len(words), len(seen))
            if smaller_size > 0 and (overlap / smaller_size) > 0.80:
                is_dup = True
                break

        if not is_dup:
            unique_chunks.append(chunk)
            seen_word_sets.append(words)

    return unique_chunks

class WebGroundedRAGPipeline:
    """
    Stateless orchestrator that integrates Phase 1 through 10.
    Conducts web-grounded educational retrieval generation and runs immediate cleanup.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None) -> None:
        """Initializes and pre-warms the pipeline components."""
        logger.debug("Initializing and pre-warming WebGroundedRAGPipeline components...")
        self.searcher: Optional[WebSearcher] = WebSearcher()
        self.filter_engine: Optional[ResultFilter] = ResultFilter()
        self.cache_manager = cache_manager or CacheManager()
        self.loader: Optional[WebsiteLoader] = WebsiteLoader(self.cache_manager)
        self.chunker: Optional[DocumentChunker] = DocumentChunker()
        self.embedder: Optional[ChunkEmbedder] = ChunkEmbedder()
        self.prompt_builder: Optional[PromptBuilder] = PromptBuilder()
        self.gemini_client: Optional[GeminiContentGenerator] = GeminiContentGenerator()
        self.chat_history: List[Dict[str, str]] = []
        logger.info("Pipeline components pre-warmed and ready.")

    @_time_pipeline_run
    def run_pipeline(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = TOP_K,
        max_search_results: int = MAX_SEARCH_RESULTS,
        max_sources_to_scrape: int = MAX_SOURCES_TO_SCRAPE,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_id: Optional[str] = None,
        research_depth: str = "quick"
    ) -> Dict[str, Any]:
        """
        Executes the multi-level cached RAG pipeline with conversation context resolution
        and user-controlled research depth:
        1. Context Loading & Follow-Up Query Reformulation (BEFORE Cache Lookup)
        2. Query Normalization, Hashing, & Intent/Freshness Classification using Resolved Query
        3. Research Depth validation and source target configuration
        4. Stampede Protection (Request Coalescing)
        5. Cache Sufficiency Check (L1 Redis, L2 Mongo, L3 SQLite Semantic) considering research depth
        6. Full 10-Phase Web-Grounded RAG Execution (on Cache Miss or Insufficient Cache Depth)
        7. Cache Persistence (L1 Redis + L2 Mongo + L3 SQLite) & User History Recording
        """
        DEPTH_ORDER = {"quick": 1, "standard": 2, "deep": 3}
        clean_depth = (research_depth or "quick").lower().strip()
        if clean_depth not in RESEARCH_DEPTH_LEVELS:
            clean_depth = "quick"
        target_sources = RESEARCH_DEPTH_LEVELS[clean_depth]
        max_output_tokens = RESEARCH_DEPTH_CONFIG.get(clean_depth, {}).get("max_output_tokens", 500)

        # Expose depth internally to pipeline resource counts
        max_sources_to_scrape = target_sources
        max_search_results = max(target_sources + 5, 10)

        clean_chat_id = chat_id if (chat_id and str(chat_id).strip() != "None" and str(chat_id).strip() != "") else None

        logger.info(f"[CHAT] user_id={user_id} chat_id={clean_chat_id}")
        logger.info(f"[RESEARCH] depth={clean_depth} target_sources={target_sources}")
        logger.info(f"[GENERATION] depth={clean_depth} max_output_tokens={max_output_tokens}")
        logger.info(f"RAG Pipeline Run started for query: '{query}' (user_id='{user_id}', chat_id='{clean_chat_id}', research_depth='{clean_depth}')")

        total_start = time.perf_counter()

        # Load chat context scoped strictly by (user_id, chat_id)
        context_started = time.perf_counter()
        if chat_history is not None:
            active_history = chat_history
        elif clean_chat_id and user_id:
            active_history = self.cache_manager.get_chat_context(user_id, clean_chat_id, limit=10)
        else:
            active_history = []

        logger.info(f"[CONTEXT] Loaded {len(active_history)} messages for user_id={user_id} chat_id={clean_chat_id}")
        _log_rag_time("conversation context lookup", context_started)

        # Context Resolution BEFORE Cache Lookup
        from cache.context_resolver import resolve_context_query
        context_resolution_started = time.perf_counter()
        res_status, resolved_query = resolve_context_query(query, active_history)
        _log_rag_time("context resolution", context_resolution_started)

        if res_status == "AMBIGUOUS":
            logger.warning(f"[CONTEXT] Unable to resolve reference for query: '{query}'")
            msg = "Could you please clarify what topic or item you are referring to?"
            return {
                "success": True,
                "answer": msg,
                "sources": [],
                "research_depth": clean_depth,
                "sources_analyzed": 0,
                "sources_used": 0,
                "timings": {"total_request_ms": (time.perf_counter() - total_start) * 1000},
                "statistics": {"Status": "Ambiguous context reference"},
                "url_metrics": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "error": None,
                "status": "AMBIGUOUS_CONTEXT",
                "cache_mode": "AMBIGUOUS_CONTEXT"
            }

        effective_query = resolved_query
        if res_status == "REWRITTEN":
            logger.info(f"[CONTEXT] Resolved follow-up: \"{query}\" -> \"{effective_query}\"")
        else:
            logger.info(f"[CONTEXT] Independent query - no reformulation required")

        timings: Dict[str, float] = {}
        statistics: Dict[str, Any] = {}
        url_metrics: List[Dict[str, Any]] = []
        token_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        db_manager: Optional[VectorStoreManager] = None
        temp_collection_name = f"temp_rag_col_{int(time.time())}"

        normalized_query, query_hash = normalize_and_hash(effective_query)
        classification = classify_query(normalized_query)
        freshness = classification.get("freshness", "stable_educational")

        def is_depth_sufficient(cached_d: Optional[str], req_d: str) -> bool:
            c_val = (cached_d or "quick").lower()
            r_val = (req_d or "quick").lower()
            return DEPTH_ORDER.get(c_val, 1) >= DEPTH_ORDER.get(r_val, 1)

        # Request Coalescing (Stampede Protection)
        is_leader, event = self.cache_manager.acquire_coalesce_lock(query_hash)
        if not is_leader and event is not None:
            logger.info(f"[Coalesce] Waiting for in-flight request for hash {query_hash[:12]}...")
            event.wait(timeout=35)
            # Re-check exact L1 Redis or L2 Mongo answer generated by leader
            exact = self.cache_manager.get_exact(query_hash) or self.cache_manager.get_mongo_answer(query_hash)
            if exact:
                cached_d = exact.get("research_depth", "quick")
                if is_depth_sufficient(cached_d, clean_depth):
                    exact.setdefault("timings", {})["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                    exact["cache_mode"] = "REDIS_HIT"
                    exact["research_depth"] = clean_depth
                    self._record_user_history(user_id, query_hash, query, normalized_query, classification, exact, "REDIS_HIT", chat_id=chat_id, resolved_query=effective_query)
                    return exact

        cache_timings: Dict[str, float] = {}
        try:
            # Skip stale caches for real-time / dynamic queries (e.g., live prices, news)
            if freshness != "real_time":
                # ── Tier 1: L1 Redis Exact Cache Lookup ─────────────────────────
                lookup_start = time.perf_counter()
                exact = self.cache_manager.get_exact(query_hash)
                cache_timings["exact_cache_lookup_ms"] = (time.perf_counter() - lookup_start) * 1000
                _log_rag_time("exact cache lookup (Redis)", lookup_start)
                if exact:
                    cached_d = exact.get("research_depth", "quick")
                    logger.info(f"[RESEARCH CACHE] Cached depth={cached_d}")
                    logger.info(f"[RESEARCH CACHE] Requested depth={clean_depth}")
                    if is_depth_sufficient(cached_d, clean_depth):
                        logger.info("[RESEARCH CACHE] SUFFICIENT")
                        logger.info("[RAG-TIME] cache mode: Redis exact hit")
                        exact["cache_mode"] = "REDIS_HIT"
                        exact["research_depth"] = clean_depth
                        exact.setdefault("timings", {})["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                        exact["timings"].update(cache_timings)
                        logger.info(
                            f"[CACHE ROUTE] REDIS EXACT HIT | user_id='{user_id}' | "
                            f"normalized_query='{normalized_query}' | query_hash='{query_hash[:12]}...' | "
                            f"intent='{classification.get('intent')}' | scope='{classification.get('scope')}' | "
                            f"similarity=1.000 | cache source=Redis L1"
                        )
                        self._record_user_history(user_id, query_hash, query, normalized_query, classification, exact, "REDIS_HIT", chat_id=chat_id, resolved_query=effective_query)
                        return exact
                    else:
                        logger.info("[RESEARCH CACHE] INSUFFICIENT")

                # ── Tier 2: L2 MongoDB Persistent Reusable Answer Lookup ─────────
                mongo_start = time.perf_counter()
                mongo_hit = self.cache_manager.get_mongo_answer(query_hash)
                cache_timings["mongo_cache_lookup_ms"] = (time.perf_counter() - mongo_start) * 1000
                _log_rag_time("reusable-answer lookup (MongoDB)", mongo_start)
                if mongo_hit:
                    cached_d = mongo_hit.get("research_depth", "quick")
                    logger.info(f"[RESEARCH CACHE] Cached depth={cached_d}")
                    logger.info(f"[RESEARCH CACHE] Requested depth={clean_depth}")
                    if is_depth_sufficient(cached_d, clean_depth):
                        logger.info("[RESEARCH CACHE] SUFFICIENT")
                        logger.info("[RAG-TIME] cache mode: MongoDB reusable-answer hit")
                        mongo_hit["cache_mode"] = "MONGODB_HIT"
                        mongo_hit["research_depth"] = clean_depth
                        mongo_hit.setdefault("timings", {})["total_request_ms"] = (time.perf_counter() - total_start) * 1000
                        mongo_hit["timings"].update(cache_timings)
                        logger.info(
                            f"[CACHE ROUTE] MONGODB GLOBAL ANSWER HIT | user_id='{user_id}' | "
                            f"normalized_query='{normalized_query}' | query_hash='{query_hash[:12]}...' | "
                            f"intent='{classification.get('intent')}' | scope='{classification.get('scope')}' | "
                            f"similarity=1.000 | cache source=MongoDB L2"
                        )
                        self._record_user_history(user_id, query_hash, query, normalized_query, classification, mongo_hit, "MONGODB_HIT", chat_id=chat_id, resolved_query=effective_query)
                        return mongo_hit
                    else:
                        logger.info("[RESEARCH CACHE] INSUFFICIENT")

                # ── Tier 3: L3 SQLite Semantic Cache & Knowledge Reuse ───────────
                semantic_lookup_start = time.perf_counter()
                query_embedding = self.cache_manager.get_query_embedding(query_hash)
                if query_embedding is None:
                    embedding_started = time.perf_counter()
                    query_embedding = self.embedder.embed_texts([normalized_query])[0]
                    _log_rag_time("query embedding", embedding_started)
                    self.cache_manager.set_query_embedding(query_hash, query_embedding)

                decision = self.cache_manager.decide_semantic(effective_query, normalized_query, query_embedding)
                _log_rag_time("semantic cache lookup (SQLite)", semantic_lookup_start)
                if decision.type == "ANSWER_HIT":
                    cached_d = getattr(decision, "research_depth", "quick")
                    logger.info(f"[RESEARCH CACHE] Cached depth={cached_d}")
                    logger.info(f"[RESEARCH CACHE] Requested depth={clean_depth}")
                    if is_depth_sufficient(cached_d, clean_depth):
                        logger.info("[RESEARCH CACHE] SUFFICIENT")
                        logger.info("[RAG-TIME] cache mode: SQLite semantic answer hit")
                        hit_resp = self._build_cache_hit_response(decision, cache_timings, total_start)
                        hit_resp["cache_mode"] = "SEMANTIC_HIT"
                        hit_resp["research_depth"] = clean_depth
                        hit_resp["sources_analyzed"] = target_sources
                        hit_resp["sources_used"] = len(hit_resp.get("sources", []))
                        logger.info(f"[SemanticCache] GLOBAL ANSWER HIT | similarity={decision.similarity:.3f}")
                        logger.info(
                            f"[CACHE ROUTE] SQLITE SEMANTIC ANSWER HIT | user_id='{user_id}' | "
                            f"normalized_query='{normalized_query}' | query_hash='{query_hash[:12]}...' | "
                            f"intent='{classification.get('intent')}' | scope='{classification.get('scope')}' | "
                            f"similarity={decision.similarity:.3f} | cache source=SQLite L3"
                        )
                        self._record_user_history(user_id, query_hash, query, normalized_query, classification, hit_resp, "SEMANTIC_HIT", chat_id=chat_id, resolved_query=effective_query)
                        return hit_resp
                    else:
                        logger.info("[RESEARCH CACHE] INSUFFICIENT")

                if decision.type == "KNOWLEDGE_REUSE" and is_depth_sufficient(getattr(decision, "research_depth", "quick"), clean_depth):
                    reused = self._run_with_cached_knowledge(
                        effective_query, decision, query_hash, normalized_query, query_embedding,
                        active_history, top_k, total_start, cache_timings
                    )
                    if reused is not None:
                        logger.info("[RAG-TIME] cache mode: SQLite knowledge reuse")
                        reused["cache_mode"] = "KNOWLEDGE_REUSE"
                        reused["research_depth"] = clean_depth
                        reused["sources_analyzed"] = target_sources
                        reused["sources_used"] = len(reused.get("sources", []))
                        logger.info(f"[SemanticCache] KNOWLEDGE REUSE HIT | similarity={decision.similarity:.3f}")
                        logger.info(
                            f"[CACHE ROUTE] KNOWLEDGE REUSE | user_id='{user_id}' | "
                            f"normalized_query='{normalized_query}' | query_hash='{query_hash[:12]}...' | "
                            f"intent='{classification.get('intent')}' | scope='{classification.get('scope')}' | "
                            f"similarity={decision.similarity:.3f} | cache source=SQLite L3 Chunks"
                        )
                        self._record_user_history(user_id, query_hash, query, normalized_query, classification, reused, "KNOWLEDGE_REUSE", chat_id=chat_id, resolved_query=effective_query)
                        return reused
            else:
                logger.info(f"[Freshness] Real-time query detected ('{effective_query}'). Bypassing stale cache for fresh search.")
                logger.info("[RAG-TIME] cache mode: bypassed for real-time query")
                embedding_started = time.perf_counter()
                query_embedding = self.embedder.embed_texts([normalized_query])[0]
                _log_rag_time("query embedding", embedding_started)
        finally:
            if is_leader:
                self.cache_manager.release_coalesce_lock(query_hash)

        logger.info(
            f"[CACHE ROUTE] FULL RAG MISS | user_id='{user_id}' | "
            f"normalized_query='{normalized_query}' | query_hash='{query_hash[:12]}...' | "
            f"intent='{classification.get('intent')}' | scope='{classification.get('scope')}' | "
            f"similarity=0.000 | cache source=None"
        )
        logger.info("[RAG-TIME] cache mode: FULL RAG MISS")

        try:
            # 0. Configuration Validation
            validate_config()

            # Ensure clients are initialized
            if not self.searcher:
                self.searcher = WebSearcher()
                self.filter_engine = ResultFilter()
                self.loader = WebsiteLoader(self.cache_manager)
                self.chunker = DocumentChunker()
                self.embedder = ChunkEmbedder()
                self.prompt_builder = PromptBuilder()
                self.gemini_client = GeminiContentGenerator()

            # ====================================================
            # QUERY REFORMULATION & OPTIMIZATION
            # ====================================================
            print_phase_header("Query Optimization")
            start = time.perf_counter()
            if active_history:
                search_query = self.searcher.reformulate_conversational_query(query, active_history)
                print_success(f"Reformulated query for search: '{search_query}'")
            else:
                search_query = self.searcher.optimize_query(query)
                print_success(f"Optimized search query: '{search_query}'")
            timings["Phase 0 (Query Optimization)"] = time.perf_counter() - start
            _log_rag_time("query optimization", start)

            fallback_mode = False
            raw_search_results = []
            filtered_results = []
            successful_docs = []

            try:
                # ====================================================
                # PHASE 1: BROAD WEB SEARCH (TAVILY)
                # ====================================================
                print_phase_header("Phase 1: Broad Web Search (Tavily)")
                t_search = time.perf_counter()
                raw_search_results = self.searcher.search(search_query, max_results=max_search_results)
                timings["Phase 1 (Web Search)"] = time.perf_counter() - t_search
                _log_rag_time("Tavily search", t_search)
                print_success(f"Discovered {len(raw_search_results)} candidate web sources.")

                # ====================================================
                # PHASE 2: URL VALIDATION & DEDUPLICATION
                # ====================================================
                print_phase_header("Phase 2: URL Validation & Deduplication")
                t_filter = time.perf_counter()
                filtered_results = self.filter_engine.filter_results(raw_search_results)
                timings["Phase 2 (URL Filtering)"] = time.perf_counter() - t_filter
                _log_rag_time("result filtering", t_filter)
                print_success(f"Retained {len(filtered_results)} unique, valid URLs.")

                if not filtered_results:
                    msg = "I couldn't find enough relevant web content to answer this question reliably."
                    logger.warning(f"Zero valid URLs for query '{query}'. Returning early.")
                    print_failure("No valid web sources were discovered.")
                    print_processing("Stopping pipeline early: zero websites loaded.")

                    total_duration = time.perf_counter() - total_start
                    timings["TOTAL RESPONSE TIME"] = total_duration

                    statistics.update({
                        "Total Duration": f"{total_duration:.3f}s",
                        "Discovered Sources": len(raw_search_results),
                        "Analyzed Sources": 0,
                        "Retrieved Chunks": 0,
                        "Cited Sources": 0,
                        "Response Length (chars)": len(msg),
                        "Status": "No web sources discovered",
                        "Research Depth": clean_depth.capitalize(),
                        "Sources Analyzed": 0,
                        "Sources Used": 0
                    })

                    return {
                        "success": True,
                        "answer": msg,
                        "sources": [],
                        "research_depth": clean_depth,
                        "sources_analyzed": 0,
                        "sources_used": 0,
                        "status": "INSUFFICIENT_EVIDENCE",
                        "timings": timings,
                        "statistics": statistics,
                        "url_metrics": [],
                        "token_usage": token_usage,
                        "error": None
                    }

                # Select candidate URLs to scrape up to max_sources_to_scrape
                candidates_to_scrape = filtered_results[:max_sources_to_scrape]

                # Incremental research tracking / logging
                existing_cached_count = sum(1 for item in candidates_to_scrape if self.cache_manager.get_page(item["url"]) is not None)
                new_unique_count = len(candidates_to_scrape) - existing_cached_count
                logger.info(f"[RESEARCH MORE] Existing sources={existing_cached_count}")
                logger.info(f"[RESEARCH MORE] Additional sources requested={target_sources}")
                logger.info(f"[RESEARCH MORE] New unique sources={new_unique_count}")

                # ====================================================
                # PHASE 3: PARALLEL WEB EXTRACTION
                # ====================================================
                print_phase_header("Phase 3: Parallel Web Extraction")
                start = time.perf_counter()
                urls_to_load = [item["url"] for item in candidates_to_scrape]
                loaded_docs = self.loader.load_multiple(urls_to_load)

                url_to_title = {item["url"]: item["title"] for item in candidates_to_scrape}
                for doc in loaded_docs:
                    doc["title"] = url_to_title.get(doc["url"], "Web Source")
                    url_metrics.append({
                        "url": doc["url"],
                        "status": "Success" if doc.get("success") else "Failed",
                        "duration": doc.get("duration", 0.0),
                        "method": doc.get("method", "Unknown"),
                        "chars": len(doc.get("content", ""))
                    })

                timings["Phase 3 (Website Loading)"] = time.perf_counter() - start
                _log_rag_time("website loading", start)

                successful_docs = [doc for doc in loaded_docs if doc.get("success", False)]
                print_success(f"Successfully scraped and extracted {len(successful_docs)} of {len(urls_to_load)} web pages.")

                if not successful_docs:
                    msg = "I couldn't find enough relevant web content to answer this question reliably."
                    logger.warning(f"All web extractions failed for query '{query}'. Returning early.")
                    print_failure("All web extractions failed.")

                    total_duration = time.perf_counter() - total_start
                    timings["TOTAL RESPONSE TIME"] = total_duration

                    statistics.update({
                        "Total Duration": f"{total_duration:.3f}s",
                        "Discovered Sources": len(raw_search_results),
                        "Analyzed Sources": 0,
                        "Retrieved Chunks": 0,
                        "Cited Sources": 0,
                        "Response Length (chars)": len(msg),
                        "Status": "Web page extraction failed",
                        "Research Depth": clean_depth.capitalize(),
                        "Sources Analyzed": 0,
                        "Sources Used": 0
                    })

                    return {
                        "success": True,
                        "answer": msg,
                        "sources": [],
                        "research_depth": clean_depth,
                        "sources_analyzed": 0,
                        "sources_used": 0,
                        "status": "INSUFFICIENT_EVIDENCE",
                        "cache_mode": "RAG_MISS",
                        "timings": timings,
                        "statistics": statistics,
                        "url_metrics": url_metrics,
                        "token_usage": token_usage,
                        "error": None
                    }
            except Exception as outer_err:
                logger.warning(f"Web extraction failed: {outer_err}. Falling back to parametric generation.")
                print_failure(f"Web extraction pipeline failed: {outer_err}")
                fallback_mode = True

            if fallback_mode:
                print_phase_header("Phase 9: Content Generation (Fallback)")
                start = time.perf_counter()
                fallback_prompt = self.prompt_builder.build_fallback_prompt(query, active_history)
                answer, token_usage = self.gemini_client.generate_response_with_tokens(fallback_prompt, max_tokens=max_output_tokens)
                timings["Phase 9 (LLM Generation)"] = time.perf_counter() - start

                total_duration = time.perf_counter() - total_start
                timings["TOTAL RESPONSE TIME"] = total_duration

                statistics.update({
                    "Total Duration": f"{total_duration:.3f}s",
                    "Discovered Sources": 0,
                    "Analyzed Sources": 0,
                    "Retrieved Chunks": 0,
                    "Cited Sources": 0,
                    "Response Length (chars)": len(answer),
                    "Fallback Mode": "Active (parametric response)",
                    "Research Depth": clean_depth.capitalize(),
                    "Sources Analyzed": 0,
                    "Sources Used": 0
                })

                active_history.append({"role": "user", "content": query})
                active_history.append({"role": "assistant", "content": answer})

                return {
                    "success": True,
                    "answer": answer,
                    "sources": [],
                    "research_depth": clean_depth,
                    "sources_analyzed": 0,
                    "sources_used": 0,
                    "cache_mode": "RAG_MISS",
                    "timings": timings,
                    "statistics": statistics,
                    "url_metrics": url_metrics,
                    "token_usage": token_usage,
                    "error": None
                }

            # Grounded RAG Pipeline
            # ====================================================
            # PHASE 4: CHUNKING
            # ====================================================
            print_phase_header("Phase 4: Chunking")
            start = time.perf_counter()
            chunks = self.chunker.chunk_documents(successful_docs)
            timings["Phase 4 (Chunking)"] = time.perf_counter() - start
            _log_rag_time("chunking", start)
            print_success(f"Split documents into {len(chunks)} text chunks.")

            if not chunks:
                raise ValueError("Document parsing yielded zero text chunks.")

            # Cap chunks if very large to optimize embedding batch size
            max_embedding_candidates = 80
            if len(chunks) > max_embedding_candidates:
                chunks = chunks[:max_embedding_candidates]

            # ====================================================
            # PHASE 5: BATCH EMBEDDING
            # ====================================================
            print_phase_header("Phase 5: Batch Embedding")
            start = time.perf_counter()
            embedded_chunks = self.embedder.embed_chunks(chunks)
            timings["Phase 5 (Embedding Generation)"] = time.perf_counter() - start
            _log_rag_time("embedding", start)
            print_success(f"Generated vector representations for {len(embedded_chunks)} chunks.")

            # ====================================================
            # PHASE 6: VECTOR STORAGE
            # ====================================================
            print_phase_header("Phase 6: Vector Storage")
            start = time.perf_counter()
            chroma_init_start = time.perf_counter()
            db_manager = VectorStoreManager()
            _log_rag_time("Chroma initialization", chroma_init_start)
            collection_started = time.perf_counter()
            col = db_manager.create_collection(temp_collection_name)
            _log_rag_time("Chroma collection creation", collection_started)
            chroma_ingest_start = time.perf_counter()
            db_manager.add_chunks(embedded_chunks)
            _log_rag_time("Chroma ingestion", chroma_ingest_start)
            timings["Phase 6 (ChromaDB Storage)"] = time.perf_counter() - start
            print_success(f"Ingested vectors into ChromaDB in-memory collection '{temp_collection_name}'.")

            # ====================================================
            # PHASE 7: SEMANTIC RETRIEVAL
            # ====================================================
            print_phase_header("Phase 7: Semantic Retrieval")
            start = time.perf_counter()
            retriever = SemanticRetriever(self.embedder, col)
            raw_retrieved_chunks = retriever.retrieve(search_query, top_k=top_k)
            retrieved_chunks = deduplicate_chunks(raw_retrieved_chunks)
            timings["Phase 7 (Retrieval)"] = time.perf_counter() - start
            _log_rag_time("retrieval", start)
            print_success(f"Retrieved {len(raw_retrieved_chunks)} chunks -> {len(retrieved_chunks)} unique deduplicated chunks.")
            logger.info(f"[RESEARCH] retrieved_chunks={len(raw_retrieved_chunks)} unique_chunks={len(retrieved_chunks)}")

            # Extract existing answer if available in context for Research More / Answer Expansion
            existing_answer = None
            if active_history:
                for msg in reversed(active_history):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        existing_answer = msg.get("content")
                        break

            # ====================================================
            # PHASE 8: PROMPT BUILDER
            # ====================================================
            print_phase_header("Phase 8: Prompt Builder")
            start = time.perf_counter()
            grounded_prompt = self.prompt_builder.build_prompt(
                query, retrieved_chunks, active_history,
                research_depth=clean_depth, existing_answer=existing_answer
            )
            timings["Phase 8 (Prompt Building)"] = time.perf_counter() - start
            _log_rag_time("prompt construction", start)
            est_prompt_tokens = len(grounded_prompt) // 4
            logger.info(f"[GENERATION] depth={clean_depth} estimated_prompt_tokens=~{est_prompt_tokens} requested_output_tokens={max_output_tokens}")
            print_success("Structured grounded prompt constructed successfully.")

            # ====================================================
            # PHASE 9: LLM GENERATION
            # ====================================================
            print_phase_header("Phase 9: Content Generation")
            start = time.perf_counter()
            answer, token_usage = self.gemini_client.generate_response_with_tokens(
                grounded_prompt, max_tokens=max_output_tokens
            )
            timings["Phase 9 (LLM Generation)"] = time.perf_counter() - start
            _log_rag_time("Groq/Gemini generation", start)
            print_success("Grounded response generated.")

            # Parse inline citations from the generated answer
            evaluation_started = time.perf_counter()
            cited_indices = set()
            for match in re.findall(r'\[(?:Source\s*)?(\d+)\]', answer):
                try:
                    idx = int(match) - 1
                    if 0 <= idx < len(retrieved_chunks):
                        cited_indices.add(idx)
                except ValueError:
                    pass

            # Map cited chunks back to unique source URLs
            cited_sources = []
            seen_cited_urls = set()

            if cited_indices:
                for idx in sorted(list(cited_indices)):
                    chunk = retrieved_chunks[idx]
                    url = chunk["url"]
                    if url not in seen_cited_urls:
                        seen_cited_urls.add(url)
                        cited_sources.append({"title": chunk["title"], "url": url})
            else:
                for chunk in retrieved_chunks:
                    url = chunk["url"]
                    if url not in seen_cited_urls:
                        seen_cited_urls.add(url)
                        cited_sources.append({"title": chunk["title"], "url": url})
            _log_rag_time("answer validation and citation processing (no separate evaluator configured)", evaluation_started)

            active_history.append({"role": "user", "content": query})
            active_history.append({"role": "assistant", "content": answer})

            # Note: Phase 10 cleanup happens in finally block and its timing is captured there
            total_duration = time.perf_counter() - total_start
            timings["TOTAL RESPONSE TIME"] = total_duration

            statistics.update({
                "Total Duration": f"{total_duration:.3f}s",
                "Discovered Sources": len(raw_search_results),
                "Analyzed Sources": len(successful_docs),
                "Retrieved Chunks": len(retrieved_chunks),
                "Cited Sources": len(cited_sources),
                "Total Chunks": len(chunks),
                "Response Length (chars)": len(answer),
                "Fallback Mode": "Inactive (grounded response)",
                "Research Depth": clean_depth.capitalize(),
                "Sources Analyzed": len(successful_docs),
                "Sources Used": len(cited_sources)
            })

            logger.info(
                f"[RESEARCH] discovered={len(raw_search_results)} "
                f"analyzed={len(successful_docs)} "
                f"used={len(cited_sources)} "
                f"cited={len(cited_sources)}"
            )

            result = {
                "success": True,
                "answer": answer,
                "sources": cited_sources,
                "research_depth": clean_depth,
                "sources_analyzed": len(successful_docs),
                "sources_used": len(cited_sources),
                "status": "SUCCESS",
                "timings": timings,
                "statistics": statistics,
                "url_metrics": url_metrics,
                "token_usage": token_usage,
                "error": None
            }
            cache_persist_started = time.perf_counter()
            self._populate_cache(
                query_hash, normalized_query, query, query_embedding, answer,
                result, embedded_chunks, successful_docs, research_depth=clean_depth
            )
            _log_rag_time("cache persistence", cache_persist_started)
            result["cache_mode"] = "RAG_MISS"
            history_persist_started = time.perf_counter()
            self._record_user_history(user_id, query_hash, query, normalized_query, classification, result, "RAG_MISS", chat_id=chat_id, resolved_query=effective_query)
            _log_rag_time("MongoDB history persistence", history_persist_started)
            return result

        except Exception as e:
            logger.error(f"[GENERATION] Pipeline failed at run: {e}")
            print_failure(f"Pipeline crashed: {e}")
            total_duration = time.perf_counter() - total_start
            timings["TOTAL RESPONSE TIME"] = total_duration

            err_str = str(e)
            if "limit" in err_str.lower() or "quota" in err_str.lower() or "unavailable" in err_str.lower():
                user_msg = (
                    "Deep research is temporarily unavailable because the AI generation limit has been reached. "
                    "Your previous answer is still available. Please try again later."
                )
            else:
                user_msg = f"Generation failed: {err_str}"

            return {
                "success": False,
                "answer": user_msg,
                "sources": [],
                "research_depth": clean_depth,
                "sources_analyzed": 0,
                "sources_used": 0,
                "timings": timings,
                "statistics": statistics,
                "url_metrics": url_metrics,
                "token_usage": token_usage,
                "error": err_str
            }
            
        finally:
            # ====================================================
            # PHASE 10: CLEANUP & TEARDOWN
            # ====================================================
            print_phase_header("Phase 10: Teardown & Cleanup")
            t_clean_start = time.perf_counter()
            
            if db_manager and db_manager.collection:
                try:
                    db_manager.delete_collection()
                except Exception as cleanup_err:
                    logger.warning(f"Error during ChromaDB cleanup: {cleanup_err}")
                    print_failure(f"ChromaDB deletion failed: {cleanup_err}")
            
            db_manager = None
            gc.collect()
            
            t_clean_elapsed = time.perf_counter() - t_clean_start
            timings["Phase 10 (Cleanup)"] = t_clean_elapsed
            _log_rag_time("Chroma cleanup and garbage collection", t_clean_start)
            
            logger.info("Garbage collection triggered. Memory freed. System stateless.")
            print_success("All temporary memories deleted. System returned to empty state.")

    def _build_cache_hit_response(
        self, decision: Any, cache_timings: Dict[str, float], total_start: float
    ) -> Dict[str, Any]:
        """Builds the normal response shape for a semantic answer hit."""
        timings = dict(cache_timings)
        timings["TOTAL RESPONSE TIME"] = time.perf_counter() - total_start
        return {
            "success": True,
            "answer": decision.answer or "",
            "sources": [],
            "timings": timings,
            "statistics": {"Cache Similarity": f"{decision.similarity:.3f}"},
            "url_metrics": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "error": None,
            "cache_mode": "SEMANTIC_ANSWER_HIT",
        }

    def _run_with_cached_knowledge(
        self, query: str, decision: Any, query_hash: str, normalized_query: str,
        query_embedding: List[float], active_history: List[Dict[str, str]],
        top_k: int, total_start: float, cache_timings: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Regenerates an answer from validated cached chunks."""
        cached_chunks = self.cache_manager.get_cached_knowledge(decision.chunk_ids)
        valid_chunks = [
            chunk for chunk in cached_chunks
            if is_domain_trusted(chunk.get("metadata", {}).get("url", ""))
        ]
        if not valid_chunks:
            return None

        db_manager = None
        try:
            db_manager = VectorStoreManager()
            collection = db_manager.create_collection(f"reuse_{int(time.time() * 1000)}")
            db_manager.add_chunks(valid_chunks)
            retriever = SemanticRetriever(self.embedder, collection)
            retrieved_chunks = retriever.retrieve(query, top_k=top_k)
            prompt = self.prompt_builder.build_prompt(query, retrieved_chunks, active_history)
            answer, token_usage = self.gemini_client.generate_response_with_tokens(prompt)
            sources = []
            seen_urls = set()
            for chunk in retrieved_chunks:
                url = chunk.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": chunk.get("title", ""), "url": url})

            result = {
                "success": True,
                "answer": answer,
                "sources": sources,
                "timings": {**cache_timings, "TOTAL RESPONSE TIME": time.perf_counter() - total_start},
                "statistics": {"Retrieved Chunks": len(retrieved_chunks)},
                "url_metrics": [],
                "token_usage": token_usage,
                "error": None,
                "cache_mode": "KNOWLEDGE_REUSE",
            }
            classification = classify_query(normalized_query)
            self.cache_manager.save_query_result(
                query_hash, normalized_query, query, classification["intent"],
                classification["scope"], classification["requirements"], answer,
                decision.source_ids, decision.chunk_ids, query_embedding, result
            )
            return result
        except Exception as error:
            logger.warning(f"[Cache] Knowledge reuse failed; continuing with full RAG: {error}")
            return None
        finally:
            if db_manager and db_manager.collection:
                try:
                    db_manager.delete_collection()
                except Exception as cleanup_error:
                    logger.warning(f"[Cache] Knowledge reuse cleanup failed: {cleanup_error}")

    def _populate_cache(
        self, query_hash: str, normalized_query: str, original_query: str,
        query_embedding: List[float], answer: str, pipeline_result: Dict[str, Any],
        embedded_chunks: List[Dict[str, Any]], loaded_docs: List[Dict[str, Any]],
        research_depth: str = "quick"
    ) -> None:
        """Persists a successful grounded run without affecting its response."""
        from cache.cache_manager import is_answer_usable
        if not is_answer_usable(answer, pipeline_result.get("status"), pipeline_result.get("success", True)):
            logger.warning(f"[Cache] Skipping cache population for unusable/insufficient answer (hash={query_hash[:12]}).")
            return
        try:
            self.cache_manager.save_full_result(
                query_hash, normalized_query, original_query, query_embedding,
                answer, pipeline_result, embedded_chunks, loaded_docs,
                research_depth=research_depth
            )
        except Exception as error:
            logger.warning(f"[Cache] Failed to populate cache: {error}")

    def _record_user_history(
        self,
        user_id: str,
        query_hash: str,
        original_query: str,
        normalized_query: str,
        classification: Dict[str, Any],
        result: Dict[str, Any],
        cache_status: str,
        chat_id: Optional[str] = None,
        resolved_query: Optional[str] = None
    ) -> None:
        """Helper to record query execution history into MongoDB."""
        try:
            sources = result.get("sources", [])
            res_depth = result.get("research_depth", "quick")
            sources_analyzed = result.get("sources_analyzed", result.get("statistics", {}).get("Analyzed Sources", len(sources)))
            sources_used = result.get("sources_used", len(sources))
            self.cache_manager.mongo.save_query_history(
                user_id=user_id,
                query_hash=query_hash,
                original_query=original_query,
                normalized_query=normalized_query,
                intent=classification.get("intent", "concept_explanation"),
                scope=classification.get("scope", ""),
                answer=result.get("answer", ""),
                citations=sources,
                sources_analyzed=sources_analyzed,
                sources_used=sources_used,
                cache_status=cache_status,
                is_cacheable=True,
                metadata={"timings": result.get("timings", {}), "research_depth": res_depth},
                chat_id=chat_id,
                resolved_query=resolved_query or original_query,
                research_depth=res_depth
            )
        except Exception as e:
            logger.warning(f"[History] Failed to record MongoDB query history: {e}")


def run_app_cli() -> None:
    """Provides a simple interactive command-line interface for the RAG assistant."""
    print("=" * 60)
    print(" WEB-GROUNDED LLM ENGINEERING RAG ASSISTANT CLI ")
    print("=" * 60)
    
    pipeline = WebGroundedRAGPipeline()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nProcessing single-shot command line query: '{query}'")
        execute_and_display(pipeline, query)
        return
        
    while True:
        try:
            print("\nEnter an engineering educational topic (or type 'exit' / 'quit' to close):")
            query = input("> ").strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting RAG CLI Assistant. Goodbye!")
                break
                
            execute_and_display(pipeline, query)
            
        except KeyboardInterrupt:
            print("\nExiting RAG CLI Assistant. Goodbye!")
            break

def execute_and_display(pipeline: WebGroundedRAGPipeline, query: str) -> None:
    """Executes the pipeline and prints formatted final output, timing, and statistics."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print(f"\nRunning grounding pipeline for topic: '{query}'...")
    
    result = pipeline.run_pipeline(query)
    
    if result["success"]:
        print("\n" + "=" * 60)
        print(" GENERATED EDUCATION NOTES ")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)
        
        # Display Sources Cited
        print("\n--- SOURCES CITED ---")
        if result["sources"]:
            for i, src in enumerate(result["sources"], 1):
                print(f"[{i}] {src['title']}")
                print(f"    Link: {src['url']}")
        else:
            print("No sources could be extracted.")
        print("-" * 60)
        
        # Display Website Loading Performance Report
        print("\n" + "=" * 60)
        print(" WEBSITE LOADING PERFORMANCE REPORT ")
        print("=" * 60)
        url_metrics = result.get("url_metrics", [])
        if url_metrics:
            for i, u_meta in enumerate(url_metrics, 1):
                status_symbol = "✓" if u_meta["status"] == "Success" else "✗"
                print(f"  {status_symbol} URL {i}: {u_meta['duration']:.2f}s | {u_meta['method']} | {u_meta['status']} ({u_meta['chars']} chars)")
                print(f"       {u_meta['url']}")
        else:
            print("  No URL loading metrics recorded.")
        print("-" * 60)
        
        # Display Token Usage Report
        token_usage = result.get("token_usage", {})
        print("\n" + "=" * 60)
        print(" LLM TOKEN USAGE REPORT ")
        print("=" * 60)
        print(f"  Input Tokens  : {token_usage.get('input_tokens', 0)}")
        print(f"  Output Tokens : {token_usage.get('output_tokens', 0)}")
        print(f"  Total Tokens  : {token_usage.get('total_tokens', 0)}")
        print("-" * 60)
        
        # Display Comprehensive Performance Report
        print("\n" + "=" * 60)
        print(" RAG PERFORMANCE REPORT ")
        print("=" * 60)
        print(f"Query: {query}\n")
        timings = result.get("timings", {})
        for phase, t_val in timings.items():
            if phase != "TOTAL RESPONSE TIME":
                print(f"  {phase:<30}: {t_val:.2f} sec")
        print("-" * 60)
        total_t = timings.get("TOTAL RESPONSE TIME", 0.0)
        print(f"  TOTAL RESPONSE TIME           : {total_t:.2f} sec")
        print("=" * 60 + "\n")

    else:
        print_failure(f"Generation failed: {result['error']}")

if __name__ == "__main__":
    run_app_cli()
