"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 1 - Web Search module using Tavily API to fetch educational results from trusted domains.
Dependencies: tavily, config.config, config.trusted_sources, utils.logger, utils.helper
"""

import sys
import time
from typing import List, Dict, Any
from utils.import_timing import timed_import

TavilyClient = timed_import("tavily").TavilyClient
Groq = timed_import("groq").Groq
# Package imports
from config.config import TAVILY_API_KEY, GROQ_API_KEY, GROQ_MODEL_NAME, MAX_SEARCH_RESULTS, validate_config
from utils.logger import setup_logger
from utils.helper import (
    print_phase_header,
    print_loading,
    print_processing,
    print_success,
    print_failure,
    print_statistics,
    PhaseTimer
)

# Initialize logger
logger = setup_logger("phase1_web_search")

class WebSearcher:
    """
    Handles querying the Tavily Search API for broad web discovery.
    Optimizes search queries and reformulates conversation histories into standalone queries using Groq.
    """
    
    def __init__(self, api_key: str = TAVILY_API_KEY, groq_api_key: str = GROQ_API_KEY) -> None:
        """
        Initializes the WebSearcher with the Tavily API key and Groq API key.
        
        Args:
            api_key (str): Tavily API key.
            groq_api_key (str): Groq API key.
        """
        if not api_key:
            raise ValueError("Tavily API key is missing. Ensure TAVILY_API_KEY is defined in your environment or .env file.")
        
        self.client = TavilyClient(api_key=api_key)
        self.groq_api_key = groq_api_key
        logger.debug("TavilyClient initialized successfully.")

    def optimize_query(self, query: str) -> str:
        """
        Uses Groq model to rewrite a conversational query into optimized keywords for web search.
        """
        logger.info(f"Optimizing raw query: '{query}'")
        optimization_started = time.perf_counter()
        try:
            if not hasattr(self, "groq_client"):
                self.groq_client = Groq(api_key=self.groq_api_key, timeout=30.0, max_retries=0)
            
            prompt = (
                "You are an expert search engine query optimizer. Your job is to convert the following raw user query "
                "into a concise, keyword-optimized search query focused on engineering and documentation. "
                "Do NOT include any introduction, explanations, or conversational filler. Return ONLY the search query.\n\n"
                f"User query: {query}\n"
                "Search query:"
            )
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            optimized = response.choices[0].message.content.strip().strip('"')
            logger.info(f"Optimized query: '{optimized}'")
            return optimized
        except Exception as e:
            logger.warning("Groq query optimization failed (%s); using raw query.", type(e).__name__)
            return query
        finally:
            logger.info("[RAG-TIME] Groq query optimization: %.3f sec", time.perf_counter() - optimization_started)

    def reformulate_conversational_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Takes conversation history and a new query, and generates a standalone keyword search query using Groq.
        """
        if not chat_history:
            return self.optimize_query(query)
            
        logger.info("Reformulating conversational query based on chat history.")
        optimization_started = time.perf_counter()
        try:
            if not hasattr(self, "groq_client"):
                self.groq_client = Groq(api_key=self.groq_api_key, timeout=30.0, max_retries=0)
                
            history_text = ""
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
                
            prompt = (
                "Given the following conversation history and a new user message, generate a standalone keyword-based "
                "search query that captures the user's intent. The query will be used to search trusted engineering documentation. "
                "Return ONLY the standalone query, without any other text.\n\n"
                f"Conversation History:\n{history_text}\n"
                f"New Message: {query}\n"
                "Standalone query:"
            )
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            reformulated = response.choices[0].message.content.strip().strip('"')
            logger.info(f"Reformulated query: '{reformulated}'")
            return reformulated
        except Exception as e:
            logger.warning("Groq query reformulation failed (%s); using raw query.", type(e).__name__)
            return query
        finally:
            logger.info("[RAG-TIME] Groq query reformulation: %.3f sec", time.perf_counter() - optimization_started)

    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        """
        Executes a Tavily search across the broader web (without domain site filters).
        
        Args:
            query (str): The search query from the user.
            max_results (int): Maximum number of candidate search results to fetch from the web.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing search results with
                                  keys: 'url', 'title', 'snippet', 'score'.
        
        Raises:
            TavilyException: If the Tavily API returns an error.
            Exception: For other unexpected runtime errors.
        """
        logger.info(f"Initiating broad web search for query: '{query}'")
        print_loading("Connecting to Tavily Search API (Broad Web Search)...")
        print_processing(f"Searching broader web for query: '{query}'...")
        search_started = time.perf_counter()
        try:
            # Execute Tavily Search on the broader web without site: filters
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                timeout=20
            )
            
            results = response.get("results", [])
            logger.info(f"Tavily API returned {len(results)} candidate search results from the broader web.")
            
            # Map Tavily fields to our standard format
            structured_results = []
            for item in results:
                structured_results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0.0)
                })
                
            return structured_results

        except Exception as e:
            logger.error("Tavily search failed (%s).", type(e).__name__)
            raise RuntimeError("Web search failed.") from e
        finally:
            logger.info("[RAG-TIME] Tavily API request: %.3f sec", time.perf_counter() - search_started)

def run_phase1(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Orchestrates the Phase 1 web search execution, handling printing, timing, and errors.
    
    Args:
        query (str): The topic to search for.
        max_results (int): The number of results to fetch.
        
    Returns:
        List[Dict[str, Any]]: The structured search results.
    """
    print_phase_header("Phase 1: Web Search")
    
    try:
        # Validate base config
        validate_config()
    except ValueError as val_err:
        print_failure(f"Configuration validation failed: {val_err}")
        logger.error(f"Configuration validation failed: {val_err}")
        sys.exit(1)
        
    searcher = WebSearcher()
    
    with PhaseTimer("Phase 1: Web Search") as timer:
        results = searcher.search(query, max_results=max_results)
        
    # Print Statistics
    stats = {
        "User Query": query,
        "Results Found": len(results),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print summaries
    for i, res in enumerate(results, 1):
        print(f"[{i}] {res['title']}")
        print(f"    URL:   {res['url']}")
        print(f"    Score: {res['score']:.4f}")
        print(f"    Snippet: {res['snippet'][:100]}...\n")
        
    return results

if __name__ == "__main__":
    # Test execution
    test_query = "Explain CPU scheduling algorithms"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
        
    try:
        run_phase1(test_query)
    except Exception as exc:
        print_failure(f"Phase 1 execution failed: {exc}")
        sys.exit(1)
