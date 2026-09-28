"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 3 - Website Loader module to fetch webpage content and parse it to clean Markdown.
         Page cache backed by SQLite via CacheManager (replaces legacy scraped_pages.json).
Dependencies: requests, docling, trafilatura, utils.logger, utils.helper, cache.cache_manager
"""

import sys
import json
import random
import concurrent.futures
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
import trafilatura

# Package imports
from config.config import BASE_DIR, MAX_CONCURRENT_REQUESTS
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
logger = setup_logger("phase3_website_loader")

# List of common User-Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

class WebsiteLoader:
    """
    Downloads and extracts clean Markdown content from webpage URLs using fast parallel processing.
    Uses Trafilatura as the primary lightweight webpage parser for high speed,
    with fallbacks to BeautifulSoup and Docling.
    Page cache is backed by SQLite via CacheManager (replaces legacy scraped_pages.json).
    The cache_manager is optional — if None, no page caching is performed.
    """

    def __init__(self, cache_manager=None) -> None:
        """
        Initializes the WebsiteLoader.

        Args:
            cache_manager: Optional CacheManager instance for SQLite page caching.
                           If None, every URL is fetched fresh (no caching).
        """
        self.cache_manager = cache_manager
        self.converter = None
        self.docling_available = None
        self.request_timeout = 5.0
        if self.cache_manager:
            logger.debug("WebsiteLoader: using SQLite page cache via CacheManager.")
        else:
            logger.debug("WebsiteLoader: no CacheManager provided — page cache disabled.")

    def _get_docling_converter(self):
        """Lazy-loads Docling DocumentConverter only when explicitly needed for PDF/documents."""
        if self.docling_available is None:
            try:
                from docling.document_converter import DocumentConverter
                self.converter = DocumentConverter()
                self.docling_available = True
            except Exception as e:
                logger.warning(f"Docling converter not available: {e}")
                self.docling_available = False
                self.converter = None
        return self.converter

    def load_with_docling(self, url: str) -> str:
        """Attempts to parse a PDF or document file directly using Docling."""
        converter = self._get_docling_converter()
        if not converter:
            raise RuntimeError("Docling is not available.")
            
        logger.info(f"Attempting Docling download/parse for document URL: {url}")
        result = converter.convert(url)
        markdown = result.document.export_to_markdown()
        
        if not markdown or not markdown.strip():
            raise ValueError("Docling conversion produced empty output.")
            
        return markdown

    def load_with_trafilatura(self, url: str) -> tuple[str, str]:
        """
        Fast primary parser that fetches webpage HTML and converts it to Markdown.
        Returns tuple of (markdown_content, method_name).
        """
        logger.info(f"Attempting Trafilatura download/parse for URL: {url}")
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        
        response = requests.get(url, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()
        html_content = response.text
        
        markdown = trafilatura.extract(
            html_content,
            output_format="markdown",
            include_comments=False,
            with_metadata=False
        )
        method = "Trafilatura"
        
        if not markdown or not markdown.strip():
            # Fallback to BeautifulSoup basic text extraction
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    element.extract()
                text = soup.get_text(separator="\n\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                markdown = "\n\n".join(lines)
                method = "BeautifulSoup"
            except Exception:
                pass
                
        if not markdown or not markdown.strip():
            raise ValueError("Extraction produced empty output.")
            
        return markdown, method

    def load_url(self, url: str) -> tuple:
        """
        Loads and parses a URL dynamically with strict 5s timeout.
        Checks SQLite page cache before making an HTTP request.
        For HTML pages, uses fast Trafilatura.
        Docling is strictly restricted to PDF/document files.
        """
        if not url:
            raise ValueError("URL string is empty.")

        # Check SQLite page cache via CacheManager
        if self.cache_manager:
            cached = self.cache_manager.get_page(url)
            if cached and cached.get("markdown"):
                logger.info(f"[Cache] Page cache HIT (SQLite) for URL: '{url}'")
                return cached["markdown"], "SQLiteCache"

        is_document = url.lower().endswith(('.pdf', '.docx', '.pptx', '.xlsx'))

        markdown = None
        method = "Failed"

        if is_document:
            try:
                markdown = self.load_with_docling(url)
                method = "Docling"
                logger.info(f"Successfully loaded document '{url}' using Docling.")
            except Exception as de:
                logger.error(f"Failed to load document '{url}' with Docling: {de}")
                raise RuntimeError(f"Docling failed on document URL '{url}': {de}") from de
        else:
            try:
                markdown, method = self.load_with_trafilatura(url)
                logger.info(f"Successfully loaded '{url}' using {method}.")
            except Exception as te:
                logger.warning(f"Fast web extraction failed for '{url}': {te}. Skipping URL.")
                raise te

        # Save to SQLite cache if successful
        if markdown:
            if self.cache_manager:
                try:
                    parsed = urlparse(url)
                    domain = (parsed.hostname or "").replace("www.", "")
                    self.cache_manager.save_page(url, markdown, title=domain, domain=domain)
                    logger.debug(f"[Cache] Page saved to SQLite for URL: '{url}'")
                except Exception as ce:
                    logger.warning(f"[Cache] Failed to save page to SQLite for '{url}': {ce}")
            return markdown, method

        raise RuntimeError(f"Could not load content from '{url}'")

    def load_single_worker(self, url: str) -> Dict[str, Any]:
        """Worker thread function to scrape a single webpage with detailed timing."""
        import time
        start_time = time.perf_counter()
        try:
            content, method = self.load_url(url)
            elapsed = time.perf_counter() - start_time
            return {
                "url": url,
                "content": content,
                "success": True,
                "duration": round(elapsed, 3),
                "method": method,
                "error": None
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            err_msg = str(e)
            return {
                "url": url,
                "content": "",
                "success": False,
                "duration": round(elapsed, 3),
                "method": "Failed",
                "error": err_msg
            }

    def load_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Loads content for multiple URLs concurrently using ThreadPoolExecutor with strict bounded timeouts.
        """
        total = len(urls)
        if total == 0:
            return []

        max_workers = min(MAX_CONCURRENT_REQUESTS, total)
        logger.info(f"Starting concurrent loading for {total} URLs with {max_workers} parallel workers.")
        print_loading(f"Fetching and parsing {total} web pages concurrently (workers={max_workers})...")

        results_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.load_single_worker, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    res = future.result(timeout=self.request_timeout + 2.0)
                    results_map[url] = res
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Extraction timed out for URL: '{url}'")
                    results_map[url] = {
                        "url": url,
                        "content": "",
                        "success": False,
                        "duration": round(self.request_timeout + 2.0, 3),
                        "method": "Failed",
                        "error": "Timeout"
                    }
                except Exception as ex:
                    results_map[url] = {
                        "url": url,
                        "content": "",
                        "success": False,
                        "duration": 0.0,
                        "method": "Failed",
                        "error": str(ex)
                    }

        success_count = sum(1 for res in results_map.values() if res.get("success"))
        timeout_count = sum(1 for res in results_map.values() if res.get("error") == "Timeout" or "timeout" in str(res.get("error", "")).lower())
        failed_count = len(urls) - success_count - timeout_count

        logger.info(f"[EXTRACTION] success={success_count} failed={failed_count} timeout={timeout_count}")

        ordered_results = []
        for i, url in enumerate(urls, 1):
            res = results_map.get(url, {"url": url, "content": "", "success": False, "duration": 0.0, "method": "Failed", "error": "Not run"})
            if res["success"]:
                print_success(f"[{i}/{total}] Loaded successfully via {res['method']} ({res['duration']}s, {len(res['content'])} chars): {url[:60]}...")
            else:
                print_failure(f"[{i}/{total}] Load failed via {res['method']} ({res['duration']}s): {res['error']} for {url[:60]}...")
            ordered_results.append(res)

        return ordered_results

def run_phase3(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Orchestrates the Phase 3 web page loading execution.
    
    Args:
        urls (List[str]): List of URLs to fetch and convert.
        
    Returns:
        List[Dict[str, Any]]: List of results with markdown contents.
    """
    print_phase_header("Phase 3: Website Loader")
    
    if not urls:
        print_failure("No URLs supplied for loading.")
        return []
        
    loader = WebsiteLoader()
    
    with PhaseTimer("Phase 3: Website Loader") as timer:
        loaded_data = loader.load_multiple(urls)
        
    # Analyze results
    success_count = sum(1 for item in loaded_data if item["success"])
    failed_count = len(loaded_data) - success_count
    total_len = sum(len(item["content"]) for item in loaded_data if item["success"])
    avg_len = total_len / success_count if success_count > 0 else 0
    
    # Statistics
    stats = {
        "Total URLS requested": len(urls),
        "Successful Loads": success_count,
        "Failed Loads": failed_count,
        "Average Length (chars)": f"{avg_len:.1f}",
        "Execution Status": "Success" if success_count > 0 else "Failure",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    return loaded_data

if __name__ == "__main__":
    # Test with standard educational page
    test_urls = [
        "https://en.wikipedia.org/wiki/Deadlock"
    ]
    if len(sys.argv) > 1:
        test_urls = sys.argv[1:]
        
    try:
        run_phase3(test_urls)
    except Exception as exc:
        print_failure(f"Phase 3 execution failed: {exc}")
        sys.exit(1)
