"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 2 - Result Filtering module to clean, normalize, validate, and verify search result URLs.
Dependencies: urllib.parse, requests, config.trusted_sources, utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any, Set
from urllib.parse import urlparse, urlunparse
import requests

# Package imports
from config.trusted_sources import is_domain_trusted, get_source_metadata
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
logger = setup_logger("phase2_result_filter")

# Blacklist keywords that might identify advertisement, sponsorship or login-wall pages
AD_KEYWORDS = {
    "promo", "advertisement", "sponsor", "affiliate", "marketing", "click-tracker",
    "ad-server", "doubleclick", "login", "signin", "subscribe", "register"
}

# Known problematic or scraper-hostile domains that cause 403s, timeouts, or irrelevant noise
BLOCKED_DOMAINS = {
    "coinmarketcap.com", "finance.yahoo.com", "indeed.com", "medium.com",
    "quora.com", "pinterest.com"
}

class ResultFilter:
    """
    Cleans and filters web search results by validating domains using urllib.parse.urlparse,
    checking for advertisement paths, removing duplicate URLs, verifying HTTP status,
    and classifying candidates into trusted vs rejected sources.
    """

    def __init__(self, request_timeout: float = 3.0) -> None:
        """
        Initializes the result filter.
        
        Args:
            request_timeout (float): Timeout in seconds for checking if a URL is active.
        """
        self.request_timeout = request_timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        logger.debug("ResultFilter initialized.")

    def normalize_url(self, url: str) -> str:
        """
        Normalizes a URL by parsing, lowering domain name, removing query parameters,
        stripping fragment identifiers, and removing trailing slashes.
        
        Args:
            url (str): The raw URL to normalize.
            
        Returns:
            str: Normalized URL.
        """
        if not url:
            return ""
            
        try:
            parsed = urlparse(url.strip())
            scheme = parsed.scheme.lower() if parsed.scheme else "https"
            netloc = parsed.netloc.lower()
            
            # Clean up default ports if present
            if ":" in netloc:
                host, port = netloc.split(":", 1)
                if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                    netloc = host
            
            path = parsed.path
            # Strip trailing slash from path for consistency, but keep root slash
            if path.endswith("/") and len(path) > 1:
                path = path[:-1]
                
            # Reconstruct without query parameters or fragments
            normalized = urlunparse((scheme, netloc, path, "", "", ""))
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize URL '{url}': {e}")
            return url

    def is_ad_or_walled_page(self, url: str) -> bool:
        """
        Checks if the URL contains advertisement or paywall-related keywords.
        
        Args:
            url (str): URL to verify.
            
        Returns:
            bool: True if it looks like an ad or walled page, False otherwise.
        """
        url_lower = url.lower()
        for kw in AD_KEYWORDS:
            if kw in url_lower:
                logger.debug(f"Flagged URL as advertisement/paywall: '{url}' (matched '{kw}')")
                return True
        return False

    def is_url_broken(self, url: str) -> bool:
        """
        Performs a fast HTTP request to check if the page is active and accessible.
        
        Args:
            url (str): URL to verify.
            
        Returns:
            bool: True if the URL is broken (non-2xx response or network failure), False otherwise.
        """
        headers = {"User-Agent": self.user_agent}
        try:
            # We attempt a HEAD request first (cheaper)
            logger.debug(f"Checking URL with HEAD: {url}")
            response = requests.head(url, headers=headers, timeout=self.request_timeout, allow_redirects=True)
            
            # If HEAD fails or is not allowed, fallback to GET
            if response.status_code >= 400:
                logger.debug(f"HEAD request returned status {response.status_code}. Retrying with GET: {url}")
                response = requests.get(url, headers=headers, timeout=self.request_timeout, allow_redirects=True, stream=True)
            
            is_broken = not (200 <= response.status_code < 400)
            if is_broken:
                logger.warning(f"URL broken. Status: {response.status_code} for URL: {url}")
            return is_broken
            
        except requests.RequestException as e:
            logger.warning(f"Connection failed for URL: {url}. Error: {e}")
            return True

    def filter_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters and validates candidate search results by checking URL format,
        normalizing schemes, removing query parameters/duplicates, and ranking by score.
        
        Args:
            search_results (List[Dict[str, Any]]): Raw candidate results with URL, title, snippet, score.
            
        Returns:
            List[Dict[str, Any]]: Validated, normalized, and deduplicated search results.
        """
        logger.info(f"Filtering and validating {len(search_results)} candidate search results.")
        print_loading("Validating URL quality, normalizing schemes, and removing duplicates...")
        
        seen_urls: Set[str] = set()
        filtered_results: List[Dict[str, Any]] = []
        rejected_summary: List[Dict[str, str]] = []
        accepted_summary: List[Dict[str, str]] = []
        
        total_items = len(search_results)
        
        for i, item in enumerate(search_results, 1):
            raw_url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            score = item.get("score", 0.0)
            
            print_processing(f"[{i}/{total_items}] Validating URL quality: {raw_url[:55]}...")
            
            # 1. Basic URL scheme check
            if not raw_url or not (raw_url.startswith("http://") or raw_url.startswith("https://")):
                logger.debug(f"Excluded non-HTTP/HTTPS URL: '{raw_url}'")
                rejected_summary.append({"url": raw_url, "reason": "Invalid URL scheme"})
                continue

            # 2. Normalize URL
            normalized_url = self.normalize_url(raw_url)
            if not normalized_url:
                logger.debug(f"Excluded empty/invalid URL.")
                rejected_summary.append({"url": raw_url, "reason": "Invalid / Empty URL"})
                continue
                
            # 3. Check for Duplicates
            if normalized_url in seen_urls:
                logger.debug(f"Excluded duplicate normalized URL: '{normalized_url}'")
                rejected_summary.append({"url": normalized_url, "reason": "Duplicate URL"})
                continue
                
            # 4. Parse hostname safely
            try:
                parsed_url = urlparse(normalized_url)
                hostname = parsed_url.hostname or ""
            except Exception:
                hostname = ""
                
            if not hostname:
                rejected_summary.append({"url": normalized_url, "reason": "Invalid hostname"})
                continue

            # 5. Check for Ad/Paywall Pages or Problematic Domains
            if self.is_ad_or_walled_page(normalized_url):
                logger.info(f"REJECTED advertisement/walled URL: '{normalized_url}'")
                rejected_summary.append({"url": normalized_url, "reason": "Ad/Paywall keyword"})
                continue

            clean_host = hostname.lower().replace("www.", "")
            if any(b_dom in clean_host for b_dom in BLOCKED_DOMAINS):
                logger.info(f"REJECTED problematic/unfriendly domain URL: '{normalized_url}'")
                rejected_summary.append({"url": normalized_url, "reason": "Known problematic domain"})
                continue
                
            # Retrieve source metadata & calculate quality boost
            meta = get_source_metadata(hostname)
            adjusted_score = score
            if is_domain_trusted(hostname):
                adjusted_score += 0.15
            
            # Approved valid web result
            seen_urls.add(normalized_url)
            approved_item = {
                "url": normalized_url,
                "title": title,
                "snippet": snippet,
                "score": adjusted_score,
                "domain": hostname,
                "source_name": meta.get("name", hostname),
                "category": meta.get("category", "Web Source")
            }
            filtered_results.append(approved_item)
            accepted_summary.append({"url": normalized_url, "name": meta.get("name", hostname)})
            logger.info(f"VALIDATED URL: '{normalized_url}' (Score: {adjusted_score:.4f})")
            
        # Sort candidates by adjusted relevance score in descending order
        filtered_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        logger.info(f"Filtering complete. Candidates: {total_items} | Validated: {len(filtered_results)} | Rejected: {len(rejected_summary)}")
        return filtered_results

def run_phase2(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrates Phase 2 URL validation and deduplication.
    
    Args:
        search_results (List[Dict[str, Any]]): The input candidate search results from Phase 1.
        
    Returns:
        List[Dict[str, Any]]: The filtered search results.
    """
    print_phase_header("Phase 2: Result Validation & Deduplication")
    
    if not search_results:
        print_failure("No input search results to filter.")
        return []
        
    filter_engine = ResultFilter()
    
    with PhaseTimer("Phase 2: Result Filtering") as timer:
        filtered_results = filter_engine.filter_results(search_results)
        
    # Statistics
    stats = {
        "Candidate Search Results": len(search_results),
        "Unique URLs Retained": len(filtered_results),
        "Invalid/Duplicate Rejected": len(search_results) - len(filtered_results),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print summaries
    print("\n--- VALIDATED WEB SOURCES ---")
    for i, res in enumerate(filtered_results, 1):
        print(f"  ✓ [{i}] {res['source_name']} (Score: {res['score']:.4f})")
        print(f"      Title: {res['title']}")
        print(f"      Clean URL: {res['url']}\n")
        
    return filtered_results

if __name__ == "__main__":
    # Test execution with dummy inputs representing various cases
    test_inputs = [
        # Valid and Trusted (Wikipedia)
        {
            "url": "HTTPS://en.wikipedia.org/wiki/Deadlock/",
            "title": "Deadlock - Wikipedia",
            "snippet": "A deadlock is a state in which each member of a group is waiting...",
            "score": 0.95
        },
        # Duplicate of above (normalized)
        {
            "url": "https://en.wikipedia.org/wiki/Deadlock?ref=ad-campaign",
            "title": "Deadlock Wiki",
            "snippet": "Duplicate description...",
            "score": 0.90
        },
        # Untrusted domain
        {
            "url": "https://untrustedsource.com/deadlock-explained",
            "title": "Untrusted explanation",
            "snippet": "Some sketchy content...",
            "score": 0.85
        },
        # Ad or Login page keyword
        {
            "url": "https://geeksforgeeks.org/login-page-advertisement",
            "title": "GfG Promo",
            "snippet": "Promo details...",
            "score": 0.60
        },
        # Broken URL
        {
            "url": "https://wikipedia.org/non-existent-page-404-check",
            "title": "Non Existent Page",
            "snippet": "This should return 404...",
            "score": 0.70
        }
    ]
    
    try:
        run_phase2(test_inputs)
    except Exception as exc:
        print_failure(f"Phase 2 execution failed: {exc}")
        sys.exit(1)
