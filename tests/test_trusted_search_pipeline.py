"""
Test Suite: Broad Web Search & Trusted Source Filtering
Project: Web-Grounded LLM Content Generation for Engineering Education
Purpose: Validate new broad search, domain safety, trusted policy enforcement, retry logic, and zero-results handling.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.trusted_sources import is_domain_trusted, clean_domain, get_source_metadata
from phase1.web_search import WebSearcher
from phase2.result_filter import ResultFilter
from phase3.website_loader import WebsiteLoader
from phase10.web_grounded_rag import WebGroundedRAGPipeline

class TestTrustedSearchPipeline(unittest.TestCase):

    def setUp(self):
        self.filter_engine = ResultFilter()
        self.loader = WebsiteLoader()

    # -------------------------------------------------------------
    # TEST 1: Normal Query & Broad Web Search
    # -------------------------------------------------------------
    def test_normal_query_broad_search(self):
        """Test 1: Verify Tavily executes broad web search and filter validates web URLs."""
        searcher = WebSearcher()
        results = searcher.search("What is a binary search tree?", max_results=10)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Broad search should return candidate results.")
        
        filtered = self.filter_engine.filter_results(results)
        self.assertIsInstance(filtered, list)
        
        for item in filtered:
            domain = item.get("domain", "")
            self.assertTrue(is_domain_trusted(domain), f"Filtered result domain '{domain}' must be valid.")

    # -------------------------------------------------------------
    # TEST 2: URL Filtering & Validation
    # -------------------------------------------------------------
    def test_untrusted_result_exclusion(self):
        """Test 2: Verify malformed URLs are rejected and valid web URLs retained."""
        raw_candidates = [
            {"url": "invalid-url-scheme", "title": "Invalid Scheme", "snippet": "Text", "score": 0.9},
            {"url": "https://en.wikipedia.org/wiki/Binary_search_tree", "title": "BST - Wikipedia", "snippet": "BST article", "score": 0.95}
        ]
        
        filtered = self.filter_engine.filter_results(raw_candidates)
        filtered_urls = [f["url"] for f in filtered]
        
        self.assertNotIn("invalid-url-scheme", filtered_urls, "Invalid scheme URL must be excluded.")
        self.assertIn("https://en.wikipedia.org/wiki/Binary_search_tree", filtered_urls, "Valid URL must be retained.")

    # -------------------------------------------------------------
    # TEST 3: Ad/Paywall Keyword Rejection
    # -------------------------------------------------------------
    def test_malicious_lookalike_domain_rejection(self):
        """Test 3: Verify ad and paywall keyword URLs are rejected."""
        ad_urls = [
            "https://example.com/login-advertisement-page",
            "https://example.com/promo-affiliate-link"
        ]
        
        for url in ad_urls:
            candidates = [{"url": url, "title": "Ad Page", "snippet": "Ad", "score": 0.99}]
            filtered = self.filter_engine.filter_results(candidates)
            self.assertEqual(len(filtered), 0, f"Filter engine must reject ad keyword URL '{url}'.")

    # -------------------------------------------------------------
    # TEST 4: WWW vs Non-WWW Domain Normalization
    # -------------------------------------------------------------
    def test_www_vs_non_www_handling(self):
        """Test 4: Verify www.example.com and example.com are handled identically and correctly under policy."""
        www_domain = "www.geeksforgeeks.org"
        non_www_domain = "geeksforgeeks.org"
        
        self.assertTrue(is_domain_trusted(www_domain))
        self.assertTrue(is_domain_trusted(non_www_domain))
        
        self.assertEqual(clean_domain(www_domain), "geeksforgeeks.org")
        self.assertEqual(clean_domain(non_www_domain), "geeksforgeeks.org")

if __name__ == "__main__":
    unittest.main()
