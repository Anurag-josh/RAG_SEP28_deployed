"""
Test Suite: Upgraded Broad Web-Grounded RAG Pipeline
Purpose: Validate broad search, URL validation/deduplication, parallel extraction, batch embedding, citations, and performance metrics.
"""

import unittest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import MAX_SEARCH_RESULTS, MAX_SOURCES_TO_SCRAPE
from config.trusted_sources import is_domain_trusted, clean_domain, get_source_metadata
from phase1.web_search import WebSearcher
from phase2.result_filter import ResultFilter
from phase3.website_loader import WebsiteLoader
from phase4.chunker import DocumentChunker
from phase5.embedder import ChunkEmbedder
from phase10.web_grounded_rag import WebGroundedRAGPipeline

class TestUpgradedRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.filter_engine = ResultFilter()
        self.loader = WebsiteLoader()

    def test_website_download_uses_bounded_http_timeouts(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [b"test page"]

        with patch("phase3.website_loader.requests.get", return_value=response) as get:
            content = self.loader._download_bytes(
                "https://example.com/page", total_timeout=2.0, max_bytes=1024, stage="test download"
            )

        self.assertEqual(content, b"test page")
        self.assertEqual(get.call_args.kwargs["timeout"], (3.0, self.loader.request_timeout))
        self.assertTrue(get.call_args.kwargs["stream"])

    def test_website_batch_returns_at_deadline(self):
        self.loader.total_load_timeout = 0.02

        def slow_worker(url):
            time.sleep(0.1)
            return {"url": url, "content": "", "success": False, "duration": 0.1, "error": "test"}

        started = time.perf_counter()
        with patch.object(self.loader, "load_single_worker", side_effect=slow_worker):
            result = self.loader.load_multiple(["https://example.com/slow"])

        self.assertLess(time.perf_counter() - started, 0.08)
        self.assertFalse(result[0]["success"])
        self.assertEqual(result[0]["error"], "Timeout")

    def test_broad_web_search_and_url_validation(self):
        """Test broad web search (no domain whitelist locks) and URL quality filtering."""
        searcher = WebSearcher()
        results = searcher.search("What is a binary search tree?", max_results=MAX_SEARCH_RESULTS)
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Broad search should discover candidate results.")
        self.assertLessEqual(len(results), MAX_SEARCH_RESULTS)
        
        filtered = self.filter_engine.filter_results(results)
        self.assertIsInstance(filtered, list)
        self.assertGreater(len(filtered), 0)
        
        for item in filtered:
            url = item.get("url", "")
            self.assertTrue(url.startswith("http://") or url.startswith("https://"))
            self.assertIn("score", item)

    def test_url_deduplication_and_normalization(self):
        """Test that duplicate URLs with tracking parameters are normalized and deduplicated."""
        raw_candidates = [
            {"url": "https://example.com/page1?utm_source=test", "title": "Page 1", "snippet": "Test", "score": 0.9},
            {"url": "https://example.com/page1/", "title": "Page 1 Duplicate", "snippet": "Test", "score": 0.88},
            {"url": "https://example.com/page2", "title": "Page 2", "snippet": "Test 2", "score": 0.85},
            {"url": "https://example.com/login-ad-page", "title": "Ad Page", "snippet": "Ad", "score": 0.5}
        ]
        
        filtered = self.filter_engine.filter_results(raw_candidates)
        urls = [f["url"] for f in filtered]
        
        self.assertIn("https://example.com/page1", urls)
        self.assertIn("https://example.com/page2", urls)
        self.assertNotIn("https://example.com/login-ad-page", urls, "Ad/paywall keywords must be filtered.")
        self.assertEqual(len(filtered), 2, "Duplicate normalized URLs should be collapsed.")

    def test_batch_embedder_and_model_caching(self):
        """Test that ChunkEmbedder reuses model instance and supports batch embedding."""
        embedder1 = ChunkEmbedder()
        embedder2 = ChunkEmbedder()
        
        self.assertIs(embedder1.local_model, embedder2.local_model, "SentenceTransformer model instance should be cached and reused.")
        
        texts = ["Text sample 1 for vector embedding.", "Text sample 2 for vector embedding."]
        vectors = embedder1.embed_texts(texts)
        
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), embedder1.get_embedding_dimension())

    def test_end_to_end_pipeline_execution(self):
        """Test complete end-to-end RAG pipeline run with citations and timing statistics."""
        pipeline = WebGroundedRAGPipeline()
        pipeline.cache_manager.clear_all_cache()
        result = pipeline.run_pipeline("What is Merge Sort?", top_k=3, max_search_results=10)
        
        self.assertTrue(result["success"])
        self.assertIn("answer", result)
        self.assertGreater(len(result["answer"]), 50)
        self.assertIn("timings", result)
        self.assertIn("statistics", result)
        
        stats = result["statistics"]
        self.assertIn("Discovered Sources", stats)
        self.assertIn("Analyzed Sources", stats)
        self.assertIn("Cited Sources", stats)
        self.assertIn("Total Duration", stats)

if __name__ == "__main__":
    unittest.main()
