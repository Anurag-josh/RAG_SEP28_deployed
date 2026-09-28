"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 5 - Embedding module using SentenceTransformers to generate 384-dimensional dense vectors.
Dependencies: sentence_transformers, torch, utils.logger, utils.helper
"""

import sys
import time
from typing import List, Dict, Any
from google import genai

# Package imports
from config.config import GEMINI_API_KEY
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
logger = setup_logger("phase5_embedder")

# Global singleton cache for local SentenceTransformer model
_LOCAL_MODEL_CACHE = {}

class ChunkEmbedder:
    """
    Generates dense vector embeddings using SentenceTransformer (all-MiniLM-L6-v2) or Gemini API with batching.
    Model instances are cached to prevent redundant initialization.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_key: str = GEMINI_API_KEY) -> None:
        """
        Initializes the ChunkEmbedder.
        
        Args:
            model_name (str): Model name identifier.
            api_key (str): Optional API key for remote embedding service.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self.local_model = None
        self.device = "cpu"
        
        # Check if local model is in global cache first
        if self.model_name in _LOCAL_MODEL_CACHE:
            self.local_model = _LOCAL_MODEL_CACHE[self.model_name]
            logger.debug(f"Reusing cached local SentenceTransformer model: {self.model_name}")
        else:
            print_loading(f"Initializing local SentenceTransformer ('{self.model_name}') model...")
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(self.model_name)
                _LOCAL_MODEL_CACHE[self.model_name] = self.local_model
                logger.info(f"Local SentenceTransformer model ('{self.model_name}') loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load local SentenceTransformer: {e}. Trying Gemini API fallback...")
                if self.api_key:
                    self.device = "api"
                    self.client = genai.Client(api_key=self.api_key)

    def get_embedding_dimension(self) -> int:
        """Returns the output vector dimension size dynamically based on model selection."""
        if self.local_model is not None:
            return 384
        return 3072

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings using batch processing.
        
        Args:
            texts (List[str]): List of texts to embed.
            
        Returns:
            List[List[float]]: List of float vector embeddings.
        """
        if not texts:
            return []
            
        if self.local_model is not None:
            logger.debug(f"Generating batch embeddings for {len(texts)} texts via local SentenceTransformer.")
            embeddings = self.local_model.encode(texts, batch_size=32, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
            
        logger.debug(f"Generating embeddings for {len(texts)} text inputs via Gemini API in parallel.")
        
        import concurrent.futures
        import time
        
        def get_single_embedding(text: str) -> List[float]:
            retries = 3
            backoff = 2.0
            for attempt in range(retries):
                try:
                    response = self.client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=text
                    )
                    return response.embeddings[0].values
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise e
            raise RuntimeError("Gemini embedding failed after maximum retries.")
            
        try:
            max_workers = min(5, len(texts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                embeddings = list(executor.map(get_single_embedding, texts))
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings via API: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embeds the text field in each chunk dictionary and adds 'embedding' key in-place.
        
        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries containing 'text'.
            
        Returns:
            List[Dict[str, Any]]: Original list of chunks updated with embedding vectors.
        """
        if not chunks:
            return []
            
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            
        return chunks

def run_phase5(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrates the Phase 5 embedding generation.
    
    Args:
        chunks (List[Dict[str, Any]]): List of chunks from Phase 4.
        
    Returns:
        List[Dict[str, Any]]: The chunks updated with embedding vectors.
    """
    print_phase_header("Phase 5: Embedding")
    
    if not chunks:
        print_failure("No chunks provided to embed.")
        return []
        
    embedder = ChunkEmbedder()
    
    print_processing(f"Generating vector representations for {len(chunks)} text chunks...")
    
    start_time = time.perf_counter()
    with PhaseTimer("Phase 5: Embedding") as timer:
        embedded_chunks = embedder.embed_chunks(chunks)
    elapsed_time = time.perf_counter() - start_time
    
    # Retrieve embedding dimension
    dimension = embedder.get_embedding_dimension()
    
    # Statistics
    stats = {
        "Embedding Model": embedder.model_name,
        "Device Utilized": embedder.device,
        "Embedding Dimension": dimension,
        "Total Chunks Embedded": len(embedded_chunks),
        "Total Embedding Time": f"{elapsed_time:.3f}s",
        "Average Chunk Embedding Time": f"{(elapsed_time/len(chunks)):.4f}s" if chunks else "N/A",
        "Execution Status": "Success"
    }
    print_statistics(stats)
    
    return embedded_chunks

if __name__ == "__main__":
    # Test with dummy chunks
    mock_chunks = [
        {
            "id": "https://example.com/deadlock#chunk-1",
            "text": "In concurrent computing, a deadlock is a state where processes wait on each other.",
            "metadata": {"url": "https://example.com/deadlock", "title": "Deadlock Wiki"}
        },
        {
            "id": "https://example.com/deadlock#chunk-2",
            "text": "Deadlock prevention algorithms work by ensuring that one of the four conditions is eliminated.",
            "metadata": {"url": "https://example.com/deadlock", "title": "Deadlock Wiki"}
        }
    ]
    
    try:
        run_phase5(mock_chunks)
    except Exception as exc:
        print_failure(f"Phase 5 execution failed: {exc}")
        sys.exit(1)
