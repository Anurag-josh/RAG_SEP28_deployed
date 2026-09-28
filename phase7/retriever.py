"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 7 - Semantic Retrieval module to perform cosine-similarity query on the vector store.
Dependencies: chromadb, phase5.embedder, utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any
from chromadb.api.models.Collection import Collection

# Package imports
from config.config import TOP_K
from phase5.embedder import ChunkEmbedder
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
logger = setup_logger("phase7_retriever")

class SemanticRetriever:
    """
    Performs vector similarity search on a ChromaDB collection using query embeddings.
    """

    def __init__(self, embedder: ChunkEmbedder, collection: Collection) -> None:
        """
        Initializes the SemanticRetriever.
        
        Args:
            embedder (ChunkEmbedder): Instance of Phase 5 embedder to embed queries.
            collection (Collection): Active ChromaDB collection containing ingested chunks.
        """
        self.embedder = embedder
        self.collection = collection
        logger.debug("SemanticRetriever initialized.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Embeds the query, retrieves candidates, and runs keyword overlap reranking for hybrid results.
        
        Args:
            query (str): The search query from the user.
            top_k (int): Number of chunks to finally return.
            
        Returns:
            List[Dict[str, Any]]: Reranked results containing chunk text, score, URL, and title.
        """
        import re
        logger.info(f"Retrieving top {top_k} matches for query: '{query}'")
        print_loading(f"Generating query embedding...")
        
        # 1. Embed query
        query_vector = self.embedder.embed_texts([query])[0]
        
        print_processing(f"Querying vector store (Semantic + Keyword hybrid search)...")
        
        try:
            # Get collection count to cap retrieval candidates
            collection_count = self.collection.count()
            fetch_k = min(max(top_k * 3, 15), max(collection_count, 1))
            logger.debug(f"Fetching {fetch_k} candidate chunks for reranking (from total {collection_count}).")
            
            # 2. Perform Cosine Similarity Search for candidates
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 3. Parse and run keyword-overlap heuristic reranking
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            # Alphanumeric word extraction for query
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "that", "this", "these", "those", "what", "how", "why", "which", "where", "who"}
            query_words = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
            
            candidate_pool = []
            for idx in range(len(ids)):
                doc_text = documents[idx]
                meta = metadatas[idx]
                distance = distances[idx]
                doc_id = ids[idx]
                
                # Cosine Similarity = 1.0 - Cosine Distance
                similarity_score = 1.0 - distance
                
                # Compute term overlap score
                text_words = set(re.findall(r'\b\w+\b', doc_text.lower()))
                overlap = len(query_words.intersection(text_words)) if query_words else 0
                keyword_score = overlap / len(query_words) if query_words else 0.0
                
                # Blend scores: 70% Semantic, 30% Keyword Density
                hybrid_score = 0.7 * similarity_score + 0.3 * keyword_score
                
                candidate_pool.append({
                    "id": doc_id,
                    "text": doc_text,
                    "semantic_score": similarity_score,
                    "keyword_score": keyword_score,
                    "score": hybrid_score,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "chunk_number": meta.get("chunk_number", 0)
                })
                
            # Sort candidate pool by hybrid score in descending order
            candidate_pool.sort(key=lambda x: x["score"], reverse=True)
            
            # Select final top_k chunks
            final_results = candidate_pool[:top_k]
            logger.info(f"Retrieved and reranked {len(final_results)} matching chunks.")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to query vector store: {e}")
            raise RuntimeError(f"Semantic retrieval failed: {e}") from e

def run_phase7(query: str, embedder: ChunkEmbedder, collection: Collection, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Orchestrates the Phase 7 retrieval.
    
    Args:
        query (str): The search query.
        embedder (ChunkEmbedder): The initialized embedder.
        collection (Collection): The active collection.
        top_k (int): Number of results to fetch.
        
    Returns:
        List[Dict[str, Any]]: List of matching chunks with details.
    """
    print_phase_header("Phase 7: Semantic Retrieval")
    
    retriever = SemanticRetriever(embedder, collection)
    
    with PhaseTimer("Phase 7: Semantic Retrieval") as timer:
        results = retriever.retrieve(query, top_k=top_k)
        
    # Statistics
    stats = {
        "Search Query": query,
        "Top-K Requested": top_k,
        "Total Retrieved": len(results),
        "Max Similarity Score": f"{results[0]['score']:.4f}" if results else "N/A",
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Display retrieved chunks
    for i, res in enumerate(results, 1):
        print(f"[{i}] SCORE: {res['score']:.4f} | {res['title']} (Chunk #{res['chunk_number']})")
        print(f"    URL:  {res['url']}")
        print(f"    Text: {res['text'][:300]}...\n" + "-" * 60)
        
    return results

if __name__ == "__main__":
    # Test execution integrating Phase 5, Phase 6, and Phase 7
    from phase6.vector_store import VectorStoreManager
    
    # 1. Initialize embedder
    test_embedder = ChunkEmbedder()
    
    # 2. Define text chunks
    mock_chunks = [
        {
            "id": "https://wikipedia.org/wiki/Deadlock#chunk-1",
            "text": (
                "A deadlock is a state in concurrent computing where a set of processes are blocked "
                "because each process is holding a resource and waiting for another resource held by "
                "some other process. It is a common problem in multiprocessing systems."
            ),
            "metadata": {"url": "https://wikipedia.org/wiki/Deadlock", "title": "Deadlock - Wikipedia", "chunk_number": 1}
        },
        {
            "id": "https://wikipedia.org/wiki/Deadlock#chunk-2",
            "text": (
                "The four necessary conditions for deadlock are: 1. Mutual Exclusion (non-shareable resources), "
                "2. Hold and Wait (processes holding resources while waiting for new ones), 3. No Preemption "
                "(resources cannot be taken away), and 4. Circular Wait (a circular chain of waiting)."
            ),
            "metadata": {"url": "https://wikipedia.org/wiki/Deadlock", "title": "Deadlock - Wikipedia", "chunk_number": 2}
        },
        {
            "id": "https://geeksforgeeks.org/cpu-scheduling#chunk-1",
            "text": (
                "CPU Scheduling is a process that allows one process to use the CPU while another process "
                "is in execution (waiting state) due to unavailability of any resource. Algorithms include "
                "First-Come First-Served (FCFS), Shortest Job First (SJF), Round Robin (RR), and Priority Scheduling."
            ),
            "metadata": {"url": "https://geeksforgeeks.org/cpu-scheduling", "title": "CPU Scheduling - GeeksforGeeks", "chunk_number": 1}
        }
    ]
    
    # 3. Generate embeddings for mock chunks
    test_chunks = test_embedder.embed_chunks(mock_chunks)
    
    # 4. Ingest into Vector Store
    db_manager = VectorStoreManager()
    col = db_manager.create_collection("test_retrieval_collection")
    db_manager.add_chunks(test_chunks)
    
    # 5. Run Semantic Retrieval test
    test_query = "What conditions are required for a deadlock to happen?"
    try:
        run_phase7(test_query, test_embedder, col, top_k=2)
    except Exception as exc:
        print_failure(f"Phase 7 execution failed: {exc}")
    finally:
        # Cleanup collection
        db_manager.delete_collection()
