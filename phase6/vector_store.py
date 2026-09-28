"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 6 - Vector Storage module using ChromaDB EphemeralClient to store embeddings in-memory.
Dependencies: chromadb, utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.models.Collection import Collection

# Package imports
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
logger = setup_logger("phase6_vector_store")

class VectorStoreManager:
    """
    Manages a transient (in-memory) ChromaDB vector storage instance.
    Handles collection creation, document insertion with metadata, and collection deletion.
    """

    def __init__(self) -> None:
        """Initializes the Ephemeral ChromaDB client (stateless in-memory)."""
        logger.debug("Initializing ChromaDB EphemeralClient...")
        try:
            self.client = chromadb.EphemeralClient()
            self.collection: Optional[Collection] = None
            self.collection_name: Optional[str] = None
            logger.info("ChromaDB EphemeralClient initialized successfully (in-memory).")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB EphemeralClient: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    def create_collection(self, name: str) -> Collection:
        """
        Creates a new collection with the given name. If it already exists, retrieves it.
        
        Args:
            name (str): Collection name.
            
        Returns:
            Collection: ChromaDB Collection object.
        """
        self.collection_name = name
        logger.info(f"Creating ChromaDB collection: '{name}'")
        print_loading(f"Creating temporary collection '{name}'...")
        
        try:
            # Ephemeral client creates/gets the collection
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity as requested by IR guidelines
            )
            logger.debug(f"Collection '{name}' ready.")
            return self.collection
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise RuntimeError(f"Collection creation failed: {e}") from e

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Adds embedded chunks to the active collection.
        
        Args:
            chunks (List[Dict[str, Any]]): Embedded chunks from Phase 5.
        """
        if not self.collection:
            raise RuntimeError("No active collection. Call create_collection() first.")
            
        if not chunks:
            logger.warning("Empty chunks list passed. Nothing to add.")
            return
            
        logger.info(f"Adding {len(chunks)} embedded chunks to ChromaDB collection '{self.collection_name}'.")
        print_processing(f"Ingesting {len(chunks)} chunks into vector store...")
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{idx}")
            text = chunk.get("text", "")
            embedding = chunk.get("embedding", [])
            meta = chunk.get("metadata", {})
            
            if not embedding:
                logger.warning(f"Chunk ID '{chunk_id}' has no embedding vector. Skipping.")
                continue
                
            ids.append(chunk_id)
            documents.append(text)
            embeddings.append(embedding)
            
            # Map clean scalar metadata values (Chroma requires string/int/float types)
            metadatas.append({
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "chunk_number": meta.get("chunk_number", 0),
                "document_id": chunk_id
            })
            
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(ids)} documents to collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{self.collection_name}': {e}")
            raise RuntimeError(f"ChromaDB ingestion failed: {e}") from e

    def delete_collection(self) -> None:
        """Explicitly deletes the current collection from the ephemeral database client."""
        if not self.collection_name:
            logger.warning("No collection is active to delete.")
            return
            
        logger.info(f"Deleting ChromaDB collection: '{self.collection_name}'")
        print_loading(f"Cleaning up and deleting collection '{self.collection_name}'...")
        
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Successfully deleted collection '{self.collection_name}'.")
            self.collection = None
            self.collection_name = None
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"ChromaDB deletion failed: {e}") from e

def run_phase6(chunks: List[Dict[str, Any]]) -> VectorStoreManager:
    """
    Orchestrates the Phase 6 vector store creation and ingestion.
    
    Args:
        chunks (List[Dict[str, Any]]): Embedded chunks from Phase 5.
        
    Returns:
        VectorStoreManager: The manager containing the loaded ephemeral collection.
    """
    print_phase_header("Phase 6: Vector Storage")
    
    if not chunks:
        print_failure("No embedded chunks provided for ingestion.")
        sys.exit(1)
        
    manager = VectorStoreManager()
    
    # We use a context/timer for ingestion
    with PhaseTimer("Phase 6: Vector Storage") as timer:
        # Create temp collection name
        temp_col_name = "temp_educational_rag_store"
        manager.create_collection(temp_col_name)
        manager.add_chunks(chunks)
        
    # Print Statistics
    stats = {
        "Collection Name": temp_col_name,
        "Storage Mode": "Ephemeral (In-Memory)",
        "Documents Ingested": len(chunks),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Demonstration of auto-deletion fallback / CLI verification
    # For CLI test execution, we delete the collection immediately to keep it clean.
    # In the integrated pipeline, we delete the collection at the very end in Phase 10.
    if __name__ == "__main__":
        print("Test execution finished. Explicitly deleting collection now...")
        manager.delete_collection()
        
    return manager

if __name__ == "__main__":
    # Test execution using mock embedded chunks (size 384)
    import random
    
    # Generate mock 384 dimensional vector
    def gen_mock_vector(dim: int = 384) -> List[float]:
        return [random.uniform(-0.1, 0.1) for _ in range(dim)]
        
    mock_embedded_chunks = [
        {
            "id": "https://example.com/deadlock#chunk-1",
            "text": "Deadlock occurs when processes are blocked waiting for resources held by each other.",
            "embedding": gen_mock_vector(),
            "metadata": {"url": "https://example.com/deadlock", "title": "Deadlock wiki", "chunk_number": 1}
        },
        {
            "id": "https://example.com/deadlock#chunk-2",
            "text": "The four conditions of deadlock are hold-and-wait, circular-wait, mutual-exclusion, no-preemption.",
            "embedding": gen_mock_vector(),
            "metadata": {"url": "https://example.com/deadlock", "title": "Deadlock wiki", "chunk_number": 2}
        }
    ]
    
    try:
        run_phase6(mock_embedded_chunks)
    except Exception as exc:
        print_failure(f"Phase 6 execution failed: {exc}")
        sys.exit(1)
