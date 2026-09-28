"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 4 - Chunker module to split extracted markdown content into semantic overlapping text chunks.
Dependencies: langchain_text_splitters, utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Package imports
from config.config import CHUNK_SIZE, CHUNK_OVERLAP
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
logger = setup_logger("phase4_chunker")

class DocumentChunker:
    """
    Splits long Markdown content from loaded webpages into semantic chunks of target sizes.
    Preserves document structure where possible (headings, lists, code paragraphs).
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        """
        Initializes the DocumentChunker.
        
        Args:
            chunk_size (int): Max character length of a single chunk.
            chunk_overlap (int): Number of characters to overlap between adjacent chunks.
        """
        if chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 characters.")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use standard separators to split content semantically: headings, code blocks, double newlines, etc.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n# ", "\n## ", "\n### ", "\n```", "\n\n", "\n", " ", ""]
        )
        logger.debug(f"DocumentChunker initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def chunk_documents(self, loaded_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits multiple documents into standard chunks.
        
        Args:
            loaded_documents (List[Dict[str, Any]]): List of dicts with 'url', 'content', and optionally 'title'.
            
        Returns:
            List[Dict[str, Any]]: Chunks formatted as dicts containing 'text' and 'metadata'.
        """
        logger.info(f"Chunking {len(loaded_documents)} loaded documents.")
        print_loading(f"Splitting {len(loaded_documents)} pages into semantic chunks...")
        
        chunks = []
        total_docs = len(loaded_documents)
        
        for idx, doc in enumerate(loaded_documents, 1):
            url = doc.get("url", "")
            title = doc.get("title", f"Document {idx}")
            content = doc.get("content", "")
            
            if not content.strip():
                logger.warning(f"Skipping empty document content for URL: {url}")
                continue
                
            print_processing(f"[{idx}/{total_docs}] Chunking: {title[:50]} ({len(content)} chars)...")
            
            # Split the document text
            split_texts = self.splitter.split_text(content)
            logger.debug(f"Document at '{url}' split into {len(split_texts)} chunks.")
            
            for chunk_idx, text in enumerate(split_texts, 1):
                # Unique ID format: url_chunkIndex
                doc_id = f"{url}#chunk-{chunk_idx}"
                
                chunks.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "url": url,
                        "title": title,
                        "chunk_number": chunk_idx,
                        "total_chunks": len(split_texts)
                    }
                })
                
        return chunks

def run_phase4(loaded_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrates the Phase 4 chunking execution.
    
    Args:
        loaded_documents (List[Dict[str, Any]]): The input loaded documents from Phase 3.
        
    Returns:
        List[Dict[str, Any]]: The split text chunks with metadata.
    """
    print_phase_header("Phase 4: Chunking")
    
    # Filter out unsuccessful loads if they are passed in from the pipeline
    valid_docs = [doc for doc in loaded_documents if doc.get("success", True) and doc.get("content")]
    
    if not valid_docs:
        print_failure("No valid document content to chunk.")
        return []
        
    chunker = DocumentChunker()
    
    with PhaseTimer("Phase 4: Chunking") as timer:
        chunks = chunker.chunk_documents(valid_docs)
        
    # Analyze statistics
    total_chunks = len(chunks)
    avg_length = sum(len(c["text"]) for c in chunks) / total_chunks if total_chunks > 0 else 0
    
    # Statistics
    stats = {
        "Source Documents": len(valid_docs),
        "Total Chunks Generated": total_chunks,
        "Average Chunk Length": f"{avg_length:.1f} chars",
        "Execution Status": "Success" if total_chunks > 0 else "Failure",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print sample chunks for inspection
    sample_size = min(3, total_chunks)
    if sample_size > 0:
        print(f"--- SAMPLE CHUNKS (Showing {sample_size} of {total_chunks}) ---")
        for i in range(sample_size):
            chunk = chunks[i]
            print(f"Chunk {i+1} Metadata: {chunk['metadata']}")
            print(f"Content:\n{chunk['text'][:250]}...\n" + "-" * 40)
            
    return chunks

if __name__ == "__main__":
    # Test with dummy educational markdown content
    mock_documents = [
        {
            "url": "https://en.wikipedia.org/wiki/Deadlock",
            "title": "Deadlock - Wikipedia",
            "content": (
                "# Deadlock\n\n"
                "In concurrent computing, a deadlock is a state in which each member of a group is waiting "
                "for another member, including itself, to take action, such as sending a message or more "
                "commonly releasing a lock.\n\n"
                "## Necessary Conditions\n\n"
                "A deadlock situation on a resource can arise if and only if all of the following conditions "
                "hold simultaneously in a system:\n"
                "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
                "2. Hold and Wait: A process must be simultaneously holding at least one resource and requesting additional resources.\n"
                "3. No Preemption: Resources cannot be preempted; they can only be released voluntarily.\n"
                "4. Circular Wait: A process must be waiting for a resource which is being held by another process.\n\n"
                "## Deadlock Prevention\n\n"
                "Deadlock prevention algorithms work by ensuring that at least one of the four necessary conditions "
                "cannot hold. For example, a process might be forced to request all its resources at once (Hold and Wait prevention)."
            ),
            "success": True
        }
    ]
    
    try:
        run_phase4(mock_documents)
    except Exception as exc:
        print_failure(f"Phase 4 execution failed: {exc}")
        sys.exit(1)
