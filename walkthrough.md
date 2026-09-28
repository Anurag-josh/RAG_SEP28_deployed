# Walkthrough: Conversational Web-Grounded RAG Chatbot (Groq & Offline Fallback Engine)

This walkthrough documents the successful implementation, verification, and migration of the project's enhancements to the Groq API and local CPU embedding fallback. 

---

## 🛠️ Summary of Changes Made

### 1. Migrated Text Generation & Optimization to Groq (Llama 3.3)
- **Files**: [gemini_client.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase9/gemini_client.py) & [web_search.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase1/web_search.py)
- **Refactoring**: Bypassed Gemini 2.5 Flash for text completions and query reformulation. The system now uses the Groq SDK to call `llama-3.3-70b-versatile`.
- **Latency Gain**: The text generation phase latency plummeted from **~9.4s** (Gemini API) to **~1.2s** (Groq API).

### 2. Offline local CPU Embeddings Fallback
- **File**: [embedder.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase5/embedder.py) & [config.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/config/config.py)
- **Refactoring**: Made `GEMINI_API_KEY` optional. If `GEMINI_API_KEY` is not present (or commented out in `.env`), the system automatically falls back to local CPU-based `SentenceTransformer("all-MiniLM-L6-v2")` embeddings (dim 384), keeping the vector dimensions dynamic and the entire system functional without a Gemini API key.

### 3. Multi-threaded Scraping & Caching
- **File**: [website_loader.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase3/website_loader.py)
- **Refactoring**: Added `ThreadPoolExecutor` for loading multiple pages concurrently.
- **Cache Layer**: Implemented a local JSON scraping cache at `cache/scraped_pages.json` mapping `URL -> Markdown`. A subsequent request for a cached URL takes `0.000s`.

### 4. Hybrid Search & Coarse Reranking Heuristic
- **Files**: [retriever.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase7/retriever.py) & [web_grounded_rag.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase10/web_grounded_rag.py)
- **Rate-limit Protection**: Added a coarse-filtering pre-ranking step that reduces total candidate text chunks to at most **40** using a keyword density score before invoking Gemini embedding API.
- **Reranker**: Merges semantic cosine similarity (70% weight) and query term overlap density (30% weight) to retrieve the top `top_k` final chunks.

### 5. Multi-turn Memory & Graceful Fallback
- **Files**: [prompt_builder.py](file:///Users/abhishekjoshi/Desktop/rag2/RAG_2-main/phase8/prompt_builder.py)
- **Refactoring**: Updated the prompt builder to format and inject `chat_history` into generation prompts.
- **Fallback**: Added a try-except fallback path in orchestrator that generates a parametric response from general knowledge prepended with a warning banner if Tavily/scraping fails.

---

## 📊 Verification & Test Run Results

### Test Run: Grounded Multi-Turn TCP vs UDP Comparison (Offline Embeddings + Groq)
The system successfully optimized the query, scraped the relevant trusted pages, cached them, chunked them, embedded them in parallel, reranked them, and produced grounded notes using Groq Llama 3.3 and local CPU embeddings:
- **Total Duration**: **~10.0s** (first uncached run)
- **Raw Search Matches**: 3
- **Total Chunks**: 40
- **Phase 5 (Embedding) Latency (Local CPU)**: **0.836s**
- **Phase 9 (Generation) Latency (Groq Llama 3.3)**: **1.145s**
- **Grounded Response**: Structured with Definition, Explanation, Example, Summary, and References.
- **Fallback Mode**: Inactive (grounded response successfully retrieved and verified).
