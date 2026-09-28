"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 8 - Prompt Builder module to construct structured prompts for Gemini grounding.
Dependencies: utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any, Optional

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
logger = setup_logger("phase8_prompt_builder")

class PromptBuilder:
    """
    Builds structured, grounded prompts containing system instructions,
    retrieved contexts (titles, URLs, and text content), and target constraints.
    """

    def __init__(self) -> None:
        """Initializes the prompt builder."""
        logger.debug("PromptBuilder initialized.")

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved text chunks with source title, URL, and an index.
        
        Args:
            retrieved_chunks (List[Dict[str, Any]]): List of matching chunks with metadata.
            
        Returns:
            str: Formatted context block.
        """
        if not retrieved_chunks:
            return "No relevant context found from trusted sources."
            
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get("title", "Untitled Webpage")
            url = chunk.get("url", "No URL provided")
            text = chunk.get("text", "")
            
            part = (
                f"[Source {idx}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content:\n{text.strip()}\n"
            )
            context_parts.append(part)
            
        return "\n".join(context_parts)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        research_depth: str = "quick",
        existing_answer: Optional[str] = None
    ) -> str:
        """
        Constructs the complete grounded prompt injected with system instructions,
        context, chat history, research depth target, and query.
        """
        clean_depth = (research_depth or "quick").lower().strip()
        logger.info(f"Building grounded prompt for query: '{query}' with {len(retrieved_chunks)} sources at depth='{clean_depth}'.")
        
        # Format the retrieved sources into the context structure
        context_block = self.format_context(retrieved_chunks)
        
        history_block = ""
        if chat_history:
            recent_history = chat_history[-3:]
            history_parts = []
            for msg in recent_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "").strip()
                if role == "Assistant" and len(content) > 250:
                    content = content[:250] + "..."
                history_parts.append(f"{role}: {content}")
            history_block = "Recent Conversation Context:\n" + "\n".join(history_parts) + "\n\n"

        # Output depth target instructions
        depth_instructions = {
            "quick": (
                "Target Synthesis Depth: QUICK (Target output: ~300-500 useful tokens, maximum budget 600)\n"
                "- Provide a concise but COMPLETE answer. Cover the essential definition, key concepts, and one useful example if appropriate.\n"
                "- Keep the answer direct and focused. Do not start optional sections or lists that cannot be completed cleanly.\n"
                "- For Quick mode especially, prefer concise bullet points and avoid Markdown tables unless essential."
            ),
            "standard": (
                "Target Synthesis Depth: STANDARD (Target output: ~600-900 useful tokens, maximum budget 1000)\n"
                "- Provide a detailed, focused, and COMPLETE answer. Cover the definition, important concepts, how it works, relevant examples, and practical considerations where useful.\n"
                "- Ensure every section, comparison, bullet list, or code example is complete before finishing."
            ),
            "deep": (
                "Target Synthesis Depth: DEEP (Target output: ~1000-1500 useful tokens, maximum budget 1600)\n"
                "- Provide a comprehensive, well-structured, and COMPLETE answer synthesizing information across all retrieved sources.\n"
                "- Include important details, multiple relevant aspects, examples, comparisons, limitations, and practical implications supported by sources.\n"
                "- Do not repeat information unnecessarily. Complete every section and structural element fully before ending."
            )
        }
        selected_depth_inst = depth_instructions.get(clean_depth, depth_instructions["quick"])

        expansion_block = ""
        if existing_answer:
            expansion_block = (
                "INSTRUCTION FOR ANSWER EXPANSION (RESEARCH MORE MODE):\n"
                "An initial answer was previously generated for this topic:\n"
                "--- PREVIOUS ANSWER ---\n"
                f"{existing_answer.strip()}\n"
                "--- END PREVIOUS ANSWER ---\n\n"
                "You are provided with NEW/ADDITIONAL retrieved sources.\n"
                "Tasks for Answer Expansion:\n"
                "1. Retain all key accurate facts and citations from the previous answer.\n"
                "2. Synthesize NEW facts, detailed explanations, practical examples, comparisons, and practical applications discovered from the newly retrieved sources.\n"
                "3. Do NOT simply repeat the previous answer with different wording. Visibly expand the explanation into a comprehensive, multi-section breakdown.\n"
                "4. Preserve existing citations and add bracketed citations (e.g. [1], [2], [3]...) for newly introduced information.\n\n"
            )

        # Build prompt sections with strict RAG and formatting rules
        prompt = (
            "System Instructions:\n"
            "You are an expert Engineering Education AI Assistant generating a user-facing answer from the provided grounded context.\n"
            "Produce a complete, self-contained response based ONLY on the provided retrieved web context.\n"
            "Do not stop in the middle of a sentence, paragraph, bullet, numbered step, table, quotation, or section.\n"
            "Use the available output budget efficiently. If the available output budget is insufficient for all optional details, prioritize the essential explanation and finish the answer cleanly rather than starting additional sections that cannot be completed.\n"
            "Never end with an unfinished sentence. Never leave Markdown structures incomplete. Never begin a section that you cannot finish.\n"
            "Do not use unnecessary introductory or repetitive text merely to consume tokens. The final response must end naturally and clearly.\n\n"
            
            f"{selected_depth_inst}\n\n"

            "Constraint Rules:\n"
            "1. ONLY answer using the facts directly mentioned in the retrieved context. Do NOT assume, extrapolate, or bring in external knowledge.\n"
            "2. If the context does not contain enough information to address the query, say EXACTLY: "
            "\"I couldn't find enough relevant web content to answer this question reliably.\"\n"
            "3. Never hallucinate or make up facts. Strict compliance is mandatory.\n"
            "4. Always cite your sources. Use bracketed numbers inline matching the Source index (e.g. [1], [2]).\n"
            "5. Table Guidance: Use a Markdown table ONLY when it meaningfully improves comparison. Do not create oversized tables with repetitive information. Prefer concise bullet points if a table would consume excessive output tokens.\n"
            "6. Quotation Guidance: Avoid long block quotations from sources. Prefer paraphrasing and concise source-supported explanations. Use short quotations only when useful. Do not allow a quotation to consume the majority of the output budget.\n"
            "7. Answer Structure Guidelines (include sections where relevant context allows and complete each section fully):\n"
            "   - **Definition / Overview**\n"
            "   - **Detailed Explanation**\n"
            "   - **Key Features / Mechanics / Formula** (if applicable)\n"
            "   - **Examples & Code Snippets** (if applicable)\n"
            "   - **Comparison & Trade-offs** (if applicable)\n"
            "   - **Practical Applications & Limitations**\n"
            "   - **Summary / Key Takeaways**\n"
            "   - **References** (A list of cited sources formatted as: [Title](URL))\n\n"
            
            f"{expansion_block}"
            
            "Retrieved Context:\n"
            f"{context_block}\n\n"
            
            f"{history_block}"
            
            "User Question:\n"
            f"{query}\n\n"
            
            "Answer:\n"
        )
        
        logger.debug("Prompt constructed successfully.")
        return prompt

    def build_fallback_prompt(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Constructs a prompt for when no search results could be retrieved from trusted engineering sources.
        Instructs the model to answer from general knowledge but prepend a warning.
        """
        logger.info(f"Building fallback prompt for query: '{query}'")
        
        history_block = ""
        if chat_history:
            history_parts = []
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_block = "Conversation History:\n" + "\n".join(history_parts) + "\n\n"
            
        prompt = (
            "System Instructions:\n"
            "You are an expert Engineering Education AI Assistant. Note that the system was UNABLE to retrieve "
            "any real-time web context from trusted sources for this query. Therefore, you must answer based on your "
            "general pre-trained engineering knowledge.\n\n"
            
            "CRITICAL REQUIREMENT:\n"
            "You MUST prepend a warning banner EXACTLY as follows at the very beginning of your response:\n"
            "**WARNING: No trusted real-time sources could be retrieved for this query. The following response is generated using general pre-trained knowledge and has not been grounded in trusted sources.**\n\n"
            
            f"{history_block}"
            
            "User Question:\n"
            f"{query}\n\n"
            
            "Answer:\n"
        )
        return prompt

def run_phase8(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Orchestrates the Phase 8 prompt construction.
    
    Args:
        query (str): User query.
        retrieved_chunks (List[Dict[str, Any]]): List of context chunks.
        
    Returns:
        str: Grounded prompt.
    """
    print_phase_header("Phase 8: Prompt Builder")
    print_loading("Assembling grounding guidelines...")
    
    builder = PromptBuilder()
    
    with PhaseTimer("Phase 8: Prompt Builder") as timer:
        print_processing("Formatting context chunks and prompt injections...")
        full_prompt = builder.build_prompt(query, retrieved_chunks)
        
    # Statistics
    stats = {
        "User Query": query,
        "Retrieved Sources": len(retrieved_chunks),
        "Prompt Length (chars)": len(full_prompt),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print the built prompt to CLI for verification
    print("--- GENERATED GROUNDED PROMPT ---")
    print(full_prompt)
    print("-" * 60 + "\n")
    
    return full_prompt

if __name__ == "__main__":
    # Test execution using mock retrieved chunks
    test_query = "Explain Deadlock conditions"
    mock_retrieved_chunks = [
        {
            "id": "https://wikipedia.org/wiki/Deadlock#chunk-2",
            "text": (
                "The four necessary conditions for deadlock are: Mutual Exclusion, Hold and Wait, "
                "No Preemption, and Circular Wait. If any one of these conditions is prevented, "
                "deadlock is avoided."
            ),
            "metadata": {"url": "https://wikipedia.org/wiki/Deadlock", "title": "Deadlock - Wikipedia", "chunk_number": 2},
            "url": "https://wikipedia.org/wiki/Deadlock",
            "title": "Deadlock - Wikipedia"
        }
    ]
    
    try:
        run_phase8(test_query, mock_retrieved_chunks)
    except Exception as exc:
        print_failure(f"Phase 8 execution failed: {exc}")
        sys.exit(1)
