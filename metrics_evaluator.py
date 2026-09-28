"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Standalone RAG evaluation metrics module to calculate faithfulness, citation quality, retrieval precision/recall, hallucination rate, and answer quality.
Dependencies: groq, config.config, utils.logger, re, json
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from config.config import GROQ_API_KEY, GROQ_MODEL_NAME

# Initialize logger
logger = logging.getLogger("metrics_evaluator")

# Configurable thresholds for hallucination rate
HALLUCINATION_LOW_THRESHOLD = 10.0
HALLUCINATION_MEDIUM_THRESHOLD = 30.0

def _get_groq_client() -> Optional[Groq]:
    """Helper to initialize and return Groq client if API key is present."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set in the environment.")
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        return None

def _extract_json(text: str) -> Any:
    """Robust helper to extract and parse JSON from LLM string output."""
    # Find code blocks
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text_to_parse = match.group(1)
    else:
        match_code = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match_code:
            text_to_parse = match_code.group(1)
        else:
            text_to_parse = text
    
    text_to_parse = text_to_parse.strip()
    return json.loads(text_to_parse)

def split_into_sentences(text: str) -> List[str]:
    """Splits the text into logical sentences while filtering out headers and pure references."""
    lines = text.split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip markdown headers
        if line.startswith("#"):
            continue
        # Skip reference listings
        if line.lower().startswith("- [") or line.lower().startswith("[source"):
            continue
        # Split line by sentence boundary punctuation
        parts = re.split(r'(?<=[.!?])\s+', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences

def _evaluate_sentences(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Batches claim-level and citation-level judgments into a single structured LLM call.
    Returns a list of evaluations matching each sentence of the answer.
    """
    client = _get_groq_client()
    if not client:
        raise ValueError("Groq client not initialized due to missing API Key.")

    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    # Map sentences to their indices and extract citation markers
    sentence_inputs = []
    for idx, s in enumerate(sentences):
        citation_matches = re.findall(r'\[(\d+)\]', s)
        citations = sorted(list(set(int(m) for m in citation_matches)))
        sentence_inputs.append({
            "index": idx,
            "sentence": s,
            "citations_to_verify": citations
        })

    # Format the retrieved sources text
    sources_text = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        sources_text += f"[Source {idx}]\n"
        sources_text += f"Title: {chunk.get('title', 'Untitled')}\n"
        sources_text += f"Content:\n{chunk.get('text', '')}\n\n"

    prompt = (
        "You are an expert evaluator for Retrieval-Augmented Generation (RAG) pipelines.\n"
        "You are given a list of retrieved sources, and a list of sentences extracted from a generated answer.\n"
        "For each sentence, evaluate:\n"
        "1. Its 'entailment_status' against the entire set of retrieved sources. Classify it as:\n"
        "   - 'SUPPORTED' if the retrieved sources directly support the statement.\n"
        "   - 'CONTRADICTED' if the statement directly conflicts with or contradicts the retrieved sources.\n"
        "   - 'UNSUPPORTED' if the statement is not mentioned or not fully supported by the retrieved sources.\n"
        "2. For each citation in 'citations_to_verify', determine if that specific source index supports the statement.\n"
        "   - For example, if citation_to_verify is 1, check if [Source 1] supports the sentence.\n"
        "   - Return true if the source supports the sentence, false otherwise.\n\n"
        "Retrieved Sources:\n"
        f"{sources_text}\n"
        "Sentences to evaluate:\n"
        f"{json.dumps(sentence_inputs, indent=2)}\n\n"
        "You must return a JSON object with a single key 'results' mapping to an array of objects. Each object must match this schema:\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"index\": int,\n"
        "      \"entailment_status\": \"SUPPORTED\" | \"UNSUPPORTED\" | \"CONTRADICTED\",\n"
        "      \"citation_verifications\": [\n"
        "        {\n"
        "          \"source_index\": int,\n"
        "          \"supports\": boolean,\n"
        "          \"explanation\": string\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Format the output strictly as a JSON object, with no markdown code fence except optionally ```json ... ```. Do not output any other text."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )

    raw_response = response.choices[0].message.content
    parsed = _extract_json(raw_response)
    results = parsed.get("results", [])
    
    # Map back original sentence strings for safety
    for r in results:
        idx = r.get("index")
        if idx is not None and 0 <= idx < len(sentences):
            r["sentence"] = sentences[idx]
            
    return results

def compute_faithfulness(answer: str, retrieved_chunks: List[Dict[str, Any]], sentence_evals: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes Groundedness/Faithfulness metrics.
    """
    try:
        if sentence_evals is None:
            sentence_evals = _evaluate_sentences(answer, retrieved_chunks)

        if not sentence_evals:
            return {
                "total_claims": 0, "grounded_claims": 0, "unsupported_claims": 0,
                "contradicted_claims": 0, "groundedness_percent": 100.0, "error": None
            }

        total = len(sentence_evals)
        grounded = sum(1 for s in sentence_evals if s.get("entailment_status") == "SUPPORTED")
        unsupported = sum(1 for s in sentence_evals if s.get("entailment_status") == "UNSUPPORTED")
        contradicted = sum(1 for s in sentence_evals if s.get("entailment_status") == "CONTRADICTED")
        
        groundedness_pct = (grounded / total) * 100.0

        return {
            "total_claims": total,
            "grounded_claims": grounded,
            "unsupported_claims": unsupported,
            "contradicted_claims": contradicted,
            "groundedness_percent": round(groundedness_pct, 2),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error computing faithfulness: {e}")
        return {
            "total_claims": 0, "grounded_claims": 0, "unsupported_claims": 0,
            "contradicted_claims": 0, "groundedness_percent": 0.0, "error": str(e)
        }

def compute_citation_metrics(answer: str, retrieved_chunks: List[Dict[str, Any]], sentence_evals: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes Citation Precision and Citation Coverage.
    """
    try:
        if sentence_evals is None:
            sentence_evals = _evaluate_sentences(answer, retrieved_chunks)

        sentences = split_into_sentences(answer)
        if not sentences:
            return {
                "sentences_with_citation": 0, "sentences_without_citation": 0,
                "citation_precision_percent": 100.0, "citation_coverage_percent": 100.0, "error": None
            }

        sentences_with_cit = 0
        sentences_without_cit = 0
        correctly_cited_count = 0

        # Create index lookup for evaluations
        evals_by_index = {s.get("index"): s for s in sentence_evals}

        for idx, s in enumerate(sentences):
            citation_matches = re.findall(r'\[(\d+)\]', s)
            has_cit = len(citation_matches) > 0
            
            if has_cit:
                sentences_with_cit += 1
                
                # Verify if LLM verified all citations as supporting
                evaluation = evals_by_index.get(idx)
                if evaluation:
                    verifications = evaluation.get("citation_verifications", [])
                    # Sentence is correctly cited if there are verifications AND all supports are True
                    if verifications and all(v.get("supports") is True for v in verifications):
                        correctly_cited_count += 1
            else:
                sentences_without_cit += 1

        total_sentences = len(sentences)
        
        cit_precision = (correctly_cited_count / sentences_with_cit * 100.0) if sentences_with_cit > 0 else 100.0
        cit_coverage = (sentences_with_cit / total_sentences * 100.0) if total_sentences > 0 else 100.0

        return {
            "sentences_with_citation": sentences_with_cit,
            "sentences_without_citation": sentences_without_cit,
            "citation_precision_percent": round(cit_precision, 2),
            "citation_coverage_percent": round(cit_coverage, 2),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error computing citation quality: {e}")
        return {
            "sentences_with_citation": 0, "sentences_without_citation": 0,
            "citation_precision_percent": 0.0, "citation_coverage_percent": 0.0, "error": str(e)
        }

def compute_retrieval_precision_recall(query: str, retrieved_chunks: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Retrieval Precision @ K and Recall @ K.
    """
    # Base retrieval similarity stats
    semantic_scores = [c.get("semantic_score", 0.0) for c in retrieved_chunks]
    avg_similarity = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
    highest_similarity = max(semantic_scores) if semantic_scores else 0.0
    lowest_similarity = min(semantic_scores) if semantic_scores else 0.0
    k = len(retrieved_chunks)

    try:
        client = _get_groq_client()
        if not client:
            raise ValueError("Groq client not initialized due to missing API Key.")

        if not all_chunks:
            return {
                "precision_at_k": 0.0, "recall_at_k": 0.0, "k": k,
                "avg_similarity": round(avg_similarity, 4), "highest_similarity": round(highest_similarity, 4),
                "lowest_similarity": round(lowest_similarity, 4), "error": "No candidate chunks in the session."
            }

        # Select candidate chunks to run through the LLM relevance judge (cap at 15 to fit TPM limits)
        chunks_input = []
        for chunk in all_chunks[:15]:
            chunks_input.append({
                "chunk_id": chunk.get("id"),
                "text": chunk.get("text", "")[:200]
            })

        prompt = (
            "You are an expert evaluator for information retrieval systems.\n"
            "Given a user query and a list of text chunks, determine whether each chunk is relevant or not relevant to answering the query.\n"
            "A chunk is relevant (1) if it contains factual details directly addressing or answering the query. Otherwise, it is not relevant (0).\n\n"
            f"User Query: {query}\n\n"
            "Text Chunks to evaluate:\n"
            f"{json.dumps(chunks_input, indent=2)}\n\n"
            "You must return a JSON object with a single key 'relevance' mapping to an array of objects matching this schema:\n"
            "{\n"
            "  \"relevance\": [\n"
            "    {\n"
            "      \"chunk_id\": string,\n"
            "      \"relevant\": 0 | 1\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Format the output strictly as a JSON object, with no markdown code fence except optionally ```json ... ```. Do not output any other text."
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )

        parsed = _extract_json(response.choices[0].message.content)
        relevance_list = parsed.get("relevance", [])
        relevance_map = {item["chunk_id"]: item["relevant"] for item in relevance_list}

        # Calculate Precision @ K
        relevant_in_top_k = 0
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("id")
            if relevance_map.get(chunk_id) == 1:
                relevant_in_top_k += 1

        precision_at_k = (relevant_in_top_k / k) if k > 0 else 0.0

        # Calculate Recall @ K (out of total relevant chunks found in candidate pool)
        total_relevant_in_pool = sum(1 for val in relevance_map.values() if val == 1)
        recall_at_k = (relevant_in_top_k / total_relevant_in_pool) if total_relevant_in_pool > 0 else 0.0

        return {
            "precision_at_k": round(precision_at_k, 2),
            "recall_at_k": round(recall_at_k, 2),
            "k": k,
            "avg_similarity": round(avg_similarity, 4),
            "highest_similarity": round(highest_similarity, 4),
            "lowest_similarity": round(lowest_similarity, 4),
            "error": None
        }

    except Exception as e:
        logger.error(f"Error computing retrieval precision/recall: {e}")
        return {
            "precision_at_k": 0.0, "recall_at_k": 0.0, "k": k,
            "avg_similarity": round(avg_similarity, 4), "highest_similarity": round(highest_similarity, 4),
            "lowest_similarity": round(lowest_similarity, 4), "error": str(e)
        }

def compute_hallucination_rate(faithfulness_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes Hallucination Rate and Risk level deterministically from Faithfulness metrics.
    """
    if faithfulness_results.get("error"):
        return {
            "hallucination_rate_percent": 0.0,
            "hallucination_risk": "UNKNOWN",
            "error": f"Faithfulness metric failed: {faithfulness_results['error']}"
        }

    total_claims = faithfulness_results.get("total_claims", 0)
    unsupported = faithfulness_results.get("unsupported_claims", 0)
    contradicted = faithfulness_results.get("contradicted_claims", 0)

    if total_claims == 0:
        rate = 0.0
    else:
        rate = ((unsupported + contradicted) / total_claims) * 100.0

    if rate < HALLUCINATION_LOW_THRESHOLD:
        risk = "LOW"
    elif rate <= HALLUCINATION_MEDIUM_THRESHOLD:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "hallucination_rate_percent": round(rate, 2),
        "hallucination_risk": risk
    }

def compute_answer_quality(query: str, answer: str) -> Dict[str, Any]:
    """
    LLM-as-judge evaluation scoring overall generated answer quality 1-5.
    """
    try:
        client = _get_groq_client()
        if not client:
            raise ValueError("Groq client not initialized due to missing API Key.")

        prompt = (
            "You are an expert educational content evaluator.\n"
            "Your task is to grade the generated answer based on the user query on a scale of 1 to 5 (where 1 is poor and 5 is excellent).\n\n"
            f"User Query: {query}\n\n"
            "Generated Answer:\n"
            f"{answer}\n\n"
            "Please evaluate the answer across the following four criteria:\n"
            "1. Accuracy: Is the content factually correct and free from errors?\n"
            "2. Completeness: Does the response fully address the topic, including definitions, explanations, examples, and formulas where applicable?\n"
            "3. Clarity: Is the response easy to understand, well-structured, and suitable for engineering education?\n"
            "4. Relevance: Is the response directly relevant to the user query?\n\n"
            "You must return a JSON object with these fields:\n"
            "{\n"
            "  \"accuracy_score\": int,\n"
            "  \"completeness_score\": int,\n"
            "  \"clarity_score\": int,\n"
            "  \"relevance_score\": int\n"
            "}\n\n"
            "Format the output strictly as a JSON object, with no markdown code fence except optionally ```json ... ```. Do not output any other text."
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )

        parsed = _extract_json(response.choices[0].message.content)
        acc = float(parsed.get("accuracy_score", 0))
        comp = float(parsed.get("completeness_score", 0))
        clar = float(parsed.get("clarity_score", 0))
        rel = float(parsed.get("relevance_score", 0))

        avg = (acc + comp + clar + rel) / 4.0

        return {
            "accuracy_score": acc,
            "completeness_score": comp,
            "clarity_score": clar,
            "relevance_score": rel,
            "answer_quality_overall": round(avg, 2),
            "error": None
        }

    except Exception as e:
        logger.error(f"Error computing answer quality: {e}")
        return {
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "clarity_score": 0.0,
            "relevance_score": 0.0,
            "answer_quality_overall": 0.0,
            "error": str(e)
        }
