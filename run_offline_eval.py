"""
Offline Evaluation Utility
Project: Web-Grounded LLM Content Generation for Engineering Education
Purpose: Execute LLM-as-a-judge evaluation metrics offline for development and quality auditing.
"""

import sys
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from phase10.web_grounded_rag import WebGroundedRAGPipeline
from metrics_evaluator import (
    _evaluate_sentences,
    compute_faithfulness,
    compute_citation_metrics,
    compute_retrieval_precision_recall,
    compute_hallucination_rate,
    compute_answer_quality
)

def run_offline_evaluation(query: str) -> None:
    print(f"\n==================================================")
    print(f" RUNNING OFFLINE RAG EVALUATION ")
    print(f" Query: '{query}'")
    print(f"==================================================\n")

    pipeline = WebGroundedRAGPipeline()
    res = pipeline.run_pipeline(query)

    if not res.get("success"):
        print(f"Pipeline execution failed: {res.get('error')}")
        return

    answer = res.get("answer", "")
    print("Generated Answer Preview:")
    print("-" * 50)
    print(answer)
    print("-" * 50 + "\n")

    print("Computing offline evaluation metrics...")

    ans_quality = compute_answer_quality(query, answer)

    eval_results = {
        "query": query,
        "pipeline_statistics": res.get("statistics"),
        "pipeline_timings": res.get("timings"),
        "answer_quality": ans_quality
    }

    print("\n==================================================")
    print(" OFFLINE EVALUATION RESULTS ")
    print("==================================================")
    print(json.dumps(eval_results, indent=2))
    print("==================================================\n")

if __name__ == "__main__":
    test_q = "Explain Deadlock conditions in operating systems"
    if len(sys.argv) > 1:
        test_q = " ".join(sys.argv[1:])
    run_offline_evaluation(test_q)
