"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 9 - Content Generation Client module with rate-limit resiliency,
         TPD-aware immediate failover, configurable free fallback providers,
         and deduplicated multi-pass continuation.
Dependencies: groq, google-genai, config.config, utils.logger, utils.helper
"""

import sys
import time
import re
from typing import Dict, Any, Tuple, Optional, List
from utils.import_timing import timed_import

Groq = timed_import("groq").Groq

# Package imports
from config.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_PRIMARY_MODEL,
    GROQ_FALLBACK_MODELS,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    validate_config
)
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
logger = setup_logger("phase9_gemini_client")


def clean_and_merge_continuation(existing_text: str, cont_text: str) -> Tuple[str, bool]:
    """
    Intelligently deduplicates and merges continuation text into existing generated answer:
    1. Removes duplicate Markdown headings (#, ##, ###, ####) generated in previous turns.
    2. Strips overlapping sentence or word prefixes between the end of existing_text and start of cont_text.
    3. Discards large duplicated paragraphs or lines already present in existing_text.
    4. Smoothly connects text boundaries.

    Returns:
        Tuple[merged_answer_text, duplicate_detected_bool]
    """
    if not cont_text or not cont_text.strip():
        return existing_text, False

    duplicate_detected = False
    cont_clean = cont_text.strip()
    existing_clean = existing_text.strip()

    # 1. Extract existing headings (e.g., "## Definition / Overview")
    existing_headings = set()
    for line in existing_clean.splitlines():
        l_str = line.strip()
        if l_str.startswith("#"):
            h_clean = re.sub(r'^#+\s*', '', l_str).lower().strip()
            if h_clean:
                existing_headings.add(h_clean)

    # 2. Filter out duplicate headings and duplicate paragraphs from continuation lines
    cont_lines = cont_clean.splitlines()
    filtered_lines = []
    skipping_duplicate_block = False

    for line in cont_lines:
        l_str = line.strip()
        if l_str.startswith("#"):
            h_clean = re.sub(r'^#+\s*', '', l_str).lower().strip()
            if h_clean in existing_headings:
                duplicate_detected = True
                skipping_duplicate_block = True
                continue
            else:
                skipping_duplicate_block = False

        if skipping_duplicate_block:
            if len(l_str) > 15 and l_str.lower() in existing_clean.lower():
                duplicate_detected = True
                continue
            else:
                skipping_duplicate_block = False

        if len(l_str) > 25 and l_str.lower() in existing_clean.lower():
            duplicate_detected = True
            continue

        filtered_lines.append(line)

    cont_clean = "\n".join(filtered_lines).strip()
    if not cont_clean:
        return existing_text, True

    # 3. Check for word/sentence prefix overlap at the exact join boundary
    overlap_window = min(len(existing_clean), 300)
    existing_suffix = existing_clean[-overlap_window:].lower()
    cont_prefix = cont_clean[:overlap_window].lower()

    overlap_len = 0
    for l in range(min(len(existing_suffix), len(cont_prefix)), 3, -1):
        if existing_suffix.endswith(cont_prefix[:l]):
            overlap_len = l
            break

    if overlap_len > 0:
        duplicate_detected = True
        cont_clean = cont_clean[overlap_len:].lstrip()

    if not cont_clean:
        return existing_text, True

    # 4. Seamless join
    if not existing_text.endswith((" ", "\n")) and not cont_clean.startswith((" ", "\n", ".", ",", "!", "?", "]", ")", ":", ";")):
        merged = existing_text + " " + cont_clean
    else:
        merged = existing_text + cont_clean

    return merged, duplicate_detected


class GeminiContentGenerator:
    """
    Interfaces with Groq and Google Gemini APIs to generate grounded educational answers.
    Features:
    - Primary provider (Groq openai/gpt-oss-120b)
    - Distinguishes daily TPD exhaustion vs temporary RPM/TPM limits
    - Free fallback chain (Secondary Groq models + Google Gemini 2.5 Flash)
    - Output token budget enforcement per research depth
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model_name: str = GROQ_MODEL_NAME,
        gemini_key: str = GEMINI_API_KEY
    ) -> None:
        """
        Initializes Groq and optional Gemini fallback clients.
        """
        if not api_key:
            raise ValueError("Groq API key is missing. Ensure GROQ_API_KEY is defined in environment or .env.")

        self.groq_api_key = api_key
        self.primary_model = model_name or GROQ_PRIMARY_MODEL
        self.fallback_groq_models = [m for m in GROQ_FALLBACK_MODELS if m != self.primary_model]
        self.gemini_key = gemini_key or GEMINI_API_KEY
        self.gemini_model = GEMINI_FALLBACK_MODEL

        logger.debug("Initializing Groq Client...")
        try:
            self.groq_client = Groq(api_key=self.groq_api_key, timeout=30.0, max_retries=0)
            self.model_name = self.primary_model
            logger.info(f"Groq primary client initialized with model: {self.primary_model}")
        except Exception as e:
            logger.error("Failed to initialize Groq client (%s).", type(e).__name__)
            raise RuntimeError("Groq client initialization failed.") from e

        self.gemini_client = None
        if self.gemini_key:
            try:
                from google import genai
                from google.genai import types
                self.gemini_client = genai.Client(
                    api_key=self.gemini_key,
                    http_options=types.HttpOptions(timeout=30000)
                )
                logger.info(f"Google Gemini fallback client initialized with model: {self.gemini_model}")
            except Exception as e:
                logger.warning("Google Gemini client initialization skipped/failed (%s).", type(e).__name__)

    def _is_daily_tpd_exceeded(self, err_msg: str) -> bool:
        """
        Checks if the error indicates a daily Tokens Per Day (TPD) quota exhaustion.
        """
        msg = err_msg.lower()
        return (
            "tokens per day" in msg or
            "tpd" in msg or
            "limit 200000" in msg or
            "daily quota" in msg or
            "quota exceeded" in msg or
            "exceeded your current quota" in msg
        )

    def _try_groq_model(
        self, model: str, prompt: str, max_tokens: int
    ) -> Tuple[Optional[str], Optional[Dict[str, int]], str, bool]:
        """
        Attempts generation using a specific Groq model with limited retry logic.
        Returns: (answer_text, token_dict, finish_reason, is_daily_quota_exhausted)
        """
        logger.info(f"[GENERATION] Attempting Groq model '{model}' with max_output_tokens={max_tokens}...")

        max_attempts = 2
        for attempt in range(max_attempts):
            attempt_started = time.perf_counter()
            try:
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens
                )

                choice = response.choices[0]
                answer_text = choice.message.content
                finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

                if not answer_text:
                    raise ValueError(f"Groq model '{model}' returned empty content.")

                usage = getattr(response, "usage", None)
                if usage:
                    token_dict = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0)
                    }
                else:
                    token_dict = {
                        "input_tokens": len(prompt) // 4,
                        "output_tokens": len(answer_text) // 4,
                        "total_tokens": (len(prompt) + len(answer_text)) // 4
                    }

                logger.info(
                    f"[GENERATION] provider=groq model={model} finish_reason={finish_reason} "
                    f"completion_tokens={token_dict['output_tokens']}"
                )
                return answer_text, token_dict, finish_reason, False

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    if self._is_daily_tpd_exceeded(err_str):
                        logger.warning(f"[GENERATION] Groq model '{model}' DAILY TPD quota exhausted. Skipping retries.")
                        return None, None, "error", True

                    if attempt < max_attempts - 1:
                        logger.warning(f"[GENERATION] Temporary rate limit on Groq '{model}'. Retrying in 2.0s (Attempt {attempt+1}/{max_attempts})...")
                        time.sleep(2.0)
                        continue
                logger.warning("[GENERATION] Groq model request failed (%s).", type(e).__name__)
                return None, None, "error", False
            finally:
                logger.info(
                    "[RAG-TIME] Groq API attempt model=%s: %.3f sec",
                    model, time.perf_counter() - attempt_started
                )

        return None, None, "error", False

    def _try_gemini_model(
        self, prompt: str, max_tokens: int
    ) -> Tuple[Optional[str], Optional[Dict[str, int]], str]:
        """
        Attempts generation using Google Gemini fallback client.
        Returns: (answer_text, token_dict, finish_reason)
        """
        if not self.gemini_client:
            logger.warning("[GENERATION] Gemini fallback client is not configured.")
            return None, None, "error"

        logger.info(f"[GENERATION] Attempting Fallback provider Google Gemini ('{self.gemini_model}')...")
        generation_started = time.perf_counter()
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=max_tokens
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=config
            )
            answer_text = response.text
            if not answer_text:
                raise ValueError("Gemini returned an empty text response.")

            candidates = getattr(response, "candidates", None)
            finish_reason = "stop"
            if candidates and len(candidates) > 0:
                raw_reason = str(getattr(candidates[0], "finish_reason", "STOP")).upper()
                if "MAX" in raw_reason or "LENGTH" in raw_reason:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"

            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tok = getattr(usage, "prompt_token_count", len(prompt) // 4)
                out_tok = getattr(usage, "candidates_token_count", len(answer_text) // 4)
                tot_tok = getattr(usage, "total_token_count", in_tok + out_tok)
                token_dict = {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot_tok}
            else:
                token_dict = {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(answer_text) // 4,
                    "total_tokens": (len(prompt) + len(answer_text)) // 4
                }

            logger.info(
                f"[GENERATION] provider=gemini model={self.gemini_model} finish_reason={finish_reason} "
                f"completion_tokens={token_dict['output_tokens']}"
            )
            return answer_text, token_dict, finish_reason
        except Exception as e:
            logger.error("[GENERATION] Gemini fallback generation failed (%s).", type(e).__name__)
            return None, None, "error"
        finally:
            logger.info("[RAG-TIME] Gemini API generation: %.3f sec", time.perf_counter() - generation_started)

    def _execute_generation_with_fallback(
        self, prompt: str, max_tokens: int
    ) -> Tuple[str, Dict[str, int], str]:
        """
        Executes primary Groq -> secondary Groq -> Gemini fallback generation chain.
        Returns: (answer, token_dict, finish_reason)
        """
        # 1. Primary Groq model
        answer, token_dict, finish_reason, daily_exhausted = self._try_groq_model(self.primary_model, prompt, max_tokens)
        if answer is not None and token_dict is not None:
            return answer, token_dict, finish_reason

        if daily_exhausted:
            logger.warning(f"[GENERATION] Primary provider ({self.primary_model}) quota exhausted. Initiating fallback chain...")

        # 2. Secondary Groq fallback models
        for fb_model in self.fallback_groq_models:
            logger.info(f"[GENERATION] Switching to secondary Groq fallback model: {fb_model}")
            answer, token_dict, finish_reason, _ = self._try_groq_model(fb_model, prompt, max_tokens)
            if answer is not None and token_dict is not None:
                return answer, token_dict, finish_reason

        # 3. Google Gemini fallback provider
        logger.info("[GENERATION] All Groq models unavailable or quota exhausted. Switching to Google Gemini fallback...")
        answer, token_dict, finish_reason = self._try_gemini_model(prompt, max_tokens)
        if answer is not None and token_dict is not None:
            return answer, token_dict, finish_reason

        error_msg = (
            "Deep research is temporarily unavailable because the AI generation limit has been reached. "
            "Please try again later."
        )
        logger.error("[GENERATION] All generation providers failed in fallback chain.")
        raise RuntimeError(error_msg)

    def is_structurally_complete(self, answer_text: str) -> bool:
        """
        Lightweight structural completeness check.
        Checks simple conditions without additional LLM calls.
        """
        if not answer_text or not answer_text.strip():
            return False

        stripped = answer_text.strip()

        if stripped.count("```") % 2 != 0:
            return False

        if stripped.endswith(":") or stripped.endswith(": "):
            return False

        incomplete_trailing = {
            "the", "a", "an", "and", "or", "but", "with", "between", "for", "of", "to",
            "in", "on", "at", "by", "from", "that", "which", "is", "are", "as", "than", "such"
        }
        words = stripped.split()
        if words:
            last_word = re.sub(r'[^\w]', '', words[-1]).lower()
            if last_word in incomplete_trailing:
                return False

        lines = stripped.splitlines()
        if lines:
            last_line = lines[-1].strip()
            if last_line.startswith("|") and not last_line.endswith("|"):
                return False

        return True

    def generate_response_with_tokens(
        self, prompt: str, max_tokens: int = 1000
    ) -> Tuple[str, Dict[str, int]]:
        """
        Submits a grounded prompt to LLM through configurable primary and fallback providers.
        Executes controlled multi-turn continuation if the response was truncated (finish_reason=length).
        """
        print_processing(f"Submitting grounded query to LLM (primary: {self.primary_model}, max_output_tokens: {max_tokens})...")
        logger.info(f"[GENERATION] max_output_tokens={max_tokens} model={self.primary_model}")

        answer, token_dict, finish_reason = self._execute_generation_with_fallback(prompt, max_tokens)

        logger.info(f"[GENERATION] finish_reason={finish_reason} completion_tokens={token_dict.get('output_tokens', 0)}")

        # Determine if continuation is required
        should_continue = False
        if finish_reason == "length":
            should_continue = True
        elif finish_reason != "stop":
            should_continue = not self.is_structurally_complete(answer)
        else:
            stripped = answer.strip()
            lines = stripped.splitlines()
            last_line = lines[-1].strip() if lines else ""
            if (
                stripped.count("```") % 2 != 0
                or stripped.endswith(",")
                or re.search(r'\b(the|a|an|and|or|with|between|of|to|is|are)\s*$', stripped.lower())
                or (last_line.startswith("|") and not last_line.endswith("|"))
            ):
                should_continue = True

        logger.info(f"[GENERATION] continuation required={'true' if should_continue else 'false'}")

        current_finish = finish_reason
        if should_continue:
            logger.warning("[GENERATION] Response truncated by token limit. Initiating intelligent multi-pass continuation.")
            max_continuation_turns = 3
            current_turn = 0

            while should_continue and current_turn < max_continuation_turns:
                current_turn += 1
                logger.info(f"[CONTINUATION] pass={current_turn}")
                prev_len = len(answer)

                continuation_prompt = (
                    "System Instructions:\n"
                    "You are continuing an educational response that was cut off before completion due to token limits.\n"
                    "Continue the previous answer from EXACTLY where it stopped. Do NOT repeat previous content or headings.\n"
                    "Complete the unfinished sentence, section, code block, or table, and finish the answer cleanly.\n"
                    "Remain strictly grounded in the retrieved context provided in the original prompt.\n\n"
                    f"Original Prompt:\n{prompt[:3000]}\n\n"
                    f"Partial Answer Generated So Far:\n{answer[-2500:]}\n\n"
                    "Continuation (resume immediately from the next word/symbol to complete the answer cleanly):"
                )

                try:
                    cont_budget = max(1000, max_tokens)
                    cont_answer, cont_tokens, current_finish = self._execute_generation_with_fallback(
                        continuation_prompt, cont_budget
                    )

                    if cont_answer and cont_answer.strip():
                        answer, dup_detected = clean_and_merge_continuation(answer, cont_answer)
                        token_dict["output_tokens"] += cont_tokens.get("output_tokens", 0)
                        token_dict["total_tokens"] += cont_tokens.get("total_tokens", 0)

                        logger.info(f"[CONTINUATION] previous_length={prev_len}")
                        logger.info(f"[CONTINUATION] new_length={len(answer)}")
                        logger.info(f"[CONTINUATION] duplicate_detected={'true' if dup_detected else 'false'}")
                    else:
                        logger.warning(f"[CONTINUATION] pass={current_turn} returned empty output")
                        break

                except Exception as cont_err:
                    logger.warning("[CONTINUATION] pass=%s attempt failed (%s).", current_turn, type(cont_err).__name__)
                    break

                # Re-evaluate stopping condition
                if current_finish == "stop" and self.is_structurally_complete(answer):
                    should_continue = False
                elif current_finish != "length" and self.is_structurally_complete(answer):
                    should_continue = False

                logger.info(f"[CONTINUATION] final_complete={'true' if not should_continue else 'false'}")

        else:
            logger.info("[GENERATION] Response complete - no continuation required")

        # Post-generation structural safety fix (guarantees complete Markdown syntax)
        stripped = answer.strip()
        if stripped.count("```") % 2 != 0:
            answer = answer + "\n```\n"

        lines = answer.strip().splitlines()
        if lines:
            last_line = lines[-1].strip()
            if last_line.startswith("|") and not last_line.endswith("|"):
                answer = answer + " |"

        # Gracefully handle any remaining incomplete trailing sentence if cut off by hard safety limit
        if current_finish == "length":
            stripped_ans = answer.strip()
            if not re.search(r'[.!?`\]\)]\s*$', stripped_ans):
                last_p = max(stripped_ans.rfind('.'), stripped_ans.rfind('!'), stripped_ans.rfind('?'))
                if last_p != -1 and last_p > len(stripped_ans) * 0.70:
                    answer = stripped_ans[:last_p + 1]

        return answer, token_dict

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Submits a grounded prompt to LLM and returns the text output.
        """
        answer_text, _ = self.generate_response_with_tokens(prompt, max_tokens=max_tokens)
        return answer_text


def run_phase9(prompt: str) -> str:
    """
    Orchestrates the Phase 9 LLM Content Generation call.
    """
    print_phase_header("Phase 9: LLM Content Generation")
    print_loading("Connecting to LLM Provider API...")

    try:
        validate_config()
    except ValueError as val_err:
        print_failure(f"Configuration validation failed: {val_err}")
        logger.error(f"Configuration validation failed: {val_err}")
        sys.exit(1)

    generator = GeminiContentGenerator()

    with PhaseTimer("Phase 9: Content Generation") as timer:
        answer = generator.generate_response(prompt)

    stats = {
        "Target Model": generator.primary_model,
        "Prompt Size (chars)": len(prompt),
        "Response Size (chars)": len(answer),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)

    print("--- RESPONSE PREVIEW ---")
    print(answer)
    print("-" * 60 + "\n")

    return answer


if __name__ == "__main__":
    test_prompt = (
        "System Instructions:\n"
        "You are an expert Engineering Education AI Assistant. Answer based ONLY on context.\n\n"
        "Retrieved Context:\n"
        "[Source 1]\n"
        "Title: CPU Scheduling\n"
        "URL: https://wikipedia.org/wiki/CPU_scheduling\n"
        "Content:\n"
        "CPU scheduling is the process by which a process is allocated the CPU for execution.\n\n"
        "User Question:\n"
        "What is CPU scheduling?\n\n"
        "Answer:\n"
    )

    try:
        run_phase9(test_prompt)
    except Exception as exc:
        print_failure(f"Phase 9 execution failed: {exc}")
        sys.exit(1)
