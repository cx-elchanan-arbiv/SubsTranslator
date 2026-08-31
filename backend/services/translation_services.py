import logging
import time
import uuid
from abc import ABC, abstractmethod
from functools import wraps

import openai
from deep_translator import GoogleTranslator as DeepGoogleTranslator

from config import get_config
from core.exceptions import TranslationServiceError
from openai_rate_limiter import rate_limiter  # Phase A+: Advanced rate limiting

# Get config and API key
config = get_config()
logger = logging.getLogger(__name__)


# Phase A: Retry decorator for translation services
def retry_on_transient_error(max_retries=2, backoff_factor=1.5):
    """Decorator for translation service retries on transient errors"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Phase A: Retry on transient errors
                    transient_indicators = [
                        "429",
                        "502",
                        "503",
                        "504",
                        "timeout",
                        "connection",
                        "network",
                    ]
                    if any(
                        indicator in error_str for indicator in transient_indicators
                    ):
                        if attempt < max_retries:
                            wait_time = backoff_factor * (2**attempt)
                            logger.warning(
                                f"🔄 Translation retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}"
                            )
                            time.sleep(wait_time)
                            continue

                    # Don't retry on client errors (400, 401, etc.)
                    break

            raise last_exception

        return wrapper

    return decorator


class Translator(ABC):
    """Abstract base class for all translation services."""

    @abstractmethod
    def translate_batch(
        self, texts: list[str], target_language: str, source_language: str = "auto"
    ) -> list[str]:
        """
        Translate a batch of texts.

        Args:
            texts (list[str]): A list of strings to translate.
            target_language (str): The target language code (e.g., 'en', 'he').
            source_language (str): The source language code (e.g., 'en', 'he'). Defaults to 'auto'.

        Returns:
            list[str]: A list of translated strings.

        Raises:
            Exception: If translation fails.
        """
        pass


#: Unicode letter ranges per target language, for the untranslated-line check.
#: Only scripts DISTINCT from Latin are listed: for Latin-target languages the
#: check cannot distinguish "untranslated" from "translated", so it stays off.
_TARGET_SCRIPT_RANGES = {
    "he": ("\u0590", "\u05ff"),
    "iw": ("\u0590", "\u05ff"),
    "ar": ("\u0600", "\u06ff"),
    "ru": ("\u0400", "\u04ff"),
    "uk": ("\u0400", "\u04ff"),
    "el": ("\u0370", "\u03ff"),
    "ja": ("\u3040", "\u30ff"),
    "ko": ("\uac00", "\ud7af"),
    "zh": ("\u4e00", "\u9fff"),
    "zh-CN": ("\u4e00", "\u9fff"),
}


def _looks_untranslated(text: str, target_language: str) -> bool:
    """A LONG output with zero target-script letters is not a translation.

    Found live: Google's error page ("Error 500 ... That's all we know.") shipped
    as the "translation" of two Hebrew-target cues. No error-page sniffing here —
    the general rule is that a 20+ character translation into Hebrew contains at
    least one Hebrew letter. Short outputs are exempt: names, numbers, "OK!" and
    interjections legitimately survive translation unchanged.
    """
    span = _TARGET_SCRIPT_RANGES.get(target_language)
    text = (text or "").strip()
    if not span or len(text) < 20:
        return False
    low, high = span
    return not any(low <= ch <= high for ch in text)


def _batch_collapsed_to_one_answer(sources: list[str], translations: list[str]) -> bool:
    """True when a translation batch is a broken response wearing a valid shape.

    Observed live: Google answered a batch with its own HTTP 500 error PAGE, and
    the library returned that page's text as the "translation" of every line —
    right count, text present, so both existing guards passed, and 19 of 30 cues
    shipped as "Error 500 (Server Error)!!1..." inside a burned video.

    The invariant that catches this without knowing anything about error pages:
    DISTINCT source lines do not legitimately translate to one IDENTICAL string.
    Two short cues collapsing ("Yes." / "Yes!" -> "כן.") is real; three or more
    distinct sources all yielding the same output is a broken response.
    """
    if len(translations) < 2:
        return False
    from collections import Counter

    counts = Counter(t.strip() for t in translations if (t or "").strip())
    if not counts:
        return False
    repeated_output, repeats = counts.most_common(1)[0]
    if repeats < 2:
        return False
    # How many DISTINCT sources produced that one output?
    distinct_sources = {
        (sources[i] or "").strip()
        for i, t in enumerate(translations)
        if (t or "").strip() == repeated_output
    }
    if repeats >= 3 and len(distinct_sources) >= 3:
        return True
    # Tail-batch hole, found live: a 2-line final batch broke entirely and slid
    # under the >=3 rule — two cues shipped as Google's error page. When an
    # ENTIRE multi-line batch of distinct sources collapses to one LONG
    # identical string, that is the same broken response; length keeps the
    # legitimate tiny collapse ("Yes!" / "Yes." -> "כן.") out of reach.
    return (
        len(distinct_sources) >= 2
        and repeats == len(translations)
        and len(repeated_output) > 40
    )


class GoogleTranslator(Translator):
    """Google Translate implementation using the deep_translator library."""

    @retry_on_transient_error(max_retries=2)  # Phase A: Add retry decorator
    def translate_batch(
        self, texts: list[str], target_language: str, source_language: str = "auto"
    ) -> list[str]:
        logger.info(
            f"Translating {len(texts)} segments using Google Translate to {target_language}."
        )

        # Map 'he' to 'iw' for compatibility with the library
        processed_target_language = "iw" if target_language == "he" else target_language

        # Split into batches using same logic as OpenAI
        max_segments = config.MAX_SEGMENTS_PER_BATCH
        all_translations = []

        for i in range(0, len(texts), max_segments):
            batch = texts[i : i + max_segments]
            batch_num = i // max_segments + 1
            total_batches = (len(texts) + max_segments - 1) // max_segments

            logger.info(
                f"Processing Google Translate batch {batch_num}/{total_batches}: {len(batch)} segments"
            )

            try:
                translator = DeepGoogleTranslator(
                    source=source_language, target=processed_target_language
                )
                batch_translations = translator.translate_batch(batch)

                if _batch_collapsed_to_one_answer(batch, batch_translations):
                    # A transient 500 usually clears immediately — one retry
                    # before failing turns most of these into a normal success.
                    logger.warning(
                        f"Google batch {batch_num} collapsed to one repeated "
                        f"answer (broken response) — retrying once"
                    )
                    time.sleep(2)
                    batch_translations = translator.translate_batch(batch)
                    if _batch_collapsed_to_one_answer(batch, batch_translations):
                        raise TranslationServiceError(
                            "google",
                            f"batch {batch_num} returned one identical answer for "
                            f"distinct lines twice in a row — the response is an "
                            f"error page, not a translation",
                        )

                # Per-line last defence: retry any line whose "translation" has
                # no target-script letters despite real length, and if the retry
                # is just as wrong, return "" — the SRT writer then keeps the
                # SOURCE text and counts a visible gap, instead of shipping
                # whatever the provider actually sent back.
                if batch_translations and len(batch_translations) == len(batch):
                    for j, line in enumerate(batch_translations):
                        if not _looks_untranslated(line, processed_target_language):
                            continue
                        logger.warning(
                            f"Google batch {batch_num} line {j + 1}: output has no "
                            f"target-script letters — retrying the line alone"
                        )
                        try:
                            retried = translator.translate(batch[j])
                        except Exception:  # noqa: BLE001 - fall through to gap
                            retried = None
                        if retried and not _looks_untranslated(
                            retried, processed_target_language
                        ):
                            batch_translations[j] = retried
                        else:
                            batch_translations[j] = ""

                if not batch_translations or len(batch_translations) != len(batch):
                    # NEVER pad with the source. The old branch filled a short or
                    # missing response with the ENGLISH it was asked to translate,
                    # and the caller had no way to tell: the job went green and the
                    # user got a .srt in which some lines were simply not translated,
                    # under the label of the service they chose. Google is the
                    # DEFAULT translator on the upload route, so this was the common
                    # path, not an edge case. A failure the user can see beats a
                    # success they cannot check.
                    raise TranslationServiceError(
                        "google",
                        f"batch {batch_num} returned "
                        f"{len(batch_translations) if batch_translations else 0} "
                        f"translations for {len(batch)} segments",
                    )

                all_translations.extend(batch_translations)

                # Small delay between batches to avoid rate limiting
                if i + max_segments < len(texts):
                    time.sleep(0.2)

            except TranslationServiceError:
                raise
            except Exception as e:
                # Same reasoning as the mismatch branch above: substituting the
                # source text turns a translation failure into an invisible one.
                logger.error(f"Google Translate batch {batch_num} failed: {e}")
                raise TranslationServiceError(
                    "google", f"batch {batch_num} failed: {e}"
                ) from e

        logger.info(
            f"Google Translate completed: {len(all_translations)} segments translated"
        )
        return all_translations


class OpenAITranslator(Translator):
    """OpenAI GPT-based translation implementation."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")
        # Fix httpx proxies issue by creating custom client
        import httpx

        http_client = httpx.Client(timeout=30.0)
        self.client = openai.OpenAI(api_key=self.api_key, http_client=http_client)

    def translate_batch(
        self, texts: list[str], target_language: str, source_language: str = "auto"
    ) -> list[str]:
        """Phase A+: Advanced OpenAI translation with rate limiting and intelligent batching"""
        logger.info(
            f"🚀 Starting OpenAI translation: {len(texts)} segments → {target_language}"
        )

        # Generate task ID for progress tracking
        task_id = str(uuid.uuid4())[:8]

        # Check for existing progress
        progress = rate_limiter.load_batch_progress(task_id)
        start_batch = progress["completed_batches"] if progress else 0

        # Split into segment-based batches (not token-based)
        batches = rate_limiter.split_into_segment_batches(texts)

        all_translations = []
        total_waited = 0
        global_offset = 0  # Track global segment index

        if config.DEBUG:
            logger.debug(
                f"📦 Split into {len(batches)} batches, starting from batch {start_batch}"
            )
        else:
            logger.info(f"Processing {len(batches)} translation batches")

        for batch_idx, batch_texts in enumerate(batches[start_batch:], start_batch):
            batch_id = f"{task_id}-{batch_idx}"
            batch_start_time = time.time()

            # Log batch info with global indices
            batch_global_start = global_offset + 1
            batch_global_end = global_offset + len(batch_texts)
            logger.info(
                f"🔄 Processing batch {batch_idx+1}/{len(batches)}: "
                f"{len(batch_texts)} segments (global #{batch_global_start}-#{batch_global_end})"
            )

            # Enhanced logging for debugging
            if config.DEBUG:
                logger.debug(
                    f"📊 Batch {batch_id} pre-request: segments={len(batch_texts)}, global_range=#{batch_global_start}-#{batch_global_end}"
                )

            try:
                # Try OpenAI for this batch with re-split fallback
                translations = self._translate_batch_with_retry_and_resplit(
                    batch_texts,
                    target_language,
                    source_language,
                    batch_id,
                    global_offset,
                )

                if translations:
                    all_translations.extend(translations)
                    # Save progress
                    rate_limiter.save_batch_progress(
                        task_id, batch_idx + 1, len(batches)
                    )
                else:
                    # Check if Google fallback is allowed
                    if config.ALLOW_GOOGLE_FALLBACK:
                        logger.warning(
                            f"🔄 Batch {batch_idx} failed with OpenAI after retries, falling back to Google"
                        )
                        google_translator = GoogleTranslator()
                        google_translations = google_translator.translate_batch(
                            batch_texts, target_language, source_language
                        )
                        all_translations.extend(google_translations)
                    else:
                        logger.error(
                            f"❌ Batch {batch_idx} failed with OpenAI after retries"
                        )
                        raise Exception(
                            "OpenAI translation failed after multiple attempts. Translation service is currently unavailable."
                        )

            except Exception as e:
                logger.error(f"❌ Batch {batch_idx} failed completely: {e}")
                # Check if Google fallback is allowed
                if config.ALLOW_GOOGLE_FALLBACK:
                    try:
                        google_translator = GoogleTranslator()
                        fallback_translations = google_translator.translate_batch(
                            batch_texts, target_language, source_language
                        )
                        all_translations.extend(fallback_translations)
                        logger.info(
                            f"✅ Batch {batch_idx} completed with Google fallback"
                        )
                    except Exception as fallback_error:
                        logger.error(
                            f"❌ Google fallback also failed for batch {batch_idx}: {fallback_error}"
                        )
                        # Use original text as last resort
                        all_translations.extend(batch_texts)
                else:
                    raise Exception(f"OpenAI translation failed completely: {str(e)}")

            # Update global offset for next batch
            global_offset += len(batch_texts)

        # Final summary logging
        expected_total = len(texts)

        if len(all_translations) == expected_total:
            logger.info(
                f"✅ Translation completed: {len(all_translations)}/{expected_total} segments | {total_waited:.1f}s total wait"
            )
        else:
            logger.warning(
                f"⚠️ Translation completed with mismatch: {len(all_translations)}/{expected_total} segments | {total_waited:.1f}s total wait"
            )

        if config.DEBUG:
            unique_translations = len(set(all_translations)) if all_translations else 0
            logger.debug(
                f"📊 Final summary: expected={expected_total}, received={len(all_translations)}, unique={unique_translations}"
            )
        return all_translations

    def _translate_batch_with_retry_and_resplit(
        self,
        texts: list[str],
        target_language: str,
        source_language: str,
        batch_id: str,
        global_offset: int = 0,
    ) -> list[str]:
        """P1: Thread-safe translation with retry and re-split fallback"""
        from config import get_config
        from openai_rate_limiter import OPENAI_SEMAPHORE

        config = get_config()

        # Try direct translation first
        try:
            # P1: Use thread-safe semaphore (blocking acquire/release)
            OPENAI_SEMAPHORE.acquire()
            try:
                return self._attempt_batch_translation(
                    texts, target_language, source_language, batch_id, global_offset
                )
            finally:
                OPENAI_SEMAPHORE.release()
        except Exception as e:
            error_str = str(e).lower()

            # Check if batch is large and should be re-split
            if "rate limit" in error_str or "budget exhausted" in error_str:
                # Check if batch is large enough to re-split (80% of max)
                if (
                    len(texts) > 1
                    and self._batch_tokens_estimate(texts)
                    > 0.8 * config.MAX_TOKENS_PER_BATCH
                ):
                    logger.warning(
                        f"🔄 Re-splitting batch {batch_id} ({len(texts)} segments)"
                    )

                    # Re-split into smaller batches
                    mini_batches = rate_limiter.split_into_batches(
                        texts, config.MAX_TOKENS_PER_BATCH // 2
                    )
                    all_translations = []

                    for i, mini_batch in enumerate(mini_batches):
                        mini_id = f"{batch_id}-split-{i}"
                        try:
                            # P1: Use thread-safe semaphore (blocking acquire/release)
                            OPENAI_SEMAPHORE.acquire()
                            try:
                                mini_result = self._attempt_batch_translation(
                                    mini_batch,
                                    target_language,
                                    source_language,
                                    mini_id,
                                    global_offset,
                                )
                                all_translations.extend(mini_result)
                            finally:
                                OPENAI_SEMAPHORE.release()
                        except Exception as mini_error:
                            logger.error(
                                f"❌ Mini-batch {mini_id} failed: {mini_error}"
                            )
                            raise mini_error  # Don't fallback to Google when OpenAI was explicitly selected

                    return all_translations

            raise e  # Don't fallback to Google when OpenAI was explicitly selected

    def _batch_tokens_estimate(self, texts: list[str]) -> int:
        """Estimate total tokens for a batch - JSON format"""
        import json

        system_prompt = """You are a translation API. Translate from source language to Hebrew.

CRITICAL REQUIREMENTS:
1. Input: JSON array with "id" and "text" fields.
2. Output: JSON array with "id" and "translation" fields.
3. PRESERVE all IDs - return same IDs in same order.
4. Translate EVERY segment separately.
Output format: Valid JSON array only."""
        system_tokens = rate_limiter.count_tokens(system_prompt)

        # Build sample JSON input
        input_segments = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
        input_json = json.dumps(input_segments, ensure_ascii=False, indent=0)
        input_tokens = rate_limiter.count_tokens(input_json)

        estimated_output = rate_limiter.estimate_output_tokens(input_tokens, len(texts))
        return system_tokens + input_tokens + estimated_output

    def _attempt_batch_translation(
        self,
        texts: list[str],
        target_language: str,
        source_language: str,
        batch_id: str,
        global_offset: int = 0,
    ) -> list[str]:
        """Attempt to translate a batch with rate limiting and retries - JSON format with IDs"""
        import json

        expected_count = len(texts)

        # Build input as JSON array with IDs
        input_segments = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
        input_json = json.dumps(input_segments, ensure_ascii=False, indent=0)

        system_prompt = f"""You are a translation API. Translate from source language to {target_language}.

CRITICAL REQUIREMENTS:
1. Input: JSON array of {expected_count} objects with "id" and "text" fields.
2. Output: JSON array of EXACTLY {expected_count} objects with "id" and "translation" fields.
3. PRESERVE all IDs from input - return same IDs in same order.
4. Translate EVERY segment separately - DO NOT merge, skip, or reorder.
5. If text is unclear, provide best-effort translation - NEVER skip.
6. Preserve URLs, numbers, proper names as-is.

Output format: Valid JSON array only - no explanations, no markdown, no extra text.
Example: [{{"id": 1, "translation": "..."}}, {{"id": 2, "translation": "..."}}]"""

        prompt_body = f"Translate these {expected_count} segments:\n{input_json}"

        # Count tokens precisely
        input_tokens = rate_limiter.count_tokens(system_prompt + prompt_body)
        estimated_output = rate_limiter.estimate_output_tokens(input_tokens, len(texts))

        if config.DEBUG:
            logger.debug(
                f"📊 Batch {batch_id}: {input_tokens} input + {estimated_output} estimated = {input_tokens + estimated_output} total tokens"
            )

        # Acquire budget with more patience
        max_budget_retries = 3
        for budget_attempt in range(max_budget_retries):
            if rate_limiter.acquire_budget(input_tokens, estimated_output):
                break
            else:
                wait_time = 70  # Wait slightly longer than a minute
                logger.warning(
                    f"💰 Budget exhausted for batch {batch_id}, waiting {wait_time}s (attempt {budget_attempt + 1}/{max_budget_retries})..."
                )
                time.sleep(wait_time)

                if budget_attempt == max_budget_retries - 1:
                    raise Exception("OpenAI budget limited. Please try again later.")

        return self._make_openai_request_with_retries(
            system_prompt,
            prompt_body,
            texts,
            batch_id,
            input_tokens,
            estimated_output,
            global_offset,
        )

    def _make_openai_request_with_retries(
        self,
        system_prompt: str,
        prompt_body: str,
        original_texts: list[str],
        batch_id: str,
        input_tokens: int,
        estimated_output: int,
        global_offset: int = 0,
    ) -> list[str]:
        """Phase A+ Hotfix: Make OpenAI request with header-aware retries - JSON format"""
        import json

        from config import get_config

        config = get_config()

        max_retries = config.MAX_OPENAI_RETRIES
        total_waited = 0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                # Make the request with config timeout
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_body},
                    ],
                    temperature=0.3,
                    timeout=config.OPENAI_REQUEST_TIMEOUT_S,
                )

                duration = time.time() - start_time

                # Extract rate limit headers
                headers = rate_limiter.extract_rate_limit_headers(response)

                # Process JSON response with ID validation
                translated_content = response.choices[0].message.content.strip()

                # Remove markdown code blocks if present
                if translated_content.startswith("```"):
                    lines = translated_content.split("\n")
                    translated_content = (
                        "\n".join(lines[1:-1]) if len(lines) > 2 else translated_content
                    )

                # Parse JSON
                try:
                    translations_json = json.loads(translated_content)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Batch {batch_id}: Invalid JSON response: {e}")
                    logger.debug(f"Raw response: {translated_content[:500]}")
                    raise Exception(f"OpenAI returned invalid JSON: {e}")

                # Validate structure
                if not isinstance(translations_json, list):
                    raise Exception(
                        f"Expected JSON array, got {type(translations_json)}"
                    )

                # Extract IDs and build translation map
                expected_ids = set(range(1, len(original_texts) + 1))
                found_ids = set()
                translation_map = {}  # id -> translation

                for item in translations_json:
                    if (
                        not isinstance(item, dict)
                        or "id" not in item
                        or "translation" not in item
                    ):
                        logger.warning(
                            f"Skipping invalid item in batch {batch_id}: {item}"
                        )
                        continue

                    item_id = item["id"]
                    found_ids.add(item_id)
                    translation_map[item_id] = item["translation"]

                # Log ID validation
                missing_ids = expected_ids - found_ids
                extra_ids = found_ids - expected_ids

                if missing_ids:
                    logger.error(
                        f"❌ Batch {batch_id}: Missing IDs: {sorted(missing_ids)}"
                    )
                if extra_ids:
                    logger.warning(
                        f"⚠️ Batch {batch_id}: Extra IDs: {sorted(extra_ids)}"
                    )

                # Build cleaned_translations by merging by ID (not index!)
                cleaned_translations = []
                for i in range(1, len(original_texts) + 1):
                    if i in translation_map:
                        cleaned_translations.append(translation_map[i])
                    else:
                        # Missing translation - will be handled below
                        cleaned_translations.append(None)

                # Validate response with ID-based logging
                valid_count = sum(1 for t in cleaned_translations if t is not None)
                json_ok = missing_ids == set() and extra_ids == set()
                ids_match = found_ids == expected_ids

                if config.DEBUG:
                    logger.debug(
                        f"📊 Batch {batch_id} post-response: json_ok={json_ok}, "
                        f"count_in={len(original_texts)}, count_out={len(translations_json)}, "
                        f"valid={valid_count}, ids_match={ids_match}"
                    )

                # CRITICAL: Validate ID matching
                if ids_match and valid_count == len(original_texts):
                    # Perfect match - log success
                    logger.info(
                        f"✅ Batch {batch_id}: Perfect ID mapping - {valid_count}/{len(original_texts)} translations"
                    )
                elif not ids_match or valid_count != len(original_texts):
                    logger.warning(
                        f"⚠️ OpenAI batch {batch_id} length mismatch: "
                        f"expected {len(original_texts)}, got {len(cleaned_translations)}"
                    )

                    # Handle missing IDs with targeted retry
                    if missing_ids:
                        missing_count = len(missing_ids)
                        # Build retry input with only missing IDs
                        missing_segments = [
                            {"id": mid, "text": original_texts[mid - 1]}
                            for mid in sorted(missing_ids)
                        ]

                        logger.warning(
                            f"🔄 Retry({batch_id}): Missing IDs={sorted(missing_ids)}"
                        )

                        # Try to recover missing translations.
                        # id -> translation (must exist before use)
                        retry_translations = {}
                        try:
                            retry_input = json.dumps(
                                missing_segments, ensure_ascii=False, indent=0
                            )
                            retry_prompt = f"Translate these {missing_count} segments:\n{retry_input}"

                            logger.info(
                                f"🔄 Retrying {missing_count} missing segments by ID..."
                            )

                            retry_response = self.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": retry_prompt},
                                ],
                                temperature=0.3,
                                max_tokens=estimated_output,
                                timeout=30,
                            )

                            retry_content = retry_response.choices[
                                0
                            ].message.content.strip()

                            # Remove markdown if present
                            if retry_content.startswith("```"):
                                lines_retry = retry_content.split("\n")
                                retry_content = (
                                    "\n".join(lines_retry[1:-1])
                                    if len(lines_retry) > 2
                                    else retry_content
                                )

                            retry_json = json.loads(retry_content)
                            if not isinstance(retry_json, list):
                                raise Exception(
                                    f"Retry returned non-array: {type(retry_json)}"
                                )

                            # Extract translations by ID from retry
                            for item in retry_json:
                                if (
                                    isinstance(item, dict)
                                    and "id" in item
                                    and "translation" in item
                                ):
                                    retry_translations[item["id"]] = item["translation"]

                            # Merge retry results into cleaned_translations by ID
                            recovered_count = 0
                            for i in range(len(cleaned_translations)):
                                idx = i + 1  # 1-based ID
                                if (
                                    cleaned_translations[i] is None
                                    and idx in retry_translations
                                ):
                                    cleaned_translations[i] = retry_translations[idx]
                                    recovered_count += 1

                            if recovered_count == missing_count:
                                logger.info(
                                    f"✅ Successfully recovered {recovered_count} missing translations by ID"
                                )
                            else:
                                logger.warning(
                                    f"⚠️ Only recovered {recovered_count}/{missing_count} missing translations"
                                )

                        except Exception as e:
                            logger.error(f"❌ Retry failed: {e}")
                            # Keep None values - will be filled with original texts below

                # Final pass: Check for None values (strict mode)
                missing_final = [
                    i + 1 for i, t in enumerate(cleaned_translations) if t is None
                ]
                if missing_final:
                    if not config.ALLOW_TRANSLATION_FALLBACK:
                        # STRICT MODE: Fail if any translations are missing
                        logger.error(
                            f"❌ STRICT MODE: Missing translations for IDs {missing_final}"
                        )
                        raise Exception(
                            f"Translation failed: missing translations for {len(missing_final)} segments"
                        )
                    else:
                        # Fallback mode: Use original texts
                        for i in range(len(cleaned_translations)):
                            if cleaned_translations[i] is None:
                                logger.warning(
                                    f"⚠️ Using original text for missing ID {i+1}"
                                )
                                cleaned_translations[i] = original_texts[i]

                # Log success
                rate_limiter.log_batch_operation(
                    batch_id=batch_id,
                    provider="openai",
                    input_tokens=input_tokens,
                    estimated_output=estimated_output,
                    duration=duration,
                    retries=attempt,
                    waited_seconds=total_waited,
                    status="success",
                    headers=headers,
                )

                return cleaned_translations

            except Exception as e:
                error_str = str(e).lower()

                # Check if this is a retryable error
                retryable = any(
                    code in error_str
                    for code in ["429", "502", "503", "504", "timeout", "connection"]
                )

                if attempt < max_retries and retryable:
                    # Phase A+ Hotfix: Header-aware backoff
                    wait_time = self._wait_for_retry_with_headers(
                        response=None, attempt=attempt + 1
                    )
                    total_waited += wait_time
                    logger.warning(
                        f"🔄 Retry {attempt + 1}/{max_retries} for batch {batch_id} after {wait_time}s"
                    )
                    continue
                else:
                    # Log failure
                    rate_limiter.log_batch_operation(
                        batch_id=batch_id,
                        provider="openai",
                        input_tokens=input_tokens,
                        estimated_output=estimated_output,
                        duration=time.time() - start_time,
                        retries=attempt,
                        waited_seconds=total_waited,
                        status=f"failed: {str(e)[:100]}",
                    )
                    raise e

        # This should never be reached as the method returns in the loop above
        raise Exception("All retry attempts failed")

    def _wait_for_retry_with_headers(self, response=None, attempt: int = 1) -> int:
        """Phase A+ Hotfix: Header-aware retry delay"""
        wait_seconds = 0

        # Check for Retry-After or x-ratelimit-reset headers
        if response and hasattr(response, "http_response"):
            headers = response.http_response.headers

            # Check Retry-After header (standard)
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after:
                try:
                    wait_seconds = int(retry_after)
                    logger.info(f"⏳ Using Retry-After header: {wait_seconds}s")
                except:
                    pass

            # Check OpenAI rate limit reset headers
            if not wait_seconds:
                reset_tokens = headers.get("x-ratelimit-reset-tokens")
                reset_requests = headers.get("x-ratelimit-reset-requests")
                if reset_tokens or reset_requests:
                    try:
                        # Use the minimum reset time if both available
                        reset_times = [
                            int(t) for t in [reset_tokens, reset_requests] if t
                        ]
                        if reset_times:
                            wait_seconds = min(reset_times)
                            logger.info(
                                f"⏳ Using rate limit reset header: {wait_seconds}s"
                            )
                    except:
                        pass

        # Fallback to exponential backoff if no headers
        if not wait_seconds:
            wait_seconds = min(2**attempt, 60)
            logger.info(
                f"⏳ Using exponential backoff: {wait_seconds}s (attempt {attempt})"
            )

        time.sleep(wait_seconds)
        return wait_seconds


def get_translator(service_name: str = "google") -> Translator:
    """
    Factory function to get a translator instance based on the service name.
    """
    if service_name == "openai":
        logger.info("Using OpenAI for translation.")
        return OpenAITranslator()
    elif service_name == "google":
        logger.info("Using Google Translate for translation.")
        return GoogleTranslator()
    else:
        logger.warning(
            f"Unknown translation service: '{service_name}'. Defaulting to Google Translate."
        )
        return GoogleTranslator()
