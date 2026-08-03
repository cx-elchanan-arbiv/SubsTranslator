"""
Phase A+: Advanced OpenAI Rate Limiting with Redis
Implements precise TPM/RPM budgets, intelligent retries, and batch processing
"""

import json
import time

import redis
import tiktoken

from config import get_config
from logging_config import get_logger

logger = get_logger(__name__)
config = get_config()


class OpenAIRateLimiter:
    """Redis-based rate limiter for OpenAI API with TPM/RPM budgets"""

    def __init__(self, redis_client: redis.Redis = None):
        self.config = get_config()

        self.redis = redis_client or redis.from_url(self.config.REDIS_URL)
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            # Fallback to cl100k_base if gpt-4o encoding not available
            logger.warning("gpt-4o encoding not found, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Conservative defaults when headers are missing
        self.default_limits = {
            "tpm": 20000,  # Tokens per minute
            "rpm": 60,  # Requests per minute
        }

        # Redis keys
        self.tpm_key = "openai:tpm_budget"
        self.rpm_key = "openai:rpm_budget"
        self.batch_progress_key = "openai:batch_progress"

    def count_tokens(self, text: str) -> int:
        """Precise token counting with tiktoken"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using approximation: {e}")
            return int(
                len(text.split()) * 1.3
            )  # Conservative fallback, ensure int return

    def estimate_output_tokens(self, input_tokens: int, segments_count: int) -> int:
        """Safer output token estimation - Phase A+ Hotfix"""
        base = int(0.8 * input_tokens) + 200
        return min(base, input_tokens + 500)

    def acquire_budget(self, input_tokens: int, estimated_output: int) -> bool:
        """Acquire budget from Redis-based TPM/RPM limits - atomic operation"""
        total_tokens = input_tokens + estimated_output

        # Lua script for atomic check-and-increment
        lua_script = """
        local tpm_key = KEYS[1]
        local rpm_key = KEYS[2]
        local total_tokens = tonumber(ARGV[1])
        local tpm_limit = tonumber(ARGV[2])
        local rpm_limit = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])

        -- Get current values
        local current_tpm = tonumber(redis.call('GET', tpm_key) or 0)
        local current_rpm = tonumber(redis.call('GET', rpm_key) or 0)

        -- Check limits
        if current_tpm + total_tokens > tpm_limit then
            return {0, current_tpm + total_tokens, tpm_limit, current_rpm, rpm_limit}
        end

        if current_rpm + 1 > rpm_limit then
            return {0, current_tpm, tpm_limit, current_rpm + 1, rpm_limit}
        end

        -- Acquire budget atomically
        redis.call('INCRBY', tpm_key, total_tokens)
        redis.call('INCR', rpm_key)
        redis.call('EXPIRE', tpm_key, ttl)
        redis.call('EXPIRE', rpm_key, ttl)

        return {1, current_tpm + total_tokens, tpm_limit, current_rpm + 1, rpm_limit}
        """

        # Execute atomic operation
        current_time = int(time.time())
        minute_window = current_time // 60

        tpm_key = f"{self.tpm_key}:{minute_window}"
        rpm_key = f"{self.rpm_key}:{minute_window}"

        try:
            result = self.redis.eval(
                lua_script,
                2,  # number of keys
                tpm_key,
                rpm_key,  # KEYS
                total_tokens,
                self.default_limits["tpm"],
                self.default_limits["rpm"],
                120,  # ARGV
            )

            success, current_tpm, tpm_limit, current_rpm, rpm_limit = result

            if not success:
                if current_tpm > tpm_limit:
                    logger.warning(f"🚨 TPM budget exceeded: {current_tpm}/{tpm_limit}")
                else:
                    logger.warning(f"🚨 RPM budget exceeded: {current_rpm}/{rpm_limit}")
                return False

            if self.config.DEBUG:
                logger.debug(
                    f"✅ Budget acquired: {total_tokens} tokens, TPM: {current_tpm}/{tpm_limit}, RPM: {current_rpm}/{rpm_limit}"
                )
            return True

        except Exception as e:
            logger.error(f"Failed to acquire budget: {e}")
            return False

    def split_into_batches(
        self, segments: list[str], max_tokens_per_batch: int = None
    ) -> list[list[str]]:
        """Split segments into batches with recursive re-split - Phase A+ Hotfix"""
        from config import get_config

        config = get_config()

        if max_tokens_per_batch is None:
            max_tokens_per_batch = config.MAX_TOKENS_PER_BATCH

        return self._recursive_split(segments, max_tokens_per_batch)

    def split_into_segment_batches(
        self, segments: list[str], max_segments_per_batch: int = None
    ) -> list[list[str]]:
        """Split segments into fixed-size batches by count, not tokens"""
        from config import get_config

        config = get_config()

        if max_segments_per_batch is None:
            max_segments_per_batch = config.MAX_SEGMENTS_PER_BATCH

        batches = []
        for i in range(0, len(segments), max_segments_per_batch):
            batch = segments[i : i + max_segments_per_batch]

            # Safety check: verify batch won't exceed token limits
            if self._batch_fits(batch, config.MAX_TOKENS_PER_BATCH):
                batches.append(batch)
            else:
                # Handle edge cases
                if len(batch) == 1:
                    # Single segment too large - log warning and include it anyway
                    logger.error(
                        f"⚠️ CRITICAL: Single segment exceeds token limit! "
                        f"Segment length: {len(batch[0])} chars, "
                        f"Preview: {batch[0][:100]}..."
                    )
                    batches.append(
                        batch
                    )  # Include it anyway, will need special handling
                else:
                    # Split in half
                    logger.warning(
                        f"Batch of {len(batch)} segments exceeds token limit, splitting..."
                    )
                    mid = len(batch) // 2
                    if mid > 0:  # Extra safety check
                        batches.extend(
                            self.split_into_segment_batches(
                                batch[:mid], max_segments_per_batch
                            )
                        )
                        batches.extend(
                            self.split_into_segment_batches(
                                batch[mid:], max_segments_per_batch
                            )
                        )
                    else:
                        # This should never happen, but just in case
                        logger.error("Cannot split batch further, including as-is")
                        batches.append(batch)

        return batches

    def _recursive_split(self, segments: list[str], max_tokens: int) -> list[list[str]]:
        """Recursively split segments until each batch fits within token limit"""
        # Try to fit all segments in one batch first
        if self._batch_fits(segments, max_tokens):
            return [segments] if segments else []

        # If single segment is too large, we have a problem
        if len(segments) == 1:
            logger.warning(
                f"⚠️ Single segment exceeds {max_tokens} tokens: {len(segments[0][:100])}..."
            )
            return [segments]  # Return it anyway, will handle in retry logic

        # Split in half and try recursively
        mid = len(segments) // 2
        left_batches = self._recursive_split(segments[:mid], max_tokens)
        right_batches = self._recursive_split(segments[mid:], max_tokens)

        return left_batches + right_batches

    def _batch_fits(
        self, segments: list[str], max_tokens: int, target_language: str = "Hebrew"
    ) -> bool:
        """Check if batch fits within token limit"""
        if not segments:
            return True

        # System prompt tokens (using the enhanced prompt)
        system_prompt = f"""Translate to {target_language}.
CRITICAL RULES:
1) Translate EVERY line, even if very short or unclear.
2) Keep EXACT numbering and order (1., 2., 3., ...), one output line per input line.
3) Do NOT skip, merge, split, or reorder lines.
4) Preserve URLs, numbers, names; if unclear, still produce a translation.
Output: numbered lines only. No explanations or extra text."""
        system_tokens = self.count_tokens(system_prompt)

        # Calculate total input tokens
        input_tokens = 0
        for i, segment in enumerate(segments):
            segment_line = f"{i+1}. {segment}"
            input_tokens += self.count_tokens(segment_line)

        # Estimate output tokens
        estimated_output = self.estimate_output_tokens(input_tokens, len(segments))
        total_tokens = system_tokens + input_tokens + estimated_output

        return total_tokens <= max_tokens

    def extract_rate_limit_headers(self, response) -> dict[str, int]:
        """Extract rate limit headers from OpenAI response"""
        headers = {}
        try:
            if hasattr(response, "http_response") and hasattr(
                response.http_response, "headers"
            ):
                raw_headers = response.http_response.headers
                headers = {
                    "tpm_remaining": int(
                        raw_headers.get("x-ratelimit-remaining-tokens", 0)
                    ),
                    "rpm_remaining": int(
                        raw_headers.get("x-ratelimit-remaining-requests", 0)
                    ),
                    "tpm_reset": int(raw_headers.get("x-ratelimit-reset-tokens", 0)),
                    "rpm_reset": int(raw_headers.get("x-ratelimit-reset-requests", 0)),
                }
        except Exception as e:
            logger.debug(f"Could not extract rate limit headers: {e}")

        return headers

    def wait_for_retry(self, response=None, attempt: int = 1) -> int:
        """Calculate wait time based on response headers or exponential backoff with jitter"""
        import random

        wait_seconds = 0

        if response:
            headers = self.extract_rate_limit_headers(response)
            if headers.get("tpm_reset") or headers.get("rpm_reset"):
                # Use the minimum reset time
                reset_times = [
                    t for t in [headers.get("tpm_reset"), headers.get("rpm_reset")] if t
                ]
                if reset_times:
                    wait_seconds = min(reset_times)

        if not wait_seconds:
            # Exponential backoff with jitter: 2^attempt seconds + random jitter, max 60
            base_wait = min(2**attempt, 60)
            # Add 0-25% jitter to prevent thundering herd
            jitter = random.uniform(0, base_wait * 0.25)
            wait_seconds = base_wait + jitter

        logger.info(f"⏳ Waiting {wait_seconds:.1f}s before retry (attempt {attempt})")
        time.sleep(wait_seconds)
        return wait_seconds

    def save_batch_progress(
        self, task_id: str, completed_batches: int, total_batches: int
    ):
        """Save progress to Redis for resume capability"""
        progress_data = {
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "timestamp": time.time(),
        }
        self.redis.setex(
            f"{self.batch_progress_key}:{task_id}", 3600, json.dumps(progress_data)
        )

    def load_batch_progress(self, task_id: str) -> dict | None:
        """Load progress from Redis"""
        try:
            data = self.redis.get(f"{self.batch_progress_key}:{task_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Could not load batch progress: {e}")
        return None

    def log_batch_operation(
        self,
        batch_id: str,
        provider: str,
        input_tokens: int,
        estimated_output: int,
        duration: float,
        retries: int,
        waited_seconds: float,
        status: str,
        headers: dict = None,
    ):
        """Phase A+ Hotfix: Enhanced structured logging for batch operations"""
        total_tokens = input_tokens + estimated_output
        headers_present = bool(headers and any(headers.values()))

        logger.info(
            "📊 Phase A+ Batch operation completed",
            batch_id=batch_id,
            provider=provider,
            in_tokens=input_tokens,
            est_out_tokens=estimated_output,
            total_tokens=total_tokens,
            duration_s=round(duration, 2),
            retries=retries,
            waited_s=round(waited_seconds, 1),
            status=status,
            rate_limit_headers_present=headers_present,
            headers=headers or {},
        )


# Global rate limiter instance
rate_limiter = OpenAIRateLimiter()

# P1: Thread-safe semaphore for concurrent translation workers
import threading

from config import get_config

_config = get_config()
OPENAI_SEMAPHORE = threading.Semaphore(_config.MAX_CONCURRENT_OPENAI_REQUESTS)
