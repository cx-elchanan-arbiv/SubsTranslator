#!/usr/bin/env python3
"""
Gemini API Transcription Module
Transcribes YouTube videos using Google Gemini 2.5 Flash
"""

import logging
import re

import yt_dlp
from google import genai
from google.genai import types

from config import get_config
from core.exceptions import TranscriptionError

logger = logging.getLogger(__name__)
config = get_config()


class GeminiTranscriptionError(TranscriptionError):
    """Custom exception for Gemini transcription errors"""

    def __init__(self, error: str):
        message = f"Gemini transcription error: {error}"
        user_message = f"Gemini transcription error: {error}"
        super().__init__(
            message=message,
            error_code="GEMINI_TRANSCRIPTION_ERROR",
            recoverable=True,
            user_message=user_message,
        )


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp string to seconds.

    Supports formats:
    - [MM:SS] -> 65.0
    - [H:MM:SS] -> 3665.0
    - [SS] -> 45.0

    Args:
        timestamp_str: Timestamp string (e.g., "1:05", "0:45", "1:30:45")

    Returns:
        float: Seconds
    """
    parts = timestamp_str.strip().split(":")

    if len(parts) == 1:
        # SS
        return float(parts[0])
    elif len(parts) == 2:
        # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # H:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        logger.warning(f"Invalid timestamp format: {timestamp_str}")
        return 0.0


def parse_gemini_response(response_text: str, video_duration: float) -> list[dict]:
    """
    Parse Gemini's transcription response into segments.

    Expected format:
    [00:01] text here
    [00:06] more text
    [01:30] even more

    Args:
        response_text: Raw response from Gemini
        video_duration: Total video duration in seconds

    Returns:
        List of segments: [{"start": 0.0, "end": 5.0, "text": "..."}]
    """
    # Regex pattern to match [MM:SS] or [H:MM:SS] followed by text
    pattern = r"\[(\d+:?\d*:?\d+)\]\s*(.+?)(?=\[|$)"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        logger.warning("No timestamp patterns found in Gemini response")
        # Fallback: return entire text as single segment
        return [{"start": 0.0, "end": video_duration, "text": response_text.strip()}]

    segments = []
    for i, (timestamp, text) in enumerate(matches):
        start = parse_timestamp(timestamp)
        text = text.strip()

        # Estimate end time
        if i < len(matches) - 1:
            # Use next segment's start time
            next_timestamp = matches[i + 1][0]
            end = parse_timestamp(next_timestamp)
        else:
            # Last segment - use video duration or estimate
            end = min(start + 5.0, video_duration)

        if text:  # Only add non-empty segments
            segments.append({"start": start, "end": end, "text": text})

    logger.info(f"Parsed {len(segments)} segments from Gemini response")
    return segments


def get_youtube_duration(youtube_url: str) -> float:
    """
    Get video duration using yt-dlp (without downloading).

    Args:
        youtube_url: YouTube video URL

    Returns:
        float: Duration in seconds
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
        "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            duration = info.get("duration", 0)
            logger.info(f"Video duration: {duration}s ({duration/60:.1f} minutes)")
            return duration
    except Exception as e:
        logger.error(f"Failed to get video duration: {e}")
        raise GeminiTranscriptionError(f"Cannot extract video metadata: {e}")


def transcribe_with_gemini(
    youtube_url: str,
    language: str = "auto",
    max_duration: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Transcribe YouTube video using Gemini API.

    Args:
        youtube_url: YouTube video URL
        language: Target language (auto = detect)
        max_duration: Maximum video duration in seconds (default: from config)
        progress_callback: Optional callback for progress updates

    Returns:
        dict: {
            "segments": [...],
            "language": "detected_lang",
            "gemini_used": True,
            "model": "gemini-2.5-flash"
        }

    Raises:
        GeminiTranscriptionError: If transcription fails
    """
    # Validate API key
    if not config.GEMINI_API_KEY:
        raise GeminiTranscriptionError("GEMINI_API_KEY not configured")

    max_duration = max_duration or config.GEMINI_MAX_DURATION

    logger.info("🚀 === GEMINI TRANSCRIPTION START ===")
    logger.info(f"📹 YouTube URL: {youtube_url}")
    logger.info(f"🌍 Language: {language}")
    logger.info(f"⏱️ Max duration: {max_duration}s ({max_duration/60:.1f} minutes)")

    try:
        # Step 1: Get video duration
        if progress_callback:
            progress_callback(
                5, "Checking video duration...", 5, "Step 0: Gemini Check", 5
            )

        duration = get_youtube_duration(youtube_url)

        # Step 2: Check duration limit
        if duration > max_duration:
            logger.warning(
                f"⚠️ Video too long: {duration}s > {max_duration}s ({duration/60:.1f} min > {max_duration/60:.1f} min)"
            )
            raise GeminiTranscriptionError(
                f"Video duration ({duration/60:.1f} minutes) exceeds maximum ({max_duration/60:.1f} minutes)"
            )

        logger.info(f"✅ Duration check passed: {duration}s")

        # Step 3: Initialize Gemini client
        if progress_callback:
            progress_callback(
                10, "Connecting to Gemini API...", 10, "Step 0: Gemini Init", 5
            )

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        model = config.GEMINI_MODEL
        logger.info(f"🤖 Using model: {model}")

        # Step 4: Prepare prompt
        if language == "auto":
            language_instruction = "Detect the language automatically."
        else:
            language_instruction = f"The video is in {language} language."

        prompt = f"""Transcribe this YouTube video with precise timestamps.

{language_instruction}

Format requirements:
- Use [MM:SS] format for timestamps
- One line per timestamp
- Include all spoken words
- Example:
[00:01] First sentence here
[00:15] Second sentence
[01:30] Another sentence

Transcribe now:"""

        # Step 5: Call Gemini API
        if progress_callback:
            progress_callback(
                20, "Sending request to Gemini...", 20, "Step 1: Gemini API", 5
            )

        logger.info("📤 Sending request to Gemini API...")

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_uri(
                    file_uri=youtube_url,
                    mime_type="video/*",
                ),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for accuracy
                max_output_tokens=8000,  # Gemini 2.5 Flash limit
            ),
        )

        if progress_callback:
            progress_callback(
                60, "Processing Gemini response...", 60, "Step 2: Parsing", 5
            )

        # Step 6: Extract response text
        if not response or not response.text:
            raise GeminiTranscriptionError("Empty response from Gemini")

        response_text = response.text
        logger.info(
            f"📥 Received response from Gemini ({len(response_text)} characters)"
        )

        # Always log the full response for debugging
        logger.info("🔍 === FULL GEMINI RESPONSE (START) ===")
        logger.info(response_text)
        logger.info("🔍 === FULL GEMINI RESPONSE (END) ===")

        # Step 7: Parse response into segments
        if progress_callback:
            progress_callback(
                70, "Parsing transcription...", 70, "Step 3: Segmentation", 5
            )

        segments = parse_gemini_response(response_text, duration)

        if not segments:
            raise GeminiTranscriptionError(
                "Failed to parse Gemini response into segments"
            )

        # Step 8: Detect language (uses Gemini's built-in language detection for "auto")
        detected_language = language if language != "auto" else "unknown"

        if progress_callback:
            progress_callback(
                85, "Transcription with Gemini completed!", 85, "Step 4: Complete", 5
            )

        result = {
            "segments": segments,
            "language": detected_language,
            "gemini_used": True,
            "model": model,
            "duration": duration,
        }

        logger.info("✅ === GEMINI TRANSCRIPTION COMPLETE ===")
        logger.info(f"📊 Segments: {len(segments)}")
        logger.info(f"🌍 Language: {detected_language}")
        logger.info(f"⏱️ Duration: {duration}s ({duration/60:.1f} minutes)")

        # Estimate cost
        estimated_input_tokens = duration * 100  # ~100 tokens/second
        estimated_output_tokens = len(response_text) * 1.3  # rough estimate
        estimated_cost = (estimated_input_tokens / 1_000_000 * 1.0) + (
            estimated_output_tokens / 1_000_000 * 2.5
        )
        logger.info(f"💰 Estimated cost: ${estimated_cost:.4f}")

        return result

    except GeminiTranscriptionError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        logger.error(f"❌ Gemini transcription failed: {e}")
        raise GeminiTranscriptionError(f"Gemini API error: {str(e)}")


def test_gemini_connection() -> bool:
    """
    Test Gemini API connection.

    Returns:
        bool: True if connection successful
    """
    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        return False

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        # Try listing models as a connection test
        models = client.models.list()
        logger.info("✅ Gemini API connection successful")
        logger.info(f"📋 Available models: {[m.name for m in list(models)[:3]]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing Gemini API connection...")
    if test_gemini_connection():
        print("✅ Connection test passed!")
    else:
        print("❌ Connection test failed!")
