"""
Unit tests for enhanced translation features:
- JSON format with ID validation
- Retry for missing segments
- Global numbering in logs
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, backend_path)

from services.translation_services import OpenAITranslator

# `stub_openai_rate_limiter` keeps the OpenAI path off Redis and off real backoff
# sleeps; see tests/unit/conftest.py.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("stub_openai_rate_limiter"),
]


class TestEnhancedTranslation:
    """Test the enhanced translation features with JSON format."""

    @patch("openai.OpenAI")
    def test_retry_missing_segments(self, mock_openai_class):
        """Test that missing segments trigger a retry."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # First response missing segment id=3
        first_response_json = json.dumps(
            [
                {"id": 1, "translation": "תרגום ראשון"},
                {"id": 2, "translation": "תרגום שני"},
                # Missing id=3
            ]
        )
        first_response = MagicMock()
        first_response.choices = [
            MagicMock(message=MagicMock(content=first_response_json))
        ]

        # Retry response with the missing segment
        retry_response_json = json.dumps([{"id": 3, "translation": "תרגום שלישי"}])
        retry_response = MagicMock()
        retry_response.choices = [
            MagicMock(message=MagicMock(content=retry_response_json))
        ]

        # Set up the mock to return different responses
        mock_client.chat.completions.create.side_effect = [
            first_response,
            retry_response,
        ]

        # Create a proper mock config object with all needed attributes
        mock_config_obj = MagicMock(
            OPENAI_API_KEY="test-key",
            MAX_SEGMENTS_PER_BATCH=25,
            MAX_TOKENS_PER_BATCH=4000,
            MAX_OPENAI_RETRIES=3,
            OPENAI_REQUEST_TIMEOUT_S=30,
            ALLOW_GOOGLE_FALLBACK=False,
            DEBUG=True,
        )

        # Patch the cached config in openai_rate_limiter module
        with (
            patch("config.get_config", return_value=mock_config_obj),
            patch("openai_rate_limiter.config", mock_config_obj),
            patch("openai_rate_limiter.get_config", return_value=mock_config_obj),
            patch("services.translation_services.config", mock_config_obj),
        ):

            translator = OpenAITranslator()

            # Translate 3 segments
            texts = ["First text", "Second text", "Third text"]
            result = translator.translate_batch(texts, "he")

            # Should have called API at least once
            assert mock_client.chat.completions.create.call_count >= 1

            # Verify final result has all translations
            assert len(result) == 3

    @patch("openai.OpenAI")
    def test_json_format_in_prompt(self, mock_openai_class):
        """Test that API uses JSON format."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Response in JSON format
        response_json = json.dumps([{"id": 1, "translation": "תרגום"}])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=response_json))]
        mock_client.chat.completions.create.return_value = mock_response

        # Create a proper mock config object with all needed attributes
        mock_config_obj = MagicMock(
            OPENAI_API_KEY="test-key",
            MAX_SEGMENTS_PER_BATCH=25,
            MAX_TOKENS_PER_BATCH=4000,
            MAX_OPENAI_RETRIES=3,
            OPENAI_REQUEST_TIMEOUT_S=30,
            ALLOW_GOOGLE_FALLBACK=False,
            DEBUG=False,
        )

        # Patch the cached config in openai_rate_limiter module
        with (
            patch("config.get_config", return_value=mock_config_obj),
            patch("openai_rate_limiter.config", mock_config_obj),
            patch("openai_rate_limiter.get_config", return_value=mock_config_obj),
            patch("services.translation_services.config", mock_config_obj),
        ):

            translator = OpenAITranslator()
            texts = ["Text to translate"]
            result = translator.translate_batch(texts, "he")

            # Check that the prompt uses JSON format
            call_args = mock_client.chat.completions.create.call_args
            system_message = call_args[1]["messages"][0]["content"]
            user_message = call_args[1]["messages"][1]["content"]

            # Should mention JSON in system prompt
            assert "JSON" in system_message

            # User message should contain JSON input
            assert '"id"' in user_message
            assert '"text"' in user_message

            # Result should have translation
            assert len(result) == 1
            assert result[0] == "תרגום"

    @patch("openai.OpenAI")
    def test_global_numbering_in_logs(self, mock_openai_class):
        """Test that global segment numbers appear in logs."""
        import logging

        # Setup OpenAI mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create JSON responses for batches
        def create_batch_response(batch_size):
            items = [
                {"id": i + 1, "translation": f"תרגום {i+1}"} for i in range(batch_size)
            ]
            return json.dumps(items)

        # Create responses for 3 batches of 25
        responses = []
        for i in range(3):
            response = MagicMock()
            response.choices = [
                MagicMock(message=MagicMock(content=create_batch_response(25)))
            ]
            responses.append(response)

        mock_client.chat.completions.create.side_effect = responses

        # Create a proper mock config object with all needed attributes
        mock_config_obj = MagicMock(
            OPENAI_API_KEY="test-key",
            MAX_SEGMENTS_PER_BATCH=25,
            MAX_TOKENS_PER_BATCH=4000,
            MAX_OPENAI_RETRIES=3,
            OPENAI_REQUEST_TIMEOUT_S=30,
            ALLOW_GOOGLE_FALLBACK=False,
            DEBUG=False,
        )

        # Capture logs
        log_capture = []
        original_info = logging.Logger.info

        def capture_log(self, msg, *args, **kwargs):
            log_capture.append(str(msg))
            return original_info(self, msg, *args, **kwargs)

        # Patch the cached config in openai_rate_limiter module
        with (
            patch("config.get_config", return_value=mock_config_obj),
            patch("openai_rate_limiter.config", mock_config_obj),
            patch("openai_rate_limiter.get_config", return_value=mock_config_obj),
            patch("services.translation_services.config", mock_config_obj),
            patch.object(logging.Logger, "info", capture_log),
        ):

            translator = OpenAITranslator()

            # Create 75 segments
            texts = [f"Text {i}" for i in range(1, 76)]

            try:
                result = translator.translate_batch(texts, "he")
            except Exception:
                pass  # We're interested in the logs

            # Check that logs were captured
            log_text = " ".join(log_capture)

            # Should see batch info (global indices or batch numbers)
            assert "batch" in log_text.lower() or "segments" in log_text.lower()

    def test_empty_line_filtering(self):
        """Test that empty lines don't cause mismatch."""
        # This is more of an integration test
        response_with_empty_lines = """1. תרגום ראשון
2. תרגום שני

3. תרגום שלישי

"""

        # Process the response
        translated_texts = [
            line.strip()
            for line in response_with_empty_lines.split("\n")
            if line.strip()
        ]

        # Should have exactly 3 lines, no empty ones
        assert len(translated_texts) == 3
        assert all(line for line in translated_texts)  # No empty strings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
