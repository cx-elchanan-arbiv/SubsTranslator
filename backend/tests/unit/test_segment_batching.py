"""
Unit tests for segment-based batching and improved translation handling
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, backend_path)

from openai_rate_limiter import OpenAIRateLimiter
from services.translation_services import GoogleTranslator, OpenAITranslator

# `stub_openai_rate_limiter` keeps the OpenAI path off Redis and off real backoff
# sleeps; see tests/unit/conftest.py.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("stub_openai_rate_limiter"),
]


class TestSegmentBatching:
    """Test the new segment-based batching functionality."""

    def test_segment_batch_splitting(self):
        """Test that segments are split into fixed-size batches."""
        # Create rate limiter with mocked Redis
        with patch("redis.from_url"):
            rate_limiter = OpenAIRateLimiter()

        # Test with 81 segments
        segments = [f"Segment {i}" for i in range(1, 82)]

        # Mock config
        with patch("config.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                MAX_SEGMENTS_PER_BATCH=25, MAX_TOKENS_PER_BATCH=5000
            )

            batches = rate_limiter.split_into_segment_batches(segments)

        # Should be split into 4 batches: 25, 25, 25, 6
        assert len(batches) == 4
        assert len(batches[0]) == 25
        assert len(batches[1]) == 25
        assert len(batches[2]) == 25
        assert len(batches[3]) == 6

        # Verify all segments are included
        all_segments = []
        for batch in batches:
            all_segments.extend(batch)
        assert all_segments == segments

    def test_token_safety_check(self):
        """Test that oversized batches are split even with segment limit."""
        with patch("redis.from_url"):
            rate_limiter = OpenAIRateLimiter()

        # Create segments where 25 would exceed token limit
        long_segment = "This is a very long segment " * 100  # ~300 tokens
        segments = [long_segment] * 25

        with patch("config.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                MAX_SEGMENTS_PER_BATCH=25, MAX_TOKENS_PER_BATCH=5000
            )

            # Mock token counting to simulate exceeding limit
            with patch.object(rate_limiter, "_batch_fits") as mock_fits:
                # First call returns False (batch too big), subsequent calls return True
                mock_fits.side_effect = [False, True, True, True, True]

                batches = rate_limiter.split_into_segment_batches(segments)

        # Should be split into more than 1 batch due to token limit
        assert len(batches) > 1

    @patch("openai.OpenAI")
    def test_openai_prompt_format(self, mock_openai_class):
        """Test that OpenAI uses JSON format with ID-based matching."""
        import json

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create response mock - now using JSON format
        response_json = json.dumps(
            [
                {"id": 1, "translation": "תרגום ראשון"},
                {"id": 2, "translation": "תרגום שני"},
            ]
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=response_json))]
        mock_client.chat.completions.create.return_value = mock_response

        # Create mock config with all required attributes
        mock_config = MagicMock(
            OPENAI_API_KEY="test-key",
            MAX_SEGMENTS_PER_BATCH=25,
            MAX_TOKENS_PER_BATCH=4000,
            MAX_OPENAI_RETRIES=3,
            OPENAI_REQUEST_TIMEOUT_S=30,
            ALLOW_GOOGLE_FALLBACK=False,
            DEBUG=True,
        )

        with (
            patch("config.get_config", return_value=mock_config),
            patch("openai_rate_limiter.config", mock_config),
            patch("openai_rate_limiter.get_config", return_value=mock_config),
            patch("services.translation_services.config", mock_config),
        ):

            translator = OpenAITranslator()

            # Translate 2 segments
            texts = ["First text", "Second text"]
            result = translator.translate_batch(texts, "he")

            # Verify the call was made
            mock_client.chat.completions.create.assert_called()

            # Check the system prompt
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            system_message = messages[0]["content"]

            # Verify JSON format is used (new format)
            assert "JSON" in system_message
            assert "id" in system_message.lower()
            assert "translation" in system_message.lower()

    def test_google_translate_batching(self):
        """Test that Google Translate now uses batching."""
        # Mock the correct import path - DeepGoogleTranslator is the alias used in the code
        with patch(
            "services.translation_services.DeepGoogleTranslator"
        ) as mock_translator_class:
            mock_translator = MagicMock()
            mock_translator_class.return_value = mock_translator

            # Mock batch translations
            def mock_translate_batch(texts):
                return [f"translated_{text}" for text in texts]

            mock_translator.translate_batch.side_effect = mock_translate_batch

            # Create mock config with required attributes
            mock_config = MagicMock(MAX_SEGMENTS_PER_BATCH=25)

            # Patch the module-level config used by GoogleTranslator
            with patch("services.translation_services.config", mock_config):
                translator = GoogleTranslator()

                # Translate 50 segments (should be 2 batches)
                texts = [f"Text {i}" for i in range(50)]
                result = translator.translate_batch(texts, "he")

                # Should have been called twice (25 + 25)
                assert mock_translator.translate_batch.call_count == 2

                # First call with 25 segments
                first_call = mock_translator.translate_batch.call_args_list[0][0][0]
                assert len(first_call) == 25
                assert first_call[0] == "Text 0"

                # Second call with 25 segments
                second_call = mock_translator.translate_batch.call_args_list[1][0][0]
                assert len(second_call) == 25
                assert second_call[0] == "Text 25"

                # Verify all results
                assert len(result) == 50
                assert all(r.startswith("translated_") for r in result)

    def test_a_short_response_raises_instead_of_padding_with_english(self):
        """Reversed on evidence: a partial answer is a failure, not a result.

        The old branch filled the missing translations with the SOURCE text and
        returned normally, so the job went green while some lines shipped in English
        under the label of the service the user chose. Google is the default
        translator on the upload route, so this was the common path rather than an
        edge case. A failure the user can see beats a success they cannot check.
        """
        from core.exceptions import TranslationServiceError

        # Mock the correct import path - DeepGoogleTranslator is the alias used in the code
        with patch(
            "services.translation_services.DeepGoogleTranslator"
        ) as mock_translator_class:
            mock_translator = MagicMock()
            mock_translator_class.return_value = mock_translator

            # Return fewer translations than expected
            mock_translator.translate_batch.return_value = [
                "תרגום 1",
                "תרגום 2",
            ]  # Only 2 instead of 5

            # Create mock config with required attributes
            mock_config = MagicMock(MAX_SEGMENTS_PER_BATCH=25)

            # Patch the module-level config used by GoogleTranslator
            with patch("services.translation_services.config", mock_config):
                translator = GoogleTranslator()

                texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
                with pytest.raises(TranslationServiceError):
                    translator.translate_batch(texts, "he")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
