#!/usr/bin/env python3
"""
Protection Tests for OpenAI Translation Mismatch Handling (JSON Format)
Tests to ensure OpenAI mismatch fixes don't regress to old fallback behavior.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, backend_dir)

from services.translation_services import OpenAITranslator

# `stub_openai_rate_limiter` keeps the OpenAI path off Redis and off real backoff
# sleeps; see tests/unit/conftest.py.
pytestmark = pytest.mark.usefixtures("stub_openai_rate_limiter")


def create_mock_config():
    """Create a mock config with all required attributes."""
    return MagicMock(
        OPENAI_API_KEY="sk-test",
        MAX_SEGMENTS_PER_BATCH=25,
        MAX_TOKENS_PER_BATCH=4000,
        MAX_OPENAI_RETRIES=3,
        OPENAI_REQUEST_TIMEOUT_S=30,
        ALLOW_GOOGLE_FALLBACK=False,
        DEBUG=False,
    )


@pytest.mark.unit
class TestOpenAIMismatchProtection:
    """Tests to protect against regression of OpenAI mismatch handling."""

    @patch("openai.OpenAI")
    def test_too_many_translations_are_truncated_not_fallback(self, mock_openai_class):
        """
        CRITICAL: When OpenAI returns too many translations, truncate them

        OLD BEHAVIOR (BAD): Return original English text
        NEW BEHAVIOR (GOOD): Truncate to correct count and return Hebrew
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # OpenAI returns 4 translations instead of 3 (too many) - JSON format
        response_json = json.dumps(
            [
                {"id": 1, "translation": "שלום"},
                {"id": 2, "translation": "עולם"},
                {"id": 3, "translation": "בדיקה"},
                {"id": 4, "translation": "נוסף"},  # Extra
            ]
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=response_json))]
        mock_client.chat.completions.create.return_value = mock_response

        mock_config = create_mock_config()

        with (
            patch("config.get_config", return_value=mock_config),
            patch("openai_rate_limiter.config", mock_config),
            patch("openai_rate_limiter.get_config", return_value=mock_config),
            patch("services.translation_services.config", mock_config),
        ):

            translator = OpenAITranslator()
            inputs = ["Hello", "World", "Test"]  # 3 inputs
            result = translator.translate_batch(inputs, "he")

            # CRITICAL: Should return Hebrew translations (first 3), NOT original English
            assert len(result) == 3, "Should return exactly 3 translations"
            # Extra ID 4 should be ignored
            assert result[0] == "שלום"
            assert result[1] == "עולם"
            assert result[2] == "בדיקה"

    @patch("openai.OpenAI")
    def test_too_few_translations_are_padded_not_fallback(self, mock_openai_class):
        """
        CRITICAL: When OpenAI returns too few translations, use what's available

        OLD BEHAVIOR (BAD): Return all original English text
        NEW BEHAVIOR (GOOD): Use available Hebrew
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # OpenAI returns only 2 translations instead of 3 (too few) - JSON format
        response_json = json.dumps(
            [
                {"id": 1, "translation": "שלום"},
                {"id": 2, "translation": "עולם"},
                # Missing id=3
            ]
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=response_json))]
        mock_client.chat.completions.create.return_value = mock_response

        mock_config = create_mock_config()

        with (
            patch("config.get_config", return_value=mock_config),
            patch("openai_rate_limiter.config", mock_config),
            patch("openai_rate_limiter.get_config", return_value=mock_config),
            patch("services.translation_services.config", mock_config),
        ):

            translator = OpenAITranslator()
            inputs = ["Hello", "World", "Test"]  # 3 inputs
            result = translator.translate_batch(inputs, "he")

            # Should have gotten at least the available translations
            assert len(result) == 3
            # First two should be Hebrew
            assert result[0] == "שלום"
            assert result[1] == "עולם"

    @patch("openai.OpenAI")
    def test_old_fallback_behavior_is_prevented(self, mock_openai_class):
        """
        REGRESSION TEST: Ensure we don't fall back to old behavior

        With JSON format and ID-based matching, translations are matched by ID
        not by position, which is more robust.
        """
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Test with IDs out of order - should still match correctly
        response_json = json.dumps(
            [
                {"id": 3, "translation": "בדיקה"},  # Out of order
                {"id": 1, "translation": "שלום"},
                {"id": 2, "translation": "עולם"},
            ]
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=response_json))]
        mock_client.chat.completions.create.return_value = mock_response

        mock_config = create_mock_config()

        with (
            patch("config.get_config", return_value=mock_config),
            patch("openai_rate_limiter.config", mock_config),
            patch("openai_rate_limiter.get_config", return_value=mock_config),
            patch("services.translation_services.config", mock_config),
        ):

            translator = OpenAITranslator()
            inputs = ["Hello", "World", "Test"]
            result = translator.translate_batch(inputs, "he")

            # CRITICAL: Should NOT return original English
            assert result != inputs, "REGRESSION: Returned original English!"

            # Should match by ID, so order should be correct
            assert len(result) == 3
            assert result[0] == "שלום"  # id=1
            assert result[1] == "עולם"  # id=2
            assert result[2] == "בדיקה"  # id=3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
