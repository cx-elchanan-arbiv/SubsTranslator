"""The targeted retry for segments OpenAI failed to return.

When a JSON batch response is missing an id, the translator re-asks for just that id
rather than re-running the batch or falling back to the source text. This is the only
test that drives that second request; ``test_openai_mismatch_protection.py`` covers what
happens when the retry ALSO comes back short.

The file's other three tests are gone: one duplicated the prompt-shape assertion in
``test_segment_batching.py``, one monkeypatched ``logging.Logger.info`` globally and then
asserted the word "batch" appeared somewhere, and one ran a list comprehension defined
inside the test itself and asserted its result — no backend code involved.
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
# sleeps; see tests/conftest.py.
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

            # Exactly two calls: the first batch, then the targeted retry for id=3.
            # The old assertion here was `call_count >= 1`, which held even if the retry
            # never fired — i.e. it could not fail on the bug it was written for.
            assert mock_client.chat.completions.create.call_count == 2

            # And the retried segment is the one that came back, in its slot.
            assert result == ["תרגום ראשון", "תרגום שני", "תרגום שלישי"]
