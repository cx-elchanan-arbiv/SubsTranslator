"""
Unit tests for the /translation-services API endpoint.
Tests the endpoint behavior with mocked dependencies.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Set test environment variables before any imports
os.environ["UPLOAD_FOLDER"] = "/tmp/test_uploads"
os.environ["DOWNLOADS_FOLDER"] = "/tmp/test_downloads"
os.environ["TESTING"] = "true"

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


@pytest.mark.unit
class TestTranslationServicesAPI:
    """Test the /translation-services endpoint behavior."""

    def test_translation_services_with_valid_openai_key(self):
        """Test that OpenAI is available when a valid key is configured."""
        from app import app

        with app.test_client() as client:
            with patch("api.health_routes.config") as mock_config:
                mock_config.OPENAI_API_KEY = (
                    "sk-test-valid-key-1234567890123456789012345"
                )

                response = client.get("/translation-services")
                assert response.status_code == 200

                data = response.get_json()
                assert "google" in data
                assert "openai" in data

                assert data["google"]["available"] is True
                assert data["openai"]["available"] is True
                assert "Advanced translation" in data["openai"]["description"]

    def test_translation_services_with_placeholder_key(self):
        """Test that OpenAI is NOT available when placeholder key is used."""
        from app import app

        with app.test_client() as client:
            with patch("api.health_routes.config") as mock_config:
                mock_config.OPENAI_API_KEY = "your-openai-api-key-here"

                response = client.get("/translation-services")
                assert response.status_code == 200

                data = response.get_json()
                assert "google" in data
                assert "openai" in data

                assert data["google"]["available"] is True
                assert data["openai"]["available"] is False
                assert "API key required" in data["openai"]["description"]


@pytest.mark.unit
class TestOpenAIKeyValidation:
    """Test the _is_valid_openai_key function."""

    def test_valid_openai_keys(self):
        """Test that valid OpenAI keys are recognized."""
        from app import _is_valid_openai_key

        valid_keys = [
            "sk-test-valid-key-1234567890123456789012345",
            "sk-test-fakekeyfortesting1234567890abcdef1234567890abcdef",
            "sk-test-anotherfakekey12345678901234567890123456789012345",
        ]

        for key in valid_keys:
            assert _is_valid_openai_key(key) is True, f"Key should be valid: {key}"

    def test_invalid_openai_keys(self):
        """Test that invalid/placeholder OpenAI keys are rejected."""
        from app import _is_valid_openai_key

        invalid_keys = [
            None,
            "",
            "your-openai-api-key-here",
            "your-api-key-here",
            "sk-your-key-here",
            "placeholder",
            "changeme",
            "replace-me",
            "invalid-key-format",
            "sk-",  # Too short
            "sk-short",  # Too short
            "not-starting-with-sk-but-long-enough-1234567890",
        ]

        for key in invalid_keys:
            assert _is_valid_openai_key(key) is False, f"Key should be invalid: {key}"

    def test_case_insensitive_placeholder_detection(self):
        """Test that placeholder detection is case insensitive."""
        from app import _is_valid_openai_key

        placeholders = [
            "Your-OpenAI-API-Key-Here",
            "YOUR-OPENAI-API-KEY-HERE",
            "PLACEHOLDER",
            "ChangeME",
            "Replace-Me",
        ]

        for placeholder in placeholders:
            assert (
                _is_valid_openai_key(placeholder) is False
            ), f"Placeholder should be detected: {placeholder}"


class TestCollapsedBatchGuard:
    """A broken response wearing a valid shape: Google once answered with its
    HTTP 500 error PAGE, and every line in the batch "translated" to that same
    page text — right count, text present, both existing guards blind. Caught
    live on a real run (19 of 30 cues shipped as "Error 500 (Server Error)!!1"
    inside a burned video). The invariant: distinct sources do not legitimately
    collapse to one identical output."""

    def test_error_page_repeated_for_distinct_lines_is_flagged(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        sources = ["We met for hours.", "It went well.", "They agreed to talk."]
        translations = ["Error 500 (Server Error)!!1"] * 3
        assert _batch_collapsed_to_one_answer(sources, translations) is True

    def test_identical_short_lines_collapsing_is_legitimate(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        # Three IDENTICAL sources may of course share one translation.
        sources = ["Yes.", "Yes.", "Yes."]
        translations = ["כן.", "כן.", "כן."]
        assert _batch_collapsed_to_one_answer(sources, translations) is False

    def test_a_normal_batch_passes(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        sources = ["Good morning.", "How are you?", "See you tomorrow."]
        translations = ["בוקר טוב.", "מה שלומך?", "נתראה מחר."]
        assert _batch_collapsed_to_one_answer(sources, translations) is False

    def test_two_repeats_are_not_enough_to_flag(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        sources = ["Yes!", "Yes.", "We signed the deal."]
        translations = ["כן.", "כן.", "חתמנו על העסקה."]
        assert _batch_collapsed_to_one_answer(sources, translations) is False


class TestTailBatchAndScriptGuards:
    """The 3-distinct-sources rule left a hole, found live: a 2-line FINAL batch
    broke entirely and two cues shipped as Google's error page. Two new layers,
    neither of which knows what an error page looks like."""

    def test_a_whole_tiny_batch_collapsing_to_one_long_string_is_flagged(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        page = "Error 500 (Server Error)!!1 That's an error. That's all we know."
        assert (
            _batch_collapsed_to_one_answer(
                ["We are nearly done.", "It was a historic day."], [page, page]
            )
            is True
        )

    def test_two_short_identical_answers_stay_legitimate(self):
        from services.translation_services import _batch_collapsed_to_one_answer

        # "Yes!" / "Yes." legitimately share one short translation.
        assert _batch_collapsed_to_one_answer(["Yes!", "Yes."], ["כן.", "כן."]) is False

    def test_long_line_without_hebrew_letters_looks_untranslated(self):
        from services.translation_services import _looks_untranslated

        assert (
            _looks_untranslated("There was an error. Please try again later.", "iw")
            is True
        )

    def test_short_latin_lines_are_exempt(self):
        from services.translation_services import _looks_untranslated

        # Names, numbers and interjections survive translation unchanged.
        assert _looks_untranslated("OK!", "iw") is False
        assert _looks_untranslated("ICC 2026", "iw") is False

    def test_real_hebrew_passes(self):
        from services.translation_services import _looks_untranslated

        assert _looks_untranslated("היום הייתה לנו פגישה מאוד טובה.", "iw") is False

    def test_latin_target_languages_are_left_alone(self):
        from services.translation_services import _looks_untranslated

        # For en/es/fr the check cannot tell translated from untranslated.
        assert (
            _looks_untranslated("This is a long English sentence here.", "en") is False
        )
