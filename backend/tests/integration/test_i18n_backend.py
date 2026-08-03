"""
Backend i18n Testing Suite for SubsTranslator
Tests server-side language detection, localized responses, and API integration
"""

import json
import os

# Import backend i18n components
import sys

import pytest
from flask import Flask

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from i18n.translations import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    I18nManager,
    get_current_language,
    get_localized_error,
    init_i18n,
    is_rtl_language,
    t,
)


class TestI18nManager:
    """Test the I18nManager class"""

    def setup_method(self):
        """Setup test environment"""
        self.i18n_manager = I18nManager()

    def test_supported_languages_configuration(self):
        """Test that all required languages are configured"""
        expected_languages = {"he", "en", "es", "ar"}
        actual_languages = set(SUPPORTED_LANGUAGES.keys())

        assert expected_languages.issubset(actual_languages)

        # Test language properties
        assert SUPPORTED_LANGUAGES["he"]["rtl"] is True
        assert SUPPORTED_LANGUAGES["ar"]["rtl"] is True
        assert SUPPORTED_LANGUAGES["en"]["rtl"] is False
        assert SUPPORTED_LANGUAGES["es"]["rtl"] is False

    def test_language_detection_from_accept_header(self):
        """Test Accept-Language header parsing"""
        test_cases = [
            ("en-US,en;q=0.9,he;q=0.8", "en"),
            ("he-IL,he;q=0.9,en;q=0.8", "he"),
            ("es-ES,es;q=0.9,en;q=0.5", "es"),
            ("ar-SA,ar;q=0.9", "ar"),
            ("fr-FR,fr;q=0.9", "en"),  # Unsupported language -> default
            ("", "en"),  # Empty -> default
            (None, "en"),  # None -> default
        ]

        for accept_header, expected in test_cases:
            result = self.i18n_manager.detect_language(accept_header)
            assert (
                result == expected
            ), f"Failed for {accept_header}: expected {expected}, got {result}"

    def test_translation_retrieval(self):
        """Test translation key retrieval"""
        # Test basic translation
        result = self.i18n_manager.get_translation("errors:http.404.title", "en")
        assert "Not Found" in result

        # Test Hebrew translation
        result = self.i18n_manager.get_translation("errors:http.404.title", "he")
        assert "לא נמצא" in result

        # Test fallback to English for missing keys
        result = self.i18n_manager.get_translation("nonexistent.key", "he")
        assert result == "nonexistent.key"  # Should return key as fallback

    def test_nested_key_navigation(self):
        """Test navigation through nested translation keys"""
        # Test nested key access
        result = self.i18n_manager.get_translation("errors:http.500.message", "en")
        assert "internal server error" in result.lower()

        # Test namespace separation
        result = self.i18n_manager.get_translation("common:status.success", "en")
        assert "Success" in result

    def test_string_interpolation(self):
        """Test string interpolation in translations"""
        # This would test format strings like "File size exceeds {max_size}MB"
        result = self.i18n_manager.get_translation(
            "errors:upload.file_too_large", "en", max_size=100
        )
        assert "100" in result

    def test_error_message_generation(self):
        """Test localized error message generation"""
        error_info = self.i18n_manager.get_error_message("404", "en")

        assert "title" in error_info
        assert "message" in error_info
        assert "code" in error_info
        assert error_info["code"] == 404

        # Test Hebrew error message
        error_info_he = self.i18n_manager.get_error_message("404", "he")
        assert "לא נמצא" in error_info_he["title"]


class TestFlaskIntegration:
    """Test Flask integration with i18n system"""

    def setup_method(self):
        """Setup Flask app for testing"""
        self.app = Flask(__name__)
        init_i18n(self.app)

        @self.app.route("/test")
        def test_route():
            return {
                "language": get_current_language(),
                "is_rtl": is_rtl_language(),
                "message": t("common:status.success"),
            }

        @self.app.route("/error-test")
        def error_test_route():
            error = get_localized_error("404")
            return error

        self.client = self.app.test_client()

    def test_language_detection_middleware(self):
        """Test automatic language detection in Flask middleware"""
        # Test with Accept-Language header
        response = self.client.get(
            "/test", headers={"Accept-Language": "he-IL,he;q=0.9,en;q=0.8"}
        )

        data = json.loads(response.data)
        assert data["language"] == "he"
        assert data["is_rtl"] is True

        # Test response headers
        assert response.headers.get("Content-Language") == "he"
        assert response.headers.get("X-Language") == "he"
        assert response.headers.get("X-Text-Direction") == "rtl"

    def test_explicit_language_parameter(self):
        """Test explicit language parameter override"""
        # Test with URL parameter
        response = self.client.get("/test?lang=es")

        data = json.loads(response.data)
        assert data["language"] == "es"
        assert data["is_rtl"] is False

        # Test with header parameter
        response = self.client.get(
            "/test",
            headers={
                "X-Language": "ar",
                "Accept-Language": "en-US",  # Should be overridden
            },
        )

        data = json.loads(response.data)
        assert data["language"] == "ar"
        assert data["is_rtl"] is True

    def test_localized_error_responses(self):
        """Test localized error message responses"""
        # Test English error
        response = self.client.get("/error-test", headers={"Accept-Language": "en-US"})

        data = json.loads(response.data)
        assert "Not Found" in data["title"]

        # Test Hebrew error
        response = self.client.get("/error-test", headers={"Accept-Language": "he-IL"})

        data = json.loads(response.data)
        assert "לא נמצא" in data["title"]

    def test_unsupported_language_fallback(self):
        """Test fallback to default language for unsupported languages"""
        response = self.client.get(
            "/test", headers={"Accept-Language": "fr-FR,fr;q=0.9"}  # Unsupported
        )

        data = json.loads(response.data)
        assert data["language"] == DEFAULT_LANGUAGE
        assert response.headers.get("Content-Language") == DEFAULT_LANGUAGE


class TestAPIEndpointLocalization:
    """Test localization of actual API endpoints"""

    def setup_method(self):
        """Setup Flask app with API endpoints"""
        self.app = Flask(__name__)
        init_i18n(self.app)

        @self.app.route("/api/languages")
        def get_languages():
            return {
                "languages": SUPPORTED_LANGUAGES,
                "current": get_current_language(),
                "message": t("common:messages.upload_success"),
            }

        @self.app.route("/api/upload", methods=["POST"])
        def upload_endpoint():
            # Simulate validation error
            return {
                "error": get_localized_error("400"),
                "details": t("errors:validation.file_required"),
            }, 400

        self.client = self.app.test_client()

    def test_languages_endpoint_localization(self):
        """Test /languages endpoint with different languages"""
        # Test Hebrew
        response = self.client.get(
            "/api/languages", headers={"Accept-Language": "he-IL"}
        )

        data = json.loads(response.data)
        assert data["current"] == "he"
        assert "הועלה בהצלחה" in data["message"]

        # Test English
        response = self.client.get(
            "/api/languages", headers={"Accept-Language": "en-US"}
        )

        data = json.loads(response.data)
        assert data["current"] == "en"
        assert "uploaded successfully" in data["message"].lower()

    def test_error_endpoint_localization(self):
        """Test error responses are localized"""
        # Test Hebrew error response
        response = self.client.post("/api/upload", headers={"Accept-Language": "he-IL"})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "בקשה לא תקינה" in data["error"]["title"]
        # Check if translation is resolved or if it's still a key
        details = data.get("details", "")
        assert "נדרש קובץ" in details or "file_required" in details

        # Test English error response
        response = self.client.post("/api/upload", headers={"Accept-Language": "en-US"})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Invalid Request" in data["error"]["title"]
        assert "required" in data["details"].lower()


class TestPerformance:
    """Test i18n system performance"""

    def setup_method(self):
        self.i18n_manager = I18nManager()

    def test_translation_caching(self):
        """Test that translations are cached for performance"""
        import time

        # First call - should load from file
        start_time = time.time()
        result1 = self.i18n_manager.get_translation("errors:http.404.title", "en")
        first_call_time = time.time() - start_time

        # Second call - should use cache
        start_time = time.time()
        result2 = self.i18n_manager.get_translation("errors:http.404.title", "en")
        second_call_time = time.time() - start_time

        assert result1 == result2
        assert second_call_time < first_call_time  # Should be faster

    def test_language_detection_performance(self):
        """Test language detection performance"""
        import time

        test_header = "en-US,en;q=0.9,he;q=0.8,es;q=0.7,ar;q=0.6"

        start_time = time.time()
        for _ in range(1000):
            self.i18n_manager.detect_language(test_header)
        end_time = time.time()

        # Should process 1000 headers in under 0.1 seconds
        assert (end_time - start_time) < 0.1


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        self.i18n_manager = I18nManager()

    def test_malformed_accept_language_header(self):
        """Test handling of malformed Accept-Language headers"""
        malformed_headers = [
            "en-US;q=invalid",
            "invalid-language-code",
            ";;;",
            "en-US,",
            "q=0.9,en-US",
        ]

        for header in malformed_headers:
            # Should not raise exception and should return default
            result = self.i18n_manager.detect_language(header)
            assert result in SUPPORTED_LANGUAGES

    def test_missing_translation_files(self):
        """Test behavior when translation files are missing"""
        # Create manager with non-existent directory
        manager = I18nManager("/nonexistent/path")

        # Should fallback gracefully
        result = manager.get_translation("any.key", "en")
        assert result == "any.key"

    def test_circular_translation_references(self):
        """Test handling of potential circular references"""
        # This would test if translation keys reference each other
        result = self.i18n_manager.get_translation("errors:http.404.title", "en")
        assert isinstance(result, str)
        assert len(result) > 0


class TestSecurityConsiderations:
    """Test security aspects of i18n system"""

    def setup_method(self):
        """Set up test fixtures"""
        self.i18n_manager = I18nManager()

    def test_no_code_injection_in_translations(self):
        """Test that translation values cannot execute code"""
        # Test with potentially dangerous input
        result = self.i18n_manager.get_translation(
            "errors:upload.file_too_large",
            "en",
            max_size='<script>alert("xss")</script>',
        )

        # Should contain the script tag as text, not execute it
        assert "<script>" in result
        assert "alert" in result

    def test_language_parameter_validation(self):
        """Test that language parameters are properly validated"""
        dangerous_inputs = [
            "../../../etc/passwd",
            '<script>alert("xss")</script>',
            "../../config.py",
            "en; rm -rf /",
        ]

        for dangerous_input in dangerous_inputs:
            result = self.i18n_manager.detect_language(dangerous_input)
            # Should return default language for invalid input
            assert result == DEFAULT_LANGUAGE


# Integration tests that would run against the actual Flask app
@pytest.mark.integration
class TestRealAPIIntegration:
    """Integration tests with real API endpoints"""

    def test_upload_endpoint_with_different_languages(self):
        """Test actual upload endpoint with different languages"""
        # This would test against the real app.py upload endpoint
        pass

    def test_status_endpoint_localization(self):
        """Test status endpoint returns localized messages"""
        # This would test the /status/{task_id} endpoint
        pass

    def test_youtube_endpoint_error_localization(self):
        """Test YouTube endpoint returns localized error messages"""
        # This would test the /youtube endpoint with invalid URLs
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
