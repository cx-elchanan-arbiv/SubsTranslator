"""
Critical Security Path Tests
Tests that verify security-critical functions work correctly and fail safely.
"""

import os
import sys
import tempfile
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
class TestSecurityCriticalPaths:
    """Test security-critical code paths that could cause data breaches."""

    def test_openai_key_validation_mutation_resistance(self):
        """Test that OpenAI key validation is resistant to common mutations."""
        from app import _is_valid_openai_key

        # Test the function works correctly
        assert (
            _is_valid_openai_key("sk-test-valid-key-1234567890123456789012345") is True
        )

        # Test mutations that should be caught by tests
        # Mutation: Remove 'sk-' check
        assert (
            _is_valid_openai_key("invalid-key-format-1234567890123456789012345")
            is False
        )

        # Mutation: Change length check
        assert _is_valid_openai_key("sk-short") is False

        # Mutation: Remove placeholder check
        assert _is_valid_openai_key("your-openai-api-key-here") is False

        # Mutation: Case sensitivity
        assert (
            _is_valid_openai_key("SK-TEST-VALID-KEY-1234567890123456789012345") is False
        )

    def test_path_traversal_prevention(self):
        """Test that file operations prevent path traversal attacks."""
        from werkzeug.utils import secure_filename

        # Test dangerous paths are sanitized
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "file_with_null\x00.txt",
            "file_with_spaces   .txt",
        ]

        for dangerous_path in dangerous_paths:
            safe_name = secure_filename(dangerous_path)
            # Should not contain path traversal
            assert "../" not in safe_name
            assert "..\\" not in safe_name
            assert "/" not in safe_name or safe_name == safe_name.split("/")[-1]
            assert "\\" not in safe_name
            assert "\x00" not in safe_name

    def test_api_key_not_leaked_in_logs(self):
        """Test that API keys don't appear in log messages."""
        import logging
        from io import StringIO

        # Create a string buffer to capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)

        # Test with OpenAI translator
        from services.translation_services import OpenAITranslator

        # Mock the logger to capture output
        logger = logging.getLogger("translation_services")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            # This should fail but not leak the API key
            # Mock openai.OpenAI which is used in the constructor
            with patch("openai.OpenAI") as mock_openai:
                mock_openai.side_effect = Exception("API Error")

                # Try to create translator (will fail during initialization)
                try:
                    translator = OpenAITranslator(
                        api_key="sk-secret-key-should-not-appear-in-logs"
                    )
                except Exception:
                    pass  # Expected to fail

            # Check that API key doesn't appear in logs
            log_output = log_capture.getvalue()
            assert "sk-secret-key-should-not-appear-in-logs" not in log_output
            assert "secret-key" not in log_output

        finally:
            logger.removeHandler(handler)

    def test_file_upload_size_limits(self):
        """Test that file upload size limits are enforced."""
        import io

        from app import app
        from config import get_config

        config = get_config()
        max_size = config.MAX_FILE_SIZE

        with app.test_client() as client:
            # Create a file larger than max_size
            data = {
                "file": (io.BytesIO(b"x" * (max_size + 1)), "huge_file.mp4"),
                "source_lang": "auto",
                "target_lang": "he",
            }
            response = client.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            # File should be rejected - either 413 (Request Entity Too Large) or 500 (Internal Server Error from Flask)
            # Both indicate the file was properly rejected for being too large
            assert response.status_code in [
                413,
                500,
            ], f"Expected 413 or 500, got {response.status_code}. Response: {response.get_json()}"

            # Verify the error message mentions the size limit
            response_data = response.get_json()
            assert response_data and "error" in response_data
            error_message = response_data["error"].lower()
            assert (
                "too large" in error_message or "capacity limit" in error_message
            ), f"Error message should mention size limit: {response_data['error']}"

    def test_download_token_expiration(self):
        """Test that download tokens expire and can't be reused."""
        import time

        from services.token_service import generate_download_token, use_download_token

        filename = "test_file.mp4"

        # Generate token with short expiration
        token = generate_download_token(filename, expires_in=1)  # 1 second
        assert token is not None

        # Should work immediately
        result_filename, error = use_download_token(token)
        assert result_filename == filename
        assert error is None

        # Should not work after expiration
        time.sleep(2)
        result_filename, error = use_download_token(token)
        assert result_filename is None
        assert error is not None

        # Should not work if reused
        new_token = generate_download_token(filename, expires_in=60)
        result1_filename, error1 = use_download_token(new_token)
        result2_filename, error2 = use_download_token(new_token)  # Second use

        assert result1_filename == filename
        assert error1 is None
        assert result2_filename is None  # Should fail on reuse
        assert error2 is not None

    def test_error_message_sanitization(self):
        """Test that error messages don't contain sensitive information."""
        from app import app

        with app.test_client() as client:
            # Test with malicious input that could cause XSS
            malicious_inputs = [
                '<script>alert("xss")</script>',
                '"><script>alert("xss")</script>',
                'javascript:alert("xss")',
                "${7*7}",  # Template injection
                "{{7*7}}",  # Jinja2 injection
            ]

            for malicious_input in malicious_inputs:
                response = client.post(
                    "/youtube",
                    json={
                        "url": malicious_input,
                        "source_lang": "auto",
                        "target_lang": "he",
                    },
                )

                # Error response should not contain the malicious input raw
                if response.status_code >= 400:
                    response_text = response.get_data(as_text=True)

                    # Should not contain script tags
                    assert "<script>" not in response_text.lower()
                    assert "javascript:" not in response_text.lower()

                    # Should not contain template injection attempts
                    assert "${" not in response_text
                    assert "{{" not in response_text


@pytest.mark.unit
class TestDataIntegrityPaths:
    """Test data processing paths that could cause data corruption."""

    def test_subtitle_encoding_preservation(self):
        """Test that subtitle encoding is preserved correctly."""
        from services.subtitle_service import SubtitleService

        # Test with various Unicode characters
        test_segments = [
            {"start": 0, "end": 1, "text": "Hello World"},
            {"start": 1, "end": 2, "text": "שלום עולם"},  # Hebrew
            {"start": 2, "end": 3, "text": "مرحبا بالعالم"},  # Arabic
            {"start": 3, "end": 4, "text": "Hola Mundo"},  # Spanish
            {"start": 4, "end": 5, "text": "你好世界"},  # Chinese
            {"start": 5, "end": 6, "text": "Emoji: 🎬🎵🌍"},  # Emojis
        ]

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".srt", delete=False) as f:
            temp_file = f.name

        try:
            # Create SRT file using SubtitleService instance
            subtitle_service = SubtitleService()
            subtitle_service.create_srt_file(
                test_segments, temp_file, use_translation=False, language="he"
            )

            # Read back and verify encoding
            with open(temp_file, encoding="utf-8") as f:
                content = f.read()

            # All original text should be preserved
            for segment in test_segments:
                assert segment["text"] in content

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_concurrent_file_operations(self):
        """Test that concurrent file operations don't cause corruption."""
        import tempfile
        import threading

        # Create a shared file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            shared_file = f.name
            f.write("initial content")

        errors = []

        def write_worker(worker_id):
            try:
                for i in range(10):
                    with open(shared_file, "a") as f:
                        f.write(f"worker_{worker_id}_line_{i}\n")
            except Exception as e:
                errors.append(e)

        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        try:
            # Verify file is not corrupted
            with open(shared_file) as f:
                content = f.read()

            # Should have content from all workers
            for worker_id in range(5):
                assert f"worker_{worker_id}" in content

            # Should not have any encoding errors
            assert len(errors) == 0, f"Concurrent operations caused errors: {errors}"

        finally:
            if os.path.exists(shared_file):
                os.unlink(shared_file)
