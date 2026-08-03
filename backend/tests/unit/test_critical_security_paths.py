"""
Critical Security Path Tests
Tests that verify security-critical functions work correctly and fail safely.
"""

import os
import sys
import tempfile

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

    def test_file_upload_size_limits(self, monkeypatch):
        """An oversized upload is rejected with a JSON error, not a stack trace.

        The cap is lowered for the duration of the test instead of allocating
        ``MAX_FILE_SIZE + 1`` bytes for real: under the config this suite actually loads
        that was a 1 GB in-memory buffer on every run.
        """
        import io

        from app import app

        monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", 1024)

        with app.test_client() as client:
            data = {
                "file": (io.BytesIO(b"x" * 2048), "huge_file.mp4"),
                "source_lang": "auto",
                "target_lang": "he",
            }
            response = client.post(
                "/upload", data=data, content_type="multipart/form-data"
            )

            assert response.status_code in (413, 500), response.status_code

            payload = response.get_json()
            assert payload and "error" in payload
            error_message = payload["error"].lower()
            assert "too large" in error_message or "capacity limit" in error_message

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
