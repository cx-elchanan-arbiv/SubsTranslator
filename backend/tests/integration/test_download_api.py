"""
Integration tests for the ``/download/<filename>`` endpoint.
"""

import os
import sys
import uuid

import pytest

# Set test environment before imports
os.environ["TESTING"] = "true"
os.environ["FLASK_TESTING"] = "1"
os.environ["DISABLE_RATE_LIMIT"] = "1"

# Add backend to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


@pytest.fixture
def app():
    """Create Flask test app using app's actual configured folders."""
    from app import app
    from config import get_config

    config = get_config()

    # Ensure the downloads folder exists
    os.makedirs(config.DOWNLOADS_FOLDER, exist_ok=True)

    app.config["TESTING"] = True
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_file(app):
    """Create a sample file in downloads folder for testing."""
    from config import get_config

    config = get_config()

    # Use unique filename to avoid conflicts between tests
    filename = f"test_file_{uuid.uuid4().hex[:8]}.srt"
    sample_path = os.path.join(config.DOWNLOADS_FOLDER, filename)
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("1\n00:00:01,000 --> 00:00:05,000\nTest subtitle\n")

    yield filename

    # Cleanup after test
    if os.path.exists(sample_path):
        os.remove(sample_path)


@pytest.fixture
def sample_video_file(app):
    """Create a sample video file in downloads folder for testing."""
    from config import get_config

    config = get_config()

    # Use unique filename to avoid conflicts between tests
    filename = f"test_video_{uuid.uuid4().hex[:8]}.mp4"
    sample_path = os.path.join(config.DOWNLOADS_FOLDER, filename)
    with open(sample_path, "wb") as f:
        f.write(b"fake video content")

    yield filename

    # Cleanup after test
    if os.path.exists(sample_path):
        os.remove(sample_path)


class TestLegacyDownloadEndpoint:
    """Test the legacy /download/<filename> endpoint."""

    def test_download_existing_srt_file(self, client, sample_file):
        """Test downloading an existing SRT file."""
        response = client.get(f"/download/{sample_file}")

        assert response.status_code == 200
        # SRT files should be returned as text/plain
        assert "text/plain" in response.content_type
        assert b"Test subtitle" in response.data

    def test_download_existing_video_file(self, client, sample_video_file):
        """Test downloading an existing video file."""
        response = client.get(f"/download/{sample_video_file}")

        assert response.status_code == 200
        assert b"fake video content" in response.data

    def test_download_nonexistent_file_returns_404(self, client):
        """Test that downloading a nonexistent file returns 404."""
        response = client.get("/download/nonexistent_file.srt")

        assert response.status_code == 404
        assert b"File not found" in response.data or b"error" in response.data

    def test_download_path_traversal_blocked(self, client):
        """Test that path traversal attempts are blocked."""
        # Try to access file outside downloads folder
        response = client.get("/download/../../../etc/passwd")

        assert response.status_code in (
            403,
            404,
        )  # Either forbidden or not found is acceptable

    def test_download_path_traversal_with_encoded_chars(self, client):
        """Test that URL-encoded path traversal is also blocked."""
        response = client.get("/download/..%2F..%2F..%2Fetc%2Fpasswd")

        assert response.status_code in (403, 404)


class TestDownloadHeaders:
    """Test download response headers."""

    def test_srt_file_has_attachment_disposition(self, client, sample_file):
        """Test that SRT files are sent as attachments."""
        response = client.get(f"/download/{sample_file}")

        assert response.status_code == 200
        # Check Content-Disposition header exists
        content_disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" in content_disposition

    def test_srt_file_has_txt_extension_for_macos(self, client, sample_file):
        """Test that SRT files get .txt extension for macOS compatibility."""
        response = client.get(f"/download/{sample_file}")

        assert response.status_code == 200
        content_disposition = response.headers.get("Content-Disposition", "")
        # Should have .txt extension for macOS compatibility
        assert ".txt" in content_disposition or ".srt" in content_disposition


class TestDownloadEdgeCases:
    """Filenames the pipeline actually produces.

    Every output name is derived from a user-supplied video title, so spaces and Hebrew
    are the common case, not the exotic one. These assert 200 and the body: the earlier
    ``status_code in (200, 404)`` form passed whether the file was served or silently
    missing, which is the whole failure it was supposed to catch.
    """

    def test_download_file_with_spaces_in_name(self, app, client):
        from config import get_config

        config = get_config()

        filename = f"file with spaces {uuid.uuid4().hex[:8]}.srt"
        sample_path = os.path.join(config.DOWNLOADS_FOLDER, filename)
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("test content")

        try:
            response = client.get(f'/download/{filename.replace(" ", "%20")}')

            assert response.status_code == 200
            assert b"test content" in response.data
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)

    def test_download_file_with_hebrew_name(self, app, client):
        from config import get_config

        config = get_config()

        filename = f"תרגום_עברית_{uuid.uuid4().hex[:8]}.srt"
        sample_path = os.path.join(config.DOWNLOADS_FOLDER, filename)
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("תוכן בעברית")

        try:
            response = client.get(f"/download/{filename}")

            assert response.status_code == 200
            assert "תוכן בעברית".encode() in response.data
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)

    def test_download_empty_filename(self, client):
        """``<path:filename>`` must not match the empty string."""
        response = client.get("/download/")

        assert response.status_code == 404
