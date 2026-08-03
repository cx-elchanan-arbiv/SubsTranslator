"""
Shared test fixtures for all test levels.
"""

import os
import shutil
import tempfile

import pytest

# Set environment variables at module import time (before app is imported)
os.environ["TESTING"] = "1"
os.environ["DISABLE_RATE_LIMIT"] = "1"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"


@pytest.fixture
def temp_dirs():
    """
    Create temporary directories for uploads and downloads.

    Yields:
        dict: Dictionary with 'root', 'uploads', and 'downloads' paths

    Cleanup:
        Automatically removes all created directories after test
    """
    root = tempfile.mkdtemp(prefix="substranslator_test_")
    uploads = os.path.join(root, "uploads")
    downloads = os.path.join(root, "downloads")

    os.makedirs(uploads)
    os.makedirs(downloads)

    yield {"root": root, "uploads": uploads, "downloads": downloads}

    # Cleanup
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mock_subprocess_success(monkeypatch):
    """
    Mock subprocess.run to return success.

    Returns a function that can be customized per test.
    """
    from types import SimpleNamespace

    def _mock_run(returncode=0, stdout="", stderr="", create_output_file=None):
        def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
            # If output file is specified in command, create it
            if create_output_file and os.path.exists(
                os.path.dirname(create_output_file)
            ):
                with open(create_output_file, "wb") as f:
                    f.write(b"\x00" * 2048)  # Write 2KB of data

            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

        return fake_run

    return _mock_run


@pytest.fixture
def mock_subprocess_timeout(monkeypatch):
    """Mock subprocess.run to raise TimeoutExpired."""
    import subprocess

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", timeout=600)

    return fake_run


@pytest.fixture
def flask_test_client(temp_dirs, monkeypatch):
    """
    Create Flask test client with proper test configuration.

    Disables rate limiting and configures test directories.
    """
    # Note: Env vars already set at module level, but monkeypatch ensures proper cleanup
    # Don't overwrite DISABLE_RATE_LIMIT since it's already set to "1" at module level

    import sys
    from pathlib import Path

    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))

    from app import app as flask_app

    flask_app.config["UPLOAD_FOLDER"] = temp_dirs["uploads"]
    flask_app.config["DOWNLOADS_FOLDER"] = temp_dirs["downloads"]
    flask_app.config["TESTING"] = True
    flask_app.config["RATELIMIT_ENABLED"] = False

    with flask_app.test_client() as c:
        yield c
