"""
Shared test fixtures for all test levels.
"""

import os
import shutil
import tempfile
import time as _real_time
import types

import pytest

# Set environment variables at module import time (before app is imported)
os.environ["TESTING"] = "1"
os.environ["DISABLE_RATE_LIMIT"] = "1"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"


@pytest.fixture
def stub_openai_rate_limiter(monkeypatch):
    """Isolate the OpenAI translation tests from Redis and from real backoff sleeps.

    ``services.translation_services`` asks the process-wide
    ``openai_rate_limiter.rate_limiter`` singleton for TPM/RPM budget before every
    request, and reports progress back to it afterwards. Both calls talk to Redis.
    In a test process Redis is unreachable (``REDIS_URL`` above points at
    ``localhost:6379/15``, which nothing serves), so ``acquire_budget`` swallows the
    connection error and reports "no budget". The caller then sleeps 70 seconds three
    times over before giving up — which is why these files used to run for minutes and
    then fail.

    Budget accounting is not what these tests exercise, so grant it. Also replace the
    ``time`` reference *inside the two modules under test* with a shim whose ``sleep``
    is a no-op; scoping the shim to those two module namespaces keeps the real
    ``time.sleep`` intact for every other test in the session.

    Yields the list of sleep durations that were requested, so a test can assert on
    backoff behaviour without paying for it in wall-clock time.
    """
    import openai_rate_limiter
    from services import translation_services

    requested_sleeps: list[float] = []

    def _no_sleep(seconds):
        requested_sleeps.append(seconds)

    fake_time = types.SimpleNamespace(sleep=_no_sleep, time=_real_time.time)

    monkeypatch.setattr(translation_services, "time", fake_time)
    monkeypatch.setattr(openai_rate_limiter, "time", fake_time)

    # Redis-backed bookkeeping on the shared singleton — stub it out entirely.
    limiter = openai_rate_limiter.rate_limiter
    monkeypatch.setattr(limiter, "acquire_budget", lambda *a, **k: True)
    monkeypatch.setattr(limiter, "save_batch_progress", lambda *a, **k: None)
    monkeypatch.setattr(limiter, "load_batch_progress", lambda *a, **k: None)

    return requested_sleeps


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
