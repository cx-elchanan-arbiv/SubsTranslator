"""Shared fixtures for the backend unit-test suite."""

import time as _real_time
import types

import pytest


@pytest.fixture
def stub_openai_rate_limiter(monkeypatch):
    """Isolate the OpenAI translation unit tests from Redis and from real backoff sleeps.

    ``services.translation_services`` asks the process-wide
    ``openai_rate_limiter.rate_limiter`` singleton for TPM/RPM budget before every
    request, and reports progress back to it afterwards. Both calls talk to Redis.
    In a unit-test process Redis is unreachable (``tests/conftest.py`` points
    ``REDIS_URL`` at ``localhost:6379/15``, which nothing serves), so
    ``acquire_budget`` swallows the connection error and reports "no budget". The
    caller then sleeps 70 seconds three times over before giving up — which is why
    these files used to run for 10-17 minutes and then fail.

    Budget accounting is not what these tests exercise, so grant it. Also replace
    the ``time`` reference *inside the two modules under test* with a shim whose
    ``sleep`` is a no-op; scoping the shim to those two module namespaces keeps the
    real ``time.sleep`` intact for every other test in the session.

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
