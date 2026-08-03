import importlib
import os
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _import_from_backend(module_name: str, backend_dir: str):
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    return importlib.import_module(module_name)


def test_google_translator_batch_success(monkeypatch):
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ts = _import_from_backend("services.translation_services", backend_dir)

    class DummyGT:
        def __init__(self, source=None, target=None):
            self.source = source
            self.target = target

        def translate_batch(self, texts):
            return [f"{t}-he" for t in texts]

    monkeypatch.setattr(ts, "DeepGoogleTranslator", DummyGT, raising=True)

    tr = ts.get_translator("google")
    out = tr.translate_batch(["a", "b", "c"], "he")
    assert out == ["a-he", "b-he", "c-he"]


def test_google_translator_fallback_on_exception(monkeypatch):
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ts = _import_from_backend("services.translation_services", backend_dir)

    class DummyGT:
        def __init__(self, source=None, target=None):
            pass

        def translate_batch(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr(ts, "DeepGoogleTranslator", DummyGT, raising=True)

    tr = ts.get_translator("google")
    inputs = ["x", "y"]
    out = tr.translate_batch(inputs, "he")
    # Fallback returns original texts
    assert out == inputs


# NOTE: The old mismatch fallback test was removed because it conflicts with the new
# smart mismatch handling logic. The new behavior tries to fix mismatches instead of
# falling back to original text. See test_openai_mismatch_protection.py for comprehensive
# tests of the new mismatch handling behavior.


def test_openai_translator_httpx_client_initialization(monkeypatch):
    """Test that OpenAI translator initializes with custom httpx client to avoid proxies issues."""
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ts = _import_from_backend("services.translation_services", backend_dir)

    # Track if httpx.Client was called and OpenAI was called with http_client
    httpx_client_called = False
    openai_called_with_http_client = False

    def mock_httpx_client(*args, **kwargs):
        nonlocal httpx_client_called
        httpx_client_called = True
        # Verify timeout is set
        assert "timeout" in kwargs
        assert kwargs["timeout"] == 30.0

        # Return a mock client
        class MockClient:
            pass

        return MockClient()

    def mock_openai_client(api_key=None, http_client=None, **kwargs):
        nonlocal openai_called_with_http_client
        # Verify that http_client is passed
        if http_client is not None:
            openai_called_with_http_client = True

        class MockOpenAI:
            pass

        return MockOpenAI()

    # Mock httpx module at import time
    import httpx

    original_httpx_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", mock_httpx_client)

    # Mock OpenAI constructor
    monkeypatch.setattr(ts.openai, "OpenAI", mock_openai_client)
    # Mock config
    monkeypatch.setattr(ts, "config", types.SimpleNamespace(OPENAI_API_KEY="sk-test"))

    # Create translator - this should call our mocks
    translator = ts.OpenAITranslator("sk-test")

    # Verify both httpx.Client was called and OpenAI was called with http_client
    assert (
        httpx_client_called
    ), "httpx.Client should be called during OpenAI translator initialization"
    assert (
        openai_called_with_http_client
    ), "OpenAI should be initialized with http_client parameter"
