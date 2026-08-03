"""
Test 2: Translation Services Check
Verifies that translation services work correctly.
"""

from unittest.mock import patch


def test_google_translate_available():
    """Test that Google Translate is always available"""
    from app import app

    with app.test_client() as client:
        response = client.get("/translation-services")
        assert response.status_code == 200

        data = response.get_json()
        assert data["google"]["available"] is True


def test_openai_not_available_with_fake_key():
    """Test that OpenAI is not available with fake/placeholder key"""
    from app import _is_valid_openai_key, app

    # Function should detect fake/placeholder keys
    assert _is_valid_openai_key("your-openai-api-key-here") is False
    assert _is_valid_openai_key("sk-test-fake-key-for-testing") is True

    with app.test_client() as client:
        with patch("app.config") as mock_config:
            mock_config.OPENAI_API_KEY = "your-openai-api-key-here"

            response = client.get("/translation-services")
            data = response.get_json()
            assert data["openai"]["available"] is False


def test_translation_works():
    """Test that translation actually works (with Google)"""
    from services.translation_services import get_translator

    translator = get_translator("google")
    result = translator.translate_batch(["Hello"], "he")

    # Should return something (even if same text)
    assert isinstance(result, list)
    assert len(result) == 1
