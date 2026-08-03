"""
Test 1: System Health Check
Verifies that the system starts up and is healthy.
"""


def test_backend_imports():
    """Test that all critical imports work"""
    from app import app
    from config import get_config
    from services.translation_services import get_translator

    assert app is not None
    assert get_config() is not None
    assert get_translator("google") is not None


def test_flask_app_starts():
    """Test that Flask application starts up"""
    from app import app

    with app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_config_loads():
    """Test that configuration loads correctly"""
    from config import get_config

    config = get_config()
    assert hasattr(config, "UPLOAD_FOLDER")
    assert hasattr(config, "DOWNLOADS_FOLDER")
    assert hasattr(config, "OPENAI_API_KEY")
