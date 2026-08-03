"""The app boots and answers /health.

Deliberately one test. The two that used to sit here — ``test_backend_imports`` and
``test_config_loads`` — asserted that ``import app`` succeeded and that ``get_config()``
had the attributes it is declared with. Neither can fail while any other test in the
suite collects, because collection imports the same modules first.
"""

import pytest


@pytest.mark.integration
def test_health_endpoint_reports_healthy():
    """Covers the wiring no unit test does: blueprints registered, ffmpeg present."""
    from app import app

    with app.test_client() as client:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["ffmpeg_installed"] is True
