"""The instant-preview endpoint — the Telegram trick, server side.

One key-less oEmbed call proxied through the backend (the browser cannot make
it itself for CORS reasons). The id is validated to exactly the YouTube shape
so nothing user-controlled reaches the outgoing URL.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

pytestmark = pytest.mark.unit


@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _oembed_response(status=200, payload=None):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload or {}
    return mock


def test_valid_id_returns_title_author_thumbnail(flask_client):
    with patch("requests.get") as get:
        get.return_value = _oembed_response(
            payload={
                "title": "Socialism creates ruination",
                "author_name": "Fox News",
                "thumbnail_url": "https://i.ytimg.com/vi/7duOWdEH2K0/hqdefault.jpg",
            }
        )
        response = flask_client.get("/youtube-preview?video_id=7duOWdEH2K0")

    assert response.status_code == 200
    data = response.get_json()
    assert data["title"] == "Socialism creates ruination"
    assert data["author"] == "Fox News"
    assert "7duOWdEH2K0" in data["thumbnail"]


@pytest.mark.parametrize(
    "bad_id",
    [
        "short",  # wrong length
        "../../etc/passwd",  # traversal junk
        "a" * 50,  # too long
        "12345678901;",  # right length, illegal char
        "",  # empty
    ],
)
def test_anything_but_an_exact_youtube_id_is_rejected(flask_client, bad_id):
    response = flask_client.get(f"/youtube-preview?video_id={bad_id}")
    assert response.status_code == 400


def test_private_or_removed_video_is_a_quiet_404(flask_client):
    """No oEmbed for private videos — the form just skips the card."""
    with patch("requests.get") as get:
        get.return_value = _oembed_response(status=404)
        response = flask_client.get("/youtube-preview?video_id=7duOWdEH2K0")
    assert response.status_code == 404
