"""
Test 4: YouTube Download Check
Verifies that YouTube download functionality works.
"""

from unittest.mock import patch


def test_youtube_endpoint_exists():
    """Test that YouTube endpoint exists"""
    from app import app

    with app.test_client() as client:
        response = client.post("/youtube")
        # Not 404 - endpoint exists
        assert response.status_code != 404


def test_youtube_url_validation():
    """Test that YouTube URLs are validated"""
    from app import app

    with app.test_client() as client:
        # Invalid URL
        response = client.post("/youtube", json={"url": "not-a-valid-url"})

        # Should return error
        assert response.status_code == 400


def test_youtube_valid_request():
    """Test that valid request is accepted"""
    from app import app

    with app.test_client() as client:
        with patch("app.download_and_process_youtube_task") as mock_task:
            mock_task.apply_async.return_value.id = "youtube-test-123"

            response = client.post(
                "/youtube",
                json={
                    "url": "https://www.youtube.com/watch?v=test123",
                    "source_lang": "auto",
                    "target_lang": "he",
                    "whisper_model": "base",
                    "translation_service": "google",
                },
            )

            # Should be accepted
            assert response.status_code in [200, 202]
            data = response.get_json()
            assert "task_id" in data
