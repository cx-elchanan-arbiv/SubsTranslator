"""
Comprehensive tests for the summary endpoint
Tests the complete flow of summary generation with proper result structure mocking
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from celery.result import AsyncResult


@pytest.fixture
def app_client():
    """Create test client with proper configuration"""
    os.environ["FLASK_TESTING"] = "1"
    os.environ["TESTING"] = "true"
    os.environ["DISABLE_RATE_LIMIT"] = "1"

    from app import app

    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"

    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_translated_srt_file():
    """Create a temporary SRT file with realistic content"""
    srt_content = """1
00:00:00,000 --> 00:00:05,000
טראמפ אמר שישראל צריכה לתקוף את עזה

2
00:00:05,000 --> 00:00:10,000
לאחר שחמאס האשים בהרג חייל ישראלי

3
00:00:10,000 --> 00:00:15,000
זהו מבחן למדיניות החוץ של ארצות הברית
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(srt_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_celery_result_success(mock_translated_srt_file):
    """
    Mock Celery AsyncResult with realistic nested structure

    This mimics the actual structure returned by tasks.py:
    {
        "status": "SUCCESS",
        "result": {
            "files": {"translated_srt": "filename.srt"},
            "user_choices": {...}
        }
    }
    """
    mock_result = Mock(spec=AsyncResult)
    mock_result.state = "SUCCESS"

    # Nested structure as returned by Celery task
    mock_result.result = {
        "status": "SUCCESS",
        "result": {
            "title": "Test Video",
            "detected_language": "en",
            "files": {
                "original_srt": "test_original.srt",
                "translated_srt": os.path.basename(mock_translated_srt_file),
                "video_with_subtitles": "test_video.mp4",
            },
            "user_choices": {
                "source_lang": "en",
                "target_lang": "he",
                "whisper_model": "medium",
                "translation_service": "google",
            },
            "timing_summary": {
                "download_video": "5.2s",
                "transcription": "10.5s",
                "translation": "3.8s",
            },
        },
    }

    return mock_result


def test_summary_endpoint_with_nested_result_structure(
    app_client, mock_celery_result_success, mock_translated_srt_file
):
    """
    Test that the summary endpoint correctly handles nested Celery result structure

    This is the critical test that verifies:
    1. Endpoint extracts files from nested result.result.files (not result.files)
    2. Handles the {"status": "SUCCESS", "result": {...}} wrapper
    3. Successfully retrieves translated_srt filename
    """
    with patch(
        "api.summary_routes.AsyncResult", return_value=mock_celery_result_success
    ):
        # Mock downloads folder to point to temp file directory
        with patch(
            "api.summary_routes.config.DOWNLOADS_FOLDER",
            os.path.dirname(mock_translated_srt_file),
        ):
            with patch("api.summary_routes._is_valid_openai_key", return_value=True):
                with patch(
                    "api.summary_routes._generate_summary_with_openai",
                    return_value="## סיכום\n\nטסט",
                ):

                    response = app_client.post(
                        "/api/summaries",
                        json={"task_id": "test-task-123", "summary_lang": "he"},
                        content_type="application/json",
                    )

                    # Should succeed now with the fix
                    assert (
                        response.status_code == 200
                    ), f"Expected 200, got {response.status_code}: {response.get_json()}"

                    data = response.get_json()
                    assert data["success"] is True
                    assert "summary" in data
                    assert data["task_id"] == "test-task-123"
                    assert data["summary_lang"] == "he"


def test_summary_endpoint_language_routing(
    app_client, mock_celery_result_success, mock_translated_srt_file
):
    """
    Test that summaries are generated in the correct language
    Tests all 13 supported languages
    """
    supported_languages = [
        "he",
        "en",
        "es",
        "ar",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "ja",
        "ko",
        "zh",
        "tr",
    ]

    for lang in supported_languages:
        with patch(
            "api.summary_routes.AsyncResult", return_value=mock_celery_result_success
        ):
            with patch(
                "api.summary_routes.config.DOWNLOADS_FOLDER",
                os.path.dirname(mock_translated_srt_file),
            ):
                with patch(
                    "api.summary_routes._is_valid_openai_key", return_value=True
                ):
                    with patch(
                        "api.summary_routes._generate_summary_with_openai"
                    ) as mock_generate:
                        mock_generate.return_value = f"## Summary in {lang}"

                        response = app_client.post(
                            "/api/summaries",
                            json={"task_id": f"test-task-{lang}", "summary_lang": lang},
                            content_type="application/json",
                        )

                        assert (
                            response.status_code == 200
                        ), f"Failed for language {lang}"

                        # Verify that _generate_summary_with_openai was called with correct language
                        mock_generate.assert_called_once()
                        args, kwargs = mock_generate.call_args
                        assert kwargs.get("lang") == lang or (
                            len(args) > 1 and args[1] == lang
                        ), f"Expected lang={lang} in call"


def test_summary_endpoint_missing_task_id(app_client):
    """Test error handling for missing task_id"""
    response = app_client.post(
        "/api/summaries", json={"summary_lang": "he"}, content_type="application/json"
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "task_id" in data["error"].lower()


def test_summary_endpoint_missing_summary_lang(app_client):
    """Test error handling for missing summary_lang"""
    response = app_client.post(
        "/api/summaries", json={"task_id": "test-123"}, content_type="application/json"
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "summary_lang" in data["error"].lower()


def test_summary_endpoint_invalid_language(app_client):
    """Test error handling for unsupported language"""
    response = app_client.post(
        "/api/summaries",
        json={"task_id": "test-123", "summary_lang": "invalid_lang"},
        content_type="application/json",
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Invalid summary_lang" in data["error"]


def test_summary_endpoint_task_not_complete(app_client):
    """Test error when task is still processing"""
    mock_result = Mock(spec=AsyncResult)
    mock_result.state = "PROGRESS"

    with patch("api.summary_routes.AsyncResult", return_value=mock_result):
        with patch("api.summary_routes._is_valid_openai_key", return_value=True):
            response = app_client.post(
                "/api/summaries",
                json={"task_id": "test-123", "summary_lang": "he"},
                content_type="application/json",
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "not completed yet" in data["error"].lower()


def test_summary_endpoint_no_translated_file(app_client):
    """Test error when task has no translated subtitles"""
    mock_result = Mock(spec=AsyncResult)
    mock_result.state = "SUCCESS"

    # Result without translated_srt
    mock_result.result = {
        "status": "SUCCESS",
        "result": {
            "files": {
                "original_srt": "test.srt"
                # translated_srt is missing!
            }
        },
    }

    with patch("api.summary_routes.AsyncResult", return_value=mock_result):
        with patch("api.summary_routes._is_valid_openai_key", return_value=True):
            response = app_client.post(
                "/api/summaries",
                json={"task_id": "test-123", "summary_lang": "he"},
                content_type="application/json",
            )

            assert response.status_code == 404
            data = response.get_json()
            assert "No translated subtitles found" in data["error"]


def test_summary_prompts_structure():
    """Test that all language prompts are properly structured"""
    from api.summary_routes import SUMMARY_PROMPTS

    expected_languages = [
        "he",
        "en",
        "es",
        "ar",
        "fr",
        "de",
        "it",
        "pt",
        "ru",
        "ja",
        "ko",
        "zh",
        "tr",
    ]

    # All languages should be present
    assert set(SUMMARY_PROMPTS.keys()) == set(
        expected_languages
    ), f"Expected {expected_languages}, got {list(SUMMARY_PROMPTS.keys())}"

    # Each language should have system and user_template
    for lang, prompts in SUMMARY_PROMPTS.items():
        assert "system" in prompts, f"Language {lang} missing 'system' prompt"
        assert "user_template" in prompts, f"Language {lang} missing 'user_template'"
        assert (
            "{text}" in prompts["user_template"]
        ), f"Language {lang} user_template missing {{text}} placeholder"


def test_srt_text_extraction():
    """Test that SRT text extraction works correctly"""
    from api.summary_routes import _extract_text_from_srt

    # Create test SRT file
    srt_content = """1
00:00:00,000 --> 00:00:05,000
First subtitle line

2
00:00:05,000 --> 00:00:10,000
Second subtitle line
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(srt_content)
        temp_path = f.name

    try:
        extracted_text = _extract_text_from_srt(temp_path)

        # Should extract only text, not numbers or timestamps
        assert "First subtitle line" in extracted_text
        assert "Second subtitle line" in extracted_text
        assert "00:00:00" not in extracted_text
        assert "-->" not in extracted_text
        assert "1" not in extracted_text or extracted_text.count(
            "1"
        ) < srt_content.count("1")

    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
