"""
Contract tests to ensure API schema stability during refactoring.
These tests verify that the response structure remains consistent.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_status_api_schema_structure():
    """Test that /status endpoint returns expected schema structure."""
    # Import after setting test environment
    import os
    import sys

    # Set test environment variables before importing app
    os.environ["UPLOAD_FOLDER"] = "/tmp/test_uploads"
    os.environ["DOWNLOADS_FOLDER"] = "/tmp/test_downloads"
    os.environ["TESTING"] = "true"

    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app import app

    with app.test_client() as client:
        # Mock AsyncResult to return predictable data
        with patch("api.video_routes.AsyncResult") as mock_async_result:
            mock_result = MagicMock()
            mock_result.state = "PROGRESS"
            mock_result.info = {
                "overall_percent": 45,
                "steps": [
                    {
                        "label": "Downloading",
                        "status": "completed",
                        "progress": 100,
                        "weight": 0.2,
                    },
                    {
                        "label": "Processing",
                        "status": "in_progress",
                        "progress": 50,
                        "weight": 0.8,
                    },
                ],
                "video_metadata": {
                    "title": "Test Video",
                    "duration": 120,
                    "duration_string": "00:02:00",
                },
                "user_choices": {
                    "source_lang": "en",
                    "target_lang": "he",
                    "auto_create_video": True,
                },
                "logs": ["Log entry 1", "Log entry 2"],
            }
            mock_async_result.return_value = mock_result

            response = client.get("/status/test-task-id")
            assert response.status_code == 200

            data = response.get_json()

            # Verify required top-level fields
            required_fields = [
                "task_id",
                "state",
                "progress",
                "video_metadata",
                "result",
                "user_choices",
                "initial_request",
                "error",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Verify progress structure
            progress = data["progress"]
            assert "overall_percent" in progress
            assert "steps" in progress
            assert isinstance(progress["steps"], list)

            # Verify step structure
            if progress["steps"]:
                step = progress["steps"][0]
                step_fields = ["label", "status", "progress", "weight"]
                for field in step_fields:
                    assert field in step, f"Missing step field: {field}"

            # Verify metadata structure (when present)
            if data["video_metadata"]:
                metadata = data["video_metadata"]
                metadata_fields = ["title", "duration", "duration_string"]
                for field in metadata_fields:
                    assert field in metadata, f"Missing metadata field: {field}"


@pytest.mark.unit
def test_youtube_endpoint_response_schema():
    """Test that /youtube endpoint returns expected response schema."""
    import os
    import sys

    # Set test environment variables before importing app
    os.environ["UPLOAD_FOLDER"] = "/tmp/test_uploads"
    os.environ["DOWNLOADS_FOLDER"] = "/tmp/test_downloads"
    os.environ["TESTING"] = "true"

    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app import app

    with app.test_client() as client:
        # Mock the Celery task
        with patch("api.video_routes.download_and_process_youtube_task") as mock_task:
            mock_task.apply_async.return_value.id = "test-task-123"

            response = client.post(
                "/youtube",
                json={
                    "url": "https://youtube.com/watch?v=test",
                    "source_lang": "en",
                    "target_lang": "he",
                    "auto_create_video": True,
                    "whisper_model": "base",  # Use valid model
                },
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 202
            data = response.get_json()

            # Verify unified task schema
            required_fields = [
                "task_id",
                "state",
                "user_choices",
                "initial_request",
                "video_metadata",
                "progress",
                "result",
                "error",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Verify initial state
            assert data["state"] == "PENDING"
            assert data["task_id"] == "test-task-123"
            assert data["user_choices"]["source_lang"] == "en"
            assert data["user_choices"]["target_lang"] == "he"


@pytest.mark.unit
def test_upload_endpoint_response_schema():
    """Test that /upload endpoint returns expected response schema."""
    import os
    import sys

    # Set test environment variables before importing app
    os.environ["UPLOAD_FOLDER"] = "/tmp/test_uploads"
    os.environ["DOWNLOADS_FOLDER"] = "/tmp/test_downloads"
    os.environ["TESTING"] = "true"

    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from io import BytesIO

    from app import app

    with app.test_client() as client:
        # Mock the Celery task and file probe
        with (
            patch("api.video_routes.process_video_task") as mock_task,
            patch("api.video_routes.probe_file_safe") as mock_probe,
        ):
            mock_task.apply_async.return_value.id = "test-upload-456"
            # Return valid file metadata to pass validation
            mock_probe.return_value = ({"duration": 120, "format": "mp4"}, None)

            # Create fake file upload
            data = {
                "file": (BytesIO(b"fake video content"), "test.mp4"),
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": "true",
                "whisper_model": "base",  # Use valid model
            }

            response = client.post(
                "/upload", data=data, content_type="multipart/form-data"
            )

            assert response.status_code == 202
            data = response.get_json()

            # Verify unified task schema (same as YouTube)
            required_fields = [
                "task_id",
                "state",
                "user_choices",
                "initial_request",
                "video_metadata",
                "progress",
                "result",
                "error",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # Verify upload-specific fields
            assert data["state"] == "PENDING"
            assert data["task_id"] == "test-upload-456"
            assert data["initial_request"]["type"] == "upload"
            assert data["initial_request"]["filename"] == "test.mp4"


@pytest.mark.unit
def test_error_response_schema():
    """Test that error responses maintain consistent schema."""
    import os
    import sys

    # Set test environment variables before importing app
    os.environ["UPLOAD_FOLDER"] = "/tmp/test_uploads"
    os.environ["DOWNLOADS_FOLDER"] = "/tmp/test_downloads"
    os.environ["TESTING"] = "true"

    backend_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app import app

    with app.test_client() as client:
        # Mock AsyncResult to avoid Redis connection
        with patch("api.video_routes.AsyncResult") as mock_async_result:
            mock_result = MagicMock()
            mock_result.state = "PENDING"
            mock_result.info = None
            mock_async_result.return_value = mock_result

            # Test invalid endpoint
            response = client.get("/status/invalid-task-id")

        # Should still return valid JSON with error info
        assert response.status_code == 200  # /status always returns 200
        data = response.get_json()

        # Even error responses should have the unified schema
        required_fields = [
            "task_id",
            "state",
            "progress",
            "video_metadata",
            "result",
            "user_choices",
            "initial_request",
            "error",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field in error response: {field}"


# Schema validation helper for future use
def validate_unified_task_schema(response_data):
    """Helper function to validate the unified task response schema."""
    required_fields = [
        "task_id",
        "state",
        "progress",
        "video_metadata",
        "result",
        "user_choices",
        "initial_request",
        "error",
    ]

    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

    # Validate progress structure
    progress = response_data["progress"]
    assert "overall_percent" in progress
    assert "steps" in progress
    assert isinstance(progress["steps"], list)

    return True
