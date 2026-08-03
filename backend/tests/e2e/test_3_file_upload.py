"""
Test 3: File Upload Check
Verifies that file upload functionality works.
"""

from io import BytesIO
from unittest.mock import patch


def test_upload_endpoint_exists():
    """Test that upload endpoint exists"""
    from app import app

    with app.test_client() as client:
        # Check without file - should return error but not 404
        response = client.post("/upload")
        assert response.status_code != 404


def test_file_upload_validation():
    """Test that file upload has validation"""
    from app import app

    with app.test_client() as client:
        # Invalid file
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"fake content"), "test.txt")},
            content_type="multipart/form-data",
        )

        # Should return error for unsupported format
        assert response.status_code in [400, 415, 422]


def test_valid_file_format_accepted():
    """Test that valid file format is accepted"""
    from app import app

    with app.test_client() as client:
        with patch("app.process_video_task") as mock_task:
            mock_task.apply_async.return_value.id = "test-123"

            response = client.post(
                "/upload",
                data={
                    "file": (BytesIO(b"fake video content"), "test.mp4"),
                    "source_lang": "auto",
                    "target_lang": "he",
                },
                content_type="multipart/form-data",
            )

            # Should be accepted and return task_id
            assert response.status_code in [200, 202]
            data = response.get_json()
            assert "task_id" in data
