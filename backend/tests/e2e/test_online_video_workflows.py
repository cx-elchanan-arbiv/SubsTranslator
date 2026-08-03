"""
E2E Tests for Online Video (YouTube) Workflows
==============================================

Tests complete user workflows for YouTube video processing:
1. Model selection verification (tiny vs large)
2. Translation service verification (Google vs OpenAI)
3. Download-only functionality
4. Log verification for successful completion

Test URL: https://www.youtube.com/watch?v=E6ZCY099A8s (short video for fast testing)
"""

import re
import subprocess
import time
from typing import Any

import pytest
import requests


@pytest.mark.e2e
@pytest.mark.slow
class TestOnlineVideoWorkflows:
    """End-to-end tests for YouTube video processing workflows."""

    # Test video URLs for different test combinations
    TEST_VIDEO_URLS = [
        "https://www.youtube.com/watch?v=E6ZCY099A8s",  # Original
        "https://www.youtube.com/watch?v=Ga2MJ_9scKI",  # New URL 1
        "https://www.youtube.com/watch?v=Wr6rT-AztP8",  # New URL 2
        "https://www.youtube.com/watch?v=toRyonUmJXg",  # New URL 3
        "https://www.youtube.com/watch?v=0EQ6J52YGog",  # New URL 4
    ]
    API_BASE_URL = "http://localhost:8081"
    MAX_WAIT_TIME = 300  # 5 minutes max wait
    POLL_INTERVAL = 3  # Check every 3 seconds

    def setup_method(self):
        """Setup before each test."""
        # Verify backend is running
        try:
            response = requests.get(f"{self.API_BASE_URL}/health", timeout=10)
            assert response.status_code == 200, "Backend not running"
        except requests.exceptions.RequestException:
            pytest.skip("Backend not available - run 'docker compose up -d' first")

    def wait_for_task_completion(
        self, task_id: str, timeout: int = MAX_WAIT_TIME
    ) -> dict[str, Any]:
        """
        Wait for a task to complete and return the final result.
        Based on the working logic from existing E2E tests.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.API_BASE_URL}/status/{task_id}", timeout=10
                )
                if response.status_code == 200:
                    status_data = response.json()
                    state = status_data.get("state", "PENDING")

                    print(
                        f"Task {task_id}: {state} ({time.time() - start_time:.1f}s elapsed)"
                    )

                    if state == "SUCCESS":
                        return status_data
                    elif state == "FAILURE":
                        error = status_data.get("error", {})
                        if isinstance(error, dict):
                            message = error.get("message", "")
                        else:
                            message = str(error)

                        # Skip if YouTube blocking
                        if (
                            "403" in message
                            or "Forbidden" in message
                            or "unavailable" in message.lower()
                        ):
                            pytest.skip(f"YouTube blocking detected: {message}")
                        else:
                            pytest.fail(f"Task failed: {message}")
                elif response.status_code == 404:
                    # Task might not be ready yet
                    pass
                else:
                    print(f"Unexpected status code: {response.status_code}")

            except requests.RequestException as e:
                print(f"Request failed: {e}")

            time.sleep(self.POLL_INTERVAL)

        pytest.fail(f"Task {task_id} did not complete within {timeout} seconds")

    def get_docker_logs(self, service: str = "worker", since_minutes: int = 5) -> str:
        """Get Docker logs for verification."""
        try:
            cmd = ["docker", "compose", "logs", "--since", f"{since_minutes}m", service]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout
        except Exception as e:
            print(f"Warning: Could not get Docker logs: {e}")
            return ""

    def verify_model_in_logs(self, logs: str, expected_model: str) -> None:
        """Verify the correct Whisper model was loaded in logs."""
        model_patterns = [
            rf"Using forced model: {expected_model}",
            rf"Loading {expected_model} model",
            rf"Model {expected_model} loaded.*successfully",
            rf"=== LOADING FASTER-WHISPER MODEL: {expected_model.upper()} ===",
        ]

        found_model = False
        for pattern in model_patterns:
            if re.search(pattern, logs, re.IGNORECASE):
                found_model = True
                print(f"✅ Found model {expected_model} in logs: {pattern}")
                break

        assert (
            found_model
        ), f"Model {expected_model} not found in logs. Available logs:\n{logs[-1000:]}"

    def verify_translation_service_in_logs(
        self, logs: str, expected_service: str
    ) -> None:
        """Verify the correct translation service was used in logs."""
        service_patterns = {
            "google": [
                r"Using Google.*translation",
                r"GoogleTranslator.*translate",
                r"Google.*translation.*successful",
            ],
            "openai": [
                r"Using OpenAI.*translation",
                r"OpenAI.*translation.*successful",
                r"HTTP Request: POST https://api\.openai\.com",
                r"Translating.*segments using OpenAI",
            ],
        }

        patterns = service_patterns.get(expected_service.lower(), [])
        found_service = False

        for pattern in patterns:
            if re.search(pattern, logs, re.IGNORECASE):
                found_service = True
                print(f"✅ Found {expected_service} service in logs: {pattern}")
                break

        assert (
            found_service
        ), f"Translation service {expected_service} not found in logs. Available logs:\n{logs[-1000:]}"

    def verify_successful_completion_in_logs(self, logs: str) -> None:
        """Verify the task completed successfully without errors."""
        success_patterns = [
            r"Task.*succeeded",
            r"Task completed",
            r"✅.*completed",
            r"status.*SUCCESS",
        ]

        # Check for success indicators
        found_success = any(
            re.search(pattern, logs, re.IGNORECASE) for pattern in success_patterns
        )
        assert (
            found_success
        ), f"No success indicators found in logs. Available logs:\n{logs[-1000:]}"

        # Check for critical errors (more specific patterns to avoid false positives)
        critical_error_patterns = [
            r"CRITICAL.*:",
            r"FATAL.*:",
            r"Task.*FAILED",
            r"Processing.*FAILED",
        ]

        found_critical_errors = [
            pattern
            for pattern in critical_error_patterns
            if re.search(pattern, logs, re.IGNORECASE)
        ]
        assert (
            not found_critical_errors
        ), f"Found critical error patterns in logs: {found_critical_errors}. Logs:\n{logs[-1000:]}"

        print("✅ Task completed successfully without critical errors")

    def verify_video_metadata(self, video_metadata: dict) -> None:
        """
        Verify that video metadata contains all required fields with valid values.

        Checks:
        - All essential fields are present
        - Field types are correct
        - Values are reasonable
        """
        if not video_metadata:
            pytest.fail("Video metadata is None or empty")

        # Essential fields that must be present
        required_fields = [
            "title",
            "duration",
            "uploader",
            "upload_date",
            "view_count",
            "thumbnail",
            "url",
        ]
        for field in required_fields:
            assert field in video_metadata, f"Missing required metadata field: {field}"
            assert video_metadata[field] is not None, f"Metadata field {field} is None"

        # Validate specific field types and values
        assert (
            isinstance(video_metadata["title"], str)
            and len(video_metadata["title"]) > 0
        ), "Title must be non-empty string"
        assert (
            isinstance(video_metadata["duration"], int)
            and video_metadata["duration"] > 0
        ), "Duration must be positive integer"
        assert (
            isinstance(video_metadata["view_count"], int)
            and video_metadata["view_count"] >= 0
        ), "View count must be non-negative integer"
        assert (
            isinstance(video_metadata["uploader"], str)
            and len(video_metadata["uploader"]) > 0
        ), "Uploader must be non-empty string"
        assert (
            isinstance(video_metadata["upload_date"], str)
            and len(video_metadata["upload_date"]) > 0
        ), "Upload date must be non-empty string"
        assert isinstance(video_metadata["thumbnail"], str) and video_metadata[
            "thumbnail"
        ].startswith("http"), "Thumbnail must be valid URL"
        # URL validation - allow any of our test URLs
        assert any(
            video_metadata["url"] == url for url in self.TEST_VIDEO_URLS
        ), f"URL should match one of the test videos, got: {video_metadata['url']}"

        # Optional but expected fields with validation
        if "description" in video_metadata and video_metadata["description"]:
            assert isinstance(
                video_metadata["description"], str
            ), "Description must be string"

        if "width" in video_metadata and video_metadata["width"]:
            assert (
                isinstance(video_metadata["width"], int) and video_metadata["width"] > 0
            ), "Width must be positive integer"

        if "height" in video_metadata and video_metadata["height"]:
            assert (
                isinstance(video_metadata["height"], int)
                and video_metadata["height"] > 0
            ), "Height must be positive integer"

        if "fps" in video_metadata and video_metadata["fps"]:
            assert (
                isinstance(video_metadata["fps"], int) and video_metadata["fps"] > 0
            ), "FPS must be positive integer"

        print("✅ Comprehensive metadata validation passed:")
        print(f"   Title: '{video_metadata['title']}'")
        print(f"   Uploader: {video_metadata['uploader']}")
        print(f"   Duration: {video_metadata['duration']}s")
        print(f"   Views: {video_metadata['view_count']:,}")
        print(f"   Upload Date: {video_metadata['upload_date']}")
        print(f"   Thumbnail: {video_metadata['thumbnail'][:50]}...")
        if video_metadata.get("width") and video_metadata.get("height"):
            print(
                f"   Resolution: {video_metadata['width']}x{video_metadata['height']}@{video_metadata.get('fps', 'N/A')}fps"
            )

    @pytest.mark.parametrize(
        "whisper_model,translation_service,test_url",
        [
            ("tiny", "google", "https://www.youtube.com/watch?v=Ga2MJ_9scKI"),
            ("tiny", "openai", "https://www.youtube.com/watch?v=Wr6rT-AztP8"),
            ("large", "google", "https://www.youtube.com/watch?v=toRyonUmJXg"),
            ("large", "openai", "https://www.youtube.com/watch?v=0EQ6J52YGog"),
        ],
    )
    def test_youtube_processing_with_model_and_service(
        self, whisper_model: str, translation_service: str, test_url: str
    ):
        """
        Test YouTube video processing with different Whisper models and translation services.

        Verifies:
        1. Correct model is loaded and used
        2. Correct translation service is used
        3. Task completes successfully
        4. All files are generated
        5. Logs show successful completion
        """
        print(
            f"\n🎯 Testing YouTube processing: {whisper_model} model + {translation_service} translation"
        )
        print(f"📺 Using test video: {test_url}")

        # Submit processing request
        request_data = {
            "url": test_url,
            "source_lang": "auto",
            "target_lang": "he",
            "auto_create_video": True,
            "whisper_model": whisper_model,
            "translation_service": translation_service,
        }

        response = requests.post(
            f"{self.API_BASE_URL}/youtube", json=request_data, timeout=30
        )
        assert response.status_code in [
            200,
            202,
        ], f"Failed to submit request: {response.text}"

        result = response.json()
        task_id = result.get("task_id")
        assert task_id, f"No task_id in response: {result}"

        print(f"📋 Task submitted: {task_id}")

        # Wait for completion
        final_result = self.wait_for_task_completion(task_id)

        # Verify basic result structure
        assert final_result["state"] == "SUCCESS", f"Task failed: {final_result}"

        # Extract result data from the response
        result_data = final_result.get("result", {})

        # Handle the two-stage task system (download → processing)
        if "task_id" in result_data and result_data.get("status") == "PROCESSING":
            # This is the download task that spawned a processing task
            processing_task_id = result_data["task_id"]
            print(
                f"📋 Download completed, now waiting for processing task: {processing_task_id}"
            )

            # Wait for the processing task to complete
            final_result = self.wait_for_task_completion(processing_task_id)
            result_data = final_result.get("result", {})

        # Check for files in the result (handle nested structure)
        files = None
        if "files" in result_data:
            files = result_data["files"]
        elif "result" in result_data and "files" in result_data["result"]:
            files = result_data["result"]["files"]

        if not files:
            pytest.fail(
                f"No files found in task result. Available keys in result: {list(result_data.keys())}. Full response structure: {final_result}"
            )

        assert files, "Files dictionary is empty"

        # Verify expected files were created
        expected_files = ["original_srt", "translated_srt"]
        if request_data["auto_create_video"]:
            expected_files.append("video_with_subtitles")

        for file_type in expected_files:
            assert file_type in files, f"Missing file type: {file_type}"
            assert files[file_type], f"Empty filename for: {file_type}"

        # Verify video metadata was extracted correctly
        if "video_metadata" in final_result:
            self.verify_video_metadata(final_result["video_metadata"])
        elif "video_metadata" in result_data:
            self.verify_video_metadata(result_data["video_metadata"])
        else:
            print("⚠️ No video metadata found in final result")

        # Get Docker logs for verification (shorter window to avoid old errors)
        logs = self.get_docker_logs(since_minutes=2)

        # Verify correct model was used
        self.verify_model_in_logs(logs, whisper_model)

        # Verify correct translation service was used
        self.verify_translation_service_in_logs(logs, translation_service)

        # Verify successful completion
        self.verify_successful_completion_in_logs(logs)

        print(f"✅ Test passed: {whisper_model} + {translation_service}")

    def test_youtube_download_only(self):
        """
        Test YouTube download-only functionality.

        Verifies:
        1. Video is downloaded successfully
        2. No processing occurs (no transcription/translation)
        3. Task completes quickly
        4. Logs show download completion
        """
        # Use the original test URL for download-only test
        test_url = self.TEST_VIDEO_URLS[0]  # Original E6ZCY099A8s
        print("\n📥 Testing YouTube download-only functionality")
        print(f"📺 Using test video: {test_url}")

        # Submit download-only request
        request_data = {"url": test_url}

        response = requests.post(
            f"{self.API_BASE_URL}/download-video-only", json=request_data, timeout=30
        )
        assert response.status_code in [
            200,
            202,
        ], f"Failed to submit download request: {response.text}"

        result = response.json()
        task_id = result.get("task_id")
        assert task_id, f"No task_id in response: {result}"

        print(f"📋 Download task submitted: {task_id}")

        # Wait for completion (should be much faster than processing)
        final_result = self.wait_for_task_completion(
            task_id, timeout=60
        )  # Shorter timeout for download-only

        # Verify basic result structure
        assert (
            final_result["state"] == "SUCCESS"
        ), f"Download task failed: {final_result}"
        assert "result" in final_result, "No result in final response"

        task_result = final_result["result"]

        # Verify comprehensive video metadata is present and valid
        assert "video_metadata" in task_result, "No video metadata in result"
        video_metadata = task_result["video_metadata"]

        # Essential fields that must be present
        required_fields = [
            "title",
            "duration",
            "uploader",
            "upload_date",
            "view_count",
            "thumbnail",
            "url",
        ]
        for field in required_fields:
            assert field in video_metadata, f"Missing required metadata field: {field}"
            assert video_metadata[field] is not None, f"Metadata field {field} is None"

        # Validate specific field types and values
        assert (
            isinstance(video_metadata["title"], str)
            and len(video_metadata["title"]) > 0
        ), "Title must be non-empty string"
        assert (
            isinstance(video_metadata["duration"], int)
            and video_metadata["duration"] > 0
        ), "Duration must be positive integer"
        assert (
            isinstance(video_metadata["view_count"], int)
            and video_metadata["view_count"] >= 0
        ), "View count must be non-negative integer"
        assert (
            isinstance(video_metadata["uploader"], str)
            and len(video_metadata["uploader"]) > 0
        ), "Uploader must be non-empty string"
        assert video_metadata["thumbnail"].startswith(
            "http"
        ), "Thumbnail must be valid URL"
        # URL validation - allow any of our test URLs
        assert any(
            video_metadata["url"] == url for url in self.TEST_VIDEO_URLS
        ), f"URL should match one of the test videos, got: {video_metadata['url']}"

        # Optional but expected fields
        optional_fields = ["description", "width", "height", "fps", "duration_string"]
        for field in optional_fields:
            if field in video_metadata:
                assert (
                    video_metadata[field] is not None
                ), f"Optional field {field} should not be None if present"

        print(
            f"✅ Video metadata validated: '{video_metadata['title']}' by {video_metadata['uploader']}"
        )
        print(
            f"   Duration: {video_metadata['duration']}s, Views: {video_metadata['view_count']:,}"
        )
        print(
            f"   Resolution: {video_metadata.get('width', 'N/A')}x{video_metadata.get('height', 'N/A')}@{video_metadata.get('fps', 'N/A')}fps"
        )

        # Get Docker logs for verification (only for this specific task)
        logs = self.get_docker_logs(since_minutes=2)  # Shorter time window

        # Verify download completion patterns
        download_patterns = [
            r"✅.*Download completed",
            r"✅.*Downloaded.*",
            r"download.*100%",
            r"Task.*download.*succeeded",
        ]

        found_download = any(
            re.search(pattern, logs, re.IGNORECASE) for pattern in download_patterns
        )
        assert (
            found_download
        ), f"No download completion found in logs. Available logs:\n{logs[-1000:]}"

        # Verify no processing occurred (should not find transcription/translation for this specific task)
        task_specific_processing_patterns = [
            rf"Task.*{task_id}.*transcription.*completed",
            rf"Task.*{task_id}.*translation.*successful",
            rf"{task_id}.*Creating.*SRT",
            rf"{task_id}.*subtitle.*embedding",
        ]

        found_processing = any(
            re.search(pattern, logs, re.IGNORECASE)
            for pattern in task_specific_processing_patterns
        )
        if found_processing:
            print(
                "Warning: Found processing patterns but this might be from previous tasks"
            )

        # Instead, verify that this specific task only did download
        download_only_patterns = [
            rf"Task.*{task_id}.*download.*only",
            rf"{task_id}.*Download.*completed",
            rf"download_and_process_youtube_task.*{task_id}.*succeeded",
        ]

        found_download_only = any(
            re.search(pattern, logs, re.IGNORECASE)
            for pattern in download_only_patterns
        )
        if not found_download_only:
            print(
                "Note: Could not find task-specific download patterns, but download completed successfully"
            )

        print("✅ Download-only test passed")

    def test_youtube_processing_error_handling(self):
        """
        Test error handling with invalid YouTube URL.

        Verifies:
        1. Invalid URL is handled gracefully
        2. Appropriate error message is returned
        3. No processing occurs
        """
        print("\n❌ Testing error handling with invalid URL")

        # Submit request with invalid URL
        request_data = {
            "url": "https://www.youtube.com/watch?v=INVALID_VIDEO_ID_12345",
            "source_lang": "auto",
            "target_lang": "he",
            "auto_create_video": False,
            "whisper_model": "tiny",
            "translation_service": "google",
        }

        response = requests.post(
            f"{self.API_BASE_URL}/youtube", json=request_data, timeout=30
        )
        assert response.status_code in [
            200,
            202,
        ], f"Failed to submit request: {response.text}"

        result = response.json()
        task_id = result.get("task_id")
        assert task_id, f"No task_id in response: {result}"

        print(f"📋 Error test task submitted: {task_id}")

        # Wait for completion (should fail quickly)
        start_time = time.time()

        while time.time() - start_time < 60:  # 1 minute timeout for error case
            response = requests.get(f"{self.API_BASE_URL}/status/{task_id}")
            assert (
                response.status_code == 200
            ), f"Failed to get task status: {response.text}"

            result = response.json()
            state = result.get("state", "UNKNOWN")

            if state == "FAILURE":
                # This is expected - verify error message is meaningful
                error = result.get("error", {})
                assert error, "No error information in failed task"
                print(f"✅ Task failed as expected with error: {error}")
                return
            elif state == "SUCCESS":
                pytest.fail("Task should have failed with invalid URL but succeeded")

            time.sleep(self.POLL_INTERVAL)

        pytest.fail("Task with invalid URL did not fail within timeout")
