"""
E2E Test for Watermark Functionality
Tests that watermark is only added when explicitly enabled by user.
"""

import os
import re
import time
from typing import Any

import pytest
import requests


@pytest.mark.e2e
@pytest.mark.skip(
    reason="Watermark E2E tests are outdated - log extraction not working with current implementation"
)
class TestWatermarkE2E:
    """End-to-end tests for watermark functionality."""

    # Test video URL - Trump Purple Heart recipients video
    TEST_VIDEO_URL = "https://www.youtube.com/watch?v=DzjrqYn0do8"

    def setup_method(self):
        """Setup before each test."""
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

    def wait_for_task_completion(
        self, task_id: str, max_wait: int = 300
    ) -> dict[Any, Any]:
        """Wait for a task to complete and return the result."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"http://localhost:8081/status/{task_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            state = status_data.get("state", "PENDING")

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

            time.sleep(3)  # Check every 3 seconds

        pytest.fail(f"Task timed out after {max_wait} seconds")

    def extract_logs_from_task_result(self, task_result: dict[Any, Any]) -> str:
        """Extract logs from task result for analysis."""
        logs = []

        # Extract from progress steps
        progress = task_result.get("progress", {})
        steps = progress.get("steps", [])

        for step in steps:
            step_logs = step.get("logs", [])
            for log_entry in step_logs:
                if isinstance(log_entry, dict):
                    logs.append(log_entry.get("message", ""))
                else:
                    logs.append(str(log_entry))

        # Also check main logs if available
        main_logs = progress.get("logs", [])
        for log_entry in main_logs:
            if isinstance(log_entry, dict):
                logs.append(log_entry.get("message", ""))
            else:
                logs.append(str(log_entry))

        return "\n".join(logs)

    def test_watermark_disabled_by_default(self):
        """Test that watermark is NOT added when checkbox is unchecked (default)."""
        print(
            f"\n🎯 Testing watermark disabled (default) with video: {self.TEST_VIDEO_URL}"
        )

        # Submit YouTube video with watermark explicitly disabled
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": True,  # We need video to test watermark
                "whisper_model": "tiny",  # Use fastest model
                "translation_service": "google",
                "watermark_enabled": False,  # Explicitly disable watermark
            },
            timeout=10,
        )

        assert response.status_code in [200, 202], f"Request failed: {response.text}"
        data = response.json()
        assert "task_id" in data, "No task_id in response"

        task_id = data["task_id"]
        print(f"📋 Task ID: {task_id}")

        # Wait for completion
        task_result = self.wait_for_task_completion(task_id, max_wait=400)

        # Extract logs for analysis
        logs = self.extract_logs_from_task_result(task_result)
        print(f"\n📝 Task logs:\n{logs}")

        # Verify watermark was skipped
        assert (
            "Skipping watermark (disabled by user)" in logs
        ), "Expected 'Skipping watermark (disabled by user)' message not found in logs"

        # Verify watermark was NOT added
        assert (
            "Adding watermark to video" not in logs
        ), "Watermark was added even though it should be disabled"

        # Check timing summary shows watermark was skipped
        result = task_result.get("result", {})
        timing_summary = result.get("timing_summary", {})
        watermark_time = timing_summary.get("add_watermark", "")

        assert (
            "skipped" in watermark_time.lower()
        ), f"Expected watermark timing to show 'skipped', got: {watermark_time}"

        print("✅ Test passed: Watermark correctly skipped when disabled")

    def test_watermark_enabled_explicitly(self):
        """Test that watermark IS added when checkbox is checked."""
        print(f"\n🎯 Testing watermark enabled with video: {self.TEST_VIDEO_URL}")

        # Submit YouTube video with watermark explicitly enabled
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": True,  # We need video to test watermark
                "whisper_model": "tiny",  # Use fastest model
                "translation_service": "google",
                "watermark_enabled": True,  # Explicitly enable watermark
                "watermark_position": "bottom-right",
                "watermark_size": "medium",
            },
            timeout=10,
        )

        assert response.status_code in [200, 202], f"Request failed: {response.text}"
        data = response.json()
        assert "task_id" in data, "No task_id in response"

        task_id = data["task_id"]
        print(f"📋 Task ID: {task_id}")

        # Wait for completion
        task_result = self.wait_for_task_completion(task_id, max_wait=400)

        # Extract logs for analysis
        logs = self.extract_logs_from_task_result(task_result)
        print(f"\n📝 Task logs:\n{logs}")

        # Verify watermark was added
        assert (
            "Adding watermark and cleaning up" in logs
        ), "Expected 'Adding watermark and cleaning up' message not found in logs"

        # Verify watermark was NOT skipped
        assert (
            "Skipping watermark (disabled by user)" not in logs
        ), "Watermark was skipped even though it should be enabled"

        # Check timing summary shows actual watermark processing time
        result = task_result.get("result", {})
        timing_summary = result.get("timing_summary", {})
        watermark_time = timing_summary.get("add_watermark", "")

        assert (
            "skipped" not in watermark_time.lower()
        ), f"Expected watermark timing to show actual time, got: {watermark_time}"

        # Should have actual processing time (number)
        assert re.search(
            r"\d+\.\d+", watermark_time
        ), f"Expected numeric timing for watermark processing, got: {watermark_time}"

        print("✅ Test passed: Watermark correctly added when enabled")

    def test_watermark_default_behavior(self):
        """Test default behavior when watermark_enabled is not specified."""
        print(
            f"\n🎯 Testing watermark default behavior (not specified) with video: {self.TEST_VIDEO_URL}"
        )

        # Submit YouTube video without specifying watermark_enabled
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": True,  # We need video to test watermark
                "whisper_model": "tiny",  # Use fastest model
                "translation_service": "google",
                # Note: No watermark_enabled specified - should default to False
            },
            timeout=10,
        )

        assert response.status_code in [200, 202], f"Request failed: {response.text}"
        data = response.json()
        assert "task_id" in data, "No task_id in response"

        task_id = data["task_id"]
        print(f"📋 Task ID: {task_id}")

        # Wait for completion
        task_result = self.wait_for_task_completion(task_id, max_wait=400)

        # Extract logs for analysis
        logs = self.extract_logs_from_task_result(task_result)
        print(f"\n📝 Task logs:\n{logs}")

        # Default should be disabled (no watermark)
        assert (
            "Skipping watermark (disabled by user)" in logs
        ), "Expected watermark to be skipped by default when not specified"

        # Verify watermark was NOT added
        assert (
            "Adding watermark to video" not in logs
        ), "Watermark was added even though it should be disabled by default"

        print("✅ Test passed: Watermark correctly disabled by default")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skip(
    reason="Watermark file upload tests are outdated - log extraction not working with current implementation"
)
class TestWatermarkFileUpload:
    """Test watermark functionality with file upload."""

    def setup_method(self):
        """Setup before each test."""
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Check if test video exists (relative to project root)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.test_video_path = os.path.join(
            project_root, "assets", "test_videos", "trump_music_song.mp4"
        )
        if not os.path.exists(self.test_video_path):
            pytest.skip(f"Test video not found: {self.test_video_path}")

    def test_file_upload_watermark_disabled(self):
        """Test watermark disabled with file upload."""
        print("\n🎯 Testing file upload with watermark disabled")

        with open(self.test_video_path, "rb") as video_file:
            files = {"file": ("test_video.mp4", video_file, "video/mp4")}
            data = {
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": "true",
                "whisper_model": "tiny",
                "translation_service": "google",
                "watermark_enabled": "false",  # Explicitly disable
            }

            response = requests.post(
                "http://localhost:8081/upload", files=files, data=data, timeout=30
            )

        assert response.status_code in [200, 202], f"Upload failed: {response.text}"
        data = response.json()
        assert "task_id" in data, "No task_id in response"

        task_id = data["task_id"]
        print(f"📋 Task ID: {task_id}")

        # Wait for completion (file upload is usually faster)
        start_time = time.time()
        max_wait = 200

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"http://localhost:8081/status/{task_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            state = status_data.get("state", "PENDING")

            if state == "SUCCESS":
                # Extract logs for analysis
                progress = status_data.get("progress", {})
                steps = progress.get("steps", [])

                logs = []
                for step in steps:
                    step_logs = step.get("logs", [])
                    for log_entry in step_logs:
                        if isinstance(log_entry, dict):
                            logs.append(log_entry.get("message", ""))
                        else:
                            logs.append(str(log_entry))

                logs_text = "\n".join(logs)
                print(f"\n📝 Task logs:\n{logs_text}")

                # Verify watermark was skipped
                assert (
                    "Skipping watermark (disabled by user)" in logs_text
                ), "Expected 'Skipping watermark (disabled by user)' message not found in logs"

                print(
                    "✅ Test passed: File upload watermark correctly skipped when disabled"
                )
                return

            elif state == "FAILURE":
                error = status_data.get("error", {})
                pytest.fail(f"Task failed: {error}")

            time.sleep(2)

        pytest.fail(f"Task timed out after {max_wait} seconds")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
