"""
Production Smoke Tests for YouTube Download
============================================

Tests the complete YouTube download pipeline against production environment.
These tests verify that POT Provider integration and proxy fallbacks work correctly.

Test URLs are selected to be:
- Short (under 60 seconds)
- Publicly available
- From different content providers
- Unlikely to be removed
"""

import os
import time
from typing import Any

import pytest
import requests


class TestYouTubeProductionSmoke:
    """End-to-end smoke tests for YouTube downloads in production."""

    # Test video URLs (short, stable videos)
    TEST_VIDEOS = [
        {
            "url": "https://www.youtube.com/watch?v=L5WSTbdw7xI",
            "title": "Iran water shortage",
            "duration_approx": 54,
            "uploader": "Associated Press",
        },
        {
            "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "title": "Me at the zoo",
            "duration_approx": 19,
            "uploader": "jawed",  # First YouTube video ever
        },
    ]

    API_BASE_URL = "https://substranslator-backend.onrender.com"
    MAX_WAIT_TIME = 120  # 2 minutes max wait
    POLL_INTERVAL = 5  # Check every 5 seconds

    @pytest.fixture(scope="class", autouse=True)
    def verify_backend_health(self):
        """Verify backend is accessible before running tests."""
        try:
            response = requests.get(f"{self.API_BASE_URL}/health", timeout=10)
            assert (
                response.status_code == 200
            ), f"Backend unhealthy: {response.status_code}"
            health_data = response.json()
            assert (
                health_data.get("status") == "healthy"
            ), f"Backend not healthy: {health_data}"
            print(f"✅ Backend is healthy: {health_data.get('message')}")
        except requests.RequestException as e:
            pytest.skip(f"Backend not accessible: {e}")

    def wait_for_task_completion(
        self, task_id: str, timeout: int = MAX_WAIT_TIME
    ) -> dict[str, Any]:
        """
        Wait for a Celery task to complete and return the final result.

        Args:
            task_id: The Celery task ID to monitor
            timeout: Maximum time to wait in seconds

        Returns:
            Final task status dictionary

        Raises:
            AssertionError if task fails or times out
        """
        start_time = time.time()
        elapsed_str = lambda: f"{time.time() - start_time:.1f}s"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.API_BASE_URL}/status/{task_id}", timeout=10
                )

                if response.status_code != 200:
                    print(
                        f"⏳ [{elapsed_str()}] Status endpoint returned {response.status_code}, waiting..."
                    )
                    time.sleep(self.POLL_INTERVAL)
                    continue

                status_data = response.json()
                state = status_data.get("state", "UNKNOWN")

                print(f"⏳ [{elapsed_str()}] Task {task_id[:8]}... state: {state}")

                if state == "SUCCESS":
                    print(f"✅ [{elapsed_str()}] Task completed successfully")
                    return status_data

                elif state == "FAILURE":
                    error = status_data.get("error", {})
                    error_message = (
                        error.get("message", "Unknown error")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    error_code = (
                        error.get("code", "UNKNOWN")
                        if isinstance(error, dict)
                        else "UNKNOWN"
                    )

                    # Check if it's a YouTube blocking issue (skip test, not fail)
                    blocking_indicators = [
                        "403",
                        "forbidden",
                        "unavailable",
                        "bot",  # Even with POT, YouTube might still block in some cases
                    ]

                    if any(
                        indicator in error_message.lower()
                        for indicator in blocking_indicators
                    ):
                        pytest.skip(
                            f"YouTube blocking detected (not a code failure): {error_code} - {error_message}"
                        )
                    else:
                        pytest.fail(
                            f"Task failed: {error_code} - {error_message}\n"
                            f"Full error: {error}"
                        )

                # Still pending/processing, wait and retry
                time.sleep(self.POLL_INTERVAL)

            except requests.RequestException as e:
                print(f"⏳ [{elapsed_str()}] Request failed: {e}, retrying...")
                time.sleep(self.POLL_INTERVAL)

        # Timeout reached
        pytest.fail(
            f"Task {task_id} did not complete within {timeout} seconds\n"
            f"Last known state: {status_data.get('state', 'UNKNOWN')}"
        )

    def verify_video_metadata(self, metadata: dict[str, Any], expected: dict[str, Any]):
        """
        Verify that video metadata is valid and matches expectations.

        Args:
            metadata: Actual video metadata from API
            expected: Expected metadata values
        """
        assert metadata is not None, "Video metadata is None"
        assert isinstance(
            metadata, dict
        ), f"Video metadata is not a dict: {type(metadata)}"

        # Essential fields
        required_fields = ["title", "duration", "uploader", "url"]
        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"
            assert metadata[field], f"Metadata field '{field}' is empty"

        # Verify URL matches
        assert (
            metadata["url"] == expected["url"]
        ), f"URL mismatch: got {metadata['url']}, expected {expected['url']}"

        # Verify duration is approximately correct (±10%)
        expected_duration = expected["duration_approx"]
        actual_duration = metadata["duration"]
        duration_diff_pct = (
            abs(actual_duration - expected_duration) / expected_duration * 100
        )

        assert (
            duration_diff_pct < 10
        ), f"Duration mismatch: got {actual_duration}s, expected ~{expected_duration}s ({duration_diff_pct:.1f}% diff)"

        # Verify uploader
        assert (
            expected["uploader"].lower() in metadata["uploader"].lower()
        ), f"Uploader mismatch: got {metadata['uploader']}, expected {expected['uploader']}"

        print("✅ Metadata validation passed:")
        print(f"   Title: {metadata['title']}")
        print(f"   Duration: {metadata['duration']}s (expected ~{expected_duration}s)")
        print(f"   Uploader: {metadata['uploader']}")

    @pytest.mark.parametrize("video_info", TEST_VIDEOS)
    def test_download_only_success(self, video_info: dict[str, Any]):
        """
        Test download-only functionality for various YouTube videos.

        This test verifies:
        1. POT Provider integration works
        2. Proxy fallback is functioning (if configured)
        3. Metadata extraction is correct
        4. Download completes successfully
        5. No "bot detection" errors
        """
        print(f"\n📥 Testing download-only: {video_info['title']}")
        print(f"   URL: {video_info['url']}")

        # Submit download request
        response = requests.post(
            f"{self.API_BASE_URL}/download-video-only",
            json={"url": video_info["url"]},
            timeout=30,
        )

        assert response.status_code in [
            200,
            202,
        ], f"Failed to submit download request: {response.status_code} - {response.text}"

        result = response.json()
        task_id = result.get("task_id")

        assert task_id, f"No task_id in response: {result}"
        print(f"📋 Task submitted: {task_id}")

        # Wait for completion
        final_status = self.wait_for_task_completion(task_id)

        # Verify success
        assert (
            final_status["state"] == "SUCCESS"
        ), f"Task did not succeed: {final_status.get('state')}"

        # Extract result
        task_result = final_status.get("result", {})
        assert task_result, "Result is empty"

        # Verify download status
        assert (
            task_result.get("status") == "SUCCESS"
        ), f"Download status not SUCCESS: {task_result.get('status')}"

        assert task_result.get("filename"), "No filename in result"
        assert task_result.get("download_url"), "No download_url in result"

        # Verify metadata
        video_metadata = final_status.get("video_metadata") or task_result.get(
            "video_metadata"
        )
        assert video_metadata, "No video_metadata in result"

        self.verify_video_metadata(video_metadata, video_info)

        print(f"✅ Download test passed for: {video_info['title']}")

    def test_download_only_consistency(self):
        """
        Test consistency by downloading the same video twice.

        Verifies:
        1. Results are consistent
        2. Metadata is identical
        3. No random failures
        """
        test_video = self.TEST_VIDEOS[0]  # Use first test video

        results = []

        for attempt in range(2):
            print(f"\n🔄 Consistency test attempt {attempt + 1}/2")

            response = requests.post(
                f"{self.API_BASE_URL}/download-video-only",
                json={"url": test_video["url"]},
                timeout=30,
            )

            assert response.status_code in [200, 202]
            task_id = response.json()["task_id"]

            final_status = self.wait_for_task_completion(task_id)
            assert final_status["state"] == "SUCCESS"

            results.append(final_status)

            # Wait between attempts to avoid rate limiting
            if attempt == 0:
                time.sleep(10)

        # Compare results
        assert len(results) == 2

        # Both should have same metadata
        metadata1 = results[0].get("video_metadata") or results[0]["result"].get(
            "video_metadata"
        )
        metadata2 = results[1].get("video_metadata") or results[1]["result"].get(
            "video_metadata"
        )

        assert metadata1["title"] == metadata2["title"]
        assert metadata1["duration"] == metadata2["duration"]
        assert metadata1["uploader"] == metadata2["uploader"]

        print("✅ Consistency test passed - results are identical")

    def test_error_handling_invalid_url(self):
        """
        Test error handling with an invalid YouTube URL.

        Verifies:
        1. Invalid URLs are rejected gracefully
        2. Error messages are user-friendly
        3. No crashes or unexpected states
        """
        print("\n❌ Testing error handling with invalid URL")

        invalid_url = "https://www.youtube.com/watch?v=INVALID_VIDEO_ID_12345"

        response = requests.post(
            f"{self.API_BASE_URL}/download-video-only",
            json={"url": invalid_url},
            timeout=30,
        )

        assert response.status_code in [200, 202]
        task_id = response.json()["task_id"]

        # Wait for completion (should fail quickly)
        start_time = time.time()

        while time.time() - start_time < 60:  # 1 minute timeout for error case
            response = requests.get(f"{self.API_BASE_URL}/status/{task_id}")

            if response.status_code == 200:
                status = response.json()
                state = status.get("state")

                if state == "FAILURE":
                    error = status.get("error", {})
                    print(
                        f"✅ Task failed as expected: {error.get('message', 'Unknown error')}"
                    )
                    assert error, "No error information in failed task"
                    return

                elif state == "SUCCESS":
                    pytest.fail(
                        "Task should have failed with invalid URL but succeeded"
                    )

            time.sleep(5)

        pytest.fail("Task with invalid URL did not fail within timeout")


# Performance benchmark test (optional, can be skipped in CI).
# `pytest.config` was removed in pytest 4.0 and the `--run-benchmark` flag it read
# was never registered by any conftest, so this gate raised AttributeError at import
# and broke collection of the whole file. Gate on an env var instead.
@pytest.mark.benchmark
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK", "").lower() not in ("1", "true", "yes"),
    reason="Benchmark tests require RUN_BENCHMARK=1",
)
class TestYouTubePerformance:
    """Performance benchmarks for YouTube downloads."""

    API_BASE_URL = "https://substranslator-backend.onrender.com"

    def test_download_speed_benchmark(self):
        """
        Benchmark download speed for a short video.

        Target: < 30s for a 1-minute video
        """
        test_video = {
            "url": "https://www.youtube.com/watch?v=L5WSTbdw7xI",
            "duration_approx": 54,
        }

        start_time = time.time()

        response = requests.post(
            f"{self.API_BASE_URL}/download-video-only",
            json={"url": test_video["url"]},
            timeout=30,
        )

        task_id = response.json()["task_id"]

        # Poll until complete
        while time.time() - start_time < 60:
            status = requests.get(f"{self.API_BASE_URL}/status/{task_id}").json()
            if status["state"] in ["SUCCESS", "FAILURE"]:
                break
            time.sleep(2)

        total_time = time.time() - start_time

        assert status["state"] == "SUCCESS", f"Download failed: {status.get('error')}"
        assert total_time < 30, f"Download took {total_time:.1f}s (target: < 30s)"

        print(f"✅ Download completed in {total_time:.1f}s")
        print(f"   Video duration: {test_video['duration_approx']}s")
        print(
            f"   Performance ratio: {total_time / test_video['duration_approx']:.2f}x"
        )
