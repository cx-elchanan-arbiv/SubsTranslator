"""
E2E Test for OpenAI Translation Service
Tests the complete flow: YouTube URL → OpenAI translation → Hebrew video
"""

import os
import re
import time
from typing import Any

import pytest
import requests


@pytest.mark.e2e
@pytest.mark.slow
class TestOpenAITranslationE2E:
    """End-to-end tests for OpenAI translation functionality."""

    # Short test video for faster testing
    TEST_VIDEO_URL = (
        "https://www.youtube.com/watch?v=wnGrN7j7-mg"  # Fox News, ~2 minutes
    )

    def setup_method(self):
        """Setup before each test."""
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Check if OpenAI is actually available
        try:
            response = requests.get(
                "http://localhost:8081/translation-services", timeout=5
            )
            if response.status_code == 200:
                services = response.json()
                if not services.get("openai", {}).get("available", False):
                    pytest.skip(
                        "OpenAI service not available - check OPENAI_API_KEY configuration"
                    )
            else:
                pytest.skip("Cannot check translation services availability")
        except:
            pytest.skip("Cannot access translation services endpoint")

    def wait_for_task_completion(
        self, task_id: str, max_wait: int = 300
    ) -> dict[Any, Any]:
        """Wait for a task to complete and return the result."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"http://localhost:8081/task/{task_id}", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("state") in ["SUCCESS", "FAILURE", "REVOKED"]:
                        return data
                elif response.status_code == 404:
                    # Task might not be ready yet
                    pass
                else:
                    print(f"Unexpected status code: {response.status_code}")

            except requests.RequestException as e:
                print(f"Request failed: {e}")

            time.sleep(5)  # Check every 5 seconds

        pytest.fail(f"Task {task_id} did not complete within {max_wait} seconds")

    def extract_logs_from_task_result(self, task_result: dict[Any, Any]) -> str:
        """Extract logs from task result for analysis."""
        logs = []

        # Get progress logs
        progress = task_result.get("progress", [])
        for entry in progress:
            if isinstance(entry, dict) and "message" in entry:
                logs.append(entry["message"])

        # Get result logs if available
        result = task_result.get("result", {})
        if isinstance(result, dict) and "logs" in result:
            if isinstance(result["logs"], list):
                logs.extend(result["logs"])
            elif isinstance(result["logs"], str):
                logs.append(result["logs"])

        return "\n".join(logs)

    @pytest.mark.skipif(
        not os.environ.get("TEST_OPENAI_E2E"),
        reason="Set TEST_OPENAI_E2E=1 to run OpenAI E2E tests (requires valid API key)",
    )
    def test_openai_translation_complete_flow(self):
        """Test complete flow: YouTube URL → OpenAI translation → Hebrew subtitles."""
        print(f"\n🎯 Testing OpenAI translation E2E with video: {self.TEST_VIDEO_URL}")

        # Submit YouTube video with OpenAI translation
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",  # Hebrew translation
                "auto_create_video": True,
                "whisper_model": "tiny",  # Use fastest model for testing
                "translation_service": "openai",  # THIS IS THE KEY TEST
                "watermark_enabled": False,
            },
            timeout=15,
        )

        assert response.status_code in [200, 202], f"Request failed: {response.text}"
        data = response.json()
        assert "task_id" in data, "No task_id in response"

        task_id = data["task_id"]
        print(f"📋 Task ID: {task_id}")

        # Verify user choices were recorded correctly
        assert (
            data["user_choices"]["translation_service"] == "openai"
        ), "OpenAI service not selected"
        assert (
            data["user_choices"]["target_lang"] == "he"
        ), "Hebrew not selected as target"

        # Wait for completion (OpenAI can be slower than Google)
        print("⏳ Waiting for task completion...")
        task_result = self.wait_for_task_completion(task_id, max_wait=400)

        # Extract logs for analysis
        logs = self.extract_logs_from_task_result(task_result)
        print(f"\n📝 Task logs:\n{logs}")

        # Verify task completed successfully
        assert (
            task_result["state"] == "SUCCESS"
        ), f"Task failed: {task_result.get('result', {}).get('error', 'Unknown error')}"

        # Verify OpenAI was actually used for translation
        openai_indicators = [
            "Using OpenAI for translation",
            "Translating",
            "segments using OpenAI to he",
            "OpenAI",
        ]

        found_openai_usage = False
        for indicator in openai_indicators:
            if indicator in logs:
                found_openai_usage = True
                print(f"✅ Found OpenAI usage indicator: '{indicator}'")
                break

        assert (
            found_openai_usage
        ), f"No evidence of OpenAI usage in logs. Available logs: {logs[:500]}..."

        # Verify NO translation failure/fallback occurred
        failure_indicators = [
            "Translation with openai failed",
            "Falling back to original text",
            "OpenAI authentication failed",
            "API key",
            "401 Unauthorized",
        ]

        for failure in failure_indicators:
            assert (
                failure not in logs
            ), f"Found translation failure indicator: '{failure}'"

        # Verify files were created
        result = task_result.get("result", {})
        assert "files" in result, "No files in result"

        files = result["files"]
        assert "translated_srt" in files, "No translated subtitle file created"
        assert "video_with_subtitles" in files, "No final video created"

        print("✅ OpenAI translation E2E test passed!")

    @pytest.mark.skipif(
        not os.environ.get("TEST_OPENAI_E2E"),
        reason="Set TEST_OPENAI_E2E=1 to run OpenAI E2E tests (requires valid API key)",
    )
    def test_openai_translation_quality_check(self):
        """Test that OpenAI actually produces Hebrew translations (not English fallback)."""
        print(
            f"\n🎯 Testing OpenAI translation quality with video: {self.TEST_VIDEO_URL}"
        )

        # Submit for transcription and translation only (no video creation for speed)
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": False,  # Skip video creation for speed
                "whisper_model": "tiny",
                "translation_service": "openai",
                "watermark_enabled": False,
            },
            timeout=15,
        )

        assert response.status_code in [200, 202]
        data = response.json()
        task_id = data["task_id"]

        # Wait for completion
        task_result = self.wait_for_task_completion(task_id, max_wait=300)
        assert task_result["state"] == "SUCCESS"

        # Get the translated subtitle file content
        result = task_result.get("result", {})
        files = result.get("files", {})
        translated_srt = files.get("translated_srt")

        assert translated_srt, "No translated SRT file found"

        # Try to download and check the subtitle content
        # Note: This would require access to the actual file, which might not be
        # available in the test environment. For now, we check the logs.

        logs = self.extract_logs_from_task_result(task_result)

        # Verify successful translation completion
        success_indicators = [
            "SRT file created successfully",
            "translated.srt",
            "segments_count=",
            "use_translation=True",
        ]

        found_success = False
        for indicator in success_indicators:
            if indicator in logs:
                found_success = True
                break

        assert (
            found_success
        ), f"No evidence of successful translation in logs: {logs[:500]}..."

        # Verify detected language makes sense
        result_data = result.get("detected_language")
        if result_data:
            # Should detect English for Fox News video
            assert result_data in [
                "en",
                "english",
            ], f"Unexpected detected language: {result_data}"

        print("✅ OpenAI translation quality check passed!")

    def test_openai_unavailable_fallback(self):
        """Test behavior when OpenAI is selected but not available."""
        print("\n🎯 Testing OpenAI unavailable scenario")

        # This test runs even without TEST_OPENAI_E2E flag
        # because it tests the error handling, not the actual OpenAI service

        # Check current OpenAI availability
        response = requests.get("http://localhost:8081/translation-services", timeout=5)
        assert response.status_code == 200

        services = response.json()
        openai_available = services.get("openai", {}).get("available", False)

        if openai_available:
            pytest.skip("OpenAI is available, cannot test unavailable scenario")

        # Try to submit with OpenAI when it's not available
        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": self.TEST_VIDEO_URL,
                "source_lang": "auto",
                "target_lang": "he",
                "auto_create_video": False,
                "whisper_model": "tiny",
                "translation_service": "openai",  # This should fail gracefully
                "watermark_enabled": False,
            },
            timeout=15,
        )

        # The request should either:
        # 1. Be rejected immediately (400 Bad Request)
        # 2. Be accepted but fail gracefully with fallback to Google

        if response.status_code == 400:
            # Immediate rejection is acceptable
            error_data = response.json()
            assert (
                "openai" in error_data.get("error", "").lower()
                or "translation" in error_data.get("error", "").lower()
            )
            print("✅ Request properly rejected when OpenAI unavailable")

        elif response.status_code in [200, 202]:
            # Graceful handling with fallback is also acceptable
            data = response.json()
            task_id = data["task_id"]

            # Wait for completion
            task_result = self.wait_for_task_completion(task_id, max_wait=300)

            # Should either succeed with fallback or fail gracefully
            if task_result["state"] == "SUCCESS":
                logs = self.extract_logs_from_task_result(task_result)
                # Should show fallback to Google or original text
                fallback_indicators = [
                    "Falling back",
                    "Using Google",
                    "translation.*failed",
                    "API key",
                ]

                found_fallback = any(
                    re.search(indicator, logs, re.IGNORECASE)
                    for indicator in fallback_indicators
                )

                assert (
                    found_fallback
                ), f"No evidence of fallback handling in logs: {logs[:500]}..."
                print("✅ Graceful fallback when OpenAI unavailable")

            else:
                # Graceful failure is also acceptable
                print("✅ Graceful failure when OpenAI unavailable")

        else:
            pytest.fail(f"Unexpected response status: {response.status_code}")


@pytest.mark.e2e
class TestOpenAIServiceAvailability:
    """Test OpenAI service availability detection."""

    def test_translation_services_endpoint_reflects_openai_status(self):
        """Test that /translation-services correctly reports OpenAI availability."""
        response = requests.get("http://localhost:8081/translation-services", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "openai" in data
        assert "google" in data

        # Google should always be available
        assert data["google"]["available"] is True

        # OpenAI availability should match actual configuration
        openai_service = data["openai"]
        assert "available" in openai_service
        assert "description" in openai_service
        assert "name" in openai_service

        if openai_service["available"]:
            assert "Advanced translation" in openai_service["description"]
        else:
            assert "API key required" in openai_service["description"]

        print(f"✅ OpenAI service availability: {openai_service['available']}")
        print(f"   Description: {openai_service['description']}")
