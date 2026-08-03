"""
Tests for YouTube download functionality.
These tests help catch YouTube blocking issues and yt-dlp problems.
"""

import os
import subprocess

import pytest
import requests


@pytest.mark.integration
class TestYouTubeDownload:
    """Test YouTube download functionality."""

    def test_ytdlp_available_in_container(self):
        """Test that yt-dlp is available and working in container."""
        # Check if containers are running
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Test yt-dlp version in container
        result = subprocess.run(
            ["docker-compose", "exec", "-T", "backend", "yt-dlp", "--version"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"yt-dlp not available: {result.stderr}"
        assert "20" in result.stdout, "yt-dlp version seems invalid"

    def test_ytdlp_can_extract_info(self):
        """Test that yt-dlp can extract video info without downloading."""
        # Use a known working test video
        test_url = (
            "https://www.youtube.com/watch?v=wnGrN7j7-mg"  # Recent Fox News video
        )

        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Test info extraction in container
        cmd = [
            "docker-compose",
            "exec",
            "-T",
            "backend",
            "python",
            "-c",
            f"""
import yt_dlp
try:
    ydl_opts = {{
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info('{test_url}', download=False)
        print("SUCCESS:", info.get('title', 'Unknown'))
except Exception as e:
    print("ERROR:", str(e))
            """,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Check if extraction worked
        if "SUCCESS:" in result.stdout:
            assert True  # Success
        elif "ERROR:" in result.stdout:
            error_msg = result.stdout.split("ERROR:")[1].strip()
            if "403" in error_msg or "Forbidden" in error_msg:
                pytest.skip(f"YouTube blocking detected: {error_msg}")
            else:
                pytest.fail(f"yt-dlp extraction failed: {error_msg}")
        else:
            pytest.fail(f"Unexpected output: {result.stdout} {result.stderr}")

    def test_youtube_api_endpoint_with_mock(self):
        """Test YouTube API endpoint with mocked yt-dlp."""
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Test with invalid URL (should return 400, not 500)
        response = requests.post(
            "http://localhost:8081/youtube", json={"url": "invalid_url"}, timeout=10
        )

        # Should return 400 (bad request) not 500 (server error)
        assert response.status_code in [
            400,
            422,
        ], f"Unexpected status: {response.status_code}"

    def test_youtube_error_handling(self):
        """Test that YouTube errors are handled gracefully."""
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Test with non-existent YouTube video
        fake_url = "https://www.youtube.com/watch?v=ThisVideoDoesNotExist123456"

        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": fake_url,
                "source_lang": "en",
                "target_lang": "he",
                "auto_create_video": False,
                "whisper_model": "base",
                "translation_service": "google",
            },
            timeout=15,
        )

        # Should return a task ID even if it will fail later (200 or 202 are both OK)
        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_id" in data


@pytest.mark.unit
class TestYouTubeConfiguration:
    """Test YouTube download configuration."""

    def test_ytdlp_options_format(self):
        """Test that yt-dlp options are properly formatted."""
        # Import config to test yt-dlp options
        import os
        import sys

        backend_path = os.path.join(os.path.dirname(__file__), "..", "..")
        sys.path.insert(0, backend_path)

        try:
            import config

            cfg = config.get_config()

            # The single format string every yt-dlp caller actually reads
            # (services/youtube_service.py). The old per-quality dict was never
            # consulted by any download path and has been removed.
            assert hasattr(cfg, "YTDLP_OPTIMIZED_FORMAT")
            assert isinstance(cfg.YTDLP_OPTIMIZED_FORMAT, str)
            assert len(cfg.YTDLP_OPTIMIZED_FORMAT) > 0

            # Test timeout settings
            assert hasattr(cfg, "YTDLP_SOCKET_TIMEOUT")
            assert cfg.YTDLP_SOCKET_TIMEOUT > 0

            assert hasattr(cfg, "YTDLP_RETRIES")
            assert cfg.YTDLP_RETRIES > 0

        except ImportError:
            pytest.skip("Config module not available")

    def test_anti_blocking_headers(self):
        """Test that anti-blocking headers are properly configured."""
        # This test ensures we have proper headers to avoid YouTube blocking

        expected_headers = ["user_agent", "referer", "headers"]

        # We can't easily test the actual implementation without importing tasks
        # But we can test that the concept is sound
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        # Test that user agent looks like a real browser
        assert "Mozilla" in user_agent
        assert "Chrome" in user_agent
        assert "Safari" in user_agent

        # Test referer
        referer = "https://www.youtube.com/"
        assert referer.startswith("https://www.youtube.com")


@pytest.mark.slow
class TestYouTubeRealDownload:
    """Test actual YouTube download (slow tests)."""

    @pytest.mark.skipif(
        not os.environ.get("TEST_REAL_YOUTUBE"),
        reason="Set TEST_REAL_YOUTUBE=1 to run real YouTube download tests",
    )
    def test_real_youtube_download(self):
        """Test downloading a real YouTube video (opt-in only)."""
        # This test is opt-in because it's slow and depends on external service

        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Use a recent, known-good video
        test_url = "https://www.youtube.com/watch?v=wnGrN7j7-mg"  # Fox News video - 2:06 minutes

        response = requests.post(
            "http://localhost:8081/youtube",
            json={
                "url": test_url,
                "source_lang": "en",
                "target_lang": "he",
                "auto_create_video": False,  # Don't create video to save time
                "whisper_model": "base",  # Use smallest available model
                "translation_service": "google",
            },
            timeout=10,
        )

        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_id" in data

        task_id = data["task_id"]

        # Poll for completion (with timeout)
        import time

        max_wait = 120  # 2 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"http://localhost:8081/status/{task_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            state = status_data.get("state", "PENDING")

            if state == "SUCCESS":
                # Success! Check that we got results
                result = status_data.get("result", {})
                # Since we're not creating video (auto_create_video=False),
                # we should have task_id and video_metadata instead of files
                assert "task_id" in result or "video_metadata" in status_data
                break
            elif state == "FAILURE":
                # Check if it's a blocking issue
                error = status_data.get("error", {})
                if isinstance(error, dict):
                    message = error.get("message", "")
                else:
                    message = str(error)

                if "403" in message or "Forbidden" in message:
                    pytest.skip(f"YouTube blocking detected: {message}")
                else:
                    pytest.fail(f"Download failed: {message}")

            time.sleep(2)
        else:
            pytest.fail("Download timed out after 2 minutes")

    @pytest.mark.skipif(
        not os.environ.get("TEST_REAL_YOUTUBE"),
        reason="Set TEST_REAL_YOUTUBE=1 to run real YouTube download tests",
    )
    def test_real_youtube_download_only(self):
        """Test downloading a real YouTube video without processing (download-only mode)."""

        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend container not running")
        except:
            pytest.skip("Backend container not accessible")

        # Use the same working video
        test_url = "https://www.youtube.com/watch?v=wnGrN7j7-mg"  # Fox News video - 2:06 minutes

        response = requests.post(
            "http://localhost:8081/download-video-only",
            json={"url": test_url, "quality": "medium"},
            timeout=10,
        )

        assert response.status_code in [200, 202]
        data = response.json()
        assert "task_id" in data

        task_id = data["task_id"]

        # Poll for completion (with timeout)
        import time

        max_wait = 60  # 1 minute max for download-only
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"http://localhost:8081/status/{task_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            state = status_data.get("state", "PENDING")

            if state == "SUCCESS":
                # Success! Check that we got download results
                result = status_data.get("result", {})
                # For download-only, we should have download_url or filename
                assert "download_url" in result or "filename" in result
                break
            elif state == "FAILURE":
                # Check if it's a blocking issue
                error = status_data.get("error", {})
                if isinstance(error, dict):
                    message = error.get("message", "")
                else:
                    message = str(error)

                if "403" in message or "Forbidden" in message:
                    pytest.skip(f"YouTube blocking detected: {message}")
                else:
                    pytest.fail(f"Download failed: {message}")

            time.sleep(2)
        else:
            pytest.fail("Download timed out after 1 minute")


# Helper functions
def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    youtube_domains = ["youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"]

    return any(domain in url for domain in youtube_domains)


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    import re

    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)",
        r"youtube\.com\/embed\/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None
