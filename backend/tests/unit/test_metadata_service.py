import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from services.metadata_service import (
    MetadataExtractionError,
    VideoMetadata,
    VideoMetadataService,
)


@pytest.mark.unit
class TestMetadataService:
    """Test metadata service with different scenarios."""

    def test_successful_metadata_extraction(self):
        """Test successful metadata extraction (OK case)."""
        service = VideoMetadataService()

        # Mock successful yt-dlp response
        mock_info = {
            "title": "Test Video",
            "duration": 120,
            "view_count": 1000,
            "upload_date": "20240101",
            "uploader": "Test Channel",
            "thumbnail": "https://example.com/thumb.jpg",
            "description": "Test description",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "webpage_url": "https://youtube.com/watch?v=test123",
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.return_value = mock_info

            metadata, error = service.extract_metadata(
                "https://youtube.com/watch?v=test123"
            )

            assert error is None
            assert isinstance(metadata, VideoMetadata)
            assert metadata.title == "Test Video"
            assert metadata.duration == 120
            assert metadata.view_count == 1000
            assert metadata.uploader == "Test Channel"

    def test_invalid_url_error(self):
        """Test invalid URL handling (unsupported site)."""
        service = VideoMetadataService()

        # Test with unsupported domain (yt-dlp will try but fail)
        with pytest.raises(MetadataExtractionError) as exc_info:
            service.extract_metadata("https://invalid-url.com")

        error = exc_info.value
        # Since we now allow yt-dlp to try any domain, it will return DOWNLOAD_ERROR
        assert error.error_code in ["EXTRACTION_ERROR", "DOWNLOAD_ERROR"]
        assert error.recoverable is True

    def test_private_video_error(self):
        """Test private video handling (PRIVATE case)."""
        service = VideoMetadataService()

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.side_effect = yt_dlp.DownloadError(
                "This video is private"
            )

            with pytest.raises(MetadataExtractionError) as exc_info:
                service.extract_metadata("https://youtube.com/watch?v=private123")

            error = exc_info.value
            assert error.error_code == "PRIVATE_VIDEO"
            assert (
                "private" in error.message.lower()
                or "unavailable" in error.message.lower()
            )
            assert error.recoverable is False

    def test_video_not_available_error(self):
        """Test video not available handling."""
        service = VideoMetadataService()

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.side_effect = yt_dlp.DownloadError(
                "Video not available"
            )

            with pytest.raises(MetadataExtractionError) as exc_info:
                service.extract_metadata("https://youtube.com/watch?v=deleted123")

            error = exc_info.value
            assert error.error_code == "VIDEO_NOT_AVAILABLE"
            assert (
                "not exist" in error.message.lower()
                or "removed" in error.message.lower()
            )
            assert error.recoverable is False

    def test_generic_extraction_error(self):
        """Test generic extraction error handling."""
        service = VideoMetadataService()

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.side_effect = Exception("Network error")

            with pytest.raises(MetadataExtractionError) as exc_info:
                service.extract_metadata("https://youtube.com/watch?v=test123")

            error = exc_info.value
            assert error.error_code == "EXTRACTION_ERROR"
            assert (
                "error" in error.message.lower()
                or "extracting" in error.message.lower()
            )
            assert error.recoverable is True

    def test_single_video_from_playlist_url(self):
        """Test that playlist URLs extract single video (noplaylist=True)."""
        service = VideoMetadataService()

        # Mock single video response (noplaylist=True means no playlist handling)
        mock_video_info = {
            "title": "Single Video",
            "duration": 120,
            "view_count": 1500,
            "upload_date": "20240101",
            "uploader": "Test Channel",
            "webpage_url": "https://youtube.com/watch?v=single123",
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.return_value = mock_video_info

            # Use a playlist URL but expect single video due to noplaylist=True
            metadata, error = service.extract_metadata(
                "https://youtube.com/watch?v=single123&list=test123"
            )

            assert error is None
            assert isinstance(metadata, VideoMetadata)
            # Should get the single video, not playlist
            assert metadata.title == "Single Video"
            assert metadata.duration == 120
            assert metadata.view_count == 1500

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        service = VideoMetadataService()
        service.cache_ttl = 3600  # Set 1 hour cache

        mock_info = {
            "title": "Cached Video",
            "duration": 180,
            "webpage_url": "https://youtube.com/watch?v=cached123",
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
            mock_ydl = MagicMock()
            mock_ydl_class.return_value.__enter__.return_value = mock_ydl
            mock_ydl.extract_info.return_value = mock_info

            # First call should hit yt-dlp
            metadata1, error1 = service.extract_metadata(
                "https://youtube.com/watch?v=cached123"
            )
            assert mock_ydl.extract_info.call_count == 1

            # Second call should use cache
            metadata2, error2 = service.extract_metadata(
                "https://youtube.com/watch?v=cached123"
            )
            assert mock_ydl.extract_info.call_count == 1  # Still 1, not called again

            # Results should be identical
            assert metadata1.title == metadata2.title
            assert metadata1.duration == metadata2.duration
