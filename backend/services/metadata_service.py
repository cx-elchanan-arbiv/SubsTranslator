"""
Enterprise-grade metadata extraction service.
Handles YouTube video metadata with proper error handling, validation, and logging.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Structured video metadata with type safety."""

    title: str
    duration: int
    duration_string: str
    view_count: int
    upload_date: str
    uploader: str
    thumbnail: str
    description: str
    width: int
    height: int
    fps: int
    filesize: int
    url: str
    quality_available: list = None


class MetadataExtractionError(Exception):
    """Custom exception for metadata extraction failures."""

    def __init__(self, message: str, error_code: str, recoverable: bool = True):
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        super().__init__(message)


class VideoMetadataService:
    """Enterprise service for video metadata extraction from multiple platforms."""

    # Supported domains for security (yt-dlp supports 1849+ sites)
    ALLOWED_DOMAINS = {
        # YouTube
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "m.youtube.com",
        "music.youtube.com",
        # Other popular video sites
        "vimeo.com",
        "dailymotion.com",
        "facebook.com",
        "fb.watch",
        "instagram.com",
        "tiktok.com",
        "twitch.tv",
        "reddit.com",
        "soundcloud.com",
        "twitter.com",
        "x.com",
        "foxnews.com",
    }

    def __init__(self):
        self._extraction_cache = {}
        self._extraction_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        self.config = get_config()

    def _validate_url(self, url: str) -> bool:
        """Validate video URL for security."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check if domain is in our known safe list
            if domain in self.ALLOWED_DOMAINS:
                # Additional validation for known domains
                if "youtube.com" in domain:
                    if (
                        "/watch" not in parsed.path
                        and "v=" not in parsed.query
                        and "/shorts/" not in parsed.path
                    ):
                        return False
                elif "youtu.be" in domain:
                    if len(parsed.path) < 2:  # Must have video ID
                        return False
                return True
            else:
                # Check configuration for unknown domains
                if not self.config.ALLOW_UNKNOWN_DOMAINS:
                    logger.warning(f"Blocked unknown domain: {domain}")
                    return False
                # For unknown domains, let yt-dlp handle validation
                # This allows support for the 1849+ sites that yt-dlp supports
                logger.info(f"Allowing unknown domain {domain} due to config")
                return True

        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return False

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        # Extract video ID for consistent caching
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "v=" in url:
            return parse_qs(urlparse(url).query).get("v", [""])[0]
        return url

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._extraction_timestamps:
            return False
        return (time.time() - self._extraction_timestamps[cache_key]) < self.cache_ttl

    def extract_metadata(self, url: str) -> tuple[VideoMetadata, str | None]:
        """
        Extract video metadata with comprehensive error handling.

        Returns:
            Tuple[VideoMetadata, Optional[error_code]]
        """
        start_time = time.time()

        try:
            # Security validation
            if not self._validate_url(url):
                raise MetadataExtractionError(
                    "Invalid or blocked URL", "INVALID_URL", recoverable=False
                )

            # Check cache
            cache_key = self._get_cache_key(url)
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached metadata for {cache_key}")
                return self._extraction_cache[cache_key], None

            # Extract metadata
            logger.info(f"Extracting metadata for: {url}")

            ydl_opts = {
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 60,
                "retries": 10,
                "fragment_retries": 10,
                "extract_flat": False,
                "restrict_filenames": True,
                "extractor_args": self.config.YTDLP_EXTRACTOR_ARGS,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)

                # Build structured metadata
                metadata = VideoMetadata(
                    title=info_dict.get("title", "Unknown Title")[:100],  # Limit length
                    duration=info_dict.get("duration", 0),
                    duration_string=info_dict.get("duration_string", "00:00"),
                    view_count=info_dict.get("view_count", 0) or 0,
                    upload_date=info_dict.get("upload_date", ""),
                    uploader=info_dict.get("uploader", "Unknown")[:50],  # Limit length
                    thumbnail=info_dict.get("thumbnail", ""),
                    description=self._safe_description(
                        info_dict.get("description", "")
                    ),
                    width=info_dict.get("width", 0) or 0,
                    height=info_dict.get("height", 0) or 0,
                    fps=info_dict.get("fps", 0) or 0,
                    filesize=info_dict.get("filesize", 0) or 0,
                    url=url,
                    quality_available=self._extract_available_qualities(info_dict),
                )

                # Cache result
                self._extraction_cache[cache_key] = metadata
                self._extraction_timestamps[cache_key] = time.time()

                extraction_time = time.time() - start_time
                logger.info(
                    f"Metadata extracted successfully in {extraction_time:.2f}s"
                )

                return metadata, None

        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            logger.error(f"yt-dlp DownloadError: {error_msg}")  # Log the actual error

            # Check for bot detection first (most specific)
            if (
                "sign in to confirm" in error_msg.lower()
                or "confirm you're not a bot" in error_msg.lower()
            ):
                raise MetadataExtractionError(
                    "YouTube is blocking downloads from our server. Please download the video to your computer and upload it as a file.",
                    "YOUTUBE_BOT_DETECTION",
                    recoverable=False,
                )
            elif "private" in error_msg.lower():
                raise MetadataExtractionError(
                    "The video is private or unavailable",
                    "PRIVATE_VIDEO",
                    recoverable=False,
                )
            elif "not available" in error_msg.lower():
                raise MetadataExtractionError(
                    "The video does not exist or has been removed",
                    "VIDEO_NOT_AVAILABLE",
                    recoverable=False,
                )
            else:
                # Log full error for debugging
                logger.error(f"Generic yt-dlp download error: {error_msg}")
                raise MetadataExtractionError(
                    f"Error accessing video: {error_msg[:100]}",
                    "DOWNLOAD_ERROR",
                    recoverable=True,
                )

        except Exception as e:
            logger.error(f"Unexpected metadata extraction error: {e}")
            raise MetadataExtractionError(
                "Error extracting video information",
                "EXTRACTION_ERROR",
                recoverable=True,
            )

    def _safe_description(self, description: str) -> str:
        """Safely truncate and clean description."""
        if not description:
            return ""
        # Remove potential harmful content, limit length
        cleaned = re.sub(r"[^\w\s\-.,!?():]", "", description[:200])
        return cleaned + "..." if len(description) > 200 else cleaned

    def _extract_available_qualities(self, info_dict: dict[str, Any]) -> list:
        """Extract available video qualities."""
        formats = info_dict.get("formats", [])
        qualities = set()

        for fmt in formats:
            if fmt.get("vcodec") != "none":  # Video track exists
                height = fmt.get("height")
                if height:
                    if height >= 1080:
                        qualities.add("1080p")
                    elif height >= 720:
                        qualities.add("720p")
                    elif height >= 480:
                        qualities.add("480p")
                    else:
                        qualities.add("360p")

        return sorted(list(qualities), reverse=True)


# Global service instance
metadata_service = VideoMetadataService()
