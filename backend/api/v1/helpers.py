"""
Shared helper functions for API v1 routes.
"""

import base64
import os
import re
from urllib.parse import urlparse

from flask import jsonify, session

from config import get_config
from i18n.translations import t
from logging_config import get_logger
from logo_manager import LogoManager
from utils.file_utils import safe_int

# Configuration
config = get_config()
logger = get_logger(__name__)

# Initialize logo manager (shared instance)
logo_manager = LogoManager(config.ASSETS_FOLDER)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return config.is_allowed_file_extension(filename)


def validate_video_url(url: str):
    """
    Validate video URL domain.

    Returns:
        Tuple of (error_response, status_code) or None if valid.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Popular supported domains (yt-dlp supports 1849+ sites)
        popular_domains = [
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
        ]

        # Check if it's a known popular domain or let yt-dlp handle it
        is_known_domain = any(
            valid_domain in domain for valid_domain in popular_domains
        )

        # For unknown domains, we'll let yt-dlp try and handle the error gracefully
        if not is_known_domain:
            logger.info(f"Unknown domain {domain}, will let yt-dlp attempt extraction")

        return None  # No error

    except Exception:
        return jsonify({"error": t("errors:validation.url_invalid_format")}), 400


def build_watermark_config(watermark_enabled: bool, request):
    """
    Build watermark configuration from form request.

    Returns:
        Tuple of (config_dict, error_message)
        error_message is None if successful
    """
    if not watermark_enabled:
        return {"enabled": False}, None

    opacity, opacity_error = safe_int(request.form.get("watermark_opacity"), 40, 0, 100)
    if opacity_error:
        return None, f"Invalid watermark_opacity: {opacity_error}"

    watermark_config = {
        "enabled": True,
        "position": request.form.get("watermark_position", "top-right"),
        "size": request.form.get("watermark_size", "medium"),
        "opacity": opacity,
    }

    # Handle custom logo upload
    if "watermark_logo" in request.files:
        logo_file = request.files["watermark_logo"]
        if logo_file and logo_file.filename:
            file_content = logo_file.read()
            extension = logo_file.filename.rsplit(".", 1)[1].lower()
            logo_path, is_new = logo_manager.save_logo(file_content, extension)
            watermark_config["custom_logo_path"] = logo_path
            session["custom_logo_path"] = logo_path
            logger.info(
                f"{'Saved new' if is_new else 'Reusing existing'} logo: {os.path.basename(logo_path)}"
            )
    else:
        # Check for logo data URL
        logo_data_url = request.form.get("watermark_logo_url")
        if logo_data_url:
            _process_logo_data_url(logo_data_url, watermark_config)
        elif "custom_logo_path" in session:
            _use_session_logo(watermark_config)

    return watermark_config, None


def build_watermark_config_from_data(watermark_enabled: bool, data: dict, request):
    """
    Build watermark configuration from JSON/form data.

    Returns:
        Tuple of (config_dict, error_message)
        error_message is None if successful
    """
    if not watermark_enabled:
        return {"enabled": False}, None

    raw_opacity = data.get("watermark_opacity", 40)
    opacity, opacity_error = safe_int(raw_opacity, 40, 0, 100)
    if opacity_error:
        return None, f"Invalid watermark_opacity: {opacity_error}"

    watermark_config = {
        "enabled": True,
        "position": data.get("watermark_position", "top-right"),
        "size": data.get("watermark_size", "medium"),
        "opacity": opacity,
    }

    # Handle custom logo upload (only available with FormData)
    if "watermark_logo" in request.files:
        logo_file = request.files["watermark_logo"]
        if logo_file and logo_file.filename:
            file_content = logo_file.read()
            extension = logo_file.filename.rsplit(".", 1)[1].lower()
            logo_path, is_new = logo_manager.save_logo(file_content, extension)
            watermark_config["custom_logo_path"] = logo_path
            session["custom_logo_path"] = logo_path
            logger.info(
                f"{'Saved new' if is_new else 'Reusing existing'} logo for YouTube: {os.path.basename(logo_path)}"
            )
    else:
        # Check for logo data URL
        if request.content_type and "application/json" in request.content_type:
            logo_data_url = data.get("watermark_logo_url")
        else:
            logo_data_url = request.form.get("watermark_logo_url")

        if logo_data_url:
            _process_logo_data_url(logo_data_url, watermark_config, "YouTube")
        elif "custom_logo_path" in session:
            _use_session_logo(watermark_config, "YouTube")

    return watermark_config, None


def _process_logo_data_url(
    logo_data_url: str, watermark_config: dict, context: str = ""
):
    """Process a base64 logo data URL and add to config."""
    try:
        match = re.match(r"data:image/([\w\+\-]+);base64,(.+)", logo_data_url)
        if match:
            file_ext = match.group(1).replace("jpeg", "jpg")
            base64_data = match.group(2)
            file_content = base64.b64decode(base64_data)
            logo_path, is_new = logo_manager.save_logo(file_content, file_ext)
            watermark_config["custom_logo_path"] = logo_path
            session["custom_logo_path"] = logo_path
            ctx = f" for {context}" if context else ""
            logger.info(
                f"{'Saved new' if is_new else 'Reusing existing'} logo from data URL{ctx}: {os.path.basename(logo_path)}"
            )
    except Exception as e:
        logger.error(f"Failed to process logo data URL: {e}")


def _use_session_logo(watermark_config: dict, context: str = ""):
    """Use logo from session if available."""
    saved_logo_path = session["custom_logo_path"]
    if os.path.exists(saved_logo_path):
        watermark_config["custom_logo_path"] = saved_logo_path
        ctx = f" for {context}" if context else ""
        logger.info(f"Using previously saved logo{ctx}: {saved_logo_path}")
    else:
        session.pop("custom_logo_path", None)
        ctx = f" for {context}" if context else ""
        logger.warning(f"Previously saved logo not found{ctx}, removed from session")
