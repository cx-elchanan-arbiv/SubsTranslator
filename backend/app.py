#!/usr/bin/env python3
"""
SubsTranslator - AI-Powered Video Subtitle Translation
Now with asynchronous processing using Celery and Redis!
"""

import os

from dotenv import load_dotenv

# Load environment variables BEFORE importing config or other modules that read env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import get_config
from logo_manager import LogoManager

# Set testing environment variables if running in CI
if os.getenv("CI") == "true" or "pytest" in os.environ.get("_", ""):
    os.environ["FLASK_ENV"] = "testing"
    os.environ["TESTING"] = "true"
    os.environ["FLASK_TESTING"] = "1"
    os.environ["DISABLE_RATE_LIMIT"] = "1"

# Get configuration
config = get_config()

# Initialize logo manager
logo_manager = LogoManager(config.ASSETS_FOLDER)

# Flask app setup
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE  # Set file size limit


@app.errorhandler(413)
def _payload_too_large(_error):
    """Flask rejects oversized bodies BEFORE any route runs, and its default 413
    is an HTML page — response.json() in the frontend then throws and the user
    got a generic techno-string. JSON with a stable code lets the UI show the
    same coded Hebrew card every other failure gets (bug #15)."""
    max_mb = config.MAX_FILE_SIZE // (1024 * 1024)
    return (
        jsonify(
            {
                "error": f"File too large. Maximum size: {max_mb}MB",
                "code": "FILE_TOO_LARGE",
                "max_mb": max_mb,
            }
        ),
        413,
    )


# SECRET_KEY is required for session security - must be set in production
_secret_key = os.getenv("SECRET_KEY")
if not _secret_key or _secret_key in (
    "your-secret-key-change-in-production",
    "changeme",
    "secret",
):
    if os.getenv("FLASK_ENV") == "production" or not os.getenv("FLASK_TESTING"):
        import warnings

        warnings.warn(
            "SECRET_KEY not set or using weak default. Set a strong SECRET_KEY in production!"
        )
    _secret_key = "dev-only-insecure-key-do-not-use-in-production"
app.config["SECRET_KEY"] = _secret_key
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["DOWNLOADS_FOLDER"] = config.DOWNLOADS_FOLDER

# CORS configuration - fully configurable via environment variable
# Set CORS_ORIGINS to a comma-separated list of allowed origins
# Example: CORS_ORIGINS="https://example.com,https://app.example.com"
cors_origins = os.getenv("CORS_ORIGINS", "")

# Local development origins - always allowed for convenience
local_origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if cors_origins:
    # Use specific origins from environment variable + local origins
    origins_list = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]
    origins_list.extend(local_origins)
else:
    # No CORS_ORIGINS set - only allow local development
    # In production, you should set CORS_ORIGINS explicitly
    origins_list = local_origins

CORS(
    app,
    resources={
        r"/*": {
            "origins": origins_list,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
        }
    },
)


# Rate limiting setup - test-friendly configuration
def get_storage_uri():
    """
    Get storage URI for rate limiter with proper fallback chain:
    1. LIMITER_STORAGE_URI (explicit limiter config)
    2. REDIS_URL (shared Redis instance)
    3. memory:// (safe fallback to prevent 500 errors)

    Note: memory:// is not consistent across multiple instances,
    but it's better than crashing production.
    """
    if os.getenv("FLASK_TESTING") == "1" or os.getenv("TESTING", "").lower() == "true":
        return "memory://"  # Use in-memory storage for tests

    # Check for explicit limiter storage URI (e.g., Upstash Redis)
    limiter_uri = os.getenv("LIMITER_STORAGE_URI")
    if limiter_uri:
        return limiter_uri

    # Fall back to shared REDIS_URL (if configured)
    if config.REDIS_URL and not config.REDIS_URL.startswith("redis://redis:"):
        return config.REDIS_URL

    # Safe fallback to prevent 500 errors in production
    # (memory:// is not ideal for multi-instance, but better than crashing)
    return "memory://"


# Initialize rate limiter only if not disabled
if os.getenv("DISABLE_RATE_LIMIT") != "1":
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["100 per hour", "20 per minute"],
        storage_uri=get_storage_uri(),
    )
else:
    # Create a mock limiter that doesn't apply any limits
    class MockLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

        def exempt(self, f):
            return f

        def request_filter(self, f):
            """Mock request filter - always allows requests"""
            return f

    limiter = MockLimiter()

# Create directories only if not in testing mode
if not (
    os.getenv("FLASK_TESTING") == "1" or os.getenv("TESTING", "").lower() == "true"
):
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.DOWNLOADS_FOLDER, exist_ok=True)
else:
    # In testing mode, create directories with proper permissions if they don't exist
    for folder in [config.UPLOAD_FOLDER, config.DOWNLOADS_FOLDER]:
        try:
            os.makedirs(folder, exist_ok=True)
        except (OSError, PermissionError):
            # If we can't create the directory, that's OK for tests
            pass

# Configure structured logging
from logging_config import (
    get_logger,
    setup_logging,
)

setup_logging(
    level=config.LOG_LEVEL,
    testing=os.getenv("TESTING", "").lower() == "true",
    json_logs=os.getenv("JSON_LOGS", "").lower() == "true",
)
logger = get_logger(__name__)

# Initialize i18n system
from i18n.translations import init_i18n

init_i18n(app)

from api.editing_routes import editing_bp

# Register blueprints.
# `_is_valid_openai_key` is re-exported on purpose: it lives in api.health_routes,
# but `from app import _is_valid_openai_key` is the established import path for the
# test suite (tests/unit/test_critical_security_paths.py and friends). Keep the alias
# here so the entrypoint stays the stable surface — it is not dead code.
from api.health_routes import _is_valid_openai_key, health_bp  # noqa: F401
from api.stats_routes import stats_bp
from api.summary_routes import summary_bp
from api.video_routes import video_bp

app.register_blueprint(health_bp)
app.register_blueprint(video_bp)
app.register_blueprint(stats_bp)
app.register_blueprint(editing_bp)
app.register_blueprint(summary_bp)

# Apply limiter exemptions to blueprint routes
limiter.exempt(health_bp)


# Exempt status polling endpoint from rate limiting (needs frequent polling)
@limiter.request_filter
def exempt_status_endpoint():
    """Exempt /status/<task_id> from rate limiting for polling"""
    return request.endpoint == "video.get_task_status"


# Initialize download token service
from services.token_service import start_cleanup_scheduler

start_cleanup_scheduler()

# Import custom exceptions
from core.exceptions import FfmpegNotInstalledError

# Import utility functions
from utils import check_ffmpeg

# =================== API ENDPOINTS ===================


# =================== SERVER STARTUP ===================

if __name__ == "__main__":
    try:
        check_ffmpeg()
        logger.info(f"🚀 Starting {config.APP_NAME} v{config.APP_VERSION}...")
        logger.info("🎉 All systems ready! Backend server is up and running! ✅")
        app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
    except FfmpegNotInstalledError as e:
        logger.error(f"Startup failed: {e}")
        exit(1)
