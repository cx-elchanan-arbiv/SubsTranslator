"""
Health, config, and metadata routes for SubsTranslator
Handles health checks, feature flags, and configuration endpoints
"""

import os
import shutil
from datetime import datetime

try:
    import psutil
except ImportError:  # optional; the fingerprint above is the load-bearing field
    psutil = None
import subprocess

from flask import Blueprint, jsonify

from config import get_config
from logging_config import get_logger

# Configuration
config = get_config()
logger = get_logger(__name__)

# Create blueprint
health_bp = Blueprint("health", __name__)


def _is_valid_openai_key(api_key: str) -> bool:
    """
    Check if OpenAI API key is valid (not None, not empty, not placeholder).

    Args:
        api_key: The API key to validate

    Returns:
        bool: True if the key appears to be valid, False otherwise
    """
    if not api_key:
        return False

    # Check for common placeholder values
    placeholder_values = {
        "your-openai-api-key-here",
        "your-api-key-here",
        "sk-your-key-here",
        "placeholder",
        "changeme",
        "replace-me",
    }

    if api_key.lower() in placeholder_values:
        return False

    # Basic format check - OpenAI keys should start with 'sk-'
    if not api_key.startswith("sk-"):
        return False

    # Key should have reasonable length (OpenAI keys are typically 51+ chars)
    if len(api_key) < 20:
        return False

    return True


@health_bp.route("/", methods=["GET"])
def root():
    """Root endpoint - service information"""
    return (
        jsonify(
            {
                "ok": True,
                "service": "SubsTranslator API",
                "version": "1.0.0",
                "environment": os.getenv("FLASK_ENV", "production"),
                "endpoints": {
                    "health": "/health",
                    "upload": "/upload",
                    "youtube": "/youtube",
                    "status": "/status/<task_id>",
                    "documentation": "https://github.com/cx-elchanan-arbiv/SubsTranslator",
                },
            }
        ),
        200,
    )


@health_bp.route("/ping", methods=["GET"])
def ping():
    """Simple ping endpoint"""
    return "pong", 200


@health_bp.route("/healthz", methods=["GET"])
def healthz():
    """Health check endpoint (alias for /health for Render compatibility)"""
    return jsonify(
        {
            "status": "healthy",
            "message": "SubsTranslator is running!",
            "ffmpeg_installed": shutil.which("ffmpeg") is not None,
        }
    )


def _running_code_fingerprint() -> str:
    """Newest mtime among the loaded service/task modules, as an ISO timestamp.

    Not a git SHA: this process may be running code that was edited on a
    bind-mounted volume after it started, which no commit id can express. What
    matters operationally is WHEN the code this process actually imported was last
    written — a long-running Celery worker keeps serving whatever it imported at
    startup, and every one of those staleness incidents looked like a code bug
    until someone checked the container's start time. Reporting it turns "is this
    running my change?" from an investigation into one request.
    """
    import os
    import sys
    from datetime import datetime

    newest = 0.0
    for name, module in list(sys.modules.items()):
        # Every package this app actually runs from — not just services/ and
        # tasks/. The old filter was blind to api/, core/, utils/, i18n/ and the
        # top-level modules, so an edit to a route or to config never moved the
        # fingerprint and the staleness comparison said "fresh" anyway.
        if not (
            name.startswith(("services.", "tasks.", "api.", "core.", "utils.", "i18n."))
            or name
            in ("config", "app", "celery_worker", "celery_config", "logging_config")
        ):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            newest = max(newest, os.path.getmtime(path))
        except OSError:
            continue
    return (
        datetime.fromtimestamp(newest).isoformat(timespec="seconds")
        if newest
        else "unknown"
    )


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    import time

    return jsonify(
        {
            "status": "healthy",
            "message": "SubsTranslator is running!",
            "ffmpeg_installed": shutil.which("ffmpeg") is not None,
            # Both are needed to spot a stale process: code newer than the start
            # time means this process has NOT loaded it.
            "code_last_modified": _running_code_fingerprint(),
            "process_started": (
                datetime.fromtimestamp(psutil.Process().create_time()).isoformat(
                    timespec="seconds"
                )
                if psutil
                else time.strftime("%Y-%m-%dT%H:%M:%S")
            ),
        }
    )


@health_bp.route("/api/features", methods=["GET"])
def get_features():
    """Return feature flags for frontend."""
    return jsonify(
        {
            "youtube_download_enabled": config.ENABLE_YOUTUBE_DOWNLOAD,
            "youtube_restricted": config.is_youtube_restricted(),
            "hosted_mode": config.HOSTED_MODE,
        }
    )


@health_bp.route("/health/deps", methods=["GET"])
def health_deps():
    """
    Diagnostic endpoint to check all dependencies.
    Returns status of Redis, Celery, ffmpeg, and yt-dlp.
    """
    deps = {}

    # Check Redis connection
    try:
        import redis

        redis_client = redis.from_url(config.REDIS_URL)
        redis_client.ping()
        deps["redis"] = "ok"
    except Exception as e:
        deps["redis"] = f"error: {e.__class__.__name__}"

    # Check Celery broker connection
    try:
        from celery_worker import celery_app

        # Ping with 1 second timeout
        celery_app.control.inspect().ping()
        deps["celery"] = "ok"
    except Exception as e:
        deps["celery"] = f"error: {e.__class__.__name__}"

    # Check ffmpeg installation
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
        deps["ffmpeg"] = "ok"
    except Exception as e:
        deps["ffmpeg"] = f"error: {e.__class__.__name__}"

    # Check yt-dlp installation and version
    try:
        import yt_dlp

        version = yt_dlp.version.__version__
        deps["yt_dlp"] = f"ok (v{version})"

        # Quick test: try to extract info from a public video
        # (Only metadata, no download - should be fast)
        try:
            test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
            ydl_opts = {
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 10,
                "extract_flat": False,
                "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(test_url, download=False)
                if info and info.get("title"):
                    deps["yt_dlp_test"] = "ok"
                else:
                    deps["yt_dlp_test"] = "warn: no metadata"
        except Exception as test_e:
            deps["yt_dlp_test"] = f"error: {str(test_e)[:50]}"
    except Exception as e:
        deps["yt_dlp"] = f"error: {e.__class__.__name__}"

    # Check rate limiter storage
    try:
        # Import get_storage_uri from app if available
        from app import get_storage_uri

        storage_uri = get_storage_uri()
        deps["limiter_storage"] = storage_uri
    except Exception as e:
        deps["limiter_storage"] = f"error: {e.__class__.__name__}"

    return jsonify(deps), 200


@health_bp.route("/languages", methods=["GET"])
def get_languages():
    """Get supported languages with proper i18n"""
    from i18n.translations import t
    from shared_config import ALL_LANGUAGES

    # Build languages dict with proper translations
    languages = {}

    # Add auto-detect with proper translation
    languages["auto"] = t("languages.autoDetect", default="Auto Detect")

    # Add all other languages with their native names and translations
    for code, info in ALL_LANGUAGES.items():
        if code != "auto":  # Skip auto as we already added it
            # Try to get translated name, fallback to native name
            translated_name = t(f"languages.{code}", default=info["nativeName"])
            languages[code] = translated_name

    return jsonify(languages)


@health_bp.route("/translation-services", methods=["GET"])
def get_translation_services():
    """Get available translation services and their status"""
    from i18n.translations import t

    services = {
        "google": {
            "name": "Google Translate",
            "available": True,
            "description": t("translationServices.google.description")
            or "Free Google translation service",
        },
        "openai": {
            "name": "OpenAI (GPT-4o)",
            "available": _is_valid_openai_key(config.OPENAI_API_KEY),
            "description": (
                (
                    t("translationServices.openai.description")
                    or "Advanced translation using GPT-4o"
                )
                if _is_valid_openai_key(config.OPENAI_API_KEY)
                else (
                    t("translationServices.openai.requiresKey")
                    or "OpenAI API key required"
                )
            ),
        },
    }

    return jsonify(services)


@health_bp.route("/whisper-models", methods=["GET"])
def get_whisper_models():
    """Get available Whisper models with user-friendly descriptions"""
    from i18n.translations import t
    from services.whisper_smart import SmartWhisperManager

    manager = SmartWhisperManager()
    model_capabilities = manager.get_available_models()

    # Check if we're in hosted mode (restricts some models to PRO)
    is_hosted = config.HOSTED_MODE
    pro_only_models = config.PRO_ONLY_MODELS

    # Format the model data for frontend consumption
    model_options = {}
    for model_name, capabilities in model_capabilities.items():
        # Check if this model is restricted in hosted mode
        is_restricted = is_hosted and model_name in pro_only_models

        model_options[model_name] = {
            "name": model_name,
            "display_name": model_name.title(),
            "accuracy": capabilities.get("accuracy", "unknown"),
            "speed": capabilities.get("speed", "unknown"),
            "languages": capabilities.get("languages", "all"),
            "description": f"{capabilities.get('accuracy', 'Unknown')} accuracy, {capabilities.get('speed', 'unknown')} speed",
            # Restriction info for frontend
            "restricted": is_restricted,
            "restrictedReason": (
                (t("whisperModels.proOnlyTooltip") or "Available for PRO users only")
                if is_restricted
                else None
            ),
            # Only show "pro" tier when in hosted mode AND model is in pro_only_models
            # In self-hosted mode, all models are "free" (unlocked)
            "tier": "pro" if (is_hosted and model_name in pro_only_models) else "free",
        }

    return jsonify(
        {
            "models": model_options,
            "default": "base",  # Default for production (2GB RAM Worker)
            "recommended": "base",  # Safe choice for most instances
            "hostedMode": is_hosted,  # Let frontend know if restrictions apply
        }
    )
