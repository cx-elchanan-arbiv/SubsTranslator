"""
Video processing routes for SubsTranslator
Handles video upload, YouTube processing, task status, and file downloads
"""

import base64
import os
import re

from celery.result import AsyncResult
from flask import Blueprint, jsonify, request, send_file, session
from werkzeug.exceptions import HTTPException

from config import get_config
from i18n.translations import t
from logging_config import get_logger
from logo_manager import LogoManager
from services import task_registry
from services.subtitle_pipeline import parse_subtitle_flags, parse_subtitle_position
from services.token_service import use_download_token
from services.url_resolver_service import resolve_video_url
from tasks import (
    download_and_process_youtube_task,
    download_youtube_only_task,
    process_video_task,
)
from utils.file_probe import probe_file_safe
from utils.file_utils import clean_filename, safe_int

# Configuration
config = get_config()
logger = get_logger(__name__)

# Initialize logo manager
logo_manager = LogoManager(config.ASSETS_FOLDER)

# Create blueprint
video_bp = Blueprint("video", __name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return config.is_allowed_file_extension(filename)


# =================== VIDEO PROCESSING ENDPOINTS ===================


@video_bp.route("/upload", methods=["POST"])
def upload_file_async():
    """Upload a file and start the processing task asynchronously."""
    try:
        # Every 4xx here carries a stable ``code``: these are the most common
        # user-facing failures, and without a code they reached a Hebrew screen
        # as raw English strings (bug #15). The code keys into errors.byCode.
        if "file" not in request.files:
            return jsonify({"error": "No file provided", "code": "NO_FILE"}), 400

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return (
                jsonify({"error": "Invalid file type", "code": "INVALID_FILE_TYPE"}),
                400,
            )

        # Check file size explicitly (in addition to Flask's MAX_CONTENT_LENGTH)
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning

        if file_size > config.MAX_FILE_SIZE:
            # Backstop only: Flask's MAX_CONTENT_LENGTH aborts oversized bodies
            # before this route runs (see the 413 handler in app.py). Same code
            # either way, so the UI cannot tell the two rejections apart.
            return (
                jsonify(
                    {
                        "error": f"File too large. Maximum size: {config.MAX_FILE_SIZE // (1024*1024)}MB",
                        "code": "FILE_TOO_LARGE",
                        "max_mb": config.MAX_FILE_SIZE // (1024 * 1024),
                    }
                ),
                413,
            )

        source_lang = request.form.get("source_lang", config.DEFAULT_SOURCE_LANG)
        target_lang = request.form.get("target_lang", config.DEFAULT_TARGET_LANG)
        auto_create_video = (
            request.form.get("auto_create_video", "false").lower() == "true"
        )
        whisper_model = request.form.get("whisper_model", config.DEFAULT_WHISPER_MODEL)
        translation_service = request.form.get("translation_service", "google")

        # Opt-in subtitle-quality toggles. Multipart form fields are always strings, so
        # parsing goes through parse_subtitle_flags rather than a bare truthiness check
        # (the string "false" is truthy).
        subtitle_flags = parse_subtitle_flags(request.form)
        subtitle_position = parse_subtitle_position(
            request.form.get("subtitle_position")
        )

        # Handle watermark configuration
        watermark_enabled = (
            request.form.get("watermark_enabled", "false").lower() == "true"
        )
        watermark_config, watermark_error = _build_watermark_config(
            watermark_enabled, request
        )
        if watermark_error:
            return jsonify({"error": watermark_error}), 400

        if not config.is_valid_whisper_model(whisper_model):
            return (
                jsonify(
                    {
                        "error": f"Invalid whisper model. Supported models: {list(config.AVAILABLE_WHISPER_MODELS)}"
                    }
                ),
                400,
            )

        # Server-side enforcement: Block restricted models in hosted mode
        if config.is_model_restricted(whisper_model):
            return (
                jsonify(
                    {
                        "error": t("whisperModels.proOnlyTooltip")
                        or "This model is available for PRO users only",
                        "code": "MODEL_RESTRICTED",
                        "restricted_model": whisper_model,
                    }
                ),
                403,
            )

        # clean_filename, not werkzeug's secure_filename: secure_filename strips
        # every non-ASCII character, so a Hebrew upload like "הרצאה.mp4" became
        # literally "mp4". clean_filename keeps the language and still guarantees
        # the same traversal safety (no separators, no "..", no leading dot) —
        # pinned by tests/unit/test_clean_filename_unit.py.
        filename = clean_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"File saved: {filename}")

        # Extract file metadata using ffprobe
        file_metadata, probe_error = probe_file_safe(filepath)

        if probe_error:
            # File probe failed - return error immediately
            error_messages = {
                "FILE_NOT_FOUND": "File was saved but could not be accessed",
                "UNSUPPORTED_MEDIA": "File format is not supported. Please upload a valid video or audio file.",
                "PROBE_FAILED": "Failed to analyze media file. The file may be corrupted or in an unsupported format.",
            }
            error_msg = error_messages.get(
                probe_error, "Failed to process uploaded file"
            )
            logger.error(f"File probe failed for {filename}: {probe_error}")

            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({"error": error_msg, "code": probe_error}), 400

        # Prepare processing_info with user choices and file metadata.
        # The four subtitle-quality toggles are user choices like any other: without
        # them in this dict, a completed job cannot be attributed to the settings that
        # produced it, which is the whole point of the experiment.
        processing_info = {
            "file_metadata": file_metadata,
            "user_choices": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "auto_create_video": auto_create_video,
                "whisper_model": whisper_model,
                "translation_service": translation_service,
                "subtitle_position": subtitle_position,
                **subtitle_flags,
            },
        }

        task = process_video_task.apply_async(
            args=[
                filepath,
                source_lang,
                target_lang,
                auto_create_video,
                whisper_model,
                translation_service,
                watermark_config,
                None,  # initial_timing_summary (not applicable for uploads)
                processing_info,  # Include metadata and user choices
            ],
            kwargs={**subtitle_flags, "subtitle_position": subtitle_position},
            queue="processing",
        )

        # Return 202 with unified schema as per spec
        return (
            jsonify(
                {
                    "task_id": task.id,
                    "state": "PENDING",
                    "user_choices": {
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "auto_create_video": auto_create_video,
                        "whisper_model": whisper_model,
                        "translation_service": translation_service,
                        "subtitle_position": subtitle_position,
                        **subtitle_flags,
                    },
                    "initial_request": {"filename": filename, "type": "upload"},
                    "file_metadata": file_metadata,
                    "video_metadata": None,
                    "progress": {"overall_percent": 0, "steps": []},
                    "result": None,
                    "error": None,
                }
            ),
            202,
        )

    except HTTPException:
        # Flask/werkzeug aborts (e.g. the 413 from MAX_CONTENT_LENGTH) carry
        # their own status and are answered by their errorhandlers — the
        # catch-all below used to swallow them into a generic 500.
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/youtube", methods=["POST"])
def process_youtube_async():
    """Process a YouTube URL asynchronously."""
    try:
        # Server-side enforcement: Block YouTube in hosted mode (PRO-only feature)
        if config.is_youtube_restricted():
            return (
                jsonify(
                    {
                        "error": t("features.youtube_pro_only")
                        or "YouTube processing is available for PRO users only",
                        "code": "YOUTUBE_RESTRICTED",
                    }
                ),
                403,
            )

        # Handle both JSON and FormData (for custom logo uploads)
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json()
            if not data:
                return jsonify({"error": t("errors:validation.no_data")}), 400
        else:
            # FormData request - convert string values to proper types
            data = request.form.to_dict()
            # Convert boolean strings to actual booleans
            if "auto_create_video" in data:
                data["auto_create_video"] = data["auto_create_video"].lower() == "true"
            if "watermark_enabled" in data:
                data["watermark_enabled"] = data["watermark_enabled"].lower() == "true"

        url = data.get("url")
        if not url:
            return jsonify({"error": t("errors:validation.url_required")}), 400

        # Basic URL validation
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": t("errors:validation.url_invalid_protocol")}), 400

        # Validate URL domain
        validation_error = _validate_video_url(url)
        if validation_error:
            return validation_error

        source_lang = data.get("source_lang", config.DEFAULT_SOURCE_LANG)
        target_lang = data.get("target_lang", config.DEFAULT_TARGET_LANG)
        auto_create_video = data.get("auto_create_video", False)
        whisper_model = data.get("whisper_model", config.DEFAULT_WHISPER_MODEL)
        translation_service = data.get("translation_service", "google")

        # Opt-in subtitle-quality toggles. `data` is a JSON dict (real booleans) OR a
        # FormData dict (strings); parse_subtitle_flags accepts both.
        subtitle_flags = parse_subtitle_flags(data)
        subtitle_position = parse_subtitle_position(data.get("subtitle_position"))

        # Handle watermark configuration
        watermark_enabled = data.get("watermark_enabled", False)
        watermark_config, watermark_error = _build_watermark_config_from_data(
            watermark_enabled, data, request
        )
        if watermark_error:
            return jsonify({"error": watermark_error}), 400

        if not config.is_valid_whisper_model(whisper_model):
            return (
                jsonify(
                    {
                        "error": t(
                            "errors:validation.whisper_model_invalid",
                            models=list(config.AVAILABLE_WHISPER_MODELS),
                        )
                    }
                ),
                400,
            )

        # Server-side enforcement: Block restricted models in hosted mode
        if config.is_model_restricted(whisper_model):
            return (
                jsonify(
                    {
                        "error": t("whisperModels.proOnlyTooltip")
                        or "This model is available for PRO users only",
                        "code": "MODEL_RESTRICTED",
                        "restricted_model": whisper_model,
                    }
                ),
                403,
            )

        task = download_and_process_youtube_task.apply_async(
            args=[
                url,
                source_lang,
                target_lang,
                auto_create_video,
                whisper_model,
                translation_service,
                watermark_config,
            ],
            kwargs={**subtitle_flags, "subtitle_position": subtitle_position},
            queue="processing",
        )

        # Return 202 with unified schema as per spec
        return (
            jsonify(
                {
                    "task_id": task.id,
                    "state": "PENDING",
                    "user_choices": {
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "auto_create_video": auto_create_video,
                        "whisper_model": whisper_model,
                        "translation_service": translation_service,
                        "url": url,
                        "subtitle_position": subtitle_position,
                        **subtitle_flags,
                    },
                    "initial_request": {},
                    "video_metadata": None,
                    "progress": {"overall_percent": 0, "steps": []},
                    "result": None,
                    "error": None,
                }
            ),
            202,
        )

    except Exception as e:
        logger.error(f"Error processing YouTube link: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/youtube-preview", methods=["GET"])
def youtube_preview():
    """Instant title+thumbnail for a YouTube id — the Telegram trick.

    Telegram's sub-second link preview is not magic: YouTube thumbnails live at
    a DETERMINISTIC url (i.ytimg.com/vi/<id>/hqdefault.jpg, no API involved),
    and the title comes from the public, key-less oEmbed endpoint. This route
    proxies that one oEmbed call server-side (the browser cannot, for CORS
    reasons) so the form can show a preview the moment a link is pasted —
    without waking the heavyweight yt-dlp probe in /resolve-url.
    """
    import requests as _requests

    video_id = request.args.get("video_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return jsonify({"error": "invalid video id"}), 400
    try:
        resp = _requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            timeout=4,
        )
    except Exception:
        return jsonify({"error": "unavailable"}), 502
    if resp.status_code != 200:
        # Private/removed videos have no oEmbed — the form just skips the card.
        return jsonify({"error": "unavailable"}), 404
    data = resp.json()
    return jsonify(
        {
            "title": data.get("title"),
            "author": data.get("author_name"),
            "thumbnail": data.get("thumbnail_url")
            or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        }
    )


@video_bp.route("/resolve-url", methods=["POST"])
def resolve_url():
    """
    Probe a URL (direct video link OR a webpage that contains video) without
    downloading. Returns {"type": "single"|"multiple"|"none", ...} so the
    frontend can proceed, show a picker, or show a friendly message.
    """
    try:
        if config.is_youtube_restricted():
            return (
                jsonify(
                    {
                        "error": t("features.youtube_pro_only")
                        or "YouTube processing is available for PRO users only",
                        "code": "YOUTUBE_RESTRICTED",
                    }
                ),
                403,
            )

        data = request.get_json()
        if not data:
            return jsonify({"error": t("errors:validation.no_data")}), 400

        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": t("errors:validation.url_required")}), 400
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": t("errors:validation.url_invalid_protocol")}), 400

        validation_error = _validate_video_url(url)
        if validation_error:
            return validation_error

        result = resolve_video_url(url)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error resolving URL: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/download-video-only", methods=["POST"])
def download_video_only():
    """Download YouTube video without processing (transcription/translation)."""
    try:
        # Server-side enforcement: Block YouTube in hosted mode (PRO-only feature)
        if config.is_youtube_restricted():
            return (
                jsonify(
                    {
                        "error": t("features.youtube_pro_only")
                        or "YouTube download is available for PRO users only",
                        "code": "YOUTUBE_RESTRICTED",
                    }
                ),
                403,
            )

        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": t("errors:validation.url_required")}), 400

        # Basic URL validation
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": t("errors:validation.url_invalid_protocol")}), 400

        # Validate URL domain
        validation_error = _validate_video_url(url)
        if validation_error:
            return validation_error

        logger.info(f"Starting download-only task for URL: {url}")

        # Use the new download-only task that doesn't do any processing
        # "mp4" keeps today's behaviour for any caller that does not ask; only an
        # explicit "mp3" changes anything. Validated here rather than passed through,
        # so an unexpected value cannot reach yt-dlp's option builder.
        media_format = str(data.get("media_format") or "mp4").lower()
        if media_format not in ("mp4", "mp3"):
            media_format = "mp4"

        task = download_youtube_only_task.apply_async(
            args=[url, "high", None, None, media_format], queue="processing"
        )

        # Return 202 with unified schema as per spec
        return (
            jsonify(
                {
                    "task_id": task.id,
                    "state": "PENDING",
                    "user_choices": {},
                    "initial_request": {
                        "url": url,
                        "quality": "high",
                        "type": "download_only",
                        # So the progress screen can say WHAT is being downloaded
                        # (🎬/🎵) from the very first render, not only at the end.
                        "media_format": media_format,
                    },
                    "video_metadata": None,
                    "progress": {"overall_percent": 0, "steps": []},
                    "result": None,
                    "error": None,
                }
            ),
            202,
        )

    except Exception as e:
        logger.error(f"Download-only task failed: {e}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Get the status of a background task with unified schema."""
    task_result = AsyncResult(task_id, app=process_video_task.app)

    status = task_result.state

    # Celery reports PENDING for ids it has never seen — an expired job and a
    # made-up id look exactly like a queued one, and a browser that remembered a
    # dead task id used to poll that PENDING forever (an eternal 0% spinner).
    # The registry knows which ids were really published; PENDING without its
    # marker means "gone", and saying so lets the frontend clean up and go home.
    if status == "PENDING" and not task_registry.is_known(task_id):
        return (
            jsonify(
                {
                    "task_id": task_id,
                    "state": "UNKNOWN",
                    "error": {
                        "code": "TASK_UNKNOWN",
                        "message": f"Task {task_id} is not known to the system",
                        "user_facing_message": (
                            "This job is no longer available — it has likely "
                            "expired. Start a new one."
                        ),
                        "recoverable": False,
                    },
                }
            ),
            404,
        )
    result = None
    error_info = None
    progress_info = {"overall_percent": 0, "steps": []}
    video_metadata = None
    user_choices = {}
    initial_request = {}
    # Bind once, up front. Each branch below only ASSIGNS when the payload actually
    # carries logs — no branch clears it. A SUCCESS payload keeps its logs on the inner
    # `result` dict, and the outer success branch used to reset this to None right after
    # collecting them, so the operator flipped EXPOSE_PROGRESS_LOGS and still saw nothing.
    progress_logs = None

    if status == "SUCCESS":
        result = task_result.result
        if result and isinstance(result, dict):
            # Extract nested data
            if "result" in result:
                inner_result = result.get("result", {})
                if inner_result and isinstance(inner_result, dict):
                    if "progress" in inner_result:
                        progress_info = {
                            "overall_percent": 100,
                            "steps": inner_result["progress"],
                        }
                    video_metadata = inner_result.get("video_metadata")
                    user_choices = inner_result.get("user_choices", {})
                    # Optional progress logs exposure
                    if config.EXPOSE_PROGRESS_LOGS and "logs" in inner_result:
                        progress_logs = inner_result.get("logs", [])

            # Check if the task actually failed despite SUCCESS state
            if result.get("status") == "DOWNLOAD_FAILED":
                status = "FAILURE"
                error_info = {
                    "code": "DOWNLOAD_FAILED",
                    "message": result.get("error", "Download failed"),
                    "user_facing_message": "Download failed. Please try again.",
                    "recoverable": True,
                }
            elif result.get("status") == "FAILURE":
                status = "FAILURE"
                error_info = {
                    "code": result.get("code", "TASK_FAILED"),
                    "message": result.get(
                        "message", result.get("error", "Task failed")
                    ),
                    "user_facing_message": result.get(
                        "user_facing_message", "Processing failed. Please try again."
                    ),
                    "recoverable": result.get("recoverable", True),
                    **(
                        {"provider": result["provider"]}
                        if result.get("provider")
                        else {}
                    ),
                }
            else:
                # Extract metadata from successful result
                video_metadata = result.get("video_metadata") or video_metadata
                user_choices = result.get("user_choices", {}) or user_choices
                initial_request = result.get("initial_request", {})
                # Optional progress logs exposure
                if config.EXPOSE_PROGRESS_LOGS and "logs" in result:
                    progress_logs = result.get("logs", [])

    elif status == "FAILURE":
        logger.info(
            f"FAILURE result type: {type(task_result.result)}, content: {task_result.result}"
        )
        if isinstance(task_result.result, Exception):
            error_message = str(task_result.result)
            error_info = {
                "code": "TASK_EXCEPTION",
                "message": error_message,
                "user_facing_message": "An error occurred during processing. Please try again.",
                "recoverable": True,
            }
        elif isinstance(task_result.result, dict) and "code" in task_result.result:
            # Use detailed error info from our improved error handling
            result_dict = task_result.result
            error_info = {
                "code": result_dict.get("code", "TASK_FAILED"),
                "message": result_dict.get("message", "Task failed"),
                "user_facing_message": result_dict.get(
                    "user_facing_message",
                    "An error occurred during processing. Please try again.",
                ),
                "recoverable": result_dict.get("recoverable", True),
                # Which translation provider was on the failing job, when known.
                **(
                    {"provider": result_dict["provider"]}
                    if result_dict.get("provider")
                    else {}
                ),
            }
            error_message = result_dict.get("message", "Task failed")
            # Optional progress logs exposure
            if config.EXPOSE_PROGRESS_LOGS and "logs" in result_dict:
                progress_logs = result_dict.get("logs", [])
        else:
            error_message = f"Task failed: {str(task_result.result)}"
            error_info = {
                "code": "TASK_FAILED",
                "message": error_message,
                "user_facing_message": "An error occurred during processing. Please try again.",
                "recoverable": True,
            }
        logger.error(f"Task {task_id} failed with error: {error_message}")

    elif status == "PROGRESS":
        if task_result.info and isinstance(task_result.info, dict):
            # Extract progress info - FLAT STRUCTURE ONLY
            progress_info = {
                "overall_percent": task_result.info.get("overall_percent", 0),
                "steps": task_result.info.get("steps", []),
            }
            # Extract metadata if available
            video_metadata = task_result.info.get("video_metadata")
            user_choices = task_result.info.get("user_choices", {})
            initial_request = task_result.info.get("initial_request", {})
            # Optional progress logs exposure
            if config.EXPOSE_PROGRESS_LOGS and "logs" in task_result.info:
                progress_logs = task_result.info.get("logs", [])

    # Tail logs if exposed
    logs_tail = None
    if config.EXPOSE_PROGRESS_LOGS and isinstance(progress_logs, list):
        tail_n = max(0, int(config.PROGRESS_LOGS_TAIL))
        logs_tail = progress_logs[-tail_n:]

    response = {
        "task_id": task_id,
        "state": status,
        "progress": progress_info,
        "video_metadata": video_metadata,
        # A failed run can still have produced files worth keeping — the transcript
        # and the .srt survive a failed burn or a failed translation, and the task
        # layer marks that payload `salvaged`. Blanking the whole result on any error
        # threw those away at the last step, so the user paid for a transcription and
        # was handed nothing. On failure, pass through the salvage (and only that).
        "result": (
            result
            if status == "SUCCESS" and not error_info
            else (
                {k: v for k, v in result.items() if k in ("files", "salvaged")}
                if isinstance(result, dict) and result.get("salvaged")
                else None
            )
        ),
        "user_choices": user_choices,
        "initial_request": initial_request,
        "error": error_info,
        **({"logs": logs_tail} if logs_tail is not None else {}),
    }

    return jsonify(response)


@video_bp.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Download processed files, preventing path traversal and (optionally) requiring a token."""
    safe_dir = os.path.abspath(config.DOWNLOADS_FOLDER)
    requested_path = os.path.abspath(os.path.join(safe_dir, filename))

    if not requested_path.startswith(safe_dir):
        return jsonify({"error": "Forbidden"}), 403

    if not os.path.exists(requested_path):
        return jsonify({"error": "File not found"}), 404

    # Optional one-time token enforcement
    if config.REQUIRE_DOWNLOAD_TOKEN:
        token = request.args.get("token")
        if not token:
            return jsonify({"error": "Token required"}), 401
        resolved_filename, err = use_download_token(token)
        if err:
            return jsonify({"error": err}), 401
        # Ensure token matches requested file
        if os.path.basename(resolved_filename) != os.path.basename(filename):
            return jsonify({"error": "Token-file mismatch"}), 403

    # Set MIME type for .srt files to text/plain with UTF-8 charset for better macOS compatibility
    if requested_path.lower().endswith(".srt"):
        return send_file(
            requested_path,
            as_attachment=True,
            mimetype="text/plain; charset=utf-8",
            download_name=os.path.basename(requested_path) + ".txt",
        )

    return send_file(requested_path, as_attachment=True)


@video_bp.route("/clear-watermark-logo", methods=["POST"])
def clear_watermark_logo():
    """Clear the saved watermark logo from session."""
    try:
        if "custom_logo_path" in session:
            logo_path = session.pop("custom_logo_path", None)
            # Optionally delete the file
            if logo_path and os.path.exists(logo_path):
                try:
                    os.remove(logo_path)
                    logger.info(f"Deleted watermark logo file: {logo_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete logo file {logo_path}: {e}")

            logger.info("Watermark logo cleared from session")
            return jsonify({"success": True, "message": "Watermark logo cleared"}), 200
        else:
            return (
                jsonify({"success": True, "message": "No watermark logo to clear"}),
                200,
            )

    except Exception as e:
        logger.error(f"Error clearing watermark logo: {str(e)}")
        return jsonify({"error": str(e)}), 500


@video_bp.route("/cleanup-logos", methods=["POST"])
def cleanup_logos():
    """Clean up old logo files (admin endpoint)"""
    try:
        # Only allow in development or with admin key
        admin_key = request.headers.get("X-Admin-Key")
        if os.getenv("FLASK_ENV") != "development" and admin_key != os.getenv(
            "ADMIN_KEY"
        ):
            return jsonify({"error": "Unauthorized"}), 401

        # Run cleanup (default: remove logos older than 24 hours)
        keep_hours = request.json.get("keep_hours", 24) if request.is_json else 24
        logo_manager.cleanup_old_logos(keep_hours)

        # Get current logo stats
        logos = logo_manager.get_all_logos()

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Cleanup completed. Kept logos from last {keep_hours} hours.",
                    "current_logos": len(logos),
                    "logos": logos,
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Failed to cleanup logos: {e}")
        return jsonify({"error": str(e)}), 500


# =================== HELPER FUNCTIONS ===================


def _validate_video_url(url):
    """Validate video URL domain. Returns error response or None if valid."""
    from urllib.parse import urlparse

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


def _build_watermark_config(watermark_enabled, request):
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


def _build_watermark_config_from_data(watermark_enabled, data, request):
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


def _process_logo_data_url(logo_data_url, watermark_config, context=""):
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


def _use_session_logo(watermark_config, context=""):
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
