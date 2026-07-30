"""
Upload routes for SubsTranslator API v1.
Handles file upload and processing initiation.
"""
import os

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import get_config
from tasks import process_video_task
from services.subtitle_pipeline import parse_subtitle_flags
from utils.file_probe import probe_file_safe
from logging_config import get_logger
from i18n.translations import t
from .helpers import allowed_file, build_watermark_config

# Configuration
config = get_config()
logger = get_logger(__name__)

# Create blueprint
upload_bp = Blueprint('upload', __name__)


@upload_bp.route("/upload", methods=["POST"])
def upload_file_async():
    """Upload a file and start the processing task asynchronously."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Check file size explicitly (in addition to Flask's MAX_CONTENT_LENGTH)
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning

        if file_size > config.MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Maximum size: {config.MAX_FILE_SIZE // (1024*1024)}MB"}), 413

        source_lang = request.form.get("source_lang", config.DEFAULT_SOURCE_LANG)
        target_lang = request.form.get("target_lang", config.DEFAULT_TARGET_LANG)
        auto_create_video = (
            request.form.get("auto_create_video", "false").lower() == "true"
        )
        whisper_model = request.form.get("whisper_model", config.DEFAULT_WHISPER_MODEL)
        translation_service = request.form.get("translation_service", "google")

        # Opt-in subtitle-quality toggles (form fields are strings; "false" is truthy)
        subtitle_flags = parse_subtitle_flags(request.form)

        # Handle watermark configuration
        watermark_enabled = (
            request.form.get("watermark_enabled", "false").lower() == "true"
        )
        watermark_config, watermark_error = build_watermark_config(watermark_enabled, request)
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
                        "error": t("whisperModels.proOnlyTooltip") or "This model is available for PRO users only",
                        "code": "MODEL_RESTRICTED",
                        "restricted_model": whisper_model
                    }
                ),
                403,
            )

        safe_filename = secure_filename(file.filename)
        filename = safe_filename.replace(" ", "_")
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
            error_msg = error_messages.get(probe_error, "Failed to process uploaded file")
            logger.error(f"File probe failed for {filename}: {probe_error}")

            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({
                "error": error_msg,
                "code": probe_error
            }), 400

        # Prepare processing_info with user choices and file metadata
        processing_info = {
            "file_metadata": file_metadata,
            "user_choices": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "auto_create_video": auto_create_video,
                "whisper_model": whisper_model,
                "translation_service": translation_service,
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
            kwargs=subtitle_flags,
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

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500
