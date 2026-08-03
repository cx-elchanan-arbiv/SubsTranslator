"""
Status routes for SubsTranslator API v1.
Handles task status checking and progress tracking.
"""

from celery.result import AsyncResult
from flask import Blueprint, jsonify

from config import get_config
from logging_config import get_logger
from tasks import process_video_task

# Configuration
config = get_config()
logger = get_logger(__name__)

# Create blueprint
status_bp = Blueprint("status", __name__)


@status_bp.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Get the status of a background task with unified schema."""
    task_result = AsyncResult(task_id, app=process_video_task.app)

    status = task_result.state
    result = None
    error_info = None
    progress_info = {"overall_percent": 0, "steps": []}
    video_metadata = None
    user_choices = {}
    initial_request = {}
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
        "result": result if status == "SUCCESS" and not error_info else None,
        "user_choices": user_choices,
        "initial_request": initial_request,
        "error": error_info,
        **({"logs": logs_tail} if logs_tail is not None else {}),
    }

    return jsonify(response)
