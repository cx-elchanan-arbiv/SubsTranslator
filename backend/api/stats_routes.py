"""
Statistics routes for SubsTranslator
Handles task statistics, daily summaries, model performance, and cost tracking
"""

import os
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

from logging_config import get_logger
from services.stats_service import (
    get_cost_breakdown,
    get_daily_summary,
    get_model_performance,
    get_stats_by_task_id,
    get_stats_count_from_file,
    get_stats_file_path,
    get_stats_file_size,
    is_stats_service_available,
)
from utils.file_utils import safe_int

# Configuration
logger = get_logger(__name__)

# Create blueprint
stats_bp = Blueprint("stats", __name__, url_prefix="/api/stats")


@stats_bp.route("/task/<task_id>", methods=["GET"])
def get_task_stats(task_id):
    """Get statistics for a specific task."""
    try:
        stats = get_stats_by_task_id(task_id)
        if stats:
            return jsonify(stats), 200
        else:
            return jsonify({"error": "Stats not found"}), 404
    except Exception as e:
        logger.error(f"Error fetching task stats: {e}")
        return jsonify({"error": str(e)}), 500


@stats_bp.route("/daily", methods=["GET"])
def get_stats_daily():
    """
    Get daily summary of processing statistics.

    Query params:
        - date: Date in YYYY-MM-DD format (optional, defaults to today)

    Example: /api/stats/daily?date=2025-01-19
    """
    try:
        date_str = request.args.get("date")

        # Validate date format if provided
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        summary = get_daily_summary(date_str)
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"Error fetching daily stats: {e}")
        return jsonify({"error": str(e)}), 500


@stats_bp.route("/model/<model_name>", methods=["GET"])
def get_stats_model(model_name):
    """
    Get performance statistics for a specific Whisper model.

    Path params:
        - model_name: Model name (base, medium, large)

    Query params:
        - days: Number of days to analyze (optional, default: 7)

    Example: /api/stats/model/base?days=30
    """
    try:
        # Validate model name
        valid_models = ["base", "medium", "large"]
        if model_name not in valid_models:
            return (
                jsonify(
                    {
                        "error": f"Invalid model name. Must be one of: {', '.join(valid_models)}"
                    }
                ),
                400,
            )

        days, days_error = safe_int(request.args.get("days"), 7, 1, 365)
        if days_error:
            return jsonify({"error": f"Invalid 'days' parameter: {days_error}"}), 400
        performance = get_model_performance(model_name, days)
        return jsonify(performance), 200
    except Exception as e:
        logger.error(f"Error fetching model stats: {e}")
        return jsonify({"error": str(e)}), 500


@stats_bp.route("/costs", methods=["GET"])
def get_stats_costs():
    """
    Get cost breakdown by translation service.

    Query params:
        - date: Starting date in YYYY-MM-DD format (optional, defaults to today)
        - days: Number of days to analyze (optional, default: 7)

    Example: /api/stats/costs?date=2025-01-19&days=30
    """
    try:
        date_str = request.args.get("date")
        days, days_error = safe_int(request.args.get("days"), 7, 1, 365)
        if days_error:
            return jsonify({"error": f"Invalid 'days' parameter: {days_error}"}), 400

        # Validate date format if provided
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        breakdown = get_cost_breakdown(date_str, days)
        return jsonify(breakdown), 200
    except Exception as e:
        logger.error(f"Error fetching cost stats: {e}")
        return jsonify({"error": str(e)}), 500


@stats_bp.route("/health", methods=["GET"])
def get_stats_health():
    """Check if statistics service is available."""
    try:
        available = is_stats_service_available()
        return jsonify(
            {
                "stats_service_available": available,
                "message": (
                    "Stats service is operational"
                    if available
                    else "Stats service unavailable (Redis connection issue)"
                ),
            }
        ), (200 if available else 503)
    except Exception as e:
        logger.error(f"Error checking stats health: {e}")
        return jsonify({"stats_service_available": False, "message": str(e)}), 503


@stats_bp.route("/download", methods=["GET"])
def download_stats_file():
    """
    Download the complete stats JSONL file.

    Returns the raw JSONL file for offline analysis.
    Each line is a JSON object representing one video processing task.

    Example usage:
        curl http://localhost:8081/api/stats/download > my_stats.jsonl

    Then analyze with pandas:
        import pandas as pd
        df = pd.read_json('my_stats.jsonl', lines=True)
    """
    try:
        stats_file = get_stats_file_path()

        if not os.path.exists(stats_file):
            return (
                jsonify(
                    {
                        "error": "Stats file not found",
                        "message": "No statistics have been recorded yet",
                    }
                ),
                404,
            )

        # Get file info
        file_size = get_stats_file_size()
        entry_count = get_stats_count_from_file()

        logger.info(
            f"📥 Downloading stats file: {entry_count} entries, {file_size / 1024:.1f}KB"
        )

        return send_file(
            stats_file,
            mimetype="application/x-ndjson",
            as_attachment=True,
            download_name="video_stats.jsonl",
        )

    except Exception as e:
        logger.error(f"Error downloading stats file: {e}")
        return jsonify({"error": str(e)}), 500


@stats_bp.route("/info", methods=["GET"])
def get_stats_file_info():
    """
    Get information about the stats file.

    Returns:
        - File size
        - Number of entries
        - File path
        - Whether file exists
    """
    try:
        stats_file = get_stats_file_path()
        exists = os.path.exists(stats_file)

        info = {
            "file_exists": exists,
            "file_path": stats_file if exists else None,
            "file_size_bytes": get_stats_file_size() if exists else 0,
            "file_size_kb": round(get_stats_file_size() / 1024, 2) if exists else 0,
            "entry_count": get_stats_count_from_file() if exists else 0,
        }

        return jsonify(info), 200

    except Exception as e:
        logger.error(f"Error getting stats file info: {e}")
        return jsonify({"error": str(e)}), 500
