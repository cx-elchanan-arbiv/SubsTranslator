"""
Watermark routes for SubsTranslator API v1.
Handles watermark logo management.
"""

import os

from flask import Blueprint, jsonify, request, session

from config import get_config
from logging_config import get_logger
from logo_manager import LogoManager

# Configuration
config = get_config()
logger = get_logger(__name__)

# Initialize logo manager
logo_manager = LogoManager(config.ASSETS_FOLDER)

# Create blueprint
watermark_bp = Blueprint("watermark", __name__)


@watermark_bp.route("/clear-watermark-logo", methods=["POST"])
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


@watermark_bp.route("/cleanup-logos", methods=["POST"])
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
