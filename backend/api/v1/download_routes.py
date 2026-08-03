"""
Download routes for SubsTranslator API v1.
Handles file downloads with security protections.
"""

import os

from flask import Blueprint, jsonify, request, send_file

from config import get_config
from logging_config import get_logger
from services.token_service import use_download_token

# Configuration
config = get_config()
logger = get_logger(__name__)

# Create blueprint
download_bp = Blueprint("download", __name__)


@download_bp.route("/download/<path:filename>", methods=["GET"])
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
