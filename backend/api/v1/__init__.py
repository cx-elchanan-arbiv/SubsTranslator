"""
API v1 - Versioned API routes for SubsTranslator

This module organizes all API endpoints under /api/v1/ prefix
for better maintainability and backwards compatibility.
"""

from flask import Blueprint

# Create main v1 blueprint
v1_bp = Blueprint("v1", __name__)

from .download_routes import download_bp
from .status_routes import status_bp

# Import and register all route blueprints
from .upload_routes import upload_bp
from .watermark_routes import watermark_bp
from .youtube_routes import youtube_bp

# Register sub-blueprints
v1_bp.register_blueprint(upload_bp)
v1_bp.register_blueprint(youtube_bp)
v1_bp.register_blueprint(status_bp)
v1_bp.register_blueprint(download_bp)
v1_bp.register_blueprint(watermark_bp)

__all__ = ["v1_bp"]
