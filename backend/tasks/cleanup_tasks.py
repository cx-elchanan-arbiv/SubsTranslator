"""
Cleanup tasks for SubsTranslator
Handles periodic file cleanup and maintenance
"""

import os
import time
from pathlib import Path

from celery_worker import celery_app
from config import get_config
from logging_config import get_logger

# Configuration
config = get_config()
logger = get_logger(__name__)

UPLOAD_FOLDER = config.UPLOAD_FOLDER
DOWNLOADS_FOLDER = config.DOWNLOADS_FOLDER
MAX_FILE_AGE = config.MAX_FILE_AGE


@celery_app.task(bind=True)
def cleanup_files_task(self):
    """Periodically cleans up old files from upload and download folders."""
    self.update_state(state="PROGRESS", meta={"status": "Starting cleanup..."})
    now = time.time()
    cleaned_files = []

    for folder in [UPLOAD_FOLDER, DOWNLOADS_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > MAX_FILE_AGE:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                    logger.info(f"Removed old file: {filename}")

    return {"status": "Cleanup complete", "cleaned_files": cleaned_files}


@celery_app.task(bind=True)
def cleanup_old_files_task(self, days=14):
    """Auto-cleanup old downloads and ensure fast_work is clean"""
    cutoff = time.time() - days * 86400
    removed_count = 0
    total_size_mb = 0

    # Clean downloads folder
    downloads_path = Path(config.DOWNLOADS_FOLDER)
    if downloads_path.exists():
        for file_path in downloads_path.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink(missing_ok=True)
                removed_count += 1
                total_size_mb += size_mb

    # Ensure fast_work is completely clean
    fast_work_path = Path(config.FAST_WORK_DIR)
    if fast_work_path.exists():
        for leftover in fast_work_path.glob("*"):
            if leftover.is_file():
                logger.warning(f"Removing leftover temp file: {leftover}")
                leftover.unlink(missing_ok=True)

    logger.info(
        "Cleanup completed",
        removed_files=removed_count,
        freed_space_mb=round(total_size_mb, 1),
        retention_days=days,
    )

    return {
        "status": "SUCCESS",
        "removed_files": removed_count,
        "freed_space_mb": round(total_size_mb, 1),
    }
