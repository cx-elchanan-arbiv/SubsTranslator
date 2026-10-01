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

    # fast_work leftovers: only files nobody has written to for FAST_WORK_MAX_AGE.
    # This used to delete EVERY file here, age regardless — safe only because the
    # worker runs one job at a time; with more it would delete downloads still in
    # progress. A running download keeps touching its files, and a failed one now
    # removes its own (youtube_service._remove_job_leftovers), so what is left for
    # this sweep is what a killed worker or a crash abandoned.
    fast_work_cutoff = time.time() - config.FAST_WORK_MAX_AGE
    fast_work_removed = 0
    fast_work_size_mb = 0
    fast_work_path = Path(config.FAST_WORK_DIR)
    if fast_work_path.exists():
        for leftover in fast_work_path.glob("*"):
            if leftover.is_file() and leftover.stat().st_mtime < fast_work_cutoff:
                logger.warning(f"Removing abandoned temp file: {leftover}")
                fast_work_size_mb += leftover.stat().st_size / (1024 * 1024)
                leftover.unlink(missing_ok=True)
                fast_work_removed += 1

    logger.info(
        "Cleanup completed",
        removed_files=removed_count,
        freed_space_mb=round(total_size_mb, 1),
        retention_days=days,
        fast_work_removed=fast_work_removed,
        fast_work_freed_mb=round(fast_work_size_mb, 1),
    )

    return {
        "status": "SUCCESS",
        "removed_files": removed_count,
        "freed_space_mb": round(total_size_mb, 1),
        "fast_work_removed": fast_work_removed,
        "fast_work_freed_mb": round(fast_work_size_mb, 1),
    }
