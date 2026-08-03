"""
Tasks package for SubsTranslator
Re-exports all Celery tasks for backwards compatibility
"""

# Cleanup tasks
from .cleanup_tasks import (
    cleanup_files_task,
    cleanup_old_files_task,
)

# Download tasks
from .download_tasks import (
    download_and_process_youtube_task,
    download_highest_quality_video_task,
    download_youtube_only_task,
)

# Processing tasks
from .processing_tasks import (
    create_video_with_subtitles_from_segments,
    process_video_task,
)

# Progress manager (for internal use)
from .progress_manager import ProgressManager

__all__ = [
    # Cleanup
    "cleanup_files_task",
    "cleanup_old_files_task",
    # Processing
    "process_video_task",
    "create_video_with_subtitles_from_segments",
    # Download
    "download_and_process_youtube_task",
    "download_highest_quality_video_task",
    "download_youtube_only_task",
    # Utility
    "ProgressManager",
]
