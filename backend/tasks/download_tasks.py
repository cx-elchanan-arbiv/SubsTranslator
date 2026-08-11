"""
Download tasks for SubsTranslator
Handles YouTube video downloads and related processing
"""

import os
import shutil
import time

import yt_dlp

from celery_worker import celery_app
from config import get_config
from logging_config import get_logger
from services.subtitle_pipeline import parse_subtitle_position, resolve_flags
from services.video_processing_service import verify_and_convert_video_format
from services.youtube_service import (
    download_youtube_video,
    download_youtube_video_with_progress,
)
from utils.file_utils import clean_filename

from .processing_tasks import process_video_task
from .progress_manager import ProgressManager

# Configuration
config = get_config()
logger = get_logger(__name__)

DOWNLOADS_FOLDER = config.DOWNLOADS_FOLDER


@celery_app.task(bind=True, name="download_and_process_youtube_task")
def download_and_process_youtube_task(
    self,
    url,
    source_lang,
    target_lang,
    auto_create_video,
    whisper_model,
    translation_service="google",
    watermark_config=None,
    start_time=None,
    end_time=None,
    spotting_v2=True,
    translation_v2=True,
    translation_style="clean",
    render_v2=True,
    subtitle_position="bottom",
):
    """
    Downloads a video from YouTube and then triggers the main processing task.

    Args:
        start_time: Optional start time for partial video processing (format: HH:MM:SS, MM:SS, or SS)
        end_time: Optional end time for partial video processing (format: HH:MM:SS, MM:SS, or SS)
        spotting_v2, translation_v2, translation_style, render_v2: the user-facing
            subtitle-quality toggles. This task only forwards them to
            ``process_video_task`` — it never acts on them itself. They must be forwarded
            explicitly: a flag dropped here would silently give the user the legacy
            pipeline for every YouTube job. See :mod:`services.subtitle_pipeline`.
        subtitle_position: burned-in subtitle placement, forwarded unchanged after
            validation. Defaults to the historical bottom-centre placement.
    """
    # One resolved copy of the four toggles: forwarded to process_video_task AND
    # reported back as user choices, so a YouTube job can be attributed to the
    # settings that produced it exactly like an upload can.
    subtitle_flags = resolve_flags(
        spotting_v2=spotting_v2,
        translation_v2=translation_v2,
        translation_style=translation_style,
        render_v2=render_v2,
    )
    subtitle_position = parse_subtitle_position(subtitle_position)

    steps_config = [
        {"label": "Downloading", "weight": 0.1},
        {"label": "Audio Processing", "weight": 0.1},
        {"label": "Loading AI Model", "weight": 0.05},
        {"label": "AI Transcription", "weight": 0.3},
        {"label": "Translation", "weight": 0.2},
        {"label": "SRT Creation", "weight": 0.1},
        {"label": "Video Creation", "weight": 0.15 if auto_create_video else 0},
    ]
    progress_manager = ProgressManager(self, steps_config)

    try:
        progress_manager.set_step_status(0, "in_progress")
        progress_manager.log(f"Starting download from YouTube: {url}", step_index=0)

        # Early metadata extraction - extract info without downloading
        progress_manager.log("Extracting initial video information...", step_index=0)
        progress_manager.set_step_progress(0, 5)

        try:
            temp_ydl_opts = {
                "noplaylist": True,
                "quiet": True,
                "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            }
            with yt_dlp.YoutubeDL(temp_ydl_opts) as temp_ydl:
                info_dict = temp_ydl.extract_info(url, download=False)

                # Check if this is a playlist URL
                if "list=" in url and "&index=" in url:
                    progress_manager.log(
                        "Playlist detected - downloading only the selected video (not the entire list)",
                        step_index=0,
                    )
                elif "list=" in url:
                    progress_manager.log(
                        "Playlist detected - downloading only one video (not the entire list)",
                        step_index=0,
                    )

                # Build metadata object early
                video_metadata = {
                    "title": info_dict.get("title", "Unknown Title"),
                    "duration": info_dict.get("duration", 0),
                    "duration_string": info_dict.get("duration_string", "00:00"),
                    "view_count": info_dict.get("view_count", 0),
                    "upload_date": info_dict.get("upload_date", ""),
                    "uploader": info_dict.get("uploader", "Unknown"),
                    "thumbnail": info_dict.get("thumbnail", ""),
                    "description": (
                        info_dict.get("description", "")[:200] + "..."
                        if info_dict.get("description", "")
                        else ""
                    ),
                    "width": info_dict.get("width", 0),
                    "height": info_dict.get("height", 0),
                    "fps": info_dict.get("fps", 0),
                    "filesize": info_dict.get("filesize", 0),
                    "url": url,
                }

                # Send early metadata update to frontend
                progress_manager.log(
                    f"Found: {video_metadata['title']} ({video_metadata['duration_string']})",
                    step_index=0,
                )

                # Set metadata in progress manager to preserve it
                progress_manager.set_metadata(
                    video_metadata=video_metadata,
                    user_choices={
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "auto_create_video": auto_create_video,
                        "whisper_model": whisper_model,
                        "url": url,
                        "translation_service": translation_service,
                        "subtitle_position": subtitle_position,
                        **subtitle_flags,
                    },
                )

                progress_manager.set_step_progress(0, 10)

                progress_manager.log("Starting download...", step_index=0)

        except Exception as e:
            progress_manager.log(
                f"Could not extract early info: {str(e)}", step_index=0
            )
            video_metadata = None

        download_start_time = time.time()

        # Define progress callback for download
        def download_progress_callback(percent):
            progress_manager.set_step_progress(
                0, int(percent), f"Downloading: {percent:.1f}%"
            )

        # Log time range if provided
        if start_time and end_time:
            progress_manager.log(
                f"Requested time range: {start_time} - {end_time}", step_index=0
            )

        # Now do the actual download
        if video_metadata:
            # If we have metadata, just download without re-extracting
            video_path, _ = download_youtube_video(
                url,
                "high",
                progress_callback=download_progress_callback,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            # Fallback: download and extract metadata together
            video_path, video_metadata = download_youtube_video(
                url,
                "high",
                progress_callback=download_progress_callback,
                start_time=start_time,
                end_time=end_time,
            )
        download_time = f"{time.time() - download_start_time:.1f}"
        progress_manager.log("Finalizing download...", step_index=0)
        progress_manager.set_step_progress(0, 99)
        time.sleep(1)
        progress_manager.complete_step(0)

        initial_timing = {"download_video": download_time}

        processing_info = {
            "video_metadata": video_metadata,
            "user_choices": {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "auto_create_video": auto_create_video,
                "whisper_model": whisper_model,
                "url": url,
                "translation_service": translation_service,
                "subtitle_position": subtitle_position,
                **subtitle_flags,
                **(
                    {"start_time": start_time, "end_time": end_time}
                    if start_time and end_time
                    else {}
                ),
            },
        }

        progress_manager.set_step_status(0, "completed")
        progress_manager.log("Chaining to processing task...", step_index=1)
        task = process_video_task.apply_async(
            args=[
                video_path,
                source_lang,
                target_lang,
                auto_create_video,
                whisper_model,
                translation_service,
                watermark_config,
                initial_timing,
                processing_info,
            ],
            kwargs={**subtitle_flags, "subtitle_position": subtitle_position},
            queue="processing",
        )

        return {
            "task_id": task.id,
            "status": "PROCESSING",
            "video_metadata": video_metadata,
            "user_choices": processing_info["user_choices"],
        }
    except Exception as e:
        import traceback

        error_msg = f"YouTube Task failed: {str(e)}"
        traceback_msg = traceback.format_exc()
        progress_manager.log(error_msg)
        progress_manager.set_step_error(0, str(e))
        return {"status": "FAILURE", "error": error_msg, "traceback": traceback_msg}


@celery_app.task(bind=True)
def download_highest_quality_video_task(self, url):
    """Download YouTube video at highest quality (best video + best audio merged)."""
    steps_config = [{"label": "Downloading", "weight": 1.0}]
    progress_manager = ProgressManager(self, steps_config)

    try:
        progress_manager.set_step_status(0, "in_progress")
        progress_manager.log(
            f"Downloading highest quality video from: {url}", step_index=0
        )

        try:
            temp_ydl_opts = {
                "noplaylist": True,
                "quiet": True,
                "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            }
            with yt_dlp.YoutubeDL(temp_ydl_opts) as temp_ydl:
                info_dict = temp_ydl.extract_info(url, download=False)
                title = info_dict.get("title", "video")
                video_id = info_dict.get("id", "unknown")
        except Exception as e:
            progress_manager.log(
                f"Could not extract info, using defaults: {str(e)}", step_index=0
            )
            title = "video"
            video_id = "unknown"

        safe_title = clean_filename(title)
        safe_filename = f"{safe_title}_HQ_{video_id}"

        preferred_chain = [
            "bv*[height<=1080]+ba/bv*+ba",
            "bestvideo[height<=1080]+bestaudio",
            "18/22/136+140/137+140/298+140",
            "best",
        ]

        base_opts = {
            "outtmpl": os.path.join(DOWNLOADS_FOLDER, f"{safe_filename}.%(ext)s"),
            "noplaylist": True,
            "socket_timeout": 60,
            "fragment_retries": 10,
            "retries": 10,
            "cache_dir": "/tmp/yt-dlp",
            "merge_output_format": "mp4",
            "restrict_filenames": True,
            "continue_dl": True,
            "hls_prefer_native": True,
            # android_vr client so YouTube returns full-res formats (not SABR 360p)
            "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            "format_sort": ["res:1080", "fps", "codec:avc1:m4a", "ext:mp4"],
            "compat_opts": ["format-sort-force"],
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
            ],
            "progress_hooks": [
                lambda d: progress_manager.set_step_progress(
                    0,
                    (
                        (d["downloaded_bytes"] / d["total_bytes"]) * 100
                        if d["status"] == "downloading" and d.get("total_bytes")
                        else (99 if d["status"] == "finished" else 0)
                    ),
                    message=d["_percent_str"],
                )
            ],
        }

        last_err = None
        info_dict = None

        for format_string in preferred_chain:
            try:
                progress_manager.log(f"Trying format: {format_string}", step_index=0)
                ydl_opts = dict(base_opts, format=format_string)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)

                    # Log cleanup: Only show format details in DEBUG mode
                    if config.DEBUG:
                        chosen_fmt = info_dict.get("requested_formats") or [info_dict]
                        for f in chosen_fmt:
                            logger.debug(
                                f"Chosen format: itag={f.get('format_id')} vcodec={f.get('vcodec')} acodec={f.get('acodec')} proto={f.get('protocol')} ext={f.get('ext')} res={f.get('height')}p"
                            )

                    break

            except Exception as e:
                progress_manager.log(
                    f"Format '{format_string}' failed: {str(e)}", step_index=0
                )
                logger.warning(f"Format '{format_string}' failed: {e}")
                last_err = e
                continue

        if info_dict is None:
            raise RuntimeError(f"All format attempts failed. Last error: {last_err}")

        final_filename = os.path.join(DOWNLOADS_FOLDER, f"{safe_filename}.mp4")

        if not os.path.exists(final_filename):
            download_dir = DOWNLOADS_FOLDER
            possible_files = [
                f
                for f in os.listdir(download_dir)
                if safe_filename in f and f.endswith(".mp4")
            ]

            if not possible_files:
                title_part = safe_title[:50] if len(safe_title) > 50 else safe_title
                possible_files = [
                    f
                    for f in os.listdir(download_dir)
                    if title_part in f and f.endswith(".mp4")
                ]

            if not possible_files:
                all_mp4s = [f for f in os.listdir(download_dir) if f.endswith(".mp4")]
                if all_mp4s:
                    all_mp4s.sort(
                        key=lambda x: os.path.getmtime(os.path.join(download_dir, x)),
                        reverse=True,
                    )
                    possible_files = [all_mp4s[0]]
                    logger.info(f"Using newest file: {all_mp4s[0]}")

            if possible_files:
                actual_file = os.path.join(download_dir, possible_files[0])
                if actual_file != final_filename:
                    os.rename(actual_file, final_filename)
                    logger.info(
                        f"Renamed {possible_files[0]} to {os.path.basename(final_filename)}"
                    )
            else:
                raise Exception(
                    f"Downloaded file not found. Expected: {os.path.basename(final_filename)}"
                )

        if (
            not os.path.exists(final_filename)
            or os.path.getsize(final_filename) < 1_000_000
        ):
            raise RuntimeError("Downloaded file missing or too small (< 1MB)")

        progress_manager.log("Verifying video format...", step_index=0)
        compatible_filename = verify_and_convert_video_format(final_filename)
        file_size = os.path.getsize(compatible_filename) / (1024 * 1024)
        filename_only = os.path.basename(compatible_filename)
        progress_manager.complete_step(0)

        return {
            "status": "SUCCESS",
            "title": title,
            "filename": filename_only,
            "file_size_mb": round(file_size, 1),
            "duration": info_dict.get("duration", 0),
            "download_url": f"/download/{filename_only}",
            "message": f"Video downloaded successfully! Size: {round(file_size, 1)}MB (QuickTime compatible)",
            "file_path": compatible_filename,
            "format_verified": True,
        }

    except Exception as e:
        import traceback

        error_msg = f"Video download failed: {str(e)}"
        traceback_msg = traceback.format_exc()
        progress_manager.log(error_msg)
        progress_manager.set_step_error(0, str(e))
        return {"status": "FAILURE", "error": error_msg, "traceback": traceback_msg}


@celery_app.task(bind=True, name="download_youtube_only_task")
def download_youtube_only_task(
    self, url, quality="high", start_time=None, end_time=None
):
    """
    Enterprise-grade YouTube video download with robust error handling.

    Args:
        url: Video URL
        quality: Video quality (high, medium, low)
        start_time: Optional start time for partial download (format: HH:MM:SS, MM:SS, or SS)
        end_time: Optional end time for partial download (format: HH:MM:SS, MM:SS, or SS)
    """
    import os
    import sys

    # Add backend root to path (parent of tasks folder) for state_manager import
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    from services.metadata_service import MetadataExtractionError, metadata_service
    from state_manager import EnterpriseStateManager, StepStatus

    steps_config = [
        {"label": "Downloading Video", "weight": 1.0, "indeterminate": False}
    ]

    state_manager = EnterpriseStateManager(self, steps_config)

    try:
        # Step 1: Extract metadata
        state_manager.set_step_status(0, StepStatus.IN_PROGRESS)
        state_manager.log("Starting download-only from YouTube")
        state_manager.set_step_progress(0, 5, "Extracting initial info...")

        try:
            video_metadata, error_code = metadata_service.extract_metadata(url)

            if error_code:
                return state_manager.fail_task(
                    error_code,
                    f"Failed to extract metadata: {error_code}",
                    "Cannot access video information",
                    recoverable=False,
                )

            # Convert dataclass to dict for JSON serialization
            metadata_dict = {
                "title": video_metadata.title,
                "duration": video_metadata.duration,
                "duration_string": video_metadata.duration_string,
                "view_count": video_metadata.view_count,
                "upload_date": video_metadata.upload_date,
                "uploader": video_metadata.uploader,
                "thumbnail": video_metadata.thumbnail,
                "description": video_metadata.description,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "fps": video_metadata.fps,
                "filesize": video_metadata.filesize,
                "url": url,
                "quality_available": video_metadata.quality_available,
            }

            # Build initial_request with optional time range
            initial_request = {
                "url": url,
                "quality": quality,
                "type": "download_only",
            }
            if start_time and end_time:
                initial_request["start_time"] = start_time
                initial_request["end_time"] = end_time

            state_manager.set_metadata(
                video_metadata=metadata_dict,
                initial_request=initial_request,
            )

            state_manager.log(
                f"Found: {video_metadata.title} ({video_metadata.duration_string})"
            )
            state_manager.set_step_progress(0, 10, "Preparing download...")

        except MetadataExtractionError as e:
            return state_manager.fail_task(
                e.error_code, e.message, e.message, e.recoverable
            )
        except Exception as e:
            return state_manager.fail_task(
                "METADATA_EXTRACTION_ERROR",
                str(e),
                "Error extracting video information",
                recoverable=True,
            )

        # Step 2: Download video
        state_manager.set_step_progress(0, 15, "Starting download...")

        # Log time range if provided
        if start_time and end_time:
            state_manager.log(f"Downloading time range: {start_time} - {end_time}")

        download_start_time = time.time()
        video_path, _ = download_youtube_video_with_progress(
            url, quality, state_manager, start_time, end_time
        )

        download_time = f"{time.time() - download_start_time:.1f}"

        state_manager.log("Download completed successfully!", step_index=0)
        state_manager.set_step_progress(0, 100, "Download completed!")
        state_manager.complete_step(0)

        # Get just the filename for download
        filename_only = os.path.basename(video_path)

        # Copy file to downloads folder so it can be served (if not already there)
        from config import get_config

        config = get_config()
        downloads_path = os.path.join(config.DOWNLOADS_FOLDER, filename_only)

        # Only copy if the file is not already in the downloads folder
        if os.path.normpath(video_path) != os.path.normpath(downloads_path):
            try:
                # Ensure downloads directory has proper permissions
                os.makedirs(os.path.dirname(downloads_path), mode=0o755, exist_ok=True)
                shutil.copy2(video_path, downloads_path)
            except (OSError, PermissionError) as e:
                logger.error(f"Failed to copy video to downloads: {e}")
                # Try to fix permissions and retry
                try:
                    os.chmod(os.path.dirname(downloads_path), 0o755)
                    shutil.copy2(video_path, downloads_path)
                except Exception as retry_error:
                    raise Exception(
                        f"Failed to copy video after permission fix: {retry_error}"
                    )
        else:
            # File is already in downloads folder, no need to copy
            pass

        # Return success with download link
        return {
            "status": "SUCCESS",
            "message": "Video downloaded successfully",
            "filename": filename_only,
            "title": video_metadata.title,
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            "duration": video_metadata.duration,
            "download_url": f"/download/{filename_only}",
            "format_verified": True,
            "video_metadata": metadata_dict,
            "timing": {"download_video": download_time},
        }

    except Exception as e:
        import traceback

        error_str = str(e)
        traceback_msg = traceback.format_exc()

        # Provide specific error messages for common issues
        # Check for bot detection first (most specific)
        if (
            "Sign in to confirm" in error_str and "bot" in error_str.lower()
        ) or "bot detection blocked" in error_str.lower():
            user_message = "YouTube is blocking downloads from our server. Please download the video to your computer and upload it as a file."
            error_code = "YOUTUBE_BOT_DETECTION"
        elif "403" in error_str and "Forbidden" in error_str:
            user_message = "YouTube has blocked this video. Please try with a different video or try again later."
            error_code = "YOUTUBE_ACCESS_BLOCKED"
        elif "404" in error_str or "Video unavailable" in error_str:
            user_message = (
                "Video not found or unavailable. Please check the link and try again."
            )
            error_code = "VIDEO_NOT_FOUND"
        elif "Private video" in error_str or "private" in error_str.lower():
            user_message = "The video is private and cannot be downloaded."
            error_code = "PRIVATE_VIDEO"
        elif "age" in error_str.lower() or (
            "restricted" in error_str.lower() and "age" not in error_str.lower()
        ):
            user_message = "The video is age-restricted and cannot be downloaded without signing in."
            error_code = "AGE_RESTRICTED"
        else:
            user_message = "An error occurred during download. Please try with a different video or try again later."
            error_code = "DOWNLOAD_ERROR"

        return state_manager.fail_task(
            error_code,
            f"YouTube download-only failed: {error_str}",
            user_message,
            recoverable=True,
        )
