"""
YouTube download service for SubsTranslator
Handles video downloads from YouTube with metadata extraction
"""

import os
import shutil
import time

import yt_dlp

from config import get_config
from logging_config import get_logger
from performance_monitor import (
    log_download_performance,
    log_move_performance,
    performance_monitor,
)
from utils.file_utils import clean_filename, parse_time_to_seconds
from ytdlp_hooks import create_clean_progress_hook

# Configuration
config = get_config()
logger = get_logger(__name__)

DOWNLOADS_FOLDER = config.DOWNLOADS_FOLDER


def download_youtube_video(
    url, quality="medium", progress_callback=None, start_time=None, end_time=None
):
    """
    Download video from YouTube and extract comprehensive metadata.

    Args:
        url: Video URL
        quality: Video quality
        progress_callback: Optional callback for progress updates
        start_time: Optional start time for partial download (format: HH:MM:SS, MM:SS, or SS)
        end_time: Optional end time for partial download (format: HH:MM:SS, MM:SS, or SS)
    """
    try:
        # FAKE mode: return deterministic metadata and copy a local test video
        if config.USE_FAKE_YTDLP or (isinstance(url, str) and "mocked_video" in url):
            fake_src = os.path.join(config.FAKE_ASSETS_DIR, config.FAKE_VIDEO_SOURCE)
            os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
            base_name = clean_filename(os.path.splitext(os.path.basename(fake_src))[0])
            dst_path = os.path.join(DOWNLOADS_FOLDER, f"{base_name}.mp4")
            shutil.copy2(fake_src, dst_path)
            metadata = {
                "title": base_name,
                "duration": 5,
                "duration_string": "00:00:05",
                "view_count": 1,
                "upload_date": "20250101",
                "uploader": "Mocked",
                "thumbnail": "",
                "description": "Mocked video for tests",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "filesize": (
                    os.path.getsize(dst_path) if os.path.exists(dst_path) else 0
                ),
                "url": url,
            }
            logger.info(f"🧪 FAKE download used. Copied {fake_src} -> {dst_path}")
            return dst_path, metadata

        # Phase A: Use fast workspace for I/O operations
        work_dir = config.FAST_WORK_DIR
        final_dir = DOWNLOADS_FOLDER
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        start_time_ts = time.time()

        ydl_opts = {
            "format": config.YTDLP_OPTIMIZED_FORMAT,  # Phase A: Optimized for remux-only
            "outtmpl": f"{work_dir}/%(title).120B.%(ext)s",  # Phase A: Use fast workspace
            "extract_flat": False,
            "noplaylist": True,
            "restrict_filenames": True,
            "retries": config.YTDLP_RETRIES,  # Phase A: Reduced retries
            "fragment_retries": config.YTDLP_FRAGMENT_RETRIES,  # Phase A: Reduced retries
            "socket_timeout": config.YTDLP_SOCKET_TIMEOUT,  # Phase A: Faster timeout
            "continue_dl": True,
            "merge_output_format": "mp4",
            # Log cleanup: Reduce yt-dlp verbosity
            "quiet": not config.DEBUG,  # Only show yt-dlp logs in DEBUG mode
            "no_warnings": not config.DEBUG,  # Suppress warnings unless debugging
            # Use android_vr client so YouTube returns full-resolution formats
            # (web/android are SABR-capped to 360p). See config.YTDLP_EXTRACTOR_ARGS
            "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            # Phase A: Only faststart, no re-encoding
            "postprocessor_args": {
                "ffmpeg": [
                    "-movflags",
                    "+faststart",
                ]  # Removed all codec args for remux-only
            },
        }

        # Add time range support via postprocessor (trim after download)
        if start_time and end_time:
            try:
                start_seconds = parse_time_to_seconds(start_time)
                end_seconds = parse_time_to_seconds(end_time)

                # Use ffmpeg to trim after download (most reliable approach)
                ydl_opts["postprocessor_args"]["ffmpeg"].extend(
                    [
                        "-ss",
                        str(start_seconds),
                        "-to",
                        str(end_seconds),
                        "-c",
                        "copy",  # Stream copy (no re-encoding) for speed
                    ]
                )
                logger.info(
                    f"🎯 Time range: {start_time} - {end_time} (will trim after download)"
                )
            except ValueError as e:
                logger.warning(
                    f"⚠️ Invalid time format: {e}. Downloading full video instead."
                )
                # Continue without time range if parsing fails

        # Use clean progress hooks from ytdlp_hooks

        if progress_callback:

            def progress_wrapper(d):
                if d["status"] == "downloading":
                    if "total_bytes" in d and d["total_bytes"]:
                        percent = (d["downloaded_bytes"] / d["total_bytes"]) * 100
                    elif "total_bytes_estimate" in d and d["total_bytes_estimate"]:
                        percent = (
                            d["downloaded_bytes"] / d["total_bytes_estimate"]
                        ) * 100
                    else:
                        percent = 50  # Fallback
                    progress_callback(min(95, max(15, percent)))

            # Use both the clean progress hook and our progress callback
            ydl_opts["progress_hooks"] = [
                create_clean_progress_hook(),
                progress_wrapper,
            ]
        else:
            # Use only the clean progress hook
            ydl_opts["progress_hooks"] = [create_clean_progress_hook()]

        logger.info("🎯 Starting optimized download")
        if config.DEBUG:
            logger.debug(f"📂 Working in fast storage: {work_dir}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if config.DEBUG:
                logger.debug(f"📡 Extracting info from URL: {url}")
            info = ydl.extract_info(url, download=True)
            work_filename = ydl.prepare_filename(info)

            if os.path.exists(work_filename):
                title = info.get("title", "video")
                cleaned_title = clean_filename(title)
                work_dir_path = os.path.dirname(work_filename)
                ext = os.path.splitext(work_filename)[1]
                cleaned_work_filename = os.path.join(
                    work_dir_path, f"{cleaned_title}{ext}"
                )

                if work_filename != cleaned_work_filename:
                    os.rename(work_filename, cleaned_work_filename)
                    work_filename = cleaned_work_filename
                    logger.info(f"🔧 Renamed file: {os.path.basename(work_filename)}")

                # Phase A: Move from fast workspace to final directory
                final_filename = work_filename.replace(work_dir, final_dir)
                move_start = time.time()

                # Use copy2 + remove instead of move to handle cross-device links
                try:
                    # Ensure directory exists with proper permissions
                    final_dir_path = os.path.dirname(final_filename)
                    os.makedirs(final_dir_path, mode=0o755, exist_ok=True)

                    # Copy file and then remove original
                    shutil.copy2(work_filename, final_filename)
                    os.chmod(final_filename, 0o644)  # Ensure file permissions
                    os.remove(work_filename)
                except (OSError, PermissionError) as e:
                    logger.warning(f"Copy failed ({e}), trying fallback move")
                    try:
                        # Fallback: try regular move
                        shutil.move(work_filename, final_filename)
                    except (OSError, PermissionError) as move_error:
                        logger.error(
                            f"Move also failed ({move_error}), trying with elevated permissions"
                        )
                        # Final fallback: try to fix permissions and move again
                        try:
                            os.chmod(os.path.dirname(final_filename), 0o755)
                            shutil.move(work_filename, final_filename)
                        except Exception as final_error:
                            raise Exception(
                                f"Failed to move file after all attempts: {final_error}"
                            )

                move_duration = time.time() - move_start

                # Phase A: Enhanced performance monitoring
                file_size_mb = os.path.getsize(final_filename) / (1024 * 1024)
                total_duration = time.time() - start_time_ts

                # Log download performance
                log_download_performance(
                    file_size_mb, total_duration, "download_and_merge"
                )

                # Log file move performance
                log_move_performance(
                    file_size_mb, move_duration, "fast_work", "downloads"
                )

                # Check system resources
                performance_monitor.check_system_resources()

                filename = final_filename
            else:
                filename = work_filename

            logger.info(f"✅ Phase A download completed: {os.path.basename(filename)}")

        metadata = {
            "title": info.get("title", "Unknown Title"),
            "duration": info.get("duration", 0),
            "duration_string": info.get("duration_string", "00:00"),
            "view_count": info.get("view_count", 0),
            "upload_date": info.get("upload_date", ""),
            "uploader": info.get("uploader", "Unknown"),
            "thumbnail": info.get("thumbnail", ""),
            "description": (
                info.get("description", "")[:200] + "..."
                if info.get("description", "")
                else ""
            ),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "fps": info.get("fps", 0),
            "filesize": info.get("filesize", 0),
            "url": url,
        }

        logger.info(
            f"✅ Downloaded: {metadata['title']} ({metadata['duration_string']}, {metadata['view_count']:,} views)"
        )
        return filename, metadata
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        # Convert to structured exception with proper error codes
        from core.exceptions import handle_youtube_error

        raise handle_youtube_error(e, url)


def download_youtube_video_with_progress(
    url, quality="medium", progress_manager=None, start_time=None, end_time=None
):
    """
    Download video from YouTube with real-time progress updates.

    Args:
        url: Video URL
        quality: Video quality (high, medium, low)
        progress_manager: Optional progress tracking manager
        start_time: Optional start time for partial download (format: HH:MM:SS, MM:SS, or SS)
        end_time: Optional end time for partial download (format: HH:MM:SS, MM:SS, or SS)
    """
    try:
        # FAKE mode shortcut
        if config.USE_FAKE_YTDLP or (isinstance(url, str) and "mocked_video" in url):
            fake_src = os.path.join(config.FAKE_ASSETS_DIR, config.FAKE_VIDEO_SOURCE)
            os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
            base_name = clean_filename(os.path.splitext(os.path.basename(fake_src))[0])
            dst_path = os.path.join(DOWNLOADS_FOLDER, f"{base_name}.mp4")
            shutil.copy2(fake_src, dst_path)
            metadata = {
                "title": base_name,
                "duration": 5,
                "duration_string": "00:00:05",
                "view_count": 1,
                "upload_date": "20250101",
                "uploader": "Mocked",
                "thumbnail": "",
                "description": "Mocked video for tests",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "filesize": (
                    os.path.getsize(dst_path) if os.path.exists(dst_path) else 0
                ),
                "url": url,
            }
            if progress_manager:
                progress_manager.set_step_progress(0, 95, "Preparing metadata...")
                progress_manager.log(
                    f"📊 Video info: {metadata['title']} ({metadata['duration_string']})"
                )
                progress_manager.set_step_progress(0, 99, "Finalizing download...")
            logger.info(f"🧪 FAKE download used. Copied {fake_src} -> {dst_path}")
            return dst_path, metadata

        # Phase A: Use fast workspace for download-only tasks too
        work_dir = config.FAST_WORK_DIR
        final_dir = DOWNLOADS_FOLDER
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        ydl_opts = {
            "format": config.YTDLP_OPTIMIZED_FORMAT,  # Phase A: Use optimized format
            "outtmpl": f"{work_dir}/%(title)s.%(ext)s",  # Phase A: Use fast workspace
            "extract_flat": False,
            "noplaylist": True,  # Force single video download only
            "socket_timeout": config.YTDLP_SOCKET_TIMEOUT,
            "fragment_retries": config.YTDLP_FRAGMENT_RETRIES,
            "retries": config.YTDLP_RETRIES,
            "cache_dir": config.YTDLP_CACHE_DIR,
            "merge_output_format": config.YTDLP_MERGE_OUTPUT_FORMAT,
            # Log cleanup: Reduce yt-dlp verbosity
            "quiet": not config.DEBUG,
            "no_warnings": not config.DEBUG,
            # Use android_vr client so YouTube returns full-resolution formats
            # (web/android are SABR-capped to 360p). See config.YTDLP_EXTRACTOR_ARGS
            "extractor_args": config.YTDLP_EXTRACTOR_ARGS,
            # Phase A: Only faststart, no re-encoding
            "postprocessor_args": {
                "ffmpeg": [
                    "-movflags",
                    "+faststart",
                ]  # Removed all codec args for remux-only
            },
            "restrict_filenames": config.YTDLP_RESTRICT_FILENAMES,
            "continue_dl": config.YTDLP_CONTINUE_DL,
        }

        # Add time range support via postprocessor (more reliable than download_ranges)
        if start_time and end_time:
            try:
                start_seconds = parse_time_to_seconds(start_time)
                end_seconds = parse_time_to_seconds(end_time)

                # Use ffmpeg to trim after download (most reliable approach)
                # -ss: start time, -to: end time, -c copy: no re-encoding (fast)
                ydl_opts["postprocessor_args"]["ffmpeg"].extend(
                    [
                        "-ss",
                        str(start_seconds),
                        "-to",
                        str(end_seconds),
                        "-c",
                        "copy",  # Stream copy (no re-encoding) for speed
                    ]
                )

                if progress_manager:
                    progress_manager.log(
                        f"🎯 Time range requested: {start_time} - {end_time}"
                    )
                    progress_manager.log("Will trim video after download using ffmpeg")
            except ValueError as e:
                if progress_manager:
                    progress_manager.log(
                        f"⚠️ Invalid time format: {e}. Downloading full video instead."
                    )
                # Continue without time range if parsing fails

        # Add real progress hooks.
        # HLS sources (e.g. Fox News, TED) download video and audio as SEPARATE
        # streams, each reporting 0->100%, and their size estimates fluctuate per
        # fragment. Mapping raw percent would make the bar jump backwards (audio
        # restarts at 0; estimates wobble). Keep it monotonic: never show less
        # than the max already reached during this download step.
        download_progress = {"max": 15, "streams": []}

        def bump(pct, msg):
            """Move the download step forward only. Single monotonic gate for
            BOTH the byte-progress hook and the post-download hardcoded steps,
            so the bar can never jump backwards across the whole download."""
            if not progress_manager:
                return
            pct = max(int(pct), download_progress["max"])
            download_progress["max"] = pct
            progress_manager.set_step_progress(0, pct, msg)

        def _stream_band(d):
            """Map the current yt-dlp stream to a sub-range of the download step.
            HLS/merged downloads fetch video then audio as separate streams. We
            key off stream ORDER (not codec): yt-dlp's vcodec/acodec are often
            absent for HLS audio in the hook, so the only reliable signal is the
            order in which distinct format_ids appear. 1st = video bulk (15-80),
            any later = audio tail (80-92). Every stream is registered so the
            ordinal stays correct. Returns (low, high)."""
            info = d.get("info_dict") or {}
            fid = info.get("format_id") or "single"
            seen = download_progress["streams"]
            if fid not in seen:
                seen.append(fid)
            return (15, 80) if seen.index(fid) == 0 else (80, 92)

        def progress_hook(d):
            if d["status"] == "downloading" and progress_manager:
                lo, hi = _stream_band(d)
                # Prefer HLS fragment count: it's the only stable signal for HLS
                # (byte totals are absent for audio and wobble for video). Fall
                # back to bytes for progressive downloads.
                frag_n = d.get("fragment_count")
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if frag_n:
                    raw = (d.get("fragment_index") or 0) / frag_n
                elif total:
                    raw = (d.get("downloaded_bytes") or 0) / total
                else:
                    raw = None
                if raw is not None:
                    percent = lo + max(0.0, min(1.0, raw)) * (
                        hi - lo
                    )  # 0..1 within stream
                else:
                    percent = download_progress["max"]  # keep current if unknown

                progress_msg = d.get("_percent_str", f"{percent:.1f}%")
                speed = d.get("_speed_str", "")
                eta = d.get("_eta_str", "")

                status_msg = f"Downloading: {progress_msg}"
                if speed:
                    status_msg += f" | {speed}"
                if eta:
                    status_msg += f" | Time remaining: {eta}"

                bump(percent, status_msg)  # monotonic clamp inside bump

            elif d["status"] == "finished" and progress_manager:
                # One stream finished — advance to ITS band end, not 95 (an audio
                # stream may still follow). The post-download code takes it to 95.
                _lo, hi = _stream_band(d)
                bump(hi, "Finishing stream...")

        ydl_opts["progress_hooks"] = [progress_hook]

        if progress_manager:
            progress_manager.log("🎯 Starting download with yt-dlp...")
            bump(15, "Configuring download...")

        logger.info(f"🎯 Starting download with options: {ydl_opts}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if progress_manager:
                bump(20, "Extracting video info...")
                progress_manager.log("📡 Extracting info from URL...")

            logger.info(f"📡 Extracting info from URL: {url}")
            info = ydl.extract_info(url, download=True)

            if progress_manager:
                bump(92, "Download complete, processing...")
                progress_manager.log("📥 Download complete, processing...")

            original_filename = ydl.prepare_filename(info)

            if os.path.exists(original_filename):
                title = info.get("title", "video")
                cleaned_title = clean_filename(title)
                work_dir_path = os.path.dirname(original_filename)
                ext = os.path.splitext(original_filename)[1]
                cleaned_work_filename = os.path.join(
                    work_dir_path, f"{cleaned_title}{ext}"
                )

                if original_filename != cleaned_work_filename:
                    os.rename(original_filename, cleaned_work_filename)
                    work_filename = cleaned_work_filename
                    logger.info(f"🔧 Renamed file: {os.path.basename(work_filename)}")
                else:
                    work_filename = original_filename

                # Phase A: Move from fast workspace to final directory
                final_filename = work_filename.replace(work_dir, final_dir)

                if progress_manager:
                    bump(93, "Moving to final directory...")

                try:
                    # Ensure directory exists with proper permissions
                    os.makedirs(
                        os.path.dirname(final_filename), mode=0o755, exist_ok=True
                    )
                    shutil.move(work_filename, final_filename)
                except (OSError, PermissionError) as e:
                    logger.error(f"Failed to move file: {e}")
                    # Try to fix permissions and retry
                    try:
                        os.chmod(os.path.dirname(final_filename), 0o755)
                        shutil.move(work_filename, final_filename)
                    except Exception as retry_error:
                        raise Exception(
                            f"Failed to move file after permission fix: {retry_error}"
                        )
                filename = final_filename

                if progress_manager:
                    progress_manager.log(
                        "🔧 Moved from fast storage to downloads folder..."
                    )
            else:
                filename = original_filename

            if progress_manager:
                bump(95, "Preparing metadata...")
                progress_manager.log(
                    f"📊 Video info: {info.get('title', 'Unknown')} ({info.get('duration_string', '00:00')})"
                )

            logger.info(f"✅ Phase A download completed: {os.path.basename(filename)}")

        metadata = {
            "title": info.get("title", "Unknown Title"),
            "duration": info.get("duration", 0),
            "duration_string": info.get("duration_string", "00:00"),
            "view_count": info.get("view_count", 0),
            "upload_date": info.get("upload_date", ""),
            "uploader": info.get("uploader", "Unknown"),
            "thumbnail": info.get("thumbnail", ""),
            "description": (
                info.get("description", "")[:200] + "..."
                if info.get("description", "")
                else ""
            ),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "fps": info.get("fps", 0),
            "filesize": info.get("filesize", 0),
            "url": url,
        }

        if progress_manager:
            bump(95, "Preparing metadata...")
            progress_manager.log(
                f"📊 Video info: {metadata['title']} ({metadata['duration_string']})"
            )

        logger.info(
            f"✅ Downloaded: {metadata['title']} ({metadata['duration_string']}, {metadata['view_count']:,} views)"
        )
        return filename, metadata

    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        # Convert to structured exception with proper error codes
        from core.exceptions import handle_youtube_error

        raise handle_youtube_error(e, url)
