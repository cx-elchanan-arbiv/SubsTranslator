"""
Subtitle service for creating and processing SRT files and video subtitles.
Handles SRT file generation, Hebrew/RTL text processing, and video subtitle embedding.
"""

import json
import os
import re
import shutil
import subprocess
import time  # Phase A: Added for performance monitoring
from collections.abc import Callable
from typing import Optional, Union

from config import get_config
from core.exceptions import FFmpegProcessError, FFmpegTimeoutError, FileNotFoundError
from logging_config import get_logger, log_external_service_call
from performance_monitor import performance_monitor  # Phase A: Import performance monitoring
from services.subtitle_engine import build_ass
from utils.rtl_utils import add_rtl_markers, clean_rtl_text, is_rtl_language

logger = get_logger(__name__)
config = get_config()

#: Where the container keeps Noto Sans Hebrew (the font `subtitle_engine`'s ASS style
#: names). Handed to FFmpeg's `ass` filter as `fontsdir` so libass finds it without
#: depending on a fontconfig cache being warm.
ASS_FONTS_DIR = os.getenv("ASS_FONTS_DIR", "/usr/share/fonts/truetype/hebrew")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remaining = seconds % 60
    milliseconds = int((seconds_remaining - int(seconds_remaining)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remaining):02d},{milliseconds:03d}"


def _ffmpeg_escape_filter_arg(value: str) -> str:
    """Escape string for safe use inside ffmpeg filter arguments wrapped with single quotes.
    Escapes backslashes and single quotes to avoid breaking the filter parser.
    """
    if value is None:
        return ""
    return value.replace("\\", "\\\\").replace("'", "\\'")


class SubtitleService:
    """Service for handling subtitle operations."""

    def __init__(self):
        self.logger = logger
        self.config = config

    def create_srt_file(
        self,
        segments: list[dict],
        output_path: str,
        use_translation: bool = False,
        language: str = "en",
    ) -> str:
        """
        Create an SRT file from segments with enhanced formatting, including RTL support.

        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text', and optionally 'translated_text'
            output_path: Path where the SRT file will be saved
            use_translation: Whether to use translated text instead of original
            language: Target language code for RTL handling

        Returns:
            Path to the created SRT file

        Raises:
            Exception: If SRT file creation fails
        """
        self.logger.info(
            "Creating SRT file",
            segments_count=len(segments),
            output_path=output_path,
            use_translation=use_translation,
            language=language,
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])

                    text = (
                        segment.get("translated_text")
                        if use_translation
                        else segment.get("text", "")
                    )
                    # Falsy, not `is None`. `subtitle_pipeline.normalize_cues`
                    # always emits the key and writes "" when there is no
                    # translation (spotting_v2 on, translation off), so an
                    # identity check here wrote a whole file of blank subtitles.
                    if not text:
                        text = segment.get("text", "") or ""

                    # Clean up text
                    text = text.replace("\n", " ").replace("\r", " ")

                    # Add RTL markers for Hebrew/Arabic using enhanced rtl_utils
                    if use_translation and is_rtl_language(language):
                        text = clean_rtl_text(text)
                        text = add_rtl_markers(text)

                    f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

            self.logger.info(
                "SRT file created successfully",
                output_path=output_path,
                segments_count=len(segments),
            )
            return output_path

        except Exception as e:
            self.logger.error(
                "SRT creation failed", output_path=output_path, error=str(e)
            )
            raise

    def fix_rtl_text_for_subtitles(self, text: str) -> str:
        """Fix RTL text direction and punctuation for video subtitles.
        Supports Hebrew, Arabic, Persian, Urdu, and other RTL languages.

        Args:
            text: Input text that may contain RTL characters

        Returns:
            Text with proper RTL formatting and punctuation fixes
        """
        if not text:
            return text

        # Use Unicode bidirectional algorithm to detect RTL characters
        import unicodedata

        has_rtl = any(unicodedata.bidirectional(char) in ("R", "AL") for char in text)

        if has_rtl:
            # Fix parentheses direction
            text = text.replace("(", "֮TEMP֮")
            text = text.replace(")", "(")
            text = text.replace("֮TEMP֮", ")")

            # Fix brackets direction
            text = text.replace("[", "֮TEMP֮")
            text = text.replace("]", "[")
            text = text.replace("֮TEMP֮", "]")

            # Fix numbers for RTL display - don't reverse them, just add LTR markers
            def fix_number(match):
                number = match.group(0)
                # Add Left-to-Right marks around numbers to preserve their direction
                return f"\u200e{number}\u200e"

            text = re.sub(r"\d[\d,\.]*", fix_number, text)

            # Add RTL override markers
            text = "\u202e" + text + "\u202c"

        return text

    def create_video_with_subtitles(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        target_language: str = "en",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Create video with burned-in subtitles, reporting progress.

        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path where output video will be saved
            target_language: Language code for subtitle styling
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(
                "Starting video subtitle embedding",
                operation="subtitle_embedding_start",
                video_path=os.path.basename(video_path),
                srt_path=os.path.basename(srt_path),
                target_language=target_language,
            )

            # FAKE mode: skip FFmpeg; just copy input to output
            if self.config.USE_FAKE_YTDLP:
                try:
                    shutil.copy2(video_path, output_path)
                    self.logger.info(
                        "FAKE mode: copied video without subtitle processing"
                    )
                    return True
                except Exception as e:
                    self.logger.error("FAKE video creation failed", error=str(e))
                    return False

            if not os.path.exists(srt_path):
                raise FileNotFoundError(srt_path)

            # Process SRT file for RTL languages
            clean_srt_path = srt_path.replace(".srt", "_clean.srt")
            try:
                with open(srt_path, encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        self.logger.error("SRT file is empty", srt_path=srt_path)
                        return False

                lines = content.split("\n")
                processed_lines = []

                for line in lines:
                    if (
                        line.strip()
                        and not line.strip().isdigit()
                        and "-->" not in line
                    ):
                        processed_line = self.fix_rtl_text_for_subtitles(line)
                        processed_lines.append(processed_line)
                    else:
                        processed_lines.append(line)

                clean_content = "\n".join(processed_lines)

                with open(clean_srt_path, "w", encoding="utf-8-sig") as f:
                    f.write(clean_content)

            except Exception as e:
                self.logger.error(
                    "Cannot process SRT file", srt_path=srt_path, error=str(e)
                )
                return False

            # Configure fonts and styling
            hebrew_fonts = [
                "Noto Sans Hebrew",
                "DejaVu Sans",
                "Liberation Sans",
                "Arial Hebrew Scholar",
                "Arial Hebrew",
                "David",
                "Arial Unicode MS",
            ]

            font_fallback = ",".join(hebrew_fonts)

            rtl_languages = ["he", "ar", "fa", "ur", "yi"]
            is_rtl = any(target_language.startswith(lang) for lang in rtl_languages)

            if is_rtl:
                subtitle_style = (
                    f"FontName={hebrew_fonts[0]},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=3,Shadow=2,MarginV=40,Alignment=2"
                )
                self.logger.info(
                    "Using enhanced RTL settings",
                    target_language=target_language,
                    font=hebrew_fonts[0],
                )
            else:
                subtitle_style = (
                    f"FontName={font_fallback},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=2,Shadow=1,MarginV=30,Alignment=2"
                )

            # Build FFmpeg command
            escaped_srt = _ffmpeg_escape_filter_arg(clean_srt_path)
            escaped_style = _ffmpeg_escape_filter_arg(subtitle_style)
            vf_filter = (
                f"subtitles='{escaped_srt}':force_style='{escaped_style}':charenc=UTF-8"
            )
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                vf_filter,
                "-c:a",
                "copy",
                # Move the moov atom to the front so the result starts playing
                # before it has fully downloaded (parity with create_video_with_ass).
                "-movflags",
                "+faststart",
                "-y",
                "-progress",
                "pipe:2",
                output_path,
            ]

            # Log cleanup: Only log FFmpeg start in DEBUG mode, otherwise use summary logging
            if config.DEBUG:
                self.logger.debug(
                    "Running FFmpeg subtitle embedding",
                    operation="ffmpeg_subtitle_start", 
                    command=" ".join(cmd[:5]) + "...",  # Only show first few args
                )
            else:
                self.logger.info(
                    "Starting video subtitle embedding",
                    operation="subtitle_embedding_start",
                    srt_path=os.path.basename(srt_path),
                    target_language=target_language,
                    video_path=os.path.basename(video_path),
                )

            # Phase A: Enhanced FFmpeg performance monitoring
            ffmpeg_start_time = time.time()
            
            # Execute FFmpeg with progress tracking
            if progress_callback:
                success = self._run_ffmpeg_with_progress(
                    cmd, video_path, progress_callback
                )
            else:
                success = self._run_ffmpeg_simple(cmd)
                
            ffmpeg_duration = time.time() - ffmpeg_start_time
            
            # Phase A: Log FFmpeg performance
            try:
                # Get video duration for performance calculation
                probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=10)
                probe_data = json.loads(probe_result.stdout)
                video_duration = float(probe_data.get("format", {}).get("duration", 0))
                
                performance_monitor.log_ffmpeg_performance(video_duration, ffmpeg_duration, "subtitle_embedding")
            except:
                # Fallback if we can't get duration
                self.logger.info(f"📊 Phase A: FFmpeg subtitle embedding took {ffmpeg_duration:.1f}s")

            # Cleanup temporary file
            if os.path.exists(clean_srt_path):
                os.remove(clean_srt_path)

            if (
                success
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0
            ):
                self.logger.info(
                    "Video with subtitles created successfully",
                    operation="subtitle_embedding_complete",
                    output_path=os.path.basename(output_path),
                    file_size_mb=round(os.path.getsize(output_path) / (1024 * 1024), 2),
                )
                return True
            else:
                self.logger.error(
                    "Output video file was not created or is empty",
                    output_path=output_path,
                )
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(
                "Video creation failed",
                error=str(e),
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            self._cleanup_temp_file(srt_path)
            return False
        except Exception as e:
            self.logger.error("Unexpected error in video creation", error=str(e))
            self._cleanup_temp_file(srt_path)
            return False

    def _run_ffmpeg_with_progress(
        self, cmd: list[str], video_path: str, progress_callback: Callable[[int], None]
    ) -> bool:
        """Run FFmpeg with progress tracking."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
        )

        try:
            # Get video duration for progress calculation
            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path,
            ]
            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.config.FFPROBE_TIMEOUT,
            )
            probe_data = json.loads(probe_result.stdout)
            total_duration = float(probe_data.get("format", {}).get("duration", 0))
        except:
            total_duration = 0

        stderr_data = ""

        while True:
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_data += stderr_line
                if "time=" in stderr_line and total_duration > 0:
                    try:
                        time_str = stderr_line.split("time=")[1].split()[0]
                        if ":" in time_str:
                            time_parts = time_str.split(":")
                            current_seconds = (
                                float(time_parts[0]) * 3600
                                + float(time_parts[1]) * 60
                                + float(time_parts[2])
                            )
                            progress_percent = min(
                                95, (current_seconds / total_duration) * 100
                            )
                            progress_callback(30 + int(progress_percent * 0.45))
                    except:
                        pass

            if process.poll() is not None:
                break

        stdout_data, remaining_stderr = process.communicate()
        stderr_data += remaining_stderr

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stdout_data, stderr_data
            )

        return True

    def _run_ffmpeg_simple(self, cmd: list[str]) -> bool:
        """Run FFmpeg without progress tracking."""
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.FFMPEG_RUN_TIMEOUT,
            )
            return True
        except subprocess.TimeoutExpired:
            raise FFmpegTimeoutError(
                "subtitle_embedding", self.config.FFMPEG_RUN_TIMEOUT
            )
        except subprocess.CalledProcessError as e:
            raise FFmpegProcessError("subtitle_embedding", e.stderr)

    def _cleanup_temp_file(self, srt_path: str) -> None:
        """Clean up temporary SRT file."""
        clean_srt_path = srt_path.replace(".srt", "_clean.srt")
        if os.path.exists(clean_srt_path):
            try:
                os.remove(clean_srt_path)
            except OSError:
                pass  # Ignore cleanup errors

    def create_video_with_subtitles_and_watermark(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        watermark_path: str,
        target_language: str = "en",
        watermark_position: tuple = ("right", "bottom"),
        watermark_opacity: float = 0.4,
        watermark_size_height: int = 80,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Create video with both subtitles and watermark in a single FFmpeg pass.
        
        This is more efficient than running two separate FFmpeg operations.
        
        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path where output video will be saved
            watermark_path: Path to watermark image
            target_language: Language code for subtitle styling
            watermark_position: Watermark position tuple
            watermark_opacity: Watermark opacity (0.0 to 1.0)
            watermark_size_height: Watermark height in pixels
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(
                "Starting combined subtitle+watermark embedding",
                operation="combined_embedding_start",
                video_path=os.path.basename(video_path),
                srt_path=os.path.basename(srt_path),
                # None-safe: `os.path.basename(None)` raises TypeError, and this log line
                # runs BEFORE the missing-watermark guard below. Without the guard here
                # too, a null logo path fails the whole render from inside a log
                # statement — the exact failure the guard exists to prevent.
                watermark_path=(
                    os.path.basename(watermark_path) if watermark_path else None
                ),
                target_language=target_language,
            )

            # FAKE mode: skip FFmpeg; just copy input to output
            if self.config.USE_FAKE_YTDLP:
                try:
                    shutil.copy2(video_path, output_path)
                    self.logger.info(
                        "FAKE mode: copied video without processing"
                    )
                    return True
                except Exception as e:
                    self.logger.error("FAKE video creation failed", error=str(e))
                    return False

            if not os.path.exists(srt_path):
                raise FileNotFoundError(srt_path)
                
            # Same guard as create_video_with_ass: a missing logo must degrade to a
            # subtitles-only render, never fail the whole job in FFmpeg. `not
            # watermark_path` is part of the check because os.path.exists(None)
            # raises TypeError rather than answering False.
            if not watermark_path or not os.path.exists(watermark_path):
                self.logger.warning(
                    "Watermark file not found, falling back to subtitles only",
                    watermark_path=watermark_path,
                )
                return self.create_video_with_subtitles(
                    video_path, srt_path, output_path, target_language, progress_callback
                )

            # Process SRT file for RTL languages (same as in create_video_with_subtitles)
            clean_srt_path = srt_path.replace(".srt", "_clean.srt")
            try:
                with open(srt_path, encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        self.logger.error("SRT file is empty", srt_path=srt_path)
                        return False

                lines = content.split("\n")
                processed_lines = []

                for line in lines:
                    if (
                        line.strip()
                        and not line.strip().isdigit()
                        and "-->" not in line
                    ):
                        processed_line = self.fix_rtl_text_for_subtitles(line)
                        processed_lines.append(processed_line)
                    else:
                        processed_lines.append(line)

                clean_content = "\n".join(processed_lines)

                with open(clean_srt_path, "w", encoding="utf-8-sig") as f:
                    f.write(clean_content)

            except Exception as e:
                self.logger.error(
                    "Cannot process SRT file", srt_path=srt_path, error=str(e)
                )
                return False

            # Configure fonts and styling (same as before)
            hebrew_fonts = [
                "Noto Sans Hebrew",
                "DejaVu Sans",
                "Liberation Sans",
                "Arial Hebrew Scholar",
                "Arial Hebrew",
                "David",
                "Arial Unicode MS",
            ]

            font_fallback = ",".join(hebrew_fonts)

            rtl_languages = ["he", "ar", "fa", "ur", "yi"]
            is_rtl = any(target_language.startswith(lang) for lang in rtl_languages)

            if is_rtl:
                subtitle_style = (
                    f"FontName={hebrew_fonts[0]},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=3,Shadow=2,MarginV=40,Alignment=2"
                )
            else:
                subtitle_style = (
                    f"FontName={font_fallback},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=2,Shadow=1,MarginV=30,Alignment=2"
                )

            # Configure watermark position
            position_map = {
                ("right", "bottom"): "W-w-10:H-h-10",
                ("left", "bottom"): "10:H-h-10",
                ("right", "top"): "W-w-10:10",
                ("left", "top"): "10:10",
                ("center", "center"): "(W-w)/2:(H-h)/2",
                ("center", "above_subtitles"): "(W-w)/2:H-h-210",
                ("upper_right", "comfortable"): "W-w-50:50",
            }
            pos_str = position_map.get(watermark_position, "W-w-10:H-h-10")

            # Build combined filter complex
            escaped_srt = _ffmpeg_escape_filter_arg(clean_srt_path)
            escaped_style = _ffmpeg_escape_filter_arg(subtitle_style)
            
            # Combined filter: first apply subtitles, then overlay watermark
            filter_complex = (
                f"[0:v]subtitles='{escaped_srt}':force_style='{escaped_style}':charenc=UTF-8[v1];"
                f"[1:v]scale=-1:{watermark_size_height},format=rgba,colorchannelmixer=aa={watermark_opacity}[logo];"
                f"[v1][logo]overlay={pos_str}[vout]"
            )
            
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                watermark_path,
                "-filter_complex",
                filter_complex,
                "-map", "[vout]",
                "-map", "0:a",
                "-c:a", "copy",
                "-preset", "fast",
                # Parity with create_video_with_ass: moov atom up front.
                "-movflags", "+faststart",
                "-y",
                "-progress", "pipe:2",
                output_path,
            ]

            # Log start
            if config.DEBUG:
                self.logger.debug(
                    "Running combined FFmpeg subtitle+watermark embedding",
                    operation="ffmpeg_combined_start",
                    command=" ".join(cmd[:5]) + "...",
                )
            else:
                self.logger.info(
                    "Starting combined video processing",
                    operation="combined_embedding_start",
                )

            # Phase A: Enhanced FFmpeg performance monitoring
            ffmpeg_start_time = time.time()
            
            # Execute FFmpeg with progress tracking
            if progress_callback:
                success = self._run_ffmpeg_with_progress(
                    cmd, video_path, progress_callback
                )
            else:
                success = self._run_ffmpeg_simple(cmd)
                
            ffmpeg_duration = time.time() - ffmpeg_start_time
            
            # Phase A: Log FFmpeg performance
            try:
                # Get video duration for performance calculation
                probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=10)
                probe_data = json.loads(probe_result.stdout)
                video_duration = float(probe_data.get("format", {}).get("duration", 0))
                
                performance_monitor.log_ffmpeg_performance(video_duration, ffmpeg_duration, "combined_subtitle_watermark")
            except:
                # Fallback if we can't get duration
                self.logger.info(f"📊 Phase A: FFmpeg combined processing took {ffmpeg_duration:.1f}s")

            # Cleanup temporary file
            if os.path.exists(clean_srt_path):
                os.remove(clean_srt_path)

            if (
                success
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0
            ):
                self.logger.info(
                    "Video with subtitles and watermark created successfully",
                    operation="combined_embedding_complete",
                    output_path=os.path.basename(output_path),
                    file_size_mb=round(os.path.getsize(output_path) / (1024 * 1024), 2),
                )
                return True
            else:
                self.logger.error(
                    "Output video file was not created or is empty",
                    output_path=output_path,
                )
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(
                "Combined video creation failed",
                error=str(e),
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            self._cleanup_temp_file(srt_path)
            return False
        except Exception as e:
            self.logger.error("Unexpected error in combined video creation", error=str(e))
            self._cleanup_temp_file(srt_path)
            return False

    # ------------------------------------------------------------------
    # render_v2: ASS subtitles via FFmpeg's `ass` filter (opt-in)
    # ------------------------------------------------------------------

    def _probe_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Return the video's real pixel dimensions, or the 1080p default on failure.

        ``build_ass`` sizes the font and margins as fractions of the height and sets
        ``PlayResX/Y`` to match, so passing the true dimensions is what makes the result
        look identical at 720p and 4K. The legacy render path only ever probes
        ``-show_format`` (duration) — resolution was never needed before.
        """
        try:
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", video_path,
            ]
            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.config.FFPROBE_TIMEOUT,
            )
            streams = json.loads(probe_result.stdout).get("streams", [])
            if streams:
                width = int(streams[0].get("width") or 0)
                height = int(streams[0].get("height") or 0)
                if width > 0 and height > 0:
                    return width, height
        except Exception as e:  # ffprobe missing, timeout, odd container, ...
            self.logger.warning("Could not probe video dimensions", error=str(e))
        self.logger.warning(
            "Falling back to 1920x1080 for ASS sizing", video_path=os.path.basename(video_path)
        )
        return 1920, 1080

    def create_video_with_ass(
        self,
        video_path: str,
        cues: list[dict],
        output_path: str,
        target_language: str = "en",
        use_translation: bool = True,
        watermark_path: Optional[str] = None,
        watermark_position: tuple = ("right", "bottom"),
        watermark_opacity: float = 0.4,
        watermark_size_height: int = 80,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Burn in subtitles from a generated ``.ass`` file (the ``render_v2`` path).

        Why a second render function instead of a branch inside the SRT ones: FFmpeg's
        ``subtitles`` filter has no ``shaping`` option — passing one is a hard failure
        ("Option not found") — so complex-shaping Hebrew requires the ``ass`` filter,
        which in turn requires an ASS file. The style, bidi isolates and two-line wrap all
        come from :mod:`services.subtitle_engine`; nothing here re-implements them.

        The watermark, when requested, is composed in the same single FFmpeg pass as the
        subtitles, exactly as :meth:`create_video_with_subtitles_and_watermark` does.

        Args:
            cues: cue dicts (``start``, ``end``, ``text``, optionally ``translated_text``)
                in the pipeline's common shape — see ``services.subtitle_pipeline``.
            use_translation: render ``translated_text`` (falling back to ``text`` per cue)
                rather than the source text.
            watermark_path: when given and present on disk, overlay it in the same pass.

        Returns:
            True on success. The generated ``.ass`` file is kept next to the output so the
            exact rendered cues can be inspected after the fact.
        """
        ass_path = os.path.splitext(output_path)[0] + ".ass"
        try:
            self.logger.info(
                "Starting ASS subtitle embedding (render_v2)",
                operation="ass_embedding_start",
                video_path=os.path.basename(video_path),
                cues_count=len(cues or []),
                target_language=target_language,
                watermark=bool(watermark_path),
            )

            # FAKE mode: skip FFmpeg; just copy input to output
            if self.config.USE_FAKE_YTDLP:
                try:
                    shutil.copy2(video_path, output_path)
                    self.logger.info("FAKE mode: copied video without ASS processing")
                    return True
                except Exception as e:
                    self.logger.error("FAKE video creation failed", error=str(e))
                    return False

            render_cues = []
            for cue in cues or []:
                text = cue.get("translated_text") if use_translation else cue.get("text")
                if not text:
                    text = cue.get("text") or ""
                text = str(text).replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    continue
                render_cues.append(
                    {"start": cue.get("start", 0), "end": cue.get("end", 0), "text": text}
                )

            if not render_cues:
                self.logger.error("No renderable cues for ASS subtitles")
                return False

            video_w, video_h = self._probe_video_dimensions(video_path)
            # Prefix match, exactly as the legacy render functions do, so "he-IL" is
            # still recognised as RTL (utils.rtl_utils.is_rtl_language demands an exact
            # code and would answer False for it).
            rtl_languages = ("he", "ar", "fa", "ur", "yi")
            is_rtl = any((target_language or "").startswith(lang) for lang in rtl_languages)
            ass_content = build_ass(
                render_cues, video_w=video_w, video_h=video_h, rtl=is_rtl
            )
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            self.logger.info(
                "ASS file written",
                ass_path=os.path.basename(ass_path),
                cues_count=len(render_cues),
                video_resolution=f"{video_w}x{video_h}",
            )

            # `shaping=complex` is valid ONLY on the `ass` filter (the `subtitles` filter
            # rejects it outright), and it is what makes Hebrew glyph shaping correct.
            ass_filter = (
                f"ass='{_ffmpeg_escape_filter_arg(ass_path)}'"
                f":fontsdir={ASS_FONTS_DIR}:shaping=complex"
            )

            # Explicit encoder args: the ass filter re-encodes video, and the defaults
            # differ from the legacy path's. faststart keeps the moov atom up front.
            encode_args = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "copy",
            ]

            if watermark_path and os.path.exists(watermark_path):
                position_map = {
                    ("right", "bottom"): "W-w-10:H-h-10",
                    ("left", "bottom"): "10:H-h-10",
                    ("right", "top"): "W-w-10:10",
                    ("left", "top"): "10:10",
                    ("center", "center"): "(W-w)/2:(H-h)/2",
                    ("center", "above_subtitles"): "(W-w)/2:H-h-210",
                    ("upper_right", "comfortable"): "W-w-50:50",
                }
                pos_str = position_map.get(watermark_position, "W-w-10:H-h-10")
                filter_complex = (
                    f"[0:v]{ass_filter}[v1];"
                    f"[1:v]scale=-1:{watermark_size_height},format=rgba,"
                    f"colorchannelmixer=aa={watermark_opacity}[logo];"
                    f"[v1][logo]overlay={pos_str}[vout]"
                )
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-i", watermark_path,
                    "-filter_complex", filter_complex,
                    "-map", "[vout]",
                    "-map", "0:a",
                    *encode_args,
                    "-y",
                    "-progress", "pipe:2",
                    output_path,
                ]
            else:
                if watermark_path:
                    self.logger.warning(
                        "Watermark file not found, rendering ASS subtitles only",
                        watermark_path=watermark_path,
                    )
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", ass_filter,
                    *encode_args,
                    "-y",
                    "-progress", "pipe:2",
                    output_path,
                ]

            if config.DEBUG:
                self.logger.debug(
                    "Running FFmpeg ASS embedding",
                    operation="ffmpeg_ass_start",
                    command=" ".join(cmd),
                )

            ffmpeg_start_time = time.time()
            if progress_callback:
                success = self._run_ffmpeg_with_progress(cmd, video_path, progress_callback)
            else:
                success = self._run_ffmpeg_simple(cmd)
            ffmpeg_duration = time.time() - ffmpeg_start_time

            try:
                probe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", video_path,
                ]
                probe_result = subprocess.run(
                    probe_cmd, capture_output=True, text=True, check=True, timeout=10
                )
                video_duration = float(
                    json.loads(probe_result.stdout).get("format", {}).get("duration", 0)
                )
                performance_monitor.log_ffmpeg_performance(
                    video_duration, ffmpeg_duration, "ass_subtitle_embedding"
                )
            except Exception:
                self.logger.info(f"📊 FFmpeg ASS embedding took {ffmpeg_duration:.1f}s")

            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.logger.info(
                    "Video with ASS subtitles created successfully",
                    operation="ass_embedding_complete",
                    output_path=os.path.basename(output_path),
                    ass_path=os.path.basename(ass_path),
                    file_size_mb=round(os.path.getsize(output_path) / (1024 * 1024), 2),
                )
                return True

            self.logger.error(
                "Output video file was not created or is empty", output_path=output_path
            )
            return False

        except subprocess.CalledProcessError as e:
            self.logger.error(
                "ASS video creation failed",
                error=str(e),
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            return False
        except Exception as e:
            self.logger.error("Unexpected error in ASS video creation", error=str(e))
            return False

    def add_watermark_to_video(
        self,
        input_video_path: str,
        watermark_path: str,
        output_video_path: str,
        position: tuple = ("right", "bottom"),
        opacity: float = 0.4,
        size_height: int = 80,
    ) -> Optional[str]:
        """Add watermark/logo to video using FFmpeg.

        Args:
            input_video_path: Path to input video
            watermark_path: Path to watermark image
            output_video_path: Path for output video
            position: Watermark position tuple
            opacity: Watermark opacity (0.0 to 1.0)
            size_height: Watermark height in pixels

        Returns:
            Path to output video if successful, None otherwise
        """
        try:
            # Log cleanup: Only log watermark details in DEBUG mode
            if config.DEBUG:
                self.logger.debug(
                    "Adding watermark to video",
                    operation="watermark_start",
                    input_video=os.path.basename(input_video_path),
                    watermark=os.path.basename(watermark_path),
                    position=position,
                    opacity=opacity,
                )
            else:
                self.logger.info("Adding watermark to video")

            if not os.path.exists(watermark_path):
                self.logger.warning(
                    "Watermark file not found, skipping watermark",
                    watermark_path=watermark_path,
                )
                shutil.copy2(input_video_path, output_video_path)
                return output_video_path

            position_map = {
                ("right", "bottom"): "W-w-10:H-h-10",
                ("left", "bottom"): "10:H-h-10",
                ("right", "top"): "W-w-10:10",
                ("left", "top"): "10:10",
                ("center", "center"): "(W-w)/2:(H-h)/2",
                ("center", "above_subtitles"): "(W-w)/2:H-h-210",
                ("upper_right", "comfortable"): "W-w-50:50",
            }
            pos_str = position_map.get(position, "W-w-10:H-h-10")

            filter_complex = f"[1:v]scale=-1:{size_height},format=rgba,colorchannelmixer=aa={opacity}[logo];[0:v][logo]overlay={pos_str}"

            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_video_path,
                "-i",
                watermark_path,
                "-filter_complex",
                filter_complex,
                "-c:a",
                "copy",
                "-preset",
                "fast",
                output_video_path,
            ]

            log_external_service_call(self.logger, "ffmpeg", "watermark", success=True)

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.config.FFMPEG_RUN_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                raise FFmpegTimeoutError("watermark", self.config.FFMPEG_RUN_TIMEOUT)
            except subprocess.CalledProcessError as e:
                raise FFmpegProcessError("watermark", e.stderr)

            if result.returncode == 0:
                self.logger.info(
                    "Watermark added successfully",
                    operation="watermark_complete",
                    output_video=os.path.basename(output_video_path),
                )
                return output_video_path
            else:
                self.logger.error("FFmpeg watermark error", stderr=result.stderr)
                shutil.copy2(input_video_path, output_video_path)
                return output_video_path

        except Exception as e:
            self.logger.error("Error adding watermark", error=str(e))
            try:
                shutil.copy2(input_video_path, output_video_path)
                return output_video_path
            except:
                return None


# Create service instance for easy import
subtitle_service = SubtitleService()

# Export functions for backward compatibility
create_srt_file = subtitle_service.create_srt_file
fix_rtl_text_for_subtitles = subtitle_service.fix_rtl_text_for_subtitles
# Backward compatibility
fix_hebrew_text_for_subtitles = subtitle_service.fix_rtl_text_for_subtitles
create_video_with_subtitles = subtitle_service.create_video_with_subtitles
create_video_with_subtitles_and_watermark = subtitle_service.create_video_with_subtitles_and_watermark
create_video_with_ass = subtitle_service.create_video_with_ass
add_watermark_to_video = subtitle_service.add_watermark_to_video
