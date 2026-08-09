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

import numpy as np

from config import get_config
from core.exceptions import FFmpegProcessError, FFmpegTimeoutError, FileNotFoundError
from logging_config import get_logger, log_external_service_call
from performance_monitor import (  # Phase A: Import performance monitoring
    performance_monitor,
)
from services.subtitle_engine import build_ass, hebrew_typography, layout_params
from utils.rtl_utils import add_rtl_markers, clean_rtl_text, is_rtl_language

logger = get_logger(__name__)
config = get_config()

#: Where the container keeps Noto Sans Hebrew (the font `subtitle_engine`'s ASS style
#: names). Handed to FFmpeg's `ass` filter as `fontsdir` so libass finds it without
#: depending on a fontconfig cache being warm.
ASS_FONTS_DIR = os.getenv("ASS_FONTS_DIR", "/usr/share/fonts/truetype/hebrew")

#: Subtitle placement -> alignment code for the LEGACY renderer, i.e. FFmpeg's
#: ``subtitles`` filter driven by ``force_style``.
#:
#: This is NOT the same numbering as :data:`subtitle_engine.ASS_ALIGNMENT`, and the
#: difference is the whole reason this constant exists. The ``ass`` filter reads a real
#: ``.ass`` file and uses ASS v4+ numpad codes (7 8 9 / 4 5 6 / 1 2 3). A ``force_style``
#: override on the ``subtitles`` filter is parsed with the older SSA v4 semantics, where
#: the code is ``horizontal + 4 for top + 8 for middle``: 1-3 bottom, 5-7 top, 9-11
#: middle. The two schemes agree on 2 (bottom centre) and on nothing else that matters.
#:
#: Measured, not inferred — every value below was rendered on a black frame and the ink
#: located (see ``test_subtitle_layout_render.TestBurnedSubtitlePosition``). Feeding the
#: ASS codes to this path put ``top`` at middle-LEFT and ``side`` at the TOP.
SSA_ALIGNMENT = {"bottom": 2, "top": 6, "side": 11}

# ----------------------------------------------------------------------------------
# Lower-third ("chyron") collision detection — constants
# ----------------------------------------------------------------------------------
#: Frames sampled across the video to decide whether the subtitle band is occupied.
#: Seven is enough to see through several atypical shots and still cheap: ONE decode per
#: position now scores BOTH bands, because the crop covers them together and the split
#: happens in numpy.
CHYRON_SAMPLE_FRAMES = 7

#: Where in the video those frames are taken, as fractions of its duration. FIXED, so
#: the decision is reproducible for a given file — a random or "every N seconds" sample
#: would make the rendered output depend on the clock.
#:
#: The first 10% is no longer skipped. It was, on the theory that titles and fades are
#: unrepresentative; the corpus says otherwise — a broadcast SUPER (the name-and-title
#: card) lives almost exclusively in the first few seconds of an interview, and it is
#: precisely the graphic a subtitle must not land on. Missing it to avoid a fade is the
#: wrong trade, and the quantile rule below is what makes an unrepresentative sample
#: harmless.
CHYRON_SAMPLE_POSITIONS = (0.03, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9)

#: 8-bit luma step between neighbouring pixels that counts as an edge. Well above sensor
#: noise and compression dither, low enough to catch anti-aliased caption text on a
#: translucent bar.
CHYRON_EDGE_THRESHOLD = 24

#: Fraction of neighbouring-pixel pairs that must be edges for the band to count as
#: OCCUPIED, averaged over the sampled frames.
#:
#: MEASURED, not guessed: 35 clips from this project's own uploads and downloads were
#: scored (news, podcasts, vertical phone video, music video, archive footage, a
#: 320x180 stub). The scores fall into two groups with a wide empty gap between them:
#:
#:     music video / clean studio / phone video   0.0000 - 0.0104
#:     ordinary footage, no lower third           0.0245 - 0.0606
#:     ------------------ empty band -------------------------
#:     Fox News clip with a chyron (IMG_2870)     0.0873
#:     NATO briefing with a chyron                0.0944
#:     Hebrew news bulletin                       0.1318
#:
#: 0.070 sits inside that gap: 15% above the busiest clip WITHOUT a lower third and 20%
#: below the least busy clip WITH one. Choosing the gap rather than hugging either side
#: is what makes the threshold survive a clip neither group anticipated.
#:
#: The asymmetry of the two errors is deliberate and points the other way from the usual
#: instinct: a false POSITIVE moves the subtitle up ~9% of the frame on a video that did
#: not need it, which a viewer barely registers; a false NEGATIVE reproduces today's
#: behaviour exactly, which is a known-bad but familiar output. Neither is a failure, so
#: the threshold is placed for accuracy rather than for a safety margin on one side.
CHYRON_BUSY_THRESHOLD = 0.070

#: Bottom margin, as a fraction of the height, used when the default band is occupied.
#: The default is 0.12; 0.24 lifts the subtitle box clear of the lower-third strip on
#: 16:9 news graphics while keeping it in the bottom third of the picture, where a
#: viewer's eye expects subtitles to be.
CHYRON_MARGIN_V_FRAC = 0.24

#: Multiplier turning ``font_px * max_lines`` into the pixel height the subtitle BOX
#: occupies — line spacing plus the opaque box's padding and 3px outline. Only used to
#: aim the sampling crop at the right strip of picture; it is not a rendering value.
CHYRON_BOX_HEIGHT_FACTOR = 1.35

#: Quantile of the per-frame scores that decides, instead of the median.
#:
#: The median was chosen to survive fades to black, and it does — but it also survives
#: the thing being looked for. A chyron that is on screen for a third of the video (a
#: super, an intermittent breaking-news bar, a lower third that comes and goes with the
#: shot) leaves a MINORITY of samples busy, and the median is by definition blind to a
#: minority. Measured on the corpus: a podcast whose subtitle band is busy in 2 of 5
#: samples scored a median of 0.0073 — 10x under the threshold — while those two samples
#: read 0.0556 and 0.0589.
#:
#: The 75th percentile keeps the fade-to-black robustness (two dark frames out of seven
#: cannot pull it down) while letting a real minority of busy frames be seen. Re-scored
#: over the whole corpus it changes no verdict that was already right and introduces no
#: false positive: the busiest clip WITHOUT a lower third reaches 0.0486 at this
#: quantile, still comfortably under 0.070.
CHYRON_SCORE_QUANTILE = 0.75

#: A band is also called occupied when at least this many INDIVIDUAL samples are over
#: the threshold, whatever the quantile says. Belt and braces for the intermittent case
#: with many samples: two independent frames showing a graphic is not a fluke.
CHYRON_MIN_BUSY_SAMPLES = 2


def _edge_density(band) -> float:
    """Fraction of neighbouring pixel pairs in a greyscale array that form an edge.

    Split out from :meth:`SubtitleService.detect_lower_third` so the scoring can be
    unit-tested on synthetic arrays with no FFmpeg, no file and no video.

    Horizontal and vertical neighbours are counted together against the total number of
    pairs, which keeps the result a dimensionless ratio — a 4K band and a 480p band of
    the same picture score the same, so one threshold serves every resolution.

    Computed in ``int16``: ``uint8`` subtraction wraps around, so ``5 - 250`` would come
    out as 11 instead of -245 and a strong edge would read as none at all.
    """
    values = band.astype(np.int16)
    horizontal = np.abs(np.diff(values, axis=1))
    vertical = np.abs(np.diff(values, axis=0))
    pairs = horizontal.size + vertical.size
    if pairs == 0:
        return 0.0
    edges = int((horizontal > CHYRON_EDGE_THRESHOLD).sum()) + int(
        (vertical > CHYRON_EDGE_THRESHOLD).sum()
    )
    return edges / pairs


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

                    # Hebrew punctuation repair, applied to the SRT and not only to the
                    # ASS. `build_ass` has always called `hebrew_typography`, so the
                    # burned-in picture read התנ״ך while the .srt shipped beside it read
                    # התנ"ך with an ASCII quote — the same cue, two different spellings,
                    # and the one a user can copy out of was the wrong one.
                    # `clean_rtl_text` does not cover this: its `fix_hebrew_quotes` only
                    # rewrites a PAIR of quotes, and an acronym mark has no partner.
                    text = hebrew_typography(text)

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
        progress_callback: Callable[[int], None] | None = None,
        subtitle_position: str = "bottom",
    ) -> bool:
        """Create video with burned-in subtitles, reporting progress.

        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path where output video will be saved
            target_language: Language code for subtitle styling
            progress_callback: Optional callback for progress updates
            subtitle_position: ``bottom``, ``top`` or ``side`` (middle-right)

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

            alignment = SSA_ALIGNMENT.get(subtitle_position, SSA_ALIGNMENT["bottom"])

            if is_rtl:
                subtitle_style = (
                    f"FontName={hebrew_fonts[0]},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=3,Shadow=2,MarginV=40,"
                    f"Alignment={alignment}"
                )
                self.logger.info(
                    "Using enhanced RTL settings",
                    target_language=target_language,
                    font=hebrew_fonts[0],
                )
            else:
                subtitle_style = (
                    f"FontName={font_fallback},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=2,Shadow=1,MarginV=30,"
                    f"Alignment={alignment}"
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
                    probe_cmd, capture_output=True, text=True, check=True, timeout=10
                )
                probe_data = json.loads(probe_result.stdout)
                video_duration = float(probe_data.get("format", {}).get("duration", 0))

                performance_monitor.log_ffmpeg_performance(
                    video_duration, ffmpeg_duration, "subtitle_embedding"
                )
            except:
                # Fallback if we can't get duration
                self.logger.info(
                    f"📊 Phase A: FFmpeg subtitle embedding took {ffmpeg_duration:.1f}s"
                )

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
        progress_callback: Callable[[int], None] | None = None,
        subtitle_position: str = "bottom",
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
            subtitle_position: ``bottom``, ``top`` or ``side`` (middle-right)

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
                    self.logger.info("FAKE mode: copied video without processing")
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
                    video_path,
                    srt_path,
                    output_path,
                    target_language=target_language,
                    progress_callback=progress_callback,
                    subtitle_position=subtitle_position,
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

            alignment = SSA_ALIGNMENT.get(subtitle_position, SSA_ALIGNMENT["bottom"])

            if is_rtl:
                subtitle_style = (
                    f"FontName={hebrew_fonts[0]},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=3,Shadow=2,MarginV=40,"
                    f"Alignment={alignment}"
                )
            else:
                subtitle_style = (
                    f"FontName={font_fallback},FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,BackColour=&H80000000,Outline=2,Shadow=1,MarginV=30,"
                    f"Alignment={alignment}"
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
                "-map",
                "[vout]",
                "-map",
                "0:a",
                "-c:a",
                "copy",
                "-preset",
                "fast",
                # Parity with create_video_with_ass: moov atom up front.
                "-movflags",
                "+faststart",
                "-y",
                "-progress",
                "pipe:2",
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
                    probe_cmd, capture_output=True, text=True, check=True, timeout=10
                )
                probe_data = json.loads(probe_result.stdout)
                video_duration = float(probe_data.get("format", {}).get("duration", 0))

                performance_monitor.log_ffmpeg_performance(
                    video_duration, ffmpeg_duration, "combined_subtitle_watermark"
                )
            except:
                # Fallback if we can't get duration
                self.logger.info(
                    f"📊 Phase A: FFmpeg combined processing took {ffmpeg_duration:.1f}s"
                )

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
            self.logger.error(
                "Unexpected error in combined video creation", error=str(e)
            )
            self._cleanup_temp_file(srt_path)
            return False

    # ------------------------------------------------------------------
    # render_v2: ASS subtitles via FFmpeg's `ass` filter (opt-in)
    # ------------------------------------------------------------------

    def probe_video_dimensions(self, video_path: str) -> tuple[int, int]:
        """Return the video's real pixel dimensions, or the 1080p default on failure.

        ``subtitle_engine.layout_params`` derives the font size, the margins AND the
        per-line character budget from these two numbers, so they are needed *before*
        translation, not only at render time — which is why this is public. The legacy
        render path only ever probes ``-show_format`` (duration); resolution was never
        needed before.

        Never raises: an unprobeable file falls back to 1920x1080, i.e. to exactly the
        landscape behaviour that predates the width-aware layout.
        """
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                video_path,
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
            "Falling back to 1920x1080 for ASS sizing",
            video_path=os.path.basename(video_path),
        )
        return 1920, 1080

    #: Kept so nothing that reached for the private name breaks; the public one is
    #: the same method.
    _probe_video_dimensions = probe_video_dimensions

    def _probe_duration(self, video_path: str) -> float:
        """Video duration in seconds, or 0.0 when it cannot be determined."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.FFPROBE_TIMEOUT,
            )
            return max(0.0, float((result.stdout or "").strip()))
        except Exception:  # unprobeable file, odd container, ffprobe missing
            return 0.0

    def _grab_band(
        self, video_path: str, at: float, x: int, y: int, width: int, height: int
    ):
        """Decode ONE frame's subtitle band as an 8-bit greyscale array.

        Cropping inside FFmpeg rather than in Python is the whole reason this is cheap:
        only the band crosses the pipe, so a 4K frame costs the same as a 480p one.

        Returns ``None`` for any failure — a frame that will not decode is a frame this
        detector simply does not get a vote from, never an exception into the render path.
        """
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{at:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            f"crop={width}:{height}:{x}:{y},format=gray",
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception:
            return None
        expected = width * height
        if result.returncode != 0 or len(result.stdout) < expected:
            return None
        return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(
            height, width
        )

    def _score_bands(self, video_path: str, at: float, geometry: dict):
        """Decode ONE frame once and score BOTH bands from it.

        One decode, two scores. The bands are adjacent, so a single crop covers them and
        numpy splits the array — which is why doubling the number of measurements did not
        double the cost. The two scores are computed on the two sub-arrays SEPARATELY and
        are never averaged together: see :meth:`detect_lower_third` for why merging them
        is the bug this method exists to avoid.

        Returns ``(subtitle_score, bottom_score)``, either of which may be ``None``.
        """
        frame = self._grab_band(
            video_path,
            at,
            geometry["x"],
            geometry["top"],
            geometry["w"],
            geometry["total_h"],
        )
        if frame is None:
            return None, None
        split = geometry["box_h"]
        subtitle = frame[:split] if split > 1 else None
        bottom = frame[split:] if frame.shape[0] - split > 1 else None
        return (
            (
                round(_edge_density(subtitle), 4)
                if subtitle is not None and subtitle.size
                else None
            ),
            (
                round(_edge_density(bottom), 4)
                if bottom is not None and bottom.size
                else None
            ),
        )

    @staticmethod
    def _band_verdict(samples: list) -> dict:
        """Turn one band's per-frame scores into a verdict.

        Busy when the :data:`CHYRON_SCORE_QUANTILE` of the samples is over the threshold,
        OR when at least :data:`CHYRON_MIN_BUSY_SAMPLES` individual frames are — the
        second rule catching the intermittent graphic that a quantile still averages away.
        """
        usable = [s for s in samples if s is not None]
        verdict = {
            "samples": samples,
            "score": None,
            "busy_samples": 0,
            "busy": False,
        }
        if not usable:
            return verdict
        verdict["score"] = round(float(np.quantile(usable, CHYRON_SCORE_QUANTILE)), 4)
        verdict["busy_samples"] = sum(1 for s in usable if s > CHYRON_BUSY_THRESHOLD)
        verdict["busy"] = bool(
            verdict["score"] > CHYRON_BUSY_THRESHOLD
            or verdict["busy_samples"] >= CHYRON_MIN_BUSY_SAMPLES
        )
        return verdict

    def detect_lower_third(self, video_path: str, layout: dict) -> dict:
        """Decide whether the picture under the subtitles is already occupied.

        The defect this closes
        ----------------------
        On news footage the subtitle box lands ON the broadcaster's chyron. At the
        default ``margin_v`` of 0.12 that is 86px up from the bottom of a 720p frame —
        which is exactly where a Fox News lower third sits. Two layers of white text on
        two coloured bars, and neither is readable. It is not a subtitle bug; it is a
        subtitle put somewhere the picture was already using.

        The defect the FIRST version of this had
        ----------------------------------------
        It sampled exactly one rectangle: the one the subtitle box would occupy. That
        rectangle stops at ``height - margin_v``, so **everything in the bottom margin
        was invisible to it** — and a lower third that begins twelve pixels below the
        subtitle box is still a lower third the subtitle will collide with, because the
        opaque box, its outline and a two-line cue are all taller than the estimate.
        Measured on a real BREAKING NEWS clip: the sampled band scored 0.0093 (7x under
        the threshold, verdict "clear") while the bar itself, 12px lower, scored 0.1146
        — 1.6x OVER it. The detector was not wrong about its band; it was looking at the
        wrong band.

        So TWO bands are scored, SEPARATELY:

        * the subtitle band — where the box goes;
        * the bottom strip — ``height - margin_v`` down to the bottom edge, i.e. exactly
          the picture the first version could not see.

        Busy in EITHER raises the box. They are deliberately not merged into one taller
        band: averaging a busy strip with a clear band dilutes the score below the
        threshold and reproduces the original miss. On that same clip the merged band
        scores 0.0450 — still under 0.070, still "clear", still wrong.

        How the frames are judged
        -------------------------
        Seven frames spread across the whole video (:data:`CHYRON_SAMPLE_POSITIONS`,
        including the first 10%, where broadcast supers live) are cropped inside FFmpeg
        and scored for **edge density**: the fraction of neighbouring pixel pairs
        differing by more than :data:`CHYRON_EDGE_THRESHOLD`. Graphics and caption text
        are dense in hard edges; skin, sky, walls and bokeh are not. The score is a
        *ratio*, so it does not care how large the frame is.

        The verdict is the 75th percentile of the samples, not their median — see
        :data:`CHYRON_SCORE_QUANTILE` — with a second rule for two or more individually
        busy frames.

        Guarantees
        ----------
        * **Deterministic** — fixed sample positions, fixed threshold, no clock, no RNG.
        * **Fast** — seven input-seeks with an in-FFmpeg crop; ONE decode scores both
          bands.
        * **Cannot fail a render** — every failure path returns ``busy=False``, i.e. the
          behaviour that shipped before this existed.
        * **Conservative on missing data** — fewer than three usable frames means no
          decision is made rather than a guess.

        Returns:
            ``{"busy": bool, "score": float|None, "threshold": float, "samples": [...],
            "sample_times": [...], "bands": {"subtitle": {...}, "bottom": {...}},
            "band": {...}, "reason": str}`` — always all keys, suitable for dropping
            straight into the research archive. ``score`` and ``samples`` describe the
            band that decided (the busier one), so existing readers keep working.
        """
        decision = {
            "busy": False,
            "score": None,
            "threshold": CHYRON_BUSY_THRESHOLD,
            "samples": [],
            "sample_times": [],
            "bands": None,
            "band": None,
            "reason": "not evaluated",
        }
        try:
            duration = self._probe_duration(video_path)
            if duration <= 0:
                decision["reason"] = "duration unknown — leaving the default margin"
                return decision

            height = int(layout["video_h"])
            width = int(layout["video_w"])
            margin_v = max(0, int(layout["margin_v"]))
            box_h = int(
                round(
                    layout["font_px"] * layout["max_lines"] * CHYRON_BOX_HEIGHT_FACTOR
                )
            )
            bottom = height - margin_v
            top = max(0, bottom - box_h)
            x = int(layout["margin_h"])
            band_w = min(max(1, width - 2 * x), max(1, width - x))
            box_h = min(max(1, bottom - top), max(1, height - top))
            total_h = min(max(1, height - top), height - top)
            geometry = {
                "x": x,
                "top": top,
                "w": band_w,
                "box_h": box_h,
                "total_h": total_h,
            }
            decision["band"] = {"x": x, "y": top, "w": band_w, "h": box_h}
            decision["bands"] = {
                "subtitle": {"x": x, "y": top, "w": band_w, "h": box_h},
                "bottom": {
                    "x": x,
                    "y": top + box_h,
                    "w": band_w,
                    "h": max(0, total_h - box_h),
                },
            }

            started = time.time()
            subtitle_samples: list = []
            bottom_samples: list = []
            times: list = []
            for position in CHYRON_SAMPLE_POSITIONS[:CHYRON_SAMPLE_FRAMES]:
                at = duration * position
                sub_score, bottom_score = self._score_bands(video_path, at, geometry)
                if sub_score is None and bottom_score is None:
                    continue
                times.append(round(at, 2))
                subtitle_samples.append(sub_score)
                bottom_samples.append(bottom_score)

            decision["sample_times"] = times
            decision["elapsed_s"] = round(time.time() - started, 3)
            if len(times) < 3:
                decision["reason"] = (
                    f"only {len(times)} of {len(CHYRON_SAMPLE_POSITIONS)} frames "
                    "decoded — not enough to judge, leaving the default margin"
                )
                return decision

            subtitle = self._band_verdict(subtitle_samples)
            bottom_band = self._band_verdict(bottom_samples)
            decision["bands"]["subtitle"].update(subtitle)
            decision["bands"]["bottom"].update(bottom_band)

            decider = "subtitle"
            if bottom_band["busy"] and not subtitle["busy"]:
                decider = "bottom"
            elif bottom_band["busy"] and subtitle["busy"]:
                decider = (
                    "bottom"
                    if (bottom_band["score"] or 0) > (subtitle["score"] or 0)
                    else "subtitle"
                )
            elif (bottom_band["score"] or 0) > (subtitle["score"] or 0):
                decider = "bottom"

            winner = subtitle if decider == "subtitle" else bottom_band
            decision["busy"] = bool(subtitle["busy"] or bottom_band["busy"])
            decision["score"] = winner["score"]
            decision["samples"] = winner["samples"]
            decision["decided_by"] = decider
            decision["reason"] = (
                f"{decider} band p{int(CHYRON_SCORE_QUANTILE * 100)} edge density "
                f"{winner['score']} "
                f"{'>' if decision['busy'] else '<='} {CHYRON_BUSY_THRESHOLD} "
                f"({winner['busy_samples']} of {len(times)} samples individually busy; "
                f"subtitle band {subtitle['score']}, bottom strip {bottom_band['score']})"
            )
            return decision
        except Exception as exc:  # noqa: BLE001 - detection may never fail a render
            self.logger.warning(
                "Lower-third detection failed — keeping the default subtitle margin",
                error=str(exc),
            )
            decision["reason"] = f"detection error: {type(exc).__name__}: {exc}"
            return decision

    def create_video_with_ass(
        self,
        video_path: str,
        cues: list[dict],
        output_path: str,
        target_language: str = "en",
        use_translation: bool = True,
        watermark_path: str | None = None,
        watermark_position: tuple = ("right", "bottom"),
        watermark_opacity: float = 0.4,
        watermark_size_height: int = 80,
        progress_callback: Callable[[int], None] | None = None,
        layout: "dict | None" = None,
        recorder=None,
        detect_lower_third: bool = True,
        subtitle_position: str = "bottom",
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
            layout: the ``subtitle_engine.layout_params`` dict the upstream stages
                budgeted their character counts against. Omitted, it is derived here
                from the probed dimensions — the same inputs, so the same answer; the
                parameter exists so the caller can prove they agree rather than assume it.
            recorder: optional research recorder (duck-typed — anything with
                ``update_meta(**fields)``). The lower-third decision and its score are
                archived through it. ``None`` disables archiving; a recorder that raises
                is ignored, because recording a render may never fail one.
            detect_lower_third: run the chyron collision check (:meth:`detect_lower_third`)
                and raise the subtitle box when the band is occupied. On by default —
                this is an automatic product behaviour, not an experiment. The parameter
                exists so tests can pin the geometry without decoding frames.
            subtitle_position: ``bottom``, ``top`` or ``side`` (middle-right). Lower-
                third avoidance only applies to the bottom placement.

        Returns:
            True on success. The generated ``.ass`` file is kept next to the output so the
            exact rendered cues can be inspected after the fact.

        Note:
            Raising the box changes ``margin_v`` ONLY. Every character budget upstream
            stages committed to — ``font_px``, ``max_line_chars``, ``max_chars_per_cue`` —
            is derived from the width and is untouched, so moving the box cannot make
            text that was going to fit stop fitting.
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
                text = (
                    cue.get("translated_text") if use_translation else cue.get("text")
                )
                if not text:
                    text = cue.get("text") or ""
                text = str(text).replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    continue
                render_cues.append(
                    {
                        "start": cue.get("start", 0),
                        "end": cue.get("end", 0),
                        "text": text,
                    }
                )

            if not render_cues:
                self.logger.error("No renderable cues for ASS subtitles")
                return False

            if layout:
                video_w, video_h = layout["video_w"], layout["video_h"]
            else:
                video_w, video_h = self.probe_video_dimensions(video_path)
                layout = layout_params(video_w, video_h)

            # Automatic lower-third avoidance. The owner's call: the subtitle moves
            # itself rather than asking the user to notice the collision and tick a box.
            chyron = None
            if detect_lower_third and subtitle_position == "bottom":
                chyron = self.detect_lower_third(video_path, layout)
                if chyron["busy"]:
                    raised = layout_params(
                        video_w, video_h, margin_v_frac=CHYRON_MARGIN_V_FRAC
                    )
                    # Copy: the caller's layout is the one upstream budgeted against and
                    # must not change under it. Only margin_v moves.
                    layout = dict(layout)
                    chyron["margin_v_before"] = layout["margin_v"]
                    chyron["margin_v_after"] = raised["margin_v"]
                    layout["margin_v"] = raised["margin_v"]
                    self.logger.info(
                        "Lower third detected under the subtitle band — raising the "
                        "subtitle box",
                        operation="chyron_avoid",
                        score=chyron["score"],
                        threshold=chyron["threshold"],
                        margin_v_before=chyron["margin_v_before"],
                        margin_v_after=chyron["margin_v_after"],
                    )
                else:
                    self.logger.info(
                        "Subtitle band is clear — keeping the default margin",
                        operation="chyron_clear",
                        score=chyron["score"],
                        threshold=chyron["threshold"],
                    )
                if recorder is not None:
                    try:
                        recorder.update_meta(lower_third=chyron)
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning(
                            "Research recorder raised while archiving the lower-third "
                            "decision — ignored",
                            error=str(exc),
                        )

            # Prefix match, exactly as the legacy render functions do, so "he-IL" is
            # still recognised as RTL (utils.rtl_utils.is_rtl_language demands an exact
            # code and would answer False for it).
            rtl_languages = ("he", "ar", "fa", "ur", "yi")
            is_rtl = any(
                (target_language or "").startswith(lang) for lang in rtl_languages
            )
            ass_content = build_ass(
                render_cues,
                video_w=video_w,
                video_h=video_h,
                rtl=is_rtl,
                layout=layout,
                position=subtitle_position,
            )
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            self.logger.info(
                "ASS file written",
                ass_path=os.path.basename(ass_path),
                cues_count=len(render_cues),
                video_resolution=f"{video_w}x{video_h}",
                font_px=layout["font_px"],
                max_line_chars=layout["max_line_chars"],
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
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "copy",
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
                    "-i",
                    video_path,
                    "-i",
                    watermark_path,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[vout]",
                    "-map",
                    "0:a",
                    *encode_args,
                    "-y",
                    "-progress",
                    "pipe:2",
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
                    "-i",
                    video_path,
                    "-vf",
                    ass_filter,
                    *encode_args,
                    "-y",
                    "-progress",
                    "pipe:2",
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
                success = self._run_ffmpeg_with_progress(
                    cmd, video_path, progress_callback
                )
            else:
                success = self._run_ffmpeg_simple(cmd)
            ffmpeg_duration = time.time() - ffmpeg_start_time

            try:
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

            if (
                success
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0
            ):
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
    ) -> str | None:
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
create_video_with_subtitles_and_watermark = (
    subtitle_service.create_video_with_subtitles_and_watermark
)
create_video_with_ass = subtitle_service.create_video_with_ass
add_watermark_to_video = subtitle_service.add_watermark_to_video
