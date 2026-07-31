"""
Video processing tasks for SubsTranslator
Handles transcription, translation, and video creation
"""
import os
import shutil
import time

from celery_worker import celery_app
from config import get_config
from logging_config import (
    get_logger,
    log_task_complete,
    log_task_error,
    log_task_start,
)
from services.subtitle_engine import reflow_dangling_connectors, words_to_cues
from services.subtitle_pipeline import (
    any_enabled,
    flags_summary,
    normalize_cues,
    resolve_flags,
)
from services.subtitle_service import subtitle_service
from services.stats_service import save_video_stats
from services.transcription_service import (
    transcribe_and_translate_streamed,
    transcribe_video,
    transcribe_with_words,
    translate_segments,
)
from services.translation_v2 import cps_report, enforce_cps, translate_cues
from utils.file_utils import clean_filename

from .progress_manager import ProgressManager

# Configuration
config = get_config()
logger = get_logger(__name__)

DOWNLOADS_FOLDER = config.DOWNLOADS_FOLDER


class TranslationFailedWithSalvage(Exception):
    """Translation failed, but the source-language SRT was already written to disk.

    The job still FAILS — substituting the untranslated source for a translation and
    calling it a success is the one thing this pipeline must never do. This exception
    only carries the name of the file that survived, so the failure payload can offer
    it instead of leaving the user with nothing after a long transcription.
    """

    def __init__(self, message: str, original_srt: str):
        super().__init__(message)
        self.original_srt = original_srt


@celery_app.task(bind=True)
def process_video_task(
    self,
    video_path,
    source_lang,
    target_lang,
    auto_create_video,
    whisper_model,
    translation_service="google",
    watermark_config=None,
    initial_timing_summary=None,
    processing_info=None,
    spotting_v2=False,
    translation_v2=False,
    translation_style="clean",
    render_v2=False,
):
    """
    Celery task to process a video file, with detailed, user-facing progress updates.

    The last four arguments are the independent, user-facing subtitle-quality toggles
    (see :mod:`services.subtitle_pipeline`). All four default to OFF, in which case this
    task takes exactly the legacy path it always did:

        spotting_v2       re-spot cues from Whisper WORD timestamps
                          (``subtitle_engine.words_to_cues``) instead of using Whisper's
                          own speech segments.
        translation_v2    translate with whole-scene context + a reading-speed pass
                          (``translation_v2``) instead of the legacy batch translator.
                          Implies transcribe-then-translate rather than the legacy
                          overlapped streaming.
        translation_style ``"clean"`` drops spoken filler, ``"faithful"`` keeps it. Only
                          meaningful while ``translation_v2`` is on.
        render_v2         burn in from a generated ``.ass`` file via FFmpeg's ``ass``
                          filter instead of SRT + the ``subtitles`` filter.
    """
    steps_config = [
        {
            "label": "Audio Processing",
            "subtitle": "Extracting audio stream",
            "weight": 0.05,
            "indeterminate": True,
        },
        {
            "label": "Loading AI Model",
            "subtitle": "Loading AI transcription model",
            "weight": 0.10,
            "indeterminate": True,
        },
        {
            "label": "AI Transcription",
            "subtitle": "Converting speech to text",
            "weight": 0.25,
        },
        {
            "label": "Translation",
            "subtitle": "Processing language conversion",
            "weight": 0.15,
            "indeterminate": True,
        },
        {
            "label": "Creating Subtitle Files",
            "subtitle": "Generating SRT files",
            "weight": 0.05,
            "indeterminate": True,
        },
        {
            "label": "Embedding Subtitles",
            "subtitle": "Adding subtitles to video",
            "weight": 0.35,
        },
        {
            "label": "Finalizing Video",
            "subtitle": "Adding watermark and cleaning up",
            "weight": 0.05,
            "indeterminate": True,
        },
    ]
    progress_manager = ProgressManager(self, steps_config)

    try:
        start_time = time.time()
        task_id = self.request.id

        # Structured logging for task start
        log_task_start(
            logger,
            "process_video",
            task_id=task_id,
            video_path=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
            whisper_model=whisper_model,
            translation_service=translation_service,
        )

        progress_manager.log(f"Starting video processing for {video_path}")

        timing_summary = initial_timing_summary or {}
        raw_base_name = os.path.splitext(os.path.basename(video_path))[0]
        base_name = clean_filename(raw_base_name)
        progress_manager.log(f"Cleaned filename: '{raw_base_name}' -> '{base_name}'")

        progress_manager.set_step_status(0, "in_progress")
        progress_manager.log("Extracting audio stream...", step_index=0)
        time.sleep(1)
        progress_manager.complete_step(0)

        progress_manager.set_step_status(1, "in_progress")
        progress_manager.log("Loading AI transcription model...", step_index=1)
        transcribe_start = time.time()

        def model_loading_callback():
            progress_manager.log("AI Model loaded.", step_index=1)
            progress_manager.complete_step(1)
            progress_manager.set_step_status(2, "in_progress")
            progress_manager.log("Converting speech to text...", step_index=2)

        def transcription_progress_callback(
            step_progress, step_status, overall_progress, current_step, total_steps
        ):
            progress_manager.set_step_progress(
                2, int(overall_progress), message=f"AI processing: {step_status}"
            )

        # === Resolved subtitle-quality flags for this job (all OFF => legacy path) ===
        flags = resolve_flags(
            spotting_v2=spotting_v2,
            translation_v2=translation_v2,
            translation_style=translation_style,
            render_v2=render_v2,
        )
        logger.info(f"Subtitle quality flags [task={task_id}]: {flags_summary(flags)}")

        # Experiment read-outs. Only the translation_v2 path can fill these in; every
        # other path leaves the honest zero rather than an invented number.
        translation_usage_totals = {"tokens": 0, "cost_usd": 0.0}
        cps_over_budget = 0

        wants_translation = bool(target_lang and target_lang != "auto")

        # Gemini transcription returns no word timestamps, so spotting cannot run on it.
        if flags["spotting_v2"] and whisper_model == "gemini":
            logger.warning(
                "spotting_v2 is not available with the gemini model (no word timestamps) "
                "- falling back to legacy segmentation for this job"
            )
            flags["spotting_v2"] = False

        # The v2 stages need transcription and translation to be separate steps; the
        # legacy path deliberately overlaps them.
        use_v2_transcription = flags["spotting_v2"] or (
            flags["translation_v2"] and wants_translation
        )

        if use_v2_transcription:
            # === v2: transcribe fully, then re-spot and/or translate with full context ===
            logger.info(f"Using v2 subtitle pipeline ({flags_summary(flags)})")

            youtube_url = (
                processing_info.get("user_choices", {}).get("url")
                if processing_info
                else None
            )

            transcription_result = transcribe_with_words(
                video_path,
                source_lang=source_lang,
                model_preference=whisper_model,
                progress_callback=transcription_progress_callback,
                model_callback=model_loading_callback,
                youtube_url=youtube_url,
                collect_words=flags["spotting_v2"],
            )
            detected_language = transcription_result["language"]
            transcribe_duration = time.time() - transcribe_start
            timing_summary["transcribe_v2"] = f"{transcribe_duration:.1f}"
            progress_manager.complete_step(2)

            whisper_segments = transcription_result.get("segments") or []
            words = transcription_result.get("words") or []
            if flags["spotting_v2"] and words:
                cues = words_to_cues(words)
                logger.info(
                    f"spotting_v2: {len(words)} words -> {len(cues)} cues "
                    f"(Whisper produced {len(whisper_segments)} segments)"
                )
            else:
                if flags["spotting_v2"]:
                    logger.warning(
                        "spotting_v2: no word timestamps returned - using Whisper segments"
                    )
                cues = whisper_segments
            segments = normalize_cues(cues)

            # Salvage anchor: the SOURCE subtitles are on disk before a single
            # translation token is spent. Translation failing is still a visible job
            # failure (never a silent English substitution), but the user gets back a
            # usable transcript instead of nothing after a long transcription.
            progress_manager.log("Writing source subtitles...", step_index=3)
            original_srt_path = subtitle_service.create_srt_file(
                segments,
                os.path.join(DOWNLOADS_FOLDER, f"{base_name}_original.srt"),
                use_translation=False,
            )
            logger.info(
                f"v2: source SRT written before translation: "
                f"{os.path.basename(original_srt_path)}"
            )

            progress_manager.set_step_status(3, "in_progress")
            if wants_translation:
                translate_start = time.time()
                if flags["translation_v2"]:
                    progress_manager.log(
                        f"Translating with whole-scene context "
                        f"(style={flags['translation_style']})...",
                        step_index=3,
                    )

                    def translation_progress(done, total, message):
                        percent = int(done * 100 / total) if total else 0
                        progress_manager.set_step_progress(3, percent, message=message)
                        progress_manager.log(message, step_index=3)

                    try:
                        translated = translate_cues(
                            segments,
                            target_lang,
                            style=flags["translation_style"],
                            progress_callback=translation_progress,
                        )
                        translation_usage = translated.usage
                        logger.info(f"translation_v2 usage: {translation_usage.as_dict()}")
                        condensed = enforce_cps(
                            translated, progress_callback=translation_progress
                        )
                        logger.info(
                            f"translation_v2 enforce_cps usage: {condensed.usage.as_dict()}"
                        )
                        translation_usage_totals = {
                            "tokens": translation_usage.total_tokens
                            + condensed.usage.total_tokens,
                            "cost_usd": translation_usage.cost_usd
                            + condensed.usage.cost_usd,
                        }
                        segments = normalize_cues(condensed)
                        segments = reflow_dangling_connectors(segments)
                    except Exception as exc:
                        # FAIL VISIBLY — but tell the caller which file survived.
                        logger.error(
                            f"translation_v2 failed after the source SRT was written: {exc}"
                        )
                        raise TranslationFailedWithSalvage(
                            str(exc), os.path.basename(original_srt_path)
                        ) from exc

                    over_budget = [c for c in cps_report(segments) if not c["ok"]]
                    cps_over_budget = len(over_budget)
                    logger.info(
                        f"translation_v2 reading speed: {cps_over_budget}/"
                        f"{len(segments)} cues still over 17 CPS / 84 chars"
                    )
                else:
                    progress_manager.log("Translating cues...", step_index=3)
                    segments = normalize_cues(
                        translate_segments(
                            segments, target_lang, service=translation_service
                        )
                    )
                timing_summary["translate_v2"] = f"{time.time() - translate_start:.1f}"
            else:
                progress_manager.log("Skipping translation.", step_index=3)
            progress_manager.complete_step(3)

            logger.info(
                f"v2 pipeline complete: {len(segments)} cues | {transcribe_duration:.1f}s "
                f"transcription | {detected_language} -> {target_lang or detected_language}"
            )
        # P1: Use pipeline overlap if translation is needed, otherwise use standard transcription
        elif wants_translation:
            # === P1 Pipeline Overlap: Transcribe + Translate Simultaneously ===
            logger.info("Using P1 pipeline overlap (transcribe + translate simultaneously)")

            # Extract YouTube URL for Gemini support (if available)
            youtube_url = processing_info.get("user_choices", {}).get("url") if processing_info else None
            logger.info(f"DEBUG: Extracted youtube_url = {youtube_url}")
            logger.info(f"DEBUG: whisper_model = {whisper_model}")

            transcription_result = transcribe_and_translate_streamed(
                video_path,
                target_language=target_lang,
                source_lang=source_lang,
                model_preference=whisper_model,
                translation_service=translation_service,
                progress_callback=transcription_progress_callback,
                model_callback=model_loading_callback,
                youtube_url=youtube_url,
            )
            segments = transcription_result["segments"]
            detected_language = transcription_result["language"]
            transcribe_duration = time.time() - transcribe_start
            timing_summary["transcribe_and_translate"] = f"{transcribe_duration:.1f}"

            # Both steps completed together
            progress_manager.complete_step(2)
            progress_manager.set_step_status(3, "in_progress")
            progress_manager.complete_step(3)

            logger.info(
                f"P1 Pipeline complete: {len(segments)} segments | {transcribe_duration:.1f}s total | "
                f"{detected_language} -> {target_lang}"
            )
        else:
            # === Standard Sequential Flow: Transcribe only (no translation) ===
            logger.info("Using standard transcription (no translation needed)")

            transcription_result = transcribe_video(
                video_path,
                source_lang=source_lang,
                model_preference=whisper_model,
                progress_callback=transcription_progress_callback,
                model_callback=model_loading_callback,
            )
            segments = transcription_result["segments"]
            detected_language = transcription_result["language"]
            transcribe_duration = time.time() - transcribe_start
            timing_summary["transcribe_video"] = f"{transcribe_duration:.1f}"
            progress_manager.complete_step(2)

            logger.info(
                f"Transcription completed: {len(segments)} segments | {transcribe_duration:.1f}s | {detected_language} detected"
            )

            # Skip translation step
            progress_manager.set_step_status(3, "in_progress")
            progress_manager.log("Skipping translation.", step_index=3)
            progress_manager.complete_step(3)

        progress_manager.set_step_status(4, "in_progress")
        progress_manager.log("Creating subtitle files...", step_index=4)
        # Rewritten unconditionally: the v2 branch already wrote this file before
        # translation as a salvage anchor, and `segments` may have been reflowed
        # since. Same path, same content for the source text.
        original_srt_path = subtitle_service.create_srt_file(
            segments,
            os.path.join(DOWNLOADS_FOLDER, f"{base_name}_original.srt"),
            use_translation=False,
        )
        translated_srt_path = subtitle_service.create_srt_file(
            segments,
            os.path.join(DOWNLOADS_FOLDER, f"{base_name}_translated.srt"),
            use_translation=True,
            language=target_lang or detected_language,
        )
        progress_manager.complete_step(4)

        final_video_path = None
        progress_manager.set_step_status(5, "in_progress")
        progress_manager.log(
            f"Rendering {len(segments)} cues into the video "
            f"({'ASS/render_v2' if flags['render_v2'] else 'SRT/legacy'})...",
            step_index=5,
        )
        if auto_create_video:
            video_creation_start = time.time()
            video_with_subtitles_path = os.path.join(
                DOWNLOADS_FOLDER, f"{base_name}_with_subtitles.mp4"
            )

            def video_progress_callback(current_progress):
                progress_manager.set_step_progress(
                    5,
                    current_progress,
                    message=f"Embedding subtitles... {current_progress}%",
                )

            # Check if watermark is enabled BEFORE creating video
            watermark_enabled = watermark_config and watermark_config.get(
                "enabled", False
            )

            if watermark_enabled:
                # Get watermark path (custom or default)
                watermark_path = watermark_config.get(
                    "custom_logo_path"
                ) or config.WATERMARK_PATHS.get("default", "/app/assets/logo.png")

                # Map frontend position to backend position format
                position_map = {
                    "top-right": ("upper_right", "comfortable"),
                    "top-left": ("left", "top"),
                    "bottom-right": ("right", "bottom"),
                    "bottom-left": ("left", "bottom"),
                }
                position = position_map.get(
                    watermark_config.get("position", "bottom-right"),
                    ("right", "bottom"),
                )

                # Map size to height
                size_map = {"small": 60, "medium": 80, "large": 120}
                size_height = size_map.get(watermark_config.get("size", "medium"), 80)

                # Convert opacity from 0-100 to 0.0-1.0 for FFmpeg
                opacity_float = watermark_config.get("opacity", 40) / 100.0

                # Use combined function for better performance
                final_video_path_output = os.path.join(
                    DOWNLOADS_FOLDER, f"{base_name}_final.mp4"
                )

                if flags["render_v2"]:
                    progress_manager.log(
                        "Creating video with ASS subtitles and watermark (render_v2)...",
                        step_index=5,
                    )
                    video_creation_success = subtitle_service.create_video_with_ass(
                        video_path,
                        segments,
                        final_video_path_output,
                        target_language=target_lang or detected_language,
                        use_translation=wants_translation,
                        watermark_path=watermark_path,
                        watermark_position=position,
                        watermark_size_height=size_height,
                        watermark_opacity=opacity_float,
                        progress_callback=video_progress_callback,
                    )
                else:
                    progress_manager.log("Creating video with subtitles and watermark (combined)...", step_index=5)

                    video_creation_success = subtitle_service.create_video_with_subtitles_and_watermark(
                        video_path,
                        translated_srt_path,
                        final_video_path_output,
                        watermark_path,
                        target_lang or detected_language,
                        watermark_position=position,
                        watermark_size_height=size_height,
                        watermark_opacity=opacity_float,
                        progress_callback=video_progress_callback,
                    )

                final_video_path = final_video_path_output if video_creation_success else None

                timing_summary["create_video_with_subtitles_and_watermark"] = (
                    f"{time.time() - video_creation_start:.1f}"
                )
                # Mark both steps as complete since we did them combined
                progress_manager.complete_step(5)
                progress_manager.set_step_status(6, "completed")
                progress_manager.complete_step(6)

            else:
                # No watermark
                if flags["render_v2"]:
                    progress_manager.log(
                        "Creating video with ASS subtitles (render_v2)...", step_index=5
                    )
                    video_creation_success = subtitle_service.create_video_with_ass(
                        video_path,
                        segments,
                        video_with_subtitles_path,
                        target_language=target_lang or detected_language,
                        use_translation=wants_translation,
                        progress_callback=video_progress_callback,
                    )
                else:
                    # use original function
                    video_creation_success = subtitle_service.create_video_with_subtitles(
                        video_path,
                        translated_srt_path,
                        video_with_subtitles_path,
                        target_lang or detected_language,
                        progress_callback=video_progress_callback,
                    )

                progress_manager.log("Finalizing video...", step_index=5)
                progress_manager.set_step_progress(5, 99)
                time.sleep(1)

                timing_summary["create_video_with_subtitles"] = (
                    f"{time.time() - video_creation_start:.1f}"
                )
                progress_manager.complete_step(5)

                progress_manager.set_step_status(6, "in_progress")

                # Check if video creation was successful before proceeding
                if not video_creation_success:
                    progress_manager.log("Video with subtitles creation failed...", step_index=6)
                    final_video_path = None
                    timing_summary["add_watermark"] = "0.0 (failed)"
                    progress_manager.complete_step(6)
                else:
                    progress_manager.log(
                        "Skipping watermark (disabled by user)...", step_index=6
                    )
                    # Just rename the file to final without watermark
                    final_video_path_output = os.path.join(
                        DOWNLOADS_FOLDER, f"{base_name}_final.mp4"
                    )
                    shutil.copy2(video_with_subtitles_path, final_video_path_output)
                    final_video_path = final_video_path_output
                    timing_summary["add_watermark"] = "0.0 (skipped)"
                    progress_manager.complete_step(6)

            # Clean up intermediate files only if we used the two-step process
            if (
                final_video_path
                and os.path.exists(final_video_path)
                and not watermark_enabled  # Only cleanup if we didn't use combined function
                and os.path.exists(video_with_subtitles_path)
            ):
                try:
                    os.remove(video_with_subtitles_path)
                except OSError as e:
                    progress_manager.log(
                        f"Error removing intermediate file: {e.strerror}"
                    )
        else:
            progress_manager.log("Skipping video creation as per user request.")
            progress_manager.complete_step(5)
            progress_manager.complete_step(6)

        files_result = {
            "original_srt": os.path.basename(original_srt_path),
            "translated_srt": os.path.basename(translated_srt_path),
            "video_with_subtitles": (
                os.path.basename(final_video_path) if final_video_path else None
            ),
        }

        final_result = {
            "title": base_name,
            "detected_language": detected_language,
            "files": files_result,
            "timing_summary": timing_summary,
            "progress": progress_manager.steps,
        }

        if processing_info:
            # Support both video_metadata (YouTube) and file_metadata (uploads)
            final_result["video_metadata"] = processing_info.get("video_metadata") or processing_info.get("file_metadata", {})
            final_result["user_choices"] = processing_info.get("user_choices", {})

        # The four subtitle-quality toggles are part of what the user chose, and the
        # experiment they belong to is only measurable if each result says which
        # settings produced it. Reported from the RESOLVED flags, not the raw kwargs,
        # so e.g. the gemini/spotting_v2 fallback is visible here too.
        final_result.setdefault("user_choices", {})
        final_result["user_choices"] = {**final_result["user_choices"], **flags}

        # Structured logging for task completion
        duration = time.time() - start_time
        log_task_complete(
            logger,
            "process_video",
            duration=duration,
            task_id=task_id,
            detected_language=detected_language,
            files_created=len([f for f in files_result.values() if f]),
        )

        # Save statistics to Redis
        try:
            # Extract video metadata
            video_metadata = final_result.get("video_metadata", {})
            video_duration = video_metadata.get("duration", 0)

            # Get video size
            video_size_mb = 0
            if os.path.exists(video_path):
                video_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            # Calculate transcription metrics
            transcription_duration = transcribe_duration if 'transcribe_duration' in locals() else 0
            transcription_speed_ratio = 0
            if transcription_duration > 0 and video_duration > 0:
                transcription_speed_ratio = video_duration / transcription_duration

            # Get translation duration if available
            translation_duration = 0
            if "translate" in timing_summary:
                try:
                    translation_duration = float(timing_summary["translate"])
                except (ValueError, TypeError):
                    translation_duration = 0

            # Get embedding duration if available
            embedding_duration = 0
            if "embed_subtitles" in timing_summary:
                try:
                    embedding_duration = float(timing_summary["embed_subtitles"])
                except (ValueError, TypeError):
                    embedding_duration = 0

            # Build stats dictionary
            stats = {
                "task_id": task_id,
                "video_url": video_metadata.get("url", "uploaded"),
                "video_duration": video_duration,
                "video_size_mb": round(video_size_mb, 2),
                "transcription_model": whisper_model,
                "transcription_duration": round(transcription_duration, 2),
                "transcription_speed_ratio": round(transcription_speed_ratio, 2),
                "translation_service": translation_service,
                "translation_duration": round(translation_duration, 2),
                # Real numbers on the translation_v2 path (TokenUsage is accumulated
                # across translate_cues + enforce_cps). The legacy translators do not
                # report usage at all, so those runs still record zero rather than a
                # made-up figure.
                "translation_tokens": translation_usage_totals["tokens"],
                "translation_cost_usd": round(translation_usage_totals["cost_usd"], 6),
                "embedding_duration": round(embedding_duration, 2),
                "total_duration": round(duration, 2),
                "status": "success",
                "error_message": None,
                # The experiment's independent variables + one quality read-out, so a
                # run can be attributed to the settings that produced it.
                "spotting_v2": flags["spotting_v2"],
                "translation_v2": flags["translation_v2"],
                "translation_style": flags["translation_style"],
                "render_v2": flags["render_v2"],
                "subtitle_pipeline_v2": any_enabled(flags),
                "cues": len(segments),
                "cps_over_budget": cps_over_budget,
            }

            # Save to Redis
            save_video_stats(stats)
            logger.info(f"Stats saved for task {task_id[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save stats (non-critical): {e}")
            # Don't fail the task if stats saving fails

        return {"status": "SUCCESS", "result": final_result}
    except Exception as e:
        import traceback

        error_msg = f"Error in process_video_task: {e}\n{traceback.format_exc()}"
        progress_manager.log(error_msg)

        # Structured logging for task error
        log_task_error(logger, "process_video", e, task_id=task_id)

        for i, step in enumerate(progress_manager.steps):
            if step["status"] == "in_progress":
                progress_manager.set_step_error(i, str(e))
                break

        failure = {"status": "FAILURE", "error": str(e)}
        if isinstance(e, TranslationFailedWithSalvage):
            # Still a failure — but the transcription was expensive and its output
            # is already on disk, so hand it over instead of throwing it away.
            failure["files"] = {"original_srt": e.original_srt}
            failure["salvaged"] = True
            logger.info(
                f"Translation failed; offering salvaged source SRT {e.original_srt}"
            )
        return failure


@celery_app.task(bind=True)
def create_video_with_subtitles_from_segments(
    self, video_path, segments, base_filename
):
    """Create video with subtitles from provided segments"""
    try:
        self.update_state(state="PROGRESS", meta={"status": "Processing subtitles..."})

        srt_filename = f"{base_filename}_subtitles.srt"
        srt_path = os.path.join(DOWNLOADS_FOLDER, srt_filename)
        subtitle_service.create_srt_file(
            segments, srt_path, use_translation=True, language="he"
        )

        self.update_state(
            state="PROGRESS", meta={"status": "Creating video with subtitles..."}
        )

        temp_video_with_subs = os.path.join(
            DOWNLOADS_FOLDER, f"{base_filename}_temp_subs.mp4"
        )
        final_video = os.path.join(
            DOWNLOADS_FOLDER, f"{base_filename}_with_subtitles.mp4"
        )

        target_lang = "he"

        if subtitle_service.create_video_with_subtitles(
            video_path, srt_path, temp_video_with_subs, target_lang
        ):
            self.update_state(state="PROGRESS", meta={"status": "Adding watermark..."})
            watermark_path = config.WATERMARK_PATHS.get(
                "default", "/app/assets/logo.png"
            )

            if subtitle_service.add_watermark_to_video(
                input_video_path=temp_video_with_subs,
                watermark_path=watermark_path,
                output_video_path=final_video,
                position=("upper_right", "comfortable"),
                opacity=0.45,
                size_height=config.DEFAULT_LOGO_SIZE,
            ):
                video_with_subtitles = os.path.basename(final_video)
                if os.path.exists(temp_video_with_subs):
                    os.remove(temp_video_with_subs)
            else:
                video_with_subtitles = os.path.basename(temp_video_with_subs)

            return {
                "status": "SUCCESS",
                "title": base_filename,
                "files": {
                    "srt": srt_filename,
                    "video_with_subtitles": video_with_subtitles,
                },
            }
        else:
            return {
                "status": "FAILURE",
                "error": "Failed to create video with subtitles",
            }

    except Exception as e:
        import traceback

        error_msg = f"Task failed: {str(e)}"
        traceback_msg = traceback.format_exc()
        logger.error(f"{error_msg}\n{traceback_msg}")
        return {"status": "FAILURE", "error": error_msg, "traceback": traceback_msg}
