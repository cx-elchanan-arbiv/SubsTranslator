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
from services.research_recorder import start_run
from services.stats_service import save_video_stats
from services.subtitle_engine import (
    absorb_dropped_time,
    drop_hallucinated_cues,
    layout_params,
    reflow_dangling_connectors,
    words_to_cues,
)
from services.subtitle_pipeline import (
    any_enabled,
    flags_summary,
    normalize_cues,
    parse_glossary,
    parse_subtitle_position,
    resolve_flags,
)
from services.subtitle_service import subtitle_service
from services.transcription_service import (
    transcribe_and_translate_streamed,
    transcribe_video,
    transcribe_with_words,
    translate_segments,
)
from services.translation_v2 import cps_report, enforce_cps, translate_cues
from services.whisper_smart import smart_whisper
from utils.file_utils import clean_filename
from utils.video_utils import get_video_duration, relative_energy_probe

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


#: What the user is shown when the burn-in fails. The task result carries it as
#: ``user_facing_message``, which is the field ``/status/<task_id>`` hands to the UI —
#: without it the UI falls back to its generic "Processing failed" string.
RENDER_FAILED_USER_MESSAGE = (
    "Subtitles could not be burned into the video. The subtitle files (.srt) were "
    "created successfully and can still be downloaded."
)


def _render_produced_a_file(reported_success, output_path) -> bool:
    """Did the render really produce a video, or did it only claim to?

    The renderers return a bool derived from FFmpeg's exit status, which cannot tell
    "wrote a playable video" apart from "exited 0 and wrote nothing". The path this
    guards is handed straight to the user as a download link, so the file existing and
    being non-empty is the only part of the claim worth trusting. Anything the renderer
    calls a failure stays a failure regardless.
    """
    if not reported_success:
        return False
    try:
        return os.path.getsize(output_path) > 0
    except OSError:
        return False


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
    subtitle_position="bottom",
):
    """
    Celery task to process a video file, with detailed, user-facing progress updates.

    The quality arguments are independent, user-facing subtitle improvements
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
        subtitle_position place burned-in subtitles at the bottom, top or right side;
                          defaults to the historical bottom-centre placement.
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
    start_time = time.time()
    task_id = self.request.id

    # Full-run archive for corpus analysis. `start_run` never raises and always returns
    # a usable object (an inert stand-in when recording is off or the directory could
    # not be made), so nothing below has to branch on it. See services/research_recorder.
    recorder = start_run(task_id)
    recorder.update_meta(
        whisper_model=whisper_model,
        translation_service=translation_service,
        source_lang=source_lang,
        target_lang=target_lang,
        auto_create_video=bool(auto_create_video),
        watermark_enabled=bool((watermark_config or {}).get("enabled", False)),
    )
    recorder.record_source_video(video_path)

    try:
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
        subtitle_position = parse_subtitle_position(subtitle_position)
        logger.info(f"Subtitle quality flags [task={task_id}]: {flags_summary(flags)}")
        recorder.update_meta(**flags, subtitle_position=subtitle_position)

        # Experiment read-outs. Only the translation_v2 path can fill these in.
        #
        # `cps_over_budget` starts as None, NOT 0, and that distinction is the whole
        # point: 0 means "measured, and nothing was over budget" — a perfect score. A
        # legacy run never measures reading speed at all, and recording its silence as a
        # perfect score made every legacy row in the corpus index read as a win over the
        # v2 rows that honestly reported their misses. Absent and zero are different
        # claims and the archive has to be able to tell them apart.
        translation_usage_totals = {"tokens": 0, "cost_usd": 0.0}
        cps_over_budget = None
        translation_mode = None

        wants_translation = bool(target_lang and target_lang != "auto")

        # Optional terminology glossary. No UI surfaces it yet, so it is empty on every
        # job today and the prompt is byte-identical to before — the plumbing exists so
        # the UI can be wired without touching the pipeline again, and so a researcher
        # can drive it from an enqueued task right now.
        glossary = parse_glossary(
            (processing_info or {}).get("user_choices", {}).get("glossary")
        )
        if glossary:
            recorder.update_meta(glossary=glossary)

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

        # Layout is derived once per job and threaded through every stage below. It is
        # None on the legacy path, where the renderer derives its own.
        layout = None

        if use_v2_transcription:
            # === v2: transcribe fully, then re-spot and/or translate with full context ===
            logger.info(f"Using v2 subtitle pipeline ({flags_summary(flags)})")

            # The frame's DIMENSIONS decide the character budget, so they have to be
            # known before a single cue is spotted or a single translation token is
            # spent — not at render time, which is where they used to be read. On a
            # 720x1280 portrait clip the budget is 66 characters per cue, not 84;
            # asking the model for 84 there buys text the frame cannot draw, and
            # `WrapStyle: 2` renders the surplus off the edge of the picture rather
            # than wrapping it. Probing never raises: it falls back to 1920x1080, i.e.
            # to the landscape numbers that predate this.
            video_w, video_h = subtitle_service.probe_video_dimensions(video_path)
            layout = layout_params(video_w, video_h)
            # The end of the picture. Every timing rule below is bounded by it: a cue
            # that ends after the video does is time that does not exist.
            video_duration = get_video_duration(video_path) or 0.0
            logger.info(
                f"Subtitle layout [task={task_id}]: {video_w}x{video_h} -> "
                f"font {layout['font_px']}px, {layout['max_line_chars']} chars/line, "
                f"{layout['max_chars_per_cue']} chars/cue"
            )
            recorder.update_meta(layout=layout, video_duration_probe=video_duration)

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
            asr_primed = bool(transcription_result.get("asr_primed"))
            recorder.update_meta(
                detected_language=detected_language,
                pipeline="v2",
                asr_primed=asr_primed,
            )
            recorder.add_timing("transcribe", transcribe_duration)
            recorder.save_segments(whisper_segments)
            recorder.save_words(words)
            if flags["spotting_v2"] and words:
                cues = words_to_cues(
                    words,
                    max_line=layout["max_line_chars"],
                    max_lines=layout["max_lines"],
                    asr_primed=asr_primed,
                    video_duration=video_duration or None,
                )
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

            # Hallucination gate. Whisper occasionally invents a whole sentence and
            # stamps it onto a fraction of a second of near-silence; until now those
            # were translated at full price and burned into the picture, where a
            # fluent Hebrew fabrication is far harder to spot than an English one.
            # Runs on the SOURCE, before translation: the CPS signature only means
            # something in the language that was spoken, dropping early costs no
            # tokens, and neighbouring cues never receive a fabricated sentence as
            # scene context.
            # The audio itself gets a vote. Whisper mis-TIMES real speech constantly
            # (cross-talk compresses its word timestamps), and the only thing that
            # separates "spoken too fast to be plausible" from "never spoken at all" is
            # whether anyone was making a noise at that moment. Built lazily: the
            # baseline measurement is not taken unless a cue is actually in question.
            segments, dropped_cues = drop_hallucinated_cues(
                segments, energy_probe=relative_energy_probe(video_path)
            )
            if dropped_cues:
                logger.warning(
                    f"hallucination gate [task={task_id}]: dropped "
                    f"{len(dropped_cues)} source cue(s) as probable ASR hallucination "
                    f"before translation"
                )
                recorder.save_dropped_cues(dropped_cues, reason="hallucination_gate_v2")
                # No holes: the previous cue holds the screen over the freed time
                # instead of the picture going bare.
                segments = absorb_dropped_time(segments, dropped_cues)

            recorder.save_cues("pre_translation", segments)

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

                    # Text the gate refused to SHIP is still text the model should
                    # READ: a dropped cue vanishing from the prompt entirely is how a
                    # translator loses the antecedent of the next line's pronoun. These
                    # ride along marked [CONTEXT-ONLY] and are filtered back out below.
                    context_carriers = [
                        dict(cue, context_only=True)
                        for cue in dropped_cues
                        if cue.get("context_only")
                    ]
                    translate_input = segments
                    if context_carriers:
                        translate_input = sorted(
                            list(segments) + context_carriers,
                            key=lambda cue: float(cue.get("start") or 0),
                        )
                        logger.info(
                            f"translation_v2: {len(context_carriers)} dropped cue(s) "
                            f"handed to the model as [CONTEXT-ONLY]"
                        )
                    try:
                        translated = translate_cues(
                            translate_input,
                            target_lang,
                            style=flags["translation_style"],
                            max_chars_per_cue=layout["max_chars_per_cue"],
                            glossary=glossary,
                            progress_callback=translation_progress,
                            recorder=recorder,
                            source_lang=detected_language,
                        )
                        translation_mode = getattr(translated, "mode", "translate")
                        translation_usage = translated.usage
                        logger.info(
                            f"translation_v2 usage ({translation_mode}): "
                            f"{translation_usage.as_dict()}"
                        )
                        condensed = enforce_cps(
                            translated,
                            max_chars_per_cue=layout["max_chars_per_cue"],
                            progress_callback=translation_progress,
                            recorder=recorder,
                            video_duration=video_duration or None,
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
                        segments = [
                            cue
                            for cue in normalize_cues(condensed)
                            if not cue.get("context_only")
                        ]
                        recorder.save_cues("post_translation", segments)
                        segments = reflow_dangling_connectors(
                            segments, max_chars=layout["max_chars_per_cue"]
                        )
                        recorder.save_cues("post_reflow", segments)
                        recorder.update_meta(
                            translation_mode=translation_mode,
                            context_only_cues=len(context_carriers),
                            translation_tokens=translation_usage_totals["tokens"],
                            translation_cost_usd=round(
                                translation_usage_totals["cost_usd"], 6
                            ),
                            translation_usage={
                                "translate_cues": translation_usage.as_dict(),
                                "enforce_cps": condensed.usage.as_dict(),
                            },
                        )
                    except Exception as exc:
                        # FAIL VISIBLY — but tell the caller which file survived.
                        logger.error(
                            f"translation_v2 failed after the source SRT was written: {exc}"
                        )
                        raise TranslationFailedWithSalvage(
                            str(exc), os.path.basename(original_srt_path)
                        ) from exc

                    over_budget = [
                        c
                        for c in cps_report(
                            segments, max_chars_per_cue=layout["max_chars_per_cue"]
                        )
                        if not c["ok"]
                    ]
                    cps_over_budget = len(over_budget)
                    logger.info(
                        f"translation_v2 reading speed: {cps_over_budget}/"
                        f"{len(segments)} cues still over 17 CPS / "
                        f"{layout['max_chars_per_cue']} chars"
                    )
                else:
                    progress_manager.log("Translating cues...", step_index=3)
                    # Same contract as the translation_v2 branch above: a translation
                    # that did not happen is a FAILED job, never an untranslated .srt
                    # shipped under a green tick. The legacy translator used to answer
                    # a failure by copying the source text into `translated_text` and
                    # returning normally, which is indistinguishable from success here.
                    # Written to hold either way: if that silent fallback is still in
                    # place this code is unreachable, and the moment it raises — with
                    # TranslationFailedWithSalvage or anything else — the failure is
                    # reported with the source SRT that was written before translation.
                    try:
                        segments = normalize_cues(
                            translate_segments(
                                segments, target_lang, service=translation_service
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            f"legacy translation failed after the source SRT was "
                            f"written: {exc}"
                        )
                        raise TranslationFailedWithSalvage(
                            str(exc), os.path.basename(original_srt_path)
                        ) from exc
                    recorder.save_cues("post_translation", segments)
                timing_summary["translate_v2"] = f"{time.time() - translate_start:.1f}"
                recorder.add_timing("translate", time.time() - translate_start)
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
            logger.info(
                "Using P1 pipeline overlap (transcribe + translate simultaneously)"
            )

            # Extract YouTube URL for Gemini support (if available)
            youtube_url = (
                processing_info.get("user_choices", {}).get("url")
                if processing_info
                else None
            )
            logger.info(f"DEBUG: Extracted youtube_url = {youtube_url}")
            logger.info(f"DEBUG: whisper_model = {whisper_model}")

            try:
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
            except TranslationFailedWithSalvage as exc:
                # This path overlaps transcription and translation, so unlike the v2
                # branch there is no source SRT on disk to fall back to: when the
                # translator gives up, the transcription that was already paid for is
                # inside the exception or nowhere. Write it out here, then let the
                # failure through — the alternative, which is what used to happen, is
                # `translated_text = text` and an untranslated video reported as a win.
                # Defensive by design: the raise may not exist yet, and when it does it
                # may hand over a filename it wrote itself or the salvaged segments.
                salvage_name = getattr(exc, "original_srt", None)
                salvaged_segments = getattr(exc, "segments", None)
                if not salvage_name and salvaged_segments:
                    salvage_name = os.path.basename(
                        subtitle_service.create_srt_file(
                            normalize_cues(salvaged_segments),
                            os.path.join(DOWNLOADS_FOLDER, f"{base_name}_original.srt"),
                            use_translation=False,
                        )
                    )
                    logger.info(
                        f"legacy translation failed; salvaged transcript written to "
                        f"{salvage_name}"
                    )
                raise TranslationFailedWithSalvage(
                    str(exc), salvage_name or ""
                ) from exc
            segments = transcription_result["segments"]
            detected_language = transcription_result["language"]
            transcribe_duration = time.time() - transcribe_start
            timing_summary["transcribe_and_translate"] = f"{transcribe_duration:.1f}"

            # The corpus needs legacy runs too — a v1-vs-v2 comparison is exactly what
            # it is for. The legacy path overlaps transcription and translation, so
            # there is one combined timing and no separate pre-translation cue list.
            recorder.update_meta(
                detected_language=detected_language, pipeline="legacy_p1"
            )
            recorder.add_timing("transcribe_and_translate", transcribe_duration)
            recorder.save_segments(segments)

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

            recorder.update_meta(
                detected_language=detected_language, pipeline="legacy_transcribe_only"
            )
            recorder.add_timing("transcribe", transcribe_duration)
            recorder.save_segments(segments)

            logger.info(
                f"Transcription completed: {len(segments)} segments | {transcribe_duration:.1f}s | {detected_language} detected"
            )

            # Skip translation step
            progress_manager.set_step_status(3, "in_progress")
            progress_manager.log("Skipping translation.", step_index=3)
            progress_manager.complete_step(3)

        # The model that ACTUALLY transcribed this job, which is not necessarily the one
        # the user asked for: whisper_smart downgrades on low memory, on a load failure,
        # on a transcribe exception, on a Gemini fallback and on "small". Every one of
        # those used to be invisible from here, because the stats below recorded
        # `whisper_model` — the REQUEST — as if it were the result. That is why Redis
        # holds 104 records under `stats:index:model:large` that cannot be trusted to
        # describe runs of `large`. Read the transcriber's own report first, fall back
        # to what the model manager last loaded, and only then to the request.
        model_actually_used = (
            transcription_result.get("model_used")
            or getattr(smart_whisper, "last_model_used", None)
            or whisper_model
        )
        if model_actually_used != whisper_model:
            logger.warning(
                f"Transcription model downgrade [task={task_id}]: '{whisper_model}' "
                f"was requested but '{model_actually_used}' actually ran"
            )
        recorder.update_meta(
            transcription_model_requested=whisper_model,
            transcription_model_used=model_actually_used,
        )

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
        recorder.copy_output(original_srt_path)
        recorder.copy_output(translated_srt_path)

        final_video_path = None
        # Set only when the burn-in / watermark step really failed. It is what turns
        # this run into a reported FAILURE at the end instead of a green job with a
        # download button that silently is not there.
        render_error = None
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
                        layout=layout,
                        recorder=recorder,
                        subtitle_position=subtitle_position,
                    )
                else:
                    progress_manager.log(
                        "Creating video with subtitles and watermark (combined)...",
                        step_index=5,
                    )

                    video_creation_success = (
                        subtitle_service.create_video_with_subtitles_and_watermark(
                            video_path,
                            translated_srt_path,
                            final_video_path_output,
                            watermark_path,
                            target_lang or detected_language,
                            watermark_position=position,
                            watermark_size_height=size_height,
                            watermark_opacity=opacity_float,
                            progress_callback=video_progress_callback,
                            subtitle_position=subtitle_position,
                        )
                    )

                timing_summary["create_video_with_subtitles_and_watermark"] = (
                    f"{time.time() - video_creation_start:.1f}"
                )
                # One call did the work of both steps, so both stand or fall with it.
                # This branch used to complete BOTH unconditionally, with no success
                # check at all — a failed combined render was displayed as "Embedding
                # Subtitles - completed 100%" and "Finalizing Video - completed 100%".
                if _render_produced_a_file(
                    video_creation_success, final_video_path_output
                ):
                    final_video_path = final_video_path_output
                    progress_manager.complete_step(5)
                    progress_manager.set_step_status(6, "in_progress")
                    progress_manager.complete_step(6)
                else:
                    final_video_path = None
                    render_error = (
                        "Burning the subtitles and watermark into the video failed - "
                        f"no usable {os.path.basename(final_video_path_output)} was "
                        f"produced."
                    )
                    progress_manager.set_step_error(5, render_error)
                    progress_manager.set_step_error(6, render_error)

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
                        layout=layout,
                        recorder=recorder,
                        subtitle_position=subtitle_position,
                    )
                else:
                    # use original function
                    video_creation_success = (
                        subtitle_service.create_video_with_subtitles(
                            video_path,
                            translated_srt_path,
                            video_with_subtitles_path,
                            target_lang or detected_language,
                            progress_callback=video_progress_callback,
                            subtitle_position=subtitle_position,
                        )
                    )

                timing_summary["create_video_with_subtitles"] = (
                    f"{time.time() - video_creation_start:.1f}"
                )

                # Verify BEFORE completing the step. The old order ran
                # complete_step(5) first and only then asked whether the burn had
                # worked, and its failure branch went on to call complete_step(6) as
                # well — so a run that produced no video at all still reported both
                # steps completed, 100% overall and no step in an error state.
                if not _render_produced_a_file(
                    video_creation_success, video_with_subtitles_path
                ):
                    final_video_path = None
                    render_error = (
                        "Burning the subtitles into the video failed - no usable "
                        f"{os.path.basename(video_with_subtitles_path)} was produced."
                    )
                    progress_manager.set_step_error(5, render_error)
                    timing_summary["add_watermark"] = "0.0 (not reached)"
                else:
                    progress_manager.log("Finalizing video...", step_index=5)
                    progress_manager.set_step_progress(5, 99)
                    time.sleep(1)
                    progress_manager.complete_step(5)

                    progress_manager.set_step_status(6, "in_progress")
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

        # Archive the produced artefacts. The .ass sits next to the rendered MP4 and is
        # the only record of the exact geometry the frame was drawn with, so it is worth
        # more to the corpus than the video itself.
        if final_video_path:
            recorder.copy_output(os.path.splitext(final_video_path)[0] + ".ass")
            recorder.copy_output(final_video_path)

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
            final_result["video_metadata"] = processing_info.get(
                "video_metadata"
            ) or processing_info.get("file_metadata", {})
            final_result["user_choices"] = processing_info.get("user_choices", {})

        # The four subtitle-quality toggles are part of what the user chose, and the
        # experiment they belong to is only measurable if each result says which
        # settings produced it. Reported from the RESOLVED flags, not the raw kwargs,
        # so e.g. the gemini/spotting_v2 fallback is visible here too.
        final_result.setdefault("user_choices", {})
        final_result["user_choices"] = {**final_result["user_choices"], **flags}
        final_result["user_choices"]["subtitle_position"] = subtitle_position

        # Structured logging. A run whose video never rendered is not a completion, and
        # logging it as one is how this failure stayed invisible in the logs too.
        duration = time.time() - start_time
        if render_error:
            log_task_error(
                logger,
                "process_video",
                RuntimeError(render_error),
                task_id=task_id,
            )
        else:
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
            transcription_duration = (
                transcribe_duration if "transcribe_duration" in locals() else 0
            )
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
                # The model that RAN, not the one that was asked for — see
                # `model_actually_used` above. `transcription_model` keeps its name
                # because stats_service indexes on it (stats:index:model:<name>), so
                # that index now means what it always claimed to mean.
                "transcription_model": model_actually_used,
                "transcription_model_requested": whisper_model,
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
                # Was hardcoded "success" with a null error, which meant a run that
                # produced no video was archived, indexed and charted as a win.
                "status": "failure" if render_error else "success",
                "error_message": render_error,
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

        recorder.update_meta(
            cues=len(segments),
            cps_over_budget=cps_over_budget,
            timing_summary=timing_summary,
            files=files_result,
        )
        recorder.add_timing("total", duration)

        if render_error:
            # The job failed, and says so. The transcript and both SRT files are real,
            # already on disk and handed over here rather than discarded — but a run
            # that produced no video is not a SUCCESS, and returning one is what left
            # the user staring at a finished-looking job whose download button had
            # quietly vanished with no error anywhere.
            recorder.update_meta(render_error=render_error)
            recorder.finish(success=False, error=render_error)
            logger.error(f"Video rendering failed [task={task_id}]: {render_error}")
            return {
                "status": "FAILURE",
                "code": "RENDER_FAILED",
                "error": render_error,
                "message": render_error,
                "user_facing_message": RENDER_FAILED_USER_MESSAGE,
                "recoverable": True,
                "files": files_result,
                "salvaged": True,
            }

        recorder.finish(success=True)

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
        if isinstance(e, TranslationFailedWithSalvage) and e.original_srt:
            # Still a failure — but the transcription was expensive and its output
            # is already on disk, so hand it over instead of throwing it away.
            # Guarded on the filename: the legacy overlapped path can fail before any
            # SRT exists, and advertising a file that was never written would send the
            # UI to a download that 404s.
            failure["files"] = {"original_srt": e.original_srt}
            failure["salvaged"] = True
            logger.info(
                f"Translation failed; offering salvaged source SRT {e.original_srt}"
            )

        # A failed run is archived too — with its prompts, its partial cue lists and
        # the exception. Those are the most informative rows in the corpus.
        recorder.finish(success=False, error=f"{type(e).__name__}: {e}")
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
