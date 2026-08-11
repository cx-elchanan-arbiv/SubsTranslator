"""
Transcription and translation service for SubsTranslator
Handles video transcription and subtitle translation with various AI models
"""

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from config import get_config
from core.exceptions import (
    AudioExtractionError,
    FFmpegProcessError,
    FFmpegTimeoutError,
    TranslationServiceError,
)
from logging_config import get_logger
from performance_monitor import performance_monitor
from services.subtitle_engine import MAX_SOURCE_WORDS_PER_SEC, RATE_SIGNAL_MIN_DUR
from services.translation_services import get_translator
from services.whisper_smart import resolve_model, smart_whisper

# Configuration
config = get_config()
logger = get_logger(__name__)


# =============================================================================
# ASR punctuation priming (v2 path only)
# =============================================================================
#: Text handed to Whisper as ``initial_prompt`` on the v2 transcription path.
#:
#: The failure it fixes
#: --------------------
#: ``large-v3`` intermittently falls into an **unpunctuated-lowercase attractor**: it
#: emits a whole clip as one run-on lowercase stream with not a single ``.``, ``,`` or
#: capital letter. Everything downstream is built on sentences — ``words_to_cues``
#: splits on terminal punctuation, the translator is asked to preserve it — so when the
#: attractor hits, the v2 pipeline produces broken cues on ~3/4 of the clip.
#:
#: A 7-way ablation on a known-bad clip (beam size 2 vs 5, VAD on/off, int8 vs float32,
#: ``chunk_length`` present/absent, ``condition_on_previous_text`` on/off, priming
#: on/off) showed beam size, VAD, compute type and chunking change **nothing**:
#: 0 terminals in every one of them. Only two levers moved it, and priming moved it
#: furthest — 0 -> 13 terminals, 0 -> 18 capitals — while also recovering casing on
#: acronyms (ICC, IDF) and correcting a real mistranscription.
#:
#: Why a *transcript excerpt* and not an instruction
#: ------------------------------------------------
#: ``initial_prompt`` is **not** a system prompt. faster-whisper feeds it to the decoder
#: as the tokens *preceding* the audio, so the model continues in whatever style it
#: establishes. Telling it "use punctuation" would be text to imitate, not an order to
#: obey. So this is written as a fragment of a punctuated transcript.
#:
#: What it deliberately does and does not contain
#: ---------------------------------------------
#: * ``.`` ``,`` and ``?`` plus sentence-initial capitals — exactly the tokens the
#:   attractor suppresses.
#: * Spoken filler ("Well", "I mean"). Load-bearing, not decoration: a clean *written*
#:   primer tidies the transcript, and ``style="faithful"`` only has filler to keep if
#:   the ASR emitted it in the first place. Priming in the register actually being
#:   transcribed — speech — is what keeps the word count honest.
#: * **No topic, no domain, no named entity.** A primer naming "television interview",
#:   "news" or any proper noun biases what the model hears in unclear audio.
#: * Short. It costs prompt tokens on every window and the ceiling is 224 tokens.
#:
#: What it must NOT contain, learned the hard way
#: ---------------------------------------------
#: **No short emphatic sentences.** The first draft ended "Really? Yes, really!" and
#: opened with "Hello." — and on the known-GOOD clip that draft made large-v3 fabricate
#: a sentence that was never spoken, twice:
#:
#:     truth   "...tried to take my life in Butler, Pennsylvania, Thomas generously
#:              mailed me one of his Purple Hearts."
#:     output  "...tried to take my life in Butler, Pennsylvania, I was killed by a
#:              police officer."   +   "I was killed by a police officer in Butler,
#:                                      Pennsylvania."  (0.26s, 219 CPS)
#:
#: Reproduced deterministically, and absent both without a primer and with this one.
#: The mechanism: ``initial_prompt`` is text to CONTINUE, so short punchy assertions in
#: it are cheap for the model to imitate with invented content. Removing those two
#: sentences — changing nothing else — removed the fabrication while keeping the entire
#: punctuation win. A primer that reads like a stretch of ordinary connected speech
#: gives the model a STYLE to copy without handing it a SHAPE to fill in.
#:
#: Verified across three clips (two known-bad, one known-good) against the unprimed
#: baseline; see the branch's review notes for the table. Guards that looked promising
#: and were rejected on evidence: ``hallucination_silence_threshold`` and
#: ``no_repeat_ngram_size`` changed nothing at all, and ``repetition_penalty=1.1``
#: suppressed the fabrication only by also thinning real speech.
ASR_PUNCTUATION_PRIMER = (
    "So, what do you think about that? Well, I mean, it depends, "
    "and it is not that simple."
)

#: Language-matched primers, keyed by the base language code the caller asked for.
#:
#: ``initial_prompt`` is text the decoder CONTINUES, so its language is not incidental:
#: priming a Hebrew clip with an English sentence asks the model to continue English
#: into Hebrew audio, which is at best a wasted 20 tokens and at worst a nudge toward
#: transliteration. Each entry is a straight translation of the English primer — same
#: shape, same punctuation, same register, same deliberate absence of proper nouns and
#: short emphatic sentences (see above: those cost a fabricated line).
#:
#: Only used when the caller NAMES the source language. ``source_lang="auto"`` keeps the
#: English primer, because guessing the language from a primer is exactly backwards.
ASR_PUNCTUATION_PRIMERS = {
    "he": "אז מה אתה חושב על זה? טוב, זאת אומרת, זה תלוי, וזה לא כל כך פשוט.",
    "en": ASR_PUNCTUATION_PRIMER,
}


def asr_primer_for(source_lang) -> str:
    """The punctuation primer to hand Whisper for this source language.

    Falls back to the English primer for ``"auto"``, ``None`` and any language with no
    entry of its own — the English one still demonstrates punctuation, which is the
    behaviour being primed.
    """
    code = str(source_lang or "auto").strip().lower().replace("_", "-").split("-")[0]
    return ASR_PUNCTUATION_PRIMERS.get(code, ASR_PUNCTUATION_PRIMER)


#: Whether the v2 path lets Whisper condition each window on its own previous output.
#:
#: This is the *other* lever the ablation moved, and the two interact. Conditioning is
#: what LOCKS the attractor in: once one window comes out unpunctuated, it is fed back
#: as context and every later window imitates it. Switching it off alone gave a partial
#: recovery (0 -> 5 terminals on the known-bad clip); priming alone gave a full one.
#:
#: Kept ON, and this is not a preference — with it OFF the primer would be nearly
#: useless. faster-whisper 1.2.0 builds each window's prompt from
#: ``all_tokens[prompt_reset_since:]``, and when ``condition_on_previous_text`` is False
#: it sets ``prompt_reset_since = len(all_tokens)`` after EVERY window. The initial
#: prompt lives at the head of ``all_tokens``, so from window two onward it has been
#: reset past: **priming would only ever reach the first 30 seconds.**
#:
#: That is not theory. Measured on a 159s clip with the primer and this flag False, the
#: transcript starts punctuated and correctly cased and then relapses into the attractor
#: — "far exactly what but", "evrit is writ is" — for the remainder: 49 terminals
#: against 131 with it True, and mangled words on top. With it True the punctuated style
#: the primer establishes propagates forward, which is exactly what conditioning is for.
#:
#: The cost of True is that errors propagate too; that is what made the FIRST draft of
#: the primer fabricate a sentence. The answer was to fix the primer (see its docstring),
#: not to disable the mechanism carrying it.
ASR_CONDITION_ON_PREVIOUS_TEXT = True


# =============================================================================
# Fabrication guards (v2 path only)
# =============================================================================
# Everything below is PURE — arrays and dicts in, arrays and dicts out, no model, no
# clock, no logging. That is deliberate: these are the rules that decide what reaches
# the viewer, and a rule you cannot unit-test is a rule you are hoping about.


#: How far below the file's OWN RMS a sample must sit before it counts as silence
#: rather than sound, for the trailing-silence trim.
#:
#: This used to be an absolute amplitude, 0.004 (~-48 dBFS), and an absolute threshold
#: is a claim about the recording LEVEL that nothing in this pipeline controls. On a
#: broadcast-levelled track (-23 LUFS is ~0.07 RMS linear) 0.004 sits 26 dB below the
#: programme, which is what it was tuned against and where it behaves well. On a quietly
#: recorded interview peaking at 0.01 it sits 8 dB below the PEAK — i.e. inside the
#: speech — and the trim eats the last word.
#:
#: So the same ratio, measured against the file's own level: -26 dB relative to the
#: whole file's RMS. That reproduces the old behaviour byte-for-byte on the material the
#: old number was tuned on and rescales it for everything else. It mirrors
#: :data:`subtitle_engine.ENERGY_VETO_DB`, which is relative for exactly this reason.
TRAILING_SILENCE_REL_DB = -26.0

#: Absolute floor under the relative threshold above, in linear amplitude.
#:
#: Two LSBs of 16-bit PCM. A digitally silent tail is not mathematically zero — it
#: carries dither at ±1 LSB (1/32768) — and on a very quiet file the relative threshold
#: alone can fall UNDER that dither, at which point the trim finds "sound" everywhere
#: and never fires. This is measured, not theoretical: a ±1-LSB change is inaudible (it
#: is below the noise floor of any microphone) and is exactly the perturbation that was
#: shown to flip Whisper's transcript, which is why silence has to be recognised as
#: silence even when the file is quiet.
DITHER_FLOOR = 2.0 / 32768.0


def _trim_trailing_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    silence_rel_db: float = TRAILING_SILENCE_REL_DB,
    pad_s: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Cut the dead air off the END of the audio, so Whisper has nothing to invent into.

    THE MEASURED FAILURE: Whisper decodes in fixed windows, and a window holding the
    last of the speech plus a stretch of silence is still a window it must emit tokens
    for — so it writes something. Three judged clips, three shapes of the same defect:
    0.72s of trailing silence produced the invented sentence "It stuck. That's what
    it's all about."; a second clip grew an entire phantom segment sitting in digital
    silence; a third fell into a repeat loop on the tail. The audio after the last
    spoken word carries nothing the transcript needs, so the cheapest fix available is
    to stop handing it to the decoder.

    Silence is RELATIVE to the file's own level, never an absolute amplitude — see
    :data:`TRAILING_SILENCE_REL_DB` for the quiet-interview case an absolute threshold
    got wrong, and :data:`DITHER_FLOOR` for the bound underneath it. This is the same
    choice :data:`subtitle_engine.ENERGY_VETO_DB` makes: this pipeline does not control
    the recording level of what it is handed, so every level test it applies has to be
    expressed against the material in front of it.

    Only the TAIL is touched, never the head. A clip's opening is where a quiet first
    syllable lives, and this pipeline has already lost a first second once (see the
    beam-size note in :func:`transcribe_with_words`); trimming the front would be
    re-buying that bug to solve a different one.

    ``pad_s`` of silence is deliberately LEFT IN PLACE. Whisper needs the release of the
    final word — an abrupt cut on the last loud sample truncates it into something else.

    Args:
        audio: mono float samples, as produced by :func:`_extract_audio_np`.
        sample_rate: samples per second of ``audio`` (16000 on this path).
        silence_rel_db: dB below the file's own RMS at which a sample stops counting as
            sound. See :data:`TRAILING_SILENCE_REL_DB`.
        pad_s: how much silence to keep after the last such sample.

    Returns:
        ``(audio, seconds_trimmed)``. The INPUT array is returned unchanged (and
        ``0.0``) when there is nothing to do: empty input, no sample anywhere above the
        threshold (a silent or pathologically quiet track is not something to truncate
        to nothing), or a tail already no longer than ``pad_s``.
    """
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return audio, 0.0

    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    threshold = max(rms * (10.0 ** (float(silence_rel_db) / 20.0)), DITHER_FLOOR)

    loud = np.nonzero(np.abs(audio) > threshold)[0]
    if loud.size == 0:
        return audio, 0.0

    keep = int(loud[-1]) + 1 + int(round(max(0.0, pad_s) * sample_rate))
    if keep >= len(audio):
        return audio, 0.0
    return audio[:keep], (len(audio) - keep) / float(sample_rate)


def _translation_failed(message: str, segments=None):
    """Build the pipeline's documented "translation failed, transcript survived" error.

    THE DEFECT THIS CLOSES: three places on the LEGACY (default) path caught a
    translation failure and assigned the SOURCE text to ``translated_text``. The job
    then finished GREEN and the user was handed an .srt in the wrong language with
    nothing anywhere saying so — the single worst outcome this pipeline can produce,
    because it is indistinguishable from success until someone reads the file. The v2
    path has had the right contract for a while: FAIL, and hand back whatever
    transcript was produced so the expensive part is not thrown away.

    ``TranslationFailedWithSalvage`` lives in ``tasks.processing_tasks``, which imports
    this module, so it is imported LAZILY: a module-level import would be circular. If
    that import is unavailable (this module is importable with no Celery at all), the
    failure is still a failure — it is raised as a ``TranslationServiceError`` instead
    of being silently downgraded to a wrong-language success.

    The transcript rides along on ``.segments``. ``original_srt`` is left empty because
    this module does not know where — or whether — the caller wrote one; the caller,
    which does, is the one that fills it in.
    """
    try:
        from tasks.processing_tasks import TranslationFailedWithSalvage
    except Exception:  # pragma: no cover - Celery-free import context
        error = TranslationServiceError("translation", message)
    else:
        error = TranslationFailedWithSalvage(message, "")
    error.segments = list(segments or [])
    return error


def _normalized_segment_text(text: str) -> str:
    """Case- and punctuation-insensitive form, for spotting a verbatim repeat.

    Mirrors ``subtitle_engine._normalized_text`` rather than importing it: that one is
    private to the gate it serves, and these two comparisons must be free to diverge —
    this module compares RAW ASR segments, that one compares finished cues.
    """
    return " ".join(
        "".join(
            ch for ch in str(text or "").lower() if ch.isalnum() or ch.isspace()
        ).split()
    )


#: Words in a repeated segment below which the repetition is speech, not a decode loop.
#:
#: "No. No." and "Wait, wait." are things people say, and Whisper transcribes them
#: correctly; a decode loop is a whole CLAUSE emitted twice ("thanks for watching" ->
#: "thanks for watching"). Requiring four words keeps the loop signature and stops the
#: signal firing on deliberate short repetition, which is a rhetorical device this
#: pipeline is elsewhere instructed to PRESERVE (see ``translation_v2``'s repetition
#: rule) — flagging it here and preserving it there is the same contradiction wearing
#: two hats.
REPEAT_LOOP_MIN_WORDS = 4


def _segment_suspects(segments: list[dict]) -> list[dict]:
    """Raw ASR segments worth a second look. **DETECTION ONLY — nothing is deleted.**

    WHY THIS EXISTS AT SEGMENT LEVEL, when ``subtitle_engine.drop_hallucinated_cues``
    already gates CUES: by the time a cue exists, ``words_to_cues`` has re-spotted the
    transcript, and re-spotting DILUTES timing — it hands a cue the silence around it,
    so a fabrication's speech rate is averaged away before anything measures it.
    Measured on the RAW segment, against the timestamps Whisper itself produced, the
    shape is still visible.

    WHY IT NO LONGER DELETES ANYTHING
    ---------------------------------
    It used to drop these segments and the word timestamps inside their spans. Two
    measurements ended that:

    * Across 7 real runs the deletion **never once fired**. The quality win credited to
      it came from :func:`_choose_asr_pass` — picking the cleaner of two COMPLETE
      transcripts — which is non-destructive and is still here.
    * Whisper is chaotically input-sensitive. One sample in 624,153 changed by a single
      LSB (inaudible, below any microphone's noise floor) flipped 11 words of
      transcript, reproduced 3/3 in both directions; an inaudible ±1-LSB dither on 0.1%
      of samples moved 13.1% of the words on one clip. A rate computed from one decode
      of one file is therefore not evidence that a person did not say something.

    So the counts feed :func:`_choose_asr_pass` and the job log, and nothing else. A
    suspect segment ships.

    Two signals:

    * **fast words per second** — over :data:`subtitle_engine.MAX_SOURCE_WORDS_PER_SEC`,
      and only on segments lasting at least
      :data:`subtitle_engine.RATE_SIGNAL_MIN_DUR`; below that the number describes
      Whisper's 0.02s timestamp quantiser rather than the speaker.
    * **verbatim repeat of the previous segment**, at least
      :data:`REPEAT_LOOP_MIN_WORDS` words long — Whisper's decode loop, which emits the
      same clause over and over once it latches. Compared with punctuation, casing and
      whitespace normalised away, because the loop varies those.

    A zero or negative duration is NOT a signal. It was, and it contradicted the sibling
    gate: ``subtitle_engine.drop_hallucinated_cues`` explicitly KEEPS a zero-length cue
    because "a zero-length cue has infinite CPS by arithmetic alone, which says nothing
    about whether its text is real". Two gates reading the same signal in opposite
    directions is not two opinions, it is one bug.

    Blank segments are never suspect: there is no fabrication in a segment with no
    words, and they are skipped for the repeat comparison too so two empties in a row
    are not read as a loop.

    Pure: the input list and its dicts are not mutated.

    Returns:
        Copies of the suspect segments in input order, each carrying ``suspect_reason``
        (a short greppable string) and ``index`` (its position in ``segments``, for the
        log line).
    """
    suspects: list[dict] = []
    previous_norm = ""

    for index, segment in enumerate(segments or []):
        text = str((segment or {}).get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(segment.get("start", 0) or 0)
            duration = float(segment.get("end", 0) or 0) - start
        except (TypeError, ValueError):
            start, duration = 0.0, 0.0

        normalized = _normalized_segment_text(text)
        reason = None
        if duration >= RATE_SIGNAL_MIN_DUR:
            words_per_sec = len(text.split()) / duration
            if words_per_sec > MAX_SOURCE_WORDS_PER_SEC:
                reason = f"fast_wps({words_per_sec:.2f}>{MAX_SOURCE_WORDS_PER_SEC:.2f})"
        if (
            reason is None
            and normalized
            and normalized == previous_norm
            and len(normalized.split()) >= REPEAT_LOOP_MIN_WORDS
        ):
            reason = "repeat_of_previous"

        previous_norm = normalized
        if reason:
            suspects.append(dict(segment, suspect_reason=reason, index=index))

    return suspects


def _choose_asr_pass(
    unprimed_suspects: int, primed_suspects: int, *, attractor: bool
) -> tuple[str, str]:
    """Which of the two ASR passes to ship: ``("unprimed"|"primed", why)``.

    The primed re-run exists for two different reasons and they want opposite tie-breaks,
    so the rule is written down here once instead of being spread through the caller:

    * **Fewer fabrications wins.** That is the whole point of running twice — the primer
      is known to change what the model HEARS (it once rewrote "work out how" into
      "work hard"), so it is neither trusted nor distrusted by default. It is measured.
    * **A tie goes to the unprimed pass**, because the unprimed transcript is the one
      that did not have a sentence of someone else's text prepended to it, and a
      re-run that bought nothing should not be paid for in decode drift.
    * **Except under the unpunctuated attractor**, where a tie goes to the PRIMED pass.
      That re-run was triggered by a transcript with zero terminal punctuation, which
      breaks every downstream stage (see :data:`ASR_PUNCTUATION_PRIMER`); a tie on
      fabrications means the primer cost nothing and fixed the thing it was called for.

    Pure, and takes COUNTS rather than lists so the decision can be read and tested on
    its own.
    """
    if primed_suspects < unprimed_suspects:
        return "primed", (
            f"primed pass has fewer impossible segments "
            f"({primed_suspects} < {unprimed_suspects})"
        )
    if primed_suspects > unprimed_suspects:
        return "unprimed", (
            f"primed pass invented more ({primed_suspects} > {unprimed_suspects}) — "
            "keeping the first pass"
        )
    if attractor:
        return "primed", (
            f"tie on impossible segments ({unprimed_suspects}) — keeping the primed "
            "pass, which is the one that escaped the unpunctuated attractor"
        )
    return "unprimed", (
        f"tie on impossible segments ({unprimed_suspects}) — keeping the unprimed "
        "first pass, which the primer cannot have rewritten"
    )


def _covers(word: dict, start: float, end: float) -> bool:
    """Does this word entry fall inside ``[start, end]``?

    Overlap, not containment: Whisper's word times and its segment times come from
    different passes and disagree at the edges by a tick or two. The second clause
    catches a zero-length span, where nothing can overlap anything.
    """
    try:
        word_start = float(word.get("s", 0) or 0)
        word_end = float(word.get("e", 0) or 0)
    except (TypeError, ValueError):
        return False
    return (word_end > start and word_start < end) or (start <= word_start <= end)


# =============================================================================
# Coverage invariant (v2 path only) — the OPPOSITE of a deletion rule
# =============================================================================
#: Uncovered audio at the END of the decode, in seconds, below which nothing is done.
#:
#: MEASURED: on one corpus clip the ASR simply STOPPED 5.25 seconds before the end of
#: the audio. Nothing was wrong with the transcript it produced; it was just short.
#: Re-decoding that region alone recovered the exact missing sentence ("if you enjoyed
#: the video don't forget to like and subscribe"). Under a second is ordinary — a final
#: consonant's release, the pad :func:`_trim_trailing_silence` deliberately leaves — and
#: re-decoding it would buy an empty window and a chance to invent into it.
COVERAGE_MIN_UNCOVERED_S = 1.0

#: How far below the whole file's RMS the uncovered tail may sit and still be re-decoded.
#:
#: MEASURED on the recovered clip: the uncovered tail sat at -14.4 dB against a file
#: average of -19.3 dB — i.e. it was 4.9 dB LOUDER than the programme. That is not a
#: fade-out, it is speech nobody transcribed. -20 dB relative is deliberately generous:
#: this mechanism only ever ADDS text, and the far more expensive error is the one that
#: was measured (a sentence silently missing) rather than a redundant decode of room
#: tone. Validated across all 11 corpus clips: it fires on the 3 with a speech-level
#: tail and stays silent on the other 8.
COVERAGE_MAX_QUIET_DB = -20.0


def _relative_db(region: np.ndarray, whole: np.ndarray) -> float | None:
    """Level of ``region`` in dB RELATIVE to ``whole``'s RMS. ``None`` if unmeasurable.

    Relative, never absolute, for the same reason :data:`TRAILING_SILENCE_REL_DB` is:
    nothing in this pipeline controls the recording level of what it is handed, so
    "loud" only means anything against the material itself.
    """
    if region is None or whole is None or len(region) == 0 or len(whole) == 0:
        return None
    region_rms = float(np.sqrt(np.mean(np.square(region.astype(np.float64)))))
    whole_rms = float(np.sqrt(np.mean(np.square(whole.astype(np.float64)))))
    if region_rms <= 0.0 or whole_rms <= 0.0:
        return None
    return 20.0 * float(np.log10(region_rms / whole_rms))


def _uncovered_tail(
    segments: list[dict],
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_uncovered_s: float = COVERAGE_MIN_UNCOVERED_S,
    max_quiet_db: float = COVERAGE_MAX_QUIET_DB,
) -> dict | None:
    """The stretch of audio after the last segment, when it still sounds like speech.

    THE COVERAGE INVARIANT: every second of audio handed to the decoder should be
    accounted for by some segment. It is the exact opposite of a deletion rule — it can
    only ever discover that something is MISSING, and the caller's only permitted
    response is to add.

    ``_trim_trailing_silence`` runs first, on the same array, so trailing DIGITAL
    silence is already gone by the time coverage is measured: anything still uncovered
    here is audio the decoder was given and did not describe. The two mechanisms
    compose, and in that order — measuring coverage against the untrimmed file would
    report every quiet ending as a gap.

    Pure: numpy in, a dict or ``None`` out; nothing is decoded and nothing is mutated.

    Returns:
        ``{"start", "end", "duration", "db"}`` in seconds (``db`` relative to the whole
        array's RMS), or ``None`` when there is no gap, the gap is shorter than
        ``min_uncovered_s``, or the gap is quieter than ``max_quiet_db`` — which is what
        an ordinary fade-out or a room-tone ending looks like.
    """
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return None
    total_s = len(audio) / float(sample_rate)

    last_end = 0.0
    for segment in segments or []:
        try:
            last_end = max(last_end, float((segment or {}).get("end", 0) or 0))
        except (TypeError, ValueError):
            continue
    last_end = min(max(last_end, 0.0), total_s)

    if total_s - last_end < min_uncovered_s:
        return None

    first = int(last_end * sample_rate)
    level = _relative_db(audio[first:], audio)
    if level is None or level < max_quiet_db:
        return None
    return {
        "start": last_end,
        "end": total_s,
        "duration": total_s - last_end,
        "db": round(level, 2),
    }


def _shifted(items: list[dict], offset: float, keys: tuple) -> list[dict]:
    """Copies of ``items`` with the named time keys moved later by ``offset`` seconds.

    A region decode returns timestamps relative to the region, so they mean nothing
    until they are put back where the audio came from. Pure.
    """
    out = []
    for item in items or []:
        shifted = dict(item)
        for key in keys:
            try:
                shifted[key] = float(item.get(key, 0) or 0) + offset
            except (TypeError, ValueError):
                pass
        out.append(shifted)
    return out


def _recovery_is_new(recovered: list[dict], previous_text: str) -> tuple[bool, str]:
    """Is a tail re-decode worth appending? ``(yes, why_not)``.

    Two ways a recovery is worse than nothing, and both have been seen from Whisper on
    short windows: an EMPTY result (a window with nothing to say still has to emit
    tokens, and sometimes emits none), and a verbatim REPEAT of the segment before the
    gap, which is the decode loop reaching across the seam. Either one would put a
    duplicate line on screen in the name of completeness.

    Pure.
    """
    text = " ".join(str((s or {}).get("text") or "").strip() for s in recovered or [])
    if not text.strip():
        return False, "the re-decode returned no text"
    if _normalized_segment_text(text) == _normalized_segment_text(previous_text):
        return False, "the re-decode repeated the segment before the gap verbatim"
    return True, ""


def _word_loss_report(
    segments: list[dict], words: list[dict], *, min_gap: int = 2
) -> list[dict]:
    """Segments whose TEXT carries words the word list does not — visibility only.

    THE MEASURED FAILURE: a profanity present in a segment's text was simply absent from
    the ASR word list. ``words_to_cues`` spots from words, so the rebuilt cue silently
    dropped it — and the delivered file said the opposite of the source, with nothing
    anywhere in the logs to say a word had gone missing. The two outputs of one Whisper
    pass are allowed to disagree; disagreeing SILENTLY is what made this expensive.

    This filters nothing and fixes nothing. It exists so the next occurrence is one grep
    away instead of one re-listen away.

    Args:
        min_gap: how many more whitespace tokens the segment text must have than the
            words covering its span before it is worth a line in the log. Off-by-one is
            routine — Whisper hyphenates and re-joins at window edges — so 2 is the
            smallest gap that is not just tokenisation noise.

    Returns:
        ``[{"start", "end", "text_tokens", "word_tokens", "gap"}]``, in segment order.
        Empty when there are no words at all: that is the Gemini/no-word-timestamps
        case, which is reported elsewhere and is not a per-segment loss.
    """
    if not segments or not words:
        return []

    report: list[dict] = []
    for segment in segments:
        text = str((segment or {}).get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        text_tokens = len(text.split())
        word_tokens = sum(1 for w in words if _covers(w, start, end))
        gap = text_tokens - word_tokens
        if gap >= min_gap:
            report.append(
                {
                    "start": start,
                    "end": end,
                    "text_tokens": text_tokens,
                    "word_tokens": word_tokens,
                    "gap": gap,
                }
            )
    return report


def transcribe_and_translate_streamed(
    video_path,
    target_language,
    source_lang="auto",
    quality="balanced",
    model_preference="large",
    translation_service="google",
    progress_callback=None,
    model_callback=None,
    youtube_url=None,  # NEW: For Gemini support
):
    """
    P1 Step 1: Pipeline overlap - transcribe and translate simultaneously.

    Streams segments from Whisper as they're transcribed and translates them
    in parallel batches, reducing total time from sequential (transcribe + translate)
    to overlapped (max(transcribe, translate/parallelism)).

    Args:
        video_path: Path to video file
        target_language: Target language for translation
        source_lang: Source language (auto-detect if "auto")
        quality: Transcription quality preference
        model_preference: Whisper model to use
        translation_service: Translation service ("google" or "openai")
        progress_callback: Optional callback for progress updates
        model_callback: Optional callback when model is loaded

    Returns:
        dict: {
            "segments": List of segments with both text and translated_text,
            "language": Detected language
        }
    """
    logger.info(
        "🚀 === P1: Pipeline Overlap - Streaming Transcription + Concurrent Translation ==="
    )

    # Get parallelism settings from environment
    parallelism = int(os.environ.get("TRANSLATION_PARALLELISM", "4"))
    batch_size = int(os.environ.get("TRANSLATION_BATCH_SIZE", "20"))

    logger.info(
        f"⚙️ Translation parallelism: {parallelism} workers, batch size: {batch_size} segments"
    )

    try:
        # FAKE mode: return small deterministic segments
        if config.USE_FAKE_YTDLP:
            if progress_callback:
                progress_callback(
                    25, "Starting FAKE transcription...", 85, "Step 1: FAKE Whisper", 5
                )
            fake_segments = [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "translated_text": "Hello world",
                },
                {
                    "start": 2.5,
                    "end": 4.0,
                    "text": "This is a test",
                    "translated_text": "This is a test",
                },
            ]
            return {
                "segments": fake_segments,
                "language": (source_lang if source_lang != "auto" else "en"),
            }

        # === Phase 1: Audio Extraction (same as transcribe_video) ===
        logger.info("📹 Step 1/3: Extracting audio from video...")

        # Probe audio format
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            video_path,
        ]
        try:
            probe_result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=config.FFPROBE_TIMEOUT,
            )
            streams = probe_result.stdout
        except subprocess.TimeoutExpired:
            raise FFmpegTimeoutError("audio_probe", config.FFPROBE_TIMEOUT)
        except subprocess.CalledProcessError as e:
            raise FFmpegProcessError(
                "audio_probe", e.stderr.decode() if e.stderr else "Unknown error"
            )

        try:
            audio_streams = json.loads(streams).get("streams", [])
        except json.JSONDecodeError:
            audio_streams = []

        if not audio_streams:
            raise ValueError("No audio stream found in the video file")

        audio_info = audio_streams[0]
        codec = audio_info.get("codec_name")
        sample_rate = int(audio_info.get("sample_rate", 0))
        channels = int(audio_info.get("channels", 0))

        is_optimal_format = (
            codec == "pcm_s16le" and sample_rate == 16000 and channels == 1
        )

        if is_optimal_format:
            logger.info("✅ Audio already in optimal format")
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-nostdin",
                "-f",
                "s16le",
                "-acodec",
                "copy",
                "-",
            ]
        else:
            logger.info(
                f"🔄 Re-encoding audio: {codec} @ {sample_rate}Hz, {channels}ch → 16kHz mono"
            )
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-nostdin",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-",
            ]

        if progress_callback:
            progress_callback(
                15, "Processing audio...", 60, "Step 1: Audio processing", 5
            )

        try:
            process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            audio_buffer, stderr = process.communicate(
                timeout=config.FFMPEG_RUN_TIMEOUT
            )

            if process.returncode != 0:
                raise AudioExtractionError(video_path, stderr.decode())
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise FFmpegTimeoutError("audio_extraction", config.FFMPEG_RUN_TIMEOUT)

        if progress_callback:
            progress_callback(
                20, "Preparing audio data...", 75, "Step 1: Data preparation", 5
            )

        audio_np = (
            np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        )
        audio_duration = len(audio_np) / 16000  # 16kHz sample rate

        logger.info(f"📊 Audio extracted: {audio_duration:.1f}s duration")

        # === Phase 2: Transcription ===
        # Check if Gemini is requested - if so, use transcribe_smart() instead of manual streaming
        if model_preference == "gemini":
            logger.info("🎯 Gemini requested - using smart transcription path")

            if progress_callback:
                progress_callback(
                    25,
                    "Starting transcription with Gemini...",
                    25,
                    "Step 1: Gemini AI",
                    5,
                )

            if model_callback:
                model_callback()

            # Use transcribe_smart which handles Gemini
            result = smart_whisper.transcribe_smart(
                audio_np,
                language=source_lang,
                duration=audio_duration,
                quality_preference=quality,
                model_preference=model_preference,
                progress_callback=progress_callback,
                youtube_url=youtube_url,
            )

            segments = result["segments"]
            detected_language = result["language"]

            # Translate segments
            if progress_callback:
                progress_callback(
                    75, "Translating transcription...", 75, "Step 2: Translation", 5
                )

            translator = get_translator(translation_service)

            # Collect all texts for batch translation
            texts_to_translate = [seg["text"] for seg in segments if seg.get("text")]

            # Translate in batch
            if texts_to_translate:
                translations = translator.translate_batch(
                    texts_to_translate,
                    target_language=target_language,
                    source_language=detected_language,
                )

                # Map translations back to segments
                translation_idx = 0
                for segment in segments:
                    if segment.get("text"):
                        segment["translated_text"] = translations[translation_idx]
                        translation_idx += 1

            if progress_callback:
                progress_callback(100, "Completed!", 100, "Step 3: Complete", 5)

            return {
                "segments": segments,
                "language": detected_language,
            }

        # === Phase 2: Load Whisper Model (non-Gemini path) ===
        logger.info("🤖 Step 2/3: Loading Whisper model...")

        if progress_callback:
            progress_callback(
                25,
                "Starting transcription with Whisper...",
                25,
                "Step 1: Whisper AI",
                5,
            )

        if model_callback:
            model_callback()

        # Choose and load model. `resolve_model` and not a local copy of the rule:
        # this path called `load_model` directly, so the memory guard and every
        # downgrade WARNING lived in a helper production never reached.
        if model_preference and model_preference in ["tiny", "base", "medium", "large"]:
            model_name = resolve_model(model_preference)
        else:
            model_name = "tiny"

        model = smart_whisper.load_model(model_name)

        # Transcription options
        options = {
            "word_timestamps": True,
            "beam_size": 2 if model_name in ["large", "medium"] else 5,
            "chunk_length": 30,
            "condition_on_previous_text": True,
        }

        if source_lang != "auto":
            options["language"] = source_lang

        logger.info(
            f"💾 Transcription settings: model={model_name}, beam_size={options['beam_size']}"
        )

        # === Phase 3: P1 Concurrent Translation - Streaming + Parallel Batches ===
        logger.info(
            f"🔄 Step 3/3: Streaming transcription with {parallelism}x concurrent translation..."
        )

        transcription_start = time.time()

        # Get translator
        translator = get_translator(translation_service)

        # Start transcription stream
        segments_iter, info = model.transcribe(audio_np, **options)
        detected_language = info.language

        logger.info(f"🌍 Detected language: {detected_language}")

        # Storage for results
        current_batch = []
        batch_futures = {}  # Maps future -> (batch_index, batch_segments)
        completed_segments = {}  # Maps global_index -> segment_with_translation
        next_segment_index = 0
        batch_index = 0

        # Create thread pool for parallel translation
        executor = ThreadPoolExecutor(max_workers=parallelism)

        def translate_batch_worker(batch_segments, batch_idx, service):
            """Worker function to translate a batch of segments"""
            thread_id = threading.get_ident()
            try:
                logger.info(
                    f"🔄 [Thread-{thread_id}] Translating batch #{batch_idx}: {len(batch_segments)} segments"
                )

                # Extract texts
                texts = [seg["text"] for seg in batch_segments]

                # Translate
                translated_texts = translator.translate_batch(
                    texts, target_language, source_language=detected_language
                )

                # Assign translations back
                for i, seg in enumerate(batch_segments):
                    seg["translated_text"] = translated_texts[i]

                logger.info(
                    f"✅ [Thread-{thread_id}] Batch #{batch_idx} translated successfully"
                )
                return batch_segments

            except Exception as e:
                logger.error(
                    f"❌ [Thread-{thread_id}] Batch #{batch_idx} translation failed: {e}"
                )
                # NO source-text fallback. Assigning `seg["text"]` to
                # `translated_text` here is what shipped untranslated .srt files under a
                # green job — see :func:`_translation_failed`. The failure travels; the
                # collector below turns it into one visible job failure.
                raise

        # Process segments as they arrive
        try:
            for segment in segments_iter:
                # Convert to dict format
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "index": next_segment_index,
                }

                current_batch.append(segment_dict)
                next_segment_index += 1

                # Update progress
                if progress_callback and audio_duration:
                    transcription_progress = (segment.end / audio_duration) * 100
                    step_progress = 30 + int(transcription_progress * 0.55)
                    progress_callback(
                        step_progress,
                        f"Transcription + Translation: {segment.end:.0f}s/{audio_duration:.0f}s",
                        step_progress,
                        "Step 1+2: Whisper + Translation",
                        5,
                    )

                # When batch is full, submit for translation
                if len(current_batch) >= batch_size:
                    batch_to_translate = current_batch.copy()
                    logger.info(
                        f"📤 Submitting batch #{batch_index} to thread pool (inflight={len(batch_futures)})"
                    )
                    future = executor.submit(
                        translate_batch_worker,
                        batch_to_translate,
                        batch_index,
                        translation_service,
                    )
                    batch_futures[future] = (batch_index, batch_to_translate)
                    batch_index += 1
                    current_batch = []

            # Submit final partial batch if any
            if current_batch:
                logger.info(
                    f"📤 Submitting final batch #{batch_index} to thread pool (inflight={len(batch_futures)})"
                )
                future = executor.submit(
                    translate_batch_worker,
                    current_batch,
                    batch_index,
                    translation_service,
                )
                batch_futures[future] = (batch_index, current_batch)

            logger.info(
                f"✅ Transcription complete: {next_segment_index} segments, {len(batch_futures)} batches"
            )

            # Collect translation results as they complete
            logger.info(
                f"⏳ Waiting for {len(batch_futures)} translation batches to complete..."
            )

            failed_batches = []
            for future in as_completed(batch_futures):
                batch_idx, original_batch = batch_futures[future]
                try:
                    translated_batch = (
                        future.result()
                    )  # P1 FIX: Use .result() instead of .get()
                    # Store by index for ordering
                    for seg in translated_batch:
                        completed_segments[seg["index"]] = seg
                    logger.info(f"✅ Collected batch #{batch_idx} results")
                except Exception as e:
                    logger.error(f"❌ Failed to collect batch #{batch_idx}: {e}")
                    failed_batches.append((batch_idx, e))
                    # The SOURCE text is kept — it is the salvage — but it is never
                    # promoted to `translated_text`. A cue with no translation must
                    # look like a cue with no translation.
                    for seg in original_batch:
                        seg.pop("translated_text", None)
                        completed_segments[seg["index"]] = seg

            # Reconstruct segments in order
            all_segments = []
            for i in range(next_segment_index):
                if i in completed_segments:
                    seg = completed_segments[i]
                    # Remove index field before returning
                    del seg["index"]
                    all_segments.append(seg)

            if failed_batches:
                # FAIL VISIBLY, carrying the transcript. Substituting the source for a
                # translation and returning success is the one outcome this pipeline
                # must never produce — see :func:`_translation_failed`.
                raise _translation_failed(
                    f"{len(failed_batches)} of {len(batch_futures)} translation "
                    f"batch(es) failed (first: {failed_batches[0][1]}) — "
                    f"{len(all_segments)} transcribed segments salvaged",
                    segments=all_segments,
                )

            logger.info(f"✅ All translations complete: {len(all_segments)} segments")

        finally:
            executor.shutdown(wait=True)

        transcription_duration = time.time() - transcription_start

        # Log performance
        performance_monitor.log_transcription_performance(
            audio_duration,
            transcription_duration,
            model_name,
            segments_count=len(all_segments),
        )

        logger.info(
            f"🎉 Pipeline overlap complete! Total time: {transcription_duration:.1f}s "
            f"for {audio_duration:.1f}s audio"
        )

        if progress_callback:
            progress_callback(
                90,
                "Transcription and translation completed",
                90,
                "Step 1+2: Processing results",
                5,
            )

        return {
            "segments": all_segments,
            "language": detected_language,
        }

    except Exception as e:
        logger.error(f"Pipeline overlap failed: {e}")
        raise


def transcribe_video(
    video_path,
    source_lang="auto",
    quality="balanced",
    model_preference="large",
    progress_callback=None,
    model_callback=None,
    youtube_url=None,  # FIXED: Added missing parameter
):
    """
    Transcribe video using Whisper by streaming audio from FFmpeg directly.
    """
    try:
        # FAKE mode: return small deterministic segments without running Whisper
        if config.USE_FAKE_YTDLP:
            if progress_callback:
                progress_callback(
                    25, "Starting FAKE transcription...", 85, "Step 1: FAKE Whisper", 5
                )
            fake_segments = [
                {"start": 0.0, "end": 2.0, "text": "Hello world"},
                {"start": 2.5, "end": 4.0, "text": "This is a test"},
            ]
            return {
                "segments": fake_segments,
                "language": (source_lang if source_lang != "auto" else "en"),
            }

        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            video_path,
        ]
        try:
            probe_result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=config.FFPROBE_TIMEOUT,
            )
            streams = probe_result.stdout
        except subprocess.TimeoutExpired:
            raise FFmpegTimeoutError("audio_probe", config.FFPROBE_TIMEOUT)
        except subprocess.CalledProcessError as e:
            raise FFmpegProcessError(
                "audio_probe", e.stderr.decode() if e.stderr else "Unknown error"
            )

        try:
            audio_streams = json.loads(streams).get("streams", [])
        except json.JSONDecodeError:
            audio_streams = []

        if not audio_streams:
            raise ValueError(
                "No audio stream found in the video file. The file may be corrupted or unsupported."
            )

        audio_info = audio_streams[0]
        codec = audio_info.get("codec_name")
        sample_rate = int(audio_info.get("sample_rate", 0))
        channels = int(audio_info.get("channels", 0))

        is_optimal_format = (
            codec == "pcm_s16le" and sample_rate == 16000 and channels == 1
        )

        if is_optimal_format:
            logger.info(
                "✅ Audio is already in the optimal format. Extracting without re-encoding."
            )
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-nostdin",
                "-f",
                "s16le",
                "-acodec",
                "copy",
                "-",
            ]
        else:
            logger.info(
                f"Audio format is {codec} @ {sample_rate}Hz, {channels}ch. Re-encoding to 16kHz mono pcm_s16le."
            )
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-nostdin",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-",
            ]

        if progress_callback:
            progress_callback(
                15, "Processing audio...", 60, "Step 1: Audio processing", 5
            )

        try:
            process = subprocess.Popen(
                ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            audio_buffer, stderr = process.communicate(
                timeout=config.FFMPEG_RUN_TIMEOUT
            )

            if process.returncode != 0:
                raise AudioExtractionError(video_path, stderr.decode())
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise FFmpegTimeoutError("audio_extraction", config.FFMPEG_RUN_TIMEOUT)

        if progress_callback:
            progress_callback(
                20, "Preparing audio data...", 75, "Step 1: Data preparation", 5
            )

        audio_np = (
            np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if progress_callback:
            progress_callback(
                25,
                "Starting transcription with Whisper...",
                25,
                "Step 1: Whisper AI",
                5,
            )

        if model_callback:
            model_callback()

        # Phase A: Enhanced transcription performance monitoring
        transcription_start = time.time()
        # Calculate duration for progress tracking
        audio_duration = len(audio_np) / 16000  # 16kHz sample rate

        result = smart_whisper.transcribe_smart(
            audio_np,
            language=source_lang,
            duration=audio_duration,
            quality_preference=quality,
            model_preference=model_preference,
            progress_callback=progress_callback,
            youtube_url=youtube_url,
        )
        transcription_duration = time.time() - transcription_start

        # Phase A: Log transcription performance
        segments_count = (
            len(result.get("segments", [])) if isinstance(result, dict) else 0
        )
        performance_monitor.log_transcription_performance(
            audio_duration,
            transcription_duration,
            model_preference or "auto",
            segments_count=segments_count,
        )

        if progress_callback:
            progress_callback(
                90, "Transcription completed", 90, "Step 1: Processing results", 5
            )

        return result

    except (subprocess.CalledProcessError, ValueError, Exception) as e:
        logger.error(f"Transcription failed: {e}")
        raise


def _extract_audio_np(video_path, progress_callback=None):
    """
    Decode a video's audio to the 16 kHz mono float32 array Whisper expects.

    Extracted for the v2 subtitle pipeline (:func:`transcribe_with_words`). The two
    legacy entry points keep their own inlined copy on purpose: with all feature flags
    off their behaviour — including log wording and ordering — must stay byte-identical
    to what shipped, so they are deliberately not refactored onto this helper.

    Returns:
        ``(audio_np, audio_duration_seconds)``
    """
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "a",
        video_path,
    ]
    try:
        probe_result = subprocess.run(
            ffprobe_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=config.FFPROBE_TIMEOUT,
        )
        streams = probe_result.stdout
    except subprocess.TimeoutExpired:
        raise FFmpegTimeoutError("audio_probe", config.FFPROBE_TIMEOUT)
    except subprocess.CalledProcessError as e:
        raise FFmpegProcessError(
            "audio_probe", e.stderr.decode() if e.stderr else "Unknown error"
        )

    try:
        audio_streams = json.loads(streams).get("streams", [])
    except json.JSONDecodeError:
        audio_streams = []

    if not audio_streams:
        raise ValueError("No audio stream found in the video file")

    audio_info = audio_streams[0]
    codec = audio_info.get("codec_name")
    sample_rate = int(audio_info.get("sample_rate", 0))
    channels = int(audio_info.get("channels", 0))

    if codec == "pcm_s16le" and sample_rate == 16000 and channels == 1:
        logger.info("✅ Audio already in optimal format")
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-nostdin",
            "-f",
            "s16le",
            "-acodec",
            "copy",
            "-",
        ]
    else:
        logger.info(
            f"🔄 Re-encoding audio: {codec} @ {sample_rate}Hz, {channels}ch → 16kHz mono"
        )
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-nostdin",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]

    if progress_callback:
        progress_callback(15, "Processing audio...", 60, "Step 1: Audio processing", 5)

    try:
        process = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        audio_buffer, stderr = process.communicate(timeout=config.FFMPEG_RUN_TIMEOUT)
        if process.returncode != 0:
            raise AudioExtractionError(video_path, stderr.decode())
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise FFmpegTimeoutError("audio_extraction", config.FFMPEG_RUN_TIMEOUT)

    if progress_callback:
        progress_callback(
            20, "Preparing audio data...", 75, "Step 1: Data preparation", 5
        )

    audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    audio_duration = len(audio_np) / 16000  # 16kHz sample rate
    logger.info(f"📊 Audio extracted: {audio_duration:.1f}s duration")
    return audio_np, audio_duration


def transcribe_with_words(
    video_path,
    source_lang="auto",
    quality="balanced",
    model_preference="large",
    progress_callback=None,
    model_callback=None,
    youtube_url=None,
    collect_words=True,
):
    """
    Transcribe only — and, unlike every legacy path, KEEP the word timestamps.

    This is the transcription stage of the opt-in v2 subtitle pipeline. It exists because
    the legacy paths both throw the words away: ``transcribe_and_translate_streamed``
    reduces each Whisper segment to ``{start, end, text}`` while translating batches
    overlapped with transcription, and ``whisper_smart.transcribe_smart`` does the same.
    ``subtitle_engine.words_to_cues`` needs the words, so the v2 path transcribes first
    and translates afterwards in one whole-scene call (see ``translation_v2``) — the
    overlap optimisation is traded for cross-cue context, deliberately.

    Model selection and the progress-callback contract mirror
    ``transcribe_and_translate_streamed``. The Whisper options no longer do, in exactly
    ONE respect: this path passes :data:`ASR_PUNCTUATION_PRIMER` as ``initial_prompt``.
    That is a deliberate, measured divergence — see the constant's docstring for the
    ablation — and it is confined to the v2 path so the legacy path stays byte-identical
    to what shipped. Everything else (model, beam size, chunk length,
    ``condition_on_previous_text``) is still shared, so a v2/legacy comparison of the
    same request still isolates the flags plus this one prompt.

    What this returns is not raw Whisper output. Four mechanisms run over it, each one
    bought with a measured defect that reached a delivered file — and **not one of them
    deletes transcript**:

    * :func:`_trim_trailing_silence` before the decode — it shortens the decoder's
      INPUT, so there is no window of dead air for the model to invent into.
    * :func:`_segment_suspects` + :func:`_choose_asr_pass` after it. The suspect counts
      choose between two COMPLETE transcripts (non-destructive, and the source of the
      measured quality win) and are otherwise logged. They used to delete the segments
      they flagged; see :func:`_segment_suspects` for the two measurements that ended
      that.
    * :func:`_uncovered_tail` — the coverage invariant, which is the opposite of a
      deletion rule: audio the decoder was handed and did not describe gets re-decoded
      and APPENDED. It runs after the trim, so trailing digital silence is already gone
      and cannot be mistaken for a gap.
    * :func:`_word_loss_report` as a pure log.

    Args:
        collect_words: gather per-word timestamps. Whisper is asked for them either way
            (``word_timestamps=True``, as the legacy path already does), so this only
            controls whether they are retained.

    Returns:
        dict: ``{"segments": [{"start","end","text"}, ...],
        "words": [{"s","e","w"}, ...], "language": str, "asr_primed": bool}``.
        ``words`` is EMPTY for the Gemini model, which returns no word timing — callers
        that need spotting must fall back to segment-based cues in that case.
        ``asr_primed`` reports whether the pass that was actually SHIPPED used the
        primer (false for Gemini and for FAKE mode); ``subtitle_engine.words_to_cues``
        takes it so its unpunctuated-ASR fallback can say whether it fired *despite* the
        primer, which is a materially different (and much more interesting) event.
    """
    logger.info("🚀 === v2 transcription (word timestamps retained) ===")

    # FAKE mode: deterministic output, no Whisper, no network.
    if config.USE_FAKE_YTDLP:
        if progress_callback:
            progress_callback(
                25, "Starting FAKE transcription...", 85, "Step 1: FAKE Whisper", 5
            )
        return {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world."},
                {"start": 2.5, "end": 4.0, "text": "This is a test."},
            ],
            "words": (
                [
                    {"s": 0.0, "e": 1.0, "w": "Hello"},
                    {"s": 1.0, "e": 2.0, "w": "world."},
                    {"s": 2.5, "e": 3.0, "w": "This"},
                    {"s": 3.0, "e": 3.3, "w": "is"},
                    {"s": 3.3, "e": 3.6, "w": "a"},
                    {"s": 3.6, "e": 4.0, "w": "test."},
                ]
                if collect_words
                else []
            ),
            "language": (source_lang if source_lang != "auto" else "en"),
            "asr_primed": False,
        }

    audio_np, audio_duration = _extract_audio_np(video_path, progress_callback)

    # === Gemini: no word timestamps available ===
    if model_preference == "gemini":
        logger.info(
            "🎯 Gemini requested — transcribing via smart path (no word timestamps)"
        )
        if progress_callback:
            progress_callback(
                25, "Starting transcription with Gemini...", 25, "Step 1: Gemini AI", 5
            )
        if model_callback:
            model_callback()

        result = smart_whisper.transcribe_smart(
            audio_np,
            language=source_lang,
            duration=audio_duration,
            quality_preference=quality,
            model_preference=model_preference,
            progress_callback=progress_callback,
            youtube_url=youtube_url,
        )
        return {
            "segments": result.get("segments", []),
            "words": [],
            "language": result.get("language"),
            "asr_primed": False,  # not Whisper — there is no initial_prompt to give
        }

    # === Whisper ===
    if progress_callback:
        progress_callback(
            25, "Starting transcription with Whisper...", 25, "Step 1: Whisper AI", 5
        )
    if model_callback:
        model_callback()

    # Same selection rule as the legacy streamed path, so both paths pick the same
    # model — and both now go through the memory guard rather than around it.
    if model_preference and model_preference in ["tiny", "base", "medium", "large"]:
        model_name = resolve_model(model_preference)
    else:
        model_name = "tiny"

    model = smart_whisper.load_model(model_name)

    options = {
        "word_timestamps": True,
        # `large` gets the full beam. The old `2` was a speed concession for the 2GB
        # prod worker, and it is exactly the degraded-search condition under which the
        # judged runs mis-decoded real speech ("work out how" -> "work hard. You've
        # got how") and dropped the first second of a clip. Re-decoded at beam 5:
        # both defects gone, on the same audio, same model. `medium` keeps 2 — its
        # entire corpus evidence base (R1-R8) was built at that setting.
        "beam_size": 2 if model_name == "medium" else 5,
        "chunk_length": 30,
        "condition_on_previous_text": ASR_CONDITION_ON_PREVIOUS_TEXT,
        # Punctuation priming is MEDIUM medicine. Without it, medium returns some
        # clips entirely unpunctuated and every stage below (sentence spotting, cue
        # splitting, translation punctuation) is built on sentences that then do not
        # exist. large-v3 at beam 5 punctuates reliably on its own — measured across
        # five judged clips — and the primer occasionally REWRITES its hearing (the
        # "work hard" mutation above survived priming at beam 5, and vanished only
        # when the primer was dropped). So: prime medium/base, leave large alone.
        # See ASR_PUNCTUATION_PRIMER.
        "initial_prompt": (
            None if model_name == "large" else asr_primer_for(source_lang)
        ),
    }
    if source_lang != "auto":
        options["language"] = source_lang

    logger.info(
        f"💾 v2 transcription settings: model={model_name}, beam_size={options['beam_size']}, "
        f"collect_words={collect_words}, "
        f"punctuation_priming={'off (large trusts its own punctuation)' if options['initial_prompt'] is None else 'on (primer language=' + str(source_lang or 'auto').split('-')[0] + ')'}, "
        f"condition_on_previous_text={ASR_CONDITION_ON_PREVIOUS_TEXT}"
    )

    # Silence at the end of the file is the cheapest hallucination there is — see
    # :func:`_trim_trailing_silence`. Done here, on the decoder's input only:
    # ``audio_duration`` still describes the MEDIA, so progress and the performance
    # log keep reporting the clip the user submitted.
    audio_np, trimmed_s = _trim_trailing_silence(audio_np, 16000)
    if trimmed_s >= 0.5:
        # f-strings, not %-args, everywhere in this function. ``logging_config``
        # configures structlog with ``wrapper_class=structlog.BoundLogger``, whose
        # ``warning(event, *args)`` raises TypeError on the second positional argument
        # — so a %-style log line here is a job that dies inside a log statement.
        logger.info(
            f"✂️ v2 transcription: trimmed {trimmed_s:.2f}s of trailing silence before "
            f"ASR — that is decode window Whisper would have had to emit tokens for"
        )

    transcription_start = time.time()

    def _materialize(run_options, audio=None):
        # `audio` overrides the full array for the tail re-decode, which hands the model
        # ONE region and gets region-relative timestamps back; the caller shifts them.
        segments_iter, run_info = model.transcribe(
            audio_np if audio is None else audio, **run_options
        )
        out_segments: list[dict] = []
        out_words: list[dict] = []
        for segment in segments_iter:
            out_segments.append(
                {"start": segment.start, "end": segment.end, "text": segment.text}
            )
            if collect_words:
                for word in getattr(segment, "words", None) or []:
                    text = getattr(word, "word", None)
                    if text is None or not str(text).strip():
                        continue
                    out_words.append(
                        {
                            "s": float(getattr(word, "start", segment.start) or 0.0),
                            "e": float(getattr(word, "end", segment.end) or 0.0),
                            "w": str(text),
                        }
                    )

            if progress_callback and audio_duration:
                step_progress = 30 + int((segment.end / audio_duration) * 100 * 0.55)
                progress_callback(
                    step_progress,
                    f"Transcription: {segment.end:.0f}s/{audio_duration:.0f}s",
                    step_progress,
                    "Step 1: Whisper AI",
                    5,
                )
        return out_segments, out_words, run_info

    def _terminal_count(seg_list):
        text = " ".join(s["text"] for s in seg_list)
        return text, sum(text.count(mark) for mark in ".!?")

    segments, words, info = _materialize(options)
    detected_language = info.language
    logger.info(f"🌍 Detected language: {detected_language}")

    # Conditional re-prime, the reconciliation of two evidence sets. Judged runs
    # showed the primer occasionally REWRITES large's hearing ("work out how" ->
    # "work hard"), so large now starts unprimed; the ablation above showed some
    # clips fall into the unpunctuated attractor without it. So: trust large first,
    # and pay for one primed re-run only when the first pass shows damage. Clips that
    # come back clean (the common case at beam 5) never see the primer at all.
    #
    # TWO triggers now, not one. The attractor signature (enough words to be a real
    # transcript, zero terminal punctuation) was the original. The second is
    # FABRICATION: :func:`_segment_suspects` measuring text that cannot have been
    # spoken in the time it is stamped on. Which pass then ships is decided by
    # measurement, not by which one ran second — see :func:`_choose_asr_pass`.
    chosen_primed = options["initial_prompt"] is not None
    chosen_options = options
    transcript, terminals = _terminal_count(segments)
    suspects = _segment_suspects(segments)
    attractor = terminals == 0 and len(transcript.split()) >= 40

    if not chosen_primed and (attractor or suspects):
        symptom = (
            "the unpunctuated-attractor signature and unusual segments"
            if attractor and suspects
            else (
                "the unpunctuated-attractor signature"
                if attractor
                else "unusual segments"
            )
        )
        logger.warning(
            f"⚠️ v2 transcription: re-running once WITH the punctuation primer — "
            f"unprimed pass shows {symptom} ({len(transcript.split())} words, "
            f"{terminals} terminals, {len(suspects)} unusual segment(s))"
        )
        primed_options = dict(options, initial_prompt=asr_primer_for(source_lang))
        try:
            primed_segments, primed_words, primed_info = _materialize(primed_options)
        except Exception as exc:  # noqa: BLE001 - a second opinion may never fail a job
            logger.warning(
                f"⚠️ v2 transcription: the primed re-run failed ({exc}) — keeping the "
                f"unprimed pass"
            )
        else:
            primed_suspects = _segment_suspects(primed_segments)
            choice, why = _choose_asr_pass(
                len(suspects), len(primed_suspects), attractor=attractor
            )
            # THE audit line: every later "why did this clip ship what it shipped?"
            # starts here, so it carries both measurements and the rule that used them.
            logger.info(
                f"🎚️ v2 ASR pass selection: chose the {choice} pass — {why} "
                f"(unprimed: {len(segments)} segments/{len(suspects)} unusual; "
                f"primed: {len(primed_segments)} segments/{len(primed_suspects)} unusual)"
            )
            if choice == "primed":
                chosen_primed = True
                chosen_options = primed_options
                segments, words, info = primed_segments, primed_words, primed_info
                suspects = primed_suspects
                detected_language = info.language
        transcript, terminals = _terminal_count(segments)

    # Suspect segments are REPORTED, never removed — see :func:`_segment_suspects` for
    # the two measurements that ended the deletion (it never fired across 7 real runs,
    # and a one-LSB audio change flips whole sentences, so a rate is not evidence that
    # nobody spoke). The counts have already done their only job: choosing between two
    # complete transcripts. What is left is a grep for the next reviewer.
    for suspect in suspects:
        logger.warning(
            f"🔎 v2 ASR: segment {float(suspect.get('start', 0) or 0):.2f}-"
            f"{float(suspect.get('end', 0) or 0):.2f}s looks unusual "
            f"[{suspect.get('suspect_reason')}] — KEPT (this is a measurement, not a "
            f"verdict; nothing is deleted on it): "
            f"{str(suspect.get('text') or '').strip()[:80]!r}"
        )

    # === Coverage invariant: audio the decoder was handed and did not describe ===
    #
    # THE MEASURED FAILURE: on one clip the ASR stopped 5.25s before the end while the
    # uncovered tail was LOUDER than the file average (-14.4 dB vs -19.3 dB). Nothing
    # flagged it — a transcript that is merely SHORT looks exactly like a transcript.
    # Re-decoding just that region recovered the missing sentence verbatim. This only
    # ever appends; it never deletes and never replaces.
    gap = _uncovered_tail(segments, audio_np, 16000)
    if gap is not None:
        logger.warning(
            f"🕳️ v2 ASR coverage: {gap['duration']:.2f}s of audio after the last "
            f"segment ({gap['start']:.2f}-{gap['end']:.2f}s) is not described by any "
            f"segment, and it is at {gap['db']:+.1f} dB against the file — that is "
            f"speech level. Re-decoding just that region."
        )
        try:
            region = audio_np[int(gap["start"] * 16000) :]
            # No conditioning: the region is decoded on its own, so there is no
            # previous output to carry a loop or a style across the seam into it.
            tail_segments, tail_words, _tail_info = _materialize(
                dict(chosen_options, condition_on_previous_text=False),
                audio=region,
            )
        except Exception as exc:  # noqa: BLE001 - a recovery may never fail a job
            logger.warning(
                f"⚠️ v2 ASR coverage: the tail re-decode failed ({exc}) — keeping the "
                f"transcript as it is"
            )
        else:
            previous_text = str(segments[-1].get("text") or "") if segments else ""
            usable, why_not = _recovery_is_new(tail_segments, previous_text)
            if not usable:
                logger.warning(f"🕳️ v2 ASR coverage: nothing appended — {why_not}")
            else:
                tail_segments = _shifted(tail_segments, gap["start"], ("start", "end"))
                tail_words = _shifted(tail_words, gap["start"], ("s", "e"))
                segments = list(segments) + tail_segments
                words = list(words) + tail_words
                recovered_text = " ".join(
                    str(s.get("text") or "").strip() for s in tail_segments
                )[:160]
                logger.warning(
                    f"🕳️ v2 ASR coverage: RECOVERED {len(tail_segments)} segment(s) / "
                    f"{len(tail_words)} word(s) from the uncovered tail: "
                    f"{recovered_text!r}"
                )
                transcript, terminals = _terminal_count(segments)

    transcription_duration = time.time() - transcription_start
    performance_monitor.log_transcription_performance(
        audio_duration, transcription_duration, model_name, segments_count=len(segments)
    )

    if collect_words and not words:
        logger.warning(
            f"⚠️ v2 transcription: Whisper returned no word timestamps for "
            f"{len(segments)} segments"
        )

    # Segments whose text says more than their words do — see :func:`_word_loss_report`.
    # Nothing is filtered here; this is the log line that turns "the delivered file lost
    # a word and inverted a sentence" from an audit finding into a grep.
    for loss in _word_loss_report(segments, words):
        logger.warning(
            f"⚠️ v2 ASR word loss: segment {loss['start']:.2f}-{loss['end']:.2f}s has "
            f"{loss['text_tokens']} text tokens but only {loss['word_tokens']} word "
            f"timestamps ({loss['gap']} missing) — spotting rebuilds cues from WORDS, "
            f"so those tokens will not reach the viewer"
        )

    # Punctuation health of the transcript we actually got. This is the read-out that
    # tells a reviewer, from the job log alone, whether the attractor was avoided on
    # this clip — the numbers the ablation is scored on, measured live on every run.
    commas = transcript.count(",")
    capitals = sum(1 for ch in transcript if ch.isupper())
    logger.info(
        f"📝 v2 ASR punctuation health: {terminals} terminals, {commas} commas, "
        f"{capitals} capitals over {len(transcript)} chars "
        f"(priming={'off' if not chosen_primed else 'on'})"
    )
    if segments and terminals == 0:
        logger.warning(
            f"⚠️ v2 ASR returned ZERO terminal punctuation across {len(segments)} "
            f"segments despite every configured measure — the unpunctuated attractor "
            f"was not escaped on this clip; downstream spotting will fall back to "
            f"speech pauses"
        )

    logger.info(
        f"✅ v2 transcription complete: {len(segments)} segments, {len(words)} words "
        f"in {transcription_duration:.1f}s"
    )

    if progress_callback:
        progress_callback(
            90, "Transcription completed", 90, "Step 1: Processing results", 5
        )

    return {
        # Whether the pass that actually SHIPPED carried a primer — not whether one was
        # configured, and not whether a primed run happened. This used to be a hardcoded
        # True, which made it a lie on every unprimed `large` job and therefore made
        # ``words_to_cues``'s "unpunctuated DESPITE the primer" signal — the whole reason
        # the flag is passed down — report the opposite of what occurred.
        "asr_primed": chosen_primed,
        "segments": segments,
        "words": words,
        "language": detected_language,
    }


def translate_segments(
    segments, target_language, service="google", progress_callback=None
):
    """Translate segments using the specified translation service."""
    if not segments or not target_language:
        return segments

    try:
        # FAKE mode: produce deterministic translations locally (no network)
        if config.USE_FAKE_YTDLP:
            for segment in segments:
                base_text = segment.get("text", "")
                segment["translated_text"] = (
                    base_text if target_language == "en" else f"{base_text}"
                )
            return segments

        if progress_callback:
            progress_callback(
                52,
                "Preparing text for translation...",
                30,
                "Step 2: Text preparation",
                5,
            )

        original_texts = [segment["text"] for segment in segments]

        if progress_callback:
            progress_callback(
                54,
                f"Connecting to {service.capitalize()}...",
                45,
                f"Step 2: Connecting to {service.capitalize()}",
                5,
            )

        translator = get_translator(service)

        if progress_callback:
            progress_callback(
                57, "Translating text...", 65, "Step 2: Active translation", 5
            )

        translated_texts = translator.translate_batch(original_texts, target_language)

        if progress_callback:
            progress_callback(
                62, "Processing translations...", 85, "Step 2: Processing results", 5
            )

        # Flexible validation: allow minor mismatches but handle them gracefully
        if not translated_texts:
            raise TranslationServiceError(
                service, "Translation service returned no results"
            )

        # Handle length mismatches gracefully
        if len(translated_texts) != len(original_texts):
            logger.warning(
                f"Translation count mismatch: expected {len(original_texts)}, got {len(translated_texts)}"
            )

            if len(translated_texts) > len(original_texts):
                # Trim excess translations
                logger.warning(
                    f"Trimming {len(translated_texts) - len(original_texts)} excess translations"
                )
                translated_texts = translated_texts[: len(original_texts)]
            elif len(translated_texts) < len(original_texts):
                # Fill missing translations with original text
                missing_count = len(original_texts) - len(translated_texts)
                logger.warning(
                    f"Filling {missing_count} missing translations with original text"
                )
                for i in range(len(translated_texts), len(original_texts)):
                    translated_texts.append(original_texts[i])

        # Final sanity check
        if len(translated_texts) != len(original_texts):
            raise TranslationServiceError(
                service,
                f"Cannot reconcile translation count: expected {len(original_texts)}, final {len(translated_texts)}",
            )

        for i, segment in enumerate(segments):
            segment["translated_text"] = translated_texts[i]

        if progress_callback:
            progress_callback(
                64,
                "Translation completed successfully",
                100,
                "Step 2: Saving results",
                5,
            )

        logger.info(
            f"✅ Translated {len(segments)} segments to {target_language} using {service}."
        )
        return segments

    except Exception as e:
        # NO source-text fallback, and no green job on a failed translation. This
        # `except` used to write `segment["text"]` into `translated_text` for every
        # segment and RETURN — the caller could not tell the difference, so the user got
        # an untranslated .srt and a successful job. See :func:`_translation_failed`.
        logger.error(
            f"Translation with {service} failed for language '{target_language}': {e}. "
            f"Failing the job and salvaging {len(segments)} transcribed segment(s) — "
            f"the source text is NOT substituted for a translation."
        )
        for segment in segments:
            segment.pop("translated_text", None)
        raise _translation_failed(
            f"Translation with {service} failed for '{target_language}': {e}",
            segments=segments,
        ) from e
