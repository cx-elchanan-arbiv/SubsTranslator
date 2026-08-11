"""
The guards standing between raw Whisper output and the v2 subtitle pipeline.

Every one of them exists because a measured defect reached a delivered file: text
invented into trailing silence, a decode loop repeating one clause, a word present in a
segment's text but missing from its word timestamps — and, in the other direction, a
transcript that simply STOPPED 5.25 seconds early over audio that was louder than the
programme.

**None of them deletes transcript any more.** The deletion that used to sit here never
fired once across 7 real runs, and the quality win credited to it came from pass
selection, which is non-destructive. What ended the argument is the input sensitivity of
the model itself: one sample in 624,153 changed by a single LSB — inaudible — flips 11
words of transcript, reproduced 3/3 in both directions. A rate measured off one decode
of one file cannot justify deleting what a person may have said.

The helpers under test are deliberately PURE — arrays and dicts in, arrays and dicts
out — precisely so the rules that decide what a viewer sees can be pinned here rather
than inferred from a job log.
"""

import os
import sys

import numpy as np
import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.subtitle_engine import (  # noqa: E402
    MAX_SOURCE_WORDS_PER_SEC,
    RATE_SIGNAL_MIN_DUR,
)
from services.transcription_service import (  # noqa: E402
    COVERAGE_MAX_QUIET_DB,
    DITHER_FLOOR,
    REPEAT_LOOP_MIN_WORDS,
    _choose_asr_pass,
    _recovery_is_new,
    _relative_db,
    _segment_suspects,
    _shifted,
    _trim_trailing_silence,
    _uncovered_tail,
    _word_loss_report,
)

SR = 16000


def _silence(seconds, sample_rate=SR):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _tone(seconds, amplitude=0.5, sample_rate=SR):
    """A loud, unambiguous block of "speech" — every sample above any silence floor."""
    return np.full(int(seconds * sample_rate), amplitude, dtype=np.float32)


def _segment(text, start, end):
    return {"start": start, "end": end, "text": text}


def _words(spec):
    """``[(start, end, word), ...]`` -> the word dicts the v2 path carries."""
    return [{"s": s, "e": e, "w": w} for s, e, w in spec]


# =============================================================================
# A1 — trailing-silence trim
# =============================================================================
@pytest.mark.unit
class TestTrimTrailingSilence:
    """0.72s of tail silence produced "It stuck. That's what it's all about." — invented.

    Whisper must emit tokens for a decode window whether or not anyone spoke in it, so
    silence at the end of the file is the cheapest hallucination there is.
    """

    def test_a_silent_tail_is_cut_back_to_the_pad(self):
        audio = np.concatenate([_tone(1.0), _silence(2.0)])
        trimmed, seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)

        assert len(trimmed) == pytest.approx(1.35 * SR, abs=2)
        assert seconds == pytest.approx(1.65, abs=0.01)

    def test_the_pad_is_actually_left_behind(self):
        """The final word's release lives in that pad; an abrupt cut truncates it."""
        audio = np.concatenate([_tone(1.0), _silence(2.0)])
        trimmed, _seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)
        assert len(trimmed) > len(_tone(1.0)), "the pad was not kept"

    def test_audio_with_no_speech_at_all_is_returned_untouched(self):
        """A silent or near-silent track is not something to truncate to nothing."""
        audio = _silence(3.0)
        trimmed, seconds = _trim_trailing_silence(audio, SR)
        assert seconds == 0.0
        assert len(trimmed) == len(audio)

    def test_a_tail_shorter_than_the_pad_is_left_alone(self):
        audio = np.concatenate([_tone(1.0), _silence(0.2)])
        trimmed, seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)
        assert seconds == 0.0
        assert len(trimmed) == len(audio)

    def test_leading_silence_is_never_trimmed(self):
        """The front is where a quiet first syllable lives — this path has lost one before."""
        audio = np.concatenate([_silence(1.0), _tone(1.0), _silence(2.0)])
        trimmed, seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)

        assert seconds == pytest.approx(1.65, abs=0.01)
        assert len(trimmed) == pytest.approx(2.35 * SR, abs=2)
        assert float(np.abs(trimmed[: int(1.0 * SR)]).max()) == 0.0

    def test_speech_running_to_the_very_end_is_untouched(self):
        audio = _tone(2.0)
        trimmed, seconds = _trim_trailing_silence(audio, SR)
        assert seconds == 0.0
        assert len(trimmed) == len(audio)

    def test_empty_audio_is_not_an_error(self):
        trimmed, seconds = _trim_trailing_silence(np.array([], dtype=np.float32), SR)
        assert seconds == 0.0
        assert len(trimmed) == 0

    def test_room_tone_far_under_the_programme_still_counts_as_silence(self):
        """The threshold is what separates "quiet" from "nothing"."""
        audio = np.concatenate(
            [_tone(1.0), np.full(int(2.0 * SR), 0.001, dtype=np.float32)]
        )
        _trimmed, seconds = _trim_trailing_silence(audio, SR)
        assert seconds == pytest.approx(1.65, abs=0.01)

    def test_a_quiet_recording_does_not_lose_its_last_word(self):
        """The defect an ABSOLUTE threshold has: it is a claim about recording level.

        1s of speech at 0.02, then a final quiet word at 0.002 — under the old absolute
        0.004 floor, so the old rule called it silence and cut it off. Everything here
        is real speech; only the level is low.
        """
        audio = np.concatenate(
            [
                _tone(1.0, amplitude=0.02),
                _tone(0.5, amplitude=0.002),
                _silence(2.0),
            ]
        )
        trimmed, seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)

        assert len(trimmed) / SR == pytest.approx(1.85, abs=0.01), "the last word went"
        assert seconds == pytest.approx(1.65, abs=0.01)

    def test_the_threshold_scales_with_the_file(self):
        """Same waveform, two levels, same decision. That is what 'relative' means."""
        loud = np.concatenate([_tone(1.0, amplitude=0.5), _silence(2.0)])
        quiet = np.concatenate([_tone(1.0, amplitude=0.005), _silence(2.0)])

        assert _trim_trailing_silence(loud, SR)[1] == pytest.approx(
            _trim_trailing_silence(quiet, SR)[1], abs=1e-6
        )

    def test_dither_on_a_silent_tail_is_still_silence(self):
        """A digitally silent tail carries ±1 LSB, which is inaudible and not speech.

        Without the absolute floor under the relative threshold, a quiet file's
        threshold falls beneath its own dither and the trim never fires.
        """
        rng = np.random.default_rng(0)
        dither = rng.choice(
            np.array([-1.0, 0.0, 1.0], dtype=np.float32) / 32768.0, int(2.0 * SR)
        ).astype(np.float32)
        audio = np.concatenate([_tone(1.0, amplitude=0.01), dither])

        _trimmed, seconds = _trim_trailing_silence(audio, SR, pad_s=0.35)
        assert float(np.abs(dither).max()) <= DITHER_FLOOR
        assert seconds == pytest.approx(1.65, abs=0.01)


# =============================================================================
# A2 — raw-segment fabrication signatures
# =============================================================================
@pytest.mark.unit
class TestSegmentSuspects:
    """DETECTION ONLY. These flags choose between two complete transcripts and then log.

    They used to delete the segments they flagged. That deletion never fired once across
    7 real runs, and Whisper is chaotically input-sensitive — a one-LSB change to a
    single sample flips whole sentences — so a rate is a suspicion, never a verdict.
    """

    def test_a_fast_speech_rate_is_suspect(self):
        # 25 words in 2.5s = 10 w/s, over the bound and over the duration floor.
        text = " ".join(f"word{i}" for i in range(25))
        suspects = _segment_suspects([_segment(text, 0.0, 2.5)])

        assert len(suspects) == 1
        assert suspects[0]["suspect_reason"].startswith("fast_wps")
        assert suspects[0]["index"] == 0

    def test_the_bound_is_the_engine_constant_not_a_local_copy(self):
        """One number, one place: a segment just over it is suspect, just under is not."""
        over = " ".join(["w"] * int(MAX_SOURCE_WORDS_PER_SEC * 2.5 + 2))
        under = " ".join(["w"] * int(MAX_SOURCE_WORDS_PER_SEC * 2.5 - 2))

        assert len(_segment_suspects([_segment(over, 0.0, 2.5)])) == 1
        assert _segment_suspects([_segment(under, 0.0, 2.5)]) == []

    def test_no_rate_is_computed_below_the_duration_floor(self):
        """9 words in 0.44s is 20 w/s — and it is not a measurement of anything.

        Whisper quantises word times to 0.02s and its error is roughly constant in
        absolute time, so on a sub-second span the rate describes the quantiser. This is
        the same floor the cue-level gate applies, from the same constant.
        """
        assert RATE_SIGNAL_MIN_DUR == 2.0
        text = "it stuck that is what it is all about"
        assert _segment_suspects([_segment(text, 10.0, 10.44)]) == []

    def test_real_speech_at_five_words_a_second_passes(self):
        text = " ".join(f"word{i}" for i in range(10))
        assert _segment_suspects([_segment(text, 0.0, 2.0)]) == []

    def test_the_corpus_fastest_genuine_cue_passes(self):
        """7.03 w/s, real cross-talk. The bound now clears it by 28%, not by 3%."""
        text = "and they were screaming and yelling at each other."
        assert len(text.split()) / (10.06 - 8.78) == pytest.approx(7.03, abs=0.01)
        assert _segment_suspects([_segment(text, 8.78, 10.06)]) == []

    def test_a_verbatim_repeat_of_the_previous_segment_is_suspect(self):
        segments = [
            _segment("Thanks for watching my video.", 0.0, 3.0),
            _segment("thanks for watching my video", 3.0, 6.0),
        ]
        suspects = _segment_suspects(segments)

        assert [s["index"] for s in suspects] == [1]
        assert suspects[0]["suspect_reason"] == "repeat_of_previous"

    def test_a_short_repetition_is_speech_not_a_decode_loop(self):
        """ "No. No." is a person insisting. A decode loop repeats a whole clause."""
        assert REPEAT_LOOP_MIN_WORDS == 4
        segments = [_segment("No.", 0.0, 1.0), _segment("No.", 1.2, 2.2)]
        assert _segment_suspects(segments) == []

    def test_a_repeat_two_segments_apart_is_not_a_loop(self):
        """Only the immediate repeat is the decode loop; recurrence is just speech."""
        segments = [
            _segment("Yes I really do think so.", 0.0, 3.0),
            _segment("Something else entirely here.", 3.0, 6.0),
            _segment("Yes I really do think so.", 6.0, 9.0),
        ]
        assert _segment_suspects(segments) == []

    def test_zero_duration_with_text_is_NOT_suspect(self):
        """The sibling gate KEEPS zero-length cues; two gates must not disagree.

        ``subtitle_engine.drop_hallucinated_cues`` keeps them because "zero length only
        means arithmetic infinity, not that the text is fake". Reading the same signal
        in the opposite direction here was one bug, not two opinions.
        """
        assert _segment_suspects([_segment("Words on no time at all.", 5.0, 5.0)]) == []

    def test_negative_duration_is_not_suspect_either(self):
        assert _segment_suspects([_segment("Backwards.", 5.0, 4.0)]) == []

    def test_a_blank_segment_is_never_suspect(self):
        """There is nothing fabricated in a segment with no words."""
        assert _segment_suspects([_segment("   ", 5.0, 5.0)]) == []

    def test_two_blank_segments_are_not_a_repeat_loop(self):
        segments = [_segment("", 0.0, 1.0), _segment("  ", 1.0, 2.0)]
        assert _segment_suspects(segments) == []

    def test_the_input_is_not_mutated(self):
        segments = [_segment(" ".join(f"w{i}" for i in range(25)), 0.0, 2.5)]
        before = [dict(s) for s in segments]
        _segment_suspects(segments)
        assert segments == before

    def test_no_segments_is_no_suspects(self):
        assert _segment_suspects([]) == []
        assert _segment_suspects(None) == []


# =============================================================================
# A3 — which of the two ASR passes ships
# =============================================================================
@pytest.mark.unit
class TestChooseAsrPass:
    """The primer is known to change what the model HEARS, so it is measured, not trusted."""

    def test_the_pass_with_fewer_fabrications_wins(self):
        choice, why = _choose_asr_pass(3, 1, attractor=False)
        assert choice == "primed"
        assert "1" in why and "3" in why

    def test_a_primed_pass_that_invented_more_is_discarded(self):
        choice, _why = _choose_asr_pass(1, 4, attractor=False)
        assert choice == "unprimed"

    def test_a_tie_goes_to_the_unprimed_first_pass(self):
        """A re-run that bought nothing is not paid for in decode drift."""
        assert _choose_asr_pass(2, 2, attractor=False)[0] == "unprimed"
        assert _choose_asr_pass(0, 0, attractor=False)[0] == "unprimed"

    def test_a_tie_under_the_attractor_keeps_the_primed_pass(self):
        """That re-run was called for zero terminal punctuation, and it fixed it.

        Without this exception the tie-break would silently undo the whole
        unpunctuated-attractor escape, which is the older and better-evidenced of the
        two reasons this re-run exists.
        """
        assert _choose_asr_pass(0, 0, attractor=True)[0] == "primed"

    def test_fabrication_still_outranks_the_attractor(self):
        """Escaping the attractor is not worth buying invented dialogue."""
        assert _choose_asr_pass(0, 2, attractor=True)[0] == "unprimed"

    def test_every_outcome_explains_itself(self):
        for args in ((3, 1), (1, 3), (2, 2)):
            for attractor in (True, False):
                _choice, why = _choose_asr_pass(*args, attractor=attractor)
                assert isinstance(why, str) and why.strip()


# =============================================================================
# B — the coverage invariant: the OPPOSITE of a deletion rule
# =============================================================================
@pytest.mark.unit
class TestUncoveredTail:
    """The ASR stopped 5.25s early over audio LOUDER than the programme (-14.4 vs -19.3).

    A transcript that is merely SHORT looks exactly like a transcript, so nothing
    anywhere reported it. Re-decoding that one region recovered the missing sentence
    verbatim ("if you enjoyed the video don't forget to like and subscribe").
    """

    def test_a_loud_uncovered_tail_is_reported(self):
        audio = _tone(10.0)
        segments = [_segment("Everything up to here.", 0.0, 4.75)]

        gap = _uncovered_tail(segments, audio, SR)

        assert gap is not None
        assert gap["start"] == pytest.approx(4.75)
        assert gap["end"] == pytest.approx(10.0)
        assert gap["duration"] == pytest.approx(5.25)
        assert gap["db"] == pytest.approx(0.0, abs=0.5)

    def test_a_fully_covered_transcript_reports_nothing(self):
        audio = _tone(10.0)
        assert _uncovered_tail([_segment("All of it.", 0.0, 10.0)], audio, SR) is None

    def test_a_sub_second_gap_is_ordinary_and_ignored(self):
        """The final consonant's release, and the pad the trim deliberately leaves."""
        audio = _tone(10.0)
        assert _uncovered_tail([_segment("x", 0.0, 9.3)], audio, SR) is None

    def test_a_quiet_tail_is_a_fade_out_not_a_gap(self):
        audio = np.concatenate([_tone(5.0, amplitude=0.5), _tone(5.0, amplitude=0.001)])
        assert _uncovered_tail([_segment("x", 0.0, 5.0)], audio, SR) is None

    def test_the_quiet_bound_is_generous_on_purpose(self):
        """This mechanism only ever ADDS; the expensive error is the missing sentence."""
        assert COVERAGE_MAX_QUIET_DB == -20.0
        audio = np.concatenate([_tone(5.0, amplitude=0.5), _tone(5.0, amplitude=0.1)])
        gap = _uncovered_tail([_segment("x", 0.0, 5.0)], audio, SR)
        assert gap is not None and gap["db"] < 0

    def test_no_segments_at_all_is_the_whole_file(self):
        gap = _uncovered_tail([], _tone(10.0), SR)
        assert gap is not None and gap["start"] == 0.0

    def test_a_segment_running_past_the_audio_is_not_a_negative_gap(self):
        assert _uncovered_tail([_segment("x", 0.0, 99.0)], _tone(10.0), SR) is None

    def test_empty_audio_is_not_an_error(self):
        assert _uncovered_tail([_segment("x", 0.0, 1.0)], np.array([]), SR) is None

    def test_it_changes_nothing(self):
        segments = [_segment("x", 0.0, 1.0)]
        before = [dict(s) for s in segments]
        _uncovered_tail(segments, _tone(10.0), SR)
        assert segments == before


@pytest.mark.unit
class TestRelativeDb:
    """Level is always measured against the file itself — nothing here sets the gain."""

    def test_the_same_level_is_zero_db(self):
        audio = _tone(2.0)
        assert _relative_db(audio, audio) == pytest.approx(0.0, abs=1e-6)

    def test_half_the_amplitude_is_about_minus_six_db(self):
        whole = _tone(2.0, amplitude=0.5)
        region = _tone(1.0, amplitude=0.25)
        assert _relative_db(region, whole) == pytest.approx(-6.02, abs=0.05)

    def test_digital_silence_is_unmeasurable_not_minus_infinity(self):
        assert _relative_db(_silence(1.0), _tone(2.0)) is None

    def test_empty_inputs_are_unmeasurable(self):
        assert _relative_db(np.array([]), _tone(1.0)) is None
        assert _relative_db(_tone(1.0), np.array([])) is None


@pytest.mark.unit
class TestShifted:
    """A region decode returns region-relative times; they mean nothing until moved."""

    def test_segment_times_move_by_the_offset(self):
        moved = _shifted([_segment("Recovered.", 0.0, 2.0)], 100.0, ("start", "end"))
        assert moved[0]["start"] == 100.0 and moved[0]["end"] == 102.0
        assert moved[0]["text"] == "Recovered."

    def test_word_times_use_their_own_keys(self):
        moved = _shifted(_words([(0.0, 0.5, "hi")]), 10.0, ("s", "e"))
        assert moved[0]["s"] == 10.0 and moved[0]["e"] == 10.5

    def test_the_input_is_not_mutated(self):
        original = [_segment("x", 0.0, 1.0)]
        _shifted(original, 5.0, ("start", "end"))
        assert original[0]["start"] == 0.0

    def test_unparseable_times_are_left_alone_rather_than_crashing(self):
        moved = _shifted([{"start": None, "end": "x"}], 5.0, ("start", "end"))
        assert moved == [{"start": 5.0, "end": "x"}]


@pytest.mark.unit
class TestRecoveryIsNew:
    """A recovery that adds a duplicate is worse than the gap it was filling."""

    def test_real_recovered_text_is_appended(self):
        ok, why = _recovery_is_new(
            [_segment("if you enjoyed the video don't forget to subscribe", 0.0, 3.0)],
            "That is all I wanted to say.",
        )
        assert ok is True and why == ""

    def test_an_empty_re_decode_is_skipped(self):
        ok, why = _recovery_is_new([_segment("   ", 0.0, 3.0)], "Anything.")
        assert ok is False and "no text" in why

    def test_no_segments_at_all_is_skipped(self):
        assert _recovery_is_new([], "Anything.")[0] is False

    def test_a_verbatim_repeat_of_the_segment_before_the_gap_is_skipped(self):
        """The decode loop reaching across the seam — punctuation and case vary."""
        ok, why = _recovery_is_new(
            [_segment("thanks for watching", 0.0, 3.0)], "Thanks for watching."
        )
        assert ok is False and "repeated" in why


# =============================================================================
# D1 — silent source-word loss
# =============================================================================
@pytest.mark.unit
class TestWordLossReport:
    """A profanity in a segment's text was missing from its words, so the cue lost it.

    The delivered file said the opposite of the source and nothing in the logs said a
    word had gone. This reports; it does not filter.
    """

    def test_a_missing_run_of_words_is_reported(self):
        segments = [_segment("one two three four five", 0.0, 2.0)]
        words = _words([(0.0, 0.4, "one"), (0.4, 0.8, "two")])

        report = _word_loss_report(segments, words)

        assert len(report) == 1
        assert report[0]["gap"] == 3
        assert report[0]["text_tokens"] == 5 and report[0]["word_tokens"] == 2
        assert report[0]["start"] == 0.0 and report[0]["end"] == 2.0

    def test_a_gap_of_one_is_tokenisation_noise_and_is_ignored(self):
        segments = [_segment("one two three", 0.0, 2.0)]
        words = _words([(0.0, 0.4, "one"), (0.4, 0.8, "two")])
        assert _word_loss_report(segments, words) == []

    def test_a_complete_segment_reports_nothing(self):
        segments = [_segment("one two three", 0.0, 2.0)]
        words = _words([(0.0, 0.4, "one"), (0.4, 0.8, "two"), (0.8, 1.2, "three")])
        assert _word_loss_report(segments, words) == []

    def test_words_outside_the_span_do_not_count_as_coverage(self):
        segments = [_segment("one two three four", 10.0, 12.0)]
        words = _words([(0.0, 0.4, "one"), (0.4, 0.8, "two")])
        assert _word_loss_report(segments, words)[0]["word_tokens"] == 0

    def test_no_words_at_all_is_the_gemini_case_and_reports_nothing(self):
        """No word timestamps is reported elsewhere; it is not a per-segment loss."""
        assert _word_loss_report([_segment("one two three", 0.0, 2.0)], []) == []

    def test_it_changes_nothing(self):
        segments = [_segment("one two three four five", 0.0, 2.0)]
        words = _words([(0.0, 0.4, "one")])
        before_segments = [dict(s) for s in segments]
        before_words = [dict(w) for w in words]

        _word_loss_report(segments, words)

        assert segments == before_segments and words == before_words


# =============================================================================
# E — the legacy path must fail visibly rather than ship the source as a translation
# =============================================================================
@pytest.mark.unit
class TestLegacyTranslationFailure:
    """An untranslated .srt under a green job is the worst outcome this pipeline has.

    It is indistinguishable from success until a human reads the file. All three legacy
    fallbacks that produced it now raise the same exception the v2 path already uses,
    carrying the transcript so the expensive stage is not thrown away.
    """

    @staticmethod
    def _segments():
        return [
            {"start": 0.0, "end": 1.0, "text": "Hello there."},
            {"start": 1.0, "end": 2.0, "text": "Second line."},
        ]

    @staticmethod
    def _with_a_dead_translator(monkeypatch):
        from services import transcription_service as svc

        monkeypatch.setattr(svc.config, "USE_FAKE_YTDLP", False, raising=False)
        monkeypatch.setattr(
            svc,
            "get_translator",
            lambda service: (_ for _ in ()).throw(RuntimeError("provider down")),
        )
        return svc

    def test_a_failing_translator_raises_instead_of_returning_the_source(
        self, monkeypatch
    ):
        from tasks.processing_tasks import TranslationFailedWithSalvage

        svc = self._with_a_dead_translator(monkeypatch)

        with pytest.raises(TranslationFailedWithSalvage) as caught:
            svc.translate_segments(self._segments(), "he", service="google")

        assert "provider down" in str(caught.value)

    def test_no_segment_is_left_claiming_a_translation_it_does_not_have(
        self, monkeypatch
    ):
        from tasks.processing_tasks import TranslationFailedWithSalvage

        svc = self._with_a_dead_translator(monkeypatch)
        segments = self._segments()

        with pytest.raises(TranslationFailedWithSalvage):
            svc.translate_segments(segments, "he", service="google")

        assert all("translated_text" not in s for s in segments)

    def test_the_transcript_rides_along_as_salvage(self, monkeypatch):
        from tasks.processing_tasks import TranslationFailedWithSalvage

        svc = self._with_a_dead_translator(monkeypatch)

        with pytest.raises(TranslationFailedWithSalvage) as caught:
            svc.translate_segments(self._segments(), "he", service="google")

        assert [s["text"] for s in caught.value.segments] == [
            "Hello there.",
            "Second line.",
        ]

    def test_the_exception_is_the_one_the_v2_path_already_raises(self):
        """Same class, so the existing salvage handler needs no second contract."""
        from services.transcription_service import _translation_failed
        from tasks.processing_tasks import TranslationFailedWithSalvage

        assert isinstance(_translation_failed("x", []), TranslationFailedWithSalvage)


# =============================================================================
# B (wiring) — the coverage recovery actually appends, on the real code path
# =============================================================================
class _FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class _FakeSegment:
    def __init__(self, start, end, text, words=()):
        self.start, self.end, self.text = start, end, text
        self.words = list(words)


class _FakeInfo:
    language = "en"


class _FakeModel:
    """Returns one canned segment list per ``transcribe`` call, recording each call."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append({"samples": len(audio), "options": dict(options)})
        index = min(len(self.calls) - 1, len(self._script) - 1)
        return iter(self._script[index]), _FakeInfo()


@pytest.mark.unit
class TestCoverageRecoveryWiring:
    """The ASR stopped 5.25s early on real audio and nothing said so.

    The pure helpers are pinned above; this pins the WIRING — that a detected gap
    actually produces a second decode whose output is appended with corrected
    timestamps, and that a covered transcript costs nothing.
    """

    @staticmethod
    def _install(monkeypatch, model, seconds=20.0):
        from services import transcription_service as svc

        audio = _tone(seconds)
        monkeypatch.setattr(svc.config, "USE_FAKE_YTDLP", False, raising=False)
        monkeypatch.setattr(
            svc, "_extract_audio_np", lambda path, cb=None: (audio, seconds)
        )
        monkeypatch.setattr(svc.smart_whisper, "load_model", lambda name: model)
        return svc

    def test_an_uncovered_tail_is_re_decoded_and_appended(self, monkeypatch):
        model = _FakeModel(
            [
                [
                    _FakeSegment(
                        0.0,
                        10.0,
                        "Everything up to here.",
                        [_FakeWord(0.0, 1.0, "Everything")],
                    )
                ],
                [
                    _FakeSegment(
                        0.0,
                        3.0,
                        "don't forget to like and subscribe.",
                        [_FakeWord(0.0, 1.0, "don't")],
                    )
                ],
            ]
        )
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert len(model.calls) == 2, "the tail was not re-decoded"
        assert [s["text"] for s in result["segments"]] == [
            "Everything up to here.",
            "don't forget to like and subscribe.",
        ]

    def test_the_recovered_timestamps_are_moved_back_to_where_the_audio_was(
        self, monkeypatch
    ):
        model = _FakeModel(
            [
                [_FakeSegment(0.0, 10.0, "First.", [_FakeWord(0.0, 1.0, "First.")])],
                [
                    _FakeSegment(
                        0.0,
                        3.0,
                        "and then the second thing happened.",
                        [_FakeWord(0.5, 1.5, "and")],
                    )
                ],
            ]
        )
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert result["segments"][1]["start"] == pytest.approx(10.0)
        assert result["segments"][1]["end"] == pytest.approx(13.0)
        assert result["words"][1]["s"] == pytest.approx(10.5)

    def test_a_one_word_recovery_is_refused(self, monkeypatch):
        """Measured: six firings, one genuine (11 words), five fragments.

        "more", "Uh-huh.", "country." — a decoder handed a window it cannot read
        emits the shortest thing that terminates, so a fragment cannot be told from
        an artefact. Skipping a real interjection is cheap; burning an invented one
        into the video is not.
        """
        model = _FakeModel(
            [
                [_FakeSegment(0.0, 10.0, "First.", [_FakeWord(0.0, 1.0, "First.")])],
                [_FakeSegment(0.0, 3.0, "more", [_FakeWord(0.5, 1.5, "more")])],
            ]
        )
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert len(result["segments"]) == 1
        assert "more" not in result["segments"][0]["text"]

    def test_the_re_decode_is_not_conditioned_on_the_previous_output(self, monkeypatch):
        """A region decoded on its own has no earlier output to carry a loop across."""
        model = _FakeModel(
            [
                [_FakeSegment(0.0, 10.0, "First.", [])],
                [_FakeSegment(0.0, 3.0, "Second.", [])],
            ]
        )
        svc = self._install(monkeypatch, model)
        svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert model.calls[0]["options"]["condition_on_previous_text"] is True
        assert model.calls[1]["options"]["condition_on_previous_text"] is False

    def test_only_the_uncovered_region_is_handed_to_the_second_decode(
        self, monkeypatch
    ):
        model = _FakeModel(
            [
                [_FakeSegment(0.0, 10.0, "First.", [])],
                [_FakeSegment(0.0, 3.0, "Second.", [])],
            ]
        )
        svc = self._install(monkeypatch, model, seconds=20.0)
        svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert model.calls[0]["samples"] == 20 * SR
        assert model.calls[1]["samples"] == 10 * SR

    def test_a_covered_transcript_costs_no_second_decode(self, monkeypatch):
        model = _FakeModel(
            [[_FakeSegment(0.0, 19.9, "All of it, right to the end.", [])]]
        )
        svc = self._install(monkeypatch, model)

        svc.transcribe_with_words("/x.mp4", model_preference="large")
        assert len(model.calls) == 1

    def test_a_repeated_recovery_is_skipped_rather_than_duplicated(self, monkeypatch):
        model = _FakeModel(
            [
                [_FakeSegment(0.0, 10.0, "Thanks for watching.", [])],
                [_FakeSegment(0.0, 3.0, "thanks for watching", [])],
            ]
        )
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")

        assert len(model.calls) == 2
        assert [s["text"] for s in result["segments"]] == ["Thanks for watching."]

    def test_an_empty_recovery_is_skipped(self, monkeypatch):
        model = _FakeModel([[_FakeSegment(0.0, 10.0, "First.", [])], []])
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")
        assert [s["text"] for s in result["segments"]] == ["First."]

    def test_a_failing_re_decode_never_fails_the_job(self, monkeypatch):
        class _Exploding(_FakeModel):
            def transcribe(self, audio, **options):
                if self.calls:
                    self.calls.append({"samples": len(audio), "options": options})
                    raise RuntimeError("CUDA out of memory")
                return super().transcribe(audio, **options)

        model = _Exploding([[_FakeSegment(0.0, 10.0, "First.", [])]])
        svc = self._install(monkeypatch, model)

        result = svc.transcribe_with_words("/x.mp4", model_preference="large")
        assert [s["text"] for s in result["segments"]] == ["First."]

    def test_a_suspect_segment_is_logged_and_KEPT(self, monkeypatch):
        """The deletion is gone. What is left is a grep for the next reviewer."""
        import structlog

        fast = " ".join(f"word{i}" for i in range(30))  # 12 w/s over 2.5s
        model = _FakeModel(
            [[_FakeSegment(0.0, 2.5, fast, []), _FakeSegment(2.5, 19.9, "Rest.", [])]]
        )
        svc = self._install(monkeypatch, model)

        with structlog.testing.capture_logs() as captured:
            result = svc.transcribe_with_words("/x.mp4", model_preference="large")
        logged = "\n".join(str(entry.get("event", "")) for entry in captured)

        assert len(result["segments"]) == 2, "a suspect segment was deleted"
        assert "KEPT" in logged and "not a verdict" in logged
        assert "dropping" not in logged and "fabricated" not in logged
