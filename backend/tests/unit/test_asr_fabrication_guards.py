"""
The guards standing between raw Whisper output and the v2 subtitle pipeline.

Every one of them exists because a fabrication reached a delivered file: text invented
into trailing silence, words stamped on time too short to have said them, a decode loop
repeating one sentence, and a word present in a segment's text but missing from its word
timestamps. The helpers under test are deliberately PURE — arrays and dicts in, arrays
and dicts out — precisely so the rules that decide what a viewer sees can be pinned here
rather than inferred from a job log.
"""

import os
import sys

import numpy as np
import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.subtitle_engine import IMPOSSIBLE_WORDS_PER_SEC  # noqa: E402
from services.transcription_service import (  # noqa: E402
    _choose_asr_pass,
    _drop_segments,
    _segment_suspects,
    _trim_trailing_silence,
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

    def test_room_tone_under_the_threshold_still_counts_as_silence(self):
        """The threshold is what separates "quiet" from "nothing" — 0.001 is nothing."""
        audio = np.concatenate(
            [_tone(1.0), np.full(int(2.0 * SR), 0.001, dtype=np.float32)]
        )
        _trimmed, seconds = _trim_trailing_silence(audio, SR, threshold=0.004)
        assert seconds == pytest.approx(1.65, abs=0.01)


# =============================================================================
# A2 — raw-segment fabrication signatures
# =============================================================================
@pytest.mark.unit
class TestSegmentSuspects:
    """Measured at CUE level the fabrications hide; measured on the RAW segment they do not.

    Re-spotting hands a cue the silence around it, which averages an impossible speech
    rate back down into the plausible range — 9 invented words in 0.44s survived the cue
    gate intact.
    """

    def test_an_impossible_speech_rate_is_suspect(self):
        # 15 words in 2.0s = 7.5 w/s — the slowest fabrication actually measured.
        text = " ".join(f"word{i}" for i in range(15))
        suspects = _segment_suspects([_segment(text, 0.0, 2.0)])

        assert len(suspects) == 1
        assert suspects[0]["suspect_reason"].startswith("impossible_wps")
        assert suspects[0]["index"] == 0

    def test_the_measured_20_words_per_second_case(self):
        """9 invented words in 0.44s — the run that reached a delivered file."""
        text = "it stuck that is what it is all about"
        suspects = _segment_suspects([_segment(text, 10.0, 10.44)])
        assert len(suspects) == 1

    def test_real_speech_at_five_words_a_second_passes(self):
        text = " ".join(f"word{i}" for i in range(10))
        assert _segment_suspects([_segment(text, 0.0, 2.0)]) == []

    def test_the_corpus_fastest_genuine_cue_passes(self):
        """7.03 w/s, real cross-talk. The threshold sits ABOVE this on purpose."""
        text = "and they were screaming and yelling at each other."
        assert len(text.split()) / (10.06 - 8.78) == pytest.approx(7.03, abs=0.01)
        assert _segment_suspects([_segment(text, 8.78, 10.06)]) == []

    def test_a_verbatim_repeat_of_the_previous_segment_is_suspect(self):
        segments = [
            _segment("Thanks for watching.", 0.0, 3.0),
            _segment("thanks for watching", 3.0, 6.0),
        ]
        suspects = _segment_suspects(segments)

        assert [s["index"] for s in suspects] == [1]
        assert suspects[0]["suspect_reason"] == "repeat_of_previous"

    def test_a_repeat_two_segments_apart_is_not_a_loop(self):
        """Only the immediate repeat is the decode loop; recurrence is just speech."""
        segments = [
            _segment("Yes.", 0.0, 3.0),
            _segment("Something else entirely.", 3.0, 6.0),
            _segment("Yes.", 6.0, 9.0),
        ]
        assert _segment_suspects(segments) == []

    def test_zero_duration_with_text_is_suspect(self):
        suspects = _segment_suspects([_segment("Words on no time at all.", 5.0, 5.0)])
        assert len(suspects) == 1
        assert suspects[0]["suspect_reason"] == "impossible_wps(no_duration)"

    def test_negative_duration_is_suspect(self):
        suspects = _segment_suspects([_segment("Backwards.", 5.0, 4.0)])
        assert len(suspects) == 1

    def test_a_blank_segment_is_never_suspect(self):
        """There is nothing fabricated in a segment with no words."""
        assert _segment_suspects([_segment("   ", 5.0, 5.0)]) == []

    def test_two_blank_segments_are_not_a_repeat_loop(self):
        segments = [_segment("", 0.0, 1.0), _segment("  ", 1.0, 2.0)]
        assert _segment_suspects(segments) == []

    def test_the_input_is_not_mutated(self):
        segments = [_segment("Words on no time at all.", 5.0, 5.0)]
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


@pytest.mark.unit
class TestDropSegments:
    """A dropped segment whose WORDS stay behind is a fabrication that ships anyway.

    ``words_to_cues`` spots from the word stream alone, so removing the segment without
    removing its words puts the invented text straight back on screen.
    """

    def test_the_suspect_segment_and_its_words_both_go(self):
        segments = [
            _segment("Real speech here.", 0.0, 2.0),
            _segment("Invented.", 2.0, 2.1),
            _segment("More real speech.", 3.0, 5.0),
        ]
        words = _words(
            [(0.0, 1.0, "Real"), (2.0, 2.1, "Invented."), (3.0, 4.0, "More")]
        )
        suspects = [dict(segments[1], suspect_reason="impossible_wps(x)", index=1)]

        kept_segments, kept_words = _drop_segments(segments, words, suspects)

        assert [s["text"] for s in kept_segments] == [
            "Real speech here.",
            "More real speech.",
        ]
        assert [w["w"] for w in kept_words] == ["Real", "More"]

    def test_a_zero_length_span_still_takes_its_words(self):
        segments = [_segment("Phantom.", 5.0, 5.0)]
        words = _words([(5.0, 5.0, "Phantom.")])
        suspects = [dict(segments[0], suspect_reason="x", index=0)]

        kept_segments, kept_words = _drop_segments(segments, words, suspects)
        assert kept_segments == [] and kept_words == []

    def test_nothing_suspect_changes_nothing(self):
        segments = [_segment("Real.", 0.0, 2.0)]
        words = _words([(0.0, 1.0, "Real.")])
        kept_segments, kept_words = _drop_segments(segments, words, [])
        assert kept_segments == segments and kept_words == words

    def test_the_input_lists_are_not_mutated(self):
        segments = [_segment("a", 0.0, 1.0), _segment("b", 1.0, 2.0)]
        words = _words([(0.0, 1.0, "a"), (1.0, 2.0, "b")])
        suspects = [dict(segments[1], suspect_reason="x", index=1)]

        _drop_segments(segments, words, suspects)
        assert len(segments) == 2 and len(words) == 2


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
# The shared threshold
# =============================================================================
@pytest.mark.unit
class TestImpossibleWordsPerSecIsShared:
    def test_the_segment_gate_uses_the_engine_constant(self):
        """One number, one place. A segment just over it is suspect; just under is not."""
        over = " ".join(["w"] * int(IMPOSSIBLE_WORDS_PER_SEC * 2 + 2))
        under = " ".join(["w"] * int(IMPOSSIBLE_WORDS_PER_SEC * 2 - 2))

        assert len(_segment_suspects([_segment(over, 0.0, 2.0)])) == 1
        assert _segment_suspects([_segment(under, 0.0, 2.0)]) == []
