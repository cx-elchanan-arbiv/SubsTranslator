"""
Unit tests for services/subtitle_engine.py — the subtitle spotting engine.

The fixture-driven tests run on real Whisper word timestamps captured from the
30s Fox News interview that exposed the defects this engine removes
(tests/fixtures/words_sample.json, 83 words). They assert the properties a
professional editor's output has and ours did not:

  * no unreadable flash cues (< 1.2s),
  * a question is never glued to its answer,
  * mid-sentence fragments ("things.") are folded back into their sentence,
  * Netflix Hebrew Timed Text limits: <= 2 lines, <= 42 chars per line.
"""
import json
import sys
from pathlib import Path

import pytest

backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from services.subtitle_engine import (  # noqa: E402
    LRI,
    MAX_LINE_CHARS,
    MIN_CUE_DUR,
    PDI,
    RLI,
    STYLE_NAME,
    bidi_isolate,
    build_ass,
    gershayim,
    words_to_cues,
    wrap_two_lines,
)

WORDS_SAMPLE = Path(__file__).resolve().parent.parent / "fixtures" / "words_sample.json"

RLO = "\u202e"  # RIGHT-TO-LEFT OVERRIDE — must never be emitted


@pytest.fixture(scope="module")
def sample_words():
    """Real word timestamps from the measured 30s clip."""
    with open(WORDS_SAMPLE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_cues(sample_words):
    """Cues produced from the real fixture with production defaults."""
    return words_to_cues(sample_words)


@pytest.mark.unit
class TestWordsToCuesOnRealClip:
    """words_to_cues() on real Whisper word timestamps."""

    def test_fixture_is_the_measured_clip(self, sample_words):
        assert len(sample_words) == 83

    def test_no_flash_cues(self, sample_cues):
        # The defect: a 0.3s cue holding only "things." and a 0.2s "Yeah.".
        short = [c for c in sample_cues if c["end"] - c["start"] < MIN_CUE_DUR]
        assert short == [], f"cues shorter than {MIN_CUE_DUR}s: {short}"

    def test_monotonic_and_non_overlapping(self, sample_cues):
        assert len(sample_cues) > 1
        for previous, following in zip(sample_cues, sample_cues[1:]):
            assert previous["start"] < previous["end"]
            assert previous["end"] <= following["start"], (previous, following)

    def test_question_ends_its_cue_and_answer_starts_a_new_one(self, sample_cues):
        idx = next(
            i
            for i, c in enumerate(sample_cues)
            if "Do you worry about that?" in c["text"]
        )
        assert sample_cues[idx]["text"].endswith("Do you worry about that?")
        # The answer must never share the cue with the question.
        assert sample_cues[idx + 1]["text"].startswith("Yeah, I think about it.")

    def test_short_fragment_is_merged_into_its_sentence(self, sample_cues):
        texts = [c["text"] for c in sample_cues]
        assert "It could complicate things." not in texts, "0.94s fragment left alone"
        holder = next(t for t in texts if "complicate things." in t)
        assert "Do you worry about that?" in holder

    def test_netflix_line_limits(self, sample_cues):
        for cue in sample_cues:
            lines = wrap_two_lines(cue["text"])
            assert len(lines) <= 2, cue
            for line in lines:
                assert len(line) <= MAX_LINE_CHARS, (len(line), line)

    def test_cue_duration_ceiling(self, sample_cues):
        for cue in sample_cues:
            assert cue["end"] - cue["start"] <= 6.0 + 1e-6, cue

    def test_wording_and_order_preserved(self, sample_cues, sample_words):
        spoken = " ".join(w["w"].strip() for w in sample_words)
        assert " ".join(c["text"] for c in sample_cues) == spoken

    def test_empty_input(self):
        assert words_to_cues([]) == []

    def test_single_word_never_recurses_forever(self):
        # A lone word longer than the cue budget must not split into empty halves.
        cues = words_to_cues([{"s": 0.0, "e": 0.4, "w": "x" * 200}])
        assert len(cues) == 1
        assert cues[0]["end"] - cues[0]["start"] >= MIN_CUE_DUR

    def test_min_gap_is_respected_between_cues(self, sample_cues):
        # Lead-out extends into dead air but leaves the default 0.08s breath.
        for previous, following in zip(sample_cues, sample_cues[1:]):
            gap = following["start"] - previous["end"]
            assert gap >= 0.08 - 1e-6, (previous, following, gap)


@pytest.mark.unit
class TestWrapTwoLines:
    """wrap_two_lines() picks a readable break, not an arbitrary one."""

    def test_short_text_stays_on_one_line(self):
        assert wrap_two_lines("I served for five years.") == [
            "I served for five years."
        ]

    def test_break_after_sentence_end_is_preferred(self):
        text = "It could complicate things. Do you worry about that?"
        assert wrap_two_lines(text) == [
            "It could complicate things.",
            "Do you worry about that?",
        ]

    def test_break_after_comma_beats_a_more_balanced_split(self):
        text = "aaaaa aaaaa aaaaa, aaaaa aaaaa aaaaa aaaaa aaaaa"
        assert len(text) > MAX_LINE_CHARS
        top, bottom = wrap_two_lines(text)
        assert top.endswith(",")
        assert len(bottom) <= MAX_LINE_CHARS

    def test_top_line_is_the_shorter_one_on_a_tie(self):
        text = " ".join(["aaaaa"] * 9)  # no punctuation anywhere to bias the score
        top, bottom = wrap_two_lines(text)
        assert len(top) < len(bottom)
        assert (len(top), len(bottom)) == (23, 29)

    def test_both_lines_fit_the_limit(self):
        text = "And you're about to land in a country that recognizes the ICC."
        lines = wrap_two_lines(text)
        assert len(lines) == 2
        assert all(len(line) <= MAX_LINE_CHARS for line in lines)

    def test_unbreakable_word_falls_back_to_a_hard_split(self):
        lines = wrap_two_lines("y" * 60)
        assert len(lines) == 2
        assert len(lines[0]) == MAX_LINE_CHARS

    def test_custom_max_line(self):
        lines = wrap_two_lines("one two three four", max_line=10)
        assert all(len(line) <= 10 for line in lines)


@pytest.mark.unit
class TestGershayim:
    """gershayim() fixes Hebrew acronym marks only."""

    def test_hebrew_acronym(self):
        assert gershayim('צה"ל') == "צה״ל"

    def test_inside_a_sentence(self):
        assert gershayim('הוא שירת בצה"ל חמש שנים') == "הוא שירת בצה״ל חמש שנים"

    def test_english_quotes_untouched(self):
        text = 'he said "hello" to me'
        assert gershayim(text) == text

    def test_quote_around_hebrew_phrase_untouched(self):
        # Not between two letters -> a real quotation mark, left alone.
        text = 'הוא אמר "שלום" לכולם'
        assert gershayim(text) == text

    def test_empty(self):
        assert gershayim("") == ""


@pytest.mark.unit
class TestBidiIsolate:
    """bidi_isolate() applies the render-validated RLI/LRI/PDI treatment."""

    def test_exact_isolate_placement(self):
        line = "ב-ICC זה יכול לסבך דברים ב-2026"
        expected = (
            RLI
            + "ב-"
            + LRI
            + "ICC"
            + PDI
            + " זה יכול לסבך דברים ב-"
            + LRI
            + "2026"
            + PDI
            + PDI
        )
        assert bidi_isolate(line) == expected

    def test_pure_hebrew_gets_only_the_outer_isolate(self):
        line = "אני שירתתי חמש שנים"
        assert bidi_isolate(line) == RLI + line + PDI

    def test_never_uses_rtl_override(self):
        assert RLO not in bidi_isolate("ב-ICC זה יכול לסבך דברים")

    def test_visible_text_is_unchanged(self):
        line = "צה״ל is tough"
        out = bidi_isolate(line)
        assert "".join(c for c in out if c not in (RLI, LRI, PDI)) == line

    def test_empty(self):
        assert bidi_isolate("") == ""


@pytest.mark.unit
class TestBuildAss:
    """build_ass() emits a valid, render-ready ASS v4+ script."""

    CUES = [
        # 45 chars -> must wrap onto two lines, breaking after the full stop.
        {
            "start": 14.02,
            "end": 16.95,
            "text": "זה עלול לסבך את העניינים בהמשך. אתה דואג מזה?",
        },
        {"start": 17.03, "end": 18.29, "text": "כן, אני חושב על זה."},
    ]

    @pytest.fixture()
    def ass(self):
        return build_ass(self.CUES, video_w=1280, video_h=720)

    def test_script_info_matches_video(self, ass):
        assert "[Script Info]" in ass
        assert "ScriptType: v4.00+" in ass
        assert "PlayResX: 1280" in ass
        assert "PlayResY: 720" in ass
        # WrapStyle 2 = never auto-wrap; we control the line breaks ourselves.
        assert "WrapStyle: 2" in ass
        assert "ScaledBorderAndShadow: yes" in ass

    def test_style_line(self, ass):
        style = next(a for a in ass.splitlines() if a.startswith("Style:"))
        fields = style.split(":", 1)[1].strip().split(",")
        # Never "Default": libass injects its own Default style and resolves
        # names backwards, so ours could lose to its Arial fallback.
        assert fields[0] == STYLE_NAME != "Default"
        assert fields[1] == "Noto Sans Hebrew"
        assert fields[2] == "44"  # round(720 * 0.061)
        assert fields[3] == "&H00FFFFFF"  # PrimaryColour
        assert fields[6] == "&H14000000"  # BackColour (the box)
        assert fields[7] == "1"  # Bold
        assert fields[15] == "4"  # BorderStyle: opaque box
        assert fields[16] == "3"  # Outline
        assert fields[17] == "0"  # Shadow
        assert fields[18] == "2"  # Alignment: bottom centre
        assert fields[21] == "86"  # MarginV: round(720 * 0.12)

    def test_font_and_margin_scale_with_height(self):
        style = next(
            line
            for line in build_ass(self.CUES, video_w=3840, video_h=2160).splitlines()
            if line.startswith("Style:")
        )
        fields = style.split(":", 1)[1].strip().split(",")
        assert fields[2] == "132"  # round(2160 * 0.061)
        assert fields[21] == "259"  # round(2160 * 0.12)

    def test_events(self, ass):
        events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        assert len(events) == 2
        assert events[0].startswith(
            f"Dialogue: 0,0:00:14.02,0:00:16.95,{STYLE_NAME},,0,0,0,,"
        )

    def test_two_line_cue_uses_ass_newline(self, ass):
        first = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
        assert "\\N" in first
        assert "\n" not in first

    def test_rtl_isolates_each_line(self, ass):
        first = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
        body = first.split(",,", 1)[1]
        assert body.count(RLI) == 2  # one per line
        assert RLO not in ass

    def test_rtl_can_be_disabled(self):
        out = build_ass(self.CUES, video_w=1280, video_h=720, rtl=False)
        assert RLI not in out
        assert LRI not in out
        assert PDI not in out

    def test_gershayim_applied_to_cue_text(self):
        out = build_ass(
            [{"start": 0.0, "end": 2.0, "text": 'שירתתי בצה"ל'}],
            video_w=1280,
            video_h=720,
        )
        assert "צה״ל" in out
        assert 'צה"ל' not in out

    def test_braces_cannot_inject_ass_markup(self):
        out = build_ass(
            [{"start": 0.0, "end": 2.0, "text": "{\\fs80}שלום"}],
            video_w=1280,
            video_h=720,
        )
        assert "{" not in out and "}" not in out

    def test_no_cues(self):
        out = build_ass([], video_w=1280, video_h=720)
        assert "[Events]" in out
        assert "Dialogue:" not in out


@pytest.mark.unit
def test_end_to_end_words_to_ass(sample_words):
    """The whole engine: words -> cues -> ASS, no defects in the output."""
    cues = words_to_cues(sample_words)
    ass = build_ass(cues, video_w=1280, video_h=720)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == len(cues)
    assert RLO not in ass
    for event in events:
        body = event.split(",,", 1)[1]
        assert len(body.split("\\N")) <= 2


@pytest.mark.unit
class TestSparsePunctuationFallback:
    """Unpunctuated ASR output (large-v3 artifact) must split on speech pauses."""

    def _words(self):
        # 16 unpunctuated words, clear 0.6s pauses after words 5 and 10
        words, t = [], 0.0
        for i in range(16):
            words.append({"s": t, "e": t + 0.25, "w": f" word{i}"})
            t += 0.3
            if i in (4, 9):
                t += 0.6  # speech pause
        return words

    def test_pause_split_engages_without_punctuation(self):
        from services.subtitle_engine import words_to_cues
        cues = words_to_cues(self._words())
        assert len(cues) >= 3, "pauses should become sentence boundaries"
        assert "word4" in cues[0]["text"] and "word5" not in cues[0]["text"]

    def test_punctuated_input_unaffected(self):
        from services.subtitle_engine import words_to_cues
        # same words but properly punctuated: fallback must NOT engage
        words = self._words()
        for i in (4, 9, 15):
            words[i]["w"] = words[i]["w"] + "."
        cues = words_to_cues(words)
        texts = [c["text"] for c in cues]
        assert any(t.rstrip().endswith("word4.") for t in texts)
