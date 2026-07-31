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
    reflow_dangling_connectors,
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

    @pytest.mark.parametrize(
        "text",
        [
            "y" * 60,
            "y" * 100,
            "https://example.com/" + "a" * 200,
            " ".join(["word"] * 60),               # 300 chars, plenty of spaces
            "supercalifragilistic " * 12,          # long tokens AND spaces
            "א" * 90,                              # Hebrew, no spaces
            "מילה " * 40,                          # Hebrew with spaces
            "x" * 43,                              # one over the limit
            "ab " + "c" * 200,
        ],
    )
    def test_no_line_ever_exceeds_the_limit(self, text):
        """The old last-resort split returned `text[max_line:]` verbatim, so anything
        over 2 x max_line produced a line that runs straight off the frame."""
        for line in wrap_two_lines(text):
            assert len(line) <= MAX_LINE_CHARS, f"overflowing line: {len(line)} chars"

    def test_hard_wrap_loses_no_text(self):
        text = "supercalifragilistic " * 12
        joined = "".join(wrap_two_lines(text))
        assert joined.replace(" ", "") == text.replace(" ", "").strip()

    def test_hard_wrap_prefers_whitespace(self):
        lines = wrap_two_lines(" ".join(["word"] * 60))
        assert all(not line.startswith(" ") and not line.endswith(" ") for line in lines)
        assert all("word" in line for line in lines)
        # No word was chopped in half when a space was available.
        assert all(
            all(token == "word" for token in line.split()) for line in lines
        )


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
    """bidi_isolate() wraps the line RTL and each MAXIMAL Latin/digit run LTR.

    The rendered proof of these placements lives in
    ``tests/integration/test_bidi_render.py`` — it puts every one of these strings
    through the real ``ass`` filter and compares pixels against an independently
    written visual layout. These unit tests only pin the structure that produces it.

    What changed and why: the previous implementation isolated each Latin word and
    each digit group *separately*. Sibling isolates are laid out right-to-left
    relative to each other inside an RTL line, so "Microsoft Azure" rendered as
    "Azure Microsoft", "3.5" as "5.3" and "COVID-19" as "19-COVID". Only maximal
    runs are safe.
    """

    def test_multi_word_latin_run_gets_ONE_isolate(self):
        # The regression that made this rewrite necessary: two isolates here render
        # as "Azure Microsoft".
        out = bidi_isolate("היום Microsoft Azure עלה")
        assert out == RLI + "היום " + LRI + "Microsoft Azure" + PDI + " עלה" + PDI
        assert out.count(LRI) == 1

    def test_decimal_is_one_run_not_two_digit_groups(self):
        out = bidi_isolate("זה עלה 3.5 אחוז")
        assert out == RLI + "זה עלה " + LRI + "3.5" + PDI + " אחוז" + PDI

    def test_hyphenated_alnum_is_one_run(self):
        out = bidi_isolate("התפרצות COVID-19 קשה")
        assert out == RLI + "התפרצות " + LRI + "COVID-19" + PDI + " קשה" + PDI

    def test_hebrew_prefixed_acronym_and_year(self):
        out = bidi_isolate("ב-ICC זה יכול לסבך דברים ב-2026")
        assert out == (
            RLI
            + "ב-" + LRI + "ICC" + PDI
            + " זה יכול לסבך דברים ב-" + LRI + "2026" + PDI
            + PDI
        )

    def test_gershayim_and_prefixed_acronym_mid_sentence(self):
        out = bidi_isolate("צה״ל אמר ש-CNN דיווח")
        assert out == RLI + "צה״ל אמר ש-" + LRI + "CNN" + PDI + " דיווח" + PDI

    @pytest.mark.parametrize(
        "line, run",
        [
            ("זה עלה 50% השנה", "50%"),          # ET after a number
            ("המחיר הוא $25 בלבד", "$25"),        # ET before a number
            ("דו״ח AT&T פורסם", "AT&T"),          # ON inside a Latin run
            ("הם קנו Microsoft Azure 2024 אתמול", "Microsoft Azure 2024"),
            ("הוא אמר Boeing 737 היום", "Boeing 737"),
        ],
    )
    def test_run_boundaries_follow_unicode_not_a_hand_written_class(self, line, run):
        """A literal character class forgot %, $ and & and rendered "50%" as "%50"."""
        assert LRI + run + PDI in bidi_isolate(line)

    def test_two_latin_runs_separated_by_hebrew_stay_separate(self):
        out = bidi_isolate("ICC אמר ש-CNN טועה")
        assert out.count(LRI) == 2

    def test_pure_hebrew_gets_only_the_outer_isolate(self):
        line = "אני שירתתי חמש שנים"
        assert bidi_isolate(line) == RLI + line + PDI

    def test_line_without_any_rtl_character_is_left_alone(self):
        """libass's native base direction is already LTR, and wrapping an English
        line in RLI kicks its sentence-final period over to the left edge."""
        line = "Hello world."
        assert bidi_isolate(line) == line

    def test_never_uses_rtl_override(self):
        assert RLO not in bidi_isolate("ב-ICC זה יכול לסבך דברים")

    def test_visible_text_is_unchanged(self):
        line = "צה״ל is tough"
        out = bidi_isolate(line)
        assert "".join(c for c in out if c not in (RLI, LRI, PDI)) == line

    def test_isolates_are_balanced(self):
        for line in [
            "ב-ICC זה יכול לסבך דברים ב-2026",
            "היום Microsoft Azure עלה",
            "זה עלה 50% השנה",
            "שלום עולם",
        ]:
            out = bidi_isolate(line)
            assert out.count(RLI) + out.count(LRI) == out.count(PDI)

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
        """COUNT-based: if the pause fallback engaged, the extra pauses would add cues.

        The previous version of this test asserted that some cue ends with "word4.",
        which is true whether or not the fallback runs — it could not fail. This one
        compares the cue COUNT against the same words with the pauses removed: the
        punctuation-only splitter cannot see pauses, so the two must agree.
        """
        from services.subtitle_engine import words_to_cues

        punctuated = self._words()
        for i in (4, 9, 15):
            punctuated[i]["w"] = punctuated[i]["w"] + "."

        # Same words, same punctuation, but evenly spaced (no pause to split on).
        flat, t = [], 0.0
        for word in punctuated:
            flat.append({"s": t, "e": t + 0.25, "w": word["w"]})
            t += 0.3

        assert len(words_to_cues(punctuated)) == len(words_to_cues(flat)), (
            "cue count changed with the pauses -> the pause fallback engaged on "
            "punctuated input"
        )

    def test_a_single_terminal_disables_the_fallback(self):
        """The trigger is `terminals == 0`, not a ratio: one full stop is enough."""
        from services.subtitle_engine import words_to_cues

        words = self._words()
        words[-1]["w"] = words[-1]["w"] + "."  # 1 terminal in 16 words

        flat, t = [], 0.0
        for word in words:
            flat.append({"s": t, "e": t + 0.25, "w": word["w"]})
            t += 0.3

        assert len(words_to_cues(words)) == len(words_to_cues(flat))

    def test_fallback_needs_at_least_eight_words(self):
        from services.subtitle_engine import words_to_cues

        words, t = [], 0.0
        for i in range(4):
            words.append({"s": t, "e": t + 0.25, "w": f"word{i}"})
            t += 1.5  # huge pauses; too few words for the fallback to engage
        assert len(words_to_cues(words)) == 1


@pytest.mark.unit
class TestMergeGap:
    """Short fragments are merged into the previous cue, but not across dead air."""

    def test_does_not_merge_across_a_long_silence(self):
        """The reviewer's repro: two one-word utterances 5 seconds apart."""
        words = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 5.5, "e": 6.0, "w": "Bye."},
        ]
        cues = words_to_cues(words)
        assert len(cues) == 2, f"merged across a 5s silence: {cues}"
        assert cues[0]["text"] == "Hello."
        assert cues[1]["text"] == "Bye."

    def test_still_merges_across_a_short_silence(self):
        words = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 0.9, "e": 1.3, "w": "Bye."},
        ]
        cues = words_to_cues(words)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello. Bye."

    def test_gap_is_measured_from_the_previous_cue_end(self):
        from services.subtitle_engine import MERGE_MAX_GAP

        just_inside = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 0.5 + MERGE_MAX_GAP - 0.01, "e": 0.5 + MERGE_MAX_GAP + 0.3, "w": "Bye."},
        ]
        just_outside = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 0.5 + MERGE_MAX_GAP + 0.01, "e": 0.5 + MERGE_MAX_GAP + 0.3, "w": "Bye."},
        ]
        assert len(words_to_cues(just_inside)) == 1
        assert len(words_to_cues(just_outside)) == 2


@pytest.mark.unit
class TestDegenerateTimings:
    """No cue may ever have a zero or negative duration, whatever the input."""

    @staticmethod
    def _assert_sane(cues):
        for cue in cues:
            assert cue["end"] > cue["start"], f"non-positive duration: {cue}"
            assert cue["start"] >= 0, f"negative start survived: {cue}"
        for earlier, later in zip(cues, cues[1:]):
            assert earlier["end"] <= later["start"], "cues overlap"
            assert earlier["start"] <= later["start"], "cues out of order"

    def test_all_timestamps_identical(self):
        words = [{"s": 5.0, "e": 5.0, "w": f"w{i}."} for i in range(6)]
        cues = words_to_cues(words)
        assert cues
        self._assert_sane(cues)

    def test_negative_timestamps_are_clamped_before_processing(self):
        words = [
            {"s": -3.0, "e": -2.5, "w": "First."},
            {"s": -1.0, "e": 0.5, "w": "Second."},
            {"s": 2.0, "e": 3.0, "w": "Third."},
        ]
        cues = words_to_cues(words)
        self._assert_sane(cues)

    def test_end_before_start(self):
        words = [
            {"s": 4.0, "e": 1.0, "w": "Backwards."},
            {"s": 6.0, "e": 5.0, "w": "Again."},
        ]
        self._assert_sane(words_to_cues(words))

    def test_no_text_is_lost_when_cues_are_folded(self):
        words = [{"s": 2.0, "e": 2.0, "w": w} for w in ("alpha.", "beta.", "gamma.")]
        cues = words_to_cues(words)
        joined = " ".join(c["text"] for c in cues)
        for word in ("alpha", "beta", "gamma"):
            assert word in joined
        self._assert_sane(cues)

    @pytest.mark.parametrize(
        "words",
        [
            [],
            [{"s": 0.0, "e": 0.0, "w": "only."}],
            [{"s": 0.0, "e": 0.0, "w": "a"}, {"s": 0.0, "e": 0.0, "w": "b"}],
            [{"s": 1.0, "e": 1.0, "w": "x."}, {"s": 1.0001, "e": 1.0001, "w": "y."}],
            [{"s": "bad", "e": "worse", "w": "junk"}],
            [{"s": float("nan"), "e": 1.0, "w": "nan"}],
        ],
    )
    def test_property_over_degenerate_inputs(self, words):
        self._assert_sane(words_to_cues(words))


@pytest.mark.unit
class TestReflowDanglingConnectors:
    """A stranded single-letter Hebrew prefix belongs to the next cue's first word."""

    @staticmethod
    def _cues(*pairs):
        return [
            {"start": float(i) * 2, "end": float(i) * 2 + 1.5, "translated_text": text}
            for i, text in enumerate(pairs)
        ]

    def test_moves_a_dangling_vav(self):
        """Real output from this pipeline: a cue ending on a bare "ו"."""
        cues = self._cues("המטוס עמד לנחות ו", "הנוסעים התכוננו")
        out = reflow_dangling_connectors(cues)
        assert out[0]["translated_text"] == "המטוס עמד לנחות"
        assert out[1]["translated_text"] == "והנוסעים התכוננו"

    def test_moves_a_dangling_bet(self):
        cues = self._cues("החברה שמכירה ב", "מחירים נמוכים")
        out = reflow_dangling_connectors(cues)
        assert out[0]["translated_text"] == "החברה שמכירה"
        assert out[1]["translated_text"] == "במחירים נמוכים"

    def test_keeps_an_existing_maqaf(self):
        cues = self._cues("הוא עבד ש-", "CNN דיווח")
        out = reflow_dangling_connectors(cues)
        assert out[1]["translated_text"] == "ש-CNN דיווח"

    def test_adds_a_maqaf_before_a_latin_word(self):
        cues = self._cues("הוא עבד ב", "Google שנתיים")
        out = reflow_dangling_connectors(cues)
        assert out[1]["translated_text"] == "ב-Google שנתיים"

    def test_no_op_without_a_dangler(self):
        cues = self._cues("משפט שלם.", "משפט נוסף.")
        assert reflow_dangling_connectors(cues) == cues

    def test_letter_that_is_part_of_a_word_is_not_moved(self):
        cues = self._cues("זה מה שקרה", "אתמול בערב")
        assert reflow_dangling_connectors(cues) == cues

    def test_last_cue_dangler_stays(self):
        cues = self._cues("משפט ראשון.", "והוא הלך ו")
        out = reflow_dangling_connectors(cues)
        assert out[-1]["translated_text"] == "והוא הלך ו"

    def test_not_moved_across_a_long_gap(self):
        cues = [
            {"start": 0.0, "end": 1.0, "translated_text": "המטוס נחת ו"},
            {"start": 5.0, "end": 6.0, "translated_text": "הנוסעים ירדו"},
        ]
        assert reflow_dangling_connectors(cues) == cues

    def test_not_moved_when_the_next_cue_would_overflow(self):
        cues = [
            {"start": 0.0, "end": 1.0, "translated_text": "המטוס נחת ו"},
            {"start": 1.2, "end": 3.0, "translated_text": "א" * 84},
        ]
        assert reflow_dangling_connectors(cues) == cues

    def test_a_cue_that_is_only_the_connector_is_left_alone(self):
        cues = self._cues("ו", "הנוסעים ירדו")
        assert reflow_dangling_connectors(cues) == cues

    def test_timings_are_never_touched(self):
        cues = self._cues("המטוס נחת ו", "הנוסעים ירדו")
        out = reflow_dangling_connectors(cues)
        assert [(c["start"], c["end"]) for c in out] == [
            (c["start"], c["end"]) for c in cues
        ]

    def test_input_is_not_mutated(self):
        cues = self._cues("המטוס נחת ו", "הנוסעים ירדו")
        before = [dict(c) for c in cues]
        reflow_dangling_connectors(cues)
        assert cues == before

    def test_only_one_token_moves_per_cue(self):
        cues = self._cues("הוא הלך ו ב", "הבית")
        out = reflow_dangling_connectors(cues)
        assert out[0]["translated_text"] == "הוא הלך ו"
        assert out[1]["translated_text"] == "בהבית"

    def test_empty_input(self):
        assert reflow_dangling_connectors([]) == []
