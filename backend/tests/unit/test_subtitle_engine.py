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
        # One documented exception: a complete sentence sitting right before a
        # question may ship slightly under the floor, because merging it into the
        # question is the judged spoiler defect (the viewer reads the next question
        # while this one is being answered) and the engine refuses. Slightly under —
        # never a flash.
        for cue, following in zip(sample_cues, sample_cues[1:] + [None]):
            duration = cue["end"] - cue["start"]
            if duration >= MIN_CUE_DUR:
                continue
            assert duration >= 1.0, f"flash cue: {cue}"
            assert following is not None and following["text"].rstrip().endswith(
                "?"
            ), f"below-floor cue not explained by the question guard: {cue}"

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

    def test_complete_sentence_is_never_glued_to_the_next_question(self, sample_cues):
        """The reversal of what this test used to pin, on judged evidence.

        The old behaviour merged the 0.94s "It could complicate things." INTO
        "Do you worry about that?" — a complete sentence glued to the question that
        follows it. In a rapid interview that exact merge put the interviewer's NEXT
        question on screen 2.5 seconds before it was asked. The engine now keeps the
        statement alone (a hair under the duration floor, logged as a compromise)
        and the question arrives when it is asked.
        """
        texts = [c["text"] for c in sample_cues]
        assert "It could complicate things." in texts
        holder = next(t for t in texts if "Do you worry about that?" in t)
        assert "complicate things" not in holder

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
        # Exception mirrors test_no_flash_cues: a below-floor sentence that the
        # question guard refused to merge takes every millisecond up to the next
        # cue's start — duration was judged worth more than the breath there.
        for previous, following in zip(sample_cues, sample_cues[1:]):
            gap = following["start"] - previous["end"]
            if previous["end"] - previous["start"] < MIN_CUE_DUR:
                assert gap >= -1e-6, (previous, following, gap)
                continue
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
            " ".join(["word"] * 60),  # 300 chars, plenty of spaces
            "supercalifragilistic " * 12,  # long tokens AND spaces
            "א" * 90,  # Hebrew, no spaces
            "מילה " * 40,  # Hebrew with spaces
            "x" * 43,  # one over the limit
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
        assert all(
            not line.startswith(" ") and not line.endswith(" ") for line in lines
        )
        assert all("word" in line for line in lines)
        # No word was chopped in half when a space was available.
        assert all(all(token == "word" for token in line.split()) for line in lines)


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

    def test_gershayim_and_prefixed_acronym_mid_sentence(self):
        out = bidi_isolate("צה״ל אמר ש-CNN דיווח")
        assert out == RLI + "צה״ל אמר ש-" + LRI + "CNN" + PDI + " דיווח" + PDI

    @pytest.mark.parametrize(
        "line, run",
        [
            ("זה עלה 50% השנה", "50%"),  # ET after a number
            ("המחיר הוא $25 בלבד", "$25"),  # ET before a number
            ("דו״ח AT&T פורסם", "AT&T"),  # ON inside a Latin run
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

    @pytest.mark.parametrize(
        "position,alignment", [("bottom", "2"), ("top", "8"), ("side", "6")]
    )
    def test_position_maps_to_ass_alignment(self, position, alignment):
        ass = build_ass(self.CUES, video_w=1280, video_h=720, position=position)
        style = next(line for line in ass.splitlines() if line.startswith("Style:"))
        fields = style.split(":", 1)[1].strip().split(",")
        assert fields[18] == alignment

    def test_invalid_position_preserves_bottom_default(self):
        ass = build_ass(self.CUES, video_w=1280, video_h=720, position="somewhere")
        style = next(line for line in ass.splitlines() if line.startswith("Style:"))
        assert style.split(":", 1)[1].strip().split(",")[18] == "2"

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
        """Four words with 0.5s pauses: enough pause for the FALLBACK, too few words.

        The pauses are deliberately between :data:`PAUSE_SPLIT_GAP` (0.35) and
        :data:`TURN_SPLIT_GAP` (0.7) — long enough that the sparse-punctuation fallback
        would split on every one of them if it engaged, short enough that the speaker-turn
        splitter (which engages on every transcript, punctuated or not) will not. So one
        cue means exactly one thing: the fallback stayed out.
        """
        from services.subtitle_engine import (
            PAUSE_SPLIT_GAP,
            TURN_SPLIT_GAP,
            words_to_cues,
        )

        gap = 0.5
        assert PAUSE_SPLIT_GAP < gap < TURN_SPLIT_GAP
        words, t = [], 0.0
        for i in range(4):
            words.append({"s": t, "e": t + 0.25, "w": f"word{i}"})
            t += 0.25 + gap
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
        """The effective ceiling is min(MERGE_MAX_GAP, TURN_SPLIT_GAP).

        It used to be MERGE_MAX_GAP alone, and that was a contradiction: a 0.72s silence
        was simultaneously long enough to be called a speaker TURN and short enough to
        merge across, so it merged — gluing "Let's give them a new task." to another
        man's "Yeah, of course." in a real job. A silence cannot be both.
        """
        from services.subtitle_engine import MERGE_MAX_GAP, TURN_SPLIT_GAP

        effective = min(MERGE_MAX_GAP, TURN_SPLIT_GAP)
        just_inside = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 0.5 + effective - 0.01, "e": 0.5 + effective + 0.3, "w": "Bye."},
        ]
        just_outside = [
            {"s": 0.0, "e": 0.5, "w": "Hello."},
            {"s": 0.5 + effective + 0.01, "e": 0.5 + effective + 0.3, "w": "Bye."},
        ]
        assert len(words_to_cues(just_inside)) == 1
        assert len(words_to_cues(just_outside)) == 2

    def test_a_turn_length_silence_is_never_merged_across(self):
        """D's repro, reduced: the 0.72s pause inside 'בייב' cue 9."""
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 25.42, "e": 25.76, "w": "Let's"},
            {"s": 25.76, "e": 26.12, "w": "give"},
            {"s": 26.12, "e": 26.52, "w": "task."},
            {"s": 27.24, "e": 27.60, "w": "Yeah,"},  # 0.72s later: a different speaker
            {"s": 27.72, "e": 27.96, "w": "course."},
        ]
        cues = words_to_cues(words)
        assert len(cues) == 2, f"two speakers glued into one cue: {cues}"
        assert cues[0]["text"] == "Let's give task."
        assert cues[1]["text"] == "Yeah, course."


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


# =============================================================================
# P5 — geresh (U+05F3) in foreign names and loanwords
# =============================================================================
@pytest.mark.unit
class TestGeresh:
    """The v2 path shipped ג'ורג' with an ASCII apostrophe; the legacy path did not.

    The regression is narrow but it is a typography error on every transliterated name,
    which on news content is most proper nouns in the clip.
    """

    def test_between_two_hebrew_letters(self):
        from services.subtitle_engine import GERESH, geresh

        assert geresh("ג'ז") == f"ג{GERESH}ז"

    def test_word_final_apostrophe_also_converts(self):
        """The whole point: ג'ורג' has a WORD-FINAL geresh a between-letters rule misses."""
        from services.subtitle_engine import GERESH, geresh

        assert geresh("ג'ורג'") == f"ג{GERESH}ורג{GERESH}"

    def test_curly_apostrophe_converts_too(self):
        """GPT-4o emits U+2019 interchangeably with ASCII '."""
        from services.subtitle_engine import GERESH, geresh

        assert geresh("צ’ארלס") == f"צ{GERESH}ארלס"

    @pytest.mark.parametrize(
        "text",
        [
            "don't",
            "it's",
            "Charles' car",
            "he said 'hello' loudly",
            "O'Brien",
        ],
    )
    def test_english_apostrophes_are_never_touched(self, text):
        from services.subtitle_engine import geresh

        assert geresh(text) == text

    def test_mixed_line_converts_only_the_hebrew_side(self):
        from services.subtitle_engine import GERESH, geresh

        out = geresh("ג'ורג' said don't")
        assert out == f"ג{GERESH}ורג{GERESH} said don't"

    @pytest.mark.parametrize("text", ["שלום'", "'דבר'", "המקור'"])
    def test_other_hebrew_letters_keep_their_quote(self, text):
        """ר/ם/ו are common word-final letters; converting after them would rewrite
        every closing single quote in Hebrew. Only ג/ז/צ/ץ are in scope."""
        from services.subtitle_engine import geresh

        assert geresh(text) == text

    def test_empty_and_none(self):
        from services.subtitle_engine import geresh

        assert geresh("") == ""
        assert geresh(None) is None

    def test_hebrew_typography_applies_both_marks(self):
        from services.subtitle_engine import GERESH, GERSHAYIM, hebrew_typography

        out = hebrew_typography("צה\"ל ו-ג'ורג'")
        assert f"צה{GERSHAYIM}ל" in out
        assert f"ג{GERESH}ורג{GERESH}" in out

    def test_build_ass_emits_geresh_not_ascii(self):
        """The renderer is the layer the user actually sees; assert at that boundary."""
        from services.subtitle_engine import GERESH, build_ass

        out = build_ass(
            [{"start": 0.0, "end": 3.0, "text": "ג'ורג' הגיע"}],
            video_w=1280,
            video_h=720,
        )
        body = "\n".join(l for l in out.splitlines() if l.startswith("Dialogue:"))
        assert GERESH in body
        assert "'" not in body


# =============================================================================
# P4 — hallucination gate
# =============================================================================
@pytest.mark.unit
class TestDropHallucinatedCues:
    """The gate is SCORED now: one signal is a suspicion, agreement is a verdict.

    Every threshold below was measured against this project's own 20-run corpus archive.
    The rule these tests replaced — "drop anything over 35 CPS" — deleted 8 real cues
    there and caught 0 fabrications, so the tests that pinned it were pinning a defect.
    """

    @staticmethod
    def _cue(text, start, end, **extra):
        cue = {"start": start, "end": end, "text": text}
        cue.update(extra)
        return cue

    @staticmethod
    def _stats(run=0, words=8):
        return {
            "words": words,
            "degenerate": run,
            "degenerate_run": run,
            "min_word_dur": 0.0 if run else 0.2,
        }

    def test_normal_speech_is_kept(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cues = [self._cue("This is a perfectly ordinary subtitle cue.", 0.0, 3.0)]
        kept, dropped = drop_hallucinated_cues(cues)
        assert kept == cues and dropped == []

    def test_impossible_density_is_dropped_on_its_own(self):
        from services.subtitle_engine import drop_hallucinated_cues

        # 57 characters in 0.26s = 219 CPS — a real fabricated Trump line.
        text = "I was killed by a police officer in Butler, Pennsylvania."
        kept, dropped = drop_hallucinated_cues([self._cue(text, 59.06, 59.32)])
        assert kept == []
        assert len(dropped) == 1
        assert dropped[0]["cps"] > 200
        assert any(s.startswith("cps_impossible") for s in dropped[0]["signals"])

    def test_fast_speech_alone_is_no_longer_enough(self):
        """The measured regression: real standup delivery at 39 CPS was being deleted."""
        from services.subtitle_engine import drop_hallucinated_cues

        real = self._cue(
            "and they were screaming and yelling at each other.", 8.78, 10.06
        )
        kept, dropped = drop_hallucinated_cues([real])
        assert (10.06 - 8.78) and len(real["text"]) / (10.06 - 8.78) > 35
        assert kept == [real] and dropped == []

    def test_the_words_per_second_bound_sits_between_the_two_populations(self):
        """LOWERED 8.0 -> 7.25, and the exact value is the evidence, not a preference.

        Invented runs were measured at 7.5-7.7 w/s and walked straight under the old
        8.0. The corpus's fastest GENUINE cue is the cross-talk line above at 7.03 w/s,
        which a hard signal must never touch — it deletes on its own, with no
        audio-energy veto to rescue it. The constant therefore has to live strictly
        inside (7.03, 7.5], and this test is what stops a later round rounding it to a
        nicer number in either direction.
        """
        from services.subtitle_engine import IMPOSSIBLE_WORDS_PER_SEC

        fastest_genuine = 9 / (10.06 - 8.78)  # 7.03 w/s, real cross-talk
        slowest_fabrication = 7.5
        assert fastest_genuine < IMPOSSIBLE_WORDS_PER_SEC <= slowest_fabrication

    def test_an_invented_run_at_seven_and_a_half_words_a_second_is_dropped(self):
        """The measured fabrication rate that used to slide under the old 8.0."""
        from services.subtitle_engine import drop_hallucinated_cues

        text = " ".join(f"word{i}" for i in range(15))  # 15 words in 2s = 7.5 w/s
        kept, dropped = drop_hallucinated_cues([self._cue(text, 0.0, 2.0)])

        assert kept == []
        assert any(s.startswith("wps_impossible") for s in dropped[0]["signals"])

    def test_two_soft_signals_together_do_drop(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue("x" * 40, 0.0, 1.0, word_stats=self._stats(run=5))
        kept, dropped = drop_hallucinated_cues([cue])
        assert kept == []
        assert dropped[0]["signals"] == [
            "cps_fast(40.0>35)",
            "degenerate_words(run=5)",
        ]

    def test_a_degenerate_run_alone_is_not_enough(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue(
            "Do you have a boyfriend?", 59.14, 60.0, word_stats=self._stats(run=6)
        )
        kept, _dropped = drop_hallucinated_cues([cue])
        assert kept == [cue]

    def test_a_short_degenerate_run_is_not_a_signal(self):
        from services.subtitle_engine import DEGENERATE_WORD_RUN, drop_hallucinated_cues

        cue = self._cue(
            "x" * 40, 0.0, 1.0, word_stats=self._stats(run=DEGENERATE_WORD_RUN - 1)
        )
        kept, dropped = drop_hallucinated_cues([cue])
        assert kept == [cue] and dropped == []

    def test_an_exact_repeat_of_the_previous_cue_is_dropped(self):
        from services.subtitle_engine import drop_hallucinated_cues

        first = self._cue("That was some day we had together, though.", 105.62, 107.96)
        loop = self._cue("that was some day we had together though", 108.12, 109.94)
        kept, dropped = drop_hallucinated_cues([first, loop])
        assert kept == [first]
        assert dropped[0]["signals"] == ["duplicate_of_previous"]

    def test_a_duplicate_is_not_handed_back_as_context(self):
        """Its text is by definition still in the cue that was kept."""
        from services.subtitle_engine import drop_hallucinated_cues

        cues = [
            self._cue("Thank you very much.", 0.0, 1.5),
            self._cue("Thank you very much.", 1.6, 3.0),
        ]
        _kept, dropped = drop_hallucinated_cues(cues)
        assert dropped[0]["context_only"] is False

    def test_a_dropped_cue_is_offered_to_the_translator_as_context(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue("x" * 40, 0.0, 1.0, word_stats=self._stats(run=5))
        _kept, dropped = drop_hallucinated_cues([cue])
        assert dropped[0]["context_only"] is True

    def test_two_consecutive_copies_of_a_hallucination_both_go(self):
        from services.subtitle_engine import drop_hallucinated_cues

        bad = self._cue("y" * 200, 2.0, 2.5)
        kept, dropped = drop_hallucinated_cues([bad, dict(bad, start=2.6, end=3.1)])
        assert kept == [] and len(dropped) == 2

    def test_loud_audio_vetoes_a_soft_signal_drop(self):
        """Cross-talk: Whisper compressed the timing, but someone was plainly talking."""
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue("x" * 40, 0.0, 1.0, word_stats=self._stats(run=5))
        kept, dropped = drop_hallucinated_cues([cue], energy_probe=lambda s, e: -0.1)
        assert kept == [cue] and dropped == []

    def test_quiet_audio_does_not_veto(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue("x" * 40, 0.0, 1.0, word_stats=self._stats(run=5))
        kept, dropped = drop_hallucinated_cues([cue], energy_probe=lambda s, e: -14.9)
        assert kept == []
        assert dropped[0]["energy_db"] == -14.9

    def test_the_veto_cannot_save_a_hard_signal(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cue = self._cue("z" * 200, 0.0, 1.0)  # 200 CPS
        kept, _dropped = drop_hallucinated_cues([cue], energy_probe=lambda s, e: 3.0)
        assert kept == []

    def test_the_probe_is_only_consulted_for_candidates(self):
        from services.subtitle_engine import drop_hallucinated_cues

        calls = []

        def probe(start, end):
            calls.append((start, end))
            return 0.0

        drop_hallucinated_cues(
            [
                self._cue("An ordinary line.", 0.0, 3.0),
                self._cue("x" * 40, 5.0, 6.0, word_stats=self._stats(run=5)),
            ],
            energy_probe=probe,
        )
        assert calls == [(5.0, 6.0)]

    def test_a_probe_that_raises_never_fails_the_gate(self):
        from services.subtitle_engine import drop_hallucinated_cues

        def probe(start, end):
            raise RuntimeError("ffmpeg exploded")

        cue = self._cue("x" * 40, 0.0, 1.0, word_stats=self._stats(run=5))
        kept, dropped = drop_hallucinated_cues([cue], energy_probe=probe)
        assert kept == [] and len(dropped) == 1

    def test_boundary_exactly_at_threshold_is_kept(self):
        """Strictly-greater-than, so a cue measuring exactly the limit survives."""
        from services.subtitle_engine import (
            IMPOSSIBLE_SOURCE_CPS,
            drop_hallucinated_cues,
        )

        text = "x" * int(IMPOSSIBLE_SOURCE_CPS)
        kept, dropped = drop_hallucinated_cues([self._cue(text, 0.0, 1.0)])
        assert len(text) / 1.0 == IMPOSSIBLE_SOURCE_CPS
        assert len(kept) == 1 and dropped == []

    def test_zero_duration_cue_is_kept_not_dropped(self):
        """Infinite CPS by arithmetic says nothing about whether the text is real."""
        from services.subtitle_engine import drop_hallucinated_cues

        kept, dropped = drop_hallucinated_cues([self._cue("Hello.", 4.0, 4.0)])
        assert len(kept) == 1 and dropped == []

    def test_blank_and_malformed_cues_are_kept(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cues = [self._cue("", 0.0, 1.0), {"start": "x", "end": None, "text": "hi"}]
        kept, dropped = drop_hallucinated_cues(cues)
        assert len(kept) == 2 and dropped == []

    def test_order_and_identity_are_preserved(self):
        from services.subtitle_engine import drop_hallucinated_cues

        good_a = self._cue("First real line.", 0.0, 2.0)
        bad = self._cue("y" * 200, 2.0, 2.5)
        good_b = self._cue("Second real line.", 3.0, 5.0)
        kept, dropped = drop_hallucinated_cues([good_a, bad, good_b])
        assert kept == [good_a, good_b]
        assert kept[0] is good_a and kept[1] is good_b  # not copies
        assert dropped[0]["text"] == bad["text"]
        assert bad == {"start": 2.0, "end": 2.5, "text": "y" * 200}  # not mutated

    def test_custom_key_and_threshold(self):
        from services.subtitle_engine import drop_hallucinated_cues

        cues = [{"start": 0.0, "end": 1.0, "translated_text": "x" * 20}]
        kept, dropped = drop_hallucinated_cues(
            cues, key="translated_text", impossible_cps=10
        )
        assert kept == [] and len(dropped) == 1

    def test_empty_input(self):
        from services.subtitle_engine import drop_hallucinated_cues

        assert drop_hallucinated_cues([]) == ([], [])
        assert drop_hallucinated_cues(None) == ([], [])

    def test_logs_the_offending_text(self, caplog):
        from services.subtitle_engine import drop_hallucinated_cues

        with caplog.at_level("WARNING"):
            drop_hallucinated_cues([self._cue("z" * 200, 0.0, 1.0)])
        assert "hallucination gate" in caplog.text
        assert "zzz" in caplog.text


@pytest.mark.unit
class TestAbsorbDroppedTime:
    """A dropped cue leaves a hole; the previous line holds the screen over it."""

    def test_previous_cue_grows_into_the_freed_span(self):
        from services.subtitle_engine import absorb_dropped_time

        kept = [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 6.0, "end": 8.0, "text": "b"},
        ]
        dropped = [{"start": 2.5, "end": 4.0, "text": "gone"}]
        out = absorb_dropped_time(kept, dropped)
        assert out[0]["end"] == 4.0
        assert out[1] == kept[1]

    def test_it_never_overlaps_the_next_surviving_cue(self):
        from services.subtitle_engine import MIN_CUE_GAP, absorb_dropped_time

        kept = [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 3.0, "end": 5.0, "text": "b"},
        ]
        dropped = [{"start": 2.1, "end": 4.5, "text": "gone"}]
        out = absorb_dropped_time(kept, dropped)
        assert out[0]["end"] == round(3.0 - MIN_CUE_GAP, 3)

    def test_it_never_exceeds_max_cue_duration(self):
        from services.subtitle_engine import MAX_CUE_DUR, absorb_dropped_time

        kept = [{"start": 0.0, "end": 5.5, "text": "a"}]
        dropped = [{"start": 5.6, "end": 20.0, "text": "gone"}]
        out = absorb_dropped_time(kept, dropped)
        assert out[0]["end"] == MAX_CUE_DUR

    def test_a_hole_before_the_first_cue_is_left_alone(self):
        from services.subtitle_engine import absorb_dropped_time

        kept = [{"start": 5.0, "end": 7.0, "text": "a"}]
        out = absorb_dropped_time(kept, [{"start": 1.0, "end": 2.0, "text": "gone"}])
        assert out == kept

    def test_nothing_dropped_is_a_no_op(self):
        from services.subtitle_engine import absorb_dropped_time

        kept = [{"start": 0.0, "end": 2.0, "text": "a"}]
        assert absorb_dropped_time(kept, []) == kept
        assert absorb_dropped_time(kept, [])[0] is not kept[0]  # copies


# =============================================================================
# P3 — MIN_CUE_DUR is a floor, not an aspiration
# =============================================================================
@pytest.mark.unit
class TestMinDurationFloor:
    """Cues of 0.72s / 0.96s / 1.06s shipped from a real job against a 1.2s floor.

    Step 4's anti-overlap clamp could push a cue below ``min_dur`` after the lead-out
    pass had granted it. The fix merges instead of shortening.
    """

    #: Rounding to milliseconds can cost ~1ms; nothing else may.
    TOL = 2e-3

    def test_crowded_cues_are_merged_not_shortened(self):
        """Two sentences 0.8s apart: the first cannot reach 1.2s, so they become one."""
        words = [
            {"s": 0.0, "e": 0.3, "w": "Hello"},
            {"s": 0.3, "e": 0.7, "w": "there."},
            {"s": 0.8, "e": 1.1, "w": "Good"},
            {"s": 1.1, "e": 1.5, "w": "morning."},
        ]
        cues = words_to_cues(words)
        for cue in cues:
            assert cue["end"] - cue["start"] >= MIN_CUE_DUR - self.TOL, cue

    def test_no_cue_below_the_floor_on_the_real_fixture(self, sample_cues):
        # Same single documented exception as TestWordsToCuesOnRealClip
        # .test_no_flash_cues: the question guard may leave one complete sentence
        # a hair under the floor rather than spoil the next question.
        for cue, following in zip(sample_cues, sample_cues[1:] + [None]):
            duration = cue["end"] - cue["start"]
            if duration >= MIN_CUE_DUR - self.TOL:
                continue
            assert duration >= 1.0, cue
            assert following is not None and following["text"].rstrip().endswith("?")

    def test_property_no_short_cues_across_many_random_transcripts(self, caplog):
        """Property test: over 200 pseudo-random transcripts, every emitted cue meets
        the floor unless the engine WARNED that it could not merge it."""
        import random

        vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
        violations = []
        for seed in range(200):
            rng = random.Random(seed)
            words, t = [], 0.0
            for i in range(rng.randint(2, 40)):
                dur = rng.uniform(0.05, 0.6)
                word = rng.choice(vocabulary)
                if rng.random() < 0.3:
                    word += rng.choice([".", "?", "!", ","])
                words.append({"s": round(t, 3), "e": round(t + dur, 3), "w": word})
                t += dur + rng.uniform(0.0, 0.9)

            caplog.clear()
            with caplog.at_level("WARNING"):
                cues = words_to_cues(words)
            compromised = "cannot be merged" in caplog.text
            for cue in cues:
                if (
                    cue["end"] - cue["start"] < MIN_CUE_DUR - self.TOL
                    and not compromised
                ):
                    violations.append((seed, cue))
        assert (
            not violations
        ), f"{len(violations)} sub-floor cues, e.g. {violations[:3]}"

    def test_impossible_merge_keeps_the_compromise_and_warns(self, caplog):
        """When the character budget forbids the merge, the short cue ships — loudly."""
        long_a = " ".join(["word"] * 12) + "."
        long_b = " ".join(["other"] * 12) + "."
        words = []
        t = 0.0
        for token in long_a.split():
            words.append({"s": round(t, 3), "e": round(t + 0.02, 3), "w": token})
            t += 0.03
        t = 0.5
        for token in long_b.split():
            words.append({"s": round(t, 3), "e": round(t + 0.02, 3), "w": token})
            t += 0.03

        with caplog.at_level("WARNING"):
            cues = words_to_cues(words)
        short = [c for c in cues if c["end"] - c["start"] < MIN_CUE_DUR - self.TOL]
        if short:
            assert "cannot be merged" in caplog.text

    def test_single_short_utterance_still_gets_its_floor(self):
        """The last cue always has dead air ahead of it, so it is never the exception."""
        cues = words_to_cues([{"s": 0.0, "e": 0.2, "w": "Hi."}])
        assert len(cues) == 1
        assert cues[0]["end"] - cues[0]["start"] >= MIN_CUE_DUR - self.TOL

    def test_cues_never_overlap_after_the_floor_pass(self, sample_cues):
        for earlier, later in zip(sample_cues, sample_cues[1:]):
            assert earlier["end"] <= later["start"] + 1e-9

    def test_no_text_is_lost_by_merging(self):
        words = [
            {"s": 0.0, "e": 0.3, "w": "Hello"},
            {"s": 0.3, "e": 0.7, "w": "there."},
            {"s": 0.8, "e": 1.1, "w": "Good"},
            {"s": 1.1, "e": 1.5, "w": "morning."},
        ]
        cues = words_to_cues(words)
        joined = " ".join(c["text"] for c in cues).split()
        assert joined == [w["w"] for w in words]


# =============================================================================
# P1 — the pause fallback reports whether it fired DESPITE priming
# =============================================================================
@pytest.mark.unit
class TestPrimedFallbackLogging:
    """Priming should make the fallback near-dead. When it still fires, say so."""

    @staticmethod
    def _unpunctuated():
        words, t = [], 0.0
        for i in range(16):
            words.append({"s": t, "e": t + 0.25, "w": f"word{i}"})
            t += 0.3
            if i in (4, 9):
                t += 0.6
        return words

    def test_primed_fallback_warns(self, caplog):
        with caplog.at_level("INFO"):
            words_to_cues(self._unpunctuated(), asr_primed=True)
        assert "DESPITE ASR punctuation priming" in caplog.text
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_unprimed_fallback_stays_informational(self, caplog):
        with caplog.at_level("INFO"):
            words_to_cues(self._unpunctuated(), asr_primed=False)
        assert "DESPITE" not in caplog.text
        assert "not punctuation-primed" in caplog.text
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_asr_primed_never_changes_the_cues(self):
        words = self._unpunctuated()
        assert words_to_cues(words, asr_primed=True) == words_to_cues(
            words, asr_primed=False
        )


# =============================================================================
# R4 — turn structure & timing hygiene
# =============================================================================
@pytest.mark.unit
class TestTurnSplitting:
    """A silence inside a cue is a speaker turn, and the word timestamps see it."""

    def test_a_sentence_is_split_at_an_internal_turn_pause(self):
        from services.subtitle_engine import TURN_SPLIT_GAP, words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "Who"},
            {"s": 0.4, "e": 0.9, "w": "protects"},
            {"s": 0.9, "e": 1.4, "w": "women"},
            # No terminal, so this is ONE sentence group — the pause is the only signal.
            # round(): 1.4 + 0.7 is 2.0999999999999996 in binary floating point, which is
            # a hair UNDER the threshold and would silently make this test vacuous.
            {"s": round(1.4 + TURN_SPLIT_GAP, 3), "e": 2.6, "w": "From"},
            {"s": 2.6, "e": 3.2, "w": "whom"},
        ]
        cues = words_to_cues(words)
        assert len(cues) == 2
        assert cues[0]["text"] == "Who protects women"
        assert cues[1]["text"] == "From whom"

    def test_a_pause_below_the_threshold_does_not_split(self):
        from services.subtitle_engine import TURN_SPLIT_GAP, words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "Who"},
            {"s": 0.4, "e": 1.4, "w": "protects"},
            {"s": 1.4 + TURN_SPLIT_GAP - 0.05, "e": 2.6, "w": "women"},
        ]
        assert len(words_to_cues(words)) == 1

    def test_turn_splitting_can_be_switched_off(self):
        """With turn_gap=0 the two unpunctuated words are one sentence again."""
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "Who"},
            {"s": 3.0, "e": 3.4, "w": "whom"},
        ]
        assert len(words_to_cues(words, turn_gap=0)) == 1
        assert len(words_to_cues(words)) == 2

    def test_no_text_is_lost_by_a_split(self):
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "one"},
            {"s": 0.4, "e": 0.8, "w": "two"},
            {"s": 1.8, "e": 2.2, "w": "three"},
            {"s": 2.2, "e": 2.6, "w": "four"},
        ]
        joined = " ".join(c["text"].replace("— ", "") for c in words_to_cues(words))
        assert joined == "one two three four"


@pytest.mark.unit
class TestDialogueDash:
    """A turn split that separates a question from the answer marks BOTH sides."""

    @staticmethod
    def _question_then_answer():
        # 'boyfriend?' has a degenerate 0.0s duration, so the terminal split is refused
        # (see TestRefuseSplitOnDegenerateTerminal) and both sentences stay in ONE group
        # — which is exactly the state in which the turn pause has to do the work.
        return [
            {"s": 0.0, "e": 0.4, "w": "Do"},
            {"s": 0.4, "e": 0.8, "w": "you"},
            {"s": 0.8, "e": 1.2, "w": "have"},
            {"s": 1.2, "e": 1.2, "w": "someone?"},
            {"s": 2.2, "e": 2.6, "w": "I"},
            {"s": 2.6, "e": 3.0, "w": "did"},
            {"s": 3.0, "e": 3.4, "w": "once"},
        ]

    def test_both_halves_carry_the_dash(self):
        from services.subtitle_engine import DIALOGUE_DASH, words_to_cues

        cues = words_to_cues(self._question_then_answer())
        assert len(cues) == 2
        assert cues[0]["text"] == f"{DIALOGUE_DASH}Do you have someone?"
        assert cues[1]["text"] == f"{DIALOGUE_DASH}I did once"

    def test_a_dashed_turn_is_never_merged_back(self):
        from services.subtitle_engine import DIALOGUE_DASH, words_to_cues

        cues = words_to_cues(self._question_then_answer(), max_line=200, max_lines=2)
        assert len(cues) == 2, "the merge step undid the turn split"
        assert all(c["text"].startswith(DIALOGUE_DASH) for c in cues)

    def test_no_dash_when_the_earlier_half_is_not_a_question(self):
        from services.subtitle_engine import DIALOGUE_DASH, words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "Statement"},
            {"s": 0.4, "e": 0.8, "w": "here."},
            {"s": 1.8, "e": 2.2, "w": "Another"},
            {"s": 2.2, "e": 2.6, "w": "one."},
        ]
        assert all(DIALOGUE_DASH not in c["text"] for c in words_to_cues(words))

    def test_the_dash_matches_the_translator_prompt_constant(self):
        """One dash, two modules. They must not drift."""
        from services.subtitle_engine import DIALOGUE_DASH
        from services.translation_v2 import DIALOGUE_DASH as PROMPT_DASH

        assert DIALOGUE_DASH == PROMPT_DASH == "— "


@pytest.mark.unit
class TestRefuseSplitOnDegenerateTerminal:
    """A full stop on a word with no duration is punctuation nobody spoke."""

    def test_a_zero_duration_terminal_does_not_end_a_sentence(self):
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 0.5, "w": "They"},
            {"s": 0.5, "e": 1.0, "w": "have"},
            {"s": 1.0, "e": 1.0, "w": "record."},  # 0.000s — the corpus case
            {"s": 1.0, "e": 1.5, "w": "They"},
            {"s": 1.5, "e": 2.0, "w": "have"},
            {"s": 2.0, "e": 2.6, "w": "reputation."},
        ]
        cues = words_to_cues(words)
        assert len(cues) == 1, f"split on punctuation with no speech under it: {cues}"

    def test_a_real_terminal_still_ends_a_sentence(self):
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 0.5, "w": "They"},
            {"s": 0.5, "e": 1.0, "w": "have"},
            {"s": 1.0, "e": 1.3, "w": "record."},  # 0.30s: really said
            {"s": 2.2, "e": 2.7, "w": "They"},
            {"s": 2.7, "e": 3.2, "w": "have"},
            {"s": 3.2, "e": 3.8, "w": "reputation."},
        ]
        assert len(words_to_cues(words)) == 2

    def test_the_refusal_is_logged(self, caplog):
        from services.subtitle_engine import words_to_cues

        with caplog.at_level("INFO"):
            words_to_cues(
                [
                    {"s": 0.0, "e": 0.5, "w": "Hello"},
                    {"s": 0.5, "e": 0.5, "w": "there."},
                    {"s": 0.5, "e": 1.0, "w": "Bye."},
                ]
            )
        assert "refused" in caplog.text


@pytest.mark.unit
class TestLeadOutCap:
    """A lead-out buys reading time; past a second it is a caption sitting on silence."""

    def test_a_cue_never_holds_more_than_the_cap_past_its_own_speech(self):
        from services.subtitle_engine import LEAD_OUT_MAX, words_to_cues

        words = [
            {"s": 0.0, "e": 0.4, "w": "From"},
            {"s": 0.4, "e": 0.8, "w": "whom?"},
            {"s": 20.0, "e": 20.5, "w": "Later."},
        ]
        cues = words_to_cues(words)
        # 3.84s on screen for a 4-character answer was the measured defect.
        assert cues[0]["end"] <= 0.8 + LEAD_OUT_MAX + 1e-6, cues[0]

    def test_the_minimum_duration_floor_still_wins_over_the_cap(self):
        from services.subtitle_engine import MIN_CUE_DUR, words_to_cues

        cues = words_to_cues([{"s": 0.0, "e": 0.1, "w": "Hi."}])
        assert cues[0]["end"] - cues[0]["start"] >= MIN_CUE_DUR - 1e-6

    def test_the_last_cue_is_capped_too(self):
        from services.subtitle_engine import LEAD_OUT_MAX, words_to_cues

        words = [
            {"s": 0.0, "e": 2.0, "w": "A"},
            {"s": 2.0, "e": 4.0, "w": "long sentence that has plenty of duration."},
        ]
        cues = words_to_cues(words)
        assert cues[-1]["end"] <= 4.0 + LEAD_OUT_MAX + 1e-6

    def test_the_cap_is_configurable(self):
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 2.0, "w": "Something"},
            {"s": 2.0, "e": 4.0, "w": "spoken here."},
            {"s": 30.0, "e": 31.0, "w": "Later."},
        ]
        tight = words_to_cues(words, lead_out_max=0.2)
        assert tight[0]["end"] <= 4.0 + 0.2 + 1e-6


@pytest.mark.unit
class TestVideoDurationClamp:
    """No cue may end after the picture does."""

    def test_the_final_lead_out_is_clamped_to_eof(self):
        from services.subtitle_engine import words_to_cues

        words = [{"s": 0.0, "e": 2.0, "w": "A sentence with real duration here."}]
        cues = words_to_cues(words, video_duration=2.3)
        assert cues[-1]["end"] == 2.3

    def test_a_clamped_cue_still_has_positive_duration(self):
        from services.subtitle_engine import words_to_cues

        cues = words_to_cues([{"s": 5.0, "e": 5.5, "w": "Late."}], video_duration=5.1)
        assert cues[0]["end"] > cues[0]["start"]

    def test_no_duration_means_no_clamp(self):
        from services.subtitle_engine import words_to_cues

        words = [{"s": 0.0, "e": 2.0, "w": "A sentence with real duration here."}]
        assert words_to_cues(words, video_duration=None)[-1]["end"] > 2.3

    def test_the_clamp_is_logged(self, caplog):
        from services.subtitle_engine import words_to_cues

        with caplog.at_level("INFO"):
            words_to_cues(
                [{"s": 0.0, "e": 2.0, "w": "Words here now."}], video_duration=2.1
            )
        assert "clamped" in caplog.text


@pytest.mark.unit
class TestWordStats:
    """Each cue carries the degeneracy read-out the hallucination gate needs."""

    def test_stats_are_attached_to_every_cue(self):
        from services.subtitle_engine import words_to_cues

        cues = words_to_cues([{"s": 0.0, "e": 0.5, "w": "Hello."}])
        assert cues[0]["word_stats"] == {
            "words": 1,
            "degenerate": 0,
            "degenerate_run": 0,
            "min_word_dur": 0.5,
        }

    def test_a_run_of_degenerate_words_is_counted(self):
        from services.subtitle_engine import words_to_cues

        words = [
            {"s": 0.0, "e": 0.5, "w": "Do"},
            {"s": 60.0, "e": 60.0, "w": "you"},
            {"s": 60.0, "e": 60.0, "w": "have"},
            {"s": 60.0, "e": 60.0, "w": "a"},
            {"s": 60.0, "e": 60.04, "w": "boyfriend?"},
        ]
        stats = words_to_cues(words)[-1]["word_stats"]
        assert stats["degenerate_run"] >= 4
        assert stats["min_word_dur"] == 0.0


@pytest.mark.unit
class TestQuotedPhrasePairing:
    """A cue that opens ״ and closes " is visibly broken, and two of eight corpus clips
    shipped one.

    The cause: the acronym rule converts an ASCII quote sitting between two Hebrew
    letters, which is true of the OPENING quote in ב"פחד גורם" (between ב and פ) and
    false of the closing one (between ם and ?). Acronyms are single marks; quotations
    are pairs, so the pair has to be claimed first.
    """

    def typo(self, text):
        from services.subtitle_engine import hebrew_typography

        return hebrew_typography(text)

    def test_the_shipped_defect(self):
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('נלחמת באחד המתמודדים ב"פחד גורם"?') == (
            f"נלחמת באחד המתמודדים ב{G}פחד גורם{G}?"
        )

    def test_a_quotation_before_a_full_stop(self):
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('הבחור מ"הישרדות".') == f"הבחור מ{G}הישרדות{G}."

    def test_a_half_converted_pair_is_repaired(self):
        """GPT-4o emits the two marks interchangeably; both ends still have to match."""
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo(f'ב{G}פחד גורם"?') == f"ב{G}פחד גורם{G}?"

    def test_a_quotation_with_no_prefix_letter(self):
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('הם אמרו "שלום" והלכו.') == f"הם אמרו {G}שלום{G} והלכו."

    # --- what must NOT be treated as a quotation -------------------------------------
    def test_a_plain_acronym_is_untouched(self):
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('עם מפכ"ל המשטרה,') == f"עם מפכ{G}ל המשטרה,"

    def test_an_inflected_acronym_keeps_its_mark(self):
        """דו"חות and מל"טים carry MORE than one letter after the mark and are still
        acronyms — which is why the pair rule, not a letter count, is what decides."""
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('הוא הגיש דו"חות רבים.') == f"הוא הגיש דו{G}חות רבים."

    def test_two_acronyms_in_one_line_are_not_a_pair(self):
        from services.subtitle_engine import GERSHAYIM as G

        assert self.typo('בצה"ל ובמפכ"ל.') == f"בצה{G}ל ובמפכ{G}ל."

    def test_an_english_quotation_inside_a_hebrew_line_is_untouched(self):
        assert self.typo('הוא אמר "hi" ואז הלך.') == 'הוא אמר "hi" ואז הלך.'

    def test_empty_and_none(self):
        from services.subtitle_engine import quoted_phrases

        assert quoted_phrases("") == ""
        assert quoted_phrases(None) is None
