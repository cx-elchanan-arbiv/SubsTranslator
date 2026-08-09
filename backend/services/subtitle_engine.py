"""
Subtitle spotting engine — turns word-level timestamps into broadcast-quality cues
and renders them as an ASS (Advanced SubStation Alpha) file.

Why this exists
---------------
Whisper segments are *speech* segments, not *subtitle* cues. Used directly they
produce the defects measured against a professional reference on the same clip:

  * 0.2-0.3s flashes holding a single word ("things.", "Yeah."),
  * sentences cut mid-clause ("It could complicate" | "things."),
  * a question and its answer merged into one cue (no speaker turn),
  * lines far longer than a viewer can read at 24 fps.

This module re-spots the transcript from *words*: words -> sentences -> cues with
duration/length constraints, then a lead-out pass that hands each cue the dead air
in front of the next one (the single biggest CPS win, and standard editorial
practice). Constraints follow the Netflix Hebrew Timed Text Style Guide:
42 characters per line, at most 2 lines, ~17 characters per second for adults.

Those editorial limits are ceilings, not constants: the frame gets a vote. See
:func:`layout_params` — a 42-character line at 6.1% of a PORTRAIT frame's height is
wider than the frame, and ``WrapStyle: 2`` means libass renders the surplus off the
edge rather than wrapping it. Every character budget in the pipeline comes from that
one function so the text that reaches the renderer is text the renderer can fit.

Rendering notes (both re-verified by rendering in this project's own container,
FFmpeg 7.1.5 / libass 0.17.3 — see ``tests/integration/test_bidi_render.py``)
--------------------------------------------------------------------------
* FFmpeg's ``subtitles`` filter has **no** ``shaping`` option ("Option not
  found", hard failure). The ``ass`` filter does. That is why this engine emits
  ``.ass`` — it must be rendered with ``ass=...:shaping=complex``.
* Hebrew bidi: **libass hard-defaults the paragraph direction to LTR.** It does
  not auto-detect it from the first strong character — rendering ``אבג DEF``
  with no controls puts the Hebrew on the *left*, which is the LTR answer, not
  the first-strong-character one. That single missing fact is the whole bidi
  problem here: with no controls every mixed Hebrew line comes out with its
  WORD ORDER REVERSED. So the line must declare its own direction with
  U+2067 (RLI) ... U+2069 (PDI), and each maximal Latin/digit stretch is kept
  intact with U+2066 (LRI) ... U+2069 (PDI) — see :func:`bidi_isolate` for why
  the runs must be *maximal* and why the line-level RLI is not optional.
  U+202E (RLO) — used by the legacy path in ``utils/rtl_utils.py`` — is
  deliberately never emitted: it is an override, it defeats the bidi algorithm
  instead of cooperating with it.

Every capability is an independent keyword toggle so the API/UI layer can expose
them one by one; nothing here reads config or touches the filesystem.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# --- Netflix Hebrew Timed Text limits (defaults, all overridable) -------------
#: Characters per line the layout AIMS for. It is a ceiling, not a promise: the real
#: per-video limit comes out of :func:`layout_params`, which will not let a line be
#: wider than the frame. See that function for why a fixed 42 is a portrait bug.
MAX_LINE_CHARS = 42
MAX_LINES = 2
MIN_CUE_DUR = 1.2
MAX_CUE_DUR = 6.0
MIN_CUE_GAP = 0.08

#: Speech-pause length (seconds) treated as a sentence boundary when the ASR
#: transcript arrives without punctuation (large-v3 sometimes does this).
PAUSE_SPLIT_GAP = 0.35

#: Longest silence (seconds) a short fragment may be merged across. Two utterances
#: separated by more than this are not the same breath, whatever their length: the
#: merge step exists to repair sub-second splinters, not to span dead air.
MERGE_MAX_GAP = 1.0

#: Silence INSIDE one sentence group (seconds) that is treated as a speaker turn and
#: splits the cue.
#:
#: The defect: a 0.72s pause sat unused inside one cue of a real job, gluing an
#: interviewer's question to the answer and attributing the answer to the wrong person on
#: screen. Whisper's own segment boundaries do not see it — it punctuates a turn as one
#: sentence — but the WORD timestamps carry the pause, and this engine already re-spots
#: from words. 0.7s is well above a breath or a comma pause (0.2-0.4s, which
#: :data:`PAUSE_SPLIT_GAP` treats as a mere sentence boundary) and well below the
#: :data:`MERGE_MAX_GAP` silence that separates unrelated utterances.
TURN_SPLIT_GAP = 0.7

#: Dialogue dash + space, prefixed to BOTH halves when a turn split separates a question
#: from what follows it. U+2014 EM DASH is the mark Hebrew broadcast subtitling uses to
#: say "different speaker"; the space after it is part of the convention.
DIALOGUE_DASH = "— "

#: Longest lead-out (seconds) a cue may claim from the silence in front of it.
#:
#: Without a cap, "hand every cue the dead air ahead of it" turns into "hold every line
#: until the next one starts": on a real street interview EVERY cue was stretched to a
#: zero gap and a 4-character answer ("ממי?") stayed on screen for 3.84 seconds, with 9
#: cues lingering more than a second past the end of their own speech. A lead-out buys
#: reading time for the line just spoken; past about a second it is no longer reading
#: time, it is a caption sitting on silence. :func:`translation_v2.enforce_cps` may
#: exceed this cap, but only for a cue that genuinely cannot be read in the time it has.
LEAD_OUT_MAX = 1.0

#: Characters per second above which a SOURCE cue is SUSPECTED of being an ASR
#: hallucination. **A suspicion, not a verdict** — see :func:`drop_hallucinated_cues`.
#:
#: Why 35 and not the editorial 17
#: -------------------------------
#: :data:`translation_v2.DEFAULT_MAX_CPS` (17) is a READING limit — how fast a viewer can
#: take text in. This is a different quantity entirely: a PHYSICAL limit on how fast a
#: human being can speak. Conversational English runs ~12-16 CPS and fast auctioneer-grade
#: delivery tops out near 25.
#:
#: Why it stopped being a verdict
#: ------------------------------
#: It was one, and it was wrong in both directions. Measured over this project's own
#: 20-run corpus archive, the CPS>35 rule deleted 8 source cues and **not one of them was
#: a fabrication**: two were genuine fast standup delivery in a podcast (39.1 and 36.4
#: CPS) and six were genuine CROSS-TALK in an argument, where Whisper compresses the
#: word timestamps of overlapping speakers and so inflates the measured rate of text that
#: was really spoken. In the same corpus, real speech reaches 58.9 CPS at cue level and
#: 66.7 at sentence level ("Bye." in 0.06s), while the fabrications that were actually
#: present sat at 14.7 and 30 CPS — comfortably UNDER the threshold.
#:
#: The measurement is dominated by Whisper's timestamp error, which is roughly constant
#: in absolute time and therefore enormous in relative terms on a short utterance. So 35
#: CPS is now one SOFT signal among several and can no longer drop a cue on its own.
MAX_SOURCE_CPS = 35.0

#: Characters per second at which text stops being physically utterable by any margin of
#: timestamp error, and the cue is dropped on that signal alone.
#:
#: MEASURED: the corpus's fastest genuine cue is 58.9 CPS and its fastest genuine
#: sentence 66.7; the fabricated Trump line documented in
#: ``transcription_service.ASR_PUNCTUATION_PRIMER`` measured 219 (57 characters in
#: 0.26s). 90 sits in the empty gap between those two populations — 35% above anything
#: real ever measured here and less than half of the fabrication.
IMPOSSIBLE_SOURCE_CPS = 90.0

#: A word timestamp shorter than this is DEGENERATE: Whisper's word times are quantised
#: to 0.02s, so anything under 0.05s is one or two ticks — the shape it produces when it
#: has text to place and no audio to place it on.
#:
#: On its own this means very little: ordinary function words after a sentence terminal
#: ("It", "a", "the") routinely land here in perfectly good transcripts — 10 of them in
#: one 509-word corpus file. What is diagnostic is a RUN of them
#: (:data:`DEGENERATE_WORD_RUN`), which is Whisper stamping a whole clause onto a single
#: instant, and a terminal ``.``/``?``/``!`` carried by such a word, which is a sentence
#: boundary that no speech supports (see :func:`words_to_cues`).
DEGENERATE_WORD_DUR = 0.05

#: Consecutive degenerate words that make :func:`drop_hallucinated_cues`'s
#: ``degenerate_words`` signal fire. Corpus: genuine fast speech peaks at a run of 1;
#: the two known fabricated blocks ran 5 and 6 words long.
DEGENERATE_WORD_RUN = 3

#: How far below the file's own average level a cue's audio must sit before a
#: soft-signal drop is allowed to proceed. Speech that Whisper merely mis-TIMED is still
#: speech and still makes noise; a fabrication sits on silence, applause or room tone.
#:
#: MEASURED on the corpus: the six cross-talk cues wrongly dropped by the old rule sit at
#: -0.1, -0.5 and +0.6 dB relative to their file — i.e. at full speech level — while a
#: verified Whisper loop repetition sits at -14.9 dB. -6.0 is inside that gap. The probe
#: is optional; with no probe there is no veto and the signals decide alone.
ENERGY_VETO_DB = -6.0

#: Shortest cue that can actually be seen (~1 frame at 25 fps). A cue that cannot
#: be given this much time without overlapping its neighbour is not shortened to
#: zero \u2014 it is folded into that neighbour. See :func:`words_to_cues`.
MIN_VISIBLE_DUR = 0.04

# --- Unicode bidi isolates (never RLO/U+202E) --------------------------------
# Spelled as escapes on purpose: these characters are invisible, so a literal
# would be impossible to review and trivial to delete by accident.
RLI = "\u2067"  # RIGHT-TO-LEFT ISOLATE
LRI = "\u2066"  # LEFT-TO-RIGHT ISOLATE
PDI = "\u2069"  # POP DIRECTIONAL ISOLATE

GERSHAYIM = "״"  # ״ HEBREW PUNCTUATION GERSHAYIM (acronyms: צה״ל)
GERESH = "׳"  # ׳ HEBREW PUNCTUATION GERESH (foreign phonemes: ג׳ורג׳)

# Hebrew letters only: alef-tav plus the yod/vav ligatures. Deliberately not
# the whole U+0590-U+05FF block, which also holds niqqud and punctuation.
_HEBREW_LETTERS = "\u05d0-\u05ea\u05ef-\u05f2"

#: The four Hebrew letters that actually take a geresh to carry a foreign phoneme in
#: modern Hebrew: \u05d2\u05f3 = j, \u05d6\u05f3 = zh, \u05e6\u05f3/\u05e5\u05f3 = ch/tch. Restricting the rule to these is what
#: makes it safe to fire at the END of a word as well as between two letters \u2014 and
#: firing at the end is not optional, because the defect this exists to fix is names
#: like \u05d2'\u05d5\u05e8\u05d2', whose SECOND apostrophe is word-final and a between-letters rule misses.
#:
#: Every other Hebrew letter is excluded on purpose. \u05e8, \u05d5, \u05d3 and \u05ea do take a geresh in
#: academic Arabic transliteration, but they are also very common word-FINAL letters
#: (\u05d3\u05d1\u05e8, \u05e9\u05d9\u05e8, \u05e1\u05e4\u05e8), so admitting them would rewrite the closing quote of any
#: single-quoted Hebrew word. The residual false positive \u2014 a closing ' immediately
#: after \u05d2/\u05d6/\u05e6/\u05e5 \u2014 is a far rarer shape than the names this repairs.
_GERESH_LETTERS = "\u05d2\u05d6\u05e6\u05e5"  # \u05d2 \u05d6 \u05e6 \u05e5

# A word ends a sentence when it ends with . ! ? possibly followed by a closer.
_TERMINAL_RE = re.compile(r"[.!?][\"'”’)\]]*$")
# Sentence boundary *inside* an assembled cue text.
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*(?:\s|$)")
# ASCII '"' between two Hebrew letters is an acronym mark, not a quote.
_GERSHAYIM_RE = re.compile(f'(?<=[{_HEBREW_LETTERS}])"(?=[{_HEBREW_LETTERS}])')

#: The single Hebrew letters that attach to the FRONT of a word (ב, ה, ו, כ, ל, מ, ש).
#: A quotation mark may sit between one of them and the word it introduces —
#: ב"פחד גורם" — which is exactly the shape the acronym rule above cannot tell apart
#: from צה"ל without looking at the OTHER end of the quotation.
_HEB_PREFIX_LETTERS = "בהוכלמש"

#: A QUOTED PHRASE inside Hebrew text, matched as a PAIR.
#:
#: The defect this closes: "ב\"פחד גורם\"?" came out of the renderer as ב״פחד גורם"? —
#: the opening quote converted to a gershayim (it sits between two Hebrew letters) and
#: the closing one did not (it is followed by a "?"). A cue that opens with one mark and
#: closes with another is visibly broken, and it happened to two of eight corpus clips.
#:
#: Matched as a pair, and only when the quotation opens at a word boundary (optionally
#: after one prefix letter) and closes somewhere that is NOT the middle of a word:
#:
#:   ב"פחד גורם"?   -> a quotation      (opens after the prefix ב, closes before ?)
#:   מ"הישרדות".    -> a quotation
#:   דו"חות         -> NOT a quotation  (opens after two letters; no closing mark)
#:   מפכ"ל          -> NOT a quotation  (the acronym rule keeps it)
#:
#: The content must contain a Hebrew letter, so an English quotation inside a Hebrew
#: line (``he said "hi"``) keeps its ASCII quotes exactly as the docstring promises.
#: Either mark may already be a gershayim — GPT-4o mixes them — and both come out as one.
_QUOTED_PHRASE_RE = re.compile(
    rf"(?<![{_HEBREW_LETTERS}\w])"
    rf"([{_HEB_PREFIX_LETTERS}]?)"
    rf'["{GERSHAYIM}]'
    rf'(?P<body>[^"{GERSHAYIM}\n]*[{_HEBREW_LETTERS}][^"{GERSHAYIM}\n]*)'
    rf'["{GERSHAYIM}]'
    rf"(?![{_HEBREW_LETTERS}])"
)

# An apostrophe DIRECTLY after ג/ז/צ/ץ is a geresh. No lookahead is needed and none is
# used: the lookbehind alone already protects every English apostrophe, because those
# are preceded by a Latin letter (don't, it's, Charles') and never by a Hebrew one.
# U+2019 (the curly ’) is matched too — GPT-4o emits it interchangeably with ASCII ',
# and it is the same defect wearing a different codepoint.
_GERESH_RE = re.compile(f"(?<=[{_GERESH_LETTERS}])['’]")

# Unicode bidirectional categories, used by the bidi pass below instead of a
# hand-maintained character class (which is exactly what got this wrong before:
# a literal class that forgot "%" renders "50%" as "%50").
_BIDI_STRONG_RTL = frozenset({"R", "AL"})  # Hebrew, Arabic, Thaana, ...
_BIDI_LTR_ANCHOR = frozenset({"L", "EN", "AN"})  # Latin letters and digits
_BIDI_NUMERIC_TERMINATOR = "ET"  # $ € ₪ % ‰ ° ...

# Break-preference bonuses used when choosing where to split two lines.
_BONUS_SENTENCE_BREAK = 30
_BONUS_CLAUSE_BREAK = 12
_PENALTY_TOP_HEAVY = 3

# Horizontal margin in ASS units (== pixels, since PlayResX/Y match the video).
# WrapStyle 2 disables libass's own wrapping, so this is NOT a wrap boundary — it is
# purely the title-safe inset the layout maths must keep the text inside of.
_MARGIN_H = 60

#: Hard cap on the horizontal margin as a fraction of the width, so a very narrow
#: frame cannot end up with a negative usable width. 15% x 2 leaves 70% of the frame.
#: Any video at least 400px wide keeps the full 60px (0.15 * 400 == 60), so every
#: real-world resolution — 720p, 1080p, 4K, 720x1280 portrait — is unaffected.
_MARGIN_H_MAX_FRAC = 0.15

#: Font size as a fraction of the video HEIGHT (the historical, landscape-tuned value).
DEFAULT_FONT_FRAC = 0.061

#: Bottom margin as a fraction of the video height.
DEFAULT_MARGIN_V_FRAC = 0.12

#: Floor on the font size, as a fraction of the video height. A portrait frame cannot
#: fit 42 characters at 6.1% of its (large) height, and the honest trade is a NARROWER
#: LINE, not microscopic text: below ~3.2% of height the subtitle stops being legible
#: on a phone, which is the only device that plays 9:16 video.
MIN_FONT_FRAC = 0.032

#: Rendered line width divided by (characters x font size), for the exact style
#: :func:`build_ass` emits — Noto Sans Hebrew **Bold**, BorderStyle 4 opaque box,
#: Outline 3, shaping=complex.
#:
#: MEASURED, not guessed: 30 renders (10 real Hebrew subtitle lines of 33-41 characters
#: x font sizes 60/100/160) inside this project's own container, each rasterised by the
#: same FFmpeg/libass that renders production output and measured by scanning pixel
#: columns for the box. The ratio is flat across font sizes (0.3612 / 0.3607 / 0.3623
#: for the same string at 60/100/160), which is what makes a single constant valid.
#:
#:     Hebrew        mean 0.3861   p95 0.4300   max 0.4318
#:     Hebrew+Latin  mean 0.3903   p95 0.3947   max 0.3956
#:     Latin lower   mean 0.3741   p95 0.3926   max 0.3932
#:
#: 0.44 sits ~2% above the widest real line measured, so the estimate is conservative
#: for every script this pipeline actually emits. KNOWN LIMIT: Latin ALL-CAPS measures
#: 0.51 mean / 0.62 max and would overflow this estimate — broadcast subtitles are not
#: set in caps, and the landscape path has always had that same exposure (there the
#: geometric limit is never the binding one anyway, see :func:`layout_params`).
#:
#: The measurement harness is `tests/integration/test_subtitle_layout_render.py`
#: (``test_measured_glyph_width_ratio_still_holds``), which re-derives this number from
#: live renders and fails if the font ever changes underneath it.
GLYPH_WIDTH_RATIO = 0.44

#: Never let the derived per-line budget collapse to something unwrappable. Only a
#: frame narrower than ~200px can reach this, and at that size no layout is readable.
MIN_LINE_CHARS = 8

# Style name. Deliberately NOT "Default": libass appends a built-in "Default"
# style to every track and resolves style names by scanning the list backwards,
# so a style of ours called "Default" could lose to libass's Arial fallback.
STYLE_NAME = "He"

#: Subtitle placement -> ``Alignment`` code written into the ``.ass`` style line.
#:
#: ASS v4+ numbers alignment like a numeric keypad — 7 8 9 across the top, 4 5 6 across
#: the middle, 1 2 3 across the bottom. That is what :func:`build_ass` emits and what
#: FFmpeg's ``ass`` filter reads, and it is verified by rendering.
#:
#: It is deliberately NOT shared with the legacy renderer. A ``force_style`` override on
#: FFmpeg's ``subtitles`` filter is parsed with the older SSA v4 scheme, in which the
#: same numbers mean different places — see :data:`subtitle_service.SSA_ALIGNMENT`. Two
#: renderers, two dialects; only ``2`` (bottom centre) means the same thing in both.
ASS_ALIGNMENT = {"bottom": 2, "top": 8, "side": 6}


# =============================================================================
# layout: the one place the frame's WIDTH gets a vote
# =============================================================================
def layout_params(
    video_w: int,
    video_h: int,
    *,
    font_frac: float = DEFAULT_FONT_FRAC,
    margin_v_frac: float = DEFAULT_MARGIN_V_FRAC,
    max_line_chars: int = MAX_LINE_CHARS,
    max_lines: int = MAX_LINES,
    glyph_ratio: float = GLYPH_WIDTH_RATIO,
    min_font_frac: float = MIN_FONT_FRAC,
) -> dict[str, Any]:
    """Derive every layout number for one video from BOTH of its dimensions.

    The bug this exists to kill
    ---------------------------
    Sizing the font from the height alone and capping lines at a fixed 42 characters
    is only safe while the frame is wider than it is tall. On a 720x1280 portrait
    clip the two rules produce a 78px font and a 42-character line — about 1450px of
    text on a 720px frame — and because ``WrapStyle: 2`` switches libass's own
    wrapping OFF, the surplus is not wrapped, it is drawn straight off BOTH edges of
    the picture. Verified on a real job (``IMG_8975.MP4``, 720x1280).

    The rule
    --------
    ``usable_w = video_w - 2 * margin_h`` is the only width text may occupy. A line of
    ``n`` characters at font size ``f`` renders about ``n * f * glyph_ratio`` pixels
    wide (:data:`GLYPH_WIDTH_RATIO`, measured — see its docstring). So:

    1. **font size** = ``min(height-derived, width-derived)``, then floored at
       ``min_font_frac * video_h``. The width-derived term is what a 42-character line
       would need in order to fit; on 16:9 it is never the smaller of the two, which is
       precisely why landscape output is bit-for-bit what it was before this function
       existed. On 9:16 it bites hard, and the floor stops it from answering "26px".
    2. **characters per line** = how many characters actually fit at that font size,
       capped at ``max_line_chars`` (a wider frame does not earn longer lines — 42 is
       an editorial reading-comfort limit, not a geometric one).

    Step 2 is what absorbs the floor applied in step 1: portrait keeps a legible font
    and pays for it in line length instead of in legibility.

    Worked examples (glyph_ratio 0.44, defaults):

        1280x720   font 44, 42 chars/line  <- unchanged from the height-only rule
        1920x1080  font 66, 42 chars/line  <- unchanged
        3840x2160  font 132, 42 chars/line <- unchanged
        1080x1080  font 51, 42 chars/line  <- square: font shrinks, budget intact
        720x1280   font 41, 33 chars/line  <- portrait: was 78px / 42 chars, off-frame

    Why not three lines on portrait: with the measured ratio a 720-wide frame still
    holds 33 characters per line, so two lines carry 66 of the 84-character landscape
    budget (79%). That is a mild condensation, not a mangling, and it keeps the
    standard two-line broadcast shape. A third line would buy 33 more characters at
    the cost of covering another 8% of the picture — not a trade worth making until
    the budget actually hurts. (At ``glyph_ratio`` 0.55 it would have been 26 chars /
    52 per cue, and the answer would have gone the other way. Measuring changed the
    editorial decision — hence the insistence on measuring.)

    Args:
        video_w: frame width in pixels.
        video_h: frame height in pixels.
        font_frac: font size as a fraction of the height (the landscape rule).
        margin_v_frac: bottom margin as a fraction of the height.
        max_line_chars: editorial ceiling on characters per line.
        max_lines: lines per cue.
        glyph_ratio: rendered width per character per font pixel.
        min_font_frac: floor on the font size, as a fraction of the height.

    Returns:
        ``{"video_w", "video_h", "font_px", "margin_h", "margin_v", "usable_w",
        "max_line_chars", "max_lines", "max_chars_per_cue"}``. ``max_chars_per_cue``
        is ``max_line_chars * max_lines`` — the ONE number every upstream stage
        (spotting, translation, the CPS pass, reflow) must budget against, so the text
        that arrives at the renderer is text the renderer can actually fit.
    """
    width = max(1, int(video_w or 0))
    height = max(1, int(video_h or 0))
    glyph_ratio = (
        float(glyph_ratio) if glyph_ratio and glyph_ratio > 0 else GLYPH_WIDTH_RATIO
    )
    max_line_chars = max(1, int(max_line_chars))
    max_lines = max(1, int(max_lines))

    margin_h = min(_MARGIN_H, int(width * _MARGIN_H_MAX_FRAC))
    usable_w = max(1, width - 2 * margin_h)

    font_by_height = max(1, round(height * float(font_frac)))
    # Largest font at which a full-length line still fits between the margins.
    font_by_width = int(usable_w // (max_line_chars * glyph_ratio))
    font_floor = max(1, round(height * float(min_font_frac)))
    font_px = max(font_floor, max(1, min(font_by_height, font_by_width)))

    # How many characters that font actually affords. Never more than the editorial
    # ceiling: a 4K frame could fit 100 characters per line and must not be given them.
    fitting_chars = int(usable_w // (font_px * glyph_ratio))
    effective_line_chars = min(max_line_chars, fitting_chars)
    if effective_line_chars < MIN_LINE_CHARS:
        logger.warning(
            "layout: %dx%d only affords %d chars/line at font %dpx — clamping to %d; "
            "lines may overflow this frame",
            width,
            height,
            fitting_chars,
            font_px,
            MIN_LINE_CHARS,
        )
        effective_line_chars = MIN_LINE_CHARS

    params = {
        "video_w": width,
        "video_h": height,
        "font_px": font_px,
        "margin_h": margin_h,
        "margin_v": max(0, round(height * float(margin_v_frac))),
        "usable_w": usable_w,
        "max_line_chars": effective_line_chars,
        "max_lines": max_lines,
        "max_chars_per_cue": effective_line_chars * max_lines,
    }
    if effective_line_chars < max_line_chars or font_px < font_by_height:
        logger.info(
            "layout: %dx%d -> font %dpx (height rule wanted %dpx), %d chars/line "
            "(ceiling %d), %d chars/cue — width-constrained frame",
            width,
            height,
            font_px,
            font_by_height,
            effective_line_chars,
            max_line_chars,
            params["max_chars_per_cue"],
        )
    return params


# =============================================================================
# words -> cues
# =============================================================================
def _normalize_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy/clean the input word list: strip padding, coerce times, drop empties.

    Accepts the compact ``{"s","e","w"}`` shape and Whisper's own
    ``{"start","end","word"}`` shape.

    Negative timestamps are clamped to 0 *here*, before anything downstream can
    reason about them: a negative start would make every later duration/gap
    calculation meaningless and can only come from broken ASR output.
    """
    cleaned: list[dict[str, Any]] = []
    for raw in words or []:
        text = str(raw.get("w", raw.get("word", "")) or "").strip()
        if not text:
            continue
        try:
            start = float(raw.get("s", raw.get("start", 0.0)))
            end = float(raw.get("e", raw.get("end", start)))
        except (TypeError, ValueError):
            logger.debug("subtitle_engine: skipping unparsable word %r", raw)
            continue
        if start != start or end != end:  # NaN: every comparison below would be False
            logger.debug("subtitle_engine: skipping word with NaN timing %r", raw)
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        cleaned.append({"s": start, "e": max(end, start), "w": text})
    # Stable sort: well-formed input is already monotonic, garbage gets repaired.
    cleaned.sort(key=lambda w: w["s"])
    return cleaned


def _text_of(group: list[dict[str, Any]]) -> str:
    """Assemble a cue's text from its words, with the dialogue dash when it carries one."""
    text = " ".join(w["w"] for w in group)
    if group and group[0].get("_dash"):
        return DIALOGUE_DASH + text
    return text


def _span_of(group: list[dict[str, Any]]) -> float:
    """Wall-clock span covered by a group of words."""
    return group[-1]["e"] - group[0]["s"]


def _sentence_count(text: str) -> int:
    """How many sentences the text already holds."""
    return len(_SENTENCE_END_RE.findall(text))


def _split_long_group(
    group: list[dict[str, Any]], max_chars: int
) -> list[list[dict[str, Any]]]:
    """Recursively split an over-long group at the comma nearest its middle.

    Falls back to the middle word when the group holds no comma, and gives up
    (returns the group as-is) when it cannot be split any further.
    """
    if len(group) < 2 or len(_text_of(group)) <= max_chars:
        return [group]

    commas = [i for i, w in enumerate(group[:-1]) if w["w"].endswith(",")]
    middle = len(group) // 2
    cut = min(commas, key=lambda i: abs(i - middle)) + 1 if commas else middle
    cut = max(1, min(cut, len(group) - 1))  # never produce an empty half
    return _split_long_group(group[:cut], max_chars) + _split_long_group(
        group[cut:], max_chars
    )


def _word_dur(word: dict[str, Any]) -> float:
    """Duration of one word timestamp, never negative."""
    return max(0.0, float(word["e"]) - float(word["s"]))


def _split_on_turn_gaps(
    group: list[dict[str, Any]], turn_gap: float
) -> list[list[dict[str, Any]]]:
    """Split one sentence group wherever the speech pauses for ``turn_gap`` or more.

    Whisper punctuates a question and its answer as a single sentence when the two run
    together ("...מי היה מגן על נשים? ממי?"), and the segment boundary it emits does not
    see the pause between them. The WORD timestamps do, which is the whole reason this
    engine re-spots from words: a silence of :data:`TURN_SPLIT_GAP` inside one sentence
    is a speaker turn, and a cue that spans it puts one person's words under another
    person's face.

    When the split separates a ``?``-terminated part from what follows — the cheapest
    available proxy for a turn, the same one :func:`_can_merge` uses — BOTH parts are
    marked to carry the dialogue dash (:data:`DIALOGUE_DASH`), so the two lines declare
    on screen that they belong to different speakers.

    Returns one part when there is no qualifying pause, so the caller can use it
    unconditionally.
    """
    if len(group) < 2 or turn_gap <= 0:
        return [group]

    parts: list[list[dict[str, Any]]] = []
    current = [group[0]]
    for previous, word in zip(group, group[1:]):
        if (word["s"] - previous["e"]) >= turn_gap:
            parts.append(current)
            current = [word]
        else:
            current.append(word)
    parts.append(current)

    if len(parts) == 1:
        return parts

    # Every part after the first OPENS a turn. The flag is what stops the merge steps
    # from immediately undoing the split — the two rules meet at exactly ``turn_gap``,
    # and without an explicit marker the boundary case (a gap of precisely 0.7s) is
    # split by one rule and re-merged by the other.
    for part in parts[1:]:
        part[0] = dict(part[0])
        part[0]["_turn"] = True

    for index, part in enumerate(parts[:-1]):
        if _text_of(part).rstrip().endswith("?"):
            # Copy before marking: the word dicts are shared with the caller's list.
            for target in (parts[index], parts[index + 1]):
                target[0] = dict(target[0])
                target[0]["_dash"] = True
    return parts


def _can_merge(
    previous: list[dict[str, Any]],
    group: list[dict[str, Any]],
    *,
    max_chars: int,
    max_dur: float,
    max_gap: float = MERGE_MAX_GAP,
) -> bool:
    """Is folding ``group`` onto the end of ``previous`` a legal cue merge?

    The single definition of a legal merge, used by BOTH merge steps in
    :func:`words_to_cues` — the short-fragment repair and the minimum-duration floor.
    Two steps with two hand-inlined copies of these five conditions is how they drift.

      * the combined text still fits the cue's character budget;
      * the combined span is at most ``max_dur``;
      * the silence between them is at most ``max_gap`` — two utterances further apart
        than that are not one breath, however short they are. The caller passes
        ``min(MERGE_MAX_GAP, turn_gap)``, because a silence long enough to be called a
        speaker TURN cannot also be short enough to merge across: that contradiction is
        what glued "Let's give them a new task." to another man's "Yeah, of course."
        across a measured 0.72s pause and put one speaker's words under the other's face;
      * ``previous`` does not end in ``?`` — the cheapest available proxy for a speaker
        turn, and gluing a question to its answer is the defect this engine exists to
        remove;
      * ``group`` does not open a new speaker turn (:func:`_split_on_turn_gaps` marked
        it), which is the same rule expressed in timing rather than in punctuation:
        undoing a turn split by merging would be a straight contradiction — and at a gap
        of exactly ``turn_gap`` both rules fire, so the marker, not the arithmetic, has
        to settle it;
      * ``previous`` does not already hold two sentences.
    """
    prev_text = _text_of(previous)
    return (
        len(prev_text) + 1 + len(_text_of(group)) <= max_chars
        and (group[-1]["e"] - previous[0]["s"]) <= max_dur
        and (group[0]["s"] - previous[-1]["e"]) <= max_gap
        and not prev_text.rstrip().endswith("?")
        and not group[0].get("_turn")
        and _sentence_count(prev_text) < 2
    )


def _word_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Degeneracy read-out for one cue's words, for the hallucination gate.

    Carried on the cue rather than the raw word list because the gate runs downstream of
    :func:`subtitle_pipeline.normalize_cues`, several stages away, and because four
    integers are cheap to archive where a few hundred word dicts per cue are not.

    ``degenerate_run`` is the longest CONSECUTIVE stretch of sub-:data:`DEGENERATE_WORD_DUR`
    words — the discriminating number. A single one is normal; a run of them is Whisper
    stamping a clause onto one instant. See :func:`drop_hallucinated_cues`.
    """
    degenerate = 0
    longest = current = 0
    for word in group:
        if _word_dur(word) < DEGENERATE_WORD_DUR:
            degenerate += 1
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "words": len(group),
        "degenerate": degenerate,
        "degenerate_run": longest,
        "min_word_dur": round(min((_word_dur(w) for w in group), default=0.0), 3),
    }


def words_to_cues(
    words: list[dict[str, Any]],
    *,
    max_line: int = MAX_LINE_CHARS,
    max_lines: int = MAX_LINES,
    min_dur: float = MIN_CUE_DUR,
    max_dur: float = MAX_CUE_DUR,
    min_gap: float = MIN_CUE_GAP,
    asr_primed: bool = False,
    turn_gap: float = TURN_SPLIT_GAP,
    lead_out_max: float = LEAD_OUT_MAX,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Re-spot word timestamps into readable subtitle cues.

    Pipeline:
      1. **Sentences** — group words, closing a group on a terminal ``. ! ?`` — unless
         the word carrying that terminal is itself DEGENERATE (shorter than
         :data:`DEGENERATE_WORD_DUR`). A full stop stamped on a word with no duration is
         punctuation the decoder invented, not a boundary the speaker produced, and
         splitting there hands the next stage a cue that starts mid-utterance.
      1b. **Turns** — a sentence group is split again wherever the speech pauses for
         ``turn_gap`` or more (:func:`_split_on_turn_gaps`). Whisper writes a question
         and its answer as one sentence; the word timestamps show the turn.
      2. **Split** — a sentence longer than ``max_line * max_lines`` characters is
         split at the comma nearest its middle, recursively.
      3. **Merge** — a group shorter than ``min_dur`` is folded into the previous
         cue only if the result still fits ``max_line * max_lines`` characters,
         spans at most ``max_dur``, the silence between the two is at most
         :data:`MERGE_MAX_GAP`, the previous cue does not end with ``?`` and the
         previous cue holds fewer than 2 sentences. The question rule keeps a
         question and its answer apart: it is the cheapest available proxy for a
         speaker turn, and merging across one is the defect this engine exists to
         remove. The gap rule stops "Hello." at 0.0-0.5s being glued to "Bye." at
         5.5-6.0s just because the second one is short.
      3c. **Floor** — a cue whose successor starts less than ``min_dur`` after it
         begins can never be granted ``min_dur`` by step 4, because there is no dead
         air left to grant. It is merged into that successor (the same
         :func:`_can_merge` constraints) rather than shipped under the floor. When
         that merge is impossible the short cue survives and a WARNING says so.
      4. **Lead-out** — each cue's end grows into the dead air ahead of it, by at most
         ``min(next_start - min_gap, end + lead_out_max)`` and never past
         ``start + max_dur``; when cues collide the end shrinks instead, but not below
         ``start + min_dur``. This is what buys the CPS headroom (~17 cps target).
         ``lead_out_max`` is what stops it becoming "hold every line until the next one
         starts" — see :data:`LEAD_OUT_MAX`.
      5. **Clamp** — with ``video_duration`` known, no cue may end after the picture
         does. libass simply stops drawing at EOF, so an over-hanging cue is not visible
         damage, but it makes every downstream measurement (coverage, lingering, the
         SRT handed to a user) describe time that does not exist.

    Args:
        words: word timestamps, ``{"s": float, "e": float, "w": str}``; ``w`` may
            carry leading/trailing whitespace.
        max_line: characters per line.
        max_lines: lines per cue (``max_line * max_lines`` is the cue budget).
        min_dur: shortest readable cue, seconds.
        max_dur: longest cue, seconds.
        min_gap: gap left between consecutive cues, seconds.
        asr_primed: whether the transcript came from an ASR run primed for punctuation
            (``transcription_service.ASR_PUNCTUATION_PRIMER``). Affects LOGGING ONLY,
            never behaviour. The pause fallback firing on an unprimed transcript is
            routine; firing on a PRIMED one means the primer failed on this clip, which
            is a materially more interesting event and is logged as such.
        turn_gap: internal silence treated as a speaker turn (:data:`TURN_SPLIT_GAP`).
            ``0`` disables turn splitting.
        lead_out_max: longest lead-out a cue may claim (:data:`LEAD_OUT_MAX`).
        video_duration: the source video's duration in seconds, when known. Cue ends are
            clamped to it. ``None`` skips the clamp.

    Returns:
        ``[{"start": float, "end": float, "text": str, "word_stats": {...}}, ...]``
        sorted by start, non-overlapping, and every cue with a strictly positive
        duration. Word order and wording are preserved exactly, except for the dialogue
        dash a turn split may prefix. ``word_stats`` is the degeneracy read-out
        :func:`drop_hallucinated_cues` needs — see :func:`_word_stats`.

        **Postcondition:** every cue lasts at least ``min_dur``, except ones this
        function logged a WARNING about (step 3c could not merge them) and the
        ~1ms that rounding to milliseconds can cost.
    """
    clean = _normalize_words(words)
    if not clean:
        return []

    max_chars = max(1, int(max_line) * int(max_lines))
    # A silence long enough to be a speaker TURN cannot also be short enough to merge
    # across. Without this the two rules contradict each other and the merge wins.
    merge_gap = (
        min(MERGE_MAX_GAP, turn_gap) if turn_gap and turn_gap > 0 else MERGE_MAX_GAP
    )

    # Fallback for unpunctuated ASR output (the known large-v3 failure mode):
    # with NO terminal punctuation at all there are no sentences to find, so
    # speech pauses become the sentence boundaries instead.
    #
    # The trigger is `terminals == 0`, not a ratio. A ratio silently changes the
    # spotting of ordinary transcripts — one long unpunctuated stretch inside an
    # otherwise well-punctuated file would drag the whole file onto the pause
    # heuristic. Zero terminals is the only state in which the punctuation-based
    # splitter is guaranteed to produce nothing at all, and it is exactly the
    # state large-v3 lands in. A transcript with even one "." keeps the normal
    # path untouched.
    terminals = sum(1 for w in clean if _TERMINAL_RE.search(w["w"]))
    sparse_punctuation = len(clean) >= 8 and terminals == 0
    if sparse_punctuation and asr_primed:
        # The fallback is now the SECOND line of defence: ASR punctuation priming
        # (transcription_service.ASR_PUNCTUATION_PRIMER) removed the attractor on every
        # clip it was measured against, so this branch means priming did not hold on
        # this one. Kept — a safety net you only find out about when it catches you is
        # working as designed — but it is a WARNING now, not routine information.
        logger.warning(
            "spotting: no terminal punctuation in %d words DESPITE ASR punctuation "
            "priming — the large-v3 attractor survived the primer on this clip; "
            "falling back to pause-based sentence boundaries",
            len(clean),
        )
    elif sparse_punctuation:
        logger.info(
            "spotting: no terminal punctuation in %d words — "
            "pause-based sentence fallback engaged (ASR was not punctuation-primed)",
            len(clean),
        )

    # 1. sentences
    #
    # The terminal is only believed when the word carrying it lasted long enough to have
    # been said. A `.`/`?`/`!` on a sub-50ms word is the signature of the primed decoder
    # inventing a boundary — the corpus holds ' record.', ' boyfriend?' and ' them.' at
    # exactly 0.000s — and every one of those boundaries starts the next cue in the
    # middle of an utterance, which is the first domino of the fabrication cascade. The
    # guard costs nothing: the group simply stays whole and step 2 splits it on length if
    # it needs splitting at all.
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    refused_splits = 0
    for i, word in enumerate(clean):
        current.append(word)
        boundary = bool(_TERMINAL_RE.search(word["w"]))
        if boundary and _word_dur(word) < DEGENERATE_WORD_DUR:
            boundary = False
            refused_splits += 1
        if not boundary and sparse_punctuation and i + 1 < len(clean):
            boundary = clean[i + 1]["s"] - word["e"] >= PAUSE_SPLIT_GAP
        if boundary:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    if refused_splits:
        logger.info(
            "spotting: refused %d sentence split(s) on a terminal carried by a word "
            "shorter than %.0fms — punctuation with no speech under it",
            refused_splits,
            DEGENERATE_WORD_DUR * 1000,
        )

    # 1b. speaker turns, then 2. split over-long sentences
    groups = [
        part
        for s in sentences
        for turn in _split_on_turn_gaps(s, turn_gap)
        for part in _split_long_group(turn, max_chars)
    ]

    # 3. merge short fragments into the previous cue when it is safe to do so
    merged: list[list[dict[str, Any]]] = []
    for group in groups:
        if (
            merged
            and _span_of(group) < min_dur
            and _can_merge(
                merged[-1],
                group,
                max_chars=max_chars,
                max_dur=max_dur,
                max_gap=merge_gap,
            )
        ):
            merged[-1] = merged[-1] + group
            continue
        merged.append(group)

    # 3b. degenerate-input repair. Two groups that begin at (effectively) the same
    # instant cannot both be shown, and step 4's anti-overlap clamp would resolve
    # that by pushing the first cue's end back onto its own start — a zero-length
    # event. Fold them into one cue instead: deterministic, and no text is lost.
    # Only broken word timestamps (all-identical, or an ASR hiccup) get here.
    collapsed: list[list[dict[str, Any]]] = []
    for group in merged:
        if collapsed and (group[0]["s"] - collapsed[-1][0]["s"]) < MIN_VISIBLE_DUR:
            logger.warning(
                "spotting: cues at the same instant (%.3fs) — folding %r into the "
                "previous cue; the word timestamps are degenerate",
                group[0]["s"],
                _text_of(group)[:40],
            )
            collapsed[-1] = collapsed[-1] + group
            continue
        collapsed.append(group)

    # 3c. minimum-duration FLOOR (not an aspiration).
    #
    # Step 4 grows each cue into the dead air ahead of it, but when the next cue starts
    # less than `min_dur` after this one begins there is no dead air to grow into, and
    # the anti-overlap clamp wins: cues of 0.72s, 0.96s and 1.06s shipped from a real
    # job against a declared 1.2s floor. `min_dur` was a wish, not a guarantee.
    #
    # A cue that cannot be given its floor is NOT shortened to fit — it is merged into
    # the neighbour crowding it, under exactly the constraints step 3 uses
    # (:func:`_can_merge`), so the two steps can never disagree about what a legal merge
    # is. Only when the merge is genuinely impossible — almost always the character
    # budget — does a short cue survive, and then it is logged rather than shipped
    # silently. That residue is the honest trade: text that cannot fit one cue is worse
    # than a cue held slightly too briefly.
    floored: list[list[dict[str, Any]]] = []
    for group in collapsed:
        if floored:
            previous = floored[-1]
            room = group[0]["s"] - previous[0]["s"]
            if room < min_dur:
                if _can_merge(
                    previous,
                    group,
                    max_chars=max_chars,
                    max_dur=max_dur,
                    max_gap=merge_gap,
                ):
                    floored[-1] = previous + group
                    continue
                logger.warning(
                    "spotting: %r can only be shown for %.2fs (floor %.2fs) and cannot "
                    "be merged into the next cue — shipping the compromise",
                    _text_of(previous)[:40],
                    room,
                    min_dur,
                )
        floored.append(group)

    cues = [
        {
            "start": g[0]["s"],
            "end": max(g[-1]["e"], g[0]["s"]),
            "text": _text_of(g),
            "word_stats": _word_stats(g),
        }
        for g in floored
    ]

    # 4. lead-out / anti-overlap
    for i, cue in enumerate(cues):
        next_start = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if next_start is None:
            limit = cue["end"] + lead_out_max  # last cue: a plain lead-out
        else:
            limit = min(next_start - min_gap, cue["end"] + lead_out_max)
        if limit < cue["end"]:
            cue["end"] = max(limit, cue["start"] + min_dur)  # shrink, but stay readable
        else:
            cue["end"] = min(limit, cue["start"] + max_dur)  # extend into dead air
        # The floor applies to BOTH branches. The extend branch stops at
        # `next_start - min_gap`, which is itself under the floor whenever two cues sit
        # between `min_dur` and `min_dur + min_gap` apart — a narrow window, but the
        # source of every sub-floor cue that step 3c does not already catch. When the two
        # constraints collide the gap yields, not the floor: a cue too short to read is
        # a defect, a gap 80ms narrower than ideal is not.
        if cue["end"] < cue["start"] + min_dur:
            cue["end"] = cue["start"] + min_dur
            if next_start is not None:
                cue["end"] = min(cue["end"], next_start)
        if next_start is not None and cue["end"] > next_start:
            cue["end"] = next_start  # hard guarantee: never overlap
        # Step 3b guarantees next_start - start >= MIN_VISIBLE_DUR, so the clamp
        # above can never land on or before the start. Kept as a belt-and-braces
        # assertion of the contract rather than as live arithmetic.
        cue["end"] = max(cue["end"], cue["start"] + MIN_VISIBLE_DUR)
        cue["start"] = round(cue["start"], 3)
        cue["end"] = round(cue["end"], 3)

    # 5. clamp to the end of the picture. Applied last, after every rule that could
    # have grown a cue: the lead-out on the FINAL cue is the usual offender (it has no
    # successor to bound it) and shipped ends 0.5-0.9s past EOF on real jobs.
    if video_duration and video_duration > 0:
        eof = round(float(video_duration), 3)
        clamped = 0
        for cue in cues:
            if cue["end"] > eof:
                cue["end"] = max(eof, cue["start"] + MIN_VISIBLE_DUR)
                clamped += 1
        if clamped:
            logger.info(
                "spotting: clamped %d cue end(s) to the video duration (%.3fs)",
                clamped,
                eof,
            )

    logger.debug(
        "subtitle_engine: %d words -> %d sentences -> %d cues",
        len(clean),
        len(sentences),
        len(cues),
    )
    return cues


# =============================================================================
# hallucination gate
# =============================================================================
def _normalized_text(text: str) -> str:
    """Case- and punctuation-insensitive form, for comparing two cues' wording."""
    return " ".join(
        "".join(
            ch for ch in str(text or "").lower() if ch.isalnum() or ch.isspace()
        ).split()
    )


def drop_hallucinated_cues(
    cues: list[dict[str, Any]],
    *,
    max_cps: float = MAX_SOURCE_CPS,
    impossible_cps: float = IMPOSSIBLE_SOURCE_CPS,
    key: str = "text",
    energy_probe: Any = None,
    energy_veto_db: float = ENERGY_VETO_DB,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split source cues into (kept, dropped-as-probable-ASR-hallucination).

    The defect this closes
    ----------------------
    Whisper invents text. Not often, but it does, and a fabrication rendered in fluent
    Hebrew is strictly worse than one rendered in English: it is invisible to the person
    who could have caught it.

    Why this is a SCORED gate and not a threshold
    ---------------------------------------------
    Because the threshold was measured against this project's own 20-run corpus archive
    and it was wrong in BOTH directions:

    * it deleted 8 real source cues and 0 fabrications. Two were genuine fast standup
      delivery (39.1 and 36.4 CPS); six were genuine CROSS-TALK, where Whisper compresses
      the word timestamps of overlapping speakers and the measured rate of real speech
      inflates past any threshold;
    * it missed every fabrication the corpus actually contains, because Whisper's
      inventions are not always fast. The verified one in ``clean_speech`` ("You've got
      how.", first word 0.02s long) measured 30 CPS inside a cue measuring 14.7.

    So a single number cannot do this job. What separates the two populations is the
    AGREEMENT of independent signals:

    ``cps_impossible`` (HARD)
        Above :data:`IMPOSSIBLE_SOURCE_CPS` — past anything real ever measured here, by
        a wide margin, and where the documented Whisper fabrications live.
    ``duplicate_of_previous`` (HARD)
        Word-for-word repetition of the cue before it: the loop artifact, which puts the
        same line on screen twice. The archive holds one at -14.9 dB below its own file's
        speech level.
    ``cps_fast`` (SOFT)
        Above :data:`MAX_SOURCE_CPS`. Suspicious, never sufficient.
    ``degenerate_words`` (SOFT)
        A run of at least :data:`DEGENERATE_WORD_RUN` consecutive sub-50ms word
        timestamps (from ``word_stats``, attached by :func:`words_to_cues`): Whisper
        stamping a clause onto one instant. Genuine fast speech in the corpus peaks at a
        run of 1; the fabricated blocks ran 5 and 6.

    A cue is dropped on ANY hard signal, or on TWO soft ones — unless the audio vetoes it.

    The energy veto
    ---------------
    Speech that Whisper merely mis-TIMED is still speech and still makes noise. A cue
    whose audio sits at the file's own speech level is therefore kept whatever the
    timing says, which is what rescues cross-talk. Measured: the six wrongly-dropped
    cross-talk cues sit at -0.1 to +0.6 dB relative to their file; a verified loop
    repetition sits at -14.9. The probe is OPTIONAL — with none supplied there is no veto.

    Args:
        cues: cue dicts with ``start``, ``end``, ``key`` and optionally ``word_stats``.
        max_cps: soft characters-per-second signal; see :data:`MAX_SOURCE_CPS`.
        impossible_cps: hard characters-per-second signal; see
            :data:`IMPOSSIBLE_SOURCE_CPS`.
        key: which field carries the source text.
        energy_probe: optional callable ``(start, end) -> float | None`` returning the
            window's mean audio level in dB RELATIVE to the whole file's mean (0 = the
            file's average level, negative = quieter). Duck-typed on purpose: this module
            owns no ffmpeg and reads no files. Any exception it raises is swallowed and
            treated as "no measurement".
        energy_veto_db: level at or above which a soft-signal drop is vetoed.

    Returns:
        ``(kept, dropped)``. ``kept`` holds the ORIGINAL dicts, unmodified, in input
        order. Each dropped cue is a COPY annotated with ``cps``, ``signals`` (the list
        that fired), ``energy_db`` (when measured), ``drop_reason`` and ``context_only``
        — the last of which tells the caller whether the text is still worth handing the
        translator as scene context (true for everything except a duplicate, whose text
        is by definition still present in the cue that was kept).

        Cues with a non-positive duration are KEPT: a zero-length cue has infinite CPS by
        arithmetic alone, which says nothing about whether its text is real, and the
        spotting stage already has rules for those.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    previous_norm = ""

    for cue in cues or []:
        text = str(cue.get(key) or "").strip()
        try:
            start = float(cue.get("start", 0) or 0)
            duration = float(cue.get("end", 0) or 0) - start
        except (TypeError, ValueError):
            start, duration = 0.0, 0.0
        if not text or duration <= 0:
            kept.append(cue)
            previous_norm = _normalized_text(text)
            continue

        cps = len(text) / duration
        normalized = _normalized_text(text)
        stats = cue.get("word_stats") or {}
        try:
            degenerate_run = int(stats.get("degenerate_run", 0) or 0)
        except (TypeError, ValueError):
            degenerate_run = 0

        hard: list[str] = []
        soft: list[str] = []
        if cps > impossible_cps:
            hard.append(f"cps_impossible({cps:.1f}>{impossible_cps:.0f})")
        if normalized and normalized == previous_norm:
            hard.append("duplicate_of_previous")
        if cps > max_cps:
            soft.append(f"cps_fast({cps:.1f}>{max_cps:.0f})")
        if degenerate_run >= DEGENERATE_WORD_RUN:
            soft.append(f"degenerate_words(run={degenerate_run})")

        verdict = bool(hard) or len(soft) >= 2
        energy_db = None
        if verdict and not hard and energy_probe is not None:
            try:
                measured = energy_probe(start, start + duration)
            except Exception as exc:  # noqa: BLE001 - a probe may never fail a job
                logger.warning(
                    "hallucination gate: energy probe raised %s — deciding without it",
                    exc,
                )
                measured = None
            if measured is not None:
                energy_db = round(float(measured), 2)
                if energy_db >= energy_veto_db:
                    verdict = False
                    logger.info(
                        "hallucination gate: KEEPING cue at %.2fs despite %s — its audio "
                        "is %.1f dB relative to the file, i.e. someone is speaking: %r",
                        start,
                        "+".join(soft),
                        energy_db,
                        text[:80],
                    )

        if not verdict:
            kept.append(cue)
            previous_norm = normalized
            continue

        record = dict(cue)
        record["cps"] = round(cps, 2)
        record["signals"] = hard + soft
        record["energy_db"] = energy_db
        record["context_only"] = "duplicate_of_previous" not in hard
        record["drop_reason"] = (
            "hard signal: " if hard else "corroborated signals: "
        ) + ", ".join(hard + soft)
        dropped.append(record)
        logger.warning(
            "hallucination gate: dropping cue at %.2f-%.2fs (%d chars in %.2fs, %.1f "
            "CPS) — %s: %r",
            start,
            start + duration,
            len(text),
            duration,
            cps,
            record["drop_reason"],
            text[:120],
        )
        # A dropped cue is NOT remembered as the previous text: two consecutive copies of
        # a hallucination must both go, not just the second.

    if dropped:
        logger.warning(
            "hallucination gate: dropped %d of %d source cues as probable ASR "
            "hallucination",
            len(dropped),
            len(cues or []),
        )
    return kept, dropped


def absorb_dropped_time(
    cues: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    *,
    max_dur: float = MAX_CUE_DUR,
    min_gap: float = MIN_CUE_GAP,
) -> list[dict[str, Any]]:
    """Let the cue before a dropped one hold the screen over the time it freed.

    Dropping a cue leaves a HOLE: the picture runs on with no subtitle at all, which
    reads as the pipeline having missed something (it did — deliberately, but the viewer
    cannot know that). Extending the previous cue over the freed span costs nothing, is
    the standard editorial answer, and keeps the last thing that WAS said on screen while
    the questionable moment passes.

    Bounded exactly as :func:`words_to_cues`'s lead-out is: never past ``max_dur`` from
    the cue's own start, never within ``min_gap`` of the next surviving cue, and never
    past the end of the freed span itself.

    Args:
        cues: the SURVIVING cues, in time order. Not mutated; copies are returned.
        dropped: the cues removed, in time order.

    Returns:
        A new list of cue copies, same length and same text.
    """
    out = [dict(cue) for cue in (cues or [])]
    if not out or not dropped:
        return out

    extended = 0
    for gone in dropped:
        try:
            hole_start = float(gone.get("start", 0) or 0)
            hole_end = float(gone.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if hole_end <= hole_start:
            continue
        # The last surviving cue that starts before the hole does.
        index = None
        for i, cue in enumerate(out):
            if float(cue.get("start", 0) or 0) <= hole_start:
                index = i
            else:
                break
        if index is None:
            continue
        cue = out[index]
        start = float(cue.get("start", 0) or 0)
        end = float(cue.get("end", 0) or 0)
        limit = min(hole_end, start + max_dur)
        if index + 1 < len(out):
            limit = min(limit, float(out[index + 1].get("start", 0) or 0) - min_gap)
        if limit > end:
            cue["end"] = round(limit, 3)
            extended += 1

    if extended:
        logger.info(
            "hallucination gate: extended %d cue(s) over the time %d dropped cue(s) "
            "freed, rather than leaving the picture blank",
            extended,
            len(dropped),
        )
    return out


# =============================================================================
# line wrapping
# =============================================================================
def _hard_wrap(text: str, max_line: int) -> list[str]:
    """Last-resort wrap that is *guaranteed* to emit no line longer than ``max_line``.

    Breaks at the last space inside the budget when there is one, and mid-token when
    there is not (a 100-character URL has no other option). May return more than two
    lines: an over-long line is drawn straight off the edge of the frame by libass —
    ``WrapStyle: 2`` disables its own wrapping — so a third line is strictly less bad
    than losing text off-screen. Upstream (``translation_v2`` and the spotting pass)
    budgets cues at :func:`layout_params`' ``max_chars_per_cue`` for this exact frame,
    so in practice only pathological input reaches this.
    """
    lines: list[str] = []
    rest = text
    while len(rest) > max_line:
        cut = rest.rfind(" ", 0, max_line + 1)
        if cut <= 0:
            cut = max_line
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        lines.append(rest)
    return lines or [""]


def wrap_two_lines(text: str, max_line: int = MAX_LINE_CHARS) -> list[str]:
    """Wrap a cue into one line, or the best two-line split.

    A single line is always preferred. Otherwise every word boundary that leaves
    both halves within ``max_line`` is scored and the cheapest wins:

      * length imbalance (lower is better),
      * a small penalty when the top line is the longer one — subtitles read
        better bottom-heavy,
      * a strong bonus for breaking after ``.``/``?``/``!``,
      * a smaller bonus for breaking after a comma.

    When no word boundary can satisfy ``max_line`` — one over-long token, or simply
    more text than two lines can hold — it falls back to :func:`_hard_wrap`, which
    never returns an over-long line.

    Postcondition (relied on by :func:`build_ass`): ``max(len(l) for l in result)
    <= max_line`` for every input.
    """
    text = " ".join((text or "").split())
    if len(text) <= max_line:
        return [text]

    words = text.split(" ")
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(words)):
        top = " ".join(words[:i])
        bottom = " ".join(words[i:])
        if len(top) > max_line or len(bottom) > max_line:
            continue
        score = float(abs(len(top) - len(bottom)))
        if len(top) > len(bottom):
            score += _PENALTY_TOP_HEAVY
        stripped = top.rstrip()
        if stripped.endswith((".", "?", "!")):
            score -= _BONUS_SENTENCE_BREAK
        elif stripped.endswith(","):
            score -= _BONUS_CLAUSE_BREAK
        if best is None or score < best[0]:
            best = (score, [top, bottom])

    if best is not None:
        return best[1]

    lines = _hard_wrap(text, max_line)
    logger.warning(
        "subtitle_engine: no two-line split fits %d chars for a %d-char cue — "
        "hard-wrapped into %d lines",
        max_line,
        len(text),
        len(lines),
    )
    return lines


# =============================================================================
# post-translation reflow
# =============================================================================
#: Single-letter Hebrew prefixes/conjunctions. They are *bound* morphemes — they
#: are written attached to the next word (ו + הלך = והלך), so one standing alone at
#: the end of a cue is always a cue boundary that fell in the wrong place.
DANGLING_PREFIXES = frozenset("ובלמשכה")

#: Longest silence (seconds) a dangling connector may be moved across. Past this the
#: two cues are not one phrase and moving the letter would put it under the wrong
#: moment of speech.
REFLOW_MAX_GAP = 1.5

#: A cue that would grow past this many characters is left alone: 2 lines x 42 is the
#: Netflix budget the rest of this module is built around.
REFLOW_MAX_CHARS = MAX_LINE_CHARS * MAX_LINES

_DANGLING_RE = re.compile(f"(?:^|\\s)([{''.join(DANGLING_PREFIXES)}])-?$")


def reflow_dangling_connectors(
    cues: list[dict[str, Any]],
    *,
    key: str = "translated_text",
    max_gap: float = REFLOW_MAX_GAP,
    max_chars: int = REFLOW_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Move a stranded single-letter Hebrew connector onto the next cue.

    Translation is done per cue, so a cue can end on a bound prefix whose word lives
    in the next cue — real output from this pipeline: ``"...לנחות ו"`` / ``"...שמכירה
    ב"``. On screen that is a lone letter hanging off the end of a line, and the word
    it belongs to appears a second later without it.

    This is a **pure text pass**: timings are never touched, so it cannot desync a
    cue from its audio. It is deliberately conservative — at most ONE token moves,
    never across a silence longer than ``max_gap``, and never into a cue that would
    end up longer than ``max_chars``.

    Args:
        cues: pipeline cue dicts. Not mutated; copies are returned.
        key: which field to reflow (``translated_text`` for the v2 path).

    Returns:
        A new list of cue copies, same length and same timings.
    """
    out = [dict(cue) for cue in (cues or [])]
    moved = 0
    for index in range(len(out) - 1):
        current, following = out[index], out[index + 1]
        text = str(current.get(key) or "").rstrip()
        next_text = str(following.get(key) or "").lstrip()
        if not text or not next_text:
            continue

        match = _DANGLING_RE.search(text)
        if not match:
            continue

        try:
            gap = float(following.get("start", 0)) - float(current.get("end", 0))
        except (TypeError, ValueError):
            continue
        if gap > max_gap:
            continue

        token = text[match.start(1) :]  # the letter plus an optional trailing "-"
        remainder = text[: match.start(1)].rstrip()
        if not remainder:
            continue  # the whole cue is the connector; moving it would empty the cue

        # Hebrew prefixes are written attached ("ו" + "המטוס" = "והמטוס"), except
        # before a Latin word or a numeral, where the convention is a maqaf:
        # "ב-2026", "ה-iPhone".
        if not token.endswith("-") and unicodedata.bidirectional(next_text[0]) in (
            "L",
            "EN",
            "AN",
        ):
            token += "-"
        joined = token + next_text
        if len(joined) > max_chars:
            continue

        current[key] = remainder
        following[key] = joined
        moved += 1

    if moved:
        logger.info("reflow: moved %d dangling connector(s) to the next cue", moved)
    return out


# =============================================================================
# Hebrew typography + bidi
# =============================================================================
def quoted_phrases(text: str) -> str:
    """Give a quoted Hebrew phrase the SAME mark at both ends (U+05F4 ״ … ״).

    Runs before :func:`gershayim`, because the acronym rule cannot see pairs: it
    converts an opening quote that happens to sit between two Hebrew letters
    (``ב"פחד``) and leaves the closing one alone when a ``?`` or ``.`` follows it
    (``גורם"?``), which ships a cue that opens ״ and closes ". See
    :data:`_QUOTED_PHRASE_RE` for exactly which shapes count as a quotation.

    English quotations inside a Hebrew line are untouched — the quoted text has to
    contain a Hebrew letter to match at all.
    """
    if not text:
        return text
    return _QUOTED_PHRASE_RE.sub(
        lambda m: f"{m.group(1)}{GERSHAYIM}{m.group('body')}{GERSHAYIM}", text
    )


def gershayim(text: str) -> str:
    """Replace an ASCII ``"`` used as a Hebrew acronym mark with U+05F4 (״).

    Only a quote sitting *between two Hebrew letters* is converted, so English
    quotations (``he said "hi"``) are left untouched. Quoted Hebrew PHRASES are handled
    one step earlier, by :func:`quoted_phrases`, so what reaches here is an acronym mark
    or an unpaired quote.
    """
    if not text:
        return text
    return _GERSHAYIM_RE.sub(GERSHAYIM, text)


def geresh(text: str) -> str:
    """Replace an apostrophe carrying a foreign phoneme in Hebrew with U+05F3 (׳).

    The sibling of :func:`gershayim`, and the answer to a real regression: the legacy
    path produced ג׳ורג׳ (via ``utils.rtl_utils.fix_hebrew_quotes``) while the v2 path
    shipped ג'ורג' with an ASCII apostrophe, which is simply wrong Hebrew typography and
    renders as a straight tick rather than a mark.

    Fires only after ג, ז, צ or ץ — the letters that take a geresh in modern Hebrew — so
    English apostrophes (``don't``, ``Charles'``) and quotes around Hebrew words ending
    in any other letter are untouched. Both the ASCII ``'`` and the curly ``’`` convert.

    Unlike ``fix_hebrew_quotes``, which rewrote any ``'...'`` pair into ``׳...׳`` and so
    mangled English quotations inside a Hebrew line, this never looks at pairs at all.
    """
    if not text:
        return text
    return _GERESH_RE.sub(GERESH, text)


def hebrew_typography(text: str) -> str:
    """Apply every Hebrew punctuation repair the renderer owes the text.

    One entry point so a caller cannot apply half of them, which is exactly how the
    geresh came to be missing while the gershayim was applied.

    Order matters: pairs before singles. :func:`quoted_phrases` has to claim a quotation
    before :func:`gershayim` converts half of it into an acronym mark.
    """
    return geresh(gershayim(quoted_phrases(text)))


def _ltr_runs(line: str) -> list[tuple[int, int]]:
    """Maximal ``[start, end]`` index pairs of stretches to keep left-to-right.

    A run is grown between *anchors* — characters the bidi algorithm calls strongly
    left-to-right or numeric (``L``/``EN``/``AN``: Latin letters, digits) — and is
    allowed to swallow anything between two anchors that is **not** strongly
    right-to-left. That is what keeps ``Microsoft Azure``, ``COVID-19``, ``3.5``,
    ``AT&T``, ``Boeing 737`` and ``2020 COVID-19`` intact as single left-to-right
    units. Adjacent numeric terminators (``ET``: ``$ € ₪ % ‰ °``) are absorbed at
    both edges so ``50%`` and ``$25`` stay glued to their number.

    The membership test is Unicode's own ``bidirectional`` category, deliberately
    not a hand-written character class: the class that preceded it forgot ``%``,
    ``$`` and ``&``, and rendered ``50%`` as ``%50``.
    """
    cats = [unicodedata.bidirectional(ch) for ch in line]
    anchors = [i for i, cat in enumerate(cats) if cat in _BIDI_LTR_ANCHOR]
    if not anchors:
        return []

    runs: list[list[int]] = [[anchors[0], anchors[0]]]
    for index in anchors[1:]:
        between = cats[runs[-1][1] + 1 : index]
        if any(cat in _BIDI_STRONG_RTL for cat in between):
            runs.append([index, index])  # an RTL word closes the run
        else:
            runs[-1][1] = index

    for run in runs:
        while run[0] > 0 and cats[run[0] - 1] == _BIDI_NUMERIC_TERMINATOR:
            run[0] -= 1
        while run[1] + 1 < len(cats) and cats[run[1] + 1] == _BIDI_NUMERIC_TERMINATOR:
            run[1] += 1
    return [(start, end) for start, end in runs]


def bidi_isolate(line: str) -> str:
    """Prepare one rendered line for libass's bidi pass.

    Two facts, both established by rendering in this project's container and pinned
    by ``tests/integration/test_bidi_render.py``:

    1. **libass hard-defaults the paragraph direction to LTR** — it does not infer
       it from the first strong character. So an untouched Hebrew line comes out
       with its word order reversed. ``RLI ... PDI`` around the whole line is what
       tells libass the paragraph is right-to-left, and it is not optional: without
       it ``ב-ICC ... ב-2026`` renders reversed.
    2. **Latin/numeric stretches must be isolated as MAXIMAL runs, not per word.**
       Isolating each word separately makes N sibling isolates, which the bidi
       algorithm then lays out right-to-left *relative to each other* — the exact
       source of ``Microsoft Azure`` -> ``Azure Microsoft``, ``3.5`` -> ``5.3`` and
       ``COVID-19`` -> ``19-COVID``.

    A line with no strongly right-to-left character is returned untouched: libass's
    native LTR base is already correct for it, and wrapping it in RLI would kick its
    sentence-final period over to the left-hand edge.

    Never emits RLO (U+202E) — an override would fight the bidi algorithm rather
    than inform it.
    """
    if not line:
        return line
    if not any(unicodedata.bidirectional(ch) in _BIDI_STRONG_RTL for ch in line):
        return line

    pieces: list[str] = []
    cursor = 0
    for start, end in _ltr_runs(line):
        pieces.append(line[cursor:start])
        pieces.append(f"{LRI}{line[start:end + 1]}{PDI}")
        cursor = end + 1
    pieces.append(line[cursor:])
    return f"{RLI}{''.join(pieces)}{PDI}"


# =============================================================================
# ASS output
# =============================================================================
def _ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp: ``H:MM:SS.CC`` (centiseconds)."""
    centis = max(0, int(round(float(seconds) * 100)))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_safe(text: str) -> str:
    """Neutralize characters that would be parsed as ASS markup."""
    text = " ".join((text or "").split())  # also flattens stray newlines
    return text.replace("{", "(").replace("}", ")")


def build_ass(
    cues: list[dict[str, Any]],
    *,
    video_w: int,
    video_h: int,
    font_frac: float = DEFAULT_FONT_FRAC,
    margin_v_frac: float = DEFAULT_MARGIN_V_FRAC,
    rtl: bool = True,
    layout: dict[str, Any] = None,
    position: str = "bottom",
) -> str:
    """Render cues as an ASS v4+ subtitle script.

    The style is the render-tested one: Noto Sans Hebrew, bold, white on a
    semi-transparent opaque box (``BorderStyle=4``, no drop shadow). Placement is
    bottom-centre by default, top-centre for ``top`` and middle-right for ``side``.
    ``PlayResX/Y`` match the video, so ASS units are pixels.

    Every size comes from :func:`layout_params`, which reads BOTH dimensions — the
    font is a fraction of the height only for as long as the width can carry a full
    line at that size. On 16:9 that is always, so landscape output is unchanged; on
    9:16 the font shrinks to its legibility floor and the line budget shrinks with it.

    Args:
        cues: ``[{"start","end","text"}, ...]`` as produced by :func:`words_to_cues`.
        video_w: video width in pixels.
        video_h: video height in pixels.
        font_frac: font size as a fraction of ``video_h`` (before the width check).
        margin_v_frac: bottom margin as a fraction of ``video_h``.
        rtl: apply :func:`bidi_isolate` to every line (Hebrew/Arabic targets).
        layout: a :func:`layout_params` dict to use instead of deriving one. Pass the
            SAME dict the upstream stages budgeted against — the caller that wrote the
            42-character-wide text and the renderer that has to fit it must not be
            allowed to disagree. Omitted, it is derived here from ``video_w/video_h``,
            which gives the identical answer for the identical inputs.
        position: ``bottom``, ``top`` or ``side``. Unknown values retain the historical
            bottom-centre placement.

    Returns:
        The complete ``.ass`` file content. Render it with FFmpeg's ``ass``
        filter (``ass=file.ass:shaping=complex``) — the ``subtitles`` filter has
        no ``shaping`` option and will hard-fail.
    """
    if layout is None:
        layout = layout_params(
            video_w, video_h, font_frac=font_frac, margin_v_frac=margin_v_frac
        )
    font_size = layout["font_px"]
    margin_v = layout["margin_v"]
    margin_h = layout["margin_h"]
    max_line = layout["max_line_chars"]
    alignment = ASS_ALIGNMENT.get(position, ASS_ALIGNMENT["bottom"])

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {int(video_w)}\n"
        f"PlayResY: {int(video_h)}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {STYLE_NAME},Noto Sans Hebrew,{font_size},"
        "&H00FFFFFF,&H00FFFFFF,&H00000000,&H14000000,"
        f"1,0,0,0,100,100,0,0,4,3,0,{alignment},"
        f"{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text\n"
    )

    events: list[str] = []
    for cue in cues or []:
        lines = wrap_two_lines(
            hebrew_typography(_ass_safe(cue.get("text", ""))), max_line
        )
        if rtl:
            lines = [bidi_isolate(line) for line in lines]
        body = "\\N".join(lines)
        events.append(
            f"Dialogue: 0,{_ass_timestamp(cue['start'])},"
            f"{_ass_timestamp(cue['end'])},{STYLE_NAME},,0,0,0,,{body}"
        )

    logger.debug(
        "subtitle_engine: built ASS with %d events at %dx%d "
        "(font %d, margin_h %d, margin_v %d, %d chars/line)",
        len(events),
        video_w,
        video_h,
        font_size,
        margin_h,
        margin_v,
        max_line,
    )
    return header + "\n".join(events) + "\n"
