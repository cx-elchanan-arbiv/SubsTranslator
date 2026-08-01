"""
Rendered proof of the bidi decisions in ``services/subtitle_engine.py``.

Everything :func:`services.subtitle_engine.bidi_isolate` claims is a claim about libass,
and a claim about rendering can only be settled by rendering. The unit tests pin the
*structure* of the control characters; this file pins the *pixels* they produce, through
the real ``ass`` filter, with the real font, in this project's own container
(FFmpeg 7.1.5 / libass 0.17.x).

Method
------
Each case is rendered five ways and compared against the first:

``reference``
    An **independently authored** layout: the tokens listed LEFT-TO-RIGHT in the order a
    Hebrew reader must see them, each wrapped in its own ``RLI...PDI`` so it keeps its
    natural internal direction while the sequence stays in the order written. It is
    written by hand from the meaning of the sentence rather than derived from the code
    under test, so agreeing with it is evidence and not a tautology.
``shipped``
    ``bidi_isolate(source)`` — what production emits today.
``no-controls``
    The raw string: the defect this module exists to remove.
``historical``
    The implementation this rewrite replaced, copied verbatim from git — one regex,
    ``[A-Za-z][A-Za-z0-9]*|\\d+``, isolating each Latin word and each digit group alone.
``naive-maximal``
    The shortcut considered and rejected during the rewrite: one regex sweep for
    maximal-looking Latin/digit stretches, plus an outer RLI.

Comparison metric
-----------------
Two renders are compared band by band, where a *band* is a contiguous run of inked
columns (a word, roughly). Bands must line up in count and width, and then each band's
**pixels** are differenced against its counterpart with a small alignment search.

Both halves are necessary, and the reason is worth recording. Band widths alone catch
words swapping places but are blind to reordering *inside* a band: ``3.5`` and ``5.3``
are the same three glyphs and therefore exactly the same width. Whole-image differencing
alone is useless in the other direction — one pixel of extra inter-word spacing shifts
every later glyph and swamps the signal. Per-band pixels, aligned per band, see both.
Measured separation is wide: every correct render scores under 1.4, every broken one
over 8.7.

What the renders establish
--------------------------
* **libass hard-defaults the paragraph direction to LTR.** It does not infer it from the
  first strong character. Proven byte-exactly in
  :meth:`TestParagraphDirection.test_libass_does_not_auto_detect_the_base_direction`.
  This is the fact the module rests on: with no outer RLI, every mixed Hebrew line comes
  out with its word order wrong.
* **The no-controls render is not a usable reference.** It reverses six of the ten cases
  and is right on the other four only because those cannot tell the two orders apart.
* **The historical per-word implementation breaks eight of ten**, and breaks them in
  exactly the ways ``bidi_isolate``'s docstring records: ``Microsoft Azure`` ->
  ``Azure Microsoft``, ``3.5`` -> ``5.3``, ``COVID-19`` -> ``19-COVID``, and ``50%``,
  ``$25``, ``AT&T`` all split from their symbols.
* **The naive maximal regex breaks three of ten** — ``50%``, ``$25`` and ``AT&T``. It
  hard-codes a character class where Unicode has categories, so ``%``, ``$`` and ``&``
  fall outside its runs. It does *not* break ``התפרצות COVID-19 קשה``, which had been
  the stated reason for rejecting it; the reason is real, the case cited for it was not.

Skips (never fails) when ffmpeg or Noto Sans Hebrew is unavailable, so the suite still
runs outside the container.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isfile(os.path.join(backend_dir, "services", "subtitle_engine.py")):
    # docker-compose mounts ./tests over /app/tests, so this file runs as
    # /app/tests/integration/... with /app as the backend root.
    backend_dir = "/app"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.subtitle_engine import (  # noqa: E402
    LRI,
    PDI,
    RLI,
    bidi_isolate,
    gershayim,
)

W, H = 1000, 90
RLO = "‮"  # RIGHT-TO-LEFT OVERRIDE — must never be emitted

#: Worst per-band mean-absolute-difference at which two renders are still "the same
#: layout". Correct renders measure <= 1.4 here and broken ones >= 8.7, so the exact
#: value is not delicate.
SAME_LAYOUT_MAD = 2.0

FONT_DIRS = ("/usr/share/fonts/truetype/hebrew", "/usr/share/fonts")

#: Where the rendered evidence is written. Set ``BIDI_RENDER_OUT`` to keep it.
OUT_DIR = os.environ.get("BIDI_RENDER_OUT") or tempfile.mkdtemp(prefix="bidi_render_")

_ASS_HEAD = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: He,Noto Sans Hebrew,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""


def _have_renderer():
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not installed"
    for directory in FONT_DIRS:
        if not os.path.isdir(directory):
            continue
        for _root, _dirs, files in os.walk(directory):
            if any("NotoSansHebrew" in name for name in files):
                return True, ""
    return False, "Noto Sans Hebrew not installed"


_RENDERER_OK, _RENDERER_WHY = _have_renderer()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _RENDERER_OK, reason=_RENDERER_WHY),
]


# ----------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------
def render(tag, text):
    """Render one line of ASS text onto black. Returns the PNG path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ass_path = os.path.join(OUT_DIR, f"{tag}.ass")
    png_path = os.path.join(OUT_DIR, f"{tag}.png")
    with open(ass_path, "w", encoding="utf-8") as handle:
        handle.write(_ASS_HEAD + f"Dialogue: 0,0:00:00.00,0:00:01.00,He,,0,0,0,,{text}\n")
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d=1",
            "-frames:v", "1",
            # `ass`, not `subtitles`: the subtitles filter has no `shaping` option.
            "-vf", f"ass={ass_path}:shaping=complex",
            "-y", png_path,
        ],
        check=True,
        capture_output=True,
    )
    return png_path


def digest(png_path):
    with open(png_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def _gray(png_path):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", png_path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=True,
    ).stdout


def visual_reference(tokens):
    """Author a line in VISUAL order: tokens left to right, each in its own RLI.

    ``RLI``, not ``LRI``. A token may itself be mixed — ``ב-ICC`` is a Hebrew prefix on a
    Latin acronym — and inside an ``LRI`` its base direction would be left-to-right, which
    puts the ``ב`` on the *left*. Hebrew reads right to left, so the prefix belongs on the
    right. ``RLI`` gives each token the base direction Hebrew text actually has, while
    the sequence of isolates still lays out left to right because the paragraph around
    them has no strong character and so defaults to LTR.
    """
    return " ".join(RLI + token + PDI for token in tokens)


# ----------------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------------
def _band_spans(png_path, threshold=250, merge_gap=6):
    """``([(start, width), ...], grayscale_buffer)`` for the inked column runs."""
    buffer = _gray(png_path)
    columns = [sum(buffer[y * W + x] for y in range(H)) for x in range(W)]

    runs, start = [], None
    for x, value in enumerate(columns):
        if value > threshold:
            start = x if start is None else start
        elif start is not None:
            runs.append((start, x - start))
            start = None
    if start is not None:
        runs.append((start, W - start))

    merged = []
    for begin, width in runs:  # glue the gaps *inside* a word back together
        if merged and begin - (merged[-1][0] + merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], begin + width - merged[-1][0])
        else:
            merged.append((begin, width))
    return merged, buffer


def ink_bands(png_path):
    """Widths of the inked column runs, left to right — the coarse layout fingerprint."""
    spans, _buffer = _band_spans(png_path)
    return [width for _start, width in spans]


def layout_difference(png_a, png_b, shift=3):
    """Worst per-band pixel difference between two renders. 0 means identical layout.

    Returns :data:`MISMATCH` when the band *structure* differs (different number of words,
    or a word of a visibly different width), which is already proof of a different layout
    and leaves nothing to difference.
    """
    spans_a, buffer_a = _band_spans(png_a)
    spans_b, buffer_b = _band_spans(png_b)
    if len(spans_a) != len(spans_b):
        return MISMATCH

    worst = 0.0
    for (start_a, width_a), (start_b, width_b) in zip(spans_a, spans_b):
        if abs(width_a - width_b) > 2:
            return MISMATCH
        width = min(width_a, width_b)
        best = None
        for dx in range(-shift, shift + 1):
            total = 0
            for y in range(H):
                row_a = buffer_a[y * W:(y + 1) * W]
                row_b = buffer_b[y * W:(y + 1) * W]
                for x in range(width):
                    xa, xb = start_a + x, start_b + x + dx
                    if 0 <= xa < W and 0 <= xb < W:
                        total += abs(row_a[xa] - row_b[xb])
            score = total / (H * max(width, 1))
            best = score if best is None else min(best, score)
        worst = max(worst, best)
    return worst


MISMATCH = 999.0


def same_layout(png_a, png_b):
    return layout_difference(png_a, png_b) < SAME_LAYOUT_MAD


def total_ink(png_path):
    return sum(_gray(png_path))


# ----------------------------------------------------------------------------------
# the rejected alternatives, as executable records
# ----------------------------------------------------------------------------------
#: Verbatim from ``git show HEAD~:backend/services/subtitle_engine.py`` — the regex the
#: rewrite replaced.
_HISTORICAL_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_NAIVE_MAXIMAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .\-/:]*[A-Za-z0-9]")


def historical(line):
    """The previous implementation: each Latin word and each digit group isolated alone."""
    isolated = _HISTORICAL_RE.sub(lambda m: f"{LRI}{m.group(0)}{PDI}", line)
    return f"{RLI}{isolated}{PDI}"


def naive_maximal(line):
    """One regex sweep for maximal-looking Latin/digit stretches, plus an outer RLI."""
    return RLI + _NAIVE_MAXIMAL_RE.sub(lambda m: LRI + m.group(0) + PDI, line) + PDI


# ----------------------------------------------------------------------------------
# the cases
# ----------------------------------------------------------------------------------
#: ``(id, source, expected tokens LEFT-TO-RIGHT on screen)``
CASES = [
    (
        "a_icc",
        "ב-ICC זה יכול לסבך דברים ב-2026",
        ["ב-2026", "דברים", "לסבך", "יכול", "זה", "ב-ICC"],
    ),
    ("b_sym", "שלום Microsoft Azure שלום", ["שלום", "Microsoft Azure", "שלום"]),
    ("b_asym", "היום Microsoft Azure עלה", ["עלה", "Microsoft Azure", "היום"]),
    ("c_dec", "זה עלה 3.5 אחוז", ["אחוז", "3.5", "עלה", "זה"]),
    ("d_covid", "התפרצות COVID-19 קשה", ["קשה", "COVID-19", "התפרצות"]),
    ("e_cnn", gershayim('צה"ל אמר ש-CNN דיווח'), ["דיווח", "ש-CNN", "אמר", "צה״ל"]),
    ("pct", "זה עלה 50% השנה", ["השנה", "50%", "עלה", "זה"]),
    ("usd", "המחיר הוא $25 בלבד", ["בלבד", "$25", "הוא", "המחיר"]),
    ("att", gershayim('דו"ח AT&T פורסם'), ["פורסם", "AT&T", "דו״ח"]),
    (
        "prod",
        "הם קנו Microsoft Azure 2024 אתמול",
        ["אתמול", "Microsoft Azure 2024", "קנו", "הם"],
    ),
]
CASE_IDS = [case[0] for case in CASES]
ALL_CASES = set(CASE_IDS)

#: Measured, not assumed. Each set is pinned by a test below, so a change in libass or
#: the font shows up as a failure here rather than as a silent shift in what is proven.
NO_CONTROLS_BREAKS = {"a_icc", "b_asym", "d_covid", "e_cnn", "att", "prod"}
HISTORICAL_BREAKS = {"b_sym", "b_asym", "c_dec", "d_covid", "pct", "usd", "att", "prod"}
NAIVE_BREAKS = {"pct", "usd", "att"}


@pytest.fixture(scope="module")
def evidence():
    """Render every case five ways, once, and report where the PNGs landed."""
    rendered = {}
    for case_id, source, tokens in CASES:
        rendered[case_id] = {
            "reference": render(f"{case_id}__reference", visual_reference(tokens)),
            "shipped": render(f"{case_id}__shipped", bidi_isolate(source)),
            "none": render(f"{case_id}__no_controls", source),
            "historical": render(f"{case_id}__historical", historical(source)),
            "naive": render(f"{case_id}__naive_maximal", naive_maximal(source)),
        }
    print(f"\nbidi render evidence written to: {OUT_DIR}")
    return rendered


def _report(case_id, variant, evidence):
    difference = layout_difference(evidence[case_id][variant], evidence[case_id]["reference"])
    return (
        f"{case_id}/{variant}: per-band difference {difference:.2f} "
        f"(same-layout threshold {SAME_LAYOUT_MAD})\n"
        f"  {variant:10} bands {ink_bands(evidence[case_id][variant])}\n"
        f"  reference  bands {ink_bands(evidence[case_id]['reference'])}\n"
        f"  PNGs in {OUT_DIR}"
    )


# ----------------------------------------------------------------------------------
# 1. the claim the whole module rests on
# ----------------------------------------------------------------------------------
class TestParagraphDirection:
    """``bidi_isolate``'s docstring asserts libass hard-defaults the paragraph to LTR."""

    def test_libass_does_not_auto_detect_the_base_direction(self):
        """``אבג DEF`` with no controls: which side does the Hebrew land on?

        * base RTL (first-strong auto-detection) -> visual ``DEF אבג``, Hebrew on the RIGHT
        * base LTR (a hard default)              -> visual ``אבג DEF``, Hebrew on the LEFT

        Both answers are rendered as explicit references, so the comparison assumes
        nothing about libass. Byte-exact: it is the same glyphs in the same places or it
        is not.
        """
        plain = render("basedir__no_controls", "אבג DEF")
        as_ltr = render("basedir__ref_ltr_base", visual_reference(["אבג", "DEF"]))
        as_rtl = render("basedir__ref_rtl_base", visual_reference(["DEF", "אבג"]))

        assert digest(plain) != digest(as_rtl), (
            "libass auto-detected the base direction from the first strong character — "
            "the premise of bidi_isolate() is wrong and its docstring must be fixed"
        )
        assert digest(plain) == digest(as_ltr), (
            f"no-controls matched neither base direction; the probe itself is suspect.\n"
            f"  none {digest(plain)}  ltr {digest(as_ltr)}  rtl {digest(as_rtl)}\n"
            f"  PNGs in {OUT_DIR}"
        )

    def test_an_explicit_rli_flips_the_paragraph(self):
        """RLI...PDI is what buys the right-to-left paragraph, so it is not optional."""
        probe = "אבגדהוז DEF"
        isolated = render("basedir__rli", RLI + probe + PDI)
        plain = render("basedir__plain", probe)
        as_rtl = render("basedir__wide_rtl", visual_reference(["DEF", "אבגדהוז"]))
        as_ltr = render("basedir__wide_ltr", visual_reference(["אבגדהוז", "DEF"]))

        assert same_layout(isolated, as_rtl), "RLI did not produce the RTL layout"
        assert not same_layout(isolated, as_ltr), "RLI did not change the layout at all"
        # ...and the same line without it lands on the wrong one.
        assert same_layout(plain, as_ltr)
        assert not same_layout(plain, as_rtl)


# ----------------------------------------------------------------------------------
# 2. what production emits, rendered
# ----------------------------------------------------------------------------------
class TestShippedOutputRendersCorrectly:
    """The payload: every case must render the way a Hebrew reader expects."""

    @pytest.mark.parametrize("case_id,source,tokens", CASES, ids=CASE_IDS)
    def test_shipped_matches_the_authored_layout(self, case_id, source, tokens, evidence):
        assert same_layout(evidence[case_id]["shipped"], evidence[case_id]["reference"]), (
            _report(case_id, "shipped", evidence)
            + f"\n  expected left-to-right: {tokens}"
        )

    @pytest.mark.parametrize("case_id,source,tokens", CASES, ids=CASE_IDS)
    def test_no_glyphs_are_gained_or_lost(self, case_id, source, tokens, evidence):
        """Isolates are zero-width: they may move glyphs, never add or drop ink."""
        shipped = total_ink(evidence[case_id]["shipped"])
        reference = total_ink(evidence[case_id]["reference"])
        assert abs(shipped - reference) / max(reference, 1) < 0.02, (
            f"{case_id}: ink differs by more than 2% — a glyph was dropped or added"
        )

    def test_production_never_emits_the_rtl_override(self):
        """U+202E fights the bidi algorithm instead of informing it."""
        for _case_id, source, _tokens in CASES:
            assert RLO not in bidi_isolate(source)


# ----------------------------------------------------------------------------------
# 3. the rejected alternatives, rendered — this is what makes section 2 evidence
# ----------------------------------------------------------------------------------
class TestNoControlsIsTheDefect:
    """What the pipeline produced before any bidi handling existed."""

    @pytest.mark.parametrize("case_id", sorted(NO_CONTROLS_BREAKS))
    def test_no_controls_gets_the_layout_wrong(self, case_id, evidence):
        assert not same_layout(evidence[case_id]["none"], evidence[case_id]["reference"]), (
            _report(case_id, "none", evidence)
            + "\n  no-controls rendered CORRECTLY, so this case proves nothing"
        )

    @pytest.mark.parametrize("case_id", sorted(ALL_CASES - NO_CONTROLS_BREAKS))
    def test_the_remaining_cases_cannot_discriminate(self, case_id, evidence):
        """Scoping the claim honestly.

        ``b_sym`` is palindromic in its Hebrew words, so both orders render identically;
        ``c_dec``, ``pct`` and ``usd`` have a single numeric run that lands in the same
        place either way. These four are the reason the no-controls render must not be
        used as a reference: it is right exactly where being right is free.
        """
        assert same_layout(evidence[case_id]["none"], evidence[case_id]["reference"]), (
            _report(case_id, "none", evidence)
        )


class TestHistoricalImplementationBreaks:
    """The per-word isolation this rewrite replaced, taken verbatim from git."""

    @pytest.mark.parametrize("case_id", sorted(HISTORICAL_BREAKS))
    def test_it_breaks_where_the_docstring_says_it_does(self, case_id, evidence):
        """Sibling isolates are laid out right-to-left relative to each other.

        Splitting one logical run into several isolates therefore reverses it:
        ``Microsoft Azure`` -> ``Azure Microsoft``, ``3.5`` -> ``5.3``, ``COVID-19`` ->
        ``19-COVID``, and ``%``/``$``/``&`` detach from the number or acronym they belong
        to. This is the entire reason :func:`services.subtitle_engine._ltr_runs` grows
        *maximal* runs between Unicode bidi anchors instead of matching word shapes.
        """
        assert not same_layout(
            evidence[case_id]["historical"], evidence[case_id]["reference"]
        ), (
            _report(case_id, "historical", evidence)
            + "\n  the old implementation renders this correctly — if that is real, the "
            "maximal-run logic may be simplifiable"
        )

    @pytest.mark.parametrize("case_id", sorted(ALL_CASES - HISTORICAL_BREAKS))
    def test_it_survives_single_run_lines(self, case_id, evidence):
        """Scoping: a line with one Latin run and nothing to reorder was never broken."""
        assert same_layout(
            evidence[case_id]["historical"], evidence[case_id]["reference"]
        ), _report(case_id, "historical", evidence)


class TestNaiveMaximalRegexBreaks:
    """The shortcut considered during the rewrite: one regex, no Unicode categories."""

    @pytest.mark.parametrize("case_id", sorted(NAIVE_BREAKS))
    def test_it_loses_the_characters_its_class_forgot(self, case_id, evidence):
        """``50%`` , ``$25`` and ``AT&T``: ``%``, ``$`` and ``&`` are not in the class.

        They fall outside the isolate, so they detach from the run they belong to. The
        shipped implementation asks Unicode for the character's bidi category instead of
        listing characters by hand, which is why it absorbs numeric terminators (``ET``)
        at both edges of a run and does not have this failure mode.
        """
        assert not same_layout(evidence[case_id]["naive"], evidence[case_id]["reference"]), (
            _report(case_id, "naive", evidence)
        )

    @pytest.mark.parametrize("case_id", sorted(ALL_CASES - NAIVE_BREAKS))
    def test_it_survives_the_rest_including_case_d(self, case_id, evidence):
        """Corrects the record: it does NOT break ``התפרצות COVID-19 קשה``.

        The naive regex was rejected on the strength of a report that it reversed the
        Hebrew word order on case (d). Rendered here it does not — case (d) comes out
        identical to the reference. The rejection stands on
        :meth:`test_it_loses_the_characters_its_class_forgot` instead, which is a real
        and reproducible failure; the case originally cited for it was not.
        """
        assert same_layout(evidence[case_id]["naive"], evidence[case_id]["reference"]), (
            _report(case_id, "naive", evidence)
        )


class TestTheMeasurementItselfWorks:
    """A metric that cannot fail proves nothing, so pin its sensitivity too."""

    def test_band_widths_alone_cannot_see_a_reordering_inside_a_word(self):
        """Why per-band pixels are compared and not just band widths.

        ``3.5`` and ``5.3`` are the same three glyphs, so they occupy exactly the same
        width. A width-only comparison calls them equal; the pixel comparison does not.
        """
        forward = render("metric__3_5", visual_reference(["3.5"]))
        reversed_ = render("metric__5_3", visual_reference(["5.3"]))

        assert ink_bands(forward) == ink_bands(reversed_), (
            "the premise of this test is gone: the two now differ in width"
        )
        assert not same_layout(forward, reversed_), (
            "the per-band pixel metric cannot see a digit swap — it would pass "
            "bidi_isolate() emitting 5.3 for 3.5"
        )

    def test_identical_renders_score_zero(self):
        page = render("metric__identity", visual_reference(["שלום", "Microsoft Azure"]))
        assert layout_difference(page, page) == 0.0

    def test_a_different_band_count_is_reported_as_a_mismatch(self):
        two = render("metric__two_bands", visual_reference(["שלום", "עולם"]))
        three = render("metric__three_bands", visual_reference(["שלום", "עולם", "היום"]))
        assert layout_difference(two, three) == MISMATCH


class TestDialogueDashRendersOnTheRightEdge:
    """R4's speaker-turn marker, pixel-checked.

    "— " is an em dash followed by a space, prefixed to a Hebrew line. Its Unicode bidi
    class is ON (neutral), so it takes the direction of the paragraph around it — which
    means it only lands where a Hebrew reader expects it (the RIGHT-hand edge, where the
    line starts) if the line declares its direction. ``bidi_isolate`` does that with the
    line-level RLI, and this asserts the result rather than the intention: a dash that
    renders on the left is not a dialogue marker, it is a typo.
    """

    LINE = "שלום עולם"
    DASH = "— "

    def _dashed(self, tag):
        from services.subtitle_engine import bidi_isolate

        return render(tag, bidi_isolate(self.DASH + self.LINE))

    def test_the_dash_is_at_the_right_hand_edge(self):
        """Measured INSIDE one render: the subtitle is centre-aligned, so absolute x
        positions are not comparable between two different lines."""
        spans, _buffer = _band_spans(self._dashed("dash__rtl"))
        assert len(spans) == 3, f"expected two words and a dash: {spans}"
        *text_bands, dash = spans
        assert dash[1] < min(width for _s, width in text_bands), (
            f"the rightmost run is too wide to be the dash: {spans}"
        )
        assert dash[0] > max(start + width for start, width in text_bands), (
            f"the dash at x={dash[0]} is not right of the Hebrew: {spans}"
        )

    def test_the_dash_does_not_reorder_the_words(self):
        """Adding the marker must not disturb the line it marks."""
        dashed, _ = _band_spans(self._dashed("dash__order"))
        plain, _ = _band_spans(render("dash__order_plain", _bidi(self.LINE)))
        dashed_widths = [width for _s, width in dashed[:-1]]
        plain_widths = [width for _s, width in plain]
        assert len(dashed_widths) == len(plain_widths)
        # +-2px: the line re-centres when the dash is added, and a sub-pixel shift can
        # move one antialiased column across the ink threshold.
        for a, b in zip(dashed_widths, plain_widths):
            assert abs(a - b) <= 2, (dashed_widths, plain_widths)

    def test_the_dash_survives_the_full_ass_pipeline(self):
        """Through build_ass, which is what actually reaches the renderer."""
        from services.subtitle_engine import DIALOGUE_DASH, build_ass

        body = build_ass(
            [{"start": 0.0, "end": 2.0, "text": f"{DIALOGUE_DASH}{self.LINE}"}],
            video_w=W, video_h=H, rtl=True,
        )
        event = [line for line in body.splitlines() if line.startswith("Dialogue:")][0]
        assert "—" in event
        assert event.index("—") < event.index("ש"), "the dash lost its leading position"


def _bidi(text):
    from services.subtitle_engine import bidi_isolate

    return bidi_isolate(text)
