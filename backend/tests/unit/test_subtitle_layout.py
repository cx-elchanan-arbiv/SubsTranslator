"""
Unit tests for the width-aware subtitle layout (``subtitle_engine.layout_params``).

The bug under test
------------------
Font size came from the video HEIGHT alone and the line budget was a hard-coded 42
characters. On a 720x1280 portrait clip that is a 78px font drawing 42-character lines
— roughly 1450px of text on a 720px frame — and since ``WrapStyle: 2`` turns libass's
own wrapping off, the surplus is drawn straight off BOTH edges of the picture. Verified
on a real job (IMG_8975.MP4, 720x1280).

Two obligations, both tested here:

1. **Portrait must fit.** The geometry is asserted directly, and
   ``tests/integration/test_subtitle_layout_render.py`` proves it against real rendered
   pixels.
2. **Landscape must not move.** ``TestLandscapeIsUntouched`` pins the derived numbers
   AND the byte hash of ``build_ass``'s output at six landscape resolutions against
   values captured from the pre-change code (commit 92163f6). Any drift is a failure,
   not a diff to eyeball.
"""

import hashlib
import os
import sys

import pytest


def _find_backend_dir():
    """docker-compose mounts ./tests over /app/tests, so a fixed relative path is wrong."""
    for seed in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = seed
        while True:
            if os.path.isfile(os.path.join(path, "services", "subtitle_engine.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise RuntimeError("could not locate the backend directory containing services/")


backend_dir = _find_backend_dir()
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.subtitle_engine import (  # noqa: E402
    GLYPH_WIDTH_RATIO,
    MAX_LINE_CHARS,
    MIN_FONT_FRAC,
    MIN_LINE_CHARS,
    build_ass,
    layout_params,
    wrap_two_lines,
)

#: A cue set with the awkward shapes: long English, long Hebrew, mixed Hebrew/Latin
#: with digits and a percent sign, an acronym that becomes gershayim, and a short cue.
#: Used for both the golden hashes and the overflow checks.
CUES = [
    {
        "start": 0.0,
        "end": 2.4,
        "text": "It could complicate things. Do you worry about that?",
    },
    {"start": 2.5, "end": 4.0, "text": "כן, אני חושב על זה הרבה מאוד לאחרונה."},
    {
        "start": 4.1,
        "end": 7.0,
        "text": "הם עברו ל-Microsoft Azure ב-2026, וזה עלה 15% יותר.",
    },
    {
        "start": 7.2,
        "end": 9.9,
        "text": 'שירתתי בצה"ל חמש שנים, ואני יודע בדיוק איך זה עובד שם.',
    },
    {"start": 10.0, "end": 12.0, "text": "Short one."},
]


def style_fields(ass_text):
    line = next(x for x in ass_text.splitlines() if x.startswith("Style:"))
    return line.split(":", 1)[1].strip().split(",")


#: The Events Format has 9 fields before Text: Layer, Start, End, Style, Name,
#: MarginL, MarginR, MarginV, Effect. Splitting on ",," instead lands on the empty
#: Name field and silently returns "0,0,0,,<text>" — which is how a length assertion
#: written that way passes for the wrong reason.
_EVENT_TEXT_FIELD = 9


def event_lines(ass_text):
    """Every rendered line of every event, bidi control characters stripped."""
    out = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        body = (
            line[len("Dialogue:") :]
            .strip()
            .split(",", _EVENT_TEXT_FIELD)[_EVENT_TEXT_FIELD]
        )
        for piece in body.split("\\N"):
            out.append(piece.replace("⁦", "").replace("⁧", "").replace("⁩", ""))
    return out


# ======================================================================================
# The regression guard: landscape must be bit-for-bit what it was
# ======================================================================================
@pytest.mark.unit
class TestLandscapeIsUntouched:
    """Every landscape resolution must produce exactly the pre-change output.

    The expected values below were produced by running the PREVIOUS implementation
    (``git show 92163f6:backend/services/subtitle_engine.py``) over :data:`CUES` in this
    project's own container. They are not re-derived from the current code, which would
    make the test tautological.
    """

    #: (width, height) -> (font_px, margin_v, max_line_chars)
    EXPECTED_PARAMS = {
        (1280, 720): (44, 86, 42),
        (1920, 1080): (66, 130, 42),
        (3840, 2160): (132, 259, 42),
        (854, 480): (29, 58, 42),
        (640, 360): (22, 43, 42),
        (1920, 800): (49, 96, 42),
    }

    #: (width, height, rtl) -> sha256 of the complete .ass file the OLD code emitted.
    GOLDEN_HASHES = {
        (
            1280,
            720,
            True,
        ): "9c23677460c5267b00b5b6d16d625ae084847763c088ccfbe7d0c9c1fae97b35",
        (
            1280,
            720,
            False,
        ): "3c2d8ffb658aed992e19aa6d79c779dcc2f50c9907ff372d40161895d73ce461",
        (
            1920,
            1080,
            True,
        ): "580c53e83a28833fa47fa5e878079ea720ebab2ae52bcc45b0c118361f1949eb",
        (
            1920,
            1080,
            False,
        ): "b47472584998d851f028ebb31f075ab25dedd3855a97b140e1954a3fa6183cec",
        (
            3840,
            2160,
            True,
        ): "73e327ca15d99afbc237691cc93879b0b999742bd3bf876f7c0f01f62ac9a20e",
        (
            3840,
            2160,
            False,
        ): "fd5cef7dd51abe11730cefc2a13cf885e9b40d75bf9af7b6d0766a89f9e3b4e7",
        (
            854,
            480,
            True,
        ): "a68d22ce817b4ec2b099767437129ab39085263987c2a41616c06a87329931f2",
        (
            854,
            480,
            False,
        ): "71594dd5738fa6fa7f794dd0da908d8a6a884a2178a414cff49a33870ea6a13b",
        (
            640,
            360,
            True,
        ): "c8102dd612d0dd98d8d89cdf8bbe1e385cf48fa89a7d86c0ae930c83a8f4e9fc",
        (
            640,
            360,
            False,
        ): "2f6f77d63bfcffaf1e23e361fb9840bdf168a499a9054e864cf70090b354c804",
        (
            1920,
            800,
            True,
        ): "f58b0113c5f6898e842154a5e561a6831b0617610c708211da38a7b9e0f172e1",
        (
            1920,
            800,
            False,
        ): "008517cdb63456b236f24e73b9f14546b1515783801b621f9827bd736b896607",
    }

    @pytest.mark.parametrize(("size", "expected"), sorted(EXPECTED_PARAMS.items()))
    def test_derived_params_are_the_historical_ones(self, size, expected):
        params = layout_params(*size)
        assert (
            params["font_px"],
            params["margin_v"],
            params["max_line_chars"],
        ) == expected
        # 60px margins and an 84-character cue budget are the historical constants.
        assert params["margin_h"] == 60
        assert params["max_chars_per_cue"] == 84
        assert params["max_lines"] == 2

    @pytest.mark.parametrize(("key", "digest"), sorted(GOLDEN_HASHES.items()))
    def test_build_ass_output_is_byte_identical(self, key, digest):
        width, height, rtl = key
        out = build_ass(CUES, video_w=width, video_h=height, rtl=rtl)
        assert hashlib.sha256(out.encode("utf-8")).hexdigest() == digest, (
            f"{width}x{height} rtl={rtl}: landscape rendering changed. The width-aware "
            "layout is only allowed to alter frames that could not fit a 42-character "
            "line; 16:9 always could."
        )

    def test_landscape_is_never_the_width_constrained_case(self):
        """The width term must lose to the height term on every 16:9 frame."""
        for height in (360, 480, 720, 1080, 1440, 2160):
            width = round(height * 16 / 9)
            params = layout_params(width, height)
            assert params["font_px"] == round(height * 0.061)
            assert params["max_line_chars"] == MAX_LINE_CHARS


# ======================================================================================
# Portrait: the bug
# ======================================================================================
@pytest.mark.unit
class TestPortrait:
    def test_the_reported_video_now_fits(self):
        """720x1280 — the exact frame from IMG_8975.MP4."""
        params = layout_params(720, 1280)
        # Was 78px (round(1280 * 0.061)) at 42 chars => ~1450px on a 720px frame.
        assert params["font_px"] == 41
        assert params["max_line_chars"] == 33
        assert params["max_chars_per_cue"] == 66
        assert params["usable_w"] == 600
        estimated = params["max_line_chars"] * params["font_px"] * GLYPH_WIDTH_RATIO
        assert estimated <= params["usable_w"]

    def test_the_old_rule_would_not_have_fitted(self):
        """Guards the guard: the pre-change numbers must fail the same check."""
        params = layout_params(720, 1280)
        old_font = round(1280 * 0.061)
        old_estimate = MAX_LINE_CHARS * old_font * GLYPH_WIDTH_RATIO
        assert old_font == 78
        assert old_estimate > params["usable_w"] * 2, (
            "the old parameters should overflow the frame by more than 2x — if they do "
            "not, this test is no longer measuring the bug it was written for"
        )

    def test_1080x1920_portrait(self):
        params = layout_params(1080, 1920)
        assert params["font_px"] == 61  # the 3.2% floor, not 117px
        assert params["max_line_chars"] == 35
        assert params["max_chars_per_cue"] == 70

    def test_font_never_goes_below_the_legibility_floor(self):
        """Narrow lines are the price; microscopic text is not on the table."""
        for width, height in [(720, 1280), (1080, 1920), (480, 854), (608, 1080)]:
            params = layout_params(width, height)
            assert params["font_px"] >= round(height * MIN_FONT_FRAC)

    def test_build_ass_applies_the_portrait_budget(self):
        out = build_ass(CUES, video_w=720, video_h=1280)
        fields = style_fields(out)
        assert fields[2] == "41"  # Fontsize
        assert fields[19] == "60" and fields[20] == "60"  # MarginL / MarginR
        assert fields[21] == "154"  # MarginV: round(1280 * 0.12)
        for line in event_lines(out):
            assert len(line) <= 33, f"{len(line)} chars on a 33-char frame: {line!r}"

    def test_build_ass_landscape_still_allows_42(self):
        """The portrait clamp must not leak into landscape wrapping."""
        out = build_ass(CUES, video_w=1920, video_h=1080)
        assert max(len(line) for line in event_lines(out)) > 33


# ======================================================================================
# Other aspect ratios
# ======================================================================================
@pytest.mark.unit
class TestOtherAspectRatios:
    def test_square_keeps_the_full_budget_on_a_smaller_font(self):
        params = layout_params(1080, 1080)
        assert params["font_px"] == 51  # < round(1080 * 0.061) == 66
        assert params["max_line_chars"] == MAX_LINE_CHARS
        assert params["max_chars_per_cue"] == 84

    @pytest.mark.parametrize(
        "size",
        [
            (1280, 720),
            (1920, 1080),
            (3840, 2160),
            (854, 480),
            (640, 360),
            (1080, 1080),
            (720, 1280),
            (1080, 1920),
            (480, 854),
            (240, 426),
            (1920, 800),
            (2560, 1080),
            (1440, 1080),
            (360, 640),
            (200, 400),
        ],
    )
    def test_estimated_line_width_never_exceeds_the_usable_width(self, size):
        """The invariant the whole module exists to hold."""
        params = layout_params(*size)
        estimated = params["max_line_chars"] * params["font_px"] * GLYPH_WIDTH_RATIO
        assert estimated <= params["usable_w"], (
            f"{size}: {params['max_line_chars']} chars at {params['font_px']}px is "
            f"~{estimated:.0f}px on a {params['usable_w']}px usable width"
        )

    @pytest.mark.parametrize(
        "size",
        [(1280, 720), (720, 1280), (1080, 1080), (480, 854), (3840, 2160)],
    )
    def test_wrapped_lines_respect_the_derived_budget(self, size):
        params = layout_params(*size)
        for cue in CUES:
            for line in wrap_two_lines(cue["text"], params["max_line_chars"]):
                assert len(line) <= params["max_line_chars"]

    def test_wider_frames_do_not_earn_longer_lines(self):
        """42 characters is a reading-comfort ceiling, not a geometric one."""
        assert layout_params(7680, 4320)["max_line_chars"] == MAX_LINE_CHARS
        assert layout_params(5000, 500)["max_line_chars"] == MAX_LINE_CHARS


# ======================================================================================
# Degenerate input
# ======================================================================================
@pytest.mark.unit
class TestDegenerateInput:
    def test_margin_shrinks_before_the_usable_width_can_go_negative(self):
        params = layout_params(100, 200)
        assert params["margin_h"] < 60
        assert params["usable_w"] > 0

    @pytest.mark.parametrize("size", [(0, 0), (-100, -100), (None, None), (1, 1)])
    def test_never_raises_and_never_returns_nonsense(self, size):
        params = layout_params(*size)
        assert params["font_px"] >= 1
        assert params["usable_w"] >= 1
        assert params["max_line_chars"] >= MIN_LINE_CHARS
        assert params["max_chars_per_cue"] == (
            params["max_line_chars"] * params["max_lines"]
        )

    def test_absurdly_narrow_frames_clamp_rather_than_collapse(self):
        """Below ~200px wide nothing is legible; the answer must still be usable."""
        params = layout_params(40, 400)
        assert params["max_line_chars"] >= MIN_LINE_CHARS

    def test_custom_glyph_ratio_is_honoured(self):
        """A wider font must buy fewer characters, not silently the same 42."""
        narrow = layout_params(720, 1280, glyph_ratio=0.30)
        wide = layout_params(720, 1280, glyph_ratio=0.60)
        assert narrow["max_line_chars"] > wide["max_line_chars"]

    def test_zero_glyph_ratio_falls_back_instead_of_dividing_by_zero(self):
        assert layout_params(720, 1280, glyph_ratio=0)["max_line_chars"] == 33


# ======================================================================================
# build_ass / layout agreement
# ======================================================================================
@pytest.mark.unit
class TestBuildAssHonoursAnInjectedLayout:
    def test_explicit_layout_wins_over_the_dimensions(self):
        """The renderer must use the budget upstream actually wrote text against."""
        layout = layout_params(720, 1280)
        out = build_ass(CUES, video_w=720, video_h=1280, layout=layout)
        assert out == build_ass(CUES, video_w=720, video_h=1280)

        forced = dict(layout, font_px=20, max_line_chars=20, margin_h=5, margin_v=7)
        out = build_ass(CUES, video_w=720, video_h=1280, layout=forced)
        fields = style_fields(out)
        assert fields[2] == "20"
        assert fields[19] == "5" and fields[20] == "5" and fields[21] == "7"
        for line in event_lines(out):
            assert len(line) <= 20
