"""
Pixel proof that subtitles stay inside the frame — portrait as well as landscape.

Why pixels and not arithmetic
-----------------------------
``tests/unit/test_subtitle_layout.py`` proves the *arithmetic* holds: characters x font
size x :data:`GLYPH_WIDTH_RATIO` fits the usable width. That is only as true as the
ratio, and the ratio is a property of a font file and a rasteriser, not of Python. So
this file renders the real ``.ass`` through the same FFmpeg/libass that renders
production output, reads the frames back as raw pixels and looks at where the ink
actually landed.

Three things are established here:

1. :class:`TestGlyphWidthRatio` re-derives ``GLYPH_WIDTH_RATIO`` from live renders. If
   the font, the style or libass changes underneath the layout, this goes red *before*
   anything ships off-frame.
2. :class:`TestPortraitRender` renders 720x1280 — the reported failing geometry — and
   asserts there is no ink in the outer margin columns, then re-renders the SAME cues
   with the pre-fix parameters and asserts that version DOES spill. A guard that cannot
   fail on the original bug is not a guard.
3. :class:`TestLandscapeRender` asserts 16:9 output is untouched, both as pixels and as
   a byte hash of the ``.ass``.

Requires ffmpeg, numpy and the Hebrew fonts — i.e. this project's own container. It
skips cleanly anywhere else.
"""
import hashlib
import os
import subprocess
import sys

import pytest


def _find_backend_dir():
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

numpy = pytest.importorskip("numpy")

from services.subtitle_engine import (  # noqa: E402
    GLYPH_WIDTH_RATIO,
    MAX_LINE_CHARS,
    build_ass,
    layout_params,
)

FONTS_DIR = os.getenv("ASS_FONTS_DIR", "/usr/share/fonts/truetype/hebrew")

#: A pixel is "ink" when it is clearly brighter than the black canvas. The subtitle
#: text is pure white (&H00FFFFFF); the opaque box behind it is black at 92% alpha, so
#: on a black canvas only the glyphs show — which is exactly what must stay in frame.
INK_THRESHOLD = 96

#: Real Hebrew of the length this pipeline actually produces (the CPS pass targets the
#: per-cue budget, so cues arrive close to it).
HEBREW_LINES = [
    "אני חושב שזה יכול לסבך את העניינים בהמשך הדרך",
    "הוא אמר שהמצב הביטחוני באזור משתפר משמעותית",
    "זה בדיוק מה שקורה כשלא מקשיבים לאנשי המקצוע",
    "ההחלטה התקבלה אתמול בישיבת הממשלה בירושלים",
    "מה שחשוב עכשיו זה להמשיך ולפעול בזהירות רבה",
    "אנחנו לא יודעים בדיוק מה הולך לקרות מחר בבוקר",
    "היא הסבירה לי בדיוק איך המערכת החדשה הזאת עובדת",
    "בשנה שעברה הם השקיעו סכומים גדולים מאוד בפרויקט",
    "צה״ל הודיע על סיום התרגיל הגדול באזור הצפון",
    "העלייה במחירים הגיעה לחמישה עשר אחוזים השנה",
    "הם עברו ל-Microsoft Azure ב-2026 וזה עלה 15% יותר",
    "הדוח של ה-ICC פורסם אתמול בערב אחרי הרבה עיכובים",
]


def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10, check=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_ffmpeg() or not os.path.isdir(FONTS_DIR),
    reason="needs ffmpeg + the container's Hebrew fonts",
)


def render_frames(ass_path, width, height, frames):
    """Rasterise ``frames`` one-second-apart frames of the .ass on a black canvas.

    Returns a ``(frames, height, width)`` uint8 array of luma. One frame per second at
    ``-r 1`` lines up with cues laid out one second apart, so every cue is inspected.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=1:d={frames}",
        "-vf", f"ass='{ass_path}':fontsdir={FONTS_DIR}:shaping=complex",
        "-frames:v", str(frames), "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=180).stdout
    expected = frames * width * height
    assert len(raw) == expected, f"got {len(raw)} bytes of pixels, expected {expected}"
    return numpy.frombuffer(raw, dtype=numpy.uint8).reshape(frames, height, width)


def ink_columns(frames):
    """Boolean array, one entry per pixel column: is there ink anywhere in it?"""
    return (frames > INK_THRESHOLD).any(axis=(0, 1))


def cues_from(lines):
    """One cue per second, so frame N shows cue N."""
    return [
        {"start": float(i), "end": float(i) + 0.9, "text": text}
        for i, text in enumerate(lines)
    ]


def write_ass(tmp_path, content, name="probe.ass"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


# ======================================================================================
@pytest.mark.integration
class TestGlyphWidthRatio:
    """The measured constant must still describe what libass actually draws."""

    def _line_width(self, tmp_path, text, font_size):
        """Width in pixels of one line rendered on a canvas too wide to clip it."""
        width, height = 6000, 400
        ass = (
            "[Script Info]\nScriptType: v4.00+\n"
            f"PlayResX: {width}\nPlayResY: {height}\n"
            "WrapStyle: 2\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.709\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
            " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
            " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
            " Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: He,Noto Sans Hebrew,{font_size},"
            "&H00FFFFFF,&H00FFFFFF,&H00000000,&H14000000,"
            "1,0,0,0,100,100,0,0,4,3,0,5,0,0,0,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
            " Effect, Text\n"
            f"Dialogue: 0,0:00:00.00,0:00:10.00,He,,0,0,0,,{text}\n"
        )
        path = write_ass(tmp_path, ass, "width_probe.ass")
        # White canvas: measures the opaque BOX, i.e. the full visual extent of the
        # line rather than just the glyph ink inside it.
        cmd = [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=white:s={width}x{height}:d=0.1",
            "-vf", f"ass='{path}':fontsdir={FONTS_DIR}:shaping=complex",
            "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
        frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(height, width)
        cols = numpy.flatnonzero((frame < 200).any(axis=0))
        assert cols.size, f"nothing rendered for {text!r}"
        return int(cols[-1] - cols[0] + 1)

    def test_measured_glyph_width_ratio_still_holds(self, tmp_path):
        """Every real line must be no wider than the constant predicts.

        This is the assumption the whole layout rests on, re-derived from live renders
        rather than trusted. If a font update makes Hebrew wider, this fails here — in a
        test — instead of in a delivered video.
        """
        ratios = []
        for font_size in (60, 100, 160):
            for text in HEBREW_LINES:
                width = self._line_width(tmp_path, text, font_size)
                ratios.append(width / (len(text) * font_size))
        worst = max(ratios)
        assert worst <= GLYPH_WIDTH_RATIO, (
            f"widest measured line is {worst:.4f} x font size per character, but the "
            f"layout assumes {GLYPH_WIDTH_RATIO}. Re-measure and raise the constant."
        )
        # ...and not so conservative that the budget is being thrown away.
        assert worst > GLYPH_WIDTH_RATIO * 0.75, (
            f"widest measured line is only {worst:.4f}; {GLYPH_WIDTH_RATIO} is now "
            "needlessly pessimistic and is costing characters per line"
        )

    def test_the_ratio_is_independent_of_font_size(self, tmp_path):
        """A single constant is only valid because the rendering scales linearly."""
        text = HEBREW_LINES[0]
        ratios = [
            self._line_width(tmp_path, text, size) / (len(text) * size)
            for size in (40, 80, 160)
        ]
        assert max(ratios) - min(ratios) < 0.01, ratios


# ======================================================================================
@pytest.mark.integration
class TestPortraitRender:
    """720x1280 — the geometry of the job that exposed the bug (IMG_8975.MP4)."""

    WIDTH, HEIGHT = 720, 1280

    def test_no_ink_in_the_outer_margin_columns(self, tmp_path):
        layout = layout_params(self.WIDTH, self.HEIGHT)
        cues = cues_from(HEBREW_LINES)
        path = write_ass(tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT))
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))

        columns = ink_columns(frames)
        assert columns.any(), "nothing rendered at all — the probe is not measuring"

        margin = layout["margin_h"]
        left = numpy.flatnonzero(columns)[0]
        right = numpy.flatnonzero(columns)[-1]
        assert not columns[:margin].any(), (
            f"ink at column {left} is inside the {margin}px left margin"
        )
        assert not columns[self.WIDTH - margin:].any(), (
            f"ink at column {right} is inside the {margin}px right margin"
        )
        # Belt and braces: nothing may touch the frame edges either.
        assert left > 0 and right < self.WIDTH - 1

    def test_the_pre_fix_parameters_do_spill(self, tmp_path):
        """The guard has teeth: the old font/line rule must fail this same check.

        Renders the identical cues with the pre-change layout — font = 6.1% of HEIGHT
        (78px) and a flat 42-character line — and asserts the ink reaches the frame
        edges. Without this, a broken assertion would look like a passing fix.
        """
        old_layout = dict(
            layout_params(self.WIDTH, self.HEIGHT),
            font_px=round(self.HEIGHT * 0.061),
            max_line_chars=MAX_LINE_CHARS,
        )
        assert old_layout["font_px"] == 78
        cues = cues_from(HEBREW_LINES)
        path = write_ass(
            tmp_path,
            build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT, layout=old_layout),
            "old.ass",
        )
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))
        columns = ink_columns(frames)
        assert columns[0] and columns[-1], (
            "the pre-fix parameters were expected to run text off both edges of a "
            "720x1280 frame; they did not, so this test no longer reproduces the bug"
        )

    def test_every_line_is_narrower_than_the_usable_width(self, tmp_path):
        """Per-cue, not just in aggregate: one bad cue must not hide behind eleven good ones."""
        layout = layout_params(self.WIDTH, self.HEIGHT)
        cues = cues_from(HEBREW_LINES)
        path = write_ass(tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT))
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))
        for index, frame in enumerate(frames):
            cols = numpy.flatnonzero((frame > INK_THRESHOLD).any(axis=0))
            assert cols.size, f"cue {index} rendered nothing"
            span = int(cols[-1] - cols[0] + 1)
            assert span <= layout["usable_w"], (
                f"cue {index} ({HEBREW_LINES[index]!r}) rendered {span}px wide on a "
                f"{layout['usable_w']}px usable width"
            )


# ======================================================================================
@pytest.mark.integration
class TestLandscapeRender:
    """16:9 must be exactly what it was before the layout learned about width."""

    WIDTH, HEIGHT = 1280, 720

    def test_no_ink_in_the_outer_margin_columns(self, tmp_path):
        layout = layout_params(self.WIDTH, self.HEIGHT)
        cues = cues_from(HEBREW_LINES)
        path = write_ass(tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT))
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))
        columns = ink_columns(frames)
        assert columns.any()
        assert not columns[: layout["margin_h"]].any()
        assert not columns[self.WIDTH - layout["margin_h"]:].any()

    def test_ass_bytes_are_the_pre_change_ones(self):
        """Byte-level regression, mirroring tests/unit/test_subtitle_layout.py.

        The hash was captured by running the previous implementation
        (``git show 92163f6:backend/services/subtitle_engine.py``) over these cues.
        """
        cues = cues_from(HEBREW_LINES)
        out = build_ass(cues, video_w=1280, video_h=720)
        assert (
            hashlib.sha256(out.encode("utf-8")).hexdigest()
            == "866eebdea4b40011bb442f3147de320ff280ee4349ebb762f92de621a05939bc"
        )
