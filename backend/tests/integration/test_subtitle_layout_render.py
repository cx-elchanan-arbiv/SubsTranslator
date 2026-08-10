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
    ASS_ALIGNMENT,
    GLYPH_WIDTH_RATIO,
    MAX_LINE_CHARS,
    build_ass,
    layout_params,
)
from services.subtitle_service import SSA_ALIGNMENT  # noqa: E402

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
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=10, check=True
        )
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
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r=1:d={frames}",
        "-vf",
        f"ass='{ass_path}':fontsdir={FONTS_DIR}:shaping=complex",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
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
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s={width}x{height}:d=0.1",
            "-vf",
            f"ass='{path}':fontsdir={FONTS_DIR}:shaping=complex",
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
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
        path = write_ass(
            tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT)
        )
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))

        columns = ink_columns(frames)
        assert columns.any(), "nothing rendered at all — the probe is not measuring"

        margin = layout["margin_h"]
        left = numpy.flatnonzero(columns)[0]
        right = numpy.flatnonzero(columns)[-1]
        assert not columns[
            :margin
        ].any(), f"ink at column {left} is inside the {margin}px left margin"
        assert not columns[
            self.WIDTH - margin :
        ].any(), f"ink at column {right} is inside the {margin}px right margin"
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
        path = write_ass(
            tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT)
        )
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
        path = write_ass(
            tmp_path, build_ass(cues, video_w=self.WIDTH, video_h=self.HEIGHT)
        )
        frames = render_frames(path, self.WIDTH, self.HEIGHT, len(cues))
        columns = ink_columns(frames)
        assert columns.any()
        assert not columns[: layout["margin_h"]].any()
        assert not columns[self.WIDTH - layout["margin_h"] :].any()

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


# ======================================================================================
# P6 — lower-third avoidance, verified on real footage
# ======================================================================================
#: The original calibration clip. NOT in the repo and never was — it is a one-off phone
#: capture that lived in the author's uploads folder. The five tests below that name it
#: are gated on it individually; the rest of the class runs off the committed corpus.
FOX_CLIP = "/app/uploads/IMG_2870.MP4"
CLEAN_CLIP = "/app/uploads/corpus/clean_speech.mp4"
#: The clip that proved the detector was structurally blind: its BREAKING NEWS bar sits
#: 12px BELOW the sampled subtitle band, inside the bottom margin the old band stopped at.
BOTTOM_BAR_CLIP = "/app/uploads/corpus/eng_chyron.mp4"
#: Must stay quiet: a 1920x1080 rally clip with no lower third at all.
QUIET_HD_CLIP = "/app/uploads/corpus/טראמפ בדיקה 3.mp4"

#: Gate for the tests that specifically need the uncommitted calibration clip. This used
#: to gate the WHOLE class, so a file that is not in the repo silently suppressed the six
#: tests that only ever needed the committed corpus — including the entire two-band
#: regression set, which is the part most worth running.
needs_fox_clip = pytest.mark.skipif(
    not os.path.exists(FOX_CLIP),
    reason="IMG_2870.MP4 is an uncommitted calibration clip, absent from the corpus",
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(CLEAN_CLIP) or not os.path.exists(BOTTOM_BAR_CLIP),
    reason="needs the corpus clips inside the container",
)
class TestLowerThirdAvoidance:
    """The detector's decision on the two clips it was calibrated against.

    A unit test with synthetic arrays proves the SCORING; only real footage proves the
    THRESHOLD, which is the part that was chosen from measurements and could rot when a
    codec, a scaler or the crop geometry changes.
    """

    @staticmethod
    def _decide(path):
        from services.subtitle_service import subtitle_service

        width, height = subtitle_service.probe_video_dimensions(path)
        layout = layout_params(width, height)
        return subtitle_service.detect_lower_third(path, layout), layout

    @needs_fox_clip
    def test_fox_chyron_is_detected(self):
        decision, _ = self._decide(FOX_CLIP)
        assert decision["busy"] is True, decision
        assert decision["score"] > decision["threshold"]

    def test_clean_speech_is_not_detected(self):
        decision, _ = self._decide(CLEAN_CLIP)
        assert decision["busy"] is False, decision
        assert decision["score"] <= decision["threshold"]

    @needs_fox_clip
    def test_detection_is_well_inside_the_two_second_budget(self):
        decision, _ = self._decide(FOX_CLIP)
        assert decision["elapsed_s"] < 2.0, decision

    @needs_fox_clip
    def test_detection_is_deterministic_across_runs(self):
        first, _ = self._decide(FOX_CLIP)
        second, _ = self._decide(FOX_CLIP)
        assert first["samples"] == second["samples"]
        assert first["score"] == second["score"]

    @needs_fox_clip
    def test_threshold_keeps_a_real_margin_on_both_sides(self):
        """Neither clip may sit on the fence — a threshold with no headroom is noise."""
        busy, _ = self._decide(FOX_CLIP)
        clear, _ = self._decide(CLEAN_CLIP)
        assert busy["score"] > clear["score"] * 2

    @needs_fox_clip
    def test_render_moves_the_box_up_on_the_chyron_clip(self, tmp_path):
        from services.subtitle_service import subtitle_service

        _, layout = self._decide(FOX_CLIP)
        cues = [
            {"start": 0.5, "end": 4.0, "text": "x", "translated_text": "שורת בדיקה"}
        ]
        margins = {}
        for tag, detect in (("auto", True), ("off", False)):
            out = str(tmp_path / f"fox_{tag}.mp4")
            assert subtitle_service.create_video_with_ass(
                FOX_CLIP,
                cues,
                out,
                target_language="he",
                layout=layout,
                detect_lower_third=detect,
            )
            style = [
                line
                for line in open(os.path.splitext(out)[0] + ".ass", encoding="utf-8")
                if line.startswith("Style:")
            ][0]
            margins[tag] = int(style.strip().split(",")[21])
        assert margins["auto"] > margins["off"], margins

    def test_a_chyron_in_the_bottom_margin_is_detected(self):
        """R3's central case. Before the two-band fix this scored 0.0093 and was 'clear'.

        The subtitle band alone still reads clear — that is the point. The bar is in the
        bottom strip, which the detector could not previously see at all.
        """
        if not os.path.exists(BOTTOM_BAR_CLIP):
            pytest.skip("eng_chyron.mp4 is not in the container")
        decision, _ = self._decide(BOTTOM_BAR_CLIP)
        assert decision["busy"] is True, decision
        assert decision["decided_by"] == "bottom", decision
        assert decision["bands"]["subtitle"]["score"] < decision["threshold"]
        assert decision["bands"]["bottom"]["score"] > decision["threshold"]

    def test_merging_the_two_bands_would_have_missed_it(self):
        """Kept as an executable argument against the obvious simplification.

        Scoring one taller band averages the busy strip with the clear band above it.
        Measured: 0.0450 against a 0.070 threshold — still 'clear', still wrong.
        """
        if not os.path.exists(BOTTOM_BAR_CLIP):
            pytest.skip("eng_chyron.mp4 is not in the container")
        decision, _ = self._decide(BOTTOM_BAR_CLIP)
        subtitle = decision["bands"]["subtitle"]
        bottom = decision["bands"]["bottom"]
        weighted = (
            subtitle["score"] * subtitle["h"] + bottom["score"] * bottom["h"]
        ) / (subtitle["h"] + bottom["h"])
        assert (
            weighted < decision["threshold"]
        ), f"the merged band scores {weighted:.4f} — this test's premise is stale"

    def test_a_clean_hd_clip_stays_quiet_with_both_bands_scored(self):
        """The false-positive guard: adding a band must not start firing on clean video."""
        if not os.path.exists(QUIET_HD_CLIP):
            pytest.skip("the HD rally clip is not in the container")
        decision, _ = self._decide(QUIET_HD_CLIP)
        assert decision["busy"] is False, decision
        assert decision["bands"]["bottom"]["score"] <= decision["threshold"]

    def test_render_moves_the_box_up_off_the_bottom_bar(self, tmp_path):
        from services.subtitle_service import subtitle_service

        if not os.path.exists(BOTTOM_BAR_CLIP):
            pytest.skip("eng_chyron.mp4 is not in the container")
        _, layout = self._decide(BOTTOM_BAR_CLIP)
        cues = [
            {"start": 0.5, "end": 4.0, "text": "x", "translated_text": "שורת בדיקה"}
        ]
        margins = {}
        for tag, detect in (("auto", True), ("off", False)):
            out = str(tmp_path / f"bar_{tag}.mp4")
            assert subtitle_service.create_video_with_ass(
                BOTTOM_BAR_CLIP,
                cues,
                out,
                target_language="he",
                layout=layout,
                detect_lower_third=detect,
            )
            style = [
                line
                for line in open(os.path.splitext(out)[0] + ".ass", encoding="utf-8")
                if line.startswith("Style:")
            ][0]
            margins[tag] = int(style.strip().split(",")[21])
        assert margins["auto"] > margins["off"], margins

    def test_render_leaves_the_clean_clip_exactly_where_it_was(self, tmp_path):
        from services.subtitle_service import subtitle_service

        _, layout = self._decide(CLEAN_CLIP)
        cues = [
            {"start": 0.5, "end": 4.0, "text": "x", "translated_text": "שורת בדיקה"}
        ]
        produced = {}
        for tag, detect in (("auto", True), ("off", False)):
            out = str(tmp_path / f"clean_{tag}.mp4")
            assert subtitle_service.create_video_with_ass(
                CLEAN_CLIP,
                cues,
                out,
                target_language="he",
                layout=layout,
                detect_lower_third=detect,
            )
            with open(os.path.splitext(out)[0] + ".ass", encoding="utf-8") as fh:
                produced[tag] = fh.read()
        assert (
            produced["auto"] == produced["off"]
        ), "a clear clip must render identically"


# ======================================================================================
@pytest.mark.integration
class TestBurnedSubtitlePosition:
    """Where the subtitle actually lands, for both renderers.

    The bug this pins: `subtitle_position` was wired through with ONE alignment map,
    the ASS v4+ numpad one. It is right for the `ass` filter and wrong for the legacy
    `subtitles` + `force_style` path, which libass parses with SSA v4 semantics — so
    `top` came out middle-LEFT and `side` came out at the TOP. Every unit test passed,
    because they all asserted on the argv string rather than on the picture.

    So these render and look. A frame is split into vertical and horizontal thirds and
    the ink's centre of mass has to fall in the expected one.
    """

    W, H = 640, 360

    @staticmethod
    def _ink_centre(frame):
        ys, xs = numpy.where(frame > INK_THRESHOLD)
        assert ys.size, "nothing was drawn"
        return xs.mean() / frame.shape[1], ys.mean() / frame.shape[0]

    @staticmethod
    def _band(fraction):
        return (
            "start" if fraction < 1 / 3 else ("middle" if fraction < 2 / 3 else "end")
        )

    @pytest.mark.parametrize(
        "position,want_v,want_h",
        [
            ("bottom", "end", "middle"),
            ("top", "start", "middle"),
            ("side", "middle", "end"),
        ],
    )
    def test_the_ass_renderer_puts_the_subtitle_where_asked(
        self, tmp_path, position, want_v, want_h
    ):
        ass = build_ass(
            cues_from(["שלום"]), video_w=self.W, video_h=self.H, position=position
        )
        frames = render_frames(write_ass(tmp_path, ass), self.W, self.H, 1)
        x, y = self._ink_centre(frames[0])
        assert self._band(y) == want_v, f"{position}: vertical band {y:.2f}"
        assert self._band(x) == want_h, f"{position}: horizontal band {x:.2f}"

    @pytest.mark.parametrize(
        "position,want_v,want_h",
        [
            ("bottom", "end", "middle"),
            ("top", "start", "middle"),
            ("side", "middle", "end"),
        ],
    )
    def test_the_legacy_renderer_puts_the_subtitle_where_asked(
        self, tmp_path, position, want_v, want_h
    ):
        srt = tmp_path / "p.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nMARK\n\n", encoding="utf-8")
        style = (
            "FontName=Arial,FontSize=18,Bold=1,PrimaryColour=&HFFFFFF,MarginV=40,"
            f"Alignment={SSA_ALIGNMENT[position]}"
        )
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={self.W}x{self.H}:r=1:d=1",
                "-vf",
                f"subtitles='{srt}':force_style='{style}'",
                "-frames:v",
                "1",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=180,
        ).stdout
        frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(self.H, self.W)
        x, y = self._ink_centre(frame)
        assert self._band(y) == want_v, f"{position}: vertical band {y:.2f}"
        assert self._band(x) == want_h, f"{position}: horizontal band {x:.2f}"

    def test_the_two_renderers_disagree_on_purpose(self):
        """A guard against 'tidying up' the two maps into one shared constant."""
        assert ASS_ALIGNMENT["bottom"] == SSA_ALIGNMENT["bottom"] == 2
        assert ASS_ALIGNMENT["top"] != SSA_ALIGNMENT["top"]
        assert ASS_ALIGNMENT["side"] != SSA_ALIGNMENT["side"]


# ======================================================================================
@pytest.mark.integration
class TestBurnedBidiTreatment:
    """The default burn path renders logical-order RTL text correctly.

    Pins the fix for two bugs that were burned into every default-path run:
    ``fix_rtl_text_for_subtitles`` pre-swapped bracket pairs — libass mirrors them
    itself, so ``(בערך)`` burned as ``)בערך(`` — and wrapped every line in RLO
    (U+202E), under which a Latin run split by the upstream geresh rewrite burned
    scrambled: ``I can't do it, I'll never`` came out ``ll never׳t do it, I׳I can``.

    Method, borrowed from ``test_bidi_render.py``: each logical cue runs through the
    REAL production chain (``create_srt_file`` -> ``create_video_with_subtitles``,
    the ``subtitles`` filter with ``force_style``) and the burned frame is compared,
    ink-band by ink-band, against the same sentence hand-authored in VISUAL order and
    burned through the same function with the bidi treatment switched off. The
    reference is written from the meaning of the sentence, not derived from the code
    under test, so agreement is evidence and not a tautology. A positive control
    shows the comparison can fail: re-applying the old RLO treatment must break it.
    """

    W, H = 640, 360
    RLI, LRI, PDI = "\u2067", "\u2066", "\u2069"

    #: Per-band mean absolute pixel difference at which two burns still count as the
    #: same layout. In ``test_bidi_render.py``'s calibration every correct render
    #: measured <= 1.4 and every broken one >= 8.7; identical filter + style here
    #: should land near 0.
    SAME_LAYOUT_MAD = 3.0

    #: ``(logical cue text, the same words in the order a viewer must SEE them,
    #: left to right on screen)``
    CASES = [
        ("זה (בערך) נכון", ["נכון", "(בערך)", "זה"]),
        ("בשנת 2024 זה קרה", ["קרה", "זה", "2024", "בשנת"]),
        (
            "אמרתי I can't do it, I'll never ויצאתי",
            ["ויצאתי", "I can't do it, I'll never", "אמרתי"],
        ),
    ]

    # -- plumbing ---------------------------------------------------------------------

    @pytest.fixture()
    def black_clip(self, tmp_path):
        path = str(tmp_path / "black.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={self.W}x{self.H}:r=10:d=2",
                "-c:v",
                "libx264",
                "-y",
                path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return path

    @staticmethod
    def _service(monkeypatch, treatment=None):
        from services.subtitle_service import SubtitleService

        svc = SubtitleService()
        monkeypatch.setattr(svc.config, "USE_FAKE_YTDLP", False, raising=False)
        if treatment is not None:
            svc.fix_rtl_text_for_subtitles = treatment
        return svc

    @staticmethod
    def _write_srt(tmp_path, name, line):
        path = str(tmp_path / name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"1\n00:00:00,000 --> 00:00:02,000\n{line}\n\n")
        return path

    def _burn_and_grab(self, svc, black_clip, srt_path, out_path):
        assert svc.create_video_with_subtitles(
            black_clip, srt_path, out_path, target_language="he"
        )
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                "1",
                "-i",
                out_path,
                "-frames:v",
                "1",
                "-pix_fmt",
                "gray",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=120,
        ).stdout
        frame = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(self.H, self.W)
        return frame[int(self.H * 0.55) :, :]  # the strip the subtitle lives in

    # -- comparison -------------------------------------------------------------------

    @staticmethod
    def _bands(strip):
        """``(start, width)`` per inked column run, gaps under 4px glued shut."""
        cols = (strip > INK_THRESHOLD).any(axis=0)
        xs = numpy.flatnonzero(cols)
        if xs.size == 0:
            return []
        groups = numpy.split(xs, numpy.flatnonzero(numpy.diff(xs) > 4) + 1)
        return [(int(g[0]), int(g[-1] - g[0] + 1)) for g in groups]

    def _layouts_agree(self, strip_a, strip_b):
        bands_a, bands_b = self._bands(strip_a), self._bands(strip_b)
        if not bands_a or len(bands_a) != len(bands_b):
            return False
        for (xa, wa), (xb, wb) in zip(bands_a, bands_b, strict=True):
            if abs(wa - wb) > 2:
                return False
            width = min(wa, wb)
            best = None
            for dx in range(-3, 4):
                if xb + dx < 0 or xb + dx + width > strip_b.shape[1]:
                    continue
                diff = numpy.abs(
                    strip_a[:, xa : xa + width].astype(int)
                    - strip_b[:, xb + dx : xb + dx + width].astype(int)
                ).mean()
                best = diff if best is None else min(best, diff)
            if best is None or best > self.SAME_LAYOUT_MAD:
                return False
        return True

    # -- the tests --------------------------------------------------------------------

    @pytest.mark.parametrize(
        "logical,visual_tokens",
        CASES,
        ids=["parens", "number", "english-run"],
    )
    def test_the_default_burn_matches_a_hand_authored_visual_reference(
        self, tmp_path, monkeypatch, black_clip, logical, visual_tokens
    ):
        production = self._service(monkeypatch)
        prod_srt = str(tmp_path / "prod.srt")
        production.create_srt_file(
            [{"start": 0.0, "end": 2.0, "text": "x", "translated_text": logical}],
            prod_srt,
            use_translation=True,
            language="he",
        )
        strip_prod = self._burn_and_grab(
            production, black_clip, prod_srt, str(tmp_path / "prod.mp4")
        )

        reference_line = " ".join(
            self.RLI + token + self.PDI for token in visual_tokens
        )
        verbatim = self._service(monkeypatch, treatment=lambda t: t)
        ref_srt = self._write_srt(tmp_path, "ref.srt", reference_line)
        strip_ref = self._burn_and_grab(
            verbatim, black_clip, ref_srt, str(tmp_path / "ref.mp4")
        )

        assert self._layouts_agree(strip_prod, strip_ref), (
            f"production burn of {logical!r} does not match the hand-authored "
            f"visual order {visual_tokens!r}: bands "
            f"{self._bands(strip_prod)} vs {self._bands(strip_ref)}"
        )

    def test_the_old_rlo_treatment_fails_this_comparison(
        self, tmp_path, monkeypatch, black_clip
    ):
        """Positive control: the pre-fix treatment must NOT pass the band compare,
        or the comparison above proves nothing.
        """
        logical, visual_tokens = self.CASES[2]
        rlo = self._service(monkeypatch, treatment=lambda t: "\u202e" + t + "\u202c")
        srt = self._write_srt(tmp_path, "rlo.srt", logical)
        strip_rlo = self._burn_and_grab(rlo, black_clip, srt, str(tmp_path / "rlo.mp4"))

        reference_line = " ".join(
            self.RLI + token + self.PDI for token in visual_tokens
        )
        verbatim = self._service(monkeypatch, treatment=lambda t: t)
        ref_srt = self._write_srt(tmp_path, "ref.srt", reference_line)
        strip_ref = self._burn_and_grab(
            verbatim, black_clip, ref_srt, str(tmp_path / "ref.mp4")
        )

        assert not self._layouts_agree(strip_rlo, strip_ref)
