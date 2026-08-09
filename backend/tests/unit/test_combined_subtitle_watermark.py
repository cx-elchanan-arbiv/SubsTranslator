"""
Unit tests for combined subtitle and watermark functionality
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.subtitle_service import SubtitleService


@pytest.mark.unit
class TestCombinedSubtitleWatermark:
    """Test the combined subtitle and watermark function."""

    def setup_method(self):
        """Setup test environment."""
        self.service = SubtitleService()
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.srt_path = os.path.join(self.temp_dir, "test.srt")
        self.watermark_path = os.path.join(self.temp_dir, "watermark.png")
        self.output_path = os.path.join(self.temp_dir, "output.mp4")

        # Create dummy files
        with open(self.video_path, "wb") as f:
            f.write(b"dummy video content")

        with open(self.srt_path, "w", encoding="utf-8") as f:
            f.write("""1
00:00:01,000 --> 00:00:03,000
Hello World

2
00:00:04,000 --> 00:00:06,000
Test subtitle
""")

        with open(self.watermark_path, "wb") as f:
            f.write(b"dummy image content")

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("services.subtitle_service.SubtitleService._run_ffmpeg_simple")
    def test_combined_function_single_ffmpeg_call(self, mock_run_ffmpeg):
        """Test that combined function uses single FFmpeg call."""
        # Setup mock to return success
        mock_run_ffmpeg.return_value = True

        # Create output file to simulate success
        with open(self.output_path, "wb") as f:
            f.write(b"output video")

        # Mock subprocess.run for ffprobe
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}}', returncode=0
            )

            # Call the combined function
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path,
                self.srt_path,
                self.output_path,
                self.watermark_path,
                target_language="en",
                watermark_position=("right", "bottom"),
                watermark_opacity=0.4,
                watermark_size_height=80,
            )

        # Verify success
        assert result is True

        # Verify FFmpeg was called only once
        assert mock_run_ffmpeg.call_count == 1

        # Get the command that was passed to _run_ffmpeg_simple
        ffmpeg_cmd = mock_run_ffmpeg.call_args[0][0]
        assert "ffmpeg" in ffmpeg_cmd
        assert "-filter_complex" in ffmpeg_cmd

        # Find filter_complex argument
        filter_idx = ffmpeg_cmd.index("-filter_complex") + 1
        filter_complex = ffmpeg_cmd[filter_idx]

        # Verify both subtitles and overlay are in the filter
        assert "subtitles=" in filter_complex
        assert "overlay=" in filter_complex
        assert "[vout]" in filter_complex

    @patch("services.subtitle_service.SubtitleService.create_video_with_subtitles")
    def test_fallback_when_watermark_missing(self, mock_create_video):
        """Test fallback to regular subtitle function when watermark is missing."""
        # Remove watermark file
        os.remove(self.watermark_path)

        # Setup mock
        mock_create_video.return_value = True

        # Call combined function
        result = self.service.create_video_with_subtitles_and_watermark(
            self.video_path,
            self.srt_path,
            self.output_path,
            self.watermark_path,
            target_language="en",
        )

        # Verify it fell back to regular function
        assert result is True
        mock_create_video.assert_called_once_with(
            self.video_path,
            self.srt_path,
            self.output_path,
            target_language="en",
            progress_callback=None,
            subtitle_position="bottom",
        )

    @patch("services.subtitle_service.SubtitleService._run_ffmpeg_simple")
    def test_rtl_language_support(self, mock_run_ffmpeg):
        """Test that RTL languages are handled properly."""
        # Create Hebrew SRT
        hebrew_srt = os.path.join(self.temp_dir, "hebrew.srt")
        with open(hebrew_srt, "w", encoding="utf-8") as f:
            f.write("""1
00:00:01,000 --> 00:00:03,000
שלום עולם

2
00:00:04,000 --> 00:00:06,000
בדיקה בעברית
""")

        # Setup mock
        mock_run_ffmpeg.return_value = True

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}}', returncode=0
            )

            # Create output file
            with open(self.output_path, "wb") as f:
                f.write(b"output")

            # Call with Hebrew
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path,
                hebrew_srt,
                self.output_path,
                self.watermark_path,
                target_language="he",
            )

            assert result is True

            # Verify RTL styling was applied
            ffmpeg_cmd = mock_run_ffmpeg.call_args[0][0]
            filter_idx = ffmpeg_cmd.index("-filter_complex") + 1
            filter_complex = ffmpeg_cmd[filter_idx]

            # Check for RTL-specific font settings
            assert "Noto Sans Hebrew" in filter_complex or "Hebrew" in filter_complex


@pytest.mark.unit
class TestFaststartAndMissingWatermarkParity:
    """Cross-path guarantees: all three renderers must agree on these two behaviours.

    ``create_video_with_ass`` (the ``render_v2`` path) was written after the two SRT
    renderers and got both of these right first. The SRT ones did not, and the difference
    is invisible in any single-path test — which is why these are parametrised over all
    three rather than asserted once.
    """

    def setup_method(self):
        self.service = SubtitleService()
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "video.mp4")
        self.srt_path = os.path.join(self.temp_dir, "subs.srt")
        self.watermark_path = os.path.join(self.temp_dir, "logo.png")
        self.output_path = os.path.join(self.temp_dir, "out.mp4")

        with open(self.video_path, "wb") as handle:
            handle.write(b"dummy video")
        with open(self.srt_path, "w", encoding="utf-8") as handle:
            handle.write("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n")
        with open(self.watermark_path, "wb") as handle:
            handle.write(b"dummy png")

        self.cues = [
            {"start": 1.0, "end": 3.0, "text": "Hello", "translated_text": "שלום"}
        ]

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- helpers -------------------------------------------------------------------
    def _invoke(self, path, watermark=None, subtitle_position="bottom"):
        """Run one renderer with ffmpeg stubbed; return the argv it built."""
        captured = {}

        def capture(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            with open(self.output_path, "wb") as handle:
                handle.write(b"out")
            return True

        with (
            patch.object(self.service, "_run_ffmpeg_simple", capture),
            patch.object(self.service, "_run_ffmpeg_with_progress", capture),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}, "streams": '
                '[{"width": 1280, "height": 720}]}',
                returncode=0,
            )
            if path == "srt":
                result = self.service.create_video_with_subtitles(
                    self.video_path,
                    self.srt_path,
                    self.output_path,
                    "he",
                    subtitle_position=subtitle_position,
                )
            elif path == "srt+watermark":
                result = self.service.create_video_with_subtitles_and_watermark(
                    self.video_path,
                    self.srt_path,
                    self.output_path,
                    watermark if watermark is not None else self.watermark_path,
                    target_language="he",
                    subtitle_position=subtitle_position,
                )
            elif path == "ass":
                result = self.service.create_video_with_ass(
                    self.video_path,
                    self.cues,
                    self.output_path,
                    target_language="he",
                    watermark_path=watermark,
                    subtitle_position=subtitle_position,
                )
            else:  # pragma: no cover - guard against a typo in a parametrisation
                raise AssertionError(f"unknown path {path}")
        return result, captured.get("cmd")

    # -- F12: faststart on every path ----------------------------------------------
    @pytest.mark.parametrize("path", ["srt", "srt+watermark", "ass"])
    def test_output_is_progressively_streamable(self, path):
        """``-movflags +faststart`` moves the moov atom to the front of the file.

        Without it the browser player must download the whole MP4 before it can show a
        single frame, which on a long render is the difference between "instant" and
        "thirty seconds of blank". The ASS path had it; the two SRT paths did not.
        """
        _result, cmd = self._invoke(path)
        assert cmd is not None, f"{path}: ffmpeg was never invoked"
        assert "-movflags" in cmd, f"{path}: no -movflags in {cmd}"
        assert (
            cmd[cmd.index("-movflags") + 1] == "+faststart"
        ), f"{path}: -movflags is set to something other than +faststart"

    @pytest.mark.parametrize("path", ["srt", "srt+watermark", "ass"])
    def test_faststart_precedes_the_output_path(self, path):
        """An ffmpeg output option after the output filename is silently ignored."""
        _result, cmd = self._invoke(path)
        assert cmd.index("-movflags") < cmd.index(self.output_path)

    @pytest.mark.parametrize(
        "position,alignment", [("bottom", "2"), ("top", "8"), ("side", "6")]
    )
    @pytest.mark.parametrize("path", ["srt", "srt+watermark", "ass"])
    def test_every_renderer_honours_the_subtitle_position(
        self, path, position, alignment
    ):
        """Upload/URL jobs may select either renderer; placement must match in all."""
        _result, cmd = self._invoke(path, subtitle_position=position)
        if path == "ass":
            ass_path = os.path.splitext(self.output_path)[0] + ".ass"
            with open(ass_path, encoding="utf-8") as handle:
                style = next(
                    line for line in handle if line.startswith("Style:")
                ).strip()
            fields = style.split(":", 1)[1].strip().split(",")
            assert fields[18] == alignment
        else:
            filter_flag = "-vf" if path == "srt" else "-filter_complex"
            filter_value = cmd[cmd.index(filter_flag) + 1]
            assert f"Alignment={alignment}" in filter_value

    # -- F13: a missing logo degrades, never fails ---------------------------------
    def test_legacy_path_renders_without_a_missing_watermark(self):
        """A deleted logo must cost the watermark, not the whole job."""
        os.remove(self.watermark_path)
        with patch.object(
            self.service, "create_video_with_subtitles", return_value=True
        ) as fallback:
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path,
                self.srt_path,
                self.output_path,
                self.watermark_path,
                target_language="he",
            )
        assert result is True, "a missing logo failed the render"
        assert fallback.called, "it did not fall back to the subtitles-only renderer"

    def test_legacy_path_survives_a_null_watermark_path(self):
        """``os.path.exists(None)`` raises TypeError rather than returning False.

        A null path reaches here whenever the watermark config is half-filled, so the
        guard has to test the path's truthiness before it touches the filesystem.
        """
        with patch.object(
            self.service, "create_video_with_subtitles", return_value=True
        ) as fallback:
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path,
                self.srt_path,
                self.output_path,
                None,
                target_language="he",
            )
        assert result is True
        assert fallback.called

    def test_ass_path_renders_without_a_missing_watermark(self):
        """Parity: the render_v2 path degrades the same way, in one pass."""
        os.remove(self.watermark_path)
        result, cmd = self._invoke("ass", watermark=self.watermark_path)
        assert result is True
        assert "overlay" not in " ".join(
            cmd
        ), "a missing logo still reached the filtergraph"

    def test_ass_path_still_overlays_a_watermark_that_exists(self):
        """The negative tests above would pass on a renderer that ignored watermarks."""
        result, cmd = self._invoke("ass", watermark=self.watermark_path)
        assert result is True
        assert self.watermark_path in cmd, "the logo was never passed to ffmpeg"
        assert "overlay" in " ".join(cmd), "no overlay filter was built"


# ======================================================================================
# P6 — automatic lower-third ("chyron") avoidance
# ======================================================================================
import numpy as np  # noqa: E402

from services.subtitle_engine import layout_params  # noqa: E402
from services.subtitle_service import (  # noqa: E402
    CHYRON_BUSY_THRESHOLD,
    CHYRON_MARGIN_V_FRAC,
    CHYRON_SAMPLE_POSITIONS,
    _edge_density,
)


@pytest.mark.unit
class TestEdgeDensity:
    """The busyness score, tested on synthetic arrays — no FFmpeg, no video."""

    def test_flat_band_scores_zero(self):
        assert _edge_density(np.full((40, 100), 128, dtype=np.uint8)) == 0.0

    def test_flat_black_band_scores_zero(self):
        """A fade to black must not read as busy — it is the opposite of busy."""
        assert _edge_density(np.zeros((40, 100), dtype=np.uint8)) == 0.0

    def test_dense_stripes_score_high(self):
        band = np.zeros((40, 100), dtype=np.uint8)
        band[:, ::2] = 255
        assert _edge_density(band) > 0.4

    def test_uint8_wraparound_does_not_hide_a_strong_edge(self):
        """5 - 250 wraps to 11 in uint8; the score must be computed in int16."""
        band = np.zeros((4, 4), dtype=np.uint8)
        band[:, :2] = 5
        band[:, 2:] = 250
        assert _edge_density(band) > 0.0

    def test_smooth_gradient_is_not_an_edge(self):
        """Skin, sky and bokeh vary smoothly and must not trip the detector."""
        band = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (40, 1))
        assert _edge_density(band) < 0.01

    def test_score_is_resolution_independent(self):
        """A ratio, so one threshold can serve 480p and 4K."""
        small = np.zeros((20, 50), dtype=np.uint8)
        small[:, ::2] = 255
        large = np.zeros((200, 500), dtype=np.uint8)
        large[:, ::2] = 255
        assert abs(_edge_density(small) - _edge_density(large)) < 0.02

    def test_empty_band_is_safe(self):
        assert _edge_density(np.zeros((1, 1), dtype=np.uint8)) == 0.0


@pytest.mark.unit
class TestDetectLowerThird:
    """The decision, with frame decoding stubbed so the logic is tested in isolation."""

    def setup_method(self):
        self.service = SubtitleService()
        self.layout = layout_params(1280, 720)

    def _stub(self, scores, duration=60.0):
        """Return a service whose frame grabs yield bands of the given edge densities."""
        service = self.service
        service._probe_duration = lambda _p: duration
        it = iter(scores)

        def grab(_path, _at, _x, _y, w, h):
            try:
                target = next(it)
            except StopIteration:
                return None
            if target is None:
                return None
            band = np.zeros((h, w), dtype=np.uint8)
            if target > 0:
                # Every Nth column white -> a predictable, tunable edge density.
                step = max(2, int(round(1.0 / target)))
                band[:, ::step] = 255
            return band

        service._grab_band = grab
        return service

    def test_busy_band_fires(self):
        service = self._stub([0.5] * 5)
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is True
        assert out["score"] > CHYRON_BUSY_THRESHOLD

    def test_clean_band_does_not_fire(self):
        service = self._stub([0.0] * 5)
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is False
        assert out["score"] == 0.0

    def test_the_score_survives_two_black_frames(self):
        """The dark-scene defence: fades among the samples must not veto a real chyron."""
        service = self._stub([0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5])
        assert service.detect_lower_third("v.mp4", self.layout)["busy"] is True

    def test_a_minority_of_busy_frames_now_fires(self):
        """DELIBERATE REVERSAL of the old median rule, and the reason for R3.

        A graphic that is only on screen for part of the video — a broadcast super, an
        intermittent breaking-news bar — leaves a MINORITY of samples busy, and a median
        is by definition blind to a minority. Measured: a corpus clip with 2 of 5 samples
        at 0.0556/0.0589 scored a median of 0.0073 and was called clear.
        """
        service = self._stub([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is True
        assert out["bands"]["subtitle"]["busy_samples"] == 2

    def test_one_busy_frame_among_seven_is_not_enough(self):
        service = self._stub([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert service.detect_lower_third("v.mp4", self.layout)["busy"] is False

    def test_too_few_decodable_frames_makes_no_decision(self):
        service = self._stub([0.5, None, None, None, None])
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is False
        assert "not enough to judge" in out["reason"]

    def test_unknown_duration_makes_no_decision(self):
        service = self._stub([0.5] * 5, duration=0.0)
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is False
        assert "duration unknown" in out["reason"]

    def test_exception_never_escapes(self):
        service = self.service
        service._probe_duration = lambda _p: 60.0

        def boom(*_a, **_k):
            raise RuntimeError("ffmpeg exploded")

        service._grab_band = boom
        out = service.detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is False
        assert "detection error" in out["reason"]

    def test_decision_is_deterministic(self):
        first = self._stub([0.5] * 5).detect_lower_third("v.mp4", self.layout)
        second = self._stub([0.5] * 5).detect_lower_third("v.mp4", self.layout)
        assert first["score"] == second["score"] and first["busy"] == second["busy"]

    def test_sample_positions_are_fixed_and_include_the_opening(self):
        """The first 10% is sampled now: broadcast supers live there and nowhere else."""
        assert CHYRON_SAMPLE_POSITIONS == (0.03, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9)
        assert sum(1 for p in CHYRON_SAMPLE_POSITIONS if p <= 0.1) >= 3

    def test_sample_timestamps_are_recorded(self):
        """Only the scores were archived, so a disputed verdict could not be re-checked."""
        out = self._stub([0.0] * 7, duration=100.0).detect_lower_third(
            "v.mp4", self.layout
        )
        assert out["sample_times"] == [3.0, 7.0, 10.0, 30.0, 50.0, 70.0, 90.0]

    def test_band_is_where_the_subtitle_box_would_be(self):
        service = self._stub([0.0] * 5)
        band = service.detect_lower_third("v.mp4", self.layout)["band"]
        bottom = 720 - self.layout["margin_v"]
        assert band["y"] + band["h"] == bottom
        assert band["x"] == self.layout["margin_h"]
        assert band["w"] == 1280 - 2 * self.layout["margin_h"]

    def test_result_always_has_every_key(self):
        for scores in ([0.5] * 7, [None] * 7, [0.0] * 7):
            out = self._stub(scores).detect_lower_third("v.mp4", self.layout)
            for key in (
                "busy",
                "score",
                "threshold",
                "samples",
                "sample_times",
                "band",
                "bands",
                "reason",
            ):
                assert key in out


@pytest.mark.unit
class TestTwoBandDetection:
    """R3: the band that mattered was the one the detector could not see.

    The sampled rectangle stops at ``height - margin_v``, so a chyron living in the
    bottom margin was invisible to it. Measured on a real BREAKING NEWS clip: the
    sampled band scored 0.0093 (verdict "clear") while the bar 12px lower scored 0.1146.
    """

    def setup_method(self):
        self.service = SubtitleService()
        self.layout = layout_params(1280, 720)

    def _stub_split(self, subtitle_score, bottom_score, duration=60.0):
        """A frame whose TOP half (the subtitle band) and BOTTOM strip differ."""
        service = self.service
        service._probe_duration = lambda _p: duration

        def fill(band, target):
            if target > 0:
                step = max(2, int(round(1.0 / target)))
                band[:, ::step] = 255

        def grab(_path, _at, _x, _y, w, h):
            band = np.zeros((h, w), dtype=np.uint8)
            split = int(round(self.layout["font_px"] * self.layout["max_lines"] * 1.35))
            split = min(split, h - 1)
            fill(band[:split], subtitle_score)
            fill(band[split:], bottom_score)
            return band

        service._grab_band = grab
        return service

    def test_a_busy_bottom_strip_alone_raises_the_box(self):
        out = self._stub_split(0.0, 0.5).detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is True
        assert out["decided_by"] == "bottom"
        assert out["bands"]["subtitle"]["busy"] is False
        assert out["bands"]["bottom"]["busy"] is True

    def test_a_busy_subtitle_band_alone_still_raises_the_box(self):
        out = self._stub_split(0.5, 0.0).detect_lower_third("v.mp4", self.layout)
        assert out["busy"] is True
        assert out["decided_by"] == "subtitle"

    def test_both_clear_stays_quiet(self):
        assert (
            self._stub_split(0.0, 0.0).detect_lower_third("v.mp4", self.layout)["busy"]
            is False
        )

    def test_the_bands_are_scored_separately_not_averaged(self):
        """Merging them dilutes a 0.1146 bar to 0.0450 and reproduces the original miss."""
        out = self._stub_split(0.0, 0.5).detect_lower_third("v.mp4", self.layout)
        subtitle = out["bands"]["subtitle"]["score"]
        bottom = out["bands"]["bottom"]["score"]
        assert subtitle < CHYRON_BUSY_THRESHOLD < bottom
        assert out["score"] == bottom

    def test_the_two_bands_tile_the_strip_with_no_gap_and_no_overlap(self):
        out = self._stub_split(0.0, 0.0).detect_lower_third("v.mp4", self.layout)
        subtitle, bottom = out["bands"]["subtitle"], out["bands"]["bottom"]
        assert subtitle["y"] + subtitle["h"] == bottom["y"]
        assert (
            bottom["y"] + bottom["h"] == 720
        ), "the bottom strip must reach the frame edge"
        assert subtitle["x"] == bottom["x"] and subtitle["w"] == bottom["w"]

    def test_one_decode_serves_both_bands(self):
        decodes = []
        service = self._stub_split(0.0, 0.5)
        original = service._grab_band

        def counting(*args, **kwargs):
            decodes.append(args[1])
            return original(*args, **kwargs)

        service._grab_band = counting
        service.detect_lower_third("v.mp4", self.layout)
        assert len(decodes) == len(CHYRON_SAMPLE_POSITIONS)


@pytest.mark.unit
class TestChyronRaisesTheBoxInTheRenderPath:
    """create_video_with_ass must move the box, and move ONLY the box."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video = os.path.join(self.temp_dir, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"x")
        self.out = os.path.join(self.temp_dir, "out.mp4")
        self.cues = [
            {"start": 0.0, "end": 3.0, "text": "שלום", "translated_text": "שלום"}
        ]

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _render(self, busy, **kwargs):
        service = SubtitleService()
        service.config.USE_FAKE_YTDLP = False
        layout = layout_params(1280, 720)
        decision = {
            "busy": busy,
            "score": 0.09 if busy else 0.01,
            "threshold": CHYRON_BUSY_THRESHOLD,
            "samples": [],
            "band": None,
            "reason": "stub",
        }
        service.detect_lower_third = lambda *_a, **_k: decision
        with (
            patch("subprocess.Popen") as popen,
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1024),
        ):
            proc = MagicMock()
            proc.stdout.readline.side_effect = [""]
            proc.poll.return_value = 0
            proc.returncode = 0
            proc.communicate.return_value = ("", "")
            popen.return_value = proc
            service.create_video_with_ass(
                self.video,
                self.cues,
                self.out,
                target_language="he",
                layout=layout,
                **kwargs,
            )
        ass_path = os.path.splitext(self.out)[0] + ".ass"
        with open(ass_path, encoding="utf-8") as f:
            content = f.read()
        style = [l for l in content.splitlines() if l.startswith("Style:")][0]
        return content, style.split(",")

    def test_busy_band_raises_margin_v(self):
        _, clean = self._render(False)
        _, raised = self._render(True)
        assert int(raised[21]) > int(clean[21])
        assert int(raised[21]) == round(720 * CHYRON_MARGIN_V_FRAC)

    def test_raising_the_box_does_not_change_the_text_budget(self):
        """margin_v is the only thing that moves; font and line budget are width-derived."""
        _, clean = self._render(False)
        _, raised = self._render(True)
        assert raised[2] == clean[2]  # Fontsize
        assert raised[19] == clean[19]  # MarginL
        assert raised[20] == clean[20]  # MarginR

    def test_detection_can_be_switched_off_for_deterministic_tests(self):
        _, default = self._render(False)
        _, disabled = self._render(False, detect_lower_third=False)
        assert default[21] == disabled[21]

    def test_decision_is_handed_to_the_recorder(self):
        recorder = Mock()
        self._render(True, recorder=recorder)
        recorder.update_meta.assert_called_once()
        assert recorder.update_meta.call_args.kwargs["lower_third"]["busy"] is True

    def test_a_recorder_that_raises_cannot_fail_the_render(self):
        recorder = Mock()
        recorder.update_meta.side_effect = RuntimeError("archive is on fire")
        content, _ = self._render(True, recorder=recorder)
        assert "Dialogue:" in content
