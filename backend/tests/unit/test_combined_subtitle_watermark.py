"""
Unit tests for combined subtitle and watermark functionality
"""
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from services.subtitle_service import SubtitleService


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
        with open(self.video_path, 'wb') as f:
            f.write(b'dummy video content')
        
        with open(self.srt_path, 'w', encoding='utf-8') as f:
            f.write("""1
00:00:01,000 --> 00:00:03,000
Hello World

2
00:00:04,000 --> 00:00:06,000
Test subtitle
""")
        
        with open(self.watermark_path, 'wb') as f:
            f.write(b'dummy image content')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('services.subtitle_service.SubtitleService._run_ffmpeg_simple')
    def test_combined_function_single_ffmpeg_call(self, mock_run_ffmpeg):
        """Test that combined function uses single FFmpeg call."""
        # Setup mock to return success
        mock_run_ffmpeg.return_value = True
        
        # Create output file to simulate success
        with open(self.output_path, 'wb') as f:
            f.write(b'output video')
        
        # Mock subprocess.run for ffprobe
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}}',
                returncode=0
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
                watermark_size_height=80
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
    
    @patch('services.subtitle_service.SubtitleService.create_video_with_subtitles')
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
            target_language="en"
        )
        
        # Verify it fell back to regular function
        assert result is True
        mock_create_video.assert_called_once_with(
            self.video_path,
            self.srt_path,
            self.output_path,
            "en",
            None  # progress_callback
        )
    
    @patch('services.subtitle_service.SubtitleService._run_ffmpeg_simple')
    def test_rtl_language_support(self, mock_run_ffmpeg):
        """Test that RTL languages are handled properly."""
        # Create Hebrew SRT
        hebrew_srt = os.path.join(self.temp_dir, "hebrew.srt")
        with open(hebrew_srt, 'w', encoding='utf-8') as f:
            f.write("""1
00:00:01,000 --> 00:00:03,000
שלום עולם

2
00:00:04,000 --> 00:00:06,000
בדיקה בעברית
""")
        
        # Setup mock
        mock_run_ffmpeg.return_value = True
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}}',
                returncode=0
            )
            
            # Create output file
            with open(self.output_path, 'wb') as f:
                f.write(b'output')
            
            # Call with Hebrew
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path,
                hebrew_srt,
                self.output_path,
                self.watermark_path,
                target_language="he"
            )
            
            assert result is True
            
            # Verify RTL styling was applied
            ffmpeg_cmd = mock_run_ffmpeg.call_args[0][0]
            filter_idx = ffmpeg_cmd.index("-filter_complex") + 1
            filter_complex = ffmpeg_cmd[filter_idx]
            
            # Check for RTL-specific font settings
            assert "Noto Sans Hebrew" in filter_complex or "Hebrew" in filter_complex


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

        self.cues = [{"start": 1.0, "end": 3.0, "text": "Hello", "translated_text": "שלום"}]

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -- helpers -------------------------------------------------------------------
    def _invoke(self, path, watermark=None):
        """Run one renderer with ffmpeg stubbed; return the argv it built."""
        captured = {}

        def capture(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            with open(self.output_path, "wb") as handle:
                handle.write(b"out")
            return True

        with patch.object(self.service, "_run_ffmpeg_simple", capture), patch.object(
            self.service, "_run_ffmpeg_with_progress", capture
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout='{"format": {"duration": "60.0"}, "streams": '
                       '[{"width": 1280, "height": 720}]}',
                returncode=0,
            )
            if path == "srt":
                result = self.service.create_video_with_subtitles(
                    self.video_path, self.srt_path, self.output_path, "he"
                )
            elif path == "srt+watermark":
                result = self.service.create_video_with_subtitles_and_watermark(
                    self.video_path, self.srt_path, self.output_path,
                    watermark if watermark is not None else self.watermark_path,
                    target_language="he",
                )
            elif path == "ass":
                result = self.service.create_video_with_ass(
                    self.video_path, self.cues, self.output_path,
                    target_language="he",
                    watermark_path=watermark,
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
        assert cmd[cmd.index("-movflags") + 1] == "+faststart", (
            f"{path}: -movflags is set to something other than +faststart"
        )

    @pytest.mark.parametrize("path", ["srt", "srt+watermark", "ass"])
    def test_faststart_precedes_the_output_path(self, path):
        """An ffmpeg output option after the output filename is silently ignored."""
        _result, cmd = self._invoke(path)
        assert cmd.index("-movflags") < cmd.index(self.output_path)

    # -- F13: a missing logo degrades, never fails ---------------------------------
    def test_legacy_path_renders_without_a_missing_watermark(self):
        """A deleted logo must cost the watermark, not the whole job."""
        os.remove(self.watermark_path)
        with patch.object(
            self.service, "create_video_with_subtitles", return_value=True
        ) as fallback:
            result = self.service.create_video_with_subtitles_and_watermark(
                self.video_path, self.srt_path, self.output_path,
                self.watermark_path, target_language="he",
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
                self.video_path, self.srt_path, self.output_path,
                None, target_language="he",
            )
        assert result is True
        assert fallback.called

    def test_ass_path_renders_without_a_missing_watermark(self):
        """Parity: the render_v2 path degrades the same way, in one pass."""
        os.remove(self.watermark_path)
        result, cmd = self._invoke("ass", watermark=self.watermark_path)
        assert result is True
        assert "overlay" not in " ".join(cmd), "a missing logo still reached the filtergraph"

    def test_ass_path_still_overlays_a_watermark_that_exists(self):
        """The negative tests above would pass on a renderer that ignored watermarks."""
        result, cmd = self._invoke("ass", watermark=self.watermark_path)
        assert result is True
        assert self.watermark_path in cmd, "the logo was never passed to ffmpeg"
        assert "overlay" in " ".join(cmd), "no overlay filter was built"
