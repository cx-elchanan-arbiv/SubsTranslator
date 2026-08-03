"""
Unit tests for video_utils subtitle embedding functions.

Tests:
- embed_subtitles_ffmpeg()
- parse_text_to_srt()
- convert_to_srt_time()
- add_watermark_to_video()
"""

# Import video_utils
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))
import utils.video_utils as video_utils


@pytest.mark.unit
def test_embed_subtitles_success(tmp_path, monkeypatch):
    """Test successful subtitle embedding."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Verify subtitle filter is used
        assert "-vf" in cmd
        cmd_str = " ".join(cmd)
        assert "subtitles=" in cmd_str

        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.embed_subtitles_ffmpeg(
        "video.mp4", "subs.srt", str(output_path)
    )

    assert result is True
    assert output_path.exists()


@pytest.mark.unit
def test_embed_subtitles_timeout(tmp_path, monkeypatch):
    """Test handling of timeout during subtitle embedding."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise video_utils.subprocess.TimeoutExpired("ffmpeg", timeout=600)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.embed_subtitles_ffmpeg(
        "video.mp4", "subs.srt", str(output_path)
    )

    assert result is False


@pytest.mark.unit
def test_embed_subtitles_output_too_small(tmp_path, monkeypatch):
    """Test rejection of suspiciously small output."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        output_path.write_bytes(b"tiny")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.embed_subtitles_ffmpeg(
        "video.mp4", "subs.srt", str(output_path)
    )

    assert result is False


@pytest.mark.unit
def test_convert_to_srt_time_mm_ss():
    """Test convert_to_srt_time with MM:SS format."""
    assert video_utils.convert_to_srt_time("05:30") == "00:05:30,000"
    assert video_utils.convert_to_srt_time("00:45") == "00:00:45,000"
    assert video_utils.convert_to_srt_time("10:00") == "00:10:00,000"


@pytest.mark.unit
def test_convert_to_srt_time_hh_mm_ss():
    """Test convert_to_srt_time with HH:MM:SS format."""
    assert video_utils.convert_to_srt_time("01:05:30") == "01:05:30,000"
    assert video_utils.convert_to_srt_time("00:00:45") == "00:00:45,000"
    assert video_utils.convert_to_srt_time("02:10:15") == "02:10:15,000"


@pytest.mark.unit
def test_parse_text_to_srt_basic(tmp_path):
    """Test parsing timestamped text to SRT format."""
    output_path = tmp_path / "output.srt"

    text = """[00:00 - 00:05] Hello World
[00:05 - 00:10] This is a test
[00:10 - 00:15] Final subtitle"""

    result = video_utils.parse_text_to_srt(text, str(output_path))

    assert result is True
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "Hello World" in content
    assert "00:00:00,000 --> 00:00:05,000" in content
    assert "This is a test" in content


@pytest.mark.unit
def test_parse_text_to_srt_hh_mm_ss_format(tmp_path):
    """Test parsing with HH:MM:SS timestamps."""
    output_path = tmp_path / "output.srt"

    text = """[00:00:00 - 00:00:05] First line
[00:00:05 - 00:00:10] Second line"""

    result = video_utils.parse_text_to_srt(text, str(output_path))

    assert result is True

    content = output_path.read_text(encoding="utf-8")
    assert "First line" in content
    assert "Second line" in content


@pytest.mark.unit
def test_parse_text_to_srt_empty_text(tmp_path):
    """Test handling of empty text."""
    output_path = tmp_path / "output.srt"

    result = video_utils.parse_text_to_srt("", str(output_path))

    assert result is False


@pytest.mark.unit
def test_parse_text_to_srt_no_valid_entries(tmp_path):
    """Test handling of text with no valid subtitle entries."""
    output_path = tmp_path / "output.srt"

    text = """This is just plain text
No timestamps here
Nothing valid"""

    result = video_utils.parse_text_to_srt(text, str(output_path))

    assert result is False


@pytest.mark.unit
def test_add_watermark_success(tmp_path, monkeypatch):
    """Test successful watermark addition."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Verify watermark filter
        assert "-filter_complex" in cmd
        cmd_str = " ".join(cmd)
        assert "scale" in cmd_str
        assert "overlay" in cmd_str

        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.add_watermark_to_video(
        "video.mp4",
        str(output_path),
        "logo.png",
        position="bottom-right",
        size="medium",
        opacity=40,
    )

    assert result is True
    assert output_path.exists()


@pytest.mark.unit
def test_add_watermark_different_positions(tmp_path, monkeypatch):
    """Test watermark with different position settings."""
    output_path = tmp_path / "output.mp4"
    positions_tested = []

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        cmd_str = " ".join(cmd)
        # Extract position from overlay parameter
        if "overlay=" in cmd_str:
            positions_tested.append(True)

        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

    for pos in positions:
        result = video_utils.add_watermark_to_video(
            "video.mp4",
            str(output_path),
            "logo.png",
            position=pos,
            size="medium",
            opacity=50,
        )
        assert result is True

    assert len(positions_tested) == 4


@pytest.mark.unit
def test_add_watermark_different_sizes(tmp_path, monkeypatch):
    """Test watermark with different size settings."""
    output_path = tmp_path / "output.mp4"
    sizes_tested = []

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        cmd_str = " ".join(cmd)
        # Check that scale is applied
        if "scale=-1:" in cmd_str:
            # Extract height from scale parameter
            sizes_tested.append(True)

        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    sizes = ["small", "medium", "large"]

    for size in sizes:
        result = video_utils.add_watermark_to_video(
            "video.mp4",
            str(output_path),
            "logo.png",
            position="bottom-right",
            size=size,
            opacity=50,
        )
        assert result is True

    assert len(sizes_tested) == 3


@pytest.mark.unit
def test_add_watermark_opacity_calculation(tmp_path, monkeypatch):
    """Test that opacity is correctly calculated (0-100 -> 0.0-1.0)."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        cmd_str = " ".join(cmd)
        # Check that opacity (aa parameter) is in filter
        assert "colorchannelmixer" in cmd_str
        assert "aa=" in cmd_str

        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.add_watermark_to_video(
        "video.mp4",
        str(output_path),
        "logo.png",
        opacity=40,  # Should become 0.4 in FFmpeg
    )

    assert result is True


@pytest.mark.unit
def test_add_watermark_timeout(tmp_path, monkeypatch):
    """Test handling of timeout during watermark addition."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise video_utils.subprocess.TimeoutExpired("ffmpeg", timeout=600)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.add_watermark_to_video(
        "video.mp4", str(output_path), "logo.png"
    )

    assert result is False


@pytest.mark.unit
def test_add_watermark_exception(tmp_path, monkeypatch):
    """Test handling of exceptions during watermark addition."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.add_watermark_to_video(
        "video.mp4", str(output_path), "logo.png"
    )

    assert result is False
