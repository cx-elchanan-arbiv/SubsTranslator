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
    captured = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        captured["cmd"] = list(cmd)
        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.embed_subtitles_ffmpeg(
        "video.mp4", "subs.srt", str(output_path)
    )

    assert result is True
    assert output_path.exists()

    # Asserted out here, not inside the stub: an `assert` in the stub is swallowed by
    # the function's own `except Exception` and resurfaces only as `result is False`.
    cmd = captured["cmd"]
    assert "-vf" in cmd
    assert "subtitles=subs.srt" in cmd

    # WhatsApp and most players will not show a preview without the moov atom up front.
    # This regressed once already; it is asserted on every render path.
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"


@pytest.mark.unit
def test_embed_subtitles_escapes_the_colon_in_a_subtitle_path(tmp_path, monkeypatch):
    """``:`` separates filter arguments, so an unescaped one truncates the filename.

    This is the only non-trivial line in the function and nothing covered it.
    """
    output_path = tmp_path / "output.mp4"
    captured = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        captured["cmd"] = list(cmd)
        output_path.write_bytes(b"\x00" * 4096)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    video_utils.embed_subtitles_ffmpeg(
        "video.mp4", r"C:\subs\out.srt", str(output_path)
    )

    assert "subtitles=C\\:/subs/out.srt" in captured["cmd"]


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
    """HH:MM:SS input must survive the round-trip into SRT timing lines.

    The assertions used to be ``"First line" in content`` only — which the MM:SS test
    already covers. The conversion this test exists for was never checked.
    """
    output_path = tmp_path / "output.srt"

    text = """[01:02:03 - 01:02:08] First line
[01:02:08 - 01:02:13] Second line"""

    result = video_utils.parse_text_to_srt(text, str(output_path))

    assert result is True

    content = output_path.read_text(encoding="utf-8")
    assert "01:02:03,000 --> 01:02:08,000" in content
    assert "01:02:08,000 --> 01:02:13,000" in content
    assert "First line" in content
    assert "Second line" in content


@pytest.mark.unit
def test_parse_text_to_srt_no_valid_entries(tmp_path):
    """Test handling of text with no valid subtitle entries."""
    output_path = tmp_path / "output.srt"

    text = """This is just plain text
No timestamps here
Nothing valid"""

    result = video_utils.parse_text_to_srt(text, str(output_path))

    assert result is False


@pytest.fixture
def watermark_cmd(tmp_path, monkeypatch):
    """Run add_watermark_to_video and hand back the ffmpeg argv it built.

    The three tests below used to append a bare ``True`` to a list whenever the word
    "overlay=" / "scale=-1:" appeared, then assert the list's length. Collapsing
    ``position_map`` or ``size_map`` in the source to a single constant left all of them
    green. They now read the filtergraph.
    """
    output_path = tmp_path / "output.mp4"

    def run(**kwargs):
        captured = {}

        def fake_run(cmd, capture_output=True, text=True, timeout=None, **_):
            captured["cmd"] = list(cmd)
            output_path.write_bytes(b"\x00" * 4096)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(video_utils.subprocess, "run", fake_run)
        assert video_utils.add_watermark_to_video(
            "video.mp4", str(output_path), "logo.png", **kwargs
        )
        cmd = captured["cmd"]
        return cmd[cmd.index("-filter_complex") + 1], cmd

    return run


@pytest.mark.unit
def test_add_watermark_success(watermark_cmd):
    filtergraph, cmd = watermark_cmd(position="bottom-right", size="medium", opacity=40)

    assert "scale=-1:120" in filtergraph
    assert "overlay=main_w-overlay_w-10:main_h-overlay_h-10" in filtergraph
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"


@pytest.mark.unit
@pytest.mark.parametrize(
    "position,expected",
    [
        ("top-left", "10:10"),
        ("top-right", "main_w-overlay_w-10:10"),
        ("bottom-left", "10:main_h-overlay_h-10"),
        ("bottom-right", "main_w-overlay_w-10:main_h-overlay_h-10"),
        ("nonsense", "main_w-overlay_w-10:10"),
    ],
)
def test_add_watermark_position_maps_to_overlay_coordinates(
    watermark_cmd, position, expected
):
    """Including the fallback: an unknown position must not crash the render."""
    filtergraph, _ = watermark_cmd(position=position, size="medium", opacity=50)

    assert f"overlay={expected}" in filtergraph


@pytest.mark.unit
@pytest.mark.parametrize(
    "size,expected_height",
    [("small", 80), ("medium", 120), ("large", 200), ("nonsense", 120)],
)
def test_add_watermark_size_maps_to_logo_height(watermark_cmd, size, expected_height):
    filtergraph, _ = watermark_cmd(position="bottom-right", size=size, opacity=50)

    assert f"scale=-1:{expected_height}" in filtergraph


@pytest.mark.unit
@pytest.mark.parametrize(
    "opacity,expected", [(0, "aa=0.0"), (40, "aa=0.4"), (100, "aa=1.0")]
)
def test_add_watermark_converts_percent_opacity_to_a_fraction(
    watermark_cmd, opacity, expected
):
    """0-100 from the UI becomes 0.0-1.0 for colorchannelmixer."""
    filtergraph, _ = watermark_cmd(
        position="bottom-right", size="medium", opacity=opacity
    )

    assert expected in filtergraph


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
