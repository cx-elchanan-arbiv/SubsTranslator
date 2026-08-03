"""
Unit tests for video_utils.cut_video_ffmpeg().

Tests video cutting logic without actually running FFmpeg.
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
def test_cut_method1_success(tmp_path, monkeypatch):
    """Test successful cut using Method 1 (fast copy)."""
    output_path = tmp_path / "cut.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Verify Method 1 parameters
        assert "-ss" in cmd
        assert "-t" in cmd
        assert "-c" in cmd
        assert "copy" in cmd

        # Create output file
        output_path.write_bytes(b"\x00" * 2048)

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4", str(output_path), "00:00:10", "00:00:20"
    )

    assert result is True
    assert output_path.exists()


@pytest.mark.unit
def test_cut_fallback_to_method2(tmp_path, monkeypatch):
    """Test fallback to Method 2 (filter-complex) when Method 1 fails."""
    output_path = tmp_path / "cut.mp4"
    call_count = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        call_count["n"] += 1

        if call_count["n"] == 1:
            # Method 1 fails
            return SimpleNamespace(returncode=1, stdout="", stderr="error")
        else:
            # Method 2 succeeds
            assert "-vf" in cmd
            assert "trim" in " ".join(cmd)
            assert "-af" in cmd
            assert "atrim" in " ".join(cmd)

            output_path.write_bytes(b"\x00" * 4096)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4", str(output_path), "00:00:05", "00:00:15"
    )

    assert result is True
    assert call_count["n"] == 2


@pytest.mark.unit
def test_cut_invalid_time_range(tmp_path, monkeypatch):
    """An end before the start is rejected WITHOUT spawning ffmpeg.

    The old version of this test never patched ``subprocess.run``: real ffmpeg ran
    against a nonexistent ``input.mp4``, failed, and returned False anyway — so deleting
    the ``duration <= 0`` guard entirely still left it green. Asserting that ffmpeg was
    never invoked is the whole point of an early guard.
    """
    output_path = tmp_path / "cut.mp4"
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise AssertionError("ffmpeg must not be spawned for an invalid time range")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4",
        str(output_path),
        "00:00:20",  # Start
        "00:00:10",  # End (before start)
    )

    assert result is False
    assert calls == []


@pytest.mark.unit
def test_cut_output_too_small(tmp_path, monkeypatch):
    """Test rejection of suspiciously small output."""
    output_path = tmp_path / "cut.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        output_path.write_bytes(b"small")  # Less than 1KB
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4", str(output_path), "00:00:00", "00:00:05"
    )

    assert result is False


@pytest.mark.unit
def test_time_to_seconds_hh_mm_ss():
    """Test time_to_seconds with HH:MM:SS format."""
    assert video_utils.time_to_seconds("01:30:45") == 5445.0  # 1h + 30m + 45s
    assert video_utils.time_to_seconds("00:05:30") == 330.0
    assert video_utils.time_to_seconds("02:00:00") == 7200.0


@pytest.mark.unit
def test_time_to_seconds_mm_ss():
    """Test time_to_seconds with MM:SS format."""
    assert video_utils.time_to_seconds("05:30") == 330.0
    assert video_utils.time_to_seconds("00:45") == 45.0
    assert video_utils.time_to_seconds("10:00") == 600.0


@pytest.mark.unit
def test_time_to_seconds_ss():
    """Test time_to_seconds with SS format."""
    assert video_utils.time_to_seconds("45") == 45.0
    assert video_utils.time_to_seconds("120") == 120.0
    assert video_utils.time_to_seconds("0") == 0.0


@pytest.mark.unit
def test_cut_both_methods_fail(tmp_path, monkeypatch):
    """Test failure when both cutting methods fail."""
    output_path = tmp_path / "cut.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="error")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4", str(output_path), "00:00:00", "00:00:10"
    )

    assert result is False


@pytest.mark.unit
def test_cut_exception_handling(tmp_path, monkeypatch):
    """Test handling of unexpected exceptions."""
    output_path = tmp_path / "cut.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.cut_video_ffmpeg(
        "input.mp4", str(output_path), "00:00:00", "00:00:10"
    )

    assert result is False
