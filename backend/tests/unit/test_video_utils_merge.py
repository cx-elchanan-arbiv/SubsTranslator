"""
Unit tests for video_utils.merge_videos_ffmpeg().

Tests the merge logic without actually running FFmpeg.
Uses monkeypatch to mock subprocess.run.
"""

import os

# Import video_utils
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))
import utils.video_utils as video_utils


@pytest.mark.unit
def test_merge_fast_concat_success(tmp_path, monkeypatch):
    """Test successful merge using fast concat demuxer method."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Verify fast concat method is being used
        assert "-f" in cmd
        assert "concat" in cmd
        assert "-c" in cmd
        assert "copy" in cmd

        # Create output file
        output_path.write_bytes(b"\x00" * 2048)  # 2KB file

        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("v1.mp4", "v2.mp4", str(output_path))

    assert result is True
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


@pytest.mark.unit
def test_merge_fallback_after_fast_fail(tmp_path, monkeypatch):
    """Test fallback to re-encode method when fast concat fails."""
    output_path = tmp_path / "output.mp4"
    call_count = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        call_count["n"] += 1

        if call_count["n"] == 1:
            # First call (fast concat) fails
            return SimpleNamespace(returncode=1, stdout="", stderr="different codecs")
        else:
            # Second call (re-encode fallback) succeeds
            # Verify re-encode method is being used
            assert "-filter_complex" in cmd
            assert "scale" in " ".join(cmd)
            assert "concat" in " ".join(cmd)

            # Create output file
            output_path.write_bytes(b"\x00" * 4096)

            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("a.mp4", "b.mp4", str(output_path))

    assert result is True
    assert output_path.exists()
    assert call_count["n"] == 2  # Two attempts made


@pytest.mark.unit
def test_merge_timeout(tmp_path, monkeypatch):
    """Test handling of FFmpeg timeout."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise video_utils.subprocess.TimeoutExpired("ffmpeg", timeout=600)

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("x.mp4", "y.mp4", str(output_path))

    assert result is False


@pytest.mark.unit
def test_merge_too_small_output(tmp_path, monkeypatch):
    """Test rejection of suspiciously small output files."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Create tiny file (less than 1KB)
        output_path.write_bytes(b"tiny")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("x.mp4", "y.mp4", str(output_path))

    assert result is False


@pytest.mark.unit
def test_merge_output_not_created(tmp_path, monkeypatch):
    """Test failure when output file is not created."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Don't create output file
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("x.mp4", "y.mp4", str(output_path))

    assert result is False


@pytest.mark.unit
def test_merge_both_methods_fail(tmp_path, monkeypatch):
    """Test failure when both concat methods fail."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Both methods fail
        return SimpleNamespace(returncode=1, stdout="", stderr="error")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("x.mp4", "y.mp4", str(output_path))

    assert result is False


@pytest.mark.unit
def test_merge_concat_list_cleanup(tmp_path, monkeypatch):
    """Test that temporary concat list file is cleaned up."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        # Fast concat succeeds
        output_path.write_bytes(b"\x00" * 2048)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("v1.mp4", "v2.mp4", str(output_path))

    assert result is True

    # Verify concat list file was cleaned up
    concat_list = str(output_path) + ".concat.txt"
    assert not os.path.exists(concat_list)


@pytest.mark.unit
def test_merge_exception_handling(tmp_path, monkeypatch):
    """Test handling of unexpected exceptions."""
    output_path = tmp_path / "output.mp4"

    def fake_run(cmd, capture_output=True, text=True, timeout=None, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(video_utils.subprocess, "run", fake_run)

    result = video_utils.merge_videos_ffmpeg("x.mp4", "y.mp4", str(output_path))

    assert result is False
