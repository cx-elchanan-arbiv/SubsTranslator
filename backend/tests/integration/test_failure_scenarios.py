"""How the pipeline behaves when a dependency fails underneath it.

Each test here pins a decision that was made deliberately and would otherwise be easy to
"simplify" away: which failures are swallowed, which are surfaced, and which must never
be papered over with a silent substitution.
"""

import os
import subprocess
import tempfile
from unittest.mock import Mock, patch

import pytest


@pytest.mark.integration
class TestFailureScenarios:
    def test_openai_failure_is_surfaced_not_silently_downgraded(
        self, stub_openai_rate_limiter, monkeypatch
    ):
        """The user picked OpenAI; a rate-limited OpenAI must raise, never fall back.

        ``translation_services`` says so twice, in as many words: "Don't fallback to
        Google when OpenAI was explicitly selected". Falling back would hand the user
        machine-translated text under the label of the model they paid for — a silent
        quality downgrade is worse than a visible failure. The fixture grants budget and
        no-ops the backoff sleeps; without it this test spends 210 seconds waiting on a
        Redis that is not there.
        """
        from openai import RateLimitError

        from services import translation_services
        from services.translation_services import get_translator

        # The OpenAI client is fully mocked below; the constructor still refuses to
        # build without a configured key, so an environment without one (CI) failed
        # here instead of exercising the rate-limit path. A placeholder keeps the
        # test measuring what it claims to measure and never reaches the network.
        monkeypatch.setattr(
            translation_services.config, "OPENAI_API_KEY", "test-key-not-used"
        )

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = RateLimitError(
                "Rate limit exceeded",
                response=Mock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            )

            translator = get_translator("openai")

            with pytest.raises(Exception, match="OpenAI translation failed"):
                translator.translate_batch(["Hello", "World"], "he")

    def test_a_full_disk_is_not_swallowed_by_srt_writing(self):
        """ENOSPC must reach the caller so the task fails loudly and can be retried.

        ``create_srt_file`` logs and re-raises. Downgrading that to a swallowed error
        would hand the pipeline a path to a truncated .srt and burn it into the video.
        The failure is injected at ``open``: patching ``os.write`` does not work, because
        buffered text IO does its writing below the Python ``os`` namespace — which is
        why the previous version of this test could never actually fail.
        """
        from services.subtitle_service import create_srt_file

        real_open = open

        class FullDisk:
            def __init__(self, fh):
                self._fh = fh

            def write(self, _data):
                raise OSError(28, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._fh.close()
                return False

        def open_onto_a_full_disk(path, *args, **kwargs):
            fh = real_open(path, *args, **kwargs)
            return FullDisk(fh) if str(path).endswith(".srt") else fh

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test.srt")
            segments = [{"start": 0, "end": 1, "text": "A" * 1000}]

            with patch("builtins.open", open_onto_a_full_disk):
                with pytest.raises(OSError) as excinfo:
                    create_srt_file(segments, temp_file, use_translation=False)

            assert excinfo.value.errno == 28
            assert os.path.getsize(temp_file) == 0

    def test_a_broker_outage_during_progress_reporting_does_not_kill_the_task(self):
        """``_update_celery_state`` swallows on purpose — progress is not the payload.

        If publishing progress could raise, a blip on the result backend would abort a
        transcode that had otherwise succeeded.
        """
        from state_manager import EnterpriseStateManager

        mock_task = Mock()
        mock_task.update_state.side_effect = Exception("Redis connection failed")
        manager = EnterpriseStateManager(mock_task, [{"label": "Step", "weight": 1.0}])

        manager.set_step_progress(0, 50, "halfway")

        # It failed to publish, and it kept the local step state anyway.
        assert mock_task.update_state.called
        assert manager.steps[0].progress == 50

    def test_an_unavailable_video_becomes_a_structured_youtube_error(
        self, tmp_path, monkeypatch
    ):
        """Raw yt-dlp text is mapped to a typed error the API layer can classify.

        The work/output directories are redirected into ``tmp_path``: the defaults are
        absolute container paths, so on any other machine this test used to fail on
        ``PermissionError: '/app'`` before it ever reached the behaviour it exists to
        check — a test that passes only inside one container is not a test.
        """
        from core.exceptions import YouTubeAccessError
        from services import youtube_service
        from services.youtube_service import download_youtube_video

        monkeypatch.setattr(
            youtube_service.config,
            "FAST_WORK_DIR",
            str(tmp_path / "work"),
            raising=False,
        )
        monkeypatch.setattr(
            youtube_service, "DOWNLOADS_FOLDER", str(tmp_path / "out"), raising=False
        )

        with patch("yt_dlp.YoutubeDL") as mock_ytdl:
            mock_instance = Mock()
            mock_ytdl.return_value.__enter__ = Mock(return_value=mock_instance)
            mock_ytdl.return_value.__exit__ = Mock(return_value=False)
            mock_instance.extract_info.side_effect = Exception("Video unavailable")
            mock_instance.download.side_effect = Exception("Video unavailable")

            with pytest.raises(YouTubeAccessError):
                download_youtube_video("https://www.youtube.com/watch?v=invalid")

    def test_a_killed_ffmpeg_returns_false_rather_than_exploding(self):
        from services.subtitle_service import SubtitleService

        service = SubtitleService()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                -9, ["ffmpeg"], "Process killed (SIGKILL)"
            )

            result = service.create_video_with_subtitles(
                "input.mp4", "subtitles.srt", "output.mp4", "he"
            )

            assert result is False

    def test_undecodable_bytes_do_not_cost_the_surrounding_subtitles(self):
        """One bad segment must not take the whole .srt down with it."""
        from services.subtitle_service import create_srt_file

        segments = [
            {"start": 0, "end": 1, "text": "Valid text"},
            {"start": 1, "end": 2, "text": "Invalid: \xff\xfe"},
            {"start": 2, "end": 3, "text": "More valid text"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "out.srt")
            create_srt_file(segments, temp_file, use_translation=False)

            with open(temp_file, encoding="utf-8", errors="replace") as fh:
                content = fh.read()

            assert "Valid text" in content
            assert "More valid text" in content


@pytest.mark.integration
class TestConcurrencyIssues:
    def test_concurrent_progress_updates_do_not_corrupt_step_state(self):
        """``EnterpriseStateManager`` guards its steps with an RLock; prove it holds."""
        import threading

        from state_manager import EnterpriseStateManager

        mock_task = Mock()
        manager = EnterpriseStateManager(
            mock_task, [{"label": "Step 1", "weight": 1.0}]
        )
        # Construction publishes the initial PENDING state; count only what follows.
        published_at_start = mock_task.update_state.call_count

        threads = [
            threading.Thread(
                target=manager.set_step_progress, args=(0, i * 10, f"p{i}")
            )
            for i in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert mock_task.update_state.call_count - published_at_start == 10
        # Whichever writer landed last, the value must be one that was actually written
        # — never a torn read or an out-of-range value.
        assert manager.steps[0].progress in {i * 10 for i in range(10)}
