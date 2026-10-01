"""The Stop button (services.job_cancel), from the flag to the job's last word.

What each piece must guarantee:

* the flag: read cheaply (cached), and a Redis that does not answer never stops a job;
* the progress managers: the next update after Stop raises JobCancelled — once, so
  the job's own handler can still log and report;
* a stopped download deletes its own partial files and nothing else;
* the render loop lets the stop through (it used to swallow it in a bare
  ``except:``) and kills ffmpeg instead of leaving it running;
* every task answers a stop with the CANCELLED failure, also when the stop arrives
  as the soft signal (SoftTimeLimitExceeded);
* the routes: /cancel refuses finished or unknown jobs, revokes only queued ones,
  always arranges the escalation; /status reports REVOKED as a stop.
"""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services import job_cancel  # noqa: E402
from services.job_cancel import JobCancelled  # noqa: E402

pytestmark = pytest.mark.unit

TASK_ID = "11111111-2222-3333-4444-555555555555"


class FakeRedis:
    def __init__(self):
        self.keys = set()
        self.calls = 0

    def setex(self, key, ttl, value):
        self.keys.add(key)

    def exists(self, key):
        self.calls += 1
        return int(key in self.keys)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(job_cancel, "_redis", lambda: fake)
    monkeypatch.setattr(job_cancel, "_last_check", {})
    monkeypatch.setattr(job_cancel, "_redis_down_until", 0.0)
    return fake


# --------------------------------------------------------------------------- the flag


class TestFlag:
    def test_not_requested_by_default(self, redis):
        assert job_cancel.is_cancel_requested(TASK_ID) is False

    def test_requested_after_stop(self, redis):
        job_cancel.request_cancel(TASK_ID)
        with pytest.raises(JobCancelled):
            job_cancel.raise_if_cancelled(TASK_ID)

    def test_a_cancellation_is_not_an_ordinary_exception(self):
        """Broad `except Exception` blocks in the pipeline must not swallow it."""
        assert not issubclass(JobCancelled, Exception)

    def test_redis_is_asked_at_most_twice_a_second(self, redis):
        for _ in range(50):
            job_cancel.is_cancel_requested(TASK_ID)
        assert redis.calls == 1

    def test_a_handler_reads_past_the_cache(self, redis):
        """Level 2 arrives as an exception a moment after the press; the handler
        deciding "stop or error?" must not get the cached "no" (seen live:
        a stopped download reported as DOWNLOAD_ERROR)."""
        assert job_cancel.is_cancel_requested(TASK_ID) is False  # cached "no"
        job_cancel.request_cancel(TASK_ID)
        assert job_cancel.is_cancel_requested(TASK_ID) is False  # still cached
        assert job_cancel.is_cancel_requested(TASK_ID, fresh=True) is True

    def test_an_unreadable_flag_never_stops_a_job(self, monkeypatch, redis):
        def broken():
            raise ConnectionError("redis is down")

        monkeypatch.setattr(job_cancel, "_redis", broken)
        assert job_cancel.is_cancel_requested(TASK_ID) is False

    @pytest.mark.parametrize("task_id", [None, "", MagicMock()])
    def test_no_task_id_is_never_cancelled(self, redis, task_id):
        assert job_cancel.is_cancel_requested(task_id) is False


# --------------------------------------------------------------------------- managers


def _task():
    task = MagicMock()
    task.request.id = TASK_ID
    return task


@pytest.mark.parametrize("which", ["progress_manager", "state_manager"])
def test_the_next_update_after_stop_raises_once(redis, which):
    if which == "progress_manager":
        from tasks.progress_manager import ProgressManager

        manager = ProgressManager(_task(), [{"label": "Downloading", "weight": 1.0}])
    else:
        from state_manager import EnterpriseStateManager

        manager = EnterpriseStateManager(_task(), [{"label": "Downloading"}])

    manager.set_step_progress(0, 10)  # no stop yet
    job_cancel.request_cancel(TASK_ID)
    job_cancel._last_check.clear()

    with pytest.raises(JobCancelled):
        manager.set_step_progress(0, 20)
    # The job's own handler logs and reports after this — it must not raise again.
    manager.log("stopping")
    manager.set_step_progress(0, 30)


# --------------------------------------------------------------------------- download


class StoppedYDL:
    """Writes partial files, then reports progress — which is where Stop lands."""

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        return {"id": "v", "title": "Clip", "ext": "mp4"}

    def process_ie_result(self, info, download=True):
        base = self.opts["outtmpl"].replace("%(title).100B.%(ext)s", "Clip")
        for suffix in (".f137.mp4.part", ".f137.mp4.ytdl"):
            with open(base + suffix, "wb") as handle:
                handle.write(b"partial")
        for hook in self.opts["progress_hooks"]:
            hook(
                {
                    "status": "downloading",
                    "info_dict": {"format_id": "137"},
                    "fragment_index": 1,
                    "fragment_count": 10,
                }
            )
        raise AssertionError("the stop should have ended the download")


def test_a_stopped_download_deletes_its_own_files_only(monkeypatch, tmp_path, redis):
    from services import youtube_service
    from state_manager import EnterpriseStateManager

    work = tmp_path / "work"
    work.mkdir()
    other = work / "99999999_Other.f137.mp4.part"
    other.write_bytes(b"another job")
    monkeypatch.setattr(youtube_service.yt_dlp, "YoutubeDL", StoppedYDL)
    monkeypatch.setattr(
        youtube_service,
        "config",
        MagicMock(USE_FAKE_YTDLP=False, FAST_WORK_DIR=str(work), DEBUG=False),
    )
    monkeypatch.setattr(youtube_service, "DOWNLOADS_FOLDER", str(tmp_path))

    manager = EnterpriseStateManager(_task(), [{"label": "Downloading"}])
    job_cancel.request_cancel(TASK_ID)
    job_cancel._last_check.clear()  # the flag is re-read at most every 0.5 s

    with pytest.raises(JobCancelled):
        youtube_service.download_youtube_video_with_progress(
            "https://example.com/v", progress_manager=manager, job_id=TASK_ID
        )
    assert sorted(os.listdir(work)) == [other.name]


# --------------------------------------------------------------------------- render


def test_a_stopped_render_kills_ffmpeg(tmp_path):
    """The render loop's progress update used to sit in a bare `except:`."""
    from services.subtitle_service import subtitle_service

    video = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=2",
            str(video),
        ],
        check=True,
    )
    # Stands in for ffmpeg: reports progress on stderr forever.
    fake_ffmpeg = [
        "sh",
        "-c",
        "while true; do echo 'frame=1 time=00:00:01.00 bitrate=1' >&2; sleep 0.05; done",
    ]
    spawned = []
    real_popen = subprocess.Popen

    def recording_popen(args, *rest, **kwargs):
        process = real_popen(args, *rest, **kwargs)
        if args == fake_ffmpeg:  # not the ffprobe call that precedes it
            spawned.append(process)
        return process

    def stop(_percent):
        raise JobCancelled(TASK_ID)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", recording_popen)
        with pytest.raises(JobCancelled):
            subtitle_service._run_ffmpeg_with_progress(fake_ffmpeg, str(video), stop)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None, "ffmpeg outlived the stop"


# --------------------------------------------------------------------------- tasks


@pytest.fixture
def no_result_backend(monkeypatch):
    """Run the task body in-process without writing state for a task id of None."""
    from tasks import download_tasks

    monkeypatch.setattr(
        download_tasks.download_youtube_only_task, "update_state", MagicMock()
    )


def test_the_download_only_task_reports_a_stop(monkeypatch, redis, no_result_backend):
    from services.metadata_service import metadata_service
    from tasks import download_tasks

    monkeypatch.setattr(
        metadata_service,
        "extract_metadata",
        MagicMock(side_effect=JobCancelled(TASK_ID)),
    )
    result = download_tasks.download_youtube_only_task.run("https://example.com/v")
    assert result["code"] == "CANCELLED"
    assert result["status"] == "FAILURE"


def test_the_soft_signal_of_a_stop_is_still_a_stop(
    monkeypatch, redis, no_result_backend
):
    """Level 2 arrives as SoftTimeLimitExceeded; with the flag set it is a stop."""
    from billiard.exceptions import SoftTimeLimitExceeded

    from services.metadata_service import metadata_service
    from tasks import download_tasks

    monkeypatch.setattr(
        metadata_service,
        "extract_metadata",
        MagicMock(return_value=(MagicMock(), None)),
    )
    monkeypatch.setattr(
        download_tasks,
        "download_youtube_video_with_progress",
        MagicMock(side_effect=SoftTimeLimitExceeded()),
    )
    monkeypatch.setattr(
        download_tasks, "is_cancel_requested", lambda _id, fresh=False: True
    )
    result = download_tasks.download_youtube_only_task.run("https://example.com/v")
    assert result["code"] == "CANCELLED"


def test_a_stopped_processing_job_deletes_its_own_outputs(monkeypatch, tmp_path):
    from tasks import processing_tasks

    mine = "Clip_11111111"
    for name in (f"{mine}_original.srt", f"{mine}_with_subtitles.mp4"):
        (tmp_path / name).write_bytes(b"partial")
    (tmp_path / "Clip.mp4").write_bytes(b"the source")
    (tmp_path / "Clip_22222222_original.srt").write_bytes(b"another job")
    monkeypatch.setattr(processing_tasks, "DOWNLOADS_FOLDER", str(tmp_path))

    manager = MagicMock(steps=[{"status": "in_progress"}])
    recorder = MagicMock()
    result = processing_tasks._stop_job(manager, recorder, mine)

    assert result["code"] == "CANCELLED"
    assert sorted(os.listdir(tmp_path)) == ["Clip.mp4", "Clip_22222222_original.srt"]
    manager.acknowledge_cancel.assert_called_once()
    recorder.finish.assert_called_once()


# --------------------------------------------------------------------------- routes


@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def cancel_route(monkeypatch, redis):
    from api import video_routes

    state = {"value": "PROGRESS"}
    result = MagicMock()
    type(result).state = property(lambda _self: state["value"])
    monkeypatch.setattr(video_routes, "AsyncResult", lambda *_a, **_k: result)
    monkeypatch.setattr(video_routes.task_registry, "is_known", lambda _id: True)
    escalate = MagicMock()
    monkeypatch.setattr(video_routes, "escalate_later", escalate)
    app = MagicMock()
    monkeypatch.setattr(video_routes, "process_video_task", MagicMock(app=app))
    return {"state": state, "escalate": escalate, "celery": app}


class TestCancelRoute:
    def test_a_running_job_gets_the_flag_and_the_escalation(
        self, flask_client, cancel_route, redis
    ):
        response = flask_client.post(f"/cancel/{TASK_ID}")
        assert response.status_code == 202
        assert job_cancel.is_cancel_requested(TASK_ID)
        cancel_route["escalate"].assert_called_once()
        cancel_route["celery"].control.revoke.assert_not_called()

    def test_a_queued_job_is_revoked(self, flask_client, cancel_route):
        cancel_route["state"]["value"] = "PENDING"
        assert flask_client.post(f"/cancel/{TASK_ID}").status_code == 202
        cancel_route["celery"].control.revoke.assert_called_once_with(TASK_ID)

    @pytest.mark.parametrize("finished", ["SUCCESS", "FAILURE", "REVOKED"])
    def test_a_finished_job_is_left_alone(self, flask_client, cancel_route, finished):
        cancel_route["state"]["value"] = finished
        assert flask_client.post(f"/cancel/{TASK_ID}").status_code == 409
        cancel_route["escalate"].assert_not_called()

    def test_an_invalid_id_is_refused(self, flask_client, cancel_route):
        assert flask_client.post("/cancel/not-a-task").status_code == 400


def test_status_reports_a_revoked_job_as_stopped(flask_client, cancel_route):
    cancel_route["state"]["value"] = "REVOKED"
    body = flask_client.get(f"/status/{TASK_ID}").get_json()
    assert body["state"] == "FAILURE"
    assert body["error"]["code"] == "CANCELLED"


def test_the_escalation_waits_and_skips_a_job_that_stopped(monkeypatch):
    """Level 2 is sent only to a job still active after the wait."""
    import celery.result

    app = MagicMock()
    state = {"value": "PROGRESS"}
    result = MagicMock()
    type(result).state = property(lambda _self: state["value"])
    monkeypatch.setattr(celery.result, "AsyncResult", lambda *_a, **_k: result)

    timer = job_cancel.escalate_later(app, TASK_ID, delay_s=0.05)
    timer.join(1)
    app.control.revoke.assert_called_once_with(
        TASK_ID, terminate=True, signal="SIGUSR1"
    )

    state["value"] = "SUCCESS"  # it stopped after the soft signal: no kill
    time.sleep(0.2)
    assert app.control.revoke.call_count == 1


def test_a_killed_jobs_files_are_removed_and_no_one_elses(monkeypatch, tmp_path):
    """Level 3 kills the process, which cleans nothing; the web process does."""
    work, downloads = tmp_path / "work", tmp_path / "downloads"
    work.mkdir()
    downloads.mkdir()
    tag = TASK_ID[:8]
    mine = [
        work / f"{tag}_Clip.f137.mp4.part",
        work / f"{tag}_Clip.f137.mp4.ytdl",
        downloads / f"Clip_{tag}_original.srt",
    ]
    others = [
        work / "99999999_Other.f137.mp4.part",
        downloads / "Clip_99999999_original.srt",
        downloads / "Clip.mp4",
    ]
    for path in mine + others:
        path.write_bytes(b"x")
    monkeypatch.setattr(
        job_cancel,
        "config",
        MagicMock(FAST_WORK_DIR=str(work), DOWNLOADS_FOLDER=str(downloads)),
    )

    assert job_cancel.remove_files_of(TASK_ID) == 3
    assert all(not p.exists() for p in mine)
    assert all(p.exists() for p in others)
