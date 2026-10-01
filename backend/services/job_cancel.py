"""
Stopping a running job — the "Stop" button.

Three levels, gentlest first:

1. Cooperative — the normal case. ``POST /cancel/<id>`` sets a Redis flag. Every
   progress update of the job checks it (yt-dlp's hook during a download, and each
   update of the transcription / translation / render steps) and raises
   :class:`JobCancelled`. The job stops between two pieces of work; its handler
   deletes the job's own partial files and reports ``CANCELLED``.
2. Soft signal — when the job is still running ``ESCALATE_AFTER_S`` later, it is
   inside one long call that reports no progress (an ffmpeg render, one big OpenAI
   request). Celery's revoke with SIGUSR1 raises ``SoftTimeLimitExceeded`` inside
   the task — the mechanism that ends a job at the 30-minute limit — and the same
   handlers run, seeing the flag and reporting ``CANCELLED``.
3. Kill — another ``ESCALATE_AFTER_S`` later. The process dies at once and cleans
   nothing, so the web process deletes the job's files itself: every file a job
   writes carries the head of its task id (:func:`remove_files_of`). Last resort.

A job still waiting in the queue is revoked without a signal and never starts.

``JobCancelled`` derives from ``BaseException``, not ``Exception``, on purpose: the
pipeline has broad ``except Exception`` blocks (per-batch retries, the research
recorder) that would catch a cancellation and carry on. Every task therefore
catches it explicitly at its top level.
"""

import glob
import os
import threading
import time

from config import get_config
from logging_config import get_logger
from services.task_registry import _redis

config = get_config()
logger = get_logger(__name__)

#: How long a stop request is remembered (the result itself lives as long).
CANCEL_TTL_S = config.CELERY_RESULT_EXPIRES

#: Wait before each escalation step (soft signal, then kill).
ESCALATE_AFTER_S = 30

#: Celery states of a job that has not finished yet.
ACTIVE_STATES = ("PENDING", "RECEIVED", "STARTED", "PROGRESS", "RETRY")

#: Progress updates come many times a second during a download; Redis is asked at
#: most this often per job.
_CHECK_EVERY_S = 0.5

#: After Redis fails to answer, stop asking for this long (a stop button that cannot
#: be read must not slow every progress update down to the socket timeout).
_BACKOFF_S = 30

CANCELLED_CODE = "CANCELLED"
CANCELLED_MESSAGE = "The job was stopped by the user."


class JobCancelled(BaseException):
    """The user pressed Stop on this job. See the module docstring for why BaseException."""

    def __init__(self, task_id: str):
        super().__init__(f"Job {task_id} was stopped by the user")
        self.task_id = task_id


def _key(task_id: str) -> str:
    return f"cancel_requested:{task_id}"


def request_cancel(task_id: str) -> None:
    """Record that the user asked to stop ``task_id``."""
    _redis().setex(_key(task_id), CANCEL_TTL_S, "1")


_last_check: dict[str, tuple[float, bool]] = {}
_redis_down_until = 0.0


def is_cancel_requested(task_id, fresh: bool = False) -> bool:
    """
    Has the user asked to stop ``task_id``? False when Redis cannot be reached — a
    job is never stopped by an unreadable flag.

    Progress updates answer from a short cache (they come many times a second).
    Error handlers pass ``fresh=True``: they decide whether a failure IS a stop —
    the soft signal of level 2 arrives as an exception — and a cached "no" from half
    a second before the press would file the stop as an ordinary error.
    """
    global _redis_down_until
    if not isinstance(task_id, str) or not task_id:
        return False
    now = time.monotonic()
    cached = _last_check.get(task_id)
    if not fresh and cached and now - cached[0] < _CHECK_EVERY_S:
        return cached[1]
    if not fresh and now < _redis_down_until:
        return False
    try:
        requested = bool(_redis().exists(_key(task_id)))
    except Exception as e:  # noqa: BLE001 - see the docstring
        logger.warning(f"Stop check unavailable ({e}); not asking for {_BACKOFF_S}s")
        _redis_down_until = now + _BACKOFF_S
        return False
    _last_check[task_id] = (now, requested)
    return requested


def raise_if_cancelled(task_id) -> None:
    """Raise :class:`JobCancelled` when the user asked to stop ``task_id``."""
    if is_cancel_requested(task_id):
        raise JobCancelled(task_id)


def cancelled_failure() -> dict:
    """The failure payload every task returns for a stopped job (/status reads it)."""
    return {
        "status": "FAILURE",
        "error": CANCELLED_MESSAGE,
        "code": CANCELLED_CODE,
        "message": CANCELLED_MESSAGE,
        "user_facing_message": CANCELLED_MESSAGE,
        # Starting the same job again is a perfectly good next step.
        "recoverable": True,
    }


#: How long after a kill the files are removed — the revoke is a message to the
#: worker, which then kills the process; nothing may still be writing.
_AFTER_KILL_S = 3


def remove_files_of(task_id: str) -> int:
    """
    Delete what a killed job left: in the working folder, ``<tag>_*`` (every download
    file, see youtube_service._job_tag); in the downloads folder, ``*_<tag>_*`` (the
    processing outputs, ``<title>_<tag>_original.srt`` and so on). ``tag`` is the
    first 8 hex digits of the task id — no other job's file carries it.
    """
    tag = task_id[:8]
    patterns = (
        os.path.join(glob.escape(config.FAST_WORK_DIR), f"{tag}_*"),
        os.path.join(glob.escape(config.DOWNLOADS_FOLDER), f"*_{tag}_*"),
    )
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                logger.warning(f"Could not remove {path} of killed job {task_id}: {e}")
    if removed:
        logger.info(f"🧹 Removed {removed} file(s) left by killed job {task_id}")
    return removed


def escalate_later(celery_app, task_id: str, delay_s: float = ESCALATE_AFTER_S):
    """
    Levels 2 and 3: a soft signal, then a kill, each only if the job is still active
    ``delay_s`` after the previous step. Runs on a daemon timer in the web process —
    the worker is busy with the very job being stopped (concurrency 1), so it cannot
    run a follow-up task of its own. Returns the first timer (tests cancel it).
    """

    def still_active() -> bool:
        from celery.result import AsyncResult

        return AsyncResult(task_id, app=celery_app).state in ACTIVE_STATES

    def step(signal_name: str, next_signal: str | None):
        try:
            if not still_active():
                return
            logger.warning(
                f"Job {task_id} did not stop within {delay_s}s of Stop; "
                f"sending {signal_name}"
            )
            celery_app.control.revoke(task_id, terminate=True, signal=signal_name)
            if next_signal:
                _start_timer(delay_s, step, next_signal, None)
            else:
                # The kill: the job will not clean up after itself.
                _start_timer(_AFTER_KILL_S, remove_files_of, task_id)
        except Exception as e:  # noqa: BLE001 - a timer thread must not die loudly
            logger.error(f"Stop escalation for {task_id} failed: {e}")

    return _start_timer(delay_s, step, "SIGUSR1", "SIGKILL")


def _start_timer(delay_s, fn, *args):
    timer = threading.Timer(delay_s, fn, args=args)
    timer.daemon = True
    timer.start()
    return timer
