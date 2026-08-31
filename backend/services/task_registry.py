"""Which task ids were actually published — so PENDING can mean something.

Celery answers PENDING for ids it has NEVER SEEN: an expired result, a purged
job and a made-up id are indistinguishable from a genuinely queued task. The
status route needs the difference, because a browser that remembers a dead
task id polls PENDING forever and shows an eternal "processing 0%" spinner
(bug #8 — reproduced live on a day-old ?task= link).

Every published task gets a Redis marker (see the ``before_task_publish`` hook
in ``celery_worker.py``) with the same TTL as the task result itself
(``CELERY_RESULT_EXPIRES``). While either the queue entry or the result can
still exist, the marker exists too; once it is gone, PENDING means "unknown"
and ``/status`` says so with a 404 instead of an eternal maybe.

Fail-open by design: if Redis cannot be reached, ``is_known`` answers True —
a spinner on a dead task is a smaller lie than a 404 on a live one.
"""

import redis as redis_lib

from config import get_config
from logging_config import get_logger

config = get_config()
logger = get_logger(__name__)

_client = None


def _redis():
    global _client
    if _client is None:
        _client = redis_lib.Redis.from_url(
            config.REDIS_URL,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    return _client


def _key(task_id: str) -> str:
    return f"task_submitted:{task_id}"


def mark_submitted(task_id: str) -> None:
    """Record that ``task_id`` was really handed to the queue."""
    try:
        _redis().setex(_key(task_id), config.CELERY_RESULT_EXPIRES, "1")
    except Exception as e:  # noqa: BLE001 - a marker must never block a publish
        logger.warning(f"task registry: could not mark {task_id}: {e}")


def is_known(task_id: str) -> bool:
    """Was ``task_id`` published within the result-retention window?"""
    try:
        return _redis().exists(_key(task_id)) > 0
    except Exception as e:  # noqa: BLE001 - fail open, see module docstring
        logger.warning(
            f"task registry: lookup failed for {task_id}: {e} (failing open)"
        )
        return True
