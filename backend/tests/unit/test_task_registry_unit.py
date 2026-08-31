"""Bug #8: a PENDING that Celery cannot vouch for is a 404, not a spinner.

Celery answers PENDING for task ids it has never seen, so an expired job and a
made-up id look exactly like a queued one. A browser that remembered a dead
task id polled that PENDING forever — an eternal "processing 0%" (reproduced
live on a day-old ?task= link).

The fix has two halves, and each test pins one:
  * every published task leaves a Redis marker with the result's own TTL
    (before_task_publish hook in celery_worker)
  * /status answers 404 + TASK_UNKNOWN for PENDING without a marker, which is
    the signal the frontend already treats as "clean up and go home"
"""

import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

os.environ["TESTING"] = "true"
os.environ.setdefault("DISABLE_RATE_LIMIT", "1")

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

pytestmark = pytest.mark.unit


@pytest.fixture
def flask_client():
    from app import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _pending_async_result():
    mock = MagicMock()
    mock.state = "PENDING"
    mock.result = None
    return mock


class TestStatusForUnknownTasks:
    def test_pending_without_a_marker_is_404(self, flask_client):
        from api import video_routes

        with (
            patch.object(
                video_routes, "AsyncResult", return_value=_pending_async_result()
            ),
            patch.object(video_routes.task_registry, "is_known", return_value=False),
        ):
            response = flask_client.get("/status/00000000-dead-dead-dead-000000000000")

        assert response.status_code == 404
        data = response.get_json()
        assert data["state"] == "UNKNOWN"
        assert data["error"]["code"] == "TASK_UNKNOWN"
        assert data["error"]["recoverable"] is False

    def test_pending_with_a_marker_stays_pending(self, flask_client):
        """A genuinely queued task must NOT be told it is gone."""
        from api import video_routes

        with (
            patch.object(
                video_routes, "AsyncResult", return_value=_pending_async_result()
            ),
            patch.object(video_routes.task_registry, "is_known", return_value=True),
        ):
            response = flask_client.get("/status/11111111-live-live-live-111111111111")

        assert response.status_code == 200
        assert response.get_json()["state"] == "PENDING"

    def test_non_pending_states_never_consult_the_registry(self, flask_client):
        """SUCCESS/FAILURE speak for themselves — the registry must not be able
        to shadow a real result (e.g. after a Redis hiccup)."""
        from api import video_routes

        done = MagicMock()
        done.state = "SUCCESS"
        done.result = {"status": "SUCCESS", "result": {"title": "t", "files": {}}}

        with (
            patch.object(video_routes, "AsyncResult", return_value=done),
            patch.object(
                video_routes.task_registry, "is_known", side_effect=AssertionError
            ),
        ):
            response = flask_client.get("/status/22222222-done-done-done-222222222222")

        assert response.status_code == 200


class TestTheRegistryItself:
    def test_mark_and_lookup_roundtrip_with_ttl(self):
        """Against the real Redis: the marker exists and expires on its own."""
        from services import task_registry

        task_id = f"registry-test-{uuid.uuid4()}"
        try:
            task_registry.mark_submitted(task_id)
        except Exception:
            pytest.skip("no Redis available in this environment")

        client = task_registry._redis()
        try:
            client.ping()
        except Exception:
            pytest.skip("no Redis available in this environment")

        assert task_registry.is_known(task_id) is True
        ttl = client.ttl(task_registry._key(task_id))
        # TTL is the result-retention window — while a result can exist, so can
        # the marker; afterwards PENDING legitimately means "unknown".
        from config import get_config

        assert 0 < ttl <= get_config().CELERY_RESULT_EXPIRES
        client.delete(task_registry._key(task_id))

    def test_lookup_fails_open_when_redis_is_down(self):
        """A dead registry must never 404 a live task."""
        from services import task_registry

        broken = MagicMock()
        broken.exists.side_effect = ConnectionError("redis is gone")
        with patch.object(task_registry, "_redis", return_value=broken):
            assert task_registry.is_known("whatever") is True

    def test_the_publish_hook_is_connected(self):
        """The marker only exists if something writes it. This pins the wiring:
        if the before_task_publish receiver is ever dropped, every task becomes
        'unknown' one result-TTL later and /status 404s live history."""
        from celery.signals import before_task_publish

        import celery_worker  # noqa: F401  (registers the receiver on import)

        receiver_names = [
            getattr(r, "__name__", "")
            for holder in before_task_publish.receivers
            for r in [holder[1]() if callable(holder[1]) else holder[1]]
            if r is not None
        ]
        assert "_mark_task_submitted" in receiver_names
