"""The structured-logging helpers, asserted on their output.

What this file replaces: seventeen tests that called a helper and then asserted either
nothing at all or ``logger is not None``. They could not fail. Two of them tested
structlog's own argument binding rather than any code in this repo, and the file carried
no ``unit`` mark, so CI never ran a line of it either way.

Every test below reads the captured event dict, because the event dict IS the product —
it is what lands in the log aggregator and what an operator greps during an incident.
"""

import pytest
import structlog

from logging_config import (
    TaskContext,
    add_correlation_ids,
    get_logger,
    log_external_service_call,
    log_task_complete,
    log_task_error,
    log_task_start,
)


@pytest.fixture
def logger():
    return get_logger("test")


@pytest.mark.unit
class TestTaskLifecycleHelpers:
    """``tasks/processing_tasks.py`` reports every task through these three."""

    def test_task_start_records_the_task_name(self, logger):
        with structlog.testing.capture_logs() as captured:
            log_task_start(logger, "process_video", video_id="abc")

        assert captured == [
            {
                "event": "Task started",
                "log_level": "info",
                "task_name": "process_video",
                "video_id": "abc",
            }
        ]

    def test_task_complete_rounds_the_duration(self, logger):
        """Unrounded floats make log lines unreadable and grep-hostile."""
        with structlog.testing.capture_logs() as captured:
            log_task_complete(logger, "process_video", duration=1.23456789)

        assert captured[0]["duration_seconds"] == 1.235

    def test_task_complete_omits_duration_when_it_was_not_measured(self, logger):
        with structlog.testing.capture_logs() as captured:
            log_task_complete(logger, "process_video")

        assert "duration_seconds" not in captured[0]

    def test_task_error_captures_the_exception_type_and_message(self, logger):
        """The type is what you group by; the message is what you read."""
        with structlog.testing.capture_logs() as captured:
            log_task_error(logger, "process_video", ValueError("bad input"))

        assert captured[0]["log_level"] == "error"
        assert captured[0]["error_type"] == "ValueError"
        assert captured[0]["error_message"] == "bad input"


@pytest.mark.unit
class TestExternalServiceCall:
    """Used by ``services/subtitle_service.py`` around ffmpeg invocations."""

    def test_a_successful_call_logs_at_info(self, logger):
        with structlog.testing.capture_logs() as captured:
            log_external_service_call(logger, "ffmpeg", "watermark", success=True)

        assert captured[0]["log_level"] == "info"
        assert captured[0]["service"] == "ffmpeg"
        assert captured[0]["operation"] == "watermark"

    def test_a_failed_call_is_promoted_to_warning(self, logger):
        """The severity switch is the only branch in this helper — pin it."""
        with structlog.testing.capture_logs() as captured:
            log_external_service_call(logger, "openai", "translate", success=False)

        assert captured[0]["log_level"] == "warning"
        assert captured[0]["success"] is False

    def test_duration_is_reported_in_milliseconds(self, logger):
        with structlog.testing.capture_logs() as captured:
            log_external_service_call(
                logger, "ffmpeg", "encode", success=True, duration=1.5
            )

        assert captured[0]["duration_ms"] == 1500.0


@pytest.mark.unit
class TestCorrelationIds:
    """``add_correlation_ids`` sits in the live processor chain (``setup_logging``)."""

    def test_a_task_id_is_stamped_onto_events_inside_the_context(self):
        with TaskContext("task-123", "video_processing"):
            event = add_correlation_ids(None, "info", {"event": "x"})
            assert event["task_id"] == "task-123"

    def test_the_task_id_is_gone_again_after_the_context_exits(self):
        """A leaked contextvar would mislabel every later task in the same worker."""
        with TaskContext("task-123", "video_processing"):
            pass

        assert "task_id" not in add_correlation_ids(None, "info", {"event": "x"})

    def test_a_user_id_is_carried_when_one_was_supplied(self):
        with TaskContext("task-456", "subtitles", "user-789"):
            event = add_correlation_ids(None, "info", {"event": "x"})
            assert event["user_id"] == "user-789"

    def test_nothing_is_added_when_no_context_is_active(self):
        assert add_correlation_ids(None, "info", {"event": "x"}) == {"event": "x"}
