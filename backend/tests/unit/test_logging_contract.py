"""The logging call itself must never be able to crash the caller.

`setup_logging()` used to configure ``structlog.BoundLogger``, whose bound methods
reject any positional argument after the message::

    logger.info("took %ss", elapsed)
    TypeError: _proxy_to_logger() takes from 2 to 3 positional arguments but 4 were given

Percent-style is the stdlib interface and this backend uses it in services and
tasks, so that configuration turned a diagnostic line into a crash of whatever
request reached it. It stayed invisible because ``setup_logging`` is called by
``app.py`` alone: the Celery worker ran structlog's permissive defaults while the
API ran the strict config, so the same modules executed under two different
logging contracts and only one of them was armed.

These tests pin the contract itself — every documented call shape reaches the
renderer — rather than the wrapper class that currently provides it.
"""

import logging
import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from logging_config import get_logger, setup_logging  # noqa: E402


@pytest.fixture()
def configured():
    """Apply the real production configuration, then restore structlog's default."""
    import structlog

    setup_logging()
    yield
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


@pytest.mark.unit
class TestLoggingCallShapes:
    def test_percent_style_does_not_raise(self, configured):
        get_logger("probe").info("took %ss for %d items", 1.5, 3)

    def test_percent_style_on_every_level(self, configured):
        log = get_logger("probe")
        for level in ("debug", "info", "warning", "error", "critical"):
            getattr(log, level)("value %s", level)

    def test_structured_keywords_still_work(self, configured):
        get_logger("probe").warning("something happened", task_id="abc", count=2)

    def test_message_only_still_works(self, configured):
        get_logger("probe").info("plain message")

    def test_both_styles_in_one_call(self, configured):
        get_logger("probe").info("took %ss", 0.5, task_id="abc")

    def test_json_mode_accepts_percent_style(self):
        import structlog

        try:
            setup_logging(json_logs=True)
            get_logger("probe").info("took %ss", 0.5)
        finally:
            structlog.reset_defaults()
            logging.getLogger().handlers.clear()

    def test_the_interpolation_actually_happens(self, configured, capsys):
        """Not merely 'does not raise' — the value must reach the output."""
        get_logger("probe").info("elapsed %s seconds", 42)
        captured = capsys.readouterr()
        assert "elapsed 42 seconds" in captured.out + captured.err
