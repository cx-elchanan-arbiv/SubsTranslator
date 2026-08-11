"""
Structured logging configuration for SubsTranslator.
Provides consistent logging with correlation IDs and structured data.
"""

import logging
import sys
from contextvars import ContextVar

import structlog

# Context variables for task correlation, populated by TaskContext.
task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def add_correlation_ids(logger, method_name, event_dict):
    """Add correlation IDs to log entries."""
    task_id = task_id_var.get()
    user_id = user_id_var.get()

    if task_id:
        event_dict["task_id"] = task_id
    if user_id:
        event_dict["user_id"] = user_id

    return event_dict


def setup_logging(level: str = "INFO", testing: bool = False, json_logs: bool = False):
    """Setup clean, structured logging configuration."""

    # Clean console format for better readability
    console_format = "%(asctime)s [%(levelname)-7s] %(name)-12s: %(message)s"

    # Configure standard library logging with clean format
    logging.basicConfig(
        format=console_format,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Reduce noise from external libraries
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)

    # Configure processors
    processors = [
        add_correlation_ids,
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        # Interpolates `logger.info("took %ss", n)`. See the wrapper_class note in
        # structlog.configure() below — the two go together.
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if testing:
        # For testing, use simple format
        processors.append(structlog.testing.LogCapture())
    elif json_logs:
        # For production, use JSON format
        processors.append(structlog.processors.JSONRenderer())
    else:
        # For development, use colored console output
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        # `structlog.stdlib.BoundLogger`, not `structlog.BoundLogger`, and this is a
        # correctness fix rather than a preference. Percent-style logging —
        # `logger.info("took %ss", n)` — is the stdlib interface and this backend
        # uses it in services and tasks. The generic BoundLogger refuses any
        # positional argument after the message and raises TypeError
        # ("_proxy_to_logger() takes from 2 to 3 positional arguments but 4 were
        # given") *from the log line itself*, i.e. it turns a diagnostic into a
        # crash of whatever request reached it. The stdlib wrapper moves those
        # arguments into ``positional_args`` for the formatter above instead.
        #
        # Why nobody noticed: `setup_logging` is called only by ``app.py``, so the
        # Celery worker ran on structlog's defaults and the API ran on this config.
        # The two processes were executing the same modules under two different
        # logging contracts, and only one of them was armed.
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class TaskContext:
    """Context manager for task correlation."""

    def __init__(self, task_id: str, task_type: str = None, user_id: str = None):
        self.task_id = task_id
        self.task_type = task_type
        self.user_id = user_id
        self.token = None
        self.user_token = None

    def __enter__(self):
        self.token = task_id_var.set(self.task_id)
        if self.user_id:
            # Keep the reset token. Setting this without one used to leak the user_id
            # into every LATER task handled by the same worker context, so one user's
            # id got stamped onto another user's log lines.
            self.user_token = user_id_var.set(self.user_id)

        # Also bind to structlog context
        structlog.contextvars.bind_contextvars(
            task_id=self.task_id, task_type=self.task_type
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            task_id_var.reset(self.token)
        if self.user_token:
            user_id_var.reset(self.user_token)
        structlog.contextvars.clear_contextvars()


# Convenience functions for common logging patterns
def log_task_start(logger: structlog.BoundLogger, task_name: str, **kwargs):
    """Log task start with standard format."""
    logger.info("Task started", task_name=task_name, **kwargs)


def log_task_complete(
    logger: structlog.BoundLogger, task_name: str, duration: float = None, **kwargs
):
    """Log task completion with standard format."""
    log_data = {"task_name": task_name, **kwargs}
    if duration is not None:
        log_data["duration_seconds"] = round(duration, 3)

    logger.info("Task completed", **log_data)


def log_task_error(
    logger: structlog.BoundLogger, task_name: str, error: Exception, **kwargs
):
    """Log task error with standard format."""
    logger.error(
        "Task failed",
        task_name=task_name,
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs,
    )


def log_external_service_call(
    logger: structlog.BoundLogger,
    service: str,
    operation: str,
    success: bool = True,
    duration: float = None,
    **kwargs,
):
    """Log external service calls with standard format."""
    log_data = {
        "service": service,
        "operation": operation,
        "success": success,
        **kwargs,
    }
    if duration is not None:
        log_data["duration_ms"] = round(duration * 1000, 2)

    level = "info" if success else "warning"
    getattr(logger, level)("External service call", **log_data)
