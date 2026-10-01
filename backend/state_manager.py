"""
Enterprise-grade state management for Celery tasks.
Thread-safe, consistent, with proper error handling.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from services.job_cancel import JobCancelled, raise_if_cancelled

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Standardized task states."""

    PENDING = "PENDING"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class StepStatus(Enum):
    """Standardized step statuses."""

    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TaskStep:
    """Structured step definition."""

    label: str
    weight: float
    progress: int = 0
    status: StepStatus = StepStatus.WAITING
    status_message: str = ""
    indeterminate: bool = False
    error_message: str = ""


@dataclass
class TaskMetadata:
    """Consolidated task metadata."""

    video_metadata: dict[str, Any] | None = None
    user_choices: dict[str, Any] | None = None
    initial_request: dict[str, Any] | None = None


class EnterpriseStateManager:
    """
    Thread-safe, enterprise-grade state manager for Celery tasks.

    Features:
    - Atomic state updates
    - Consistent metadata preservation
    - Comprehensive logging
    - Error recovery
    """

    def __init__(self, celery_task, steps_config: list[dict[str, Any]]):
        self.task = celery_task
        self._cancel_raised = False
        self.steps = [
            TaskStep(
                label=step["label"],
                weight=step.get("weight", 1.0),
                indeterminate=step.get("indeterminate", False),
            )
            for step in steps_config
        ]

        self.metadata = TaskMetadata()
        self.logs = []
        self.start_time = time.time()
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Initialize with PENDING state
        self._update_celery_state()

        logger.info(f"StateManager initialized with {len(self.steps)} steps")

    def _calculate_overall_progress(self) -> int:
        """Calculate weighted overall progress."""
        total_weight = sum(step.weight for step in self.steps)
        if total_weight == 0:
            return 0

        weighted_progress = sum(
            (step.progress / 100.0) * step.weight for step in self.steps
        )

        return min(100, int((weighted_progress / total_weight) * 100))

    def _build_state_meta(self) -> dict[str, Any]:
        """Build consistent state metadata."""
        # Convert steps to dict with enum values as strings
        steps_data = []
        for step in self.steps:
            step_dict = asdict(step)
            # Convert enum to string for JSON serialization
            if isinstance(step_dict["status"], StepStatus):
                step_dict["status"] = step_dict["status"].value
            steps_data.append(step_dict)

        meta = {
            "overall_percent": self._calculate_overall_progress(),
            "steps": steps_data,
            "logs": self.logs.copy(),  # Copy to prevent external mutation
            "timestamp": time.time(),
        }

        # Add metadata if available
        if self.metadata.video_metadata:
            meta["video_metadata"] = self.metadata.video_metadata
        if self.metadata.user_choices:
            meta["user_choices"] = self.metadata.user_choices
        if self.metadata.initial_request:
            meta["initial_request"] = self.metadata.initial_request

        return meta

    def _check_cancel(self):
        """
        The Stop button (services.job_cancel). Every progress update passes here, so
        a stopped job ends at its next update — in yt-dlp's hook, between
        transcription segments, between translation batches. Raised once; after
        that the job's own cancel handler may still log and update state freely.
        """
        if self._cancel_raised:
            return
        try:
            raise_if_cancelled(getattr(getattr(self.task, "request", None), "id", None))
        except JobCancelled:
            self._cancel_raised = True
            raise

    def acknowledge_cancel(self):
        """The job is handling its stop: progress updates no longer raise."""
        self._cancel_raised = True

    def _update_celery_state(self) -> None:
        """Atomic Celery state update."""
        self._check_cancel()
        try:
            meta = self._build_state_meta()
            self.task.update_state(state=TaskState.PROGRESS.value, meta=meta)

        except Exception as e:
            logger.error(f"Failed to update Celery state: {e}")
            # Don't raise - state updates are non-critical for task logic

    def set_metadata(
        self,
        video_metadata: dict[str, Any] | None = None,
        user_choices: dict[str, Any] | None = None,
        initial_request: dict[str, Any] | None = None,
    ) -> None:
        """Thread-safe metadata update."""
        with self._lock:
            if video_metadata is not None:
                self.metadata.video_metadata = video_metadata
                logger.info("Video metadata updated")

            if user_choices is not None:
                self.metadata.user_choices = user_choices
                logger.info("User choices updated")

            if initial_request is not None:
                self.metadata.initial_request = initial_request
                logger.info("Initial request updated")

            self._update_celery_state()

    def log(
        self, message: str, step_index: int | None = None, level: str = "INFO"
    ) -> None:
        """Thread-safe logging with optional step association."""
        with self._lock:
            elapsed = time.time() - self.start_time
            timestamp = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            log_entry = f"[{timestamp}] {message}"

            self.logs.append(log_entry)

            # Log to system logger as well
            getattr(logger, level.lower(), logger.info)(f"Task log: {message}")

            # Associate with step if specified
            if step_index is not None and 0 <= step_index < len(self.steps):
                self.steps[step_index].status_message = message

            self._update_celery_state()

    def set_step_status(
        self, step_index: int, status: StepStatus, message: str = ""
    ) -> None:
        """Update step status with optional message."""
        with self._lock:
            if not (0 <= step_index < len(self.steps)):
                logger.error(f"Invalid step index: {step_index}")
                return

            step = self.steps[step_index]
            old_status = step.status
            step.status = status

            if message:
                step.status_message = message

            self.log(
                f"Step '{step.label}' {old_status.value} → {status.value}"
                + (f": {message}" if message else ""),
                step_index=step_index,
            )

    def set_step_progress(
        self, step_index: int, progress: int, message: str = ""
    ) -> None:
        """Update step progress with bounds checking."""
        with self._lock:
            if not (0 <= step_index < len(self.steps)):
                logger.error(f"Invalid step index: {step_index}")
                return

            # Bound progress to valid range
            progress = max(0, min(100, progress))

            step = self.steps[step_index]
            step.progress = progress

            if message:
                step.status_message = message

            # Auto-update status based on progress
            if progress == 0:
                step.status = StepStatus.WAITING
            elif progress == 100:
                step.status = StepStatus.COMPLETED
            else:
                step.status = StepStatus.IN_PROGRESS

            self._update_celery_state()

    def set_step_error(self, step_index: int, error_message: str) -> None:
        """Set step to error state with message."""
        with self._lock:
            if not (0 <= step_index < len(self.steps)):
                logger.error(f"Invalid step index: {step_index}")
                return

            step = self.steps[step_index]
            step.status = StepStatus.ERROR
            step.error_message = error_message
            step.status_message = f"Error: {error_message}"

            self.log(
                f"Step '{step.label}' failed: {error_message}", step_index, "ERROR"
            )

    def complete_step(self, step_index: int) -> None:
        """Mark step as completed."""
        self.set_step_progress(step_index, 100, "Completed")

    def fail_task(
        self,
        error_code: str,
        error_message: str,
        user_message: str,
        recoverable: bool = True,
    ) -> dict[str, Any]:
        """
        Standardized task failure with comprehensive error info.

        Returns error dict for task result.
        """
        with self._lock:
            error_info = {
                "status": "FAILURE",
                "code": error_code,  # Use 'code' to match frontend TaskError interface
                "message": error_message,
                "user_facing_message": user_message,
                "recoverable": recoverable,
                "timestamp": time.time(),
                "logs": self.logs.copy(),
            }

            self.log(f"Task failed: {error_code} - {error_message}", level="ERROR")

            return error_info

    def complete_task(self, result: dict[str, Any]) -> dict[str, Any]:
        """Mark task as successfully completed."""
        with self._lock:
            # Mark all steps as completed
            for i in range(len(self.steps)):
                if self.steps[i].status != StepStatus.ERROR:
                    self.steps[i].status = StepStatus.COMPLETED
                    self.steps[i].progress = 100

            success_result = {
                "status": "SUCCESS",
                "result": result,
                "metadata": asdict(self.metadata),
                "execution_time": time.time() - self.start_time,
                "logs": self.logs.copy(),
            }

            self.log("Task completed successfully")

            return success_result
