"""
Progress Manager for Celery tasks
Provides step-based progress tracking for long-running tasks
"""

import time


class ProgressManager:
    """Manages step-based progress updates for Celery tasks with metadata preservation."""

    def __init__(self, task, steps_config):
        self.task = task
        self.steps = steps_config
        self.start_time = time.time()
        self.logs = []
        # Store existing metadata to preserve it
        self.video_metadata = None
        self.user_choices = {}
        self.initial_request = {}
        for step in self.steps:
            step["status"] = "waiting"
            step["progress"] = 0
            step["indeterminate"] = step.get("indeterminate", False)

    def _update_state(self):
        overall_progress = 0
        for step in self.steps:
            step_contribution = (step["progress"] / 100.0) * step.get("weight", 0) * 100
            overall_progress += step_contribution

        # Build meta with existing metadata preserved
        meta = {
            "steps": self.steps,
            "overall_percent": int(overall_progress),
            "logs": self.logs,
        }

        # Preserve existing metadata
        if self.video_metadata:
            meta["video_metadata"] = self.video_metadata
        if self.user_choices:
            meta["user_choices"] = self.user_choices
        if self.initial_request:
            meta["initial_request"] = self.initial_request

        self.task.update_state(state="PROGRESS", meta=meta)

    def log(self, message, step_index=None):
        timestamp = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.start_time)
        )
        self.logs.append(f"[{timestamp}] {message}")
        if step_index is not None:
            self.steps[step_index]["status_message"] = message
        self._update_state()

    def set_step_status(self, step_index, status):
        self.steps[step_index]["status"] = status
        self.log(
            f"Step '{self.steps[step_index]['label']}' status changed to {status}",
            step_index=step_index,
        )

    def set_step_progress(self, step_index, progress, message=None):
        self.steps[step_index]["progress"] = progress
        if message:
            self.steps[step_index]["status_message"] = message
        self._update_state()

    def complete_step(self, step_index):
        self.steps[step_index]["status"] = "completed"
        self.steps[step_index]["progress"] = 100
        self.log(
            f"Step '{self.steps[step_index]['label']}' completed", step_index=step_index
        )

    def set_metadata(
        self, video_metadata=None, user_choices=None, initial_request=None
    ):
        """Set metadata to be preserved across progress updates"""
        if video_metadata:
            self.video_metadata = video_metadata
        if user_choices:
            self.user_choices = user_choices
        if initial_request:
            self.initial_request = initial_request
        self._update_state()

    def set_step_error(self, step_index, error_message):
        self.steps[step_index]["status"] = "error"
        self.log(
            f"Step '{self.steps[step_index]['label']}' error: {error_message}",
            step_index=step_index,
        )
