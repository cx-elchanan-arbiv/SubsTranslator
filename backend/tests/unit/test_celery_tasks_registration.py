"""
Tests to verify Celery tasks are properly registered and can be imported.
This prevents runtime errors where tasks are not found by the worker.
"""

import pytest


class TestCeleryTasksRegistration:
    """Test that all Celery tasks are properly registered."""

    def test_celery_worker_imports_successfully(self):
        """Test that celery_worker can be imported without errors."""
        try:
            from celery_worker import celery_app

            assert celery_app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import celery_worker: {e}")

    def test_tasks_module_imports_successfully(self):
        """Test that tasks module can be imported without errors."""
        try:
            import tasks

            assert tasks is not None
        except ImportError as e:
            pytest.fail(f"Failed to import tasks: {e}")

    def test_all_expected_tasks_are_registered(self):
        """Test that all expected tasks are registered with Celery."""
        from celery_worker import celery_app

        registered_tasks = list(celery_app.tasks.keys())

        # Expected main tasks (tasks were reorganized into submodules)
        expected_tasks = [
            "download_and_process_youtube_task",
            "download_youtube_only_task",
            "tasks.processing_tasks.process_video_task",
            "tasks.cleanup_tasks.cleanup_files_task",
            "tasks.processing_tasks.create_video_with_subtitles_from_segments",
            "tasks.download_tasks.download_highest_quality_video_task",
        ]

        missing_tasks = []
        for task in expected_tasks:
            if task not in registered_tasks:
                missing_tasks.append(task)

        assert (
            len(missing_tasks) == 0
        ), f"Missing tasks: {missing_tasks}. Registered tasks: {registered_tasks}"

    def test_specific_task_can_be_called(self):
        """Test that a specific task can be accessed and has correct signature."""
        from celery_worker import celery_app

        # Test the main YouTube processing task
        task = celery_app.tasks.get("download_and_process_youtube_task")
        assert (
            task is not None
        ), "download_and_process_youtube_task not found in registered tasks"

        # Verify it's callable
        assert callable(task), "Task is not callable"

    def test_no_circular_import_issues(self):
        """Test that importing all task modules doesn't cause circular import errors."""
        try:
            # Import in the same order as celery_worker
            import tasks
            from celery_worker import celery_app

            # Verify celery_app picked up the task modules
            registered_tasks = celery_app.tasks.keys()
            assert "download_and_process_youtube_task" in registered_tasks
            assert (
                "tasks.download_tasks.download_highest_quality_video_task"
                in registered_tasks
            )

        except ImportError as e:
            pytest.fail(f"Circular import or import error: {e}")

    def test_task_discovery_mechanism(self):
        """Test that Celery can discover tasks automatically."""
        from celery_worker import celery_app

        # Get all non-builtin tasks (exclude celery.* tasks)
        user_tasks = [
            task for task in celery_app.tasks.keys() if not task.startswith("celery.")
        ]

        # Should have at least the main tasks
        assert (
            len(user_tasks) >= 5
        ), f"Expected at least 5 user tasks, found: {user_tasks}"

        # Verify main task is present
        assert "download_and_process_youtube_task" in user_tasks
