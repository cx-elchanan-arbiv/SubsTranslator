"""Every task the API enqueues must be registered on the worker.

This is the file that catches a whole class of silent breakage: the worker only learns
about tasks through the side-effect ``import tasks`` in ``celery_worker.py``. Drop that
import — a ``ruff --fix`` pass once did exactly that, see the note in .github/workflows/
ci.yml — or move a task between submodules without updating its name, and the app keeps
accepting uploads while nothing ever runs them.

The five other tests that lived here were ``import tasks; assert tasks is not None`` and
subsets of the check below. It also carried no ``unit`` mark, so CI never ran any of it.
"""

import pytest


@pytest.mark.unit
def test_all_expected_tasks_are_registered():
    from celery_worker import celery_app

    registered = set(celery_app.tasks)

    # Two are explicitly named at the decorator; the rest take Celery's
    # module-qualified default. Both spellings are part of the contract, because the
    # API enqueues by name.
    expected = {
        "download_and_process_youtube_task",
        "download_youtube_only_task",
        "tasks.processing_tasks.process_video_task",
        "tasks.processing_tasks.create_video_with_subtitles_from_segments",
        "tasks.cleanup_tasks.cleanup_files_task",
        "tasks.download_tasks.download_highest_quality_video_task",
    }

    missing = expected - registered
    assert not missing, f"unregistered tasks: {sorted(missing)}"
