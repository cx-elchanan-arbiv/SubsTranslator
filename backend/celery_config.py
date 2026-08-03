import ssl

from kombu import Queue

from config import get_config

# Get configuration
config = get_config()

# Celery Configuration
# This file centralizes the configuration for Celery.


"""
Select in-memory broker/backend under testing to avoid Redis dependency.
"""
broker_url = config.CELERY_BROKER_URL
result_backend = config.CELERY_RESULT_BACKEND

# SSL Configuration for Redis TLS (rediss://)
# Use broker_transport_options which is more reliable than URL params
if broker_url and broker_url.startswith("rediss://"):
    broker_transport_options = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }
    # For the result backend, we need redis_backend_use_ssl
    redis_backend_use_ssl = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }

# Task Settings
# These settings control how tasks are executed and their results are handled.
task_serializer = config.CELERY_TASK_SERIALIZER
result_serializer = config.CELERY_RESULT_SERIALIZER
accept_content = ["json"]
timezone = config.CELERY_TIMEZONE
enable_utc = config.CELERY_ENABLE_UTC
result_expires = config.CELERY_RESULT_EXPIRES

# Task Queues
# Defines the queues that Celery will use.
# This setup separates different types of tasks to allow for dedicated workers.
task_queues = (
    Queue("default", routing_key="task.default"),
    Queue("processing", routing_key="task.processing"),
    Queue("cleanup", routing_key="task.cleanup"),
)

# Task Routes
# This dictionary routes tasks to specific queues.
# This ensures that tasks are handled by the appropriate worker.
task_routes = {
    # Processing tasks
    "tasks.processing_tasks.process_video_task": {"queue": "processing"},
    "tasks.processing_tasks.create_video_with_subtitles_from_segments": {
        "queue": "processing"
    },
    # Download tasks (explicit names used in decorators)
    "download_and_process_youtube_task": {"queue": "processing"},
    "download_youtube_only_task": {"queue": "processing"},
    "tasks.download_tasks.download_highest_quality_video_task": {"queue": "processing"},
    # Cleanup tasks
    "tasks.cleanup_tasks.cleanup_files_task": {"queue": "cleanup"},
    "tasks.cleanup_tasks.cleanup_old_files_task": {"queue": "cleanup"},
}

# Beat Schedule
# This schedule is for periodic tasks, such as cleanup.
# Celery Beat will run these tasks automatically at the specified intervals.
beat_schedule = {
    "cleanup-every-6-hours": {
        "task": "tasks.cleanup_tasks.cleanup_files_task",
        "schedule": 21600.0,  # 6 hours in seconds
    },
    # Daily cleanup with fast_work directory cleaning
    "cleanup-old-files-daily": {
        "task": "tasks.cleanup_tasks.cleanup_old_files_task",
        "schedule": 86400.0,  # Daily (24 hours in seconds)
        "kwargs": {"days": 14},  # Keep files for 14 days
    },
}

# Worker Settings
# These settings control the behavior of the Celery workers.
worker_concurrency = config.WORKER_CONCURRENCY
worker_prefetch_multiplier = config.WORKER_PREFETCH_MULTIPLIER
worker_max_tasks_per_child = config.WORKER_MAX_TASKS_PER_CHILD

# Task timeout settings (in seconds)
task_soft_time_limit = config.TASK_SOFT_TIME_LIMIT
task_time_limit = config.TASK_TIME_LIMIT
task_acks_late = True  # Acknowledge tasks after completion
worker_disable_rate_limits = True  # Disable rate limits for better performance

# Connection settings
broker_connection_retry_on_startup = True  # Fix Celery 6.0 deprecation warning

"""
Enable eager execution in testing; also store eager results so AsyncResult works.
These are no-ops outside testing.
"""
try:
    task_always_eager = config.CELERY_TASK_ALWAYS_EAGER
    task_eager_propagates = config.CELERY_TASK_EAGER_PROPAGATES
    task_store_eager_result = getattr(config, "CELERY_TASK_STORE_EAGER_RESULT", False)
except AttributeError:
    pass
