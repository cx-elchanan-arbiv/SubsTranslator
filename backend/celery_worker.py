from celery import Celery

# Create Celery instance
# This initializes Celery and loads the configuration from the 'celery_config' module.
# The configuration includes broker URL, result backend, and other settings.
celery_app = Celery("tasks")
celery_app.config_from_object("celery_config")

# This import is placed here to avoid circular imports.
# The tasks module needs the `celery_app` object, so it's imported after the app is
# created. It is a SIDE-EFFECT import: importing `tasks` is what runs every
# @celery_app.task decorator and registers the tasks on the worker. It looks unused
# to a linter, but removing it silently produces a worker with zero tasks.
import tasks  # noqa: F401  (side-effect import: registers Celery tasks)

# Get logger for startup messages
from logging_config import get_logger

logger = get_logger(__name__)

# Log SSL configuration for debugging
from config import get_config

_config = get_config()
if _config.CELERY_BROKER_URL and "ssl_cert_reqs" in _config.CELERY_BROKER_URL:
    logger.info("🔒 Redis TLS configured with ssl_cert_reqs in URL")
elif _config.CELERY_BROKER_URL and _config.CELERY_BROKER_URL.startswith("rediss://"):
    logger.warning("⚠️ Redis TLS URL missing ssl_cert_reqs parameter!")

logger.info("🔧 Celery worker is ready! Task processing system initialized! 🚀")

if __name__ == "__main__":
    # This block allows running the Celery worker directly.
    # The worker will connect to the broker and start processing tasks from the defined queues.
    # Example command: celery -A celery_worker.celery_app worker -l info -Q processing
    celery_app.start()
