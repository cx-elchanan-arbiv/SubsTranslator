"""
Video Processing Statistics Service
====================================

Redis-based statistics tracking for video processing tasks.
Tracks performance metrics, costs, and processing times for analytics.

Features:
- Task-level statistics (transcription, translation, embedding)
- Automatic indexing by date, model, service
- TTL-based cleanup (30 days retention)
- Query helpers for common analytics
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any

import redis

from config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Initialize Redis connection
try:
    redis_client = redis.from_url(
        config.REDIS_URL,
        decode_responses=True,  # Auto-decode bytes to strings
        socket_timeout=5,
        socket_connect_timeout=5,
    )
    # Test connection
    redis_client.ping()
    logger.info("✅ Stats service connected to Redis")
except Exception as e:
    logger.error(f"❌ Stats service failed to connect to Redis: {e}")
    redis_client = None


# Configuration
STATS_TTL_DAYS = 30  # Keep stats for 30 days (Redis only)
STATS_PREFIX = "stats"
INDEX_PREFIX = "stats:index"

# JSONL file configuration
STATS_DIR = os.getenv("STATS_FOLDER", "/app/storage/stats")
STATS_FILE = os.path.join(STATS_DIR, "video_stats.jsonl")
_file_lock = threading.Lock()  # Thread-safe file writing


def append_video_stats_to_jsonl(stats: dict[str, Any]) -> bool:
    """
    Append video statistics to JSONL file (one JSON per line).
    Thread-safe, persistent storage for long-term analysis.

    Args:
        stats: Statistics dictionary

    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Ensure stats directory exists
        os.makedirs(STATS_DIR, exist_ok=True)

        # Add timestamp if not present
        if "timestamp" not in stats:
            stats["timestamp"] = datetime.now().isoformat()

        # Thread-safe append to file
        with _file_lock:
            with open(STATS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats, ensure_ascii=False) + "\n")

        logger.debug(
            f"📝 Appended stats to JSONL: {stats.get('task_id', 'unknown')[:8]}..."
        )
        return True

    except Exception as e:
        logger.error(f"❌ Failed to append stats to JSONL: {e}")
        return False


def save_video_stats(stats: dict[str, Any]) -> bool:
    """
    Save video processing statistics to Redis with automatic indexing.

    Args:
        stats: Dictionary containing:
            - task_id (str): Unique task identifier
            - video_url (str, optional): Video URL or "uploaded"
            - video_duration (float): Video duration in seconds
            - video_size_mb (float): Video file size in MB
            - transcription_model (str): Whisper model used (base/medium/large)
            - transcription_duration (float): Time spent transcribing (seconds)
            - transcription_speed_ratio (float): Speed ratio (e.g., 2.5x)
            - translation_service (str): Translation service used (google/openai)
            - translation_duration (float): Time spent translating (seconds)
            - translation_tokens (int): Total tokens used
            - translation_cost_usd (float): Translation cost in USD
            - embedding_duration (float): Time spent embedding subtitles (seconds)
            - total_duration (float): Total processing time (seconds)
            - status (str): "success" or "failure"
            - error_message (str, optional): Error message if failed
            - created_at (str): ISO format timestamp

    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not redis_client:
        logger.warning("⚠️ Stats service not available (Redis not connected)")
        return False

    try:
        task_id = stats.get("task_id")
        if not task_id:
            logger.error("❌ Cannot save stats: missing task_id")
            return False

        # Add timestamp if not present
        if "created_at" not in stats:
            stats["created_at"] = datetime.now().isoformat()

        # Parse date for indexing
        created_at = datetime.fromisoformat(stats["created_at"].replace("Z", "+00:00"))
        date_key = created_at.strftime("%Y-%m-%d")

        # 1. Save main stats data
        stats_key = f"{STATS_PREFIX}:{task_id}"
        redis_client.setex(stats_key, timedelta(days=STATS_TTL_DAYS), json.dumps(stats))

        # 2. Create indexes for fast queries
        ttl = timedelta(days=STATS_TTL_DAYS)

        # Date index
        redis_client.sadd(f"{INDEX_PREFIX}:date:{date_key}", task_id)
        redis_client.expire(f"{INDEX_PREFIX}:date:{date_key}", ttl)

        # Model index
        if "transcription_model" in stats:
            redis_client.sadd(
                f"{INDEX_PREFIX}:model:{stats['transcription_model']}", task_id
            )
            redis_client.expire(
                f"{INDEX_PREFIX}:model:{stats['transcription_model']}", ttl
            )

        # Translation service index
        if "translation_service" in stats:
            redis_client.sadd(
                f"{INDEX_PREFIX}:service:{stats['translation_service']}", task_id
            )
            redis_client.expire(
                f"{INDEX_PREFIX}:service:{stats['translation_service']}", ttl
            )

        # Status index
        if "status" in stats:
            redis_client.sadd(f"{INDEX_PREFIX}:status:{stats['status']}", task_id)
            redis_client.expire(f"{INDEX_PREFIX}:status:{stats['status']}", ttl)

        logger.info(f"📊 Saved stats to Redis for task {task_id[:8]}...")

        # Also append to JSONL file for persistent storage
        append_video_stats_to_jsonl(stats)

        return True

    except Exception as e:
        logger.error(f"❌ Failed to save stats to Redis: {e}")
        # Even if Redis fails, try to save to file
        try:
            append_video_stats_to_jsonl(stats)
            logger.info("✅ Stats saved to JSONL file (Redis failed)")
            return True
        except:
            return False


def get_stats_by_task_id(task_id: str) -> dict[str, Any] | None:
    """
    Get statistics for a specific task.

    Args:
        task_id: Task ID to retrieve

    Returns:
        Dictionary with stats or None if not found
    """
    if not redis_client:
        return None

    try:
        data = redis_client.get(f"{STATS_PREFIX}:{task_id}")
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to get stats for task {task_id}: {e}")
        return None


def get_stats_by_date(date_str: str) -> list[dict[str, Any]]:
    """
    Get all statistics for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of stat dictionaries
    """
    if not redis_client:
        return []

    try:
        task_ids = redis_client.smembers(f"{INDEX_PREFIX}:date:{date_str}")
        stats = []

        for task_id in task_ids:
            data = redis_client.get(f"{STATS_PREFIX}:{task_id}")
            if data:
                stats.append(json.loads(data))

        return stats
    except Exception as e:
        logger.error(f"❌ Failed to get stats for date {date_str}: {e}")
        return []


def get_stats_by_model(model: str, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Get statistics for a specific Whisper model.

    Args:
        model: Model name (base/medium/large)
        limit: Maximum number of results (optional)

    Returns:
        List of stat dictionaries
    """
    if not redis_client:
        return []

    try:
        task_ids = redis_client.smembers(f"{INDEX_PREFIX}:model:{model}")
        stats = []

        for i, task_id in enumerate(task_ids):
            if limit and i >= limit:
                break
            data = redis_client.get(f"{STATS_PREFIX}:{task_id}")
            if data:
                stats.append(json.loads(data))

        return stats
    except Exception as e:
        logger.error(f"❌ Failed to get stats for model {model}: {e}")
        return []


def get_daily_summary(date_str: str | None = None) -> dict[str, Any]:
    """
    Get daily summary of processing statistics.

    Args:
        date_str: Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dictionary with summary metrics
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    stats_list = get_stats_by_date(date_str)

    if not stats_list:
        return {
            "date": date_str,
            "total_videos": 0,
            "successful": 0,
            "failed": 0,
            "total_cost_usd": 0.0,
            "total_processing_time": 0.0,
            "models_used": {},
            "services_used": {},
        }

    # Calculate aggregations
    successful = [s for s in stats_list if s.get("status") == "success"]
    failed = [s for s in stats_list if s.get("status") == "failure"]

    # Model breakdown
    models_used = {}
    for stat in stats_list:
        model = stat.get("transcription_model", "unknown")
        models_used[model] = models_used.get(model, 0) + 1

    # Service breakdown
    services_used = {}
    for stat in stats_list:
        service = stat.get("translation_service", "unknown")
        services_used[service] = services_used.get(service, 0) + 1

    return {
        "date": date_str,
        "total_videos": len(stats_list),
        "successful": len(successful),
        "failed": len(failed),
        "total_cost_usd": round(
            sum(s.get("translation_cost_usd", 0) for s in stats_list), 4
        ),
        "total_processing_time": round(
            sum(s.get("total_duration", 0) for s in stats_list), 2
        ),
        "avg_processing_time": (
            round(
                sum(s.get("total_duration", 0) for s in stats_list) / len(stats_list), 2
            )
            if stats_list
            else 0
        ),
        "models_used": models_used,
        "services_used": services_used,
    }


def get_model_performance(model: str, days: int = 7) -> dict[str, Any]:
    """
    Calculate performance metrics for a specific Whisper model.

    Args:
        model: Model name (base/medium/large)
        days: Number of days to analyze (default: 7)

    Returns:
        Dictionary with performance metrics
    """
    stats_list = get_stats_by_model(model)

    if not stats_list:
        return {
            "model": model,
            "total_videos": 0,
            "avg_transcription_duration": 0,
            "avg_speed_ratio": 0,
            "min_speed_ratio": 0,
            "max_speed_ratio": 0,
        }

    # Filter successful tasks only
    successful = [s for s in stats_list if s.get("status") == "success"]

    if not successful:
        return {
            "model": model,
            "total_videos": len(stats_list),
            "successful_videos": 0,
            "avg_transcription_duration": 0,
            "avg_speed_ratio": 0,
        }

    durations = [s.get("transcription_duration", 0) for s in successful]
    speed_ratios = [
        s.get("transcription_speed_ratio", 0)
        for s in successful
        if s.get("transcription_speed_ratio", 0) > 0
    ]

    return {
        "model": model,
        "total_videos": len(stats_list),
        "successful_videos": len(successful),
        "avg_transcription_duration": (
            round(sum(durations) / len(durations), 2) if durations else 0
        ),
        "avg_speed_ratio": (
            round(sum(speed_ratios) / len(speed_ratios), 2) if speed_ratios else 0
        ),
        "min_speed_ratio": round(min(speed_ratios), 2) if speed_ratios else 0,
        "max_speed_ratio": round(max(speed_ratios), 2) if speed_ratios else 0,
    }


def get_cost_breakdown(date_str: str | None = None, days: int = 7) -> dict[str, Any]:
    """
    Get cost breakdown by translation service.

    Args:
        date_str: Starting date (defaults to today)
        days: Number of days to analyze (default: 7)

    Returns:
        Dictionary with cost metrics
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Get all stats for the date range
    all_stats = []
    current_date = datetime.strptime(date_str, "%Y-%m-%d")

    for i in range(days):
        check_date = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")
        all_stats.extend(get_stats_by_date(check_date))

    if not all_stats:
        return {
            "date_from": (current_date - timedelta(days=days - 1)).strftime("%Y-%m-%d"),
            "date_to": date_str,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "by_service": {},
        }

    # Calculate costs by service
    by_service = {}
    for stat in all_stats:
        service = stat.get("translation_service", "unknown")
        if service not in by_service:
            by_service[service] = {"cost_usd": 0.0, "tokens": 0, "videos": 0}
        by_service[service]["cost_usd"] += stat.get("translation_cost_usd", 0)
        by_service[service]["tokens"] += stat.get("translation_tokens", 0)
        by_service[service]["videos"] += 1

    # Round costs
    for service in by_service:
        by_service[service]["cost_usd"] = round(by_service[service]["cost_usd"], 4)

    return {
        "date_from": (current_date - timedelta(days=days - 1)).strftime("%Y-%m-%d"),
        "date_to": date_str,
        "total_cost_usd": round(
            sum(s.get("translation_cost_usd", 0) for s in all_stats), 4
        ),
        "total_tokens": sum(s.get("translation_tokens", 0) for s in all_stats),
        "total_videos": len(all_stats),
        "by_service": by_service,
    }


def delete_old_stats(days: int = 30) -> int:
    """
    Delete statistics older than specified days (cleanup utility).
    Note: TTL handles this automatically, but this can be used for manual cleanup.

    Args:
        days: Delete stats older than this many days

    Returns:
        Number of deleted entries
    """
    if not redis_client:
        return 0

    try:
        deleted = 0
        cutoff_date = datetime.now() - timedelta(days=days)

        # Scan all stats keys
        for key in redis_client.scan_iter(f"{STATS_PREFIX}:*"):
            data = redis_client.get(key)
            if data:
                stat = json.loads(data)
                created_at = datetime.fromisoformat(
                    stat["created_at"].replace("Z", "+00:00")
                )
                if created_at < cutoff_date:
                    redis_client.delete(key)
                    deleted += 1

        logger.info(f"🗑️ Deleted {deleted} old stats entries")
        return deleted

    except Exception as e:
        logger.error(f"❌ Failed to delete old stats: {e}")
        return 0


# Health check
def is_stats_service_available() -> bool:
    """Check if stats service is available."""
    if not redis_client:
        return False
    try:
        redis_client.ping()
        return True
    except:
        return False


# =================== JSONL FILE OPERATIONS ===================


def read_all_stats_from_jsonl() -> list[dict[str, Any]]:
    """
    Read all statistics from JSONL file.

    Returns:
        List of all stats dictionaries
    """
    if not os.path.exists(STATS_FILE):
        logger.warning(f"📁 Stats file not found: {STATS_FILE}")
        return []

    try:
        stats_list = []
        with open(STATS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        stats_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping invalid JSON line: {e}")
                        continue

        logger.info(f"📖 Read {len(stats_list)} stats entries from JSONL")
        return stats_list

    except Exception as e:
        logger.error(f"❌ Failed to read JSONL file: {e}")
        return []


def get_stats_file_path() -> str:
    """Get the path to the stats JSONL file."""
    return STATS_FILE


def get_stats_file_size() -> int:
    """
    Get size of stats file in bytes.

    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        if os.path.exists(STATS_FILE):
            return os.path.getsize(STATS_FILE)
        return 0
    except:
        return 0


def get_stats_count_from_file() -> int:
    """
    Count number of stats entries in JSONL file.

    Returns:
        Number of lines in file
    """
    if not os.path.exists(STATS_FILE):
        return 0

    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0
