"""
Download Token Service
Manages one-time download tokens for secure file access
"""

import os
import threading
import time
import uuid

from logging_config import get_logger

logger = get_logger(__name__)

# Token storage and synchronization
_download_tokens = {}
_download_tokens_lock = threading.Lock()
_cleanup_timer = None
_scheduler_started = False  # Guard against multiple starts


def generate_download_token(filename, expires_in=300):
    """
    Generate a one-time download token for a file.

    Args:
        filename: Name of the file to associate with the token
        expires_in: Token expiration time in seconds (default: 300 = 5 minutes)

    Returns:
        str: UUID token for one-time file download
    """
    token = str(uuid.uuid4())
    expiry_time = time.time() + expires_in

    with _download_tokens_lock:
        _download_tokens[token] = {
            "filename": filename,
            "expires_at": expiry_time,
            "used": False,
        }

    logger.info(
        "Download token generated",
        operation="token_generated",
        filename=filename,
        token_prefix=token[:8],
    )
    return token


def use_download_token(token):
    """
    Use a download token (can only be used once).

    Args:
        token: The download token to use

    Returns:
        tuple: (filename, error_message)
            - If successful: (filename, None)
            - If failed: (None, error_message)
    """
    with _download_tokens_lock:
        if token not in _download_tokens:
            return None, "Invalid token"

        token_info = _download_tokens[token]

        if token_info["used"]:
            return None, "Token already used"

        if time.time() > token_info["expires_at"]:
            del _download_tokens[token]
            return None, "Token expired"

        token_info["used"] = True
        filename = token_info["filename"]

        del _download_tokens[token]

        logger.info(
            "Download token used",
            operation="token_used",
            filename=filename,
            token_prefix=token[:8],
        )
        return filename, None


def cleanup_expired_tokens():
    """Clean up expired tokens from storage."""
    current_time = time.time()
    with _download_tokens_lock:
        expired_tokens = [
            token
            for token, info in _download_tokens.items()
            if current_time > info["expires_at"]
        ]
        for token in expired_tokens:
            filename = _download_tokens[token]["filename"]
            del _download_tokens[token]
            logger.info(
                "Expired token cleaned up", operation="token_cleanup", filename=filename
            )


def _schedule_token_cleanup():
    """Schedule periodic token cleanup (internal use only)."""
    global _cleanup_timer

    cleanup_expired_tokens()

    # Only run timer if not in testing mode
    if not os.getenv("TESTING", "").lower() == "true":
        _cleanup_timer = threading.Timer(300, _schedule_token_cleanup)
        _cleanup_timer.start()


def start_cleanup_scheduler():
    """
    Start the periodic token cleanup scheduler.

    Should be called once during application initialization.
    Only starts if not in testing mode and not already started.
    Note: In multi-worker setups (Gunicorn), each worker runs its own scheduler.
    """
    global _scheduler_started

    if _scheduler_started:
        return  # Already started in this process, skip

    if not os.getenv("TESTING", "").lower() == "true":
        _scheduler_started = True
        _schedule_token_cleanup()
        # Use debug level to avoid log noise in multi-worker setups
        logger.debug("Token cleanup scheduler started (5 minute interval)")


def stop_cleanup_scheduler():
    """
    Stop the periodic token cleanup scheduler.

    Useful for graceful shutdown or testing.
    """
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
        _cleanup_timer = None
        logger.info("Token cleanup scheduler stopped")


def get_token_stats():
    """
    Get statistics about current tokens (for monitoring/debugging).

    Returns:
        dict: Token statistics including total count, expired count, used count
    """
    current_time = time.time()
    with _download_tokens_lock:
        total = len(_download_tokens)
        expired = sum(
            1 for info in _download_tokens.values() if current_time > info["expires_at"]
        )
        used = sum(1 for info in _download_tokens.values() if info["used"])

        return {
            "total_tokens": total,
            "expired_tokens": expired,
            "used_tokens": used,
            "active_tokens": total - expired - used,
        }
