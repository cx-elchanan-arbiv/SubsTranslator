"""
File utility functions for SubsTranslator
"""

import re
import unicodedata


def safe_int(
    value: str | int | None,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> tuple[int, str | None]:
    """
    Safely convert a value to integer with validation.

    Args:
        value: The value to convert (string, int, or None)
        default: Default value if conversion fails
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value

    Returns:
        Tuple of (integer_value, error_message)
        error_message is None if successful

    Example:
        opacity, error = safe_int(request.form.get('opacity'), 40, 0, 100)
        if error:
            return jsonify({"error": error}), 400
    """
    if value is None:
        return default, None

    try:
        result = int(value)
    except (ValueError, TypeError):
        return default, f"Invalid integer value: {value}"

    if min_val is not None and result < min_val:
        return default, f"Value {result} is below minimum {min_val}"

    if max_val is not None and result > max_val:
        return default, f"Value {result} exceeds maximum {max_val}"

    return result, None


def clean_filename(filename):
    """Clean filename by removing problematic characters"""
    # First normalize Unicode characters (convert fullwidth to normal)
    normalized = unicodedata.normalize("NFKC", filename)

    # Replace any non-ASCII, non-alphanumeric characters with underscores
    cleaned = re.sub(r"[^\w\s\-_.]", "_", normalized, flags=re.ASCII)

    # Replace multiple spaces/underscores with single underscore
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("_")

    # Ensure it's not empty and not too long
    if not cleaned or cleaned == "_":
        cleaned = "video"
    if len(cleaned) > 100:
        cleaned = cleaned[:100].rstrip("_")

    return cleaned


def parse_time_to_seconds(time_str):
    """
    Parse flexible time format to seconds.

    Supports:
    - SS: "90" -> 90 seconds
    - MM:SS: "01:30" -> 90 seconds
    - HH:MM:SS: "00:01:30" -> 90 seconds

    Args:
        time_str: Time string in format SS, MM:SS, or HH:MM:SS

    Returns:
        int: Total seconds

    Raises:
        ValueError: If format is invalid
    """
    if not time_str or not isinstance(time_str, str):
        raise ValueError(f"Invalid time string: {time_str}")

    time_str = time_str.strip()

    # Case 1: Pure seconds "90"
    if time_str.isdigit():
        return int(time_str)

    # Case 2: MM:SS or HH:MM:SS
    parts = time_str.split(":")
    if len(parts) == 2:  # MM:SS
        try:
            minutes, seconds = map(int, parts)
            if seconds > 59:
                raise ValueError(f"Invalid seconds value: {seconds} (must be 0-59)")
            return minutes * 60 + seconds
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid MM:SS format: {time_str}") from e

    elif len(parts) == 3:  # HH:MM:SS
        try:
            hours, minutes, seconds = map(int, parts)
            if minutes > 59:
                raise ValueError(f"Invalid minutes value: {minutes} (must be 0-59)")
            if seconds > 59:
                raise ValueError(f"Invalid seconds value: {seconds} (must be 0-59)")
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid HH:MM:SS format: {time_str}") from e

    else:
        raise ValueError(
            f"Invalid time format: {time_str}. Expected: SS, MM:SS, or HH:MM:SS"
        )
