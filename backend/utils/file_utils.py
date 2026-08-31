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


#: Longest cleaned name, in UTF-8 BYTES, not characters. Filesystems cap a name
#: at 255 bytes, and the pipeline prepends things like ``with_subs_<uuid>_`` and
#: appends extensions — 150 leaves room for all of them. Bytes matter because
#: Hebrew is 2 bytes per character in UTF-8: a 200-character Hebrew title is 400
#: bytes, and a character cap alone would sail past the limit and crash the
#: rename with OSError.
MAX_FILENAME_BYTES = 150


def clean_filename(filename):
    """Make a filename safe for disk and URLs while keeping its language.

    Keeps every Unicode letter and digit — a Hebrew title stays Hebrew. This
    used to be ``re.ASCII``, which defined "letter" as a-z only and turned
    "שי צברי // רחמנא // מתוך אלבום צמאה 1" into ``1`` — the digit was the only
    character that survived, and it became the whole filename.

    What still gets replaced with underscores, because each class genuinely
    breaks something:
      * path separators and ``..`` runs — path traversal
      * Windows-forbidden characters ``< > : " | ? *`` — the file is downloaded
        to Windows machines
      * URL metacharacters ``# ? % &`` — the name is embedded in the download
        link, where ``#`` starts a fragment and ``%`` breaks decoding
      * control characters, and emoji (not letters, and 4 bytes each)

    Safety note: this is also used on user-supplied UPLOAD names in place of
    werkzeug's ``secure_filename`` (which strips all non-ASCII). The traversal
    properties it must uphold: no ``/`` or ``\\``, no ``..``, no leading dot.
    ``tests/unit/test_clean_filename_unit.py`` pins each of these.
    """
    # Normalize Unicode (fullwidth → normal, composed forms) before filtering.
    normalized = unicodedata.normalize("NFKC", filename)

    # \w without re.ASCII is Unicode-aware: keeps letters/digits in any script.
    cleaned = re.sub(r"[^\w\s\-.]", "_", normalized)

    # Collapse whitespace/underscore runs; collapse dot runs (kills "..").
    cleaned = re.sub(r"[\s_]+", "_", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)

    # No leading/trailing junk: hidden-file dots, stray dashes, underscores.
    cleaned = cleaned.strip("_.-")

    if not cleaned:
        cleaned = "video"

    encoded = cleaned.encode("utf-8")
    if len(encoded) > MAX_FILENAME_BYTES:
        # Truncate on the byte budget, then drop any half character the cut
        # left behind (errors="ignore") and any junk the cut exposed.
        cleaned = encoded[:MAX_FILENAME_BYTES].decode("utf-8", "ignore").rstrip("_.-")

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
