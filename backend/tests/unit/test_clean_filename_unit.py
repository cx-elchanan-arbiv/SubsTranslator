"""clean_filename: keep the language, keep the safety.

Two jobs at once, and each half of this file guards one of them:

1. **Language survives.** The old ``re.ASCII`` version defined "letter" as a-z,
   so "שי צברי // רחמנא // מתוך אלבום צמאה 1" became ``1.mp3`` — the digit was
   the only survivor and it became the entire filename. Most of this project's
   content is Hebrew, so that was the common case, not an edge.

2. **Traversal safety survives.** clean_filename now replaces werkzeug's
   secure_filename at the upload entry points (video_routes, editing_routes),
   which makes it a security boundary: user-controlled names must not escape
   the upload folder, hide as dotfiles, or smuggle URL metacharacters into the
   download links they get embedded in.
"""

import os
import sys

import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from utils.file_utils import MAX_FILENAME_BYTES, clean_filename  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Language survives
# ---------------------------------------------------------------------------


def test_the_bug_that_started_this_hebrew_title_is_not_reduced_to_a_digit():
    # The real title that came out as "1.mp3" in production.
    assert (
        clean_filename("שי צברי // רחמנא // מתוך אלבום צמאה 1")
        == "שי_צברי_רחמנא_מתוך_אלבום_צמאה_1"
    )


def test_mixed_hebrew_english_title_keeps_both():
    assert (
        clean_filename("אברהם פריד והסימפונית - שלום עליכם | Avraham Fried")
        == "אברהם_פריד_והסימפונית_-_שלום_עליכם_Avraham_Fried"
    )


def test_hebrew_upload_name_keeps_its_extension():
    # werkzeug's secure_filename turned this into literally "mp4".
    assert clean_filename("הרצאה על בינה.mp4") == "הרצאה_על_בינה.mp4"


def test_arabic_survives_too():
    assert clean_filename("محاضرة عن الذكاء.mp4") == "محاضرة_عن_الذكاء.mp4"


def test_ascii_behaviour_is_unchanged():
    # The pre-fix output for ASCII input, byte for byte — no regression for
    # every existing English-titled file.
    assert clean_filename("Me at the zoo") == "Me_at_the_zoo"


def test_emoji_are_dropped_not_kept():
    # Emoji are not letters, and at 4 UTF-8 bytes each they burn the byte
    # budget three times faster than Hebrew.
    assert clean_filename("🔥🔥 שיר מדהים 🔥🔥") == "שיר_מדהים"


# ---------------------------------------------------------------------------
# 2. Traversal and platform safety
# ---------------------------------------------------------------------------


def test_path_traversal_is_neutralised():
    cleaned = clean_filename("../../etc/passwd")
    assert "/" not in cleaned
    assert ".." not in cleaned
    assert not cleaned.startswith(".")


def test_backslashes_and_windows_drive_are_neutralised():
    cleaned = clean_filename("..\\..\\windows\\system32")
    assert "\\" not in cleaned
    assert ".." not in cleaned


def test_windows_forbidden_characters_are_removed():
    cleaned = clean_filename('a<b>c:d"e|f?g*h')
    for ch in '<>:"|?*':
        assert ch not in cleaned
    assert cleaned == "a_b_c_d_e_f_g_h"


def test_url_metacharacters_are_removed():
    # The name is embedded in /download/<name>: '#' starts a fragment,
    # '%' breaks percent-decoding, '&' and '?' split queries.
    cleaned = clean_filename("song #1 (100% legal) ?ok&fine")
    for ch in "#%?&":
        assert ch not in cleaned


def test_no_hidden_files():
    assert not clean_filename(".htaccess").startswith(".")


def test_control_characters_are_removed():
    cleaned = clean_filename("a\x00b\nc\rd")
    assert cleaned == "a_b_c_d"


def test_punctuation_only_falls_back_to_video():
    assert clean_filename("///???***") == "video"
    assert clean_filename("") == "video"


# ---------------------------------------------------------------------------
# 3. Length is capped in BYTES, at a character boundary
# ---------------------------------------------------------------------------


def test_long_hebrew_title_fits_the_filesystem():
    # 200 Hebrew characters = 400 UTF-8 bytes. A character-based cap would pass
    # this through and the rename would die with OSError(36).
    cleaned = clean_filename("א" * 200)
    assert len(cleaned.encode("utf-8")) <= MAX_FILENAME_BYTES
    # The cut must not leave half a character behind.
    cleaned.encode("utf-8").decode("utf-8")  # raises if malformed


def test_long_ascii_title_is_also_capped():
    cleaned = clean_filename("a" * 500)
    assert len(cleaned.encode("utf-8")) <= MAX_FILENAME_BYTES


def test_room_for_pipeline_prefixes():
    # The longest real prefix is "with_subs_<uuid4>_" (46 chars) plus ".mp4".
    # A maximal cleaned name must still fit a 255-byte filesystem name limit.
    cleaned = clean_filename("ב" * 300)
    full = f"with_subs_12345678-1234-1234-1234-123456789012_{cleaned}.mp4"
    assert len(full.encode("utf-8")) <= 255
