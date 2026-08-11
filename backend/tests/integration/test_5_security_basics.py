"""
Test 5: Basic Security Checks
Verifies basic security measures are in place.
"""

import os
import re

import pytest

#: The tree to scan. This file lives at ``<backend>/tests/integration/``, so the backend
#: root is two levels up.
#:
#: It used to be ``os.path.join(os.path.dirname(__file__), "..", "backend")`` — i.e.
#: ``<backend>/tests/backend``, a directory that has never existed. ``os.walk`` over a
#: missing path yields nothing and raises nothing, so the scan below walked 0
#: directories, opened 0 files, and then asserted that its empty findings list was
#: empty. Measured before the fix: "dirs walked: 0, .py files scanned: 0", and a
#: real-looking API key planted anywhere in the tree left the test green.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{48,}",  # OpenAI keys
    r'password\s*=\s*["\'][^"\']+["\']',  # Passwords
    r'secret\s*=\s*["\'][^"\']+["\']',  # Secrets
]

#: Not this project's source: caches, VCS metadata, vendored/installed packages and the
#: runtime data directories the container accumulates.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "node_modules",
    "site-packages",
    "whisper_models",
}


def scan_for_secrets(root):
    """Grep ``root``'s Python files for hardcoded credentials.

    Returns ``(findings, files_scanned)``. The file count is returned rather than
    discarded because it is the only thing that distinguishes "nothing to find" from
    "nothing was looked at" — the failure mode this whole file was in for as long as it
    has existed.
    """
    findings = []
    files_scanned = 0

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(current_root, file)
            try:
                with open(file_path, encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except OSError:
                # Unreadable is not clean: count it nowhere and say so in the findings,
                # rather than the old bare `except: continue` that hid it completely.
                findings.append((file_path, ["<unreadable>"]))
                continue

            files_scanned += 1
            for pattern in SECRET_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                # Filter out test keys
                real_matches = [m for m in matches if "test" not in m.lower()]
                if real_matches:
                    findings.append((os.path.relpath(file_path, root), real_matches))

    return findings, files_scanned


def assert_the_scan_actually_read_something(files_scanned, root):
    """A scanner that scanned nothing must never be green.

    Shared by the real scan and by the empty-tree test below, so the guard has exactly
    one implementation and that implementation is itself under test.
    """
    assert files_scanned > 0, (
        f"the secret scan read 0 Python files under {root} — it proved nothing. "
        f"Either the path is wrong again or SKIP_DIRS is excluding the source tree."
    )


def test_no_hardcoded_secrets():
    """Test that there are no hardcoded secrets in code"""
    found_secrets, files_scanned = scan_for_secrets(BACKEND_DIR)

    assert_the_scan_actually_read_something(files_scanned, BACKEND_DIR)

    assert (
        len(found_secrets) == 0
    ), f"Found hardcoded secrets in {files_scanned} scanned files: {found_secrets}"


def test_the_secret_scan_catches_a_planted_secret(tmp_path):
    """The scan has teeth: prove it on a tree with real-looking credentials in it.

    Without this, ``test_no_hardcoded_secrets`` above can only ever prove that the
    patterns matched nothing — which is exactly what it reported for months while
    matching nothing against nothing. Every string below is assembled by concatenation
    so that no scannable literal exists in this file; this file is itself inside the
    scanned tree, and a planted secret written out in one piece here would make the
    real scan fail on the scanner's own source.
    """
    planted_key = "sk-" + ("A1b2C3d4E5f6G7h8I9j0" * 3)[:50]
    # The variable name matters too: the pattern matches the WORD followed by `=` and a
    # quoted value, so naming this one after the credential it carries makes the LINE a
    # hit. Measured — the first version of this test flagged its own source file, and
    # the run above went red on it.
    planted_credential_line = "pass" + 'word = "hunter2-not-a-drill"'

    module = tmp_path / "pkg" / "leaky.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "OPENAI_KEY = " + repr(planted_key) + "\n" + planted_credential_line + "\n",
        encoding="utf-8",
    )
    # A file the scan must ignore, so a clean result is not just "matched nothing".
    (tmp_path / "pkg" / "notes.txt").write_text(planted_key, encoding="utf-8")

    findings, files_scanned = scan_for_secrets(str(tmp_path))

    assert (
        files_scanned == 1
    ), f"expected to scan the one .py file, read {files_scanned}"
    flat = [match for _path, matches in findings for match in matches]
    assert any(
        planted_key in match for match in flat
    ), f"the planted API key was not detected: {findings}"
    assert any(
        "hunter2" in match for match in flat
    ), f"the planted password was not detected: {findings}"


def test_the_secret_scan_fails_loudly_on_an_empty_tree(tmp_path):
    """Zero files scanned is a broken scanner, not a clean bill of health."""
    empty = tmp_path / "nowhere"
    empty.mkdir()

    findings, files_scanned = scan_for_secrets(str(empty))

    # "No findings" and "read nothing" together are exactly what the old test reported
    # for every run of its life, and what it called a pass.
    assert findings == []
    assert files_scanned == 0
    with pytest.raises(AssertionError, match="proved nothing"):
        assert_the_scan_actually_read_something(files_scanned, str(empty))


def test_openai_key_validation():
    """Test that OpenAI key validation works"""
    from app import _is_valid_openai_key

    # Valid keys
    assert _is_valid_openai_key("sk-test-valid-key-1234567890123456789012345") is True

    # Invalid keys
    assert _is_valid_openai_key("your-openai-api-key-here") is False
    assert _is_valid_openai_key("") is False
    assert _is_valid_openai_key(None) is False
    assert _is_valid_openai_key("invalid-format") is False


def test_file_extension_validation():
    """Test that file extension validation works"""
    from config import get_config

    config = get_config()

    # Allowed files
    assert config.is_allowed_file_extension("video.mp4") is True
    assert config.is_allowed_file_extension("audio.wav") is True

    # Disallowed files
    assert config.is_allowed_file_extension("script.exe") is False
    assert config.is_allowed_file_extension("data.txt") is False
    assert config.is_allowed_file_extension("noextension") is False
