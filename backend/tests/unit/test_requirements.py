"""requirements.txt as a policy document.

What is worth asserting here is narrow: that the file still pins what it claims to pin,
and that a package the runtime needs has not quietly fallen out of it. Everything else
this file used to contain was ``import flask; assert flask is not None`` in various
costumes — collection imports those modules before any test runs, so a missing dependency
kills the session long before such a test could report it. One of them built a fresh
virtualenv and ran a real ``pip install`` over the network.
"""

import re
from pathlib import Path

import pytest

# Resolved relative to this file, not the CWD: the suite runs from /app in the
# container but from the repo root on CI, and a bare Path("requirements.txt")
# only exists in one of them.
BACKEND_DIR = Path(__file__).resolve().parents[2]
REQUIREMENTS = BACKEND_DIR / "requirements.txt"

#: package name, then the first version operator
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(==|>=|<=|~=|!=|>|<)")


def _requirement_lines():
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            yield raw.strip(), line


@pytest.mark.unit
class TestRequirements:
    def test_every_requirement_is_version_pinned(self):
        """An unpinned dependency makes the build non-reproducible."""
        unpinned = [
            line
            for line, stripped in _requirement_lines()
            if not _REQUIREMENT.match(stripped)
        ]
        assert not unpinned, f"requirements without a version specifier: {unpinned}"

    def test_the_packages_the_runtime_imports_are_all_declared(self):
        """structlog is in this list because its absence once shipped a dead worker."""
        content = REQUIREMENTS.read_text().lower()

        for package in (
            "flask",
            "celery",
            "redis",
            "structlog",
            "yt-dlp",
            "faster-whisper",
            "deep-translator",
            "requests",
            "gunicorn",
        ):
            assert package in content, f"'{package}' missing from requirements.txt"

    def test_no_package_is_declared_twice(self):
        """Two pins for one package — pip silently honours the last one."""
        names = []
        for _, stripped in _requirement_lines():
            match = _REQUIREMENT.match(stripped)
            if match:
                # Normalise per PEP 503 so Foo_Bar and foo-bar collide as they should.
                names.append(re.sub(r"[-_.]+", "-", match.group(1)).lower())

        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert not duplicates, f"duplicate packages: {duplicates}"
