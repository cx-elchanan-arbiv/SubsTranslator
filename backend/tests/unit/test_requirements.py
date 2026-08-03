"""
Tests for requirements.txt and dependency management.
These tests help catch missing dependencies before deployment.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Resolved relative to this file, not the CWD: the suite runs from /app in the
# container but from the repo root on CI, and a bare Path("requirements.txt")
# only exists in one of them.
BACKEND_DIR = Path(__file__).resolve().parents[2]
REQUIREMENTS = BACKEND_DIR / "requirements.txt"


@pytest.mark.unit
class TestRequirements:
    """Test requirements.txt file and dependencies."""

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        req_file = REQUIREMENTS
        assert req_file.exists(), "requirements.txt not found"

    def test_requirements_file_not_empty(self):
        """Test that requirements.txt is not empty."""
        req_file = REQUIREMENTS
        content = req_file.read_text()
        assert content.strip(), "requirements.txt is empty"

    def test_requirements_format(self):
        """Test that requirements.txt has proper format."""
        req_file = REQUIREMENTS
        lines = req_file.read_text().splitlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Should have package==version or package>=version format
            has_version_spec = any(
                op in line for op in ["==", ">=", "<=", ">", "<", "~="]
            )
            assert (
                has_version_spec
            ), f"Line {line_num}: '{line}' should have version specifier (==, >=, etc.)"

            # Parse package name (before first version operator)
            for op in ["==", ">=", "<=", "~=", ">", "<"]:
                if op in line:
                    package = line.split(op)[0].strip()
                    assert package, f"Line {line_num}: Empty package name"
                    break

    def test_critical_packages_present(self):
        """Test that critical packages are in requirements.txt."""
        req_file = REQUIREMENTS
        content = req_file.read_text().lower()

        critical_packages = [
            "flask",
            "celery",
            "redis",
            "structlog",  # This would have caught our bug!
            "yt-dlp",
            "faster-whisper",
            "deep-translator",
            "requests",
            "gunicorn",
        ]

        for package in critical_packages:
            assert (
                package in content
            ), f"Critical package '{package}' missing from requirements.txt"

    def test_no_duplicate_packages(self):
        """Test that no packages are duplicated in requirements.txt."""
        req_file = REQUIREMENTS
        lines = req_file.read_text().splitlines()

        packages = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            package = line.split("==")[0].strip().lower()
            packages.append(package)

        duplicates = [pkg for pkg in set(packages) if packages.count(pkg) > 1]
        assert not duplicates, f"Duplicate packages found: {duplicates}"


@pytest.mark.unit
class TestImports:
    """Test that all required modules can be imported."""

    def test_flask_imports(self):
        """Test Flask and related imports."""
        try:
            import flask
            from flask import Flask, jsonify, request
            from flask_cors import CORS
        except ImportError as e:
            pytest.fail(f"Flask import failed: {e}")

    def test_celery_imports(self):
        """Test Celery imports."""
        try:
            import celery
            from celery import Celery
        except ImportError as e:
            pytest.fail(f"Celery import failed: {e}")

    def test_redis_imports(self):
        """Test Redis imports."""
        try:
            import redis
        except ImportError as e:
            pytest.fail(f"Redis import failed: {e}")

    def test_structlog_imports(self):
        """Test structlog imports (this would have caught our bug!)."""
        try:
            import structlog
        except ImportError as e:
            pytest.fail(f"structlog import failed: {e}")

    def test_video_processing_imports(self):
        """Test video processing related imports."""
        try:
            import yt_dlp

            # Note: faster_whisper might not be available in test environment
            # so we'll test it conditionally
        except ImportError as e:
            pytest.fail(f"Video processing import failed: {e}")

    def test_translation_imports(self):
        """Test translation service imports."""
        try:
            from deep_translator import GoogleTranslator
        except ImportError as e:
            pytest.fail(f"Translation import failed: {e}")

    def test_custom_module_imports(self):
        """Test that our custom modules can be imported."""
        # Add backend to path for testing
        # Use relative path from test file to backend directory
        test_file_dir = Path(__file__).parent.resolve()
        backend_path = (
            test_file_dir / ".." / ".."
        )  # Go up from unit/ to tests/ to backend/
        backend_path = backend_path.resolve()

        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        try:
            import config
            import logging_config

            # Test that core.exceptions exists as a file
            exceptions_path = backend_path / "core" / "exceptions.py"
            assert (
                exceptions_path.exists()
            ), f"core/exceptions.py not found at {exceptions_path}"
            # Test that services directory exists and has subtitle_service
            services_path = backend_path / "services" / "subtitle_service.py"
            assert (
                services_path.exists()
            ), f"subtitle_service.py not found at {services_path}"
        except ImportError as e:
            pytest.fail(f"Custom module import failed: {e}")


@pytest.mark.unit
class TestDependencyCompatibility:
    """Test that dependencies work together without conflicts."""

    def test_flask_celery_compatibility(self):
        """Test that Flask and Celery can work together."""
        try:
            from celery import Celery
            from flask import Flask

            # Create minimal Flask app
            app = Flask(__name__)

            # Create minimal Celery app
            celery_app = Celery("test")

            # Should not raise any conflicts
            assert app is not None
            assert celery_app is not None

        except Exception as e:
            pytest.fail(f"Flask-Celery compatibility issue: {e}")

    def test_logging_compatibility(self):
        """Test that structlog works with our logging setup."""
        try:
            import logging

            import structlog

            # Test basic structlog setup
            structlog.configure(
                processors=[structlog.processors.JSONRenderer()],
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                logger_factory=structlog.PrintLoggerFactory(),
                cache_logger_on_first_use=True,
            )

            logger = structlog.get_logger()
            logger.info("Test message", test_key="test_value")

        except Exception as e:
            pytest.fail(f"Logging compatibility issue: {e}")


@pytest.mark.slow
class TestRequirementsInstallation:
    """Test that requirements can be installed cleanly."""

    @pytest.mark.skipif(
        not os.environ.get("TEST_REQUIREMENTS_INSTALL"),
        reason="Set TEST_REQUIREMENTS_INSTALL=1 to run installation tests",
    )
    def test_requirements_install_clean(self):
        """Test that requirements.txt can be installed without conflicts."""
        # This test is slow and potentially destructive, so it's opt-in

        # Create a temporary virtual environment and test installation
        import tempfile
        import venv

        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "test_venv"

            # Create virtual environment
            venv.create(venv_dir, with_pip=True)

            # Install requirements
            pip_path = venv_dir / "bin" / "pip"
            if not pip_path.exists():
                pip_path = venv_dir / "Scripts" / "pip.exe"  # Windows

            req_file = Path("backend/requirements.txt")

            result = subprocess.run(
                [str(pip_path), "install", "-r", str(req_file)],
                capture_output=True,
                text=True,
            )

            assert (
                result.returncode == 0
            ), f"Requirements installation failed: {result.stderr}"

    def test_no_security_vulnerabilities(self):
        """Test for known security vulnerabilities in dependencies."""
        # This would require safety or similar tool
        # For now, just check that we're not using obviously old versions

        req_file = REQUIREMENTS
        content = req_file.read_text()

        # Check for some known vulnerable patterns (adjust as needed)
        vulnerable_patterns = [
            "flask==0.",  # Very old Flask versions
            "requests==2.0",  # Very old requests
            "pyyaml==3.",  # Old PyYAML versions had vulnerabilities
        ]

        for pattern in vulnerable_patterns:
            assert (
                pattern not in content.lower()
            ), f"Potentially vulnerable dependency: {pattern}"


# Helper function to check if package is available
def package_available(package_name: str) -> bool:
    """Check if a package is available for import."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False
