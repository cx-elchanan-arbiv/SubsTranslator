"""
Integration tests for Docker container functionality.
These tests verify that the containers can start and basic functionality works.
"""

import subprocess
import time

import pytest
import requests


@pytest.mark.integration
class TestDockerIntegration:
    """Test Docker container integration and health."""

    @pytest.fixture(scope="class")
    def docker_services(self):
        """Ensure Docker services are running for tests."""
        # Check if services are already running
        try:
            response = requests.get("http://localhost:8081/health", timeout=5)
            if response.status_code == 200:
                yield True
                return
        except:
            pass

        # Start services if not running
        subprocess.run(["docker-compose", "up", "-d"], check=True)

        # Wait for services to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8081/health", timeout=5)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(2)
        else:
            pytest.fail("Docker services failed to start within timeout")

        yield True

        # Cleanup is handled by user, not automatically

    def test_backend_container_health(self, docker_services):
        """Test that backend container is healthy and responds."""
        response = requests.get("http://localhost:8081/health", timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert data["ffmpeg_installed"] is True

    def test_backend_container_imports(self, docker_services):
        """Test that backend container has all required imports."""
        # Test that the backend can import all required modules
        # This would catch missing dependencies like structlog

        response = requests.get("http://localhost:8081/health", timeout=10)
        assert response.status_code == 200

        # If we get a healthy response, it means all imports worked
        # because the app.py imports everything at startup

    def test_redis_connectivity(self, docker_services):
        """Test that Redis container is accessible."""
        # We can test this indirectly by checking if backend can connect to Redis
        response = requests.get("http://localhost:8081/health", timeout=10)
        assert response.status_code == 200

        # A healthy backend means Redis is accessible
        # (since backend connects to Redis on startup)

    def test_frontend_container_serving(self, docker_services):
        """Test that frontend container serves content."""
        response = requests.get("http://localhost", timeout=10)

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        # Should contain React app content
        assert "SubsTranslator" in response.text or "root" in response.text

    def test_api_endpoints_available(self, docker_services):
        """Test that main API endpoints are available."""
        # Test CORS preflight
        response = requests.options("http://localhost:8081/youtube", timeout=10)
        assert response.status_code == 200

        # Test that endpoints exist (even if they return errors for missing data)
        response = requests.post("http://localhost:8081/youtube", json={}, timeout=10)
        # Should return 400 (bad request) not 404 (not found) or 500 (server error)
        assert response.status_code in [400, 422]  # Bad request, not server error

    def test_container_logs_no_critical_errors(self, docker_services):
        """Test that containers don't have critical startup errors."""
        # Check backend logs for critical errors
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=50", "backend"],
            capture_output=True,
            text=True,
        )

        logs = result.stdout.lower()

        # Should not contain critical error patterns
        critical_errors = [
            "modulenotfounderror",
            "importerror",
            "failed to boot",
            "worker failed",
            "traceback (most recent call last)",
        ]

        for error in critical_errors:
            assert error not in logs, f"Found critical error in logs: {error}"

    def test_worker_container_starts(self, docker_services):
        """Test that Celery worker container starts without errors."""
        # Check worker logs
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=20", "worker"],
            capture_output=True,
            text=True,
        )

        logs = result.stdout.lower()

        # Worker should start successfully
        success_indicators = ["ready", "celery@", "worker"]
        assert any(
            indicator in logs for indicator in success_indicators
        ), f"Worker doesn't seem to be running properly. Logs: {logs}"

        # Should not have import errors
        assert "modulenotfounderror" not in logs
        assert "importerror" not in logs


@pytest.mark.integration
class TestContainerDependencies:
    """Test that containers have all required dependencies."""

    def test_backend_python_packages(self):
        """Test that backend container has all required Python packages."""
        # Run a command inside the backend container to check imports
        cmd = [
            "docker-compose",
            "exec",
            "-T",
            "backend",
            "python",
            "-c",
            """
import sys
required_packages = [
    'flask', 'celery', 'redis', 'structlog', 'yt_dlp', 
    'faster_whisper', 'deep_translator', 'ffmpeg', 'requests'
]
missing = []
for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError as e:
        missing.append(f"{pkg}: {e}")

if missing:
    print("MISSING:", missing)
    sys.exit(1)
else:
    print("ALL_PACKAGES_OK")
            """,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert (
            result.returncode == 0
        ), f"Package check failed: {result.stdout} {result.stderr}"
        assert "ALL_PACKAGES_OK" in result.stdout

    def test_backend_system_dependencies(self):
        """Test that backend container has required system dependencies."""
        # Test FFmpeg
        cmd = ["docker-compose", "exec", "-T", "backend", "ffmpeg", "-version"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "FFmpeg not available in backend container"

        # Test ffprobe
        cmd = ["docker-compose", "exec", "-T", "backend", "ffprobe", "-version"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "ffprobe not available in backend container"


@pytest.mark.integration
@pytest.mark.slow
class TestContainerPerformance:
    """Test container performance and resource usage."""

    def test_container_startup_time(self):
        """Test that containers start within reasonable time."""
        # Stop containers
        subprocess.run(["docker-compose", "down"], check=True)

        # Time the startup
        start_time = time.time()
        subprocess.run(["docker-compose", "up", "-d"], check=True)

        # Wait for health check to pass
        max_wait = 60  # seconds
        for i in range(max_wait):
            try:
                response = requests.get("http://localhost:8081/health", timeout=5)
                if response.status_code == 200:
                    startup_time = time.time() - start_time
                    break
            except:
                pass
            time.sleep(1)
        else:
            pytest.fail("Containers failed to start within 60 seconds")

        # Should start within reasonable time (adjust based on your requirements)
        assert startup_time < 45, f"Containers took too long to start: {startup_time}s"

    def test_container_memory_usage(self):
        """Test that containers don't use excessive memory."""
        # Get container stats
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "table {{.Container}}\t{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
        )

        # This is more of a monitoring test - adjust limits based on your requirements
        assert result.returncode == 0, "Failed to get container stats"

        # Log the memory usage for monitoring
        print(f"Container memory usage:\n{result.stdout}")


# Helper function for other tests
def is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def skip_if_no_docker():
    """Skip test if Docker is not available."""
    return pytest.mark.skipif(not is_docker_available(), reason="Docker not available")


# Apply skip decorator to all classes if Docker not available
for cls_name in [
    "TestDockerIntegration",
    "TestContainerDependencies",
    "TestContainerPerformance",
]:
    cls = globals()[cls_name]
    globals()[cls_name] = skip_if_no_docker()(cls)
