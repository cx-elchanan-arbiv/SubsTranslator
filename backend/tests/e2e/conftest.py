"""
Configuration for E2E Tests
===========================

Shared fixtures and utilities for end-to-end tests.
"""

import subprocess
import time

import pytest
import requests


@pytest.fixture(scope="session")
def ensure_backend_running():
    """Ensure the backend is running before E2E tests."""
    api_base_url = "http://localhost:8081"
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Backend is running and healthy")
                return api_base_url
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                print(
                    f"⏳ Backend not ready, attempt {attempt + 1}/{max_retries}, waiting {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                pytest.skip("Backend not available - run 'docker compose up -d' first")

    pytest.skip("Backend health check failed after all retries")


@pytest.fixture(scope="session")
def check_required_services():
    """Check that all required services are available."""
    required_services = {
        "backend": "http://localhost:8081/health",
        "redis": "redis://localhost:6379",
        "frontend": "http://localhost",
    }

    available_services = {}

    # Check backend
    try:
        response = requests.get(required_services["backend"], timeout=5)
        available_services["backend"] = response.status_code == 200
    except:
        available_services["backend"] = False

    # Check if Docker Compose is running
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running_services = (
            result.stdout.strip().split("\n") if result.stdout.strip() else []
        )
        available_services["docker_services"] = running_services
    except:
        available_services["docker_services"] = []

    return available_services


@pytest.fixture
def api_client(ensure_backend_running):
    """Provide a configured API client for tests."""

    class APIClient:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.session = requests.Session()
            self.session.timeout = 30

        def post_youtube(self, **kwargs):
            """Submit YouTube processing request."""
            return self.session.post(f"{self.base_url}/youtube", json=kwargs)

        def post_download_only(self, **kwargs):
            """Submit download-only request."""
            return self.session.post(
                f"{self.base_url}/download-video-only", json=kwargs
            )

        def get_status(self, task_id: str):
            """Get task status."""
            return self.session.get(f"{self.base_url}/status/{task_id}")

        def get_health(self):
            """Get health status."""
            return self.session.get(f"{self.base_url}/health")

        def get_translation_services(self):
            """Get available translation services."""
            return self.session.get(f"{self.base_url}/translation-services")

    return APIClient(ensure_backend_running)


@pytest.fixture
def cleanup_downloads():
    """Clean up downloaded files after tests."""
    yield

    # Cleanup logic could go here
    # For now, we rely on the system's cleanup mechanisms
    pass


def pytest_configure(config):
    """Configure pytest for E2E tests."""
    # Add custom markers
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "youtube: Tests that require YouTube access")
    config.addinivalue_line("markers", "openai: Tests that require OpenAI API key")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for E2E tests."""
    # Add markers automatically based on test names/locations
    for item in items:
        # Mark all tests in e2e directory as e2e and slow
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)

        # Mark YouTube tests
        if "youtube" in item.name.lower() or "online_video" in item.name.lower():
            item.add_marker(pytest.mark.youtube)

        # Mark OpenAI tests
        if "openai" in item.name.lower() or "gpt" in item.name.lower():
            item.add_marker(pytest.mark.openai)
