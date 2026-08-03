"""
Integration tests for translation services configuration.
Tests with real environment configuration.
"""

import os
import sys

import pytest

# Add backend to path
backend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "backend")
)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


@pytest.mark.integration
class TestRealEnvironmentIntegration:
    """Integration tests with real environment setup."""

    def test_current_env_configuration(self):
        """Test the actual current environment configuration."""
        from app import _is_valid_openai_key
        from config import get_config

        config = get_config()

        # Check what we actually have
        actual_key = config.OPENAI_API_KEY
        is_valid = _is_valid_openai_key(actual_key)

        if actual_key == "your-openai-api-key-here":
            # This is the expected state in development without real key
            assert is_valid is False, "Placeholder key should be detected as invalid"
        elif actual_key and actual_key.startswith("sk-") and len(actual_key) > 20:
            # This would be a real key
            assert is_valid is True, "Real key should be detected as valid"
        else:
            # No key or invalid format
            assert is_valid is False, "Missing or malformed key should be invalid"
