"""
Tests to verify OpenAI API configuration and security.
This prevents issues where API keys are hardcoded or not properly configured.
"""

import re
from pathlib import Path


class TestOpenAISecurityAndConfiguration:
    """Test OpenAI API security and configuration."""

    def test_openai_translator_accepts_explicit_api_key(self):
        """Test that OpenAI translator can accept API key as parameter."""
        from services.translation_services import OpenAITranslator

        # Should not raise error with explicit API key
        translator = OpenAITranslator(api_key="sk-test-explicit-key")
        assert translator.api_key == "sk-test-explicit-key"

    def test_openai_client_initialization(self):
        """Test that OpenAI client is properly initialized."""
        from services.translation_services import OpenAITranslator

        translator = OpenAITranslator(api_key="sk-test-client-key")

        # Verify client is created
        assert hasattr(translator, "client")
        assert translator.client is not None

        # Verify client has the correct API key
        assert translator.client.api_key == "sk-test-client-key"

    def test_openai_environment_variable_usage(self):
        """Test that OpenAI uses environment variable correctly."""
        from config import get_config

        # Verify config reads from environment
        config = get_config()
        assert hasattr(config, "OPENAI_API_KEY")

        # Check that OPENAI_API_KEY attribute exists and is a string
        assert isinstance(config.OPENAI_API_KEY, str)

        # In CI with real secrets, key should start with 'sk-'
        # In tests without secrets, it may be a placeholder
        # Only validate format if it looks like a real key
        if config.OPENAI_API_KEY.startswith("sk-"):
            assert len(config.OPENAI_API_KEY) > 10, "API key too short"

    def test_no_hardcoded_api_keys_anywhere(self):
        """Test that there are no hardcoded OpenAI API keys anywhere in the codebase."""
        # Pattern to match real OpenAI API keys (not test keys)
        api_key_patterns = [
            r"sk-proj-[A-Za-z0-9]{48}T3BlbkFJ[A-Za-z0-9]{24}",  # New format
            r"sk-[A-Za-z0-9]{48}",  # Old format
        ]

        # Get project root
        project_root = Path(__file__).parent.parent

        # Check important files
        files_to_check = [
            project_root / "backend" / "**" / "*.py",
            project_root / "frontend" / "**" / "*.js",
            project_root / "frontend" / "**" / "*.ts",
            project_root / "frontend" / "**" / "*.tsx",
            project_root / "docker-compose.yml",
            project_root / ".github" / "**" / "*.yml",
            project_root / "**" / ".env",  # Check .env files too
            project_root / "**" / "*.env",  # Check any .env files
        ]

        hardcoded_keys = []
        for pattern_path in files_to_check:
            for file_path in project_root.glob(
                str(pattern_path).replace(str(project_root) + "/", "")
            ):
                try:
                    if file_path.is_file():
                        content = file_path.read_text(encoding="utf-8")

                        # Skip test files, conftest, and .idea files
                        if any(
                            skip in str(file_path).lower()
                            for skip in ["test", "conftest", ".idea", "node_modules"]
                        ):
                            continue

                        for pattern in api_key_patterns:
                            matches = re.findall(pattern, content)
                            # Filter out test keys
                            real_matches = [
                                m for m in matches if not m.startswith("sk-test")
                            ]
                            if real_matches:
                                hardcoded_keys.append((str(file_path), real_matches))

                except Exception:
                    continue  # Skip files that can't be read

        assert (
            len(hardcoded_keys) == 0
        ), f"Found hardcoded API keys in: {hardcoded_keys}"

    def test_openai_uses_environment_not_hardcoded(self):
        """Test that OpenAI configuration uses os.getenv, not hardcoded values."""
        backend_dir = Path(__file__).parent.parent / "backend"
        config_file = backend_dir / "config.py"

        if config_file.exists():
            content = config_file.read_text()

            # Should use os.getenv for API key
            assert 'os.getenv("OPENAI_API_KEY")' in content

            # Should NOT have hardcoded API keys
            assert "sk-proj-" not in content or "sk-test" in content  # Allow test keys

    def test_translation_services_uses_config_not_hardcoded(self):
        """Test that translation services uses config, not hardcoded API keys."""
        backend_dir = Path(__file__).parent.parent / "backend"
        translation_file = backend_dir / "translation_services.py"

        if translation_file.exists():
            content = translation_file.read_text()

            # Should use config.OPENAI_API_KEY
            assert "config.OPENAI_API_KEY" in content

            # Should NOT have hardcoded API keys
            assert "sk-proj-" not in content or "sk-test" in content  # Allow test keys

    def test_github_secrets_integration(self):
        """Test that the code is ready for GitHub Secrets integration."""
        from config import get_config

        config = get_config()

        # Verify that OPENAI_API_KEY is read from environment
        assert hasattr(config, "OPENAI_API_KEY")

        # In CI with GitHub secrets, this should be properly set
        # This test documents the expected behavior
        assert config.OPENAI_API_KEY is not None  # conftest sets this for tests
