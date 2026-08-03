"""
Test 5: Basic Security Checks
Verifies basic security measures are in place.
"""


def test_no_hardcoded_secrets():
    """Test that there are no hardcoded secrets in code"""
    import os
    import re

    # Check Python files in backend
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")

    secret_patterns = [
        r"sk-[a-zA-Z0-9]{48,}",  # OpenAI keys
        r'password\s*=\s*["\'][^"\']+["\']',  # Passwords
        r'secret\s*=\s*["\'][^"\']+["\']',  # Secrets
    ]

    found_secrets = []

    for root, dirs, files in os.walk(backend_dir):
        # Skip cache directories
        dirs[:] = [d for d in dirs if not d.startswith("__pycache__")]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path) as f:
                        content = f.read()

                    for pattern in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        # Filter out test keys
                        real_matches = [m for m in matches if "test" not in m.lower()]
                        if real_matches:
                            found_secrets.append((file, real_matches))
                except:
                    continue

    assert len(found_secrets) == 0, f"Found hardcoded secrets: {found_secrets}"


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
