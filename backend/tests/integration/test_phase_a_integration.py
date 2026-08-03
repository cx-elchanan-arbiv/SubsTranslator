#!/usr/bin/env python3
"""
Integration Test for Phase A - YouTube Bot Detection UX Improvements
Tests all backend changes including feature flags, error handling, and API endpoints.
"""

import json
import os
import sys
from typing import Any

import requests

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8081")
TEST_YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
)


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = (
        f"{Colors.GREEN}✓ PASS{Colors.END}"
        if passed
        else f"{Colors.RED}✗ FAIL{Colors.END}"
    )
    print(f"{status} {name}")
    if details:
        print(f"     {Colors.YELLOW}{details}{Colors.END}")


def test_features_endpoint() -> dict[str, Any]:
    """Test 1: Check /api/features endpoint exists and returns feature flags"""
    print_header("TEST 1: Feature Flags API Endpoint")

    try:
        response = requests.get(f"{API_BASE_URL}/api/features", timeout=5)

        # Check status code
        test_passed = response.status_code == 200
        print_test("Status Code 200", test_passed, f"Got: {response.status_code}")

        if not test_passed:
            return {"passed": False, "data": None}

        # Check JSON response
        data = response.json()
        print_test("Response is JSON", True, f"Keys: {list(data.keys())}")

        # Check for youtube_download_enabled field
        has_flag = "youtube_download_enabled" in data
        print_test(
            "Has youtube_download_enabled field",
            has_flag,
            f"Value: {data.get('youtube_download_enabled')}",
        )

        # Check flag value type
        if has_flag:
            is_bool = isinstance(data["youtube_download_enabled"], bool)
            print_test(
                "Flag is boolean type",
                is_bool,
                f"Type: {type(data['youtube_download_enabled']).__name__}",
            )

        return {"passed": has_flag, "data": data}

    except Exception as e:
        print_test("API Endpoint Available", False, f"Error: {str(e)}")
        return {"passed": False, "data": None}


def test_backend_config() -> bool:
    """Test 2: Check backend config.py has ENABLE_YOUTUBE_DOWNLOAD"""
    print_header("TEST 2: Backend Configuration")

    try:
        # Import config
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
        from config import Config

        # Check if ENABLE_YOUTUBE_DOWNLOAD exists
        has_attr = hasattr(Config, "ENABLE_YOUTUBE_DOWNLOAD")
        print_test(
            "Config has ENABLE_YOUTUBE_DOWNLOAD attribute",
            has_attr,
            f"Value: {getattr(Config, 'ENABLE_YOUTUBE_DOWNLOAD', None)}",
        )

        if has_attr:
            # Check default value
            default_value = Config.ENABLE_YOUTUBE_DOWNLOAD
            is_true = default_value is True
            print_test("Default value is True", is_true, f"Value: {default_value}")

        return has_attr

    except Exception as e:
        print_test("Config Import", False, f"Error: {str(e)}")
        return False


def test_exception_classes() -> bool:
    """Test 3: Check exception classes exist with correct structure"""
    print_header("TEST 3: Exception Classes")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
        from core.exceptions import YouTubeBotDetectionError, handle_youtube_error

        # Test YouTubeBotDetectionError instantiation
        test_url = "https://youtube.com/watch?v=test"
        error = YouTubeBotDetectionError(test_url)

        print_test(
            "YouTubeBotDetectionError instantiates",
            True,
            f"Error code: {error.error_code}",
        )

        # Check error code
        has_code = error.error_code == "YOUTUBE_BOT_DETECTION"
        print_test(
            "Error code is YOUTUBE_BOT_DETECTION", has_code, f"Code: {error.error_code}"
        )

        # Check user message exists
        has_message = bool(error.user_message)
        print_test(
            "Has user-facing message",
            has_message,
            f"Message length: {len(error.user_message)} chars",
        )

        # Check Hebrew in message
        has_hebrew = any("\u0590" <= c <= "\u05ff" for c in error.user_message)
        print_test(
            "User message contains Hebrew",
            has_hebrew,
            f"First 50 chars: {error.user_message[:50]}...",
        )

        # Check to_dict method
        error_dict = error.to_dict()
        has_dict_method = isinstance(error_dict, dict)
        print_test(
            "to_dict() returns dictionary",
            has_dict_method,
            f"Keys: {list(error_dict.keys())}",
        )

        # Test handle_youtube_error function
        test_exception = Exception("Sign in to confirm you're not a bot")
        handled_error = handle_youtube_error(test_exception, test_url)

        is_bot_detection = isinstance(handled_error, YouTubeBotDetectionError)
        print_test(
            "handle_youtube_error detects bot detection",
            is_bot_detection,
            f"Type: {type(handled_error).__name__}",
        )

        return has_code and has_message and is_bot_detection

    except Exception as e:
        print_test("Exception Classes", False, f"Error: {str(e)}")
        import traceback

        print(f"{Colors.RED}{traceback.format_exc()}{Colors.END}")
        return False


def test_translation_files() -> bool:
    """Test 4: Check translation files have YouTube bot detection messages"""
    print_header("TEST 4: Translation Files")

    languages = {"he": "Hebrew", "en": "English", "ar": "Arabic", "es": "Spanish"}

    all_passed = True

    for lang_code, lang_name in languages.items():
        try:
            file_path = f"frontend/public/locales/{lang_code}/common.json"

            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check if errors.youtubeBotDetection exists
            has_section = "errors" in data and "youtubeBotDetection" in data["errors"]

            if has_section:
                bot_section = data["errors"]["youtubeBotDetection"]
                required_keys = [
                    "title",
                    "message",
                    "alternativeTitle",
                    "step1",
                    "step2",
                    "step3",
                    "switchButton",
                ]
                has_all_keys = all(key in bot_section for key in required_keys)

                print_test(
                    f"{lang_name} ({lang_code}) has all keys",
                    has_all_keys,
                    f"Keys: {list(bot_section.keys())}",
                )

                all_passed = all_passed and has_all_keys
            else:
                print_test(
                    f"{lang_name} ({lang_code}) has youtubeBotDetection section",
                    False,
                    "Section missing",
                )
                all_passed = False

        except Exception as e:
            print_test(
                f"{lang_name} ({lang_code}) file readable", False, f"Error: {str(e)}"
            )
            all_passed = False

    return all_passed


def test_frontend_components() -> bool:
    """Test 5: Check frontend component files exist and have correct structure"""
    print_header("TEST 5: Frontend Components")

    try:
        # Check ProgressDisplay.tsx
        progress_display_path = "frontend/src/components/ProgressDisplay.tsx"
        with open(progress_display_path, encoding="utf-8") as f:
            content = f.read()

        # Check for bot detection handling
        has_bot_check = "isYoutubeBotDetection" in content
        print_test(
            "ProgressDisplay has bot detection check",
            has_bot_check,
            "Found: isYoutubeBotDetection variable",
        )

        # Check for YOUTUBE_BOT_DETECTION error code check
        has_error_code = "YOUTUBE_BOT_DETECTION" in content
        print_test(
            "ProgressDisplay checks error_code",
            has_error_code,
            "Found: YOUTUBE_BOT_DETECTION string",
        )

        # Check for translation keys
        has_translations = "errors.youtubeBotDetection" in content
        print_test(
            "ProgressDisplay uses translation keys",
            has_translations,
            "Found: errors.youtubeBotDetection",
        )

        # Check App.tsx
        app_path = "frontend/src/App.tsx"
        with open(app_path, encoding="utf-8") as f:
            app_content = f.read()

        # Check for feature flag state
        has_flag_state = "youtubeDownloadEnabled" in app_content
        print_test(
            "App.tsx has feature flag state",
            has_flag_state,
            "Found: youtubeDownloadEnabled",
        )

        # Check for /api/features fetch
        has_fetch = "/api/features" in app_content
        print_test(
            "App.tsx fetches feature flags",
            has_fetch,
            "Found: /api/features endpoint call",
        )

        # Check Tabs.tsx
        tabs_path = "frontend/src/components/Tabs.tsx"
        with open(tabs_path, encoding="utf-8") as f:
            tabs_content = f.read()

        # Check for youtubeEnabled prop
        has_youtube_enabled = "youtubeEnabled" in tabs_content
        print_test(
            "Tabs component has youtubeEnabled prop",
            has_youtube_enabled,
            "Found: youtubeEnabled prop",
        )

        # Check for conditional rendering
        has_conditional = (
            "youtubeEnabled &&" in tabs_content or "{youtubeEnabled &&" in tabs_content
        )
        print_test(
            "Tabs conditionally renders YouTube tab",
            has_conditional,
            "Found: conditional rendering logic",
        )

        return (
            has_bot_check
            and has_error_code
            and has_translations
            and has_flag_state
            and has_fetch
            and has_youtube_enabled
            and has_conditional
        )

    except Exception as e:
        print_test("Frontend Components", False, f"Error: {str(e)}")
        return False


def test_env_files() -> bool:
    """Test 6: Check .env example files have ENABLE_YOUTUBE_DOWNLOAD"""
    print_header("TEST 6: Environment Files")

    env_files = [".env.local", ".env.example", "backend/.env.example"]

    all_passed = True

    for env_file in env_files:
        try:
            with open(env_file, encoding="utf-8") as f:
                content = f.read()

            has_flag = "ENABLE_YOUTUBE_DOWNLOAD" in content
            print_test(
                f"{env_file} has ENABLE_YOUTUBE_DOWNLOAD",
                has_flag,
                "Found in file" if has_flag else "Not found",
            )

            all_passed = all_passed and has_flag

        except Exception as e:
            print_test(f"{env_file} readable", False, f"Error: {str(e)}")
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all integration tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Phase A Integration Test - YouTube Bot Detection UX          ║")
    print("║                 Testing All Components                            ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    results = {}

    # Run tests
    results["features_api"] = test_features_endpoint()
    results["backend_config"] = test_backend_config()
    results["exceptions"] = test_exception_classes()
    results["translations"] = test_translation_files()
    results["frontend_components"] = test_frontend_components()
    results["env_files"] = test_env_files()

    # Summary
    print_header("TEST SUMMARY")

    total_tests = len(results)
    passed_tests = sum(
        1
        for r in results.values()
        if (r if isinstance(r, bool) else r.get("passed", False))
    )

    print(f"Total Tests: {total_tests}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.END}")
    print(f"{Colors.RED}Failed: {total_tests - passed_tests}{Colors.END}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%\n")

    # Detailed results
    for test_name, result in results.items():
        passed = result if isinstance(result, bool) else result.get("passed", False)
        status = (
            f"{Colors.GREEN}✓{Colors.END}" if passed else f"{Colors.RED}✗{Colors.END}"
        )
        print(f"{status} {test_name}")

    # Feature flag data
    if results["features_api"].get("data"):
        print(f"\n{Colors.BOLD}Feature Flags:{Colors.END}")
        for key, value in results["features_api"]["data"].items():
            print(f"  - {key}: {value}")

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
