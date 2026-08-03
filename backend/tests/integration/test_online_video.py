#!/usr/bin/env python3
"""
Online Video Test Runner
========================

Specialized test runner for YouTube/Online Video E2E tests.

Usage:
    python test_online_video.py [options]

Options:
    --model MODEL       Test specific Whisper model (tiny, large, all)
    --service SERVICE   Test specific translation service (google, openai, openai-tiny, openai-large, all)
    --test TEST         Run specific test (tiny-google, tiny-openai, large-google, large-openai, download-only, error-handling, all)
    --download-only     Test only download functionality (deprecated, use --test download-only)
    --verbose           Verbose output
    --no-cleanup        Don't clean up after tests

Examples:
    python test_online_video.py --test tiny-google         # Run single test: tiny model + Google
    python test_online_video.py --test tiny-openai         # Run single test: tiny model + OpenAI
    python test_online_video.py --test large-google        # Run single test: large model + Google
    python test_online_video.py --test large-openai        # Run single test: large model + OpenAI
    python test_online_video.py --test download-only       # Test only download functionality
    python test_online_video.py --test error-handling      # Test error handling
    python test_online_video.py --model tiny --service google  # Legacy: same as --test tiny-google
    python test_online_video.py --verbose
"""

import argparse
import subprocess
import sys


def run_pytest_command(args: list, verbose: bool = False) -> int:
    """Run pytest with the given arguments."""
    cmd = ["python", "-m", "pytest"] + args

    if verbose:
        cmd.extend(["-v", "-s"])

    print(f"🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Online Video E2E tests")
    parser.add_argument(
        "--model",
        choices=["tiny", "large", "all"],
        default="all",
        help="Whisper model to test",
    )
    parser.add_argument(
        "--service",
        choices=["google", "openai", "openai-tiny", "openai-large", "all"],
        default="all",
        help="Translation service to test",
    )
    parser.add_argument(
        "--test",
        choices=[
            "tiny-google",
            "tiny-openai",
            "large-google",
            "large-openai",
            "download-only",
            "error-handling",
            "all",
        ],
        default="all",
        help="Run specific test combination",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Test only download functionality (deprecated, use --test download-only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Don't clean up after tests"
    )

    args = parser.parse_args()

    # Base pytest arguments
    pytest_args = ["backend/tests/e2e/test_online_video_workflows.py", "-m", "e2e"]

    if not args.no_cleanup:
        pytest_args.append("--tb=short")

    # Filter by specific test if requested
    if args.test != "all":
        # Handle new --test parameter
        if args.test == "tiny-google":
            pytest_args.extend(["-k", "tiny and google"])
        elif args.test == "tiny-openai":
            pytest_args.extend(["-k", "tiny and openai"])
        elif args.test == "large-google":
            pytest_args.extend(["-k", "large and google"])
        elif args.test == "large-openai":
            pytest_args.extend(["-k", "large and openai"])
        elif args.test == "download-only":
            pytest_args.extend(["-k", "download_only"])
        elif args.test == "error-handling":
            pytest_args.extend(["-k", "error_handling"])
    elif args.download_only:
        # Handle deprecated --download-only flag
        pytest_args.extend(["-k", "download_only"])
    elif args.model != "all" or args.service != "all":
        # Handle legacy --model and --service combinations
        if args.service == "openai-tiny":
            filter_expr = "(tiny and openai) or download_only or error_handling"
        elif args.service == "openai-large":
            filter_expr = "(large and openai) or download_only or error_handling"
        elif args.model != "all" and args.service != "all":
            # Both model and service specified
            service_name = (
                args.service.replace("openai-", "")
                if args.service.startswith("openai-")
                else args.service
            )
            filter_expr = (
                f"({args.model} and {service_name}) or download_only or error_handling"
            )
        elif args.model != "all":
            filter_expr = f"{args.model} or download_only or error_handling"
        elif args.service != "all":
            # Handle regular service names
            service_name = (
                args.service.replace("openai-", "")
                if args.service.startswith("openai-")
                else args.service
            )
            filter_expr = f"{service_name} or download_only or error_handling"

        pytest_args.extend(["-k", filter_expr])

    # Check if backend is running
    print("🔍 Checking if backend is running...")
    health_check = subprocess.run(
        ["curl", "-s", "http://localhost:8081/health"], capture_output=True
    )

    if health_check.returncode != 0:
        print("❌ Backend not running. Please start it first:")
        print("   docker compose up -d")
        return 1

    print("✅ Backend is running")

    # Check if required services are available
    print("🔍 Checking translation services...")
    services_check = subprocess.run(
        ["curl", "-s", "http://localhost:8081/translation-services"],
        capture_output=True,
        text=True,
    )

    if services_check.returncode == 0:
        print("✅ Translation services available")
    else:
        print("⚠️  Could not verify translation services")

    # Run the tests
    print("\n🎯 Running Online Video tests...")
    if args.test != "all":
        print(f"   Test: {args.test}")
    else:
        print(f"   Model: {args.model}")
        print(f"   Service: {args.service}")
        print(f"   Download only: {args.download_only}")
    print()

    return run_pytest_command(pytest_args, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
