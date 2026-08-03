#!/usr/bin/env python3
"""
Test Runner for SubsTranslator
=============================

Usage:
    python run_tests.py [type] [component]

Types:
    unit        - Run unit tests only (fast)
    integration - Run integration tests only (medium)
    e2e         - Run end-to-end tests only (slow)
    all         - Run all tests (default)

Components:
    backend     - Run backend tests only
    frontend    - Run frontend tests only
    all         - Run all components (default)

Examples:
    python run_tests.py unit backend
    python run_tests.py e2e --verbose
    python run_tests.py all frontend
    python run_tests.py --verbose
    python run_tests.py integration --verbose
"""

import argparse
import os
import re
import subprocess
import sys
import time


def parse_pytest_output(output):
    """Parse pytest output to extract test statistics and individual test results"""
    stats = {
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "collection_errors": 0,
        "error_details": [],
        "individual_tests": [],
    }

    # Look for collection information
    collected_match = re.search(r"collected (\d+) items?", output)
    if collected_match:
        stats["collected"] = int(collected_match.group(1))

    # Look for collection errors
    collection_errors = re.findall(r"(\d+) errors? during collection", output)
    if collection_errors:
        stats["collection_errors"] = int(collection_errors[0])

    # Look for final test results
    result_match = re.search(r"=+ (.+?) =+\s*$", output, re.MULTILINE)
    if result_match:
        result_line = result_match.group(1)

        # Extract numbers from result line
        passed_match = re.search(r"(\d+) passed", result_line)
        if passed_match:
            stats["passed"] = int(passed_match.group(1))

        failed_match = re.search(r"(\d+) failed", result_line)
        if failed_match:
            stats["failed"] = int(failed_match.group(1))

        error_match = re.search(r"(\d+) error", result_line)
        if error_match:
            stats["errors"] = int(error_match.group(1))

        skipped_match = re.search(r"(\d+) skipped", result_line)
        if skipped_match:
            stats["skipped"] = int(skipped_match.group(1))

    # Extract error details
    error_blocks = re.findall(
        r"ERROR collecting (.+?)\n(.+?)(?=ERROR|$)", output, re.DOTALL
    )
    for file_path, error_detail in error_blocks:
        # Look for specific error like IndentationError
        if "IndentationError" in error_detail:
            line_match = re.search(r"line (\d+)", error_detail)
            line_num = line_match.group(1) if line_match else "unknown"
            stats["error_details"].append(f"{file_path} line {line_num}")

    # Extract individual test results with timing
    test_lines = output.split("\n")
    for line in test_lines:
        # Look for test result lines like: "tests/unit/test_something.py::test_name PASSED [10%] 0.02s"
        test_match = re.match(
            r"(.+?)::(.*?)\s+(PASSED|FAILED|SKIPPED|ERROR)(?:\s+\[.*?\])?\s*(?:(\d+\.\d+)s)?",
            line,
        )
        if test_match:
            file_path, test_name, status, duration_str = test_match.groups()
            duration = float(duration_str) if duration_str else 0

            stats["individual_tests"].append(
                {
                    "file": file_path.replace("backend/tests/", ""),
                    "name": test_name,
                    "status": status,
                    "duration": duration,
                }
            )

    return stats


def parse_jest_output(output):
    """Parse Jest output to extract test statistics"""
    stats = {
        "test_suites_passed": 0,
        "test_suites_total": 0,
        "tests_passed": 0,
        "tests_total": 0,
        "snapshots": 0,
        "time": "",
        "success": False,
    }

    # Look for test suite results
    suite_match = re.search(r"Test Suites: (\d+) passed, (\d+) total", output)
    if suite_match:
        stats["test_suites_passed"] = int(suite_match.group(1))
        stats["test_suites_total"] = int(suite_match.group(2))

    # Look for individual test results
    test_match = re.search(r"Tests:\s+(\d+) passed, (\d+) total", output)
    if test_match:
        stats["tests_passed"] = int(test_match.group(1))
        stats["tests_total"] = int(test_match.group(2))

    # Look for time
    time_match = re.search(r"Time:\s+(.+?)$", output, re.MULTILINE)
    if time_match:
        stats["time"] = time_match.group(1).strip()

    # Check if all tests passed
    stats["success"] = (
        stats["test_suites_passed"] == stats["test_suites_total"]
        and stats["tests_passed"] == stats["tests_total"]
    )

    return stats


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def print_test_summary(backend_result=None, frontend_result=None, total_duration=None):
    """Print a comprehensive test summary"""
    print("\n" + "=" * 22 + " 🧪 TEST SUMMARY " + "=" * 22)
    print()

    # Backend summary
    if backend_result:
        print("Backend:")
        stats = backend_result["stats"]
        duration = backend_result.get("duration", 0)
        if stats["collection_errors"] > 0:
            print(f"  ✅ Collected: {stats['collected']} tests")
            print("  ❌ 0 executed (stopped at import errors)")
            print(f"  ❌ {stats['collection_errors']} errors during collection", end="")
            if stats["error_details"]:
                print(f" ({', '.join(stats['error_details'])})")
            else:
                print()
        elif backend_result["returncode"] == 0:
            total_executed = stats["passed"] + stats["failed"] + stats["errors"]
            print(f"  ✅ Collected: {stats['collected']} tests")
            print(f"  ✅ Executed: {total_executed} tests")
            if stats["passed"] > 0:
                print(f"  ✅ Passed: {stats['passed']}")
            if stats["failed"] > 0:
                print(f"  ❌ Failed: {stats['failed']}")
            if stats["errors"] > 0:
                print(f"  ❌ Errors: {stats['errors']}")
            if stats["skipped"] > 0:
                print(f"  ⏭️ Skipped: {stats['skipped']}")

            # Show individual test results if available
            if stats["individual_tests"]:
                print("\n  📋 Individual Test Results:")
                for test in stats["individual_tests"]:
                    status_icon = {
                        "PASSED": "✅",
                        "FAILED": "❌",
                        "SKIPPED": "⏭️",
                        "ERROR": "💥",
                    }.get(test["status"], "❓")
                    duration_str = (
                        f" ({test['duration']:.3f}s)" if test["duration"] > 0 else ""
                    )
                    print(
                        f"    {status_icon} {test['file']}::{test['name']}{duration_str}"
                    )
        else:
            print(
                f"  ❌ Failed to run tests (exit code: {backend_result['returncode']})"
            )
            # Show what we can parse from failed output
            if stats["individual_tests"]:
                print("\n  📋 Partial Test Results:")
                for test in stats["individual_tests"]:
                    status_icon = {
                        "PASSED": "✅",
                        "FAILED": "❌",
                        "SKIPPED": "⏭️",
                        "ERROR": "💥",
                    }.get(test["status"], "❓")
                    duration_str = (
                        f" ({test['duration']:.3f}s)" if test["duration"] > 0 else ""
                    )
                    print(
                        f"    {status_icon} {test['file']}::{test['name']}{duration_str}"
                    )

        if duration > 0:
            print(f"  ⏱️ Duration: {format_duration(duration)}")

    # Frontend summary
    if frontend_result:
        print()
        print("Frontend:")
        stats = frontend_result["stats"]
        duration = frontend_result.get("duration", 0)
        if frontend_result["returncode"] == 0 and stats["success"]:
            print(
                f"  ✅ Test Suites: {stats['test_suites_passed']}/{stats['test_suites_total']} passed"
            )
            print(f"  ✅ Tests: {stats['tests_passed']}/{stats['tests_total']} passed")
            if stats["time"]:
                print(f"  ⏱️ Jest Time: {stats['time']}")
        else:
            print(
                f"  ❌ Test Suites: {stats['test_suites_passed']}/{stats['test_suites_total']} passed"
            )
            print(f"  ❌ Tests: {stats['tests_passed']}/{stats['tests_total']} passed")

        if duration > 0:
            print(f"  ⏱️ Duration: {format_duration(duration)}")

    # Overall result
    print()
    print("Overall Result:")
    backend_ok = not backend_result or (
        backend_result["returncode"] == 0
        and backend_result["stats"]["collection_errors"] == 0
    )
    frontend_ok = not frontend_result or frontend_result["returncode"] == 0

    if backend_result:
        if backend_result["stats"]["collection_errors"] > 0:
            print("  ❌ Backend failed at collection")
        elif backend_result["returncode"] == 0:
            print("  ✅ Backend passed successfully")
        else:
            print("  ❌ Backend failed during execution")

    if frontend_result:
        if frontend_result["returncode"] == 0:
            print("  ✅ Frontend passed successfully")
        else:
            print("  ❌ Frontend failed")

    # Total duration
    if total_duration:
        print()
        print(f"🕐 Total Duration: {format_duration(total_duration)}")

    print()
    print("=" * 61)

    return backend_ok and frontend_ok


def run_backend_tests(test_type="all", verbose=False):
    """Run backend tests"""
    print(f"🔧 Running backend {test_type} tests...")

    start_time = time.time()

    cmd = ["python3", "-m", "pytest"]

    if test_type == "unit":
        cmd.extend(["backend/tests/unit", "-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["backend/tests/integration", "-m", "integration"])
    elif test_type == "e2e":
        cmd.extend(["backend/tests/e2e", "-m", "e2e"])
    else:
        cmd.append("backend/tests")

    # Always run with verbose to get individual test results
    cmd.extend(["-v"])

    if verbose:
        cmd.extend(["-s"])
        # For verbose mode, show output in real-time
        result = subprocess.run(cmd, text=True)
        # Still need to capture output for parsing, so run again quietly
        result_for_parsing = subprocess.run(
            cmd[:-1], capture_output=True, text=True
        )  # Remove -s flag
        output = result_for_parsing.stdout + result_for_parsing.stderr
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

    duration = time.time() - start_time

    # Parse output for statistics
    stats = parse_pytest_output(output)

    return {
        "returncode": result.returncode,
        "output": output,
        "stats": stats,
        "duration": duration,
    }


def run_frontend_tests(test_type="all", verbose=False):
    """Run frontend tests"""
    print(f"⚛️  Running frontend {test_type} tests...")

    start_time = time.time()

    os.chdir("frontend")

    if test_type == "unit":
        # Frontend doesn't separate unit tests by folder, so run all Jest tests
        cmd = ["npm", "test", "--", "--watchAll=false"]
    elif test_type == "integration":
        # Frontend doesn't have integration tests separate from unit tests
        cmd = ["npm", "test", "--", "--watchAll=false"]
    elif test_type == "e2e":
        cmd = ["npx", "playwright", "test", "tests/e2e"]
    else:
        cmd = ["npm", "test", "--", "--watchAll=false"]

    if verbose and test_type != "e2e":  # npm test supports --verbose
        cmd.append("--verbose")
    elif verbose and test_type == "e2e":  # Playwright has different verbose flag
        cmd.append("--reporter=list")

    if verbose:
        # For verbose mode, show output in real-time
        result = subprocess.run(cmd, text=True)
        output = ""  # No output to parse since we showed it live
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

    os.chdir("..")

    duration = time.time() - start_time

    # Parse output for statistics
    stats = parse_jest_output(output)

    return {
        "returncode": result.returncode,
        "output": output,
        "stats": stats,
        "duration": duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SubsTranslator tests")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["unit", "integration", "e2e", "all"],
        help="Type of tests to run",
    )
    parser.add_argument(
        "component",
        nargs="?",
        default="all",
        choices=["backend", "frontend", "all"],
        help="Component to test",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print(f"🚀 Running {args.test_type} tests for {args.component}")
    if args.verbose:
        print("📝 Verbose mode enabled")

    backend_result = None
    frontend_result = None

    # Run backend tests
    if args.component in ["backend", "all"]:
        backend_result = run_backend_tests(args.test_type, args.verbose)

    # Run frontend tests
    if args.component in ["frontend", "all"]:
        frontend_result = run_frontend_tests(args.test_type, args.verbose)

    # Print comprehensive summary (skip if verbose since we already saw everything)
    if args.verbose:
        print("\n" + "🎉 Tests completed! Check the output above for results.")
        # Simple success check
        backend_ok = not backend_result or backend_result["returncode"] == 0
        frontend_ok = not frontend_result or frontend_result["returncode"] == 0
        all_passed = backend_ok and frontend_ok
    else:
        all_passed = print_test_summary(backend_result, frontend_result)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
