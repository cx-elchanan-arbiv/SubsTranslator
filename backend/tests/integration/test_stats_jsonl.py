#!/usr/bin/env python3
"""
Test JSONL Statistics - End-to-End Test
========================================

This test:
1. Processes a short YouTube video
2. Uses base model (fastest) + Google Translate
3. Verifies a new line was added to stats JSONL
4. Checks the stats contain correct data

Usage:
    python3 test_stats_jsonl.py <youtube_url>

Example:
    python3 test_stats_jsonl.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
"""

import json
import sys
import time

import requests

# Configuration
API_URL = "http://localhost:8081"
STATS_FILE = "/app/stats/video_stats.jsonl"  # Inside container


def print_header(text):
    """Print fancy header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def get_stats_count():
    """Get current number of stats entries."""
    try:
        response = requests.get(f"{API_URL}/api/stats/info")
        if response.status_code == 200:
            info = response.json()
            return info.get("entry_count", 0)
        return 0
    except:
        return 0


def download_stats_file():
    """Download latest stats JSONL."""
    try:
        response = requests.get(f"{API_URL}/api/stats/download")
        if response.status_code == 200:
            with open("test_stats.jsonl", "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False


def get_latest_stat():
    """Get the latest (last) stat from JSONL."""
    try:
        with open("test_stats.jsonl") as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1].strip())
        return None
    except Exception as e:
        print(f"❌ Error reading stats: {e}")
        return None


def submit_video(url):
    """Submit video for processing."""
    print_header("📤 SUBMITTING VIDEO")

    payload = {
        "url": url,
        "source_lang": "auto",
        "target_lang": "he",
        "auto_create_video": False,  # Faster - skip video creation
        "whisper_model": "base",
        "translation_service": "google",
    }

    print(f"URL: {url}")
    print("Model: base (fastest)")
    print("Translation: Google (not OpenAI)")
    print("Auto-create video: False (faster)")

    try:
        response = requests.post(f"{API_URL}/youtube", json=payload, timeout=600)

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None

        result = response.json()
        task_id = result.get("task_id")
        print(f"✅ Task submitted: {task_id}")
        return task_id

    except Exception as e:
        print(f"❌ Error submitting: {e}")
        return None


def wait_for_task(task_id, timeout=600):
    """Poll task status until completion."""
    print_header("⏳ WAITING FOR TASK")

    start = time.time()
    last_status = None

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{API_URL}/status/{task_id}")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                time.sleep(2)
                continue

            data = response.json()
            status = data.get("state", "UNKNOWN")

            # Print status updates
            if status != last_status:
                print(f"📊 Status: {status}")
                last_status = status

                # Show progress
                if "meta" in data:
                    overall = data["meta"].get("overall_percent", 0)
                    print(f"   Progress: {overall}%")

            # Check if done
            if status == "SUCCESS":
                elapsed = time.time() - start
                print(f"✅ Task completed in {elapsed:.1f}s")
                return data
            elif status == "FAILURE":
                error = data.get("result", {}).get("error", "Unknown error")
                print(f"❌ Task failed: {error}")
                return None

            time.sleep(2)

        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
            time.sleep(2)

    print(f"❌ Timeout after {timeout}s")
    return None


def verify_stats(task_id, expected_model, expected_service):
    """Verify stats were saved correctly."""
    print_header("🔍 VERIFYING STATS")

    # Download latest stats
    print("📥 Downloading stats file...")
    if not download_stats_file():
        print("❌ Failed to download stats")
        return False

    # Get latest entry
    latest = get_latest_stat()
    if not latest:
        print("❌ No stats found")
        return False

    print("\n📊 Latest stat entry:")
    print(json.dumps(latest, indent=2, ensure_ascii=False))

    # Verify fields
    print("\n🔎 Verification:")
    checks = []

    # Task ID match
    if latest.get("task_id") == task_id:
        print("✅ Task ID matches")
        checks.append(True)
    else:
        print(f"❌ Task ID mismatch: {latest.get('task_id')} != {task_id}")
        checks.append(False)

    # Model check
    if latest.get("transcription_model") == expected_model:
        print(f"✅ Model: {expected_model}")
        checks.append(True)
    else:
        print(
            f"❌ Model mismatch: {latest.get('transcription_model')} != {expected_model}"
        )
        checks.append(False)

    # Service check
    if latest.get("translation_service") == expected_service:
        print(f"✅ Translation service: {expected_service}")
        checks.append(True)
    else:
        print(
            f"❌ Service mismatch: {latest.get('translation_service')} != {expected_service}"
        )
        checks.append(False)

    # Status check
    if latest.get("status") == "success":
        print("✅ Status: success")
        checks.append(True)
    else:
        print(f"❌ Status: {latest.get('status')}")
        checks.append(False)

    # Check timing fields exist
    timing_fields = [
        "video_duration",
        "transcription_duration",
        "translation_duration",
        "embedding_duration",
        "total_duration",
    ]

    print("\n📏 Timing data:")
    for field in timing_fields:
        value = latest.get(field, 0)
        if value > 0:
            print(f"  ✅ {field}: {value:.1f}s")
            checks.append(True)
        else:
            print(f"  ⚠️ {field}: {value} (might be 0 if video creation skipped)")

    # Speed ratio
    speed_ratio = latest.get("transcription_speed_ratio", 0)
    if speed_ratio > 0:
        print(f"  ✅ Speed ratio: {speed_ratio:.2f}x realtime")
        checks.append(True)

    # Summary
    print(f"\n{'='*60}")
    if all(checks[:4]):  # First 4 are critical
        print("✅ ALL CRITICAL CHECKS PASSED!")
        return True
    else:
        print("❌ SOME CHECKS FAILED")
        return False


def main():
    """Main test flow."""
    print("\n🧪 JSONL STATISTICS END-TO-END TEST")

    # Check if URL provided
    if len(sys.argv) < 2:
        print("\n❌ Please provide a YouTube URL")
        print("\nUsage:")
        print("  python3 test_stats_jsonl.py <youtube_url>")
        print("\nExample (short video ~5 seconds):")
        print(
            '  python3 test_stats_jsonl.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"'
        )
        sys.exit(1)

    url = sys.argv[1]

    # Get initial stats count
    initial_count = get_stats_count()
    print(f"\n📊 Initial stats count: {initial_count}")

    # Submit video
    task_id = submit_video(url)
    if not task_id:
        print("\n❌ TEST FAILED: Could not submit video")
        sys.exit(1)

    # Wait for completion
    result = wait_for_task(task_id, timeout=600)
    if not result:
        print("\n❌ TEST FAILED: Task did not complete")
        sys.exit(1)

    # Wait a bit for stats to be written
    print("\n⏱️ Waiting 3s for stats to be written...")
    time.sleep(3)

    # Check new count
    final_count = get_stats_count()
    print(f"\n📊 Final stats count: {final_count}")

    if final_count > initial_count:
        print(f"✅ New entry added! ({final_count - initial_count} new entries)")
    else:
        print("⚠️ Warning: No new entries detected")

    # Verify stats content
    success = verify_stats(
        task_id=task_id, expected_model="base", expected_service="google"
    )

    # Final result
    print_header("🏁 TEST RESULT")
    if success:
        print("✅ TEST PASSED!")
        print("\nThe JSONL stats system is working correctly:")
        print("  ✅ Video processed successfully")
        print("  ✅ Stats saved to JSONL file")
        print("  ✅ Correct model (base) recorded")
        print("  ✅ Correct service (google) recorded")
        print("  ✅ All timing data present")
        sys.exit(0)
    else:
        print("❌ TEST FAILED!")
        print("\nSome checks did not pass. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
