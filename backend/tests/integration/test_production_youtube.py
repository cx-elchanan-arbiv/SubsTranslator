#!/usr/bin/env python3
"""
Test YouTube transcription against a deployed API.
Set API_BASE environment variable to your production URL.
Example: API_BASE=https://api.example.com python test_production_youtube.py
"""

import json
import os
import time

import requests

# Allow overriding via environment variable
API_BASE = os.getenv("API_BASE", "http://localhost:8081")


def test_youtube_transcription():
    """Test YouTube URL processing"""

    # Short test video (30 seconds)
    youtube_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    print(f"🎬 Testing YouTube transcription: {youtube_url}")
    print(f"📡 API: {API_BASE}")
    print()

    # Step 1: Submit transcription request
    print("1️⃣  Submitting transcription request...")
    response = requests.post(
        f"{API_BASE}/youtube",
        json={"url": youtube_url, "source_language": "en", "whisper_model": "base"},
        timeout=30,
    )

    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:200]}")

    if response.status_code not in [200, 202]:
        print(f"❌ Failed to submit request: {response.status_code}")
        print(f"   {response.text}")
        return False

    data = response.json()
    task_id = data.get("task_id")

    if not task_id:
        print(f"❌ No task_id in response: {data}")
        return False

    print(f"✅ Task created: {task_id}")
    print()

    # Step 2: Poll for status
    print("2️⃣  Polling for task completion...")
    max_attempts = 60  # 5 minutes max
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        time.sleep(5)

        status_response = requests.get(f"{API_BASE}/status/{task_id}", timeout=10)

        if status_response.status_code != 200:
            print(f"   ⚠️  Status check failed: {status_response.status_code}")
            continue

        status_data = status_response.json()
        state = status_data.get("state", "UNKNOWN")
        progress = status_data.get("progress", 0)

        print(f"   [{attempt}] State: {state}, Progress: {progress}%")

        if state == "SUCCESS":
            print()
            print("✅ Task completed successfully!")
            print(f"   Result: {json.dumps(status_data, indent=2)[:500]}")
            return True

        elif state == "FAILURE":
            error = status_data.get("error", "Unknown error")
            print()
            print(f"❌ Task failed: {error}")
            print(f"   Full response: {json.dumps(status_data, indent=2)}")
            return False

        elif state in ["PENDING", "STARTED", "PROGRESS"]:
            # Still processing
            continue

        else:
            print(f"   ⚠️  Unknown state: {state}")

    print()
    print(f"❌ Timeout after {max_attempts} attempts")
    return False


if __name__ == "__main__":
    success = test_youtube_transcription()
    exit(0 if success else 1)
