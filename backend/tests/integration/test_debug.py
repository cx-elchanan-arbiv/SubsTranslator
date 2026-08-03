"""Quick debug script to see what's wrong with endpoints."""

import os

os.environ["DISABLE_RATE_LIMIT"] = "1"
os.environ["TESTING"] = "1"

import shutil
import tempfile

from tests.integration.ffmpeg_helpers import make_video

# Create temp dirs
temp_root = tempfile.mkdtemp(prefix="debug_")
uploads = os.path.join(temp_root, "uploads")
downloads = os.path.join(temp_root, "downloads")
os.makedirs(uploads)
os.makedirs(downloads)

# Import app AFTER setting env vars
from app import app

app.config["UPLOAD_FOLDER"] = uploads
app.config["DOWNLOADS_FOLDER"] = downloads
app.config["TESTING"] = True

# Create test video
video_path = os.path.join(uploads, "test.mp4")
print(f"Creating video at: {video_path}")
success = make_video(video_path, color="red", seconds=2)
print(f"Video created: {success}")

# Test cut-video endpoint
with app.test_client() as client:
    with open(video_path, "rb") as f:
        data = {
            "video": (f, "test.mp4"),
            "start_time": "00:00:00",
            "end_time": "00:00:01",
        }

        print("\n=== Testing /cut-video ===")
        response = client.post(
            "/cut-video", data=data, content_type="multipart/form-data"
        )
        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(f"Error: {response.get_json()}")
        else:
            print(f"Success! Size: {len(response.data)} bytes")

# Cleanup
shutil.rmtree(temp_root, ignore_errors=True)
