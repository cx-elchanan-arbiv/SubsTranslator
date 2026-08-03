#!/usr/bin/env python3
"""
Test script to verify video quality download logic.
Shows what formats are available and what gets downloaded.
"""

import json
import os
import subprocess
import sys

import yt_dlp

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config

config = get_config()


def test_video_quality(url: str):
    """Test what quality gets downloaded for a given URL"""
    print("=" * 80)
    print(f"🎯 Testing video quality for: {url}")
    print("=" * 80)
    print()

    # Step 1: Extract available formats
    print("📊 Step 1: Extracting available formats...")
    print("-" * 80)

    temp_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["web_creator"]}},
    }

    try:
        with yt_dlp.YoutubeDL(temp_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            print(f"📹 Title: {info.get('title', 'Unknown')}")
            print(f"⏱️  Duration: {info.get('duration_string', 'Unknown')}")
            print()

            # Show available video formats
            formats = info.get("formats", [])
            video_formats = [
                f for f in formats if f.get("vcodec") != "none" and f.get("height")
            ]

            print("🎬 Available video resolutions:")
            seen_heights = set()
            for f in sorted(
                video_formats, key=lambda x: x.get("height", 0), reverse=True
            ):
                height = f.get("height")
                if height and height not in seen_heights:
                    vcodec = f.get("vcodec", "unknown")[:10]
                    ext = f.get("ext", "?")
                    filesize = f.get("filesize", 0)
                    size_mb = f"{filesize / 1024 / 1024:.1f}MB" if filesize else "?"
                    print(f"  • {height}p - {vcodec} ({ext}) - {size_mb}")
                    seen_heights.add(height)

            print()

    except Exception as e:
        print(f"❌ Error extracting info: {e}")
        return False

    # Step 2: Download with our format string
    print("📥 Step 2: Downloading with our format string...")
    print("-" * 80)
    print(f"Format: {config.YTDLP_FORMAT}")
    print()

    work_dir = "/tmp/quality_test"
    os.makedirs(work_dir, exist_ok=True)

    ydl_opts = {
        "format": config.YTDLP_FORMAT,
        "outtmpl": f"{work_dir}/test_video.%(ext)s",
        "noplaylist": True,
        "quiet": False,  # Show download progress
        "extractor_args": {"youtube": {"player_client": ["web_creator"]}},
        "merge_output_format": "mp4",
    }

    downloaded_file = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)

            # Show what format was chosen
            requested_formats = info.get("requested_formats", [info])
            print()
            print("✅ Downloaded format(s):")
            for f in requested_formats:
                format_id = f.get("format_id", "?")
                height = f.get("height", "?")
                vcodec = f.get("vcodec", "?")
                acodec = f.get("acodec", "?")
                ext = f.get("ext", "?")
                print(f"  • Format ID: {format_id}")
                print(f"    Resolution: {height}p")
                print(f"    Video codec: {vcodec}")
                print(f"    Audio codec: {acodec}")
                print(f"    Extension: {ext}")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

    # Step 3: Verify with ffprobe
    print()
    print("🔍 Step 3: Verifying downloaded file with ffprobe...")
    print("-" * 80)

    if not downloaded_file or not os.path.exists(downloaded_file):
        print("❌ Downloaded file not found!")
        return False

    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            downloaded_file,
        ]

        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])

        if streams:
            video = streams[0]
            width = video.get("width")
            height = video.get("height")
            codec = video.get("codec_name")

            print("✅ Final video file:")
            print(f"  • Resolution: {width}x{height} ({height}p)")
            print(f"  • Codec: {codec}")
            print(
                f"  • File size: {os.path.getsize(downloaded_file) / 1024 / 1024:.1f}MB"
            )
            print(f"  • Path: {downloaded_file}")

            print()
            print("=" * 80)

            # Determine if it's acceptable
            if height >= 720:
                print("✅ SUCCESS: Downloaded 720p or higher!")
            elif height >= 480:
                print("⚠️  WARNING: Downloaded 480p (acceptable fallback)")
            elif height >= 360:
                print("⚠️  WARNING: Downloaded 360p (low quality fallback)")
            else:
                print("❌ FAIL: Downloaded below 360p!")

            print("=" * 80)
            return True

        else:
            print("❌ No video stream found in file!")
            return False

    except Exception as e:
        print(f"❌ ffprobe failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_quality.py <youtube_url>")
        print()
        print("Example:")
        print("  python test_quality.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
        sys.exit(1)

    url = sys.argv[1]
    success = test_video_quality(url)

    sys.exit(0 if success else 1)
