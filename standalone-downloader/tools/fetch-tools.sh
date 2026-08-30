#!/usr/bin/env bash
# Downloads the two Windows binaries the portable build ships with.
# Run on macOS/Linux — these are Windows .exe files, we only fetch and unpack them.
#
# They are deliberately NOT committed: ffmpeg.exe alone is over GitHub's 100MB
# per-file limit, and a binary blob in git history is forever.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="$here/bin"
mkdir -p "$bin"

YTDLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

if [[ ! -f "$bin/yt-dlp.exe" ]]; then
  echo "==> yt-dlp.exe"
  curl -fL --progress-bar "$YTDLP_URL" -o "$bin/yt-dlp.exe"
else
  echo "==> yt-dlp.exe already here (delete it to refresh)"
fi

if [[ ! -f "$bin/ffmpeg.exe" ]]; then
  echo "==> ffmpeg (win64)"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fL --progress-bar "$FFMPEG_URL" -o "$tmp/ffmpeg.zip"
  unzip -q "$tmp/ffmpeg.zip" -d "$tmp/x"
  found="$(find "$tmp/x" -name ffmpeg.exe -type f | head -1)"
  [[ -n "$found" ]] || { echo "ffmpeg.exe not found inside the archive" >&2; exit 1; }
  cp "$found" "$bin/ffmpeg.exe"
else
  echo "==> ffmpeg.exe already here (delete it to refresh)"
fi

echo
ls -lh "$bin"
