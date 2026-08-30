#!/usr/bin/env bash
# Builds the file you actually send: dist/simple-downloader-windows.zip
#
# Contents are laid out so that double-clicking the launcher at the top level is
# the only thing anyone has to understand.
#
# The zip is assembled by Python, not by `zip`: Info-ZIP on macOS writes non-ASCII
# names without setting the UTF-8 flag, so Windows decodes the Hebrew filenames
# with the local code page and shows mojibake. Python's zipfile sets the flag.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$here/bin/yt-dlp.exe" || ! -f "$here/bin/ffmpeg.exe" ]]; then
  echo "bin/ is empty — run tools/fetch-tools.sh first" >&2
  exit 1
fi

mkdir -p "$here/dist"
python3 - "$here" <<'PY'
import os
import sys
import zipfile

root = sys.argv[1]
out = os.path.join(root, "dist", "simple-downloader-windows.zip")

# (source, name inside the zip)
entries = [
    ("src/launch.bat", "הורדת סרטונים.bat"),
    # Same launcher under an ASCII name, in case a Hebrew filename trips up
    # whatever unpacks it on the other side. Identical bytes, no second version.
    ("src/launch.bat", "Start.bat"),
    ("src/instructions.he.txt", "קרא אותי.txt"),
    ("src/Downloader.ps1", "src/Downloader.ps1"),
    ("bin/yt-dlp.exe", "bin/yt-dlp.exe"),
    ("bin/ffmpeg.exe", "bin/ffmpeg.exe"),
]

if os.path.exists(out):
    os.remove(out)

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for source, name in entries:
        archive.write(os.path.join(root, source), name)

with zipfile.ZipFile(out) as archive:
    for info in archive.infolist():
        flag = "utf8" if info.flag_bits & 0x800 else "ascii"
        print(f"  {info.filename}  ({flag}, {info.file_size:,} bytes)")

print(f"\nbuilt: {out}  ({os.path.getsize(out) / 1048576:.0f} MB)")
PY
