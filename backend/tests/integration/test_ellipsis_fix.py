#!/usr/bin/env python3
"""
Quick test for ellipsis RTL fix
"""

import os
import sys

# Add backend directory to path (works from any location)
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, backend_dir)

from utils.rtl_utils import add_rtl_markers, clean_rtl_text

# Test cases from your minute 1
test_cases = [
    ("אוקיי, עכשיו עשינו משהו...", "Hebrew with ellipsis at end"),
    ("...מונומנטלי. החזרנו את בני הערובה.", "Hebrew with ellipsis at start"),
    ("Okay, now we did something...", "English with ellipsis at end"),
    ("...monumental. We got the hostages back.", "English with ellipsis at start"),
]

print("=" * 70)
print("Testing Ellipsis RTL Fix")
print("=" * 70)

for text, description in test_cases:
    print(f"\n📝 Test: {description}")
    print(f"   Input:  '{text}'")

    # Process like subtitle_service does
    cleaned = clean_rtl_text(text)
    processed = add_rtl_markers(cleaned)

    # Visualize unicode marks
    visual = processed.replace("\u202b", "[RLE]")
    visual = visual.replace("\u202c", "[PDF]")
    visual = visual.replace("\u200f", "[RLM]")
    visual = visual.replace("\u200e", "[LRM]")
    visual = visual.replace("\u202a", "[LRE]")

    print(f"   Output: '{processed}'")
    print(f"   Marks:  '{visual}'")

    # Check if ellipsis has RLM
    if "..." in text:
        if "\u200f..." in processed or "...\u200f" in processed:
            print("   ✅ Ellipsis has RLM mark - should display correctly!")
        else:
            print("   ❌ Ellipsis missing RLM mark - may not display correctly")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
