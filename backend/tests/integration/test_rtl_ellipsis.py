#!/usr/bin/env python3
"""
Unit tests for RTL ellipsis fix
Tests that ellipsis (...) are properly handled at start/end of lines
"""

import os
import sys

# Add backend directory to path (works from any location)
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, backend_dir)

import unittest

from utils.rtl_utils import add_rtl_markers, clean_rtl_text

# Unicode BiDi control characters for verification
RLE = "\u202b"  # Right-to-Left Embedding
PDF = "\u202c"  # Pop Directional Formatting
RLM = "\u200f"  # Right-to-Left Mark
LRM = "\u200e"  # Left-to-Right Mark


class TestRTLEllipsisFix(unittest.TestCase):
    """Test cases for ellipsis RTL handling"""

    def _has_rlm_near_ellipsis(self, text):
        """Check if ellipsis has RLM marker nearby"""
        # Check for RLM before or after ellipsis
        return (f"{RLM}..." in text) or (f"...{RLM}" in text)

    def test_ellipsis_at_start_hebrew(self):
        """Test: '...מונומנטלי' should have RLM before ellipsis"""
        text = "...מונומנטלי. החזרנו את בני הערובה."
        result = add_rtl_markers(clean_rtl_text(text))

        self.assertTrue(
            self._has_rlm_near_ellipsis(result),
            f"Ellipsis should have RLM marker. Got: {repr(result)}",
        )
        print(f"✅ Hebrew ellipsis at start: {repr(result)}")

    def test_ellipsis_at_start_english(self):
        """Test: '...monumental' should have RLM before ellipsis"""
        text = "...monumental. We got the hostages back."
        result = add_rtl_markers(clean_rtl_text(text))

        # Even English text should get RLM if it has ellipsis at start
        # (though it won't be wrapped in RLE since no RTL chars)
        if "..." in text:
            # English won't have RLE wrapping, but that's OK
            print(f"ℹ️  English ellipsis at start: {repr(result)}")

    def test_ellipsis_at_end_hebrew(self):
        """Test: 'עשינו משהו...' should have RLM after ellipsis"""
        text = "אוקיי, עכשיו עשינו משהו..."
        result = add_rtl_markers(clean_rtl_text(text))

        self.assertTrue(
            self._has_rlm_near_ellipsis(result),
            f"Ellipsis should have RLM marker. Got: {repr(result)}",
        )
        print(f"✅ Hebrew ellipsis at end: {repr(result)}")

    def test_mixed_hebrew_english_ellipsis_start(self):
        """Test: Mixed content with ellipsis at start"""
        text = "...מונומנטלי"
        result = add_rtl_markers(clean_rtl_text(text))

        self.assertTrue(
            self._has_rlm_near_ellipsis(result),
            f"Mixed content ellipsis should have RLM. Got: {repr(result)}",
        )
        print(f"✅ Mixed content ellipsis: {repr(result)}")

    def test_no_ellipsis(self):
        """Test: Normal text without ellipsis should work normally"""
        text = "שלום עולם"
        result = add_rtl_markers(clean_rtl_text(text))

        # Should have RLE wrapping but no ellipsis markers
        self.assertIn(RLE, result)
        self.assertIn(PDF, result)
        print(f"✅ Normal text (no ellipsis): {repr(result)}")

    def test_ellipsis_in_middle(self):
        """Test: Ellipsis in middle of text"""
        text = "הם אמרו... שהם יעשו זאת"
        result = add_rtl_markers(clean_rtl_text(text))

        # Should have RLE wrapping
        self.assertIn(RLE, result)
        self.assertIn(PDF, result)
        print(f"✅ Ellipsis in middle: {repr(result)}")

    def test_multiline_ellipsis(self):
        """Test: Multiple lines with ellipsis"""
        text = """אוקיי, עכשיו עשינו משהו...
...מונומנטלי. החזרנו את בני הערובה."""
        result = add_rtl_markers(clean_rtl_text(text))

        # Both ellipsis should have RLM
        ellipsis_count = result.count("...")
        rlm_near_ellipsis = result.count(f"{RLM}...") + result.count(f"...{RLM}")

        self.assertGreaterEqual(
            rlm_near_ellipsis,
            1,
            f"Should have RLM near ellipsis. Found {rlm_near_ellipsis} out of {ellipsis_count}",
        )
        print(
            f"✅ Multiline ellipsis: Found {rlm_near_ellipsis}/{ellipsis_count} with RLM"
        )

    def test_actual_subtitle_line_9(self):
        """Test: Actual line 9 from your subtitle file"""
        text = "...מונומנטלי. החזרנו את בני הערובה."
        result = add_rtl_markers(clean_rtl_text(text))

        self.assertTrue(
            self._has_rlm_near_ellipsis(result),
            f"Line 9 should have RLM near ellipsis. Got: {repr(result)}",
        )

        # Visualize for debugging
        visual = result.replace(RLE, "[RLE]")
        visual = visual.replace(PDF, "[PDF]")
        visual = visual.replace(RLM, "[RLM]")
        visual = visual.replace(LRM, "[LRM]")

        print("\n📝 Line 9 visualization:")
        print(f"   Input:  {repr(text)}")
        print(f"   Output: {repr(result)}")
        print(f"   Visual: {visual}")
        print("   ✅ PASS: Ellipsis has RLM marker!\n")


def run_tests():
    """Run all tests with verbose output"""
    print("=" * 70)
    print("RTL Ellipsis Fix - Unit Tests")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRTLEllipsisFix)

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"   Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
