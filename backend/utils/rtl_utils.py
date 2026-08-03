#!/usr/bin/env python3
"""
RTL Subtitle Utilities for Hebrew and Arabic
Fixes right-to-left text direction, punctuation, and mixed content in SRT files
"""

import re
import unicodedata


def is_rtl_language(language_code):
    """Check if language requires RTL support"""
    rtl_languages = {"he", "ar", "fa", "ur", "yi"}
    return language_code.lower() in rtl_languages


def add_rtl_markers(text):
    """Add Unicode BiDi markers for proper RTL display"""
    if not text or not text.strip():
        return text

    # Unicode BiDi control characters
    RLE = "\u202b"  # Right-to-Left Embedding
    LRE = "\u202a"  # Left-to-Right Embedding
    PDF = "\u202c"  # Pop Directional Formatting
    RLM = "\u200f"  # Right-to-Left Mark
    LRM = "\u200e"  # Left-to-Right Mark

    # Clean existing markers first
    text = re.sub(r"[\u202A-\u202E\u200E\u200F]", "", text)

    # Detect if text contains RTL characters
    has_rtl = any(unicodedata.bidirectional(char) in ("R", "AL") for char in text)
    has_ltr = any(unicodedata.bidirectional(char) == "L" for char in text)

    if not has_rtl:
        return text  # No RTL content, return as-is

    # Handle mixed RTL/LTR content
    if has_rtl and has_ltr:
        # Complex case: mixed content
        text = fix_mixed_content(text)

    # Wrap the entire text with RTL embedding
    text = f"{RLE}{text}{PDF}"

    # Fix punctuation at the end
    text = fix_rtl_punctuation(text)

    return text


def fix_mixed_content(text):
    """Fix mixed RTL/LTR content like Hebrew with English words/numbers"""
    RLE = "\u202b"  # Right-to-Left Embedding
    LRE = "\u202a"  # Left-to-Right Embedding
    PDF = "\u202c"  # Pop Directional Formatting
    LRM = "\u200e"  # Left-to-Right Mark

    # Pattern for Latin/English words (including numbers and common symbols)
    ltr_pattern = r"([a-zA-Z0-9][a-zA-Z0-9\s\-_\.,]*[a-zA-Z0-9]|[a-zA-Z0-9])"

    def wrap_ltr(match):
        content = match.group(0)
        # Don't wrap if it's just punctuation or single character
        if len(content.strip()) <= 1 or content.strip() in ".,!?;:":
            return content
        return f"{LRE}{content}{PDF}"

    # Wrap LTR sequences
    text = re.sub(ltr_pattern, wrap_ltr, text)

    # Add LRM after wrapped LTR content to maintain flow
    text = re.sub(f"({PDF})", f"\\1{LRM}", text)

    return text


#: Punctuation that should stick to the Hebrew word in front of it.
#:
#: U+05F0-U+05F4 — the Hebrew ligatures and, critically, GERESH ׳ (U+05F3) and GERSHAYIM
#: ״ (U+05F4) — are deliberately EXCLUDED, and that exclusion is a bug fix, not a
#: preference. Those two are not sentence punctuation at all: they are word-INTERNAL
#: marks carrying a phoneme (ג׳ = j, צ׳ = ch) or marking an acronym (צה״ל, מפכ״ל).
#: :func:`fix_rtl_punctuation` inserts U+200F RIGHT-TO-LEFT MARK between the preceding
#: letter and the mark, so listing them here wedged an invisible control character INSIDE
#: a word: the name ג׳וני shipped as 'ג‏׳וני'. Invisible on screen, and it breaks
#: search, copy-paste, and every downstream string comparison in the pipeline.
RTL_STICKY_PUNCTUATION = ".!?:;,"


def fix_rtl_punctuation(text):
    """Fix punctuation placement in RTL text.

    Note the deliberate omission of geresh and gershayim — see
    :data:`RTL_STICKY_PUNCTUATION`.
    """
    RLM = "\u200f"  # Right-to-Left Mark

    rtl_punctuation = RTL_STICKY_PUNCTUATION

    for punct in rtl_punctuation:
        # Add RLM before punctuation to ensure proper positioning
        text = re.sub(f"([א-ת])({re.escape(punct)})", f"\\1{RLM}\\2", text)

    # Fix ellipsis at start/end of lines (for cases like "...monumental" or "...מונומנטלי")
    text = re.sub(r"^(\.\.\.)", f"{RLM}\\1", text, flags=re.MULTILINE)  # Start of line
    text = re.sub(r"(\.\.\.)\s*$", f"\\1{RLM}", text, flags=re.MULTILINE)  # End of line

    # Fix quotes and parentheses
    text = re.sub(r'"([^"]*)"', f'"{RLM}\\1{RLM}"', text)
    text = re.sub(r"\(([^)]*)\)", f"({RLM}\\1{RLM})", text)

    return text


def clean_rtl_text(text):
    """Clean and prepare RTL text for subtitle display"""
    if not text:
        return text

    # Remove extra whitespace
    text = " ".join(text.split())

    # Normalize Hebrew/Arabic characters
    text = unicodedata.normalize("NFC", text)

    # Fix common RTL issues
    text = fix_hebrew_quotes(text)
    text = fix_arabic_diacritics(text)

    return text


def fix_hebrew_quotes(text):
    """Fix Hebrew quotation marks"""
    # Replace regular quotes with Hebrew quotes
    text = re.sub(r'"([^"]*)"', r"״\1״", text)
    text = re.sub(r"'([^']*)'", r"׳\1׳", text)
    return text


def fix_arabic_diacritics(text):
    """Handle Arabic diacritics for better readability"""
    # Remove excessive diacritics for subtitle readability
    arabic_diacritics = "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
    # Keep text readable by removing most diacritics except essential ones
    text = re.sub(f"[{arabic_diacritics}]", "", text)
    return text


def process_rtl_subtitle_file(srt_content, language_code):
    """Process entire SRT file content for RTL support"""
    if not is_rtl_language(language_code):
        return srt_content

    lines = srt_content.split("\n")
    processed_lines = []

    for line in lines:
        line = line.strip()

        # Skip empty lines, timestamps, and subtitle numbers
        if not line or line.isdigit() or "-->" in line:
            processed_lines.append(line)
            continue

        # Process subtitle text
        line = clean_rtl_text(line)
        line = add_rtl_markers(line)
        processed_lines.append(line)

    return "\n".join(processed_lines)


def create_rtl_srt(segments, filename, language_code):
    """Create RTL-optimized SRT file"""
    srt_content = []

    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])

        # Get text (translated if available, otherwise original)
        text = segment.get("translated_text", segment["text"])

        # Apply RTL processing
        if is_rtl_language(language_code):
            text = clean_rtl_text(text)
            text = add_rtl_markers(text)

        srt_content.extend(
            [
                str(i),
                f"{start_time} --> {end_time}",
                text,
                "",  # Empty line between subtitles
            ]
        )

    return "\n".join(srt_content)


def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


# Test function for development
def test_rtl_processing():
    """Test RTL processing with sample text"""
    test_cases = [
        ("שלום world", "he"),
        ("זה טקסט בעברית עם English words", "he"),
        ("مرحبا بكم في العالم", "ar"),
        ("هذا نص عربي مع English text", "ar"),
        ("Regular English text", "en"),
    ]

    results = []
    for text, lang in test_cases:
        processed = clean_rtl_text(text)
        processed = add_rtl_markers(processed)
        results.append((lang, text, processed))
    return results


if __name__ == "__main__":
    test_rtl_processing()
