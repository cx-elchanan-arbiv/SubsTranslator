"""
The RLM-inside-a-word bug, and the SRT typography gap it lived next to.

Both defects come from the same confusion: treating GERESH ׳ (U+05F3) and GERSHAYIM ״
(U+05F4) as sentence punctuation. They are not — they are word-INTERNAL marks that carry
a phoneme (ג׳ = j) or mark an acronym (צה״ל, מפכ״ל, התנ״ך).

  * ``fix_rtl_punctuation`` inserted U+200F RIGHT-TO-LEFT MARK between a Hebrew letter
    and "its punctuation", so the name ג׳וני shipped as 'ג‏׳וני' — invisible on screen,
    and enough to break search, copy-paste and every string comparison downstream. This
    function is on the LEGACY path as well as the v2 one, so the fix covers both.
  * ``create_srt_file`` never applied the gershayim/geresh normalisation that
    ``build_ass`` has always applied, so the burned-in picture read התנ״ך while the .srt
    shipped beside it read התנ"ך with an ASCII quote.
"""

import os
import sys

import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from utils.rtl_utils import (  # noqa: E402
    RTL_STICKY_PUNCTUATION,
    add_rtl_markers,
    fix_rtl_punctuation,
)

RLM = "‏"
GERESH = "׳"
GERSHAYIM = "״"


@pytest.mark.unit
class TestNoRlmInsideAWord:
    def test_geresh_does_not_get_an_rlm_before_it(self):
        assert fix_rtl_punctuation(f"ג{GERESH}וני") == f"ג{GERESH}וני"

    def test_gershayim_does_not_get_an_rlm_before_it(self):
        assert fix_rtl_punctuation(f"צה{GERSHAYIM}ל") == f"צה{GERSHAYIM}ל"

    def test_the_reported_name_is_intact(self):
        """D's find: 'ג‏׳וני' — an RLM wedged between the gimel and its geresh."""
        out = fix_rtl_punctuation(f"שמו ג{GERESH}וני והוא הגיע.")
        assert f"{RLM}{GERESH}" not in out
        assert f"ג{GERESH}וני" in out

    def test_the_marks_are_not_in_the_punctuation_class(self):
        assert GERESH not in RTL_STICKY_PUNCTUATION
        assert GERSHAYIM not in RTL_STICKY_PUNCTUATION

    def test_the_whole_hebrew_ligature_block_is_excluded(self):
        for code in range(0x05F0, 0x05F5):
            assert chr(code) not in RTL_STICKY_PUNCTUATION, hex(code)

    def test_real_sentence_punctuation_still_gets_its_rlm(self):
        """The rule's actual job is untouched."""
        assert fix_rtl_punctuation("שלום.") == f"שלום{RLM}."
        assert fix_rtl_punctuation("מה?") == f"מה{RLM}?"
        assert fix_rtl_punctuation("עצור!") == f"עצור{RLM}!"
        assert fix_rtl_punctuation("כן, טוב") == f"כן{RLM}, טוב"

    def test_the_legacy_marker_path_is_clean_too(self):
        """``add_rtl_markers`` calls ``fix_rtl_punctuation`` — same bug, same fix."""
        out = add_rtl_markers(f"ג{GERESH}ורג{GERESH} אמר שלום.")
        assert f"{RLM}{GERESH}" not in out
        assert f"ג{GERESH}ורג{GERESH}" in out


@pytest.mark.unit
class TestSrtCarriesHebrewTypography:
    """The .srt a user downloads must spell words the way the picture does."""

    @staticmethod
    def _write(tmp_path, segments, **kwargs):
        from services.subtitle_service import SubtitleService

        out = str(tmp_path / "out.srt")
        SubtitleService().create_srt_file(segments, out, **kwargs)
        with open(out, encoding="utf-8") as fh:
            return fh.read()

    def test_ascii_quote_in_an_acronym_becomes_gershayim(self, tmp_path):
        segments = [{"start": 0.0, "end": 2.0, "text": 'קראתי את התנ"ך אתמול.'}]
        body = self._write(tmp_path, segments)
        assert f"התנ{GERSHAYIM}ך" in body
        assert 'התנ"ך' not in body

    def test_ascii_apostrophe_in_a_name_becomes_geresh(self, tmp_path):
        segments = [{"start": 0.0, "end": 2.0, "text": "ג'ורג' הגיע."}]
        body = self._write(tmp_path, segments)
        assert f"ג{GERESH}ורג{GERESH}" in body

    def test_the_translated_track_gets_it_too(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "The Bible.",
                "translated_text": 'התנ"ך.',
            }
        ]
        body = self._write(tmp_path, segments, use_translation=True, language="he")
        assert f"התנ{GERSHAYIM}ך" in body

    def test_an_english_quotation_is_left_alone(self, tmp_path):
        segments = [{"start": 0.0, "end": 2.0, "text": 'He said "hello" loudly.'}]
        body = self._write(tmp_path, segments)
        assert '"hello"' in body

    def test_an_english_apostrophe_is_left_alone(self, tmp_path):
        segments = [{"start": 0.0, "end": 2.0, "text": "It's Charles' book."}]
        body = self._write(tmp_path, segments)
        assert "It's Charles' book." in body

    def test_the_srt_and_the_ass_now_agree(self, tmp_path):
        """The point of the fix, stated as an equality."""
        from services.subtitle_engine import build_ass

        text = "צה\"ל וג'ורג'."
        srt = self._write(tmp_path, [{"start": 0.0, "end": 2.0, "text": text}])
        ass = build_ass(
            [{"start": 0.0, "end": 2.0, "text": text}],
            video_w=1280,
            video_h=720,
            rtl=False,
        )
        for mark in (GERSHAYIM, GERESH):
            assert (mark in srt) == (mark in ass) is True
