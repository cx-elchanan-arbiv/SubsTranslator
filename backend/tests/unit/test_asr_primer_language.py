"""
The ASR punctuation primer is text the decoder CONTINUES, so its language matters.

Priming a Hebrew clip with an English sentence asks the model to continue English into
Hebrew audio. ``source_lang="auto"`` keeps the English primer on purpose: guessing the
language from the primer is exactly backwards.
"""

import os
import sys

import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.transcription_service import (  # noqa: E402
    ASR_PUNCTUATION_PRIMER,
    ASR_PUNCTUATION_PRIMERS,
    asr_primer_for,
)


@pytest.mark.unit
class TestAsrPrimerForLanguage:
    def test_hebrew_gets_a_hebrew_primer(self):
        primer = asr_primer_for("he")
        assert primer != ASR_PUNCTUATION_PRIMER
        assert any("א" <= ch <= "ת" for ch in primer)

    def test_a_regional_code_still_matches(self):
        assert asr_primer_for("he-IL") == asr_primer_for("he")

    def test_auto_keeps_the_english_primer(self):
        assert asr_primer_for("auto") == ASR_PUNCTUATION_PRIMER
        assert asr_primer_for(None) == ASR_PUNCTUATION_PRIMER
        assert asr_primer_for("") == ASR_PUNCTUATION_PRIMER

    def test_an_unknown_language_falls_back(self):
        assert asr_primer_for("sw") == ASR_PUNCTUATION_PRIMER

    def test_every_primer_demonstrates_the_punctuation_it_primes(self):
        for code, primer in ASR_PUNCTUATION_PRIMERS.items():
            assert primer.rstrip().endswith("."), code
            assert "," in primer, code
            assert "?" in primer, code

    def test_no_primer_names_a_topic_or_an_entity(self):
        """A primer with a proper noun biases what the model HEARS in unclear audio.

        Sentence-initial capitals are the whole point of the primer, so only capitals
        INSIDE a sentence can be proper nouns — with the English pronoun "I" as the one
        capital that is a grammatical rule rather than a name.
        """
        for code, primer in ASR_PUNCTUATION_PRIMERS.items():
            for sentence in primer.replace("?", ".").replace("!", ".").split("."):
                for word in sentence.split()[1:]:
                    if word.strip(",") == "I":
                        continue
                    assert not word[0].isupper(), (code, word)

    def test_no_primer_contains_a_short_emphatic_sentence(self):
        """The measured fabrication: short punchy assertions are cheap to imitate."""
        for code, primer in ASR_PUNCTUATION_PRIMERS.items():
            for sentence in primer.replace("?", ".").replace("!", ".").split("."):
                words = sentence.split()
                assert not (0 < len(words) <= 2), (code, sentence)
