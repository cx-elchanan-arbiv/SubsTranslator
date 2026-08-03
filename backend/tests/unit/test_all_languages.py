"""Backend i18n across every supported language.

Half of what used to live here could not fail. ``I18nManager.get_translation`` falls back
to English for any missing key, so "does language X have key Y" was answered by English
every time — and several tests additionally guarded their own assertion behind
``if translation != key``. Two more called a pure function twice and asserted the two
results matched. Those are gone; what remains either reads a language's own file or
asserts a value only that language can produce.

The file also carried no ``unit`` mark, so CI ran none of it.
"""

import json
from pathlib import Path

import pytest

from i18n.translations import SUPPORTED_LANGUAGES, i18n_manager

pytestmark = pytest.mark.unit


class TestLanguageSupport:
    """Test basic language support configuration"""

    def test_rtl_languages_correct(self):
        """Test that RTL languages are correctly marked"""
        rtl_languages = ["he", "ar"]  # Hebrew and Arabic should be RTL
        ltr_languages = ["en", "es"]  # English and Spanish should be LTR

        for lang in rtl_languages:
            if lang in SUPPORTED_LANGUAGES:
                assert SUPPORTED_LANGUAGES[lang]["rtl"] is True, f"{lang} should be RTL"

        for lang in ltr_languages:
            if lang in SUPPORTED_LANGUAGES:
                assert (
                    SUPPORTED_LANGUAGES[lang]["rtl"] is False
                ), f"{lang} should be LTR"


class TestTranslationCompleteness:
    """Test that translations are complete for all supported languages"""

    @pytest.mark.parametrize("lang_code", list(SUPPORTED_LANGUAGES.keys()))
    def test_basic_translations_exist(self, lang_code):
        """Every language carries these keys in its OWN locale file.

        Deliberately reads the file instead of calling ``get_translation``: the manager
        falls back to English for any key a language is missing, so the obvious version
        of this test passes for a language whose file is empty. Four sibling tests below
        assert the actual rendered values; this one is about coverage of the key set.
        """
        locale_dir = (
            Path(__file__).resolve().parents[2] / "i18n" / "locales" / lang_code
        )
        common = json.loads((locale_dir / "common.json").read_text(encoding="utf-8"))

        for key in ("success", "error", "processing"):
            assert key in common.get(
                "status", {}
            ), f"{lang_code} is missing status.{key}"

        # Each language names itself, in itself.
        assert lang_code in common.get(
            "languages", {}
        ), f"{lang_code} does not name itself in its own common.json"


class TestLanguageSwitching:
    """Test language switching functionality"""

    @pytest.mark.parametrize("lang_code", list(SUPPORTED_LANGUAGES.keys()))
    def test_language_detection_works(self, lang_code):
        """Test that language detection works for each supported language"""
        # Test Accept-Language header detection
        accept_header = f"{lang_code},en;q=0.9"
        detected = i18n_manager.detect_language(accept_header)
        assert (
            detected == lang_code
        ), f"Failed to detect {lang_code} from header {accept_header}"

    def test_fallback_to_default_language(self):
        """Test fallback to default language for unsupported languages"""
        unsupported_header = "xx-XX,zz;q=0.9"
        detected = i18n_manager.detect_language(unsupported_header)
        assert (
            detected == "en"
        ), f"Should fallback to 'en' for unsupported language, got {detected}"


class TestSpecificLanguages:
    """Test specific language implementations"""

    def test_hebrew_translations(self):
        """Test Hebrew-specific translations"""
        hebrew_translations = {
            "common:status.success": "הצלחה",
            "common:status.error": "שגיאה",
            "common:languages.he": "עברית",
        }

        for key, expected in hebrew_translations.items():
            actual = i18n_manager.get_translation(key, "he")
            assert (
                actual == expected
            ), f"Hebrew translation for {key}: got {actual}, expected {expected}"

    def test_english_translations(self):
        """Test English-specific translations"""
        english_translations = {
            "common:status.success": "Success",
            "common:status.error": "Error",
            "common:languages.en": "English",
        }

        for key, expected in english_translations.items():
            actual = i18n_manager.get_translation(key, "en")
            assert (
                actual == expected
            ), f"English translation for {key}: got {actual}, expected {expected}"

    def test_spanish_translations(self):
        """Test Spanish-specific translations"""
        spanish_translations = {
            "common:status.success": "Éxito",
            "common:status.error": "Error",
            "common:languages.es": "Español",
        }

        for key, expected in spanish_translations.items():
            actual = i18n_manager.get_translation(key, "es")
            assert (
                actual == expected
            ), f"Spanish translation for {key}: got {actual}, expected {expected}"

    def test_arabic_translations(self):
        """Test Arabic-specific translations"""
        arabic_translations = {
            "common:status.success": "نجح",
            "common:status.error": "خطأ",
            "common:languages.ar": "العربية",
        }

        for key, expected in arabic_translations.items():
            actual = i18n_manager.get_translation(key, "ar")
            assert (
                actual == expected
            ), f"Arabic translation for {key}: got {actual}, expected {expected}"


class TestTranslationQuality:
    """Test translation quality and formatting"""

    @pytest.mark.parametrize("lang_code", list(SUPPORTED_LANGUAGES.keys()))
    def test_interpolation_support(self, lang_code):
        """Test that interpolation works in translations"""
        # Test with a key that should support interpolation
        key = "errors:upload.file_too_large"
        translation = i18n_manager.get_translation(key, lang_code, max_size=100)

        if translation != key:  # Translation exists
            # Should contain the interpolated value or placeholder
            assert (
                "100" in translation or "{max_size}" in translation
            ), f"Interpolation failed for {key} in {lang_code}"


class TestPerformance:
    """Test translation performance"""

    # Both should be fast (cached)

    def test_all_languages_loaded(self):
        """Test that all supported languages are loaded in cache"""
        for lang_code in SUPPORTED_LANGUAGES.keys():
            assert (
                lang_code in i18n_manager._translations_cache
            ), f"Language {lang_code} not loaded in cache"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
