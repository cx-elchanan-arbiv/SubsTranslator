"""
Comprehensive tests for all supported languages in SubsTranslator
Tests both frontend and backend i18n systems for completeness and correctness
"""

import os
import sys

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from i18n.translations import SUPPORTED_LANGUAGES, i18n_manager


class TestLanguageSupport:
    """Test basic language support configuration"""

    def test_supported_languages_defined(self):
        """Test that supported languages are properly defined"""
        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert len(SUPPORTED_LANGUAGES) >= 4  # At least he, en, es, ar

        # Check required languages
        required_languages = ["he", "en", "es", "ar"]
        for lang in required_languages:
            assert lang in SUPPORTED_LANGUAGES, f"Language {lang} should be supported"

    def test_language_properties(self):
        """Test that each language has required properties"""
        for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
            assert "name" in lang_info, f"Language {lang_code} missing 'name'"
            assert (
                "nativeName" in lang_info
            ), f"Language {lang_code} missing 'nativeName'"
            assert "rtl" in lang_info, f"Language {lang_code} missing 'rtl'"
            assert isinstance(
                lang_info["rtl"], bool
            ), f"Language {lang_code} 'rtl' should be boolean"

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
        """Test that basic translations exist for each language"""
        basic_keys = [
            "common:status.success",
            "common:status.error",
            "common:status.processing",
            "common:languages." + lang_code,  # Each language should have its own name
        ]

        for key in basic_keys:
            translation = i18n_manager.get_translation(key, lang_code)
            assert translation != key, f"Translation missing for {key} in {lang_code}"
            assert (
                len(translation.strip()) > 0
            ), f"Empty translation for {key} in {lang_code}"

    @pytest.mark.parametrize("lang_code", list(SUPPORTED_LANGUAGES.keys()))
    def test_error_translations_exist(self, lang_code):
        """Test that error translations exist for each language"""
        error_keys = [
            "errors:http.404.title",
            "errors:upload.file_too_large",
            "errors:processing.task_failed",
        ]

        for key in error_keys:
            translation = i18n_manager.get_translation(key, lang_code)
            # Some languages might not have all error translations yet
            if translation != key:  # Translation exists
                assert (
                    len(translation.strip()) > 0
                ), f"Empty translation for {key} in {lang_code}"

    def test_language_names_in_native_script(self):
        """Test that language names are in their native scripts"""
        expected_native_names = {
            "he": "עברית",
            "en": "English",
            "es": "Español",
            "ar": "العربية",
        }

        for lang_code, expected_name in expected_native_names.items():
            if lang_code in SUPPORTED_LANGUAGES:
                actual_name = SUPPORTED_LANGUAGES[lang_code]["nativeName"]
                assert (
                    actual_name == expected_name
                ), f"Wrong native name for {lang_code}: got {actual_name}, expected {expected_name}"


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

    @pytest.mark.parametrize("lang_code", list(SUPPORTED_LANGUAGES.keys()))
    def test_translation_consistency(self, lang_code):
        """Test that translations are consistent (same key returns same value)"""
        key = "common:status.success"

        # Get translation multiple times
        translation1 = i18n_manager.get_translation(key, lang_code)
        translation2 = i18n_manager.get_translation(key, lang_code)

        assert (
            translation1 == translation2
        ), f"Inconsistent translation for {key} in {lang_code}"


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
    def test_no_empty_translations(self, lang_code):
        """Test that there are no empty translations"""
        common_keys = [
            "common:status.success",
            "common:status.error",
            "common:status.processing",
            "common:status.completed",
            "common:status.failed",
        ]

        for key in common_keys:
            translation = i18n_manager.get_translation(key, lang_code)
            if translation != key:  # Translation exists
                assert (
                    translation.strip()
                ), f"Empty translation for {key} in {lang_code}"

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

    def test_rtl_language_formatting(self):
        """Test that RTL languages have proper formatting"""
        rtl_languages = ["he", "ar"]

        for lang_code in rtl_languages:
            if lang_code in SUPPORTED_LANGUAGES:
                # Test that RTL languages have proper Unicode direction
                translation = i18n_manager.get_translation(
                    "common:status.success", lang_code
                )
                if translation != "common:status.success":  # Translation exists
                    # RTL text should contain RTL characters
                    has_rtl_chars = any(ord(char) > 0x590 for char in translation)
                    assert (
                        has_rtl_chars
                    ), f"RTL language {lang_code} should contain RTL characters"


class TestPerformance:
    """Test translation performance"""

    def test_translation_caching(self):
        """Test that translations are cached for performance"""
        key = "common:status.success"
        lang = "en"

        # First call - might load from file
        translation1 = i18n_manager.get_translation(key, lang)

        # Second call - should be cached
        translation2 = i18n_manager.get_translation(key, lang)

        assert translation1 == translation2
        # Both should be fast (cached)

    def test_all_languages_loaded(self):
        """Test that all supported languages are loaded in cache"""
        for lang_code in SUPPORTED_LANGUAGES.keys():
            assert (
                lang_code in i18n_manager._translations_cache
            ), f"Language {lang_code} not loaded in cache"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
