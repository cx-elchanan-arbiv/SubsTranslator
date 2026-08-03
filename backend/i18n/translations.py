"""
Backend i18n system for SubsTranslator
Provides server-side translation support with Accept-Language header detection
"""

import json
import os

from flask import g, request

# Import shared configuration
from shared_config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    is_rtl,
)


class I18nManager:
    """Manages backend translations and language detection"""

    def __init__(self, translations_dir: str = None):
        self.translations_dir = translations_dir or os.path.join(
            os.path.dirname(__file__), "locales"
        )
        self._translations_cache: dict[str, dict] = {}
        self._load_all_translations()

    def _load_all_translations(self):
        """Load all translation files into memory"""
        for lang_code in SUPPORTED_LANGUAGES.keys():
            self._load_language_translations(lang_code)

    def _load_language_translations(self, lang_code: str):
        """Load translations for a specific language"""
        lang_dir = os.path.join(self.translations_dir, lang_code)
        if not os.path.exists(lang_dir):
            return

        translations = {}
        for filename in os.listdir(lang_dir):
            if filename.endswith(".json"):
                namespace = filename[:-5]  # Remove .json extension
                filepath = os.path.join(lang_dir, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        translations[namespace] = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Error loading translation file {filepath}: {e}")

        self._translations_cache[lang_code] = translations

    def detect_language(self, accept_language: str = None) -> str:
        """Detect user's preferred language from Accept-Language header"""
        if not accept_language:
            return DEFAULT_LANGUAGE

        # Parse Accept-Language header
        languages = []
        for lang_range in accept_language.split(","):
            lang_range = lang_range.strip()
            if ";q=" in lang_range:
                lang, quality = lang_range.split(";q=", 1)
                try:
                    quality = float(quality)
                except ValueError:
                    quality = 1.0
            else:
                lang = lang_range
                quality = 1.0

            # Extract language code (ignore country code)
            lang_code = lang.split("-")[0].lower()
            if lang_code in SUPPORTED_LANGUAGES:
                languages.append((lang_code, quality))

        # Sort by quality and return best match
        if languages:
            languages.sort(key=lambda x: x[1], reverse=True)
            return languages[0][0]

        return DEFAULT_LANGUAGE

    def get_translation(self, key: str, lang_code: str = None, **kwargs) -> str:
        """Get translated string for given key and language"""
        if not lang_code:
            lang_code = getattr(g, "language", DEFAULT_LANGUAGE)

        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = DEFAULT_LANGUAGE

        translations = self._translations_cache.get(lang_code, {})

        # Handle namespaced keys (e.g., "errors:network.title")
        if ":" in key:
            namespace, key_path = key.split(":", 1)
        else:
            namespace = "common"
            key_path = key

        namespace_translations = translations.get(namespace, {})

        # Navigate nested keys (e.g., "network.title")
        value = namespace_translations
        for part in key_path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Fallback to English if key not found
                if lang_code != DEFAULT_LANGUAGE:
                    return self.get_translation(key, DEFAULT_LANGUAGE, **kwargs)
                return key  # Return key as fallback

        # Handle string interpolation
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value

        return str(value) if value is not None else key

    def get_error_message(
        self, error_code: str, lang_code: str = None
    ) -> dict[str, str]:
        """Get localized error message"""
        if not lang_code:
            lang_code = getattr(g, "language", DEFAULT_LANGUAGE)

        return {
            "title": self.get_translation(f"errors:http.{error_code}.title", lang_code),
            "message": self.get_translation(
                f"errors:http.{error_code}.message", lang_code
            ),
            "code": int(error_code) if error_code.isdigit() else None,
        }

    def get_supported_languages(self) -> dict[str, dict]:
        """Get list of supported languages with metadata"""
        return SUPPORTED_LANGUAGES


# Global i18n manager instance
i18n_manager = I18nManager()


# Flask integration functions
def init_i18n(app):
    """Initialize i18n for Flask app"""

    @app.before_request
    def detect_language():
        """Detect and set language for current request"""
        # Check for explicit language parameter
        lang = request.args.get("lang") or request.headers.get("X-Language")

        # Fall back to Accept-Language header
        if not lang:
            accept_language = request.headers.get("Accept-Language", "")
            lang = i18n_manager.detect_language(accept_language)

        # Validate and set language
        if lang not in SUPPORTED_LANGUAGES:
            lang = DEFAULT_LANGUAGE

        g.language = lang
        g.is_rtl = is_rtl(lang)

    @app.after_request
    def add_language_headers(response):
        """Add language-related headers to response"""
        if hasattr(g, "language"):
            response.headers["Content-Language"] = g.language
            response.headers["X-Language"] = g.language
            if g.is_rtl:
                response.headers["X-Text-Direction"] = "rtl"
            else:
                response.headers["X-Text-Direction"] = "ltr"
        return response


def t(key: str, **kwargs) -> str:
    """Translation function (shorthand)"""
    return i18n_manager.get_translation(key, **kwargs)


def get_localized_error(error_code: str, lang_code: str = None) -> dict:
    """Get localized error response"""
    return i18n_manager.get_error_message(error_code, lang_code)


def get_current_language() -> str:
    """Get current request language"""
    return getattr(g, "language", DEFAULT_LANGUAGE)


def is_rtl_language() -> bool:
    """Check if current language is RTL"""
    return getattr(g, "is_rtl", False)


# Decorator for API endpoints that need localized responses
