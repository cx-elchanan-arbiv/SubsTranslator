"""
Wiring layer for the opt-in subtitle-quality pipeline.

``subtitle_engine`` (spotting + ASS rendering) and ``translation_v2`` (scene-context
translation) are deliberately pure modules: they read no config, touch no filesystem and
know nothing about Flask, Celery or this app's data shapes. This module is the only place
that knows how to *reach* them from the app:

* :func:`parse_subtitle_flags` — read the four user-facing toggles off a request
  (``request.form`` or a parsed JSON dict). HTML forms send booleans as the strings
  ``"true"``/``"false"``, JSON sends real booleans, and a hand-rolled curl can send
  ``"1"`` — all three are accepted.
* :func:`resolve_flags` — re-normalise the same four values after they have travelled
  through Celery kwargs, so the pipeline never has to trust its caller.
* :func:`normalize_cues` — the common cue dict. The legacy pipeline speaks
  ``{"start","end","text","translated_text"}``; ``subtitle_engine.words_to_cues`` emits
  ``{"start","end","text"}`` and ``translation_v2.translate_cues`` adds ``"translated"``.
  Normalising after every stage is what makes all eight flag combinations composable:
  any stage can consume either predecessor's output.

The four flags are independent by design:

``spotting_v2``       re-spot cues from Whisper WORD timestamps instead of using
                      Whisper's own speech segments.
``translation_v2``    translate with whole-scene context + CPS enforcement instead of
                      the legacy fixed-window translator.
``translation_style``  ``"clean"`` (drop spoken filler) or ``"faithful"`` (keep it).
                      Only meaningful while ``translation_v2`` is on.
``render_v2``         burn in from a generated ``.ass`` file via FFmpeg's ``ass`` filter
                      instead of SRT + the ``subtitles`` filter.

All four default to OFF: with no flags set, nothing in this module runs and the legacy
pipeline is untouched.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

STYLE_CLEAN = "clean"
STYLE_FAITHFUL = "faithful"
TRANSLATION_STYLES = (STYLE_CLEAN, STYLE_FAITHFUL)

#: Flag name -> default. Every default is "behave exactly like today".
FLAG_DEFAULTS: dict[str, Any] = {
    "spotting_v2": False,
    "translation_v2": False,
    "translation_style": STYLE_CLEAN,
    "render_v2": False,
}

BOOL_FLAGS = ("spotting_v2", "translation_v2", "render_v2")

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on", "t", "y"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", "f", "n", "none", "null", ""})


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a boolean that may have arrived as a real bool, a number or a string.

    Unrecognised values fall back to ``default`` with a warning rather than raising:
    a malformed experimental toggle must never turn a working job into a 500.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    logger.warning(
        "subtitle_pipeline: unparsable boolean %r — using %s", value, default
    )
    return default


def parse_translation_style(value: Any, default: str = STYLE_CLEAN) -> str:
    """Parse the filler-removal style. Anything unrecognised falls back to ``default``."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRANSLATION_STYLES:
        return text
    if text:
        logger.warning(
            "subtitle_pipeline: unknown translation_style %r — using %r", value, default
        )
    return default


def parse_subtitle_flags(source: Any) -> dict[str, Any]:
    """Read the four toggles out of a request payload.

    Args:
        source: anything with ``.get(key, default)`` — ``request.form``,
            ``request.get_json()``'s dict, or a plain dict. ``None`` yields the defaults.

    Returns:
        ``{"spotting_v2": bool, "translation_v2": bool, "translation_style": str,
        "render_v2": bool}`` — always all four keys, always the right types.
    """
    if source is None:
        return dict(FLAG_DEFAULTS)
    getter = getattr(source, "get", None)
    if not callable(getter):
        logger.warning(
            "subtitle_pipeline: cannot read flags from %s — using defaults",
            type(source).__name__,
        )
        return dict(FLAG_DEFAULTS)

    flags = {name: parse_bool(getter(name), FLAG_DEFAULTS[name]) for name in BOOL_FLAGS}
    flags["translation_style"] = parse_translation_style(
        getter("translation_style"), FLAG_DEFAULTS["translation_style"]
    )
    return flags


def resolve_flags(
    spotting_v2: Any = None,
    translation_v2: Any = None,
    translation_style: Any = None,
    render_v2: Any = None,
) -> dict[str, Any]:
    """Normalise the four flags as received by a Celery task.

    Celery round-trips kwargs through JSON, so booleans normally survive intact — but a
    task can also be enqueued by hand or by an older client, so the values are parsed
    with the same tolerance as the HTTP layer.
    """
    return {
        "spotting_v2": parse_bool(spotting_v2, FLAG_DEFAULTS["spotting_v2"]),
        "translation_v2": parse_bool(translation_v2, FLAG_DEFAULTS["translation_v2"]),
        "translation_style": parse_translation_style(
            translation_style, FLAG_DEFAULTS["translation_style"]
        ),
        "render_v2": parse_bool(render_v2, FLAG_DEFAULTS["render_v2"]),
    }


def parse_glossary(source: Any) -> dict[str, str]:
    """Read an optional terminology glossary out of a request/job payload.

    A glossary pins a source term to one exact target rendering. Its first purpose is
    terminology CONTRAST: a clip whose whole argument is "the language is not called
    Hebrew, it is called Ivrit" translates into self-contradiction unless the two names
    are kept apart, because collapsing synonyms is normally the right instinct.

    Accepted shapes, because this value has to survive an HTML form, a JSON body and a
    Celery round-trip:

    * a real mapping — ``{"Ivrit": "עִברית"}``;
    * a JSON object as a STRING — ``'{"Ivrit": "עִברית"}'`` (what a form field carries);
    * ``None``/blank/anything else — ``{}``.

    Never raises. A malformed glossary is a warning and an empty dict, because a broken
    optional hint must not turn a working job into a failure.
    """
    if source is None:
        return {}
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return {}
        try:
            source = json.loads(text)
        except (ValueError, TypeError):
            logger.warning("subtitle_pipeline: glossary is not valid JSON — ignored")
            return {}
    if not isinstance(source, dict):
        logger.warning(
            "subtitle_pipeline: glossary must be an object, got %s — ignored",
            type(source).__name__,
        )
        return {}

    glossary: dict[str, str] = {}
    for key, value in source.items():
        if key is None or value is None:
            continue
        term, rendering = str(key).strip(), str(value).strip()
        if term and rendering:
            glossary[term] = rendering
    if glossary:
        logger.info("subtitle_pipeline: glossary with %d term(s)", len(glossary))
    return glossary


def any_enabled(flags: dict[str, Any]) -> bool:
    """True when at least one v2 stage is switched on (``translation_style`` alone is not a stage)."""
    return any(bool(flags.get(name)) for name in BOOL_FLAGS)


def flags_summary(flags: dict[str, Any]) -> str:
    """One-line, greppable rendering of the resolved flags for the job log."""
    return (
        f"spotting_v2={bool(flags.get('spotting_v2'))} "
        f"translation_v2={bool(flags.get('translation_v2'))} "
        f"translation_style={flags.get('translation_style', STYLE_CLEAN)} "
        f"render_v2={bool(flags.get('render_v2'))}"
    )


#: Keys this module owns and rewrites; everything else on a cue is passed through.
_NORMALIZED_KEYS = frozenset({"start", "end", "text", "translated_text", "translated"})


def _coerce_time(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_cues(items: Any) -> list[dict[str, Any]]:
    """Normalise any stage's output to the pipeline's common cue dict.

    Accepts legacy Whisper segments (``translated_text``), ``words_to_cues`` output (no
    translation at all) and ``translate_cues`` output (``translated``), and returns
    plain dicts — never a ``TranslationResult`` — so downstream code can copy and mutate
    them freely.

    The translation is read from the first of ``translated_text`` / ``translated`` that is
    actually **non-blank**, not merely present. That distinction is load-bearing: this
    function normalises cues *between* stages, so by the time ``translate_cues`` runs, its
    input already carries ``translated_text: ""`` from the previous pass — and
    ``translate_cues`` writes its result to ``translated``. Preferring a present-but-empty
    ``translated_text`` would silently discard every v2 translation and burn in the source
    text while reporting success.

    Keys this function does not know about are **carried through untouched**. A stage
    that starts emitting, say, ``speaker`` or ``confidence`` must not have it silently
    deleted by the normaliser sitting between it and its consumer; normalising is about
    agreeing on four keys, not about being an allow-list. ``translated`` is the one
    exception — it is consumed into ``translated_text`` and dropped, so no downstream
    code has to know which of the two spellings it is looking at.

    Returns:
        ``[{"start": float, "end": float, "text": str, "translated_text": str, ...}, ...]``
        in input order. Entries whose source *and* translated text are both blank are
        dropped (they can only produce an empty subtitle event); everything else is kept
        even if only one of the two is present.
    """
    cues: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            logger.warning("subtitle_pipeline: skipping non-dict cue %r", item)
            continue
        text = str(item.get("text") or "").strip()
        translated = ""
        for key in ("translated_text", "translated"):
            value = item.get(key)
            if value is not None and str(value).strip():
                translated = str(value).strip()
                break
        if not text and not translated:
            continue
        cue = {k: v for k, v in item.items() if k not in _NORMALIZED_KEYS}
        cue.update(
            {
                "start": _coerce_time(item.get("start")),
                "end": _coerce_time(item.get("end")),
                "text": text,
                "translated_text": translated,
            }
        )
        cues.append(cue)
    return cues
