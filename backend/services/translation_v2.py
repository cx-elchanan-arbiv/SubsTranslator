"""
translation_v2 — broadcast-quality subtitle translation (additive, opt-in).

Why this module exists
----------------------
The legacy path (``translation_services.OpenAITranslator``) treats translation as a
string-array API call: it prompts with a raw ISO code ("translate to he"), never asks
for punctuation to be preserved, and translates fixed windows of cues in parallel with
zero cross-window context. The result is unpunctuated, inconsistent Hebrew.

This module is a clean re-implementation of the *quality* concerns only:

* prompts always name the language in full ("Hebrew", never "he");
* punctuation is an explicit hard rule — every cue ends punctuated;
* the whole scene is translated in ONE request when it fits (<= 40 cues), so the model
  can keep pronouns / gender / tense consistent across cues; longer sequences are
  chunked with a read-only overlap on both sides so context never gets cut;
* Netflix Hebrew TTSG limits are first-class: 17 characters-per-second, 84 characters
  per cue (2 lines x 42);
* filler-word removal is a **user choice** (``style="clean"`` vs ``style="faithful"``),
  never hidden behaviour;
* a missing cue id is an ERROR (``TranslationV2Error``) — source text is *never*
  silently substituted for a translation.

Every capability is independently usable: ``translate_cues`` does translation,
``enforce_cps`` is a separate optional reading-speed pass.

Public API
----------
``LANGUAGE_NAMES``       code -> full English language name (prompts use the name)
``translate_cues()``     translate a cue list, returns copies with ``"translated"``
``enforce_cps()``        optional condensation pass for reading-speed violations
``build_system_prompt()``/``build_user_prompt()``  exposed for inspection & testing
``TranslationV2Error``   raised when the model will not produce every requested cue
``TokenUsage``           token/cost accounting, attached to results as ``.usage``

Nothing here imports the Flask app, Celery, the rate limiter or the legacy translator,
so it is safe to unit-test and to wire in behind a feature flag.
"""

from __future__ import annotations

import json
import logging
import math
import os

try:  # pinned: openai==1.35.13
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai is a hard dependency in production
    OpenAI = None

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o"

#: One request per scene while it fits — whole-scene context is the main quality lever.
MAX_CUES_PER_REQUEST = 40

#: Read-only cues added on each side of a chunk so context is never cut mid-thought.
OVERLAP_CUES = 3

#: Netflix Hebrew TTSG: 42 chars/line x 2 lines.
DEFAULT_MAX_CHARS_PER_CUE = 84

#: Netflix Hebrew TTSG: 17 characters per second (adult programming).
DEFAULT_MAX_CPS = 17.0

#: Low but not zero — deterministic enough for subtitles, still idiomatic.
TEMPERATURE = 0.2

DEFAULT_TIMEOUT_S = 120

#: U+05F4 HEBREW PUNCTUATION GERSHAYIM — the correct mark inside Hebrew acronyms.
GERSHAYIM = "״"

CONTEXT_MARKER = "[CONTEXT-ONLY]"

STYLES = ("clean", "faithful")

#: The app's language codes -> full English names. Prompts NEVER use the bare code:
#: "translate to he" is ambiguous to the model, "translate to Hebrew" is not.
LANGUAGE_NAMES = {
    "he": "Hebrew",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "tr": "Turkish",
}

#: USD per 1M tokens (input, output) — used for cost reporting only.
USD_PER_1M_TOKENS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


# --------------------------------------------------------------------------------------
# Errors & accounting
# --------------------------------------------------------------------------------------


class TranslationV2Error(Exception):
    """
    Raised when translation cannot be completed correctly.

    The important case is a persistent id mismatch: the model did not return a
    translation for every requested cue even after a targeted retry. We raise instead
    of padding with source text — an untranslated cue that looks like a success is
    worse than a visible failure.
    """

    def __init__(self, message: str, missing_ids=None):
        super().__init__(message)
        self.missing_ids = sorted(missing_ids) if missing_ids else []


class TokenUsage:
    """Accumulated token usage and USD cost across all requests of one call."""

    __slots__ = ("prompt_tokens", "completion_tokens", "requests", "cost_usd")

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        self.cost_usd = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.requests += 1
        rate_in, rate_out = USD_PER_1M_TOKENS.get(model, USD_PER_1M_TOKENS[DEFAULT_MODEL])
        self.cost_usd += (
            prompt_tokens / 1_000_000 * rate_in + completion_tokens / 1_000_000 * rate_out
        )

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "cost_usd": round(self.cost_usd, 6),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TokenUsage(in={self.prompt_tokens}, out={self.completion_tokens}, "
            f"requests={self.requests}, cost=${self.cost_usd:.4f})"
        )


class TranslationResult(list):
    """
    ``list[dict]`` of cues, with token accounting attached.

    Behaves exactly like the plain list callers expect (``result[0]["translated"]``),
    but also exposes ``result.usage`` (a :class:`TokenUsage`) so the caller can log or
    bill the request without a second channel.
    """

    def __init__(self, cues=(), usage: TokenUsage = None):
        super().__init__(cues)
        self.usage = usage if usage is not None else TokenUsage()


# --------------------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------------------


def language_name(code: str) -> str:
    """
    Map an app language code to its full English name.

    Regional variants are normalised ("he-IL" -> Hebrew, "zh-CN" -> Chinese). An
    unknown code raises instead of leaking the raw code into the prompt.
    """
    if not code or not isinstance(code, str):
        raise TranslationV2Error(f"Invalid target language: {code!r}")
    base = code.strip().lower().replace("_", "-").split("-")[0]
    name = LANGUAGE_NAMES.get(base)
    if not name:
        raise TranslationV2Error(
            f"Unsupported target language {code!r}; supported: "
            f"{', '.join(sorted(LANGUAGE_NAMES))}"
        )
    return name


def build_system_prompt(
    target_lang: str,
    style: str = "clean",
    *,
    max_chars_per_cue: int = DEFAULT_MAX_CHARS_PER_CUE,
    context_note: str = None,
) -> str:
    """
    Build the subtitler system prompt.

    ``style`` is a **user-facing choice**, not an internal heuristic:

    * ``"clean"``    — remove disfluencies and filler ("uh", "you know", "listen").
                       Subtitles are read, not heard; filler wastes reading time.
    * ``"faithful"`` — keep them. Required when the delivery itself matters
                       (evidence, testimony, comedy timing, verbatim reporting).

    Neither option ever paraphrases away meaning; only the treatment of filler differs.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")

    lang = language_name(target_lang)

    header = [
        f"You are a professional broadcast subtitler producing {lang} subtitles for "
        "television.",
    ]
    if context_note:
        header.append(str(context_note).strip())
    header.append(
        f"Translate each numbered cue into {lang} for burned-in subtitles."
    )

    if style == "clean":
        filler_rule = (
            'REMOVE spoken disfluencies and filler: "uh", "um", "you know", "I mean", '
            '"listen", "look", stutters and false starts. Subtitles are read, not heard '
            "— filler wastes reading time."
        )
    else:
        filler_rule = (
            'KEEP spoken disfluencies and filler: "uh", "um", "you know", "I mean", '
            '"listen", "look", stutters and false starts. Render them with their natural '
            f"{lang} equivalents — this is a faithful, verbatim rendering of the "
            "speaker's delivery."
        )

    rules = [
        f"Output natural, idiomatic spoken {lang} — never a literal, word-for-word "
        "rendering.",
        "PRESERVE sentence punctuation: . , ? ! — every cue must end with proper "
        "punctuation.",
        filler_rule,
        f"Maximum {max_chars_per_cue} characters per cue. Condense the meaning rather "
        "than exceed it.",
        "Numbers one to ten are spelled out in words; 11 and above stay as numerals.",
        "Keep proper nouns and well-known Latin acronyms as-is (ICC).",
    ]

    if lang == "Hebrew":
        rules.append(
            f"Use correct Hebrew typography: gershayim {GERSHAYIM} (U+05F4) inside "
            f"acronyms — צה{GERSHAYIM}ל, חו{GERSHAYIM}ל — never the ASCII quote \"."
        )
    elif lang == "Arabic":
        rules.append(
            "Use correct Arabic typography and punctuation: Arabic comma ، and Arabic "
            "question mark ؟."
        )

    rules += [
        "Read the whole sequence first — it is ONE continuous conversation. Keep "
        "pronouns, gender agreement and verb tense consistent ACROSS cues, and make "
        "each cue flow from the previous one.",
        "A cue that asks a question and the cue that answers it belong to DIFFERENT "
        "speakers. Translate each so it reads naturally on its own.",
        "The source is raw ASR output and may arrive WITHOUT punctuation or "
        "capitalization (a known transcription artifact). Infer sentence boundaries "
        "yourself and punctuate the translation correctly regardless of how broken "
        "the source looks.",
        "When a speaker's or addressee's gender cannot be determined from context, "
        "use MASCULINE grammatical forms consistently (the broadcast convention for "
        "Hebrew and other gendered languages). Never switch grammatical gender "
        "mid-conversation.",
        f"Cues marked {CONTEXT_MARKER} are surrounding dialogue given only so you "
        "understand the scene. Read them, never translate them, and never include "
        "their ids in your output.",
    ]

    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))

    return (
        "\n".join(header)
        + "\n\nHARD RULES\n"
        + numbered
        + '\n\nReturn ONLY JSON: {"cues":[{"id":<int>,"t":"<'
        + lang
        + ' text>"}]} — exactly one entry for every cue you were asked to translate, '
        "reusing the ids you were given."
    )


def build_user_prompt(target_lang: str, items) -> str:
    """
    Build the user message.

    ``items`` is an ordered sequence of ``(cue_id, text, is_context)`` triples. Context
    cues are rendered with the ``[CONTEXT-ONLY]`` marker and are *not* counted in the
    "translate N cues" instruction, so the model knows exactly which ids to emit.
    """
    lang = language_name(target_lang)
    translate_ids = [cid for cid, _text, is_ctx in items if not is_ctx]
    if not translate_ids:
        raise TranslationV2Error("build_user_prompt called with no translatable cues")

    lines = []
    for cue_id, text, is_ctx in items:
        clean_text = " ".join(str(text).split())
        if is_ctx:
            lines.append(f"{cue_id}. {CONTEXT_MARKER} {clean_text}")
        else:
            lines.append(f"{cue_id}. {clean_text}")

    ids_str = _format_id_list(translate_ids)
    head = (
        f"Translate {len(translate_ids)} cues into {lang}.\n"
        f"Emit exactly these ids: {ids_str}.\n"
    )
    if len(translate_ids) != len(items):
        head += (
            f"Lines marked {CONTEXT_MARKER} are context only — do not translate them "
            "and do not include them in the output.\n"
        )
    return head + "\n" + "\n".join(lines)


def _format_id_list(ids) -> str:
    """Compact id list: contiguous runs collapse to "12-51"."""
    ids = sorted(ids)
    if not ids:
        return ""
    if ids[-1] - ids[0] + 1 == len(ids):
        return f"{ids[0]}-{ids[-1]}" if len(ids) > 1 else str(ids[0])
    return ", ".join(str(i) for i in ids)


# --------------------------------------------------------------------------------------
# OpenAI plumbing
# --------------------------------------------------------------------------------------


def _resolve_client(client=None):
    """Return the given client, or build one from the configured API key."""
    if client is not None:
        return client
    if OpenAI is None:  # pragma: no cover
        raise TranslationV2Error("openai package is not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:  # lazy: keeps this module importable without the Flask config
            from config import get_config

            api_key = get_config().OPENAI_API_KEY
        except Exception:  # pragma: no cover
            api_key = None
    if not api_key:
        raise TranslationV2Error("OPENAI_API_KEY is not configured")

    try:
        import httpx

        # Mirrors the legacy translator: an explicit httpx client avoids the
        # "proxies" kwarg incompatibility seen with some httpx versions.
        return OpenAI(api_key=api_key, http_client=httpx.Client(timeout=DEFAULT_TIMEOUT_S))
    except Exception:  # pragma: no cover
        return OpenAI(api_key=api_key)


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _record_usage(usage: TokenUsage, response, model: str) -> None:
    raw = getattr(response, "usage", None)
    usage.add(
        _safe_int(getattr(raw, "prompt_tokens", 0)),
        _safe_int(getattr(raw, "completion_tokens", 0)),
        model,
    )


def _extract_content(response) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise TranslationV2Error(f"Malformed OpenAI response: {exc}")
    if not isinstance(content, str):
        raise TranslationV2Error(
            f"OpenAI response content is not text (got {type(content).__name__})"
        )
    return content.strip()


def _parse_cue_map(content: str) -> dict:
    """
    Parse ``{"cues":[{"id":int,"t":str}]}`` into ``{id: text}``.

    Tolerant about the wrapper (bare list, ``translations`` key) and about the text key
    (``t``/``text``/``translation``) because those are the shapes GPT-4o drifts to —
    but strict about ids: an entry without a usable int id is dropped and logged.
    """
    body = content
    if body.startswith("```"):
        lines = body.split("\n")
        body = "\n".join(lines[1:-1]) if len(lines) > 2 else body

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error("translation_v2: invalid JSON from model: %s", exc)
        logger.debug("translation_v2: raw response head: %s", body[:500])
        raise TranslationV2Error(f"Model returned invalid JSON: {exc}")

    if isinstance(data, dict):
        entries = data.get("cues")
        if entries is None:
            entries = data.get("translations")
        if entries is None:
            raise TranslationV2Error('Model JSON has no "cues" array')
    elif isinstance(data, list):
        entries = data
    else:
        raise TranslationV2Error(f"Unexpected model JSON type: {type(data).__name__}")

    if not isinstance(entries, list):
        raise TranslationV2Error('Model JSON "cues" is not an array')

    out = {}
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning("translation_v2: skipping non-object cue entry: %r", entry)
            continue
        cue_id = entry.get("id")
        try:
            cue_id = int(cue_id)
        except (TypeError, ValueError):
            logger.warning("translation_v2: skipping cue entry without int id: %r", entry)
            continue
        text = entry.get("t")
        if text is None:
            text = entry.get("text")
        if text is None:
            text = entry.get("translation")
        if not isinstance(text, str) or not text.strip():
            logger.warning("translation_v2: empty translation for id %s", cue_id)
            continue
        out[cue_id] = text.strip()
    return out


def _request_cue_map(client, model: str, system: str, user: str, usage: TokenUsage) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout=DEFAULT_TIMEOUT_S,
    )
    _record_usage(usage, response, model)
    return _parse_cue_map(_extract_content(response))


# --------------------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------------------


def translate_cues(
    cues,
    target_lang,
    *,
    style="clean",
    model=DEFAULT_MODEL,
    client=None,
    context_note=None,
):
    """
    Translate subtitle cues with whole-scene context.

    Args:
        cues: sequence of ``{"start": float, "end": float, "text": str}``. Extra keys
            are preserved. The input is never mutated — copies are returned.
        target_lang: app language code (``"he"``, ``"es"``, ...). The prompt always
            uses the full English name from :data:`LANGUAGE_NAMES`.
        style: ``"clean"`` removes spoken filler, ``"faithful"`` keeps it. This is a
            **user choice** surfaced in the UI, not hidden behaviour — the two values
            produce two explicitly different prompt rules.
        model: chat model id (default ``gpt-4o``).
        client: an ``openai.OpenAI`` instance. Built from ``OPENAI_API_KEY`` if omitted.
        context_note: optional one-line scene description injected into the system
            prompt, e.g. "An interview between host X and guest Y." Measurably improves
            register and speaker consistency.

    Returns:
        :class:`TranslationResult` — a ``list[dict]`` of cue copies each with an added
        ``"translated"`` key, plus a ``.usage`` attribute (:class:`TokenUsage`) carrying
        token counts and USD cost.

    Batching:
        <= 40 cues go in a SINGLE request so the model sees the whole scene. Longer
        sequences are split into 40-cue chunks, each carrying 3 read-only cues of
        context on either side; those overlap cues are marked ``[CONTEXT-ONLY]`` and are
        not re-emitted, so every cue is translated exactly once.

    Raises:
        TranslationV2Error: if the model omits cue ids even after one targeted retry.
            Source text is never substituted for a missing translation.
        ValueError: for an invalid ``style``.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")

    usage = TokenUsage()
    cue_list = list(cues or [])
    out = [dict(cue) for cue in cue_list]
    if not out:
        return TranslationResult(out, usage)

    # Blank cues never go to the model; they would be reported as "missing ids".
    texts = {}
    for idx, cue in enumerate(out):
        text = (cue.get("text") or "").strip()
        if text:
            texts[idx + 1] = text
        else:
            cue["translated"] = ""

    if not texts:
        logger.warning("translation_v2: no non-empty cue text to translate")
        return TranslationResult(out, usage)

    # Prompt first: an unsupported language must fail before we go looking for an API key.
    system = build_system_prompt(
        target_lang,
        style,
        max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE,
        context_note=context_note,
    )
    client = _resolve_client(client)

    total = len(out)
    for chunk_start, chunk_end in _chunk_bounds(total):
        target_ids = [i for i in range(chunk_start + 1, chunk_end + 1) if i in texts]
        if not target_ids:
            continue

        context_ids = [
            i
            for i in list(range(max(1, chunk_start + 1 - OVERLAP_CUES), chunk_start + 1))
            + list(range(chunk_end + 1, min(total, chunk_end + OVERLAP_CUES) + 1))
            if i in texts
        ]
        items = sorted(
            [(i, texts[i], False) for i in target_ids]
            + [(i, texts[i], True) for i in context_ids],
            key=lambda item: item[0],
        )

        translated = _request_cue_map(
            client, model, system, build_user_prompt(target_lang, items), usage
        )

        extra = set(translated) - set(target_ids)
        if extra:
            logger.warning(
                "translation_v2: model returned ids outside the request (%s) — ignored",
                sorted(extra),
            )
            for cue_id in extra:
                translated.pop(cue_id, None)

        missing = [i for i in target_ids if i not in translated]
        if missing:
            logger.warning(
                "translation_v2: missing ids %s — one targeted retry", sorted(missing)
            )
            # Initialised BEFORE the retry request that populates it. The legacy path
            # used this dict without ever creating it, so every recovery attempt died
            # with a swallowed NameError.
            retry_translations = {}
            retry_items = sorted(
                [(i, texts[i], False) for i in missing]
                + [(i, texts[i], True) for i, _t, _c in items if i not in missing],
                key=lambda item: item[0],
            )
            try:
                retry_translations = _request_cue_map(
                    client,
                    model,
                    system,
                    build_user_prompt(target_lang, retry_items),
                    usage,
                )
            except TranslationV2Error as exc:
                logger.error("translation_v2: retry request failed: %s", exc)

            for cue_id in list(missing):
                if cue_id in retry_translations:
                    translated[cue_id] = retry_translations[cue_id]

            missing = [i for i in target_ids if i not in translated]
            if missing:
                raise TranslationV2Error(
                    "Translation incomplete after retry: missing cue ids "
                    f"{sorted(missing)} ({len(missing)}/{len(target_ids)} of this chunk)",
                    missing_ids=missing,
                )

        for cue_id, text in translated.items():
            out[cue_id - 1]["translated"] = text

    logger.info(
        "translation_v2: translated %d cues -> %s (style=%s, model=%s) | "
        "tokens in=%d out=%d requests=%d cost=$%.4f",
        len(texts),
        language_name(target_lang),
        style,
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.requests,
        usage.cost_usd,
    )
    return TranslationResult(out, usage)


def _chunk_bounds(total: int):
    """Half-open [start, end) index ranges of cues to translate per request."""
    if total <= MAX_CUES_PER_REQUEST:
        return [(0, total)]
    return [
        (start, min(start + MAX_CUES_PER_REQUEST, total))
        for start in range(0, total, MAX_CUES_PER_REQUEST)
    ]


# --------------------------------------------------------------------------------------
# Reading-speed enforcement
# --------------------------------------------------------------------------------------

_CPS_SYSTEM_PROMPT = (
    "You are a professional broadcast subtitler condensing subtitles that are too long "
    "to read in the time available.\n\n"
    "HARD RULES\n"
    "1. Keep each cue in the SAME language it is already written in. Do NOT translate.\n"
    "2. Never exceed the character limit given in parentheses for that cue.\n"
    "3. Keep the meaning and the register. Drop redundancy, not information.\n"
    "4. PRESERVE sentence punctuation: . , ? ! — every cue must end with proper "
    "punctuation.\n"
    "5. Keep proper nouns, numbers and acronyms, and keep existing typography "
    f"(Hebrew gershayim {GERSHAYIM}).\n\n"
    'Return ONLY JSON: {"cues":[{"id":<int>,"t":"<shortened cue>"}]} — one entry per '
    "cue given, reusing the same ids."
)


def enforce_cps(
    cues,
    *,
    max_cps=DEFAULT_MAX_CPS,
    max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE,
    model=DEFAULT_MODEL,
    client=None,
):
    """
    Optional pass: condense cues that break the reading-speed budget.

    Defaults are the Netflix Hebrew TTSG limits: 17 characters per second and 84
    characters per cue (2 lines x 42). Per cue the effective budget is
    ``min(max_chars_per_cue, floor(max_cps * duration))``.

    Exactly ONE batched request is made, containing only the violating cues (there is
    no loop and no second pass). If a returned cue still breaks its budget, the shorter
    of {model output, original} is kept and a warning is logged — this pass improves
    readability but never blocks delivery, and never changes a compliant cue.

    Args:
        cues: cue dicts with ``start``, ``end`` and ``translated`` (as produced by
            :func:`translate_cues`). Not mutated; copies are returned.

    Returns:
        :class:`TranslationResult` (``list[dict]``) with a ``.usage`` attribute.
    """
    usage = TokenUsage()
    out = [dict(cue) for cue in (cues or [])]
    if not out:
        return TranslationResult(out, usage)

    budgets = {}
    violators = []
    for idx, cue in enumerate(out):
        text = (cue.get("translated") or "").strip()
        if not text:
            continue
        duration = _duration(cue)
        limit = max_chars_per_cue
        if duration > 0:
            limit = min(max_chars_per_cue, int(math.floor(max_cps * duration)))
        limit = max(limit, 1)
        budgets[idx + 1] = limit
        if len(text) > limit:
            violators.append(idx + 1)

    if not violators:
        logger.info(
            "translation_v2.enforce_cps: all %d cues within %.1f CPS / %d chars",
            len(out),
            max_cps,
            max_chars_per_cue,
        )
        return TranslationResult(out, usage)

    logger.info(
        "translation_v2.enforce_cps: condensing %d/%d cues over budget (ids %s)",
        len(violators),
        len(out),
        violators,
    )

    lines = [
        f"{cue_id}. (max {budgets[cue_id]} chars) {out[cue_id - 1]['translated'].strip()}"
        for cue_id in violators
    ]
    user = (
        f"Shorten these {len(violators)} subtitle cues so each fits its character "
        "limit, keeping the same language, meaning and punctuation:\n\n"
        + "\n".join(lines)
    )

    client = _resolve_client(client)
    try:
        shortened = _request_cue_map(client, model, _CPS_SYSTEM_PROMPT, user, usage)
    except TranslationV2Error as exc:
        # Cosmetic pass: a failure here must not fail the job.
        logger.warning("translation_v2.enforce_cps: condensation request failed: %s", exc)
        return TranslationResult(out, usage)

    for cue_id in violators:
        original = out[cue_id - 1]["translated"].strip()
        candidate = shortened.get(cue_id)
        if not candidate:
            logger.warning(
                "translation_v2.enforce_cps: no replacement for cue %d — keeping original",
                cue_id,
            )
            continue
        if len(candidate) > budgets[cue_id]:
            keep = candidate if len(candidate) < len(original) else original
            logger.warning(
                "translation_v2.enforce_cps: cue %d still over budget "
                "(%d chars > %d) — keeping the shorter version (%d chars)",
                cue_id,
                len(candidate),
                budgets[cue_id],
                len(keep),
            )
            out[cue_id - 1]["translated"] = keep
            continue
        out[cue_id - 1]["translated"] = candidate

    logger.info(
        "translation_v2.enforce_cps: done | tokens in=%d out=%d cost=$%.4f",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cost_usd,
    )
    return TranslationResult(out, usage)


def _duration(cue) -> float:
    try:
        return max(0.0, float(cue.get("end", 0) or 0) - float(cue.get("start", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def cps_report(cues, *, max_cps=DEFAULT_MAX_CPS, max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE):
    """
    Measure-only helper: per-cue characters, duration, CPS and whether it is in budget.

    Useful for logging quality metrics without calling the API.
    """
    report = []
    for idx, cue in enumerate(cues or [], 1):
        text = (cue.get("translated") or cue.get("text") or "").strip()
        duration = _duration(cue)
        cps = (len(text) / duration) if duration > 0 else float("inf")
        report.append(
            {
                "id": idx,
                "chars": len(text),
                "duration": round(duration, 3),
                "cps": round(cps, 2) if duration > 0 else None,
                "ok": len(text) <= max_chars_per_cue and (duration <= 0 or cps <= max_cps),
            }
        )
    return report
