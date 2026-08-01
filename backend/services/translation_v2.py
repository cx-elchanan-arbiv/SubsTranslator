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
import time

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
#:
#: DO NOT LOWER THIS. It looks like an obvious knob to turn when translation quality
#: disappoints ("smaller chunks, more attention per cue"), and that intuition was tested
#: head-to-head on the same clip and is WRONG:
#:
#:     chunk 40 (this value)   87% of cues internally punctuated, 100% terminally
#:     chunk 12                79% internally punctuated,          57% terminally
#:                             and +18% translation cost
#:
#: Smaller chunks lose more than they gain: each request sees less of the conversation,
#: so pronouns, gender, tense and terminology drift across the seams, and the model
#: punctuates a fragment less confidently than a scene. The overlap cues
#: (:data:`OVERLAP_CUES`) soften a seam but do not remove it — the only way not to pay
#: for a seam is not to create one.
#:
#: When punctuation is bad, the cause is almost never here: it is upstream, in the ASR.
#: See ``transcription_service.ASR_PUNCTUATION_PRIMER``.
MAX_CUES_PER_REQUEST = 40

#: Cues per condensation request. The CPS pass has no cross-cue context to protect
#: (each cue is shortened on its own), so the only thing the batch size controls is
#: whether the reply fits in the completion budget. An unbounded batch on a long
#: video is a truncated JSON reply, i.e. the whole pass silently lost.
MAX_CUES_PER_CPS_REQUEST = 40

#: Output budget per condensation request. Every reply cue is by construction
#: SHORTER than its input, which is capped at DEFAULT_MAX_CHARS_PER_CUE; ~64 tokens
#: per cue is a generous ceiling for that plus the JSON scaffolding.
CPS_TOKENS_PER_CUE = 64

#: Read-only cues added on each side of a chunk so context is never cut mid-thought.
OVERLAP_CUES = 3

#: Netflix Hebrew TTSG: 42 chars/line x 2 lines.
DEFAULT_MAX_CHARS_PER_CUE = 84

#: Netflix Hebrew TTSG: 17 characters per second (adult programming).
DEFAULT_MAX_CPS = 17.0

#: How far over its budget a cue must be before :func:`enforce_cps` will rewrite it.
#:
#: A cue 2 characters over an 84-character budget is about a tenth of a second of
#: reading time over — invisible. Rewriting it costs tokens AND risks the failure mode
#: this whole pass had to be rebuilt for: a model asked to shorten a nearly-compliant cue
#: has nothing safe to cut, so it cuts something unsafe. 10% is roughly "one word", the
#: smallest edit a condensation can honestly be.
CPS_TRIGGER_MARGIN = 1.10

#: The FLOOR on a condensed cue, as a fraction of its own character limit.
#:
#: The measured failure it stops: cues returned at 31 characters against a 51-character
#: limit, at 9-16 characters under budget on portrait clips, and one that deleted the
#: noun "העולמי" out of "מרכז הסחר העולמי" with 13 characters to spare. Condensation is
#: supposed to be the minimum edit that fits, and without a lower bound the model has no
#: way to know that — "under the limit" is satisfied just as well by deleting half the
#: sentence.
CPS_MIN_KEEP_FRACTION = 0.85

#: Longest a cue may be stretched to by :func:`apply_time_relief`, and the gap it leaves
#: before the next cue. Mirrors ``subtitle_engine.MAX_CUE_DUR`` / ``MIN_CUE_GAP``;
#: restated here because this module deliberately imports nothing, and pinned to them by
#: ``tests/unit/test_translation_v2.py``.
CPS_MAX_CUE_DUR = 6.0
CPS_MIN_CUE_GAP = 0.08

#: Low but not zero — deterministic enough for subtitles, still idiomatic.
TEMPERATURE = 0.2

DEFAULT_TIMEOUT_S = 120

#: U+05F4 HEBREW PUNCTUATION GERSHAYIM — the correct mark inside Hebrew acronyms.
GERSHAYIM = "״"

#: U+05F3 HEBREW PUNCTUATION GERESH — the mark that carries foreign phonemes into
#: Hebrew: ג׳ (j), ז׳ (zh), צ׳/ץ׳ (ch). An ASCII apostrophe in its place is a typography
#: error, and the v2 path was making it (ג'ורג') where the legacy path did not (ג׳ורג׳).
GERESH = "׳"

#: Longest glossary the prompt will carry. A glossary is a hard constraint repeated to
#: the model on every chunk, so it costs input tokens on every request; past a few dozen
#: entries the cost is real and the model's adherence drops anyway. Overflow is dropped
#: with a warning rather than silently truncating the user's intent.
MAX_GLOSSARY_ENTRIES = 40

CONTEXT_MARKER = "[CONTEXT-ONLY]"

#: U+2014 EM DASH + space — the dialogue dash ``subtitle_engine.DIALOGUE_DASH`` prefixes
#: onto both halves of a speaker turn. Spelled out again here rather than imported: this
#: module deliberately depends on nothing, and the two constants are pinned to each other
#: by ``tests/unit/test_subtitle_engine.py``.
DIALOGUE_DASH = "— "

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

    def __init__(self, cues=(), usage: TokenUsage = None, mode: str = "translate"):
        super().__init__(cues)
        self.usage = usage if usage is not None else TokenUsage()
        #: Which contract produced these cues — ``"translate"`` or ``"proofread"``.
        #: A plain list has no room for it and the caller has to archive it.
        self.mode = mode


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


def normalize_glossary(glossary) -> dict:
    """
    Clean a caller-supplied glossary into ``{source term: required translation}``.

    Everything about the input is untrusted — it will eventually come from a UI text
    box — so non-string keys/values, blanks and pure whitespace are dropped rather than
    allowed to reach the prompt as ``None`` or ``123``. Entries past
    :data:`MAX_GLOSSARY_ENTRIES` are dropped with a warning.

    Returns ``{}`` for ``None`` or anything that is not a mapping, so callers can pass
    the value straight through without checking it first.
    """
    if not glossary:
        return {}
    if not hasattr(glossary, "items"):
        logger.warning(
            "translation_v2: glossary must be a mapping, got %s — ignored",
            type(glossary).__name__,
        )
        return {}

    clean = {}
    for source, target in glossary.items():
        if not isinstance(source, str) or not isinstance(target, str):
            logger.warning(
                "translation_v2: dropping non-string glossary entry %r -> %r",
                source,
                target,
            )
            continue
        source, target = source.strip(), target.strip()
        if not source or not target:
            continue
        if len(clean) >= MAX_GLOSSARY_ENTRIES:
            logger.warning(
                "translation_v2: glossary longer than %d entries — dropping %r and the rest",
                MAX_GLOSSARY_ENTRIES,
                source,
            )
            break
        clean[source] = target
    return clean


def build_system_prompt(
    target_lang: str,
    style: str = "clean",
    *,
    max_chars_per_cue: int = DEFAULT_MAX_CHARS_PER_CUE,
    context_note: str = None,
    glossary: dict = None,
) -> str:
    """
    Build the subtitler system prompt.

    ``style`` is a **user-facing choice**, not an internal heuristic:

    * ``"clean"``    — remove disfluencies and filler ("uh", "you know", "listen").
                       Subtitles are read, not heard; filler wastes reading time.
    * ``"faithful"`` — keep them. Required when the delivery itself matters
                       (evidence, testimony, comedy timing, verbatim reporting).

    Neither option ever paraphrases away meaning; only the treatment of filler differs.

    ``glossary`` maps a source term to the exact target rendering it must always get
    (``{"Ivrit": "עִברית"}``). It is rendered as a numbered, quoted list at the END of the
    prompt — last position because it is the most specific instruction present and the
    one a model most easily drops when it is buried among general rules.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")

    lang = language_name(target_lang)
    glossary = normalize_glossary(glossary)

    header = [
        f"You are a professional broadcast subtitler producing {lang} subtitles for "
        "television.",
    ]
    if context_note:
        header.append(str(context_note).strip())
    header.append(
        f"Translate each numbered cue into {lang} for burned-in subtitles."
    )

    filler_examples = (
        '"uh", "um", "you know", "I mean", "listen", "look", "so", sentence-initial '
        '"like", stutters and false starts'
    )
    if style == "clean":
        filler_rule = (
            f"REMOVE spoken disfluencies and filler: {filler_examples}. Subtitles are "
            "read, not heard — filler wastes reading time. Note that these are FILLER "
            "only when they carry no meaning: sentence-initial \"like\" ("
            '"Like, I was there") is filler and is deleted, while comparative "like" '
            '("it moves like a train") is meaning and is translated. The same test '
            'applies to "look" and "listen": drop the interjection, keep the verb.'
        )
    else:
        filler_rule = (
            f"KEEP spoken disfluencies and filler: {filler_examples}. Render them with "
            f"their natural {lang} equivalents — this is a faithful, verbatim rendering "
            "of the speaker's delivery. Even here, sentence-initial \"like\" is an "
            "interjection, NOT the comparative \"like\": never render it as a "
            "comparison word."
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

    # The contrast rule. A clip whose entire POINT was "the language is not called
    # Hebrew, it is called Ivrit" had BOTH names rendered עברית, which turned 36% of its
    # cues into statements that contradict themselves ("it is not called X, it is called
    # X"). A translator collapsing two names onto one target word is normally right —
    # it is the same referent — which is exactly why it needs an explicit exception.
    rules.append(
        "When the source DISTINGUISHES two names or terms for the same thing, the "
        f"{lang} MUST keep them distinguishable. TRANSLITERATE the foreign term "
        f"phonetically into {lang} (optionally in quotes) instead of translating both "
        "names to the same word. This rule is triggered by the SYNTACTIC FRAME, not by "
        "the words in it — treat every one of these as the same construction:\n"
        "   \"it wasn't called X, it was called Y\" / \"they didn't call it X, they "
        'called it Y"\n'
        '   "X, not Y" / "we say X rather than Y" / "the real name is X, not Y"\n'
        "Worked examples. \"They didn't call it Hebrew, they called it Ivrit\" must not "
        'become "לא קראו לזה עברית. קראו לזה עברית." — the second name is '
        'transliterated: "קראו לזה עִברית". "It is not Farsi, it is Parsi" turns on a '
        "single consonant and must keep it: פארסי vs פרסי. A cue that reads \"it is not "
        'called A, it is called A" is a mistranslation, not a subtitle.'
    )
    rules.append(
        f"A cue that begins with the dialogue dash \"{DIALOGUE_DASH}\" MUST begin with "
        "that same dash in your translation. It marks a change of speaker inside the "
        "scene and the timing of that change has already been established from the "
        "audio — do not add it where it is absent and do not remove it where it is "
        "present."
    )

    if lang == "Hebrew":
        rules.append(
            f"Use correct Hebrew typography: gershayim {GERSHAYIM} (U+05F4) inside "
            f"acronyms — צה{GERSHAYIM}ל, חו{GERSHAYIM}ל — never the ASCII quote \"."
        )
        rules.append(
            f"Foreign names and loanwords take a geresh {GERESH} (U+05F3), never an "
            f"ASCII apostrophe ': ג{GERESH}ורג{GERESH}, צ{GERESH}ארלס, ג{GERESH}ז, "
            f"ז{GERESH}אנר."
        )
        rules.append(
            "Grammatical gender must agree WITHIN each noun phrase, not only across "
            "speakers: the noun, its adjectives and its demonstratives all carry the "
            f"noun's own gender (אותה מחווה מדהימה, not את אותו מחווה מדהימה)."
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
        "INFER each speaker's and each addressee's gender from ALL the evidence in the "
        "scene before choosing grammatical forms: names, forms of address, the content "
        "of what they say about themselves, and how other speakers answer them. A woman "
        "answering a question is addressed and answers in FEMININE forms. Only when "
        "nothing in the scene indicates gender at all, default to masculine. Once a "
        "speaker's gender is established, never switch it mid-conversation.",
        f"Cues marked {CONTEXT_MARKER} are surrounding dialogue given only so you "
        "understand the scene. Read them, never translate them, and never include "
        "their ids in your output.",
    ]

    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))

    glossary_block = ""
    if glossary:
        entries = "\n".join(
            f'- "{source}" -> "{target}"' for source, target in glossary.items()
        )
        glossary_block = (
            "\n\nGLOSSARY (binding — overrides every rule above)\n"
            "Render these source terms EXACTLY as given, every time they appear, "
            "including inside a sentence that contrasts one of them with another term:\n"
            + entries
        )

    return (
        "\n".join(header)
        + "\n\nHARD RULES\n"
        + numbered
        + glossary_block
        + '\n\nReturn ONLY JSON: {"cues":[{"id":<int>,"t":"<'
        + lang
        + ' text>"}]} — exactly one entry for every cue you were asked to translate, '
        "reusing the ids you were given."
    )


def same_language(source_lang, target_lang) -> bool:
    """Are these two codes the same language (ignoring region and case)?

    ``"he"``/``"he-IL"``/``"HE"`` are one language; ``"auto"``, ``None`` and anything
    unrecognised are NOT a match, because "we do not know what was spoken" is not the
    same claim as "it was spoken in the target language".
    """
    def base(code):
        if not code or not isinstance(code, str):
            return None
        head = code.strip().lower().replace("_", "-").split("-")[0]
        return head if head in LANGUAGE_NAMES else None

    source, target = base(source_lang), base(target_lang)
    return bool(source) and source == target


def build_proofread_prompt(
    lang_code: str,
    style: str = "clean",
    *,
    max_chars_per_cue: int = DEFAULT_MAX_CHARS_PER_CUE,
    context_note: str = None,
    glossary: dict = None,
) -> str:
    """Build the system prompt for SAME-LANGUAGE work: proofreading, not translating.

    The defect this closes
    ----------------------
    A Hebrew news clip subtitled into Hebrew went through the translator and paid for two
    GPT calls to hand back **23 of 24 cues byte-identical** to their input. That is the
    correct behaviour for a translator — there is nothing to translate — and it is a
    wasted opportunity, because the same call sitting in front of the same text could
    have repaired what the ASR got wrong. The broadcaster's own burned-in captions on
    that clip prove the errors were real and correctable:

        מפכה        -> מפכ״ל        (a rank, missing its gershayim)
        יצחקי ארצוג -> יצחק הרצוג   (a garbled proper noun)
        ההבטחה      -> האבטחה       (the security, heard as the promise)
        עווה        -> עבה
        שאותיר      -> שהותיר

    Every one is a MISHEARING, and every one is obvious from the surrounding sentence to
    a reader who knows the language — which is exactly what the model is.

    The contract
    ------------
    Identical to :func:`build_system_prompt`'s in every mechanical respect — same cue-id
    JSON, same character budget, same filler semantics, same typography rules — and
    opposite in intent: **preserve, do not improve.** Wording, register and word order
    are the speaker's; only what the transcriber demonstrably got wrong may change.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")

    lang = language_name(lang_code)
    glossary = normalize_glossary(glossary)

    header = [
        f"You are a professional {lang} transcript proofreader preparing broadcast "
        "subtitles.",
    ]
    if context_note:
        header.append(str(context_note).strip())
    header.append(
        f"Each numbered cue is raw {lang} speech-recognition output of {lang} speech. "
        f"Return it in {lang}, corrected. This is NOT a translation task."
    )

    if style == "clean":
        filler_rule = (
            'REMOVE spoken disfluencies and filler ("uh", "um", stutters, false starts, '
            "repeated words). Subtitles are read, not heard."
        )
    else:
        filler_rule = (
            "KEEP spoken disfluencies and filler exactly as transcribed — this is a "
            "verbatim record of the delivery."
        )

    rules = [
        "PRESERVE the speaker's own words. Do not paraphrase, do not improve the style, "
        "do not reorder, do not shorten. A cue with nothing wrong in it must come back "
        "CHARACTER-FOR-CHARACTER unchanged.",
        "CORRECT what the speech recogniser plainly got wrong, using the surrounding "
        "cues to decide what was actually said:",
        "  (a) garbled proper nouns — names of people, places, organisations and ranks;",
        "  (b) real words misheard as other real words, where the sentence only makes "
        "sense with the other one;",
        "  (c) missing or wrong punctuation, including sentence-final punctuation;",
        filler_rule,
        f"Maximum {max_chars_per_cue} characters per cue.",
        "Numbers one to ten are spelled out in words; 11 and above stay as numerals.",
        "When you are not certain a word is wrong, LEAVE IT ALONE. A faithful "
        "transcription error is recoverable by a viewer who heard the audio; an "
        "invented 'correction' is not.",
    ]

    if lang == "Hebrew":
        rules.append(
            f"Restore Hebrew typography the recogniser drops: gershayim {GERSHAYIM} "
            f"(U+05F4) inside acronyms and ranks — מפכ{GERSHAYIM}ל, צה{GERSHAYIM}ל, "
            f"חו{GERSHAYIM}ל — and geresh {GERESH} (U+05F3) in foreign names — "
            f"ג{GERESH}ורג{GERESH}. Never the ASCII \" or '."
        )
    elif lang == "Arabic":
        rules.append(
            "Use correct Arabic punctuation: Arabic comma ، and Arabic question mark ؟."
        )

    rules += [
        "Read the whole sequence first — it is ONE continuous recording, and a name "
        "spelled correctly in one cue tells you how to spell it in the next.",
        f"A cue that begins with the dialogue dash \"{DIALOGUE_DASH}\" MUST keep it.",
        f"Cues marked {CONTEXT_MARKER} are surrounding dialogue given only so you "
        "understand the scene. Read them, never return them, and never include their "
        "ids in your output.",
    ]

    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))

    glossary_block = ""
    if glossary:
        entries = "\n".join(
            f'- "{source}" -> "{target}"' for source, target in glossary.items()
        )
        glossary_block = (
            "\n\nGLOSSARY (binding — overrides every rule above)\n"
            "Whenever one of these terms appears, however the recogniser spelled it, "
            "render it EXACTLY as given:\n" + entries
        )

    return (
        "\n".join(header)
        + "\n\nHARD RULES\n"
        + numbered
        + glossary_block
        + '\n\nReturn ONLY JSON: {"cues":[{"id":<int>,"t":"<corrected '
        + lang
        + ' text>"}]} — exactly one entry for every cue you were asked to proofread, '
        "reusing the ids you were given."
    )


def build_user_prompt(target_lang: str, items, *, mode: str = "translate") -> str:
    """
    Build the user message.

    ``items`` is an ordered sequence of ``(cue_id, text, is_context)`` triples. Context
    cues are rendered with the ``[CONTEXT-ONLY]`` marker and are *not* counted in the
    "translate N cues" instruction, so the model knows exactly which ids to emit.

    ``mode`` is ``"translate"`` or ``"proofread"`` — the same JSON contract either way,
    but the instruction has to name the job the system prompt was written for.
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
    if mode == "proofread":
        head = (
            f"Proofread {len(translate_ids)} {lang} cues, correcting only what the "
            "speech recogniser got wrong.\n"
            f"Emit exactly these ids: {ids_str}.\n"
        )
    else:
        head = (
            f"Translate {len(translate_ids)} cues into {lang}.\n"
            f"Emit exactly these ids: {ids_str}.\n"
        )
    if len(translate_ids) != len(items):
        head += (
            f"Lines marked {CONTEXT_MARKER} are context only — do not return them "
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


def _record_llm(recorder, stage: str, system: str, user: str, response, meta: dict) -> None:
    """Hand one request/response pair to an optional research recorder.

    The recorder is **duck-typed and optional on purpose**: this module must stay
    importable and unit-testable with no knowledge of where research archives live, so
    it never imports ``services.research_recorder``. Anything exposing
    ``record_llm(stage, system, user, response, meta)`` will do, including a test spy.

    A recorder that raises is logged at WARNING and ignored. Archiving a run is
    strictly less important than completing it.
    """
    if recorder is None:
        return
    try:
        recorder.record_llm(stage=stage, system=system, user=user, response=response, meta=meta)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("translation_v2: research recorder raised %s — ignored", exc)


def _request_cue_map(
    client,
    model: str,
    system: str,
    user: str,
    usage: TokenUsage,
    max_tokens=None,
    *,
    recorder=None,
    stage: str = "translate",
) -> dict:
    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=DEFAULT_TIMEOUT_S,
            **kwargs,
        )
    except Exception as exc:
        # The corpus needs the failures too: a request that never came back is the
        # most interesting row in it.
        _record_llm(
            recorder,
            stage,
            system,
            user,
            None,
            {
                "model": model,
                "latency_s": round(time.time() - started, 3),
                "max_tokens": int(max_tokens) if max_tokens else None,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    latency = time.time() - started
    _record_usage(usage, response, model)
    raw_usage = getattr(response, "usage", None)
    _record_llm(
        recorder,
        stage,
        system,
        user,
        response,
        {
            "model": model,
            "latency_s": round(latency, 3),
            "max_tokens": int(max_tokens) if max_tokens else None,
            "prompt_tokens": _safe_int(getattr(raw_usage, "prompt_tokens", 0)),
            "completion_tokens": _safe_int(getattr(raw_usage, "completion_tokens", 0)),
        },
    )
    return _parse_cue_map(_extract_content(response))


# --------------------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------------------


def translate_cues(
    cues,
    target_lang,
    *,
    style="clean",
    max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE,
    model=DEFAULT_MODEL,
    client=None,
    context_note=None,
    glossary=None,
    progress_callback=None,
    recorder=None,
    source_lang=None,
):
    """
    Translate subtitle cues with whole-scene context — or PROOFREAD them when the source
    language is already the target language.

    Args:
        cues: sequence of ``{"start": float, "end": float, "text": str}``. Extra keys
            are preserved. The input is never mutated — copies are returned. A cue
            carrying a truthy ``"context_only"`` key is shown to the model as scene
            context and never translated: that is how text the hallucination gate
            REFUSED to ship still reaches the model, so the cue after it does not lose
            the antecedent of its own first pronoun.
        target_lang: app language code (``"he"``, ``"es"``, ...). The prompt always
            uses the full English name from :data:`LANGUAGE_NAMES`.
        style: ``"clean"`` removes spoken filler, ``"faithful"`` keeps it. This is a
            **user choice** surfaced in the UI, not hidden behaviour — the two values
            produce two explicitly different prompt rules.
        max_chars_per_cue: the per-cue character budget written into the prompt. The
            default 84 is 2 lines x 42 — correct only for a frame wide enough to draw
            42 characters. Callers that know the frame pass
            ``subtitle_engine.layout_params(...)["max_chars_per_cue"]`` instead: on a
            720x1280 portrait clip that is 66, and asking the model for 84 there means
            asking for text that cannot be rendered.
        model: chat model id (default ``gpt-4o``).
        client: an ``openai.OpenAI`` instance. Built from ``OPENAI_API_KEY`` if omitted.
        context_note: optional one-line scene description injected into the system
            prompt, e.g. "An interview between host X and guest Y." Measurably improves
            register and speaker consistency.
        glossary: optional ``{source term: required translation}`` map, rendered into the
            system prompt as a binding term list. Empty/``None`` adds nothing to the
            prompt at all, so a job without one is byte-identical to before. Its first
            purpose is terminology CONTRAST — a clip arguing "it is not called Hebrew,
            it is called Ivrit" needs ``{"Ivrit": "עִברית"}`` to stop both names collapsing
            onto one Hebrew word. Invalid entries are dropped, never raised on; see
            :func:`normalize_glossary`.
        progress_callback: called as ``(done, total, message)`` after each chunk.
            Translation is serial and can be minutes long on a feature-length video,
            so without this the UI sits on one frozen step for the whole pass.
            Exceptions raised by the callback are logged and swallowed — reporting
            progress must never be able to fail a translation.
        recorder: optional research recorder (duck-typed: anything with a
            ``record_llm(stage, system, user, response, meta)`` method). Every request
            this function makes — scene chunks and the targeted retry — is handed over
            verbatim. ``None`` disables it entirely, and a recorder that raises is
            logged and ignored; see :func:`_record_llm`.
        source_lang: the language actually spoken, when it is known. When it equals
            ``target_lang`` the system prompt is swapped for the ASR-PROOFREADING
            contract (:func:`build_proofread_prompt`) — same JSON, same budget, opposite
            intent: correct what the recogniser misheard and change nothing else.
            ``None`` or ``"auto"`` means "unknown", and the translation prompt is used.

    Returns:
        :class:`TranslationResult` — a ``list[dict]`` of cue copies each with an added
        ``"translated"`` key, plus a ``.usage`` attribute (:class:`TokenUsage`) carrying
        token counts and USD cost. Context-only cues come back with ``"translated": ""``
        — they are the caller's to discard.

        The result also carries ``.mode`` (``"translate"`` or ``"proofread"``) so the
        caller can log and archive which contract was actually used.

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
    proofread = same_language(source_lang, target_lang)
    mode = "proofread" if proofread else "translate"
    if not out:
        return TranslationResult(out, usage, mode=mode)

    # Blank cues never go to the model; they would be reported as "missing ids".
    # Context-only cues DO go to the model, as context, and are never asked for back.
    texts = {}
    context_only = set()
    for idx, cue in enumerate(out):
        text = (cue.get("text") or "").strip()
        if not text:
            cue["translated"] = ""
            continue
        texts[idx + 1] = text
        if cue.get("context_only"):
            context_only.add(idx + 1)
            cue["translated"] = ""

    if not texts or not (set(texts) - context_only):
        logger.warning("translation_v2: no non-empty cue text to translate")
        return TranslationResult(out, usage, mode=mode)

    # Prompt first: an unsupported language must fail before we go looking for an API key.
    build_prompt = build_proofread_prompt if proofread else build_system_prompt
    system = build_prompt(
        target_lang,
        style,
        max_chars_per_cue=max_chars_per_cue,
        context_note=context_note,
        glossary=glossary,
    )
    if proofread:
        logger.info(
            "translation_v2: source and target are both %s — running the ASR-PROOFREAD "
            "contract instead of translating",
            language_name(target_lang),
        )
    client = _resolve_client(client)

    total = len(out)
    bounds = _chunk_bounds(total)
    for chunk_index, (chunk_start, chunk_end) in enumerate(bounds):
        target_ids = [
            i
            for i in range(chunk_start + 1, chunk_end + 1)
            if i in texts and i not in context_only
        ]
        if not target_ids:
            continue

        _report(
            progress_callback,
            chunk_start,
            total,
            f"{'Proofreading' if proofread else 'Translating'} cues "
            f"{chunk_start + 1}-{chunk_end} of {total} "
            f"(chunk {chunk_index + 1}/{len(bounds)})",
        )

        # Read-only neighbours: the overlap on either side of the chunk, plus every
        # context-only cue INSIDE it (text the hallucination gate refused to ship —
        # dropping it from the prompt as well is how a translator loses the antecedent
        # of the next line's pronoun).
        context_ids = [
            i
            for i in list(range(max(1, chunk_start + 1 - OVERLAP_CUES), chunk_start + 1))
            + list(range(chunk_end + 1, min(total, chunk_end + OVERLAP_CUES) + 1))
            if i in texts
        ]
        context_ids += [
            i for i in range(chunk_start + 1, chunk_end + 1) if i in context_only
        ]
        items = sorted(
            [(i, texts[i], False) for i in target_ids]
            + [(i, texts[i], True) for i in sorted(set(context_ids))],
            key=lambda item: item[0],
        )

        translated = _request_cue_map(
            client,
            model,
            system,
            build_user_prompt(target_lang, items, mode=mode),
            usage,
            recorder=recorder,
            stage=f"{mode}_chunk_{chunk_index + 1}",
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
                    build_user_prompt(target_lang, retry_items, mode=mode),
                    usage,
                    recorder=recorder,
                    stage=f"{mode}_retry_{chunk_index + 1}",
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

        _report(
            progress_callback,
            chunk_end,
            total,
            f"Translated {chunk_end} of {total} cues",
        )

    if proofread:
        unchanged = sum(
            1
            for i, text in texts.items()
            if i not in context_only
            and " ".join(str(out[i - 1].get("translated") or "").split())
            == " ".join(text.split())
        )
        asked = len(set(texts) - context_only)
        logger.info(
            "translation_v2: proofread %d %s cues (%d returned unchanged, %d corrected)",
            asked,
            language_name(target_lang),
            unchanged,
            asked - unchanged,
        )

    logger.info(
        "translation_v2: %s %d cues -> %s (style=%s, model=%s) | "
        "tokens in=%d out=%d requests=%d cost=$%.4f",
        "proofread" if proofread else "translated",
        len(set(texts) - context_only),
        language_name(target_lang),
        style,
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.requests,
        usage.cost_usd,
    )
    return TranslationResult(out, usage, mode=mode)


def _chunk_bounds(total: int):
    """Half-open [start, end) index ranges of cues to translate per request."""
    if total <= MAX_CUES_PER_REQUEST:
        return [(0, total)]
    return [
        (start, min(start + MAX_CUES_PER_REQUEST, total))
        for start in range(0, total, MAX_CUES_PER_REQUEST)
    ]


def _report(callback, done: int, total: int, message: str) -> None:
    """Best-effort progress notification. A broken callback never fails a job."""
    if not callback:
        return
    try:
        callback(done, total, message)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("translation_v2: progress callback raised %s", exc)


# --------------------------------------------------------------------------------------
# Reading-speed enforcement
# --------------------------------------------------------------------------------------

_CPS_SYSTEM_PROMPT = (
    "You are a professional broadcast subtitler condensing subtitles that are too long "
    "to read in the time available.\n\n"
    "HARD RULES\n"
    "1. Keep each cue in the SAME language it is already written in. Do NOT translate.\n"
    "2. Never exceed the character limit given in parentheses for that cue.\n"
    "3. SHORTEN BY THE MINIMUM NECESSARY. This is the rule that matters most: you are "
    "trimming a cue to fit, not rewriting it. Aim to land just under the limit. NEVER "
    "return a cue shorter than 85% of its stated limit — a cue 15 characters under its "
    "limit is not 'safer', it is a cue that threw away words it had room for.\n"
    "4. Drop FILLER before content: redundant adverbs, repeated words, discourse "
    "markers, doubled adjectives. Never delete a noun, a name, a number, a negation or "
    "a whole clause while filler is still present.\n"
    "5. Keep the meaning and the register. Drop redundancy, not information.\n"
    "6. PRESERVE sentence punctuation: . , ? ! — every cue must end with proper "
    "punctuation.\n"
    "7. Keep proper nouns, numbers and acronyms, and keep existing typography "
    f"(Hebrew gershayim {GERSHAYIM}, geresh {GERESH}).\n"
    f"8. A cue that begins with the dialogue dash \"{DIALOGUE_DASH}\" must still begin "
    "with it.\n\n"
    'Return ONLY JSON: {"cues":[{"id":<int>,"t":"<shortened cue>"}]} — one entry per '
    "cue given, reusing the same ids."
)

#: Second-pass prompt, sent only for the cues the first pass got wrong — in either
#: direction. Both failures are named explicitly because the model has just produced one
#: of them and needs to know which.
_CPS_REASK_PROMPT = (
    "You are a professional broadcast subtitler. A previous condensation pass produced "
    "cues that are still wrong, and you are fixing exactly those.\n\n"
    "Each line gives the ORIGINAL cue, its character limit, and the ATTEMPT that "
    "failed. An attempt fails in one of two ways:\n"
    "  TOO LONG   — it still exceeds the limit. Cut more.\n"
    "  TOO SHORT  — it fits, but it threw away content it had room for. Restore what "
    "was lost from the ORIGINAL, up to just under the limit.\n\n"
    "HARD RULES\n"
    "1. Keep the cue in the SAME language it is written in. Do NOT translate.\n"
    "2. Never exceed the character limit given for that cue.\n"
    "3. Land BETWEEN 85% and 100% of the limit. Both edges are real failures.\n"
    "4. Work from the ORIGINAL, not from the failed attempt — the attempt may have "
    "deleted the most important word in the cue.\n"
    "5. Drop filler before content; keep proper nouns, numbers, negations and "
    "punctuation.\n\n"
    'Return ONLY JSON: {"cues":[{"id":<int>,"t":"<corrected cue>"}]} — one entry per '
    "cue given, reusing the same ids."
)


def _cps_budget(duration, max_cps, max_chars_per_cue):
    """Characters this cue may carry: the tighter of the reading-speed and frame limits."""
    limit = max_chars_per_cue
    if duration > 0:
        limit = min(max_chars_per_cue, int(math.floor(max_cps * duration)))
    return max(int(limit), 1)


def apply_time_relief(
    cues,
    *,
    max_cps=DEFAULT_MAX_CPS,
    max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE,
    max_dur=CPS_MAX_CUE_DUR,
    min_gap=CPS_MIN_CUE_GAP,
    video_duration=None,
):
    """Give an over-long cue more TIME before anyone considers taking away its WORDS.

    The defect this closes
    ----------------------
    ``enforce_cps`` used to reach straight for the model. But a cue is over budget
    because of a RATIO — characters over seconds — and the pipeline had just finished
    capping the denominator (``subtitle_engine.LEAD_OUT_MAX``). Deleting words to fix a
    number that a free extra second of silence would have fixed is destroying content to
    solve a problem that was not about content: the corpus holds a cue whose whole
    subject noun ("העולמי", of "מרכז הסחר העולמי") was deleted while 13 characters of
    headroom sat unused, and four portrait cues that came back 9-16 characters UNDER
    their limit because a clause had been dropped rather than trimmed.

    So: extend first, condense only what time cannot fix. This is the ONE place allowed
    to exceed ``LEAD_OUT_MAX``, and only for a cue that genuinely cannot be read in the
    time it has — a lead-out is a comfort, being unreadable is a defect.

    Bounded by everything that bounds a cue: the next cue's start (less ``min_gap``),
    ``max_dur`` from its own start, and the end of the video.

    Args:
        cues: cue dicts with ``start``, ``end`` and a translation. Not mutated.
        video_duration: hard end of the picture, when known.

    Returns:
        ``(cues, relieved)`` — new copies, and how many cues were extended.
    """
    out = [dict(cue) for cue in (cues or [])]
    relieved = 0
    for index, cue in enumerate(out):
        if cue.get("context_only"):
            continue  # never shown, so it has no reading speed to relieve
        text = _measured_text(cue)
        if not text:
            continue
        duration = _duration(cue)
        if duration <= 0:
            continue
        if len(text) <= _cps_budget(duration, max_cps, max_chars_per_cue):
            continue
        if len(text) > max_chars_per_cue:
            continue  # over the FRAME budget: no amount of time can fix that

        try:
            start = float(cue.get("start", 0) or 0)
            end = float(cue.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue

        needed_end = start + len(text) / float(max_cps)
        ceiling = start + float(max_dur)
        if index + 1 < len(out):
            try:
                ceiling = min(ceiling, float(out[index + 1].get("start", 0) or 0) - min_gap)
            except (TypeError, ValueError):
                pass
        if video_duration:
            ceiling = min(ceiling, float(video_duration))

        new_end = min(needed_end, ceiling)
        if new_end > end:
            cue["end"] = round(new_end, 3)
            relieved += 1

    if relieved:
        logger.info(
            "translation_v2.enforce_cps: gave %d over-long cue(s) more TIME instead of "
            "shortening their text",
            relieved,
        )
    return out, relieved


def _choose_candidate(original, candidates, limit):
    """Pick the best replacement for one cue: the LONGEST that fits its budget.

    "Longest that fits" is the whole point. Every candidate here is a lossy compression
    of the same sentence, so among the ones that are legal, the one that threw away the
    least is the best one — which is the exact opposite of the instinct ("shortest is
    safest") that deleted a proper noun with 13 characters to spare.

    Falls back to the shortest of {every candidate, the original} when NONE fits: the
    pass never blocks delivery, it just gets as close as it can and says so. The ORIGINAL
    is in that fallback set deliberately — a "condensation" that came back LONGER than
    what it was given has failed at the one thing it was asked to do, and shipping it
    would make the cue worse in both dimensions at once.
    """
    fitting = [c for c in candidates if c and len(c) <= limit]
    if fitting:
        return max(fitting, key=len), True
    usable = [c for c in candidates if c] + [original]
    return min(usable, key=len), False


def enforce_cps(
    cues,
    *,
    max_cps=DEFAULT_MAX_CPS,
    max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE,
    model=DEFAULT_MODEL,
    client=None,
    progress_callback=None,
    recorder=None,
    video_duration=None,
    time_relief=True,
):
    """
    Optional pass: make cues readable in the time they have — with time first, text last.

    Defaults are the Netflix Hebrew TTSG limits: 17 characters per second and 84
    characters per cue (2 lines x 42). Per cue the effective budget is
    ``min(max_chars_per_cue, floor(max_cps * duration))``.

    ``max_chars_per_cue`` is a *frame* property as much as an editorial one — pass
    ``subtitle_engine.layout_params(...)["max_chars_per_cue"]`` so a portrait video is
    condensed to what its narrow frame can render (66 chars at 720x1280) rather than to
    the landscape 84.

    What this pass does, in order
    -----------------------------
    1. **Time relief** (:func:`apply_time_relief`) — a cue that is over budget because it
       is SHORT, not because it is WORDY, is given the silence in front of it instead of
       losing words. Nothing is sent to the model for a cue this fixes.
    2. **Trigger** — only cues more than :data:`CPS_TRIGGER_MARGIN` over what is left are
       sent at all. A cue 2 characters over its budget is 0.1 seconds of reading time
       over, which no viewer has ever noticed; paying tokens to rewrite it and risking a
       deleted noun to save it is a bad trade in both directions.
    3. **Condense** — one request per batch, under a prompt with an explicit FLOOR
       (never below 85% of the limit) as well as a ceiling.
    4. **Re-ask, once** — for every cue the first pass got wrong in EITHER direction:
       still over the limit, or crushed below the floor. The re-ask is given the
       original text, the limit and the failed attempt, and the winner is chosen by
       :func:`_choose_candidate` — the longest candidate that fits.

    Why the floor exists at all: the first version of this pass had only a ceiling, and
    the model duly obeyed it by deleting whatever it liked. Measured on this project's
    own corpus it cut a cue from 53 to 31 characters when 51 were allowed, deleted the
    subject noun of "מרכז הסחר העולמי" with 13 characters of headroom, and returned four
    portrait cues 9-16 characters under their limits with whole clauses missing. A budget
    with no lower bound is not a budget, it is a licence.

    This pass improves readability but never blocks delivery: a failed batch leaves that
    batch's cues untouched, and a cue that cannot be fixed keeps the best available text
    with a warning.

    Args:
        cues: cue dicts with ``start``, ``end`` and ``translated`` (as produced by
            :func:`translate_cues`). Not mutated; copies are returned. TIMINGS MAY
            CHANGE — see step 1; that is the point.
        video_duration: end of the picture, so time relief cannot run past it.
        time_relief: set False to keep the old text-only behaviour (used by tests that
            pin condensation independently of timing).
        progress_callback: ``(done, total, message)``, called per batch. Best effort.
        recorder: optional research recorder, same contract as
            :func:`translate_cues` — every condensation request is archived verbatim.

    Returns:
        :class:`TranslationResult` (``list[dict]``) with a ``.usage`` attribute.
    """
    usage = TokenUsage()
    out = [dict(cue) for cue in (cues or [])]
    if not out:
        return TranslationResult(out, usage)

    if time_relief:
        out, _relieved = apply_time_relief(
            out,
            max_cps=max_cps,
            max_chars_per_cue=max_chars_per_cue,
            video_duration=video_duration,
        )

    budgets = {}
    originals = {}
    violators = []
    for idx, cue in enumerate(out):
        text = (cue.get("translated") or "").strip()
        if not text:
            continue
        limit = _cps_budget(_duration(cue), max_cps, max_chars_per_cue)
        budgets[idx + 1] = limit
        originals[idx + 1] = text
        if len(text) > limit * CPS_TRIGGER_MARGIN:
            violators.append(idx + 1)

    if not violators:
        logger.info(
            "translation_v2.enforce_cps: all %d cues within %.1f CPS / %d chars "
            "(or inside the %.0f%% trigger margin)",
            len(out),
            max_cps,
            max_chars_per_cue,
            (CPS_TRIGGER_MARGIN - 1) * 100,
        )
        return TranslationResult(out, usage)

    logger.info(
        "translation_v2.enforce_cps: condensing %d/%d cues more than %.0f%% over budget "
        "(ids %s)",
        len(violators),
        len(out),
        (CPS_TRIGGER_MARGIN - 1) * 100,
        violators,
    )

    client = _resolve_client(client)
    batches = [
        violators[i: i + MAX_CUES_PER_CPS_REQUEST]
        for i in range(0, len(violators), MAX_CUES_PER_CPS_REQUEST)
    ]
    still_over = 0
    over_condensed = 0
    for batch_index, batch in enumerate(batches):
        _report(
            progress_callback,
            batch_index * MAX_CUES_PER_CPS_REQUEST,
            len(violators),
            f"Condensing over-long cues (batch {batch_index + 1}/{len(batches)})",
        )
        lines = [
            f"{cue_id}. (max {budgets[cue_id]} chars, never below "
            f"{_floor_chars(budgets[cue_id])}) {originals[cue_id]}"
            for cue_id in batch
        ]
        user = (
            f"Shorten these {len(batch)} subtitle cues so each fits its character "
            "limit, keeping the same language, meaning and punctuation. Trim by the "
            "minimum necessary — do not go below the stated minimum:\n\n"
            + "\n".join(lines)
        )
        try:
            shortened = _request_cue_map(
                client,
                model,
                _CPS_SYSTEM_PROMPT,
                user,
                usage,
                max_tokens=len(batch) * CPS_TOKENS_PER_CUE,
                recorder=recorder,
                stage=f"cps_batch_{batch_index + 1}",
            )
        except TranslationV2Error as exc:
            # Cosmetic pass: a failure here must not fail the job, and must not
            # cost the batches that would have succeeded.
            logger.warning(
                "translation_v2.enforce_cps: condensation batch %d/%d failed: %s",
                batch_index + 1,
                len(batches),
                exc,
            )
            continue

        # Who still needs work, and why — both directions.
        reask = {}
        for cue_id in batch:
            candidate = (shortened.get(cue_id) or "").strip()
            limit = budgets[cue_id]
            if not candidate:
                logger.warning(
                    "translation_v2.enforce_cps: no replacement for cue %d — "
                    "keeping original",
                    cue_id,
                )
                continue
            if len(candidate) > limit:
                reask[cue_id] = (candidate, "TOO LONG")
            elif len(candidate) < _floor_chars(limit):
                reask[cue_id] = (candidate, "TOO SHORT")
            else:
                out[cue_id - 1]["translated"] = candidate

        if not reask:
            continue

        logger.info(
            "translation_v2.enforce_cps: re-asking %d cue(s) from batch %d "
            "(%d too long, %d over-condensed)",
            len(reask),
            batch_index + 1,
            sum(1 for _c, why in reask.values() if why == "TOO LONG"),
            sum(1 for _c, why in reask.values() if why == "TOO SHORT"),
        )
        reask_lines = [
            f"{cue_id}. (max {budgets[cue_id]} chars, never below "
            f"{_floor_chars(budgets[cue_id])}) [{why}: "
            f"{len(attempt)} chars] ORIGINAL: {originals[cue_id]} || ATTEMPT: {attempt}"
            for cue_id, (attempt, why) in sorted(reask.items())
        ]
        try:
            second = _request_cue_map(
                client,
                model,
                _CPS_REASK_PROMPT,
                f"Fix these {len(reask)} cues:\n\n" + "\n".join(reask_lines),
                usage,
                max_tokens=len(reask) * CPS_TOKENS_PER_CUE,
                recorder=recorder,
                stage=f"cps_reask_{batch_index + 1}",
            )
        except TranslationV2Error as exc:
            logger.warning(
                "translation_v2.enforce_cps: re-ask for batch %d failed: %s",
                batch_index + 1,
                exc,
            )
            second = {}

        for cue_id, (attempt, _why) in reask.items():
            limit = budgets[cue_id]
            best, fits = _choose_candidate(
                originals[cue_id],
                [attempt, (second.get(cue_id) or "").strip()],
                limit,
            )
            out[cue_id - 1]["translated"] = best
            if not fits:
                still_over += 1
                logger.warning(
                    "translation_v2.enforce_cps: cue %d still over budget after a "
                    "re-ask (%d chars > %d) — keeping the shortest available",
                    cue_id,
                    len(best),
                    limit,
                )
            elif len(best) < _floor_chars(limit):
                over_condensed += 1
                logger.warning(
                    "translation_v2.enforce_cps: cue %d came back over-condensed even "
                    "after a re-ask (%d chars for a %d-char budget) — shipping it, but "
                    "content was probably lost",
                    cue_id,
                    len(best),
                    limit,
                )

    _report(progress_callback, len(violators), len(violators), "Reading-speed pass done")

    logger.info(
        "translation_v2.enforce_cps: done | %d condensed, %d still over budget, "
        "%d over-condensed | tokens in=%d out=%d cost=$%.4f",
        len(violators),
        still_over,
        over_condensed,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cost_usd,
    )
    return TranslationResult(out, usage)


def _floor_chars(limit: int) -> int:
    """The fewest characters a condensed cue may carry — see :data:`CPS_MIN_KEEP_FRACTION`."""
    return int(math.ceil(int(limit) * CPS_MIN_KEEP_FRACTION))


def _duration(cue) -> float:
    try:
        return max(0.0, float(cue.get("end", 0) or 0) - float(cue.get("start", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _measured_text(cue) -> str:
    """The text a reading-speed measurement should actually be taken on.

    The translation arrives under one of TWO spellings depending on where in the
    pipeline the cue is: :func:`translate_cues` writes ``translated``, and
    ``subtitle_pipeline.normalize_cues`` immediately rewrites that to
    ``translated_text``. ``process_video_task`` normalises before measuring, so a
    reader that only knows ``translated`` silently falls through to ``text`` and
    measures the untranslated SOURCE — which is a different language, a different
    length, and therefore a meaningless CPS number reported as a real one.

    Blank is treated as absent (not merely missing), for the same reason
    ``normalize_cues`` does: the previous stage leaves ``translated_text: ""`` behind.
    """
    for key in ("translated", "translated_text", "text"):
        value = cue.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def cps_report(cues, *, max_cps=DEFAULT_MAX_CPS, max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE):
    """
    Measure-only helper: per-cue characters, duration, CPS and whether it is in budget.

    Useful for logging quality metrics without calling the API. Measures the
    TRANSLATION when there is one, under either of the two key spellings the pipeline
    uses — see :func:`_measured_text`.
    """
    report = []
    for idx, cue in enumerate(cues or [], 1):
        text = _measured_text(cue)
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
