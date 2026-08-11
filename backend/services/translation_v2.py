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
import re
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

#: Smallest LAST chunk worth sending on its own. A shorter tail is folded back into the
#: chunk before it, which is allowed to run up to ``MAX_CUES_PER_REQUEST + this - 1``.
#:
#: The defect: a 41-cue podcast (40 exactly, until the hallucination gate stopped
#: deleting a real cue) chunked into 40 + 1. Cue 41 went out alone, with three
#: [CONTEXT-ONLY] English lines and not one word of the Hebrew that had just been
#: written for the forty cues before it — 1125 prompt tokens to produce 34 tokens of
#: output with no idea what gender, register or tense the rest of the scene had settled
#: on. It duly came back in the feminine for a male speaker. A seam is a cost you accept
#: to fit inside a context window; a seam that isolates a single cue buys nothing at all.
MIN_TAIL_CUES = 10

#: Netflix Hebrew TTSG: 42 chars/line x 2 lines.
DEFAULT_MAX_CHARS_PER_CUE = 84

#: Netflix Hebrew TTSG: 17 characters per second (adult programming).
DEFAULT_MAX_CPS = 17.0

#: How far over its budget a cue may be, both BEFORE and AFTER condensation.
#:
#: A cue 2 characters over an 84-character budget is about a tenth of a second of
#: reading time over — invisible. Rewriting it costs tokens AND risks the failure mode
#: this whole pass had to be rebuilt for: a model asked to shorten a nearly-compliant cue
#: has nothing safe to cut, so it cuts something unsafe. 10% is roughly "one word", the
#: smallest edit a condensation can honestly be.
#:
#: ONE margin, used at BOTH ends of the pass — that symmetry is the fix for a measured
#: regression, not a tidiness preference. R8 shipped a 10% margin on the way IN and a
#: zero-tolerance limit on the way OUT, so a cue could be judged "not worth rewriting"
#: at 33 characters and, once the model had condensed it TO 33 characters, be judged
#: "still wrong" and sent back for a second, deeper cut. On the corpus that asymmetry
#: cost the adjective the whole cue turned on twice in eight clips:
#:
#:     "אנחנו לא יודעים אם זה מטוס מסחרי."   33 chars, limit 31 -> re-asked
#:       -> "אנחנו לא יודעים אם זה היה מטוס."  ("commercial" gone, and the NEXT cue
#:                                             is the one that says "private")
#:     "לצה״ל יש מוניטין קשוח. הם נחשבים קשוחים."  40 chars, limit 38 -> re-asked
#:       -> "לצה״ל יש מוניטין קשוח. הם נחשבים מאוד."  (a modifier with nothing to modify)
#:
#: Both attempts were inside the margin that would have stopped the pass from ever
#: touching them. If 10% over is acceptable in a cue we never touched, it is acceptable
#: in a cue we just improved.
CPS_TRIGGER_MARGIN = 1.10

#: The FLOOR on a condensed cue, as a fraction of its own character limit — now stated
#: to the MODEL in the prompt, and no longer used to trigger a second request.
#:
#: The measured failure it stops: cues returned at 31 characters against a 51-character
#: limit, at 9-16 characters under budget on portrait clips, and one that deleted the
#: noun "העולמי" out of "מרכז הסחר העולמי" with 13 characters to spare. Condensation is
#: supposed to be the minimum edit that fits, and without a lower bound the model has no
#: way to know that — "under the limit" is satisfied just as well by deleting half the
#: sentence.
#:
#: Why it stopped being an enforcement threshold: a character count is a PROXY for
#: "you deleted content", and :func:`_cps_rejection` now measures the thing itself. The
#: proxy's false positives were expensive — see :data:`CPS_MIN_BUDGET` and the removal
#: of the TOO SHORT re-ask.
CPS_MIN_KEEP_FRACTION = 0.85

#: Smallest character budget :func:`enforce_cps` will ask the model to hit.
#:
#: A budget under 20 characters is under about one second of screen time, and a cue that
#: short has no filler left to give: the honest answers are all longer than the budget,
#: so the only way to satisfy the request is to delete meaning. MEASURED over the
#: eight-clip corpus — every single condensation attempted under a 20-character budget
#: destroyed content, and none succeeded:
#:
#:     budget  7   "איפה אתם?"          -> "איפה את"      (truncated mid-word)
#:     budget 13   "אתה טועה. עִברית."   -> "אתה טועה."    (the word the clip is ABOUT)
#:     budget 14   "זו הייתה שפה כנענית." -> "זו הייתה שפה."
#:     budget 17   "זה לא העברית המקורית," -> "זה לא העברית."
#:     budget 19   "זו התרגום לאנגלית. אנחנו מדברים אנגלית." -> "זו התרגום לאנגלית."
#:
#: Above 20 the same pass produced clean, minimal trims ("והם צעקו וצרחו אחד על השני."
#: -> "והם צעקו אחד על השני." at a 21-character budget). So: leave these cues alone and
#: let them run a little fast. An over-fast correct subtitle is a readability cost; an
#: amputated one is a translation error.
CPS_MIN_BUDGET = 20

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
        rate_in, rate_out = USD_PER_1M_TOKENS.get(
            model, USD_PER_1M_TOKENS[DEFAULT_MODEL]
        )
        self.cost_usd += (
            prompt_tokens / 1_000_000 * rate_in
            + completion_tokens / 1_000_000 * rate_out
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


#: A Latin term worth locking: starts with a capital, ends on a letter or digit, and may
#: carry ``&``, ``.`` or ``-`` inside (``AT&T``, ``U.S.``, ``COVID-19``). Applied to BOTH
#: sides of a chunk, which is what makes trailing sentence punctuation harmless — "Trump."
#: reduces to ``Trump`` in the source and in the translation alike.
_LATIN_TERM_RE = re.compile(r"[A-Z][A-Za-z0-9&.\-]*[A-Za-z0-9]")


def _is_lockable_term(token: str) -> bool:
    """Is this Latin token the KIND of thing a video must spell the same way twice?

    Two shapes, both chosen because they are what actually drifted: an ACRONYM (all
    caps) and a PROPER NAME (capitalised). Everything else — lowercase words, single
    letters, "A" — is ordinary vocabulary, and pinning ordinary vocabulary to one
    rendering would freeze the translator's grammar, not its terminology.

    Both shapes require **3 characters**. The acronym rule used to accept 2, which
    swept in US, EU, TV, UN, AI — the acronyms most likely to HAVE an established
    target rendering (ארה״ב, האיחוד האירופי, טלוויזיה, האו״ם). Locking those is the
    opposite of the intent: whichever way chunk 1 happened to render one of them
    would become law for the whole video, so a mechanism built to stop drift would
    be pinning an accident instead. A 3-character floor is not a stop-list of
    specific words — it is the length below which a Latin token in this corpus is
    almost always a common abbreviation rather than a name, and it costs nothing:
    every term measured drifting (AIPAC, ISIS, NATO, JCPOA) is longer.
    """
    if not token or not token[0].isupper():
        return False
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    return len(token) >= 3


def _harvest_latin_locks(
    source_texts: list[str], translated_texts: list[str]
) -> dict[str, str]:
    """Terms this chunk left in Latin script, as glossary entries for the NEXT chunk.

    THE MEASURED FAILURE: on a 5-minute video, AIPAC stayed Latin through chunk 1 and
    turned into "איפא״ק" at its first appearance in chunk 2; ISIS and the spelling of
    "אסלאם" drifted the same way. Nothing was wrong with either rendering on its own —
    the fault is that one video used both, and each chunk is a separate request with no
    memory of the last one's decisions.

    The harvest is deliberately narrow: a term is locked ONLY when it appears in the
    chunk's source AND survives VERBATIM in the chunk's translation. That single test
    does the work of a whole heuristic — a term the model chose to transliterate or
    translate is simply absent from the translation and is never locked, so this can
    only ever say "you already kept this one in Latin, keep doing that", which is the
    exact drift that was measured. It never pushes a rendering the model did not itself
    produce.

    Pure. Returns ``{token: token}`` (identity mappings — the lock IS "leave it alone"),
    empty when nothing qualifies.
    """
    kept = set()
    for text in translated_texts or []:
        kept.update(_LATIN_TERM_RE.findall(str(text or "")))
    if not kept:
        return {}

    locks: dict[str, str] = {}
    for text in source_texts or []:
        for token in _LATIN_TERM_RE.findall(str(text or "")):
            if token in kept and _is_lockable_term(token):
                locks[token] = token
    return locks


def _script_switches(
    source_texts: list[str], translated_texts: list[str], locked: dict
) -> list[str]:
    """Terms the video has ALREADY kept in Latin that this chunk translated away.

    The known blind spot of :func:`_harvest_latin_locks`, made visible instead of
    guessed at. The lock can only ever say "keep leaving this in Latin", because a
    term the model translated is absent from the translation and there is no way to
    learn WHICH target word it became without word-level alignment or a separate
    glossary pass. So ISIS -> "דאעש" in one chunk and ISIS in another is drift the
    lock cannot prevent.

    Building the entity ledger that WOULD prevent it costs an extra model pass per
    video, and one measured occurrence is not enough to justify that. This detector
    is the cheap half: it costs nothing, it never changes output, and it turns "we
    think this drifts" into a count. If the log stays empty across real use, the
    ledger is not needed; if it fills up, it is — and the evidence will be sitting
    in the run archive either way.

    Pure. Returns the offending source terms.
    """
    if not locked:
        return []
    present = set()
    for text in translated_texts or []:
        present.update(_LATIN_TERM_RE.findall(str(text or "")))
    switched = []
    for text in source_texts or []:
        for token in _LATIN_TERM_RE.findall(str(text or "")):
            if token in locked and token not in present and token not in switched:
                switched.append(token)
    return switched


def _merge_locks(glossary, locks: dict) -> dict:
    """The caller's glossary plus harvested term locks, caller's entries winning.

    Order matters twice over. The caller's entries are inserted FIRST so that
    :data:`MAX_GLOSSARY_ENTRIES` truncation on a long video drops harvested locks rather
    than the terms a human explicitly asked for; and ``setdefault`` means a term the
    caller pinned is never overwritten by whatever the model happened to do with it in
    chunk 1. Locks stop being added at the cap for the same reason — so the prompt never
    has to be built from an over-long glossary and log a warning per chunk.

    Pure.
    """
    merged = normalize_glossary(glossary)
    for token, rendering in (locks or {}).items():
        if len(merged) >= MAX_GLOSSARY_ENTRIES:
            break
        merged.setdefault(token, rendering)
    return merged


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
    header.append(f"Translate each numbered cue into {lang} for burned-in subtitles.")

    filler_examples = (
        '"uh", "um", "you know", "I mean", "listen", "look", "so", sentence-initial '
        '"like", stutters and false starts'
    )
    if style == "clean":
        # ONE rule, not two. This used to be a "MECHANICAL RULE, no judgement required"
        # (delete any opening "you know," / "well," / "look,") followed by an EXCEPTION
        # that "outranks" it and requires exactly the judgement the first half forbids.
        # A model handed two rules that contradict each other follows whichever one it
        # read last, which is not a policy. The judgement is the rule.
        filler_rule = (
            f"REMOVE spoken disfluencies and filler: {filler_examples}. Subtitles are "
            "read, not heard — filler wastes reading time. But these words are filler "
            "only when they carry NO meaning, so decide each one on what it is doing:\n"
            '   Sentence-initial "like" ("Like, I was there") is filler and goes; '
            'comparative "like" ("it moves like a train") is meaning and is translated. '
            'The same test applies to "look" and "listen": drop the interjection, keep '
            'the verb — "Do you know what that means?" is a question, not a marker.\n'
            '   An opening discourse marker — "you know," / "I mean," / "like," / '
            '"well," / "look," / "listen," — is DROPPED when it is doing nothing but '
            'filling time ("You know, like, he really got hurt." is "הוא באמת נפגע."), '
            "and KEPT and translated when it is doing interpersonal work: direct address "
            '("look, Nick" — a parent softening what follows), softening, hedging, or '
            "carrying the speaker's stance toward what they are about to say.\n"
            "   The test, applied to that cue and not to the word in the abstract: read "
            "the sentence without it. If it says the same thing and the speaker sounds "
            "the same, it was filler. If the speaker comes out colder, blunter or more "
            "certain than they were, it was meaning and it stays."
        )
    else:
        filler_rule = (
            f"KEEP spoken disfluencies and filler: {filler_examples}. Render them with "
            f"their natural {lang} equivalents — this is a faithful, verbatim rendering "
            'of the speaker\'s delivery. Even here, sentence-initial "like" is an '
            'interjection, NOT the comparative "like": never render it as a '
            "comparison word."
        )

    rules = [
        f"Output natural, idiomatic spoken {lang} — never a literal, word-for-word "
        "rendering.",
        # THE 38%-OF-VIDEO DISASTER. This rule used to end "every cue must end with
        # proper punctuation", and combined with "read the whole batch" that is an
        # INSTRUCTION to complete a mid-sentence fragment from the cue after it. Verified
        # in the raw LLM archive: source cue 42 "President Johnson refused to greenlight
        # a preemptive" / 43 "strike by Israel... in 1967." came back with 42 holding the
        # whole sentence and 43 holding cue 44's content. The offset then grew to 3 and
        # held to the end of the chunk. A cue is a TIMED unit — it is on screen while
        # those words are being spoken — so a fragment that ends mid-sentence is correct
        # and completing it is theft from the next cue.
        "PRESERVE sentence punctuation: . , ? ! — a cue that ends a sentence in the "
        "source ends one in your translation, with the same mark.\n"
        "   A CUE IS AN INDEPENDENT UNIT. Translate the words that are in it and only "
        "those. If a cue stops mid-sentence, your translation stops mid-sentence too: do "
        "NOT complete it, do NOT pull words forward from the next cue, and do NOT push "
        "words back into the previous one. Never merge two cues into one, and never "
        "leave a cue empty because you already said its content somewhere else. The "
        "punctuation requirement does NOT apply to a fragment that continues into the "
        "next cue — such a fragment correctly ends with no terminal mark at all.\n"
        "   Each cue is on screen only while those exact words are spoken, so content "
        "moved between cues appears at the wrong moment on the picture, and every cue "
        "after it is wrong too.",
        filler_rule,
        # Measured twice in judged rounds: "Fucking shit" came back as "איזה שטויות"
        # ("what nonsense"), and in another run a profanity was silently deleted — the
        # cue shipped without it and nothing said so. Both times the model was doing
        # what it thought was a favour.
        "PROFANITY, insults and vulgarity are MEANING, not noise. Translate them at "
        "the SOURCE'S FULL INTENSITY, using the equivalent register in "
        f"{lang} — an angry speaker stays angry, a crude speaker stays crude. Never "
        "soften them into mild words, never censor or asterisk them, and never delete "
        "them. This is not a matter of taste: a subtitle that tones a speaker down is "
        'reporting a different person saying a different thing. The "clean" style '
        "removes FILLER — never profanity.",
        'PRESERVE deliberate rhetorical repetition. "a great, great honor" is a '
        "speech pattern, not a stutter — render both words. Collapsing it to one "
        "flattens the speaker. (Genuine stutters and false starts follow the filler "
        "rule above instead.)",
        "Read the WHOLE batch before translating any cue, and let the scene decide "
        'domain terms: "gains" beside a protein shake is muscle, not profit; a word '
        'being ARGUED ABOUT ("it\'s called X") is quoted as itself, not translated. '
        "When the cues around a term name its world, that world wins over the "
        "dictionary.",
        f"Maximum {max_chars_per_cue} characters per cue. Condense the meaning rather "
        "than exceed it.",
        "Numbers one to ten are spelled out in words; 11 and above stay as numerals.",
        "Keep proper nouns and well-known Latin acronyms as-is (ICC).",
        # Measured on a 5-minute video: AIPAC stayed Latin for the whole of chunk 1 and
        # became "איפא״ק" at its first occurrence in chunk 2; ISIS and the spelling of
        # "אסלאם" drifted the same way. Each chunk is a separate request, so each one
        # re-decides — and a viewer watching one video sees one term wearing two names.
        "A recurring name, acronym or term must be rendered IDENTICALLY every time it "
        "appears — throughout the WHOLE video, not merely within the cues in front of "
        "you. Pick one rendering (one script, one spelling) and never switch to another "
        "later. Any term list below may carry renderings ALREADY USED EARLIER IN THIS "
        "SAME VIDEO: reuse those exactly, even where you would have chosen differently "
        "— consistency across the video outranks your preference within this batch.",
        # "Fear Factor" came back as "פחד גורם" — two words that mean "fear" and
        # "causing", in that order, which is not a title, not a phrase and not Hebrew.
        # The same clip got "Survivor" and "The Amazing Race" right, because those two
        # have Israeli broadcast names and that one does not.
        "TITLES of television programmes, films, books, songs and companies are "
        f"translated ONLY when a real, established {lang} release title exists "
        '("Survivor" is "הישרדות", "The Amazing Race" is "המרוץ למיליון"). If you are '
        "not certain such a title exists, LEAVE THE ORIGINAL TITLE IN LATIN SCRIPT — "
        'never translate it word by word ("Fear Factor" is "Fear Factor", never '
        '"פחד גורם"). A made-up title is not a translation; it is a programme that does '
        "not exist.",
    ]

    # The contrast rule. A clip whose entire POINT was "the language is not called
    # Hebrew, it is called Ivrit" had BOTH names rendered עברית, which turned 36% of its
    # cues into statements that contradict themselves ("it is not called X, it is called
    # X"). A translator collapsing two names onto one target word is normally right —
    # it is the same referent — which is exactly why it needs an explicit exception.
    #
    # The dictated Hebrew outputs for that one clip were DELETED from this rule: it used
    # to spell out the exact strings "לא קראו לזה Hebrew. קראו לזה עברית." and
    # "מה המילה העברית ל-Hebrew?". A prompt that scripts the answer for one video is not
    # a rule, it is that video's expected output smuggled into the system prompt — it
    # cannot generalise and it makes the rule impossible to evaluate on anything else.
    # The PRINCIPLE stays, with an illustrative example (Farsi/Parsi) that is not the
    # clip it was learned on.
    rules.append(
        "When the source DISTINGUISHES two names or terms for the same thing, the "
        f"{lang} MUST keep them distinguishable. TRANSLITERATE the foreign term "
        f"phonetically into {lang} (optionally in quotes) instead of translating both "
        "names to the same word. This rule is triggered by the SYNTACTIC FRAME, not by "
        "the words in it — treat every one of these as the same construction:\n"
        '   "it wasn\'t called X, it was called Y" / "they didn\'t call it X, they '
        'called it Y"\n'
        '   "X, not Y" / "we say X rather than Y" / "the real name is X, not Y"\n'
        '"It is not Farsi, it is Parsi" turns on a single consonant and must keep it: '
        'פארסי vs פרסי. A cue that reads "it is not called A, it is called A" is a '
        "mistranslation, not a subtitle.\n"
        f"WHEN TRANSLITERATION CANNOT CARRY THE CONTRAST — because the {lang} for the "
        f"foreign term and the transliteration of it are the SAME {lang} word, or differ "
        "only by diacritics — KEEP THE FOREIGN TERM IN LATIN SCRIPT instead. Diacritics "
        "are NOT a contrast: subtitle fonts do not draw them, so two spellings that "
        "differ only in diacritics are one word on screen and a cue built on the "
        "difference between them says nothing. Rescuing such a cue with diacritics is "
        "not a solution; keeping one of the two names in Latin script is.\n"
        "WHICH one stays in Latin: the name being REJECTED, or the one being NAMED "
        "rather than used. Translate the other, so the cue still reads as a "
        f"correction in {lang} instead of as a contradiction.\n"
        "THE SAME TREATMENT APPLIES TO EVERY FRAME LISTED ABOVE — they are one problem "
        "in different syntax, and a model that fixes only the frame it was shown "
        "leaves the rest contradicting themselves. Worked through on a pair that has "
        "nothing to do with your material:\n"
        '   "They don\'t call it Greece, they call it Hellas."   -> "Greece" stays '
        f'Latin, "Hellas" goes into {lang}\n'
        '   "It is called Hellas, not Greece."                  -> the same two words, '
        "the same treatment\n"
        f'   "What is the {lang} word for Greece?"              -> "Greece" stays '
        "Latin; rendering it in the target as well turns the question into its own "
        "answer\n"
        '   "How do you say Greece in Greek?"                   -> the LANGUAGE name is '
        "translated, the WORD being asked about stays Latin\n"
        "TEST YOUR OWN OUTPUT: if a cue denies and asserts the same string, this rule "
        "applies to it and you have not applied it."
    )
    rules.append(
        f'A cue that begins with the dialogue dash "{DIALOGUE_DASH}" MUST begin with '
        "that same dash in your translation. It marks a change of speaker inside the "
        "scene and the timing of that change has already been established from the "
        "audio — do not add it where it is absent and do not remove it where it is "
        "present."
    )

    if lang == "Hebrew":
        rules.append(
            f"Use correct Hebrew typography: gershayim {GERSHAYIM} (U+05F4) inside "
            f'acronyms — צה{GERSHAYIM}ל, חו{GERSHAYIM}ל — never the ASCII quote ".'
        )
        rules.append(
            f"Foreign names and loanwords take a geresh {GERESH} (U+05F3), never an "
            f"ASCII apostrophe ': ג{GERESH}ורג{GERESH}, צ{GERESH}ארלס, ג{GERESH}ז, "
            f"ז{GERESH}אנר."
        )
        rules.append(
            # Clause (3) was DELETED: 'a counted noun above ten is singular — "243 שנה",
            # never "243 שנים"'. That is not Hebrew grammar. The singular after a large
            # number is an optional LITERARY pattern confined to units of measure, time
            # and currency; the general rule is the opposite, and the instruction was
            # making the model write "243 איש" where "243 אנשים" is what Hebrew says.
            # A prompt rule that is wrong is worse than a missing one — the model obeys
            # it.
            #
            # Clause (2) was NARROWED. "military and honorific titles take the article"
            # is true of ranks and offices before a name ("הגנרל ג'ורג' וושינגטון") and
            # false of מר / ד"ר / פרופ' ("מר כהן", never "המר כהן") and of direct
            # address ("אדוני הנשיא", not "האדוני הנשיא").
            "Hebrew micro-grammar that machine output keeps getting wrong: (1) in a "
            'definite construct chain only the LAST noun is definite — "לב הארגמן", '
            'never "הלב הארגמן"; (2) a MILITARY RANK or an OFFICE standing before a '
            'proper name takes the article — "הגנרל ג\'ורג\' וושינגטון", "הנשיא טראמפ" '
            f"— but the personal honorifics מר, ד{GERSHAYIM}ר and פרופ{GERESH} do NOT "
            '("מר כהן", never "המר כהן"), and neither does direct address '
            '("אדוני הנשיא").'
        )
        rules.append(
            "Grammatical gender must agree WITHIN each noun phrase, not only across "
            "speakers: the noun, its adjectives and its demonstratives all carry the "
            "noun's own gender (אותה מחווה מדהימה, not את אותו מחווה מדהימה)."
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
        "INFER each speaker's and each addressee's gender ONLY from EXPLICIT TEXTUAL "
        "EVIDENCE in the cues in front of you: a personal name, a gendered form of "
        "address, a gendered pronoun used about that person, or a gendered noun they "
        'use about themselves or their relationships ("my husband", "my wife", '
        '"ma\'am", "sir", "she said"). That is the whole list of admissible '
        "evidence. Tone, topic, politeness, warmth, who is asking and who is answering, "
        "and what the scene 'feels' like are NOT evidence and must never move you off "
        "the default. WITH NO SUCH EVIDENCE, USE MASCULINE FORMS. This is mandatory, "
        "not a preference: you cannot see the speakers, so a feminine form you inferred "
        "from atmosphere is a coin flip printed on the screen, while masculine is the "
        "broadcast convention for unknown gender and reads as neutral. Never invent a "
        "gender to make a line sound natural. Once a speaker's gender is fixed by "
        "evidence, never switch it mid-conversation.",
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
        f'A cue that begins with the dialogue dash "{DIALOGUE_DASH}" MUST keep it.',
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
        return OpenAI(
            api_key=api_key, http_client=httpx.Client(timeout=DEFAULT_TIMEOUT_S)
        )
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
            logger.warning(
                "translation_v2: skipping cue entry without int id: %r", entry
            )
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


# --------------------------------------------------------------------------------------
# Alignment: is cue N's translation actually a translation of cue N?
# --------------------------------------------------------------------------------------
#: Largest uniform cue shift the alignment check looks for.
#:
#: THE MEASURED FAILURE: validation checked only that every requested id was PRESENT
#: (set membership) and then bound text to cue by index. A response in which every id
#: exists but all the content has slid by one therefore passed in complete silence.
#: Verified in the raw LLM archive: source cue 42 "President Johnson refused to
#: greenlight a preemptive" / 43 "strike by Israel... in 1967." came back with 42 holding
#: the whole sentence and 43 holding cue 44's content. The offset then grew to 3 and held
#: to the end of the chunk — 38% of that video was subtitled with the wrong line.
#:
#: 3 because 3 is what was measured. A drift larger than that is not a drift, it is a
#: different response, and the anchor evidence for it would be indistinguishable from
#: noise on a 40-cue chunk.
ALIGNMENT_MAX_OFFSET = 3

#: How many more cues a SHIFTED reading must explain than the aligned one before the
#: chunk is rejected.
#:
#: One extra agreement is a coincidence — a name that happens to appear in two adjacent
#: cues. Two independent cues both agreeing with the shifted reading and disagreeing
#: with the ids is the signature of the defect. Deliberately conservative in the safe
#: direction: a false NEGATIVE costs the same silent shift that already ships today,
#: while a false POSITIVE spends a whole extra chunk request on a correct response.
ALIGNMENT_MIN_ADVANTAGE = 2

#: Plausible band for ``len(translation) / len(source)`` on a correctly bound cue.
#:
#: A weak, corroborating signal only. Hebrew usually comes back shorter than English and
#: Spanish longer, so the band is wide on purpose: its job is to be able to CONTRADICT a
#: shifted reading, never to detect one on its own.
ALIGNMENT_LENGTH_BAND = (0.4, 2.5)

#: Digit runs, and Latin-script tokens of two characters or more.
_ANCHOR_RE = re.compile(r"[0-9]+|[A-Za-z][A-Za-z0-9'’\-]*")


def _anchors(text: str) -> set:
    """The tokens of one cue that SURVIVE translation, so they can be matched across it.

    Digits survive by rule (the prompt requires "11 and above stay as numerals") and
    Latin-script tokens survive whenever the prompt's own rules keep them: proper nouns,
    acronyms, programme titles, and the terms the cross-chunk lock pins in Latin. Those
    are the only things a Hebrew or Arabic translation still shares with its English
    source, which is exactly what makes them usable as alignment anchors.

    Single Latin letters are excluded: "a", "I" and stray initials appear everywhere and
    would agree with anything. Digit runs are kept at any length — a lone "7" inside a
    line of Hebrew is a strong anchor.

    Pure. Lower-cased, so casing drift is not mistaken for a different token.
    """
    found = set()
    for match in _ANCHOR_RE.finditer(str(text or "")):
        token = match.group(0)
        if token[0].isdigit():
            found.add(token)
        elif len(token) >= 2:
            found.add(token.lower())
    return found


def _compare_key(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form: "is this the same line?"."""
    return " ".join(
        "".join(
            ch for ch in str(text or "").lower() if ch.isalnum() or ch.isspace()
        ).split()
    )


def _alignment_scores(sources: list, targets: list, max_offset: int) -> dict:
    """For each shift ``k``, how well does "target p is really source p+k" hold up?

    Two independent measurements per pair, and they are kept separate on purpose:

    ``anchor_hits``
        how many pairs share at least one :func:`_anchors` token. Content evidence, and
        the only thing allowed to DECIDE.
    ``length_fit``
        the fraction of pairs whose length ratio is inside
        :data:`ALIGNMENT_LENGTH_BAND`. Corroboration only — it can veto a shifted
        reading, never establish one.

    Pure.
    """
    low, high = ALIGNMENT_LENGTH_BAND
    scores = {}
    for offset in range(0, max(0, int(max_offset)) + 1):
        hits = 0
        fitting = 0
        pairs = 0
        for position, target in enumerate(targets):
            source_position = position + offset
            if source_position >= len(sources):
                break
            source = sources[source_position]
            pairs += 1
            if _anchors(source) & _anchors(target):
                hits += 1
            source_len = len(str(source or "").strip())
            if (
                source_len
                and low <= len(str(target or "").strip()) / source_len <= high
            ):
                fitting += 1
        scores[offset] = {
            "pairs": pairs,
            "anchor_hits": hits,
            "length_fit": round(fitting / pairs, 3) if pairs else 0.0,
        }
    return scores


def _detect_chunk_offset(
    sources: list, targets: list, *, max_offset: int = ALIGNMENT_MAX_OFFSET
) -> dict:
    """Does a uniform shift explain this chunk's content better than the ids do?

    THE INVARIANT: a response whose ids are all present is not therefore correct. This
    is a CONTENT check, run before anything is bound, and it is the only thing standing
    between the archive's measured off-by-one and a delivered file in which every
    subtitle is one line early.

    The aligned reading (``offset == 0``) is the default and wins every tie. A shift is
    declared only when it explains at least :data:`ALIGNMENT_MIN_ADVANTAGE` more cues by
    anchor content AND the length evidence does not contradict it. With no anchors
    anywhere in the chunk — a possibility, since anchors are exactly the tokens that
    survive translation — there is nothing to measure and the answer is "aligned": this
    check reports what it can see and never guesses.

    Pure.

    Returns:
        ``{"offset": int, "scores": {k: {...}}, "why": str}``. ``offset`` is 0 when the
        chunk looks correctly bound.
    """
    scores = _alignment_scores(sources or [], targets or [], max_offset)
    aligned = scores.get(0, {"anchor_hits": 0, "length_fit": 0.0, "pairs": 0})
    best_offset = 0
    best = aligned
    for offset, score in scores.items():
        if offset == 0:
            continue
        if score["anchor_hits"] > best["anchor_hits"]:
            best_offset, best = offset, score

    if not best_offset:
        return {"offset": 0, "scores": scores, "why": "no shift explains more content"}
    advantage = best["anchor_hits"] - aligned["anchor_hits"]
    if advantage < ALIGNMENT_MIN_ADVANTAGE:
        return {
            "offset": 0,
            "scores": scores,
            "why": (
                f"shift {best_offset} explains only {advantage} more cue(s) than the "
                f"ids do — under the {ALIGNMENT_MIN_ADVANTAGE} needed to reject"
            ),
        }
    if best["length_fit"] < aligned["length_fit"]:
        return {
            "offset": 0,
            "scores": scores,
            "why": (
                f"shift {best_offset} matches {advantage} more anchor(s) but its cue "
                f"lengths fit worse ({best['length_fit']} < {aligned['length_fit']}) — "
                "not enough to reject"
            ),
        }
    return {
        "offset": best_offset,
        "scores": scores,
        "why": (
            f"a uniform shift of {best_offset} matches the anchors of "
            f"{best['anchor_hits']}/{best['pairs']} cues where the ids match only "
            f"{aligned['anchor_hits']}/{aligned['pairs']}; length fit "
            f"{best['length_fit']} vs {aligned['length_fit']}"
        ),
    }


def _record_llm(
    recorder, stage: str, system: str, user: str, response, meta: dict
) -> None:
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
        recorder.record_llm(
            stage=stage, system=system, user=user, response=response, meta=meta
        )
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


def _reask_misaligned_chunk(
    *,
    request,
    target_lang: str,
    texts: dict,
    target_ids: list,
    context_items: list,
    mode: str,
    chunk_index: int,
) -> dict:
    """Get an ALIGNED translation for a chunk that came back shifted, or fail loudly.

    Two escalating attempts, because they fail for different reasons:

    1. **The same chunk, asked again.** A shift is a decoding accident, not a property
       of the request — the archive's shifted response came from a prompt that already
       said one entry per id. Most of the time asking again is enough, and it costs one
       request.
    2. **One request per cue.** A per-cue request cannot be misaligned: there is exactly
       one source cue in it, so there is nothing for the content to slide against. The
       other cues ride along as read-only context so whole-scene consistency (gender,
       tense, terminology) survives the fallback. It is expensive on purpose — it is
       reached only after a chunk has come back wrong twice.

    Never returns a shifted map, and never returns a partial one: an id the model will
    not translate even one-at-a-time raises, exactly as a missing id already does.

    Args:
        request: ``(items, stage) -> {id: text}``; the caller owns the client, model and
            prompt, so this function stays testable with a plain callable.
        context_items: the chunk's read-only neighbours, as ``(id, text, True)``.
    """
    wanted = set(target_ids)
    sources = [texts[i] for i in target_ids]

    items = sorted(
        [(i, texts[i], False) for i in target_ids] + list(context_items),
        key=lambda item: item[0],
    )
    try:
        retry = {
            i: t
            for i, t in request(items, f"{mode}_realign_{chunk_index + 1}").items()
            if i in wanted
        }
    except TranslationV2Error as exc:
        logger.error("translation_v2: the realignment request failed: %s", exc)
        retry = {}

    dropped = [i for i in target_ids if i not in retry]
    if not dropped:
        check = _detect_chunk_offset(sources, [retry[i] for i in target_ids])
        if not check["offset"]:
            logger.info(
                "translation_v2: chunk %d came back ALIGNED on the second ask — %s",
                chunk_index + 1,
                check["why"],
            )
            return retry
        logger.error(
            "translation_v2: chunk %d is STILL shifted by %d on the second ask (%s) — "
            "falling back to one request per cue, where a shift is impossible",
            chunk_index + 1,
            check["offset"],
            check["why"],
        )
    else:
        logger.error(
            "translation_v2: the realignment request dropped ids %s — falling back to "
            "one request per cue",
            sorted(dropped),
        )

    per_cue: dict = {}
    for cue_id in target_ids:
        single_items = sorted(
            [(cue_id, texts[cue_id], False)]
            + [(i, texts[i], True) for i in target_ids if i != cue_id]
            + list(context_items),
            key=lambda item: item[0],
        )
        try:
            answer = request(single_items, f"{mode}_percue_{chunk_index + 1}_{cue_id}")
        except TranslationV2Error as exc:
            logger.error(
                "translation_v2: per-cue request for %s failed: %s", cue_id, exc
            )
            continue
        if cue_id in answer:
            per_cue[cue_id] = answer[cue_id]

    still_missing = [i for i in target_ids if i not in per_cue]
    if still_missing:
        raise TranslationV2Error(
            "Translation misaligned and unrecoverable: chunk "
            f"{chunk_index + 1} came back shifted, and cue ids {sorted(still_missing)} "
            "could not be translated one at a time either",
            missing_ids=still_missing,
        )
    logger.warning(
        "translation_v2: chunk %d recovered one cue at a time (%d requests) after two "
        "shifted responses",
        chunk_index + 1,
        len(target_ids),
    )
    return per_cue


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

        Chunks are separate requests and so decide terminology separately — measured:
        AIPAC Latin in chunk 1, "איפא״ק" in chunk 2. Every chunk therefore feeds the
        terms it kept in Latin script forward as glossary entries for the ones after it
        (:func:`_harvest_latin_locks`, merged by :func:`_merge_locks` with the caller's
        glossary winning). A single-chunk job builds exactly the prompt it always did.

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
    # Terms this video has already committed to, harvested chunk by chunk and fed
    # forward as glossary entries — see :func:`_harvest_latin_locks`. Empty on chunk 1,
    # which therefore builds the byte-identical prompt it always did.
    locks: dict[str, str] = {}
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
            for i in list(
                range(max(1, chunk_start + 1 - OVERLAP_CUES), chunk_start + 1)
            )
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

        # The term lock costs one extra prompt build per chunk and nothing else. With no
        # locks yet (chunk 1, or a proofread) the prompt object built above is reused
        # unchanged.
        chunk_system = system
        if locks:
            chunk_system = build_prompt(
                target_lang,
                style,
                max_chars_per_cue=max_chars_per_cue,
                context_note=context_note,
                glossary=_merge_locks(glossary, locks),
            )

        translated = _request_cue_map(
            client,
            model,
            chunk_system,
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
                    chunk_system,
                    build_user_prompt(target_lang, retry_items, mode=mode),
                    usage,
                    recorder=recorder,
                    stage=f"{mode}_retry_{chunk_index + 1}",
                )
            except TranslationV2Error as exc:
                logger.error("translation_v2: retry request failed: %s", exc)

            # A retry that fills a hole by COPYING a line it already produced has not
            # recovered anything — it has printed one cue's words over another cue's
            # timing. The archive shows exactly that. Two cues whose SOURCE text is the
            # same are allowed to share a translation ("Yes." / "Yes."); two cues that
            # said different things are not.
            assigned = {_compare_key(t): cue_id for cue_id, t in translated.items()}
            for cue_id in list(missing):
                candidate = retry_translations.get(cue_id)
                if candidate is None:
                    continue
                holder = assigned.get(_compare_key(candidate))
                if holder is not None and _compare_key(texts.get(holder, "")) != (
                    _compare_key(texts.get(cue_id, ""))
                ):
                    logger.error(
                        "translation_v2: the retry filled cue %s with a translation "
                        "already bound to cue %s, whose SOURCE says something else — "
                        "refusing it rather than shipping cue %s's words on cue %s's "
                        "timing: %r",
                        cue_id,
                        holder,
                        holder,
                        cue_id,
                        candidate[:60],
                    )
                    continue
                translated[cue_id] = candidate
                assigned[_compare_key(candidate)] = cue_id

            missing = [i for i in target_ids if i not in translated]
            if missing:
                raise TranslationV2Error(
                    "Translation incomplete after retry: missing cue ids "
                    f"{sorted(missing)} ({len(missing)}/{len(target_ids)} of this chunk)",
                    missing_ids=missing,
                )

        # === Alignment: every id being PRESENT is not the same as being CORRECT ===
        #
        # Until this check existed, text was bound to cue by index the moment the ids
        # validated. A response where all 40 ids are there and all 40 contents have slid
        # by one passed silently, and the archive holds one that did — the shift grew to
        # 3 and held to the end of the chunk. Nothing from a shifted response is bound.
        alignment = _detect_chunk_offset(
            [texts[i] for i in target_ids], [translated[i] for i in target_ids]
        )
        if alignment["offset"]:
            logger.error(
                "translation_v2: chunk %d (cues %d-%d) came back MISALIGNED by %d — %s. "
                "NOTHING from this response is bound; re-requesting the chunk.",
                chunk_index + 1,
                target_ids[0],
                target_ids[-1],
                alignment["offset"],
                alignment["why"],
            )
            translated = _reask_misaligned_chunk(
                # `system` is bound as a default so the closure carries THIS chunk's
                # prompt (term locks and all), not whatever the loop variable holds by
                # the time it is called.
                request=lambda request_items, stage, system=chunk_system: (
                    _request_cue_map(
                        client,
                        model,
                        system,
                        build_user_prompt(target_lang, request_items, mode=mode),
                        usage,
                        recorder=recorder,
                        stage=stage,
                    )
                ),
                target_lang=target_lang,
                texts=texts,
                target_ids=target_ids,
                context_items=[
                    (i, t, True) for i, t, is_context in items if is_context
                ],
                mode=mode,
                chunk_index=chunk_index,
            )

        for cue_id, text in translated.items():
            out[cue_id - 1]["translated"] = text

        # Harvest AFTER the chunk is complete, so a retry's answers are included.
        # Never in proofread mode: source and target are the same language there, so
        # "the term survived verbatim" is true of every word and carries no information.
        if not proofread:
            chunk_sources = [texts[i] for i in target_ids if i in texts]
            chunk_targets = [translated[i] for i in target_ids if i in translated]

            # Detect first: a term this chunk translated away can still be one an
            # EARLIER chunk locked, and the lock has no way to stop that. Logging
            # only — see _script_switches for why the fix is deferred.
            switched = _script_switches(chunk_sources, chunk_targets, locks)
            if switched:
                logger.warning(
                    "translation_v2: term drift — chunk %d translated %s, which an "
                    "earlier chunk had kept in Latin script; the same video now uses "
                    "two renderings for it",
                    chunk_index + 1,
                    ", ".join(sorted(switched)),
                )

            found = _harvest_latin_locks(chunk_sources, chunk_targets)
            fresh = {t: r for t, r in found.items() if t not in locks}
            if fresh:
                logger.info(
                    "translation_v2: term lock — chunk %d kept %s in Latin script; "
                    "later chunks are told to do the same",
                    chunk_index + 1,
                    ", ".join(sorted(fresh)),
                )
            locks.update(fresh)

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
    """Half-open [start, end) index ranges of cues to translate per request.

    Even chunks of :data:`MAX_CUES_PER_REQUEST`, except that a final chunk smaller than
    :data:`MIN_TAIL_CUES` is merged into the one before it rather than sent alone — see
    that constant for the cue that came back in the wrong gender because it was.
    """
    if total <= MAX_CUES_PER_REQUEST:
        return [(0, total)]
    bounds = [
        (start, min(start + MAX_CUES_PER_REQUEST, total))
        for start in range(0, total, MAX_CUES_PER_REQUEST)
    ]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_TAIL_CUES:
        bounds[-2:] = [(bounds[-2][0], total)]
    return bounds


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
    "5. NEVER delete the LAST meaningful word of the cue — it is almost always the word "
    "the cue exists to deliver, and the sentence collapses without it. "
    '"מטוס מסחרי." must not become "מטוס.", "הם נחשבים קשוחים." must not become '
    '"הם נחשבים מאוד." (a modifier with nothing to modify), "שפה כנענית." must not '
    'become "שפה." Trim from the middle and the front; keep the end intact.\n'
    "6. Keep the meaning and the register. Drop redundancy, not information.\n"
    "7. PRESERVE sentence punctuation: . , ? ! — every cue must end with the SAME "
    "final mark it already has. A question stays a question.\n"
    "8. Keep proper nouns, numbers and acronyms, and keep existing typography "
    f"(Hebrew gershayim {GERSHAYIM}, geresh {GERESH}).\n"
    f'9. A cue that begins with the dialogue dash "{DIALOGUE_DASH}" must still begin '
    "with it.\n\n"
    'Return ONLY JSON: {"cues":[{"id":<int>,"t":"<shortened cue>"}]} — one entry per '
    "cue given, reusing the same ids."
)

#: Second-pass prompt, sent only for cues that are STILL too long after the first pass.
#:
#: There used to be a second direction — TOO SHORT, "you threw away content you had room
#: for, put it back". It was removed in R8 after being measured on the corpus: four cues
#: entered it across eight clips, none came back better, and one came back destroyed
#: ("איפה אתם?" -> "איפה?" -> "איפה את", truncated mid-word with the question mark gone).
#: The failure is structural, not a wording problem: a model handed a cue that already
#: fits and told it is nevertheless wrong will change something, and the only thing left
#: to change is text that was already correct. The 85% floor survives as GUIDANCE in the
#: first-pass prompt, where it stops over-cutting before it happens, and as
#: :func:`_cps_rejection`, which measures the deletion itself instead of its shadow.
_CPS_REASK_PROMPT = (
    "You are a professional broadcast subtitler. A previous condensation pass returned "
    "cues that were rejected, and you are fixing exactly those.\n\n"
    "Each line gives the ORIGINAL cue, its character limit, the ATTEMPT, and why the "
    "attempt was rejected. There are two reasons:\n"
    "  TOO LONG             — it still exceeds the limit. Cut more.\n"
    "  NOT A CONDENSATION   — it deleted something a condensation may not delete, and "
    "the reason names it. Put that back, and find the characters somewhere else: "
    "filler, a repeated word, a redundant modifier, a subject pronoun the verb already "
    "implies.\n\n"
    "HARD RULES\n"
    "1. Keep the cue in the SAME language it is written in. Do NOT translate.\n"
    "2. Never exceed the character limit given for that cue.\n"
    "3. Land BETWEEN 85% and 100% of the limit. Both edges are real failures.\n"
    "4. Work from the ORIGINAL, not from the failed attempt — the attempt may have "
    "deleted the most important word in the cue.\n"
    "5. NEVER delete the LAST meaningful word of the cue, and never leave a modifier "
    'whose noun you deleted ("הם נחשבים מאוד." is not a sentence). Trim from the middle '
    "and the front.\n"
    "6. Drop filler before content; keep proper nouns, numbers, negations, and the "
    "cue's final punctuation mark exactly as it is.\n"
    "7. If the cue genuinely cannot be shortened without losing meaning, return it "
    "unchanged. An over-long correct cue is better than a short broken one.\n\n"
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
                ceiling = min(
                    ceiling, float(out[index + 1].get("start", 0) or 0) - min_gap
                )
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


# --------------------------------------------------------------------------------------
# What a condensation is allowed to throw away
# --------------------------------------------------------------------------------------
#
# Everything below exists because a character count cannot tell the difference between
# these two answers to "shorten this by six characters":
#
#     "לצה״ל יש מוניטין קשוח. הם נחשבים קשוחים."   -> good
#     "לצה״ל יש מוניטין קשוח. הם נחשבים מאוד."     -> "they are considered very."
#
# Both fit. One is Hebrew. The pass used to pick by length and shipped the second one.

#: Marks that never belong to a word token.
_WORD_STRIP = " \t\r\n.,!?;:…\"'()[]{}«»„“”‘’-–—" + GERSHAYIM + GERESH

#: Hebrew points and cantillation (U+0591-U+05C7). Stripped before comparing tokens: a
#: subtitle font does not draw them, so עברית and עִברית are the same word on screen and
#: must be the same word to this code.
_NIQQUD = "".join(chr(c) for c in range(0x0591, 0x05C8))

#: Sentence-final marks. A cue that had one must still have one.
_TERMINAL_PUNCT = ".?!…"

#: Any trailing mark at all, terminal or continuing.
_TRAILING_PUNCT = _TERMINAL_PUNCT + ",:;"

#: Words a condensation MAY delete: discourse markers, intensifiers, copulas and bare
#: pronouns. Deliberately a CLOSED list — an unknown word counts as content, because the
#: cost of protecting a filler word (a cue a few characters over budget) is nothing next
#: to the cost of deleting a content word (a cue that says something else).
_DROPPABLE = frozenset("""
    אה אהה אמ המ הא נו ובכן טוב אוקיי אוקי הנה כאילו בעצם פשוט ממש מאוד די לגמרי
    כמובן למעשה בכלל הרי בערך כמעט בעצם אז גם רק עוד כבר הכי יותר פחות כך ככה
    היי הי אוי אויה וואו וואי אופס הופ יאללה אוף
    זה זו זאת הזה הזאת הזו אלה האלה הללו
    היה הייתה היתה היו יהיה תהיה
    אני אתה את הוא היא אנחנו אתם אתן הם הן
    אותי אותך אותו אותה אותנו אותם אותן
    לי לך לו לה לנו להם להן
    מזה בזה לזה כזה לכך בכך מכך
    עליו עליה עליהם ממנו ממנה מהם אליו אליה אליהם בו בה בהם
    ו ש כי אם אבל או אשר כדי של עם על אל מן כן נכון
    uh um er ah oh hey well okay ok like you know i mean so right yeah
    a an the is are was were that this it
    """.split())

#: Words a condensation may NEVER delete, whatever else is going on. Negations invert
#: the sentence; reflexives are the verb's object and leave a transitive verb dangling
#: ("הכינו את עצמכם" -> "הכינו", "get yourselves ready" -> "get ready").
#: ("אל" is deliberately absent: the negative imperative and the preposition "to" are
#: spelled the same, and the preposition is the common one.)
_NEVER_DROPPABLE = frozenset("""
    לא אין בלי ללא לעולם מעולם לאו אינו אינה אינם אינן
    עצמי עצמך עצמו עצמה עצמנו עצמכם עצמכן עצמם עצמן
    not no never none without
    """.split())

#: Words that cannot be the last word of a cue. A cue ending on a preposition, a
#: subordinator or the accusative את is a sentence someone cut in half.
#: ("גם" and "רק" are deliberately absent — "ואני גם." is a whole Hebrew sentence.)
_CANNOT_END = frozenset("""
    של את עם על אל מן אצל בין מול תחת לפי לגבי בגלל למרות כמו אחרי לפני
    כי אם אבל או אשר כדי כאשר ו ש הכי
    of to in on at by for with from and or but that which the a an
    """.split())


def _normalize_token(token: str) -> str:
    """One word, stripped of punctuation, typography marks and Hebrew points."""
    token = token.strip(_WORD_STRIP)
    if not token:
        return ""
    for point in _NIQQUD:
        if point in token:
            token = token.replace(point, "")
    return token.casefold()


def _tokens(text: str) -> list:
    """The words of a cue, normalised for comparison. Order preserved, blanks dropped."""
    return [t for t in (_normalize_token(w) for w in str(text or "").split()) if t]


def _same_word(a: str, b: str) -> bool:
    """Two tokens are the same word, allowing for a conjunctive vav on one of them.

    "והם" and "הם" are one word wearing two hats — Hebrew glues the conjunction onto the
    following word, so a condensation that drops an "and" changes the token without
    dropping anything. Nothing else is normalised away: this is a spelling equivalence,
    not a stemmer, and a stemmer here would quietly bless "קשוחים" -> "קשוח".
    """
    return a == b or a == "ו" + b or b == "ו" + a


def _last_content_token(tokens) -> str:
    """The final word that carries meaning — the one a cue is usually ABOUT.

    Every harmful condensation measured on the corpus took this word: "מסחרי" out of
    "מטוס מסחרי", "קשוחים" out of "נחשבים קשוחים", "כנענית" out of "שפה כנענית",
    "לפארסי" out of "אז זה הפך לפארסי", and — the case that first put a floor in this
    module — "העולמי" out of "מרכז הסחר העולמי". Legitimate trims took words from the
    middle or the front instead ("וצרחו" out of "צעקו וצרחו", "אני אומר" off the front).
    """
    for token in reversed(list(tokens)):
        if token not in _DROPPABLE:
            return token
    return ""


def _cps_rejection(original: str, candidate: str):
    """Why this condensation is unusable, or ``None`` if it may be shipped.

    A gate, not a grader: it says "this one is not a shorter version of that one", and
    the caller falls back to a candidate that is. Everything it checks is a shape that
    was measured coming out of the model, and every check is closed-list or structural —
    no stemming, no grammar, no guessing.
    """
    candidate = (candidate or "").strip()
    if not candidate:
        return "empty"

    original = (original or "").strip()
    original_tokens = _tokens(original)
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens:
        return "no words"

    # 1. A cue that begins with the dialogue dash still marks a change of speaker.
    if original.startswith(DIALOGUE_DASH) and not candidate.startswith(DIALOGUE_DASH):
        return "lost the dialogue dash"

    # 2. Punctuation. "איפה אתם?" came back as "איפה את" — no mark, and the question
    #    stopped being one.
    original_end = original[-1:] if original else ""
    candidate_end = candidate[-1:]
    if original_end in _TERMINAL_PUNCT and candidate_end != original_end:
        if not (original_end == "…" and candidate_end in _TERMINAL_PUNCT):
            return f"ends {candidate_end!r} where the original ended {original_end!r}"
    if original_end in _TRAILING_PUNCT and candidate_end not in _TRAILING_PUNCT:
        return "lost its closing punctuation"

    # 3. The word the cue is about — see :func:`_last_content_token`.
    tail = _last_content_token(original_tokens)
    if tail and not any(_same_word(tail, t) for t in candidate_tokens):
        return f"dropped the final content word {tail!r}"

    # 4. Negations, reflexives and numbers: deleting one of these is never a
    #    condensation. A dropped negation says the opposite of the source; a dropped
    #    figure is a broadcast correction.
    for token in original_tokens:
        if token in _NEVER_DROPPABLE or any(c.isdigit() for c in token):
            if not any(_same_word(token, t) for t in candidate_tokens):
                return f"dropped {token!r}"

    # 5. A cue may not end on a word that cannot end a sentence.
    if candidate_tokens[-1] in _CANNOT_END:
        return f"ends on {candidate_tokens[-1]!r}"

    return None


def _choose_candidate(original, candidates, limit):
    """Pick the best replacement for one cue: the longest LEGAL one that fits its budget.

    Two filters, in this order, and the order is the whole point:

    1. **Legal** — :func:`_cps_rejection` throws out candidates that are not shortened
       versions of the original at all. This runs FIRST because a length comparison
       between a good cue and a broken one is meaningless.
    2. **Longest that fits** — among the survivors, the one that threw away the least,
       which is the exact opposite of the instinct ("shortest is safest") that deleted a
       proper noun with 13 characters to spare.

    Falls back to the shortest of {legal candidates, the original} when none fits, and
    to the ORIGINAL when none is legal: the pass never blocks delivery, and an
    over-budget cue that says the right thing beats an in-budget cue that does not.

    Returns ``(text, fits, rejections)`` — ``rejections`` is ``[(candidate, why)]`` so
    the caller can log what it threw away and why.
    """
    rejections = []
    legal = []
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        why = _cps_rejection(original, candidate)
        if why:
            rejections.append((candidate, why))
        else:
            legal.append(candidate)

    fitting = [c for c in legal if len(c) <= limit]
    if fitting:
        return max(fitting, key=len), True, rejections
    usable = legal + [original]
    best = min(usable, key=len)
    return best, len(best) <= limit, rejections


#: The marks a punctuation-only edit is measured over: everything
#: :data:`_WORD_STRIP` knows about except whitespace.
_PUNCT_MARKS = frozenset(_WORD_STRIP) - frozenset(" \t\r\n")


def _depunctuated(text: str) -> str:
    """The same text with every punctuation mark removed and whitespace collapsed."""
    return " ".join(
        "".join(ch for ch in str(text or "") if ch not in _PUNCT_MARKS).split()
    )


def _punctuation_only_edit(original: str, candidate: str) -> bool:
    """Is this "rewrite" nothing but the deletion of punctuation?

    MEASURED: a re-ask came back having deleted a comma — same words, same order, one
    character shorter. That is not a condensation; it is a cue with its clause boundary
    removed, shipped as though the model had solved something. It saves nothing a viewer
    can perceive and costs the reader the pause the sentence was built around, and this
    pipeline spends real effort upstream getting that punctuation to exist at all (see
    ``transcription_service.ASR_PUNCTUATION_PRIMER``).

    Both conditions must hold: identical once punctuation is removed, AND fewer marks
    than the original. A rewrite that ADDS or MOVES punctuation while changing the words
    is a different (and legitimate) thing and is not caught here.

    Pure.
    """
    original, candidate = str(original or "").strip(), str(candidate or "").strip()
    if not original or not candidate:
        return False
    if _depunctuated(original) != _depunctuated(candidate):
        return False
    return sum(ch in _PUNCT_MARKS for ch in candidate) < sum(
        ch in _PUNCT_MARKS for ch in original
    )


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
       sent at all, and only when their budget is at least :data:`CPS_MIN_BUDGET`. A cue
       2 characters over its budget is 0.1 seconds of reading time over, which no viewer
       has ever noticed; paying tokens to rewrite it and risking a deleted noun to save
       it is a bad trade in both directions.
    3. **Condense** — one request per batch, under a prompt with an explicit FLOOR
       (never below 85% of the limit) as well as a ceiling.
    4. **Validate** — :func:`_cps_rejection` throws out a reply that is not a shortened
       version of what it was given: one that dropped the cue's last content word, its
       negation, its reflexive object, its final punctuation or its dialogue dash. A
       refused reply is not re-asked, it is simply not used — the original text stands.
    5. **Re-ask, once** — for every cue that is still over the limit *by more than the
       same margin that let it into the pass*. The re-ask is given the original text,
       the limit and the failed attempt, and the winner is chosen by
       :func:`_choose_candidate` — the longest LEGAL candidate that fits. Two guards
       stand around that choice, both bought with measured releases: a candidate whose
       only change is deleting punctuation is thrown out before the choice
       (:func:`_punctuation_only_edit`), and if nothing FITS the ORIGINAL is kept rather
       than the shortest near-miss — a rewrite that leaves the cue over budget has
       improved nothing and can only have cost meaning.

    Why the floor is guidance and not a threshold: the first version of this pass had
    only a ceiling, and the model duly obeyed it by deleting whatever it liked — it cut
    a cue from 53 to 31 characters when 51 were allowed, deleted the subject noun of
    "מרכז הסחר העולמי" with 13 characters of headroom, and returned four portrait cues
    9-16 characters under their limits with whole clauses missing. The floor stopped
    that, and then a floor VIOLATION started triggering a "restore what you lost"
    re-ask, which on the same corpus never once restored anything and did produce
    "איפה את" out of "איפה אתם?". Step 4 replaces the proxy with the measurement.

    This pass improves readability but never blocks delivery: a failed batch leaves that
    batch's cues untouched, and a cue that cannot be fixed keeps its ORIGINAL text with a
    warning. When readability and meaning genuinely conflict, meaning wins and the cue
    ships over budget — an over-fast correct subtitle is a readability cost, an amputated
    one is a translation error.

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
    too_tight = 0
    for idx, cue in enumerate(out):
        text = (cue.get("translated") or "").strip()
        if not text:
            continue
        limit = _cps_budget(_duration(cue), max_cps, max_chars_per_cue)
        budgets[idx + 1] = limit
        originals[idx + 1] = text
        if len(text) <= limit * CPS_TRIGGER_MARGIN:
            continue
        if limit < CPS_MIN_BUDGET:
            too_tight += 1
            continue
        violators.append(idx + 1)

    if too_tight:
        logger.info(
            "translation_v2.enforce_cps: left %d over-budget cue(s) alone — their "
            "budget is under %d characters, which is not a condensation job "
            "(see CPS_MIN_BUDGET)",
            too_tight,
            CPS_MIN_BUDGET,
        )

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
        violators[i : i + MAX_CUES_PER_CPS_REQUEST]
        for i in range(0, len(violators), MAX_CUES_PER_CPS_REQUEST)
    ]
    still_over = 0
    over_condensed = 0
    refused = 0
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

        # Who still needs work, and why. Two reasons, and neither of them is "you came
        # back shorter than a percentage of the limit" — see :data:`_CPS_REASK_PROMPT`.
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
            why = _cps_rejection(originals[cue_id], candidate)
            if why:
                refused += 1
                logger.warning(
                    "translation_v2.enforce_cps: refused the condensation of cue %d "
                    "(%s) — %r",
                    cue_id,
                    why,
                    candidate,
                )
                reask[cue_id] = (candidate, f"NOT A CONDENSATION — it {why}")
                continue
            if len(candidate) > limit * CPS_TRIGGER_MARGIN:
                reask[cue_id] = (candidate, f"TOO LONG — {len(candidate)} chars")
                continue
            out[cue_id - 1]["translated"] = candidate
            if len(candidate) < _floor_chars(limit):
                over_condensed += 1
            elif len(candidate) > limit:
                still_over += 1

        if not reask:
            continue

        logger.info(
            "translation_v2.enforce_cps: re-asking %d cue(s) from batch %d",
            len(reask),
            batch_index + 1,
        )
        reask_lines = [
            f"{cue_id}. (max {budgets[cue_id]} chars, never below "
            f"{_floor_chars(budgets[cue_id])}) [{why}] "
            f"ORIGINAL: {originals[cue_id]} || ATTEMPT: {attempt}"
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
            original = originals[cue_id]

            # GUARD 1, before the chooser sees them: a candidate whose only change is
            # deleting punctuation is not a candidate — see :func:`_punctuation_only_edit`.
            candidates = []
            for candidate in (attempt, (second.get(cue_id) or "").strip()):
                if not candidate:
                    continue
                if _punctuation_only_edit(original, candidate):
                    refused += 1
                    logger.warning(
                        "translation_v2.enforce_cps: refused a candidate for cue %d "
                        "(deletes punctuation and changes nothing else) — %r",
                        cue_id,
                        candidate,
                    )
                    continue
                candidates.append(candidate)

            best, fits, rejections = _choose_candidate(original, candidates, limit)
            for candidate, why in rejections:
                if candidate == attempt:
                    continue  # already counted and logged when the batch produced it
                refused += 1
                logger.warning(
                    "translation_v2.enforce_cps: refused a candidate for cue %d (%s) "
                    "— %r",
                    cue_id,
                    why,
                    candidate,
                )

            # GUARD 2: the re-ask is the LAST chance, so a rewrite that is still over the
            # limit has bought nothing — it did not make the cue readable, and every
            # rewrite carries the risk of having quietly dropped a word (measured: one
            # was released over the limit having done exactly that). Over budget for
            # over budget, the text nobody rewrote is the safer one.
            if not fits:
                still_over += 1
                out[cue_id - 1]["translated"] = original
                logger.warning(
                    "translation_v2.enforce_cps: cue %d is STILL over budget after a "
                    "re-ask (best candidate %d chars, budget %d) — keeping the ORIGINAL "
                    "text, since a rewrite that does not fit only risks meaning",
                    cue_id,
                    len(best),
                    limit,
                )
                continue

            out[cue_id - 1]["translated"] = best
            if len(best) < _floor_chars(limit):
                over_condensed += 1
                logger.warning(
                    "translation_v2.enforce_cps: cue %d came back over-condensed even "
                    "after a re-ask (%d chars for a %d-char budget) — shipping it, but "
                    "content was probably lost",
                    cue_id,
                    len(best),
                    limit,
                )

    _report(
        progress_callback, len(violators), len(violators), "Reading-speed pass done"
    )

    logger.info(
        "translation_v2.enforce_cps: done | %d condensed, %d still over budget, "
        "%d over-condensed, %d refused as not-a-condensation | "
        "tokens in=%d out=%d cost=$%.4f",
        len(violators),
        still_over,
        over_condensed,
        refused,
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


def cps_report(
    cues, *, max_cps=DEFAULT_MAX_CPS, max_chars_per_cue=DEFAULT_MAX_CHARS_PER_CUE
):
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
                "ok": len(text) <= max_chars_per_cue
                and (duration <= 0 or cps <= max_cps),
            }
        )
    return report
