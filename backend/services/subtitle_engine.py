"""
Subtitle spotting engine — turns word-level timestamps into broadcast-quality cues
and renders them as an ASS (Advanced SubStation Alpha) file.

Why this exists
---------------
Whisper segments are *speech* segments, not *subtitle* cues. Used directly they
produce the defects measured against a professional reference on the same clip:

  * 0.2-0.3s flashes holding a single word ("things.", "Yeah."),
  * sentences cut mid-clause ("It could complicate" | "things."),
  * a question and its answer merged into one cue (no speaker turn),
  * lines far longer than a viewer can read at 24 fps.

This module re-spots the transcript from *words*: words -> sentences -> cues with
duration/length constraints, then a lead-out pass that hands each cue the dead air
in front of the next one (the single biggest CPS win, and standard editorial
practice). Constraints follow the Netflix Hebrew Timed Text Style Guide:
42 characters per line, at most 2 lines, ~17 characters per second for adults.

Rendering notes (both re-verified by rendering in this project's own container,
FFmpeg 7.1.5 / libass 0.17.3 — see ``tests/integration/test_bidi_render.py``)
--------------------------------------------------------------------------
* FFmpeg's ``subtitles`` filter has **no** ``shaping`` option ("Option not
  found", hard failure). The ``ass`` filter does. That is why this engine emits
  ``.ass`` — it must be rendered with ``ass=...:shaping=complex``.
* Hebrew bidi: **libass hard-defaults the paragraph direction to LTR.** It does
  not auto-detect it from the first strong character — rendering ``אבג DEF``
  with no controls puts the Hebrew on the *left*, which is the LTR answer, not
  the first-strong-character one. That single missing fact is the whole bidi
  problem here: with no controls every mixed Hebrew line comes out with its
  WORD ORDER REVERSED. So the line must declare its own direction with
  U+2067 (RLI) ... U+2069 (PDI), and each maximal Latin/digit stretch is kept
  intact with U+2066 (LRI) ... U+2069 (PDI) — see :func:`bidi_isolate` for why
  the runs must be *maximal* and why the line-level RLI is not optional.
  U+202E (RLO) — used by the legacy path in ``utils/rtl_utils.py`` — is
  deliberately never emitted: it is an override, it defeats the bidi algorithm
  instead of cooperating with it.

Every capability is an independent keyword toggle so the API/UI layer can expose
them one by one; nothing here reads config or touches the filesystem.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# --- Netflix Hebrew Timed Text limits (defaults, all overridable) -------------
MAX_LINE_CHARS = 42
MAX_LINES = 2
MIN_CUE_DUR = 1.2
MAX_CUE_DUR = 6.0
MIN_CUE_GAP = 0.08

#: Speech-pause length (seconds) treated as a sentence boundary when the ASR
#: transcript arrives without punctuation (large-v3 sometimes does this).
PAUSE_SPLIT_GAP = 0.35

#: Longest silence (seconds) a short fragment may be merged across. Two utterances
#: separated by more than this are not the same breath, whatever their length: the
#: merge step exists to repair sub-second splinters, not to span dead air.
MERGE_MAX_GAP = 1.0

#: Shortest cue that can actually be seen (~1 frame at 25 fps). A cue that cannot
#: be given this much time without overlapping its neighbour is not shortened to
#: zero \u2014 it is folded into that neighbour. See :func:`words_to_cues`.
MIN_VISIBLE_DUR = 0.04

# --- Unicode bidi isolates (never RLO/U+202E) --------------------------------
# Spelled as escapes on purpose: these characters are invisible, so a literal
# would be impossible to review and trivial to delete by accident.
RLI = "\u2067"  # RIGHT-TO-LEFT ISOLATE
LRI = "\u2066"  # LEFT-TO-RIGHT ISOLATE
PDI = "\u2069"  # POP DIRECTIONAL ISOLATE

GERSHAYIM = "״"  # ״ HEBREW PUNCTUATION GERSHAYIM

# Hebrew letters only: alef-tav plus the yod/vav ligatures. Deliberately not
# the whole U+0590-U+05FF block, which also holds niqqud and punctuation.
_HEBREW_LETTERS = "\u05d0-\u05ea\u05ef-\u05f2"

# A word ends a sentence when it ends with . ! ? possibly followed by a closer.
_TERMINAL_RE = re.compile(r"[.!?][\"'”’)\]]*$")
# Sentence boundary *inside* an assembled cue text.
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*(?:\s|$)")
# ASCII '"' between two Hebrew letters is an acronym mark, not a quote.
_GERSHAYIM_RE = re.compile(f'(?<=[{_HEBREW_LETTERS}])"(?=[{_HEBREW_LETTERS}])')

# Unicode bidirectional categories, used by the bidi pass below instead of a
# hand-maintained character class (which is exactly what got this wrong before:
# a literal class that forgot "%" renders "50%" as "%50").
_BIDI_STRONG_RTL = frozenset({"R", "AL"})          # Hebrew, Arabic, Thaana, ...
_BIDI_LTR_ANCHOR = frozenset({"L", "EN", "AN"})    # Latin letters and digits
_BIDI_NUMERIC_TERMINATOR = "ET"                    # $ € ₪ % ‰ ° ...

# Break-preference bonuses used when choosing where to split two lines.
_BONUS_SENTENCE_BREAK = 30
_BONUS_CLAUSE_BREAK = 12
_PENALTY_TOP_HEAVY = 3

# Horizontal margins in ASS units. WrapStyle 2 disables auto-wrap (we wrap
# ourselves), so these only guard against edge-hugging on odd aspect ratios.
_MARGIN_H = 60

# Style name. Deliberately NOT "Default": libass appends a built-in "Default"
# style to every track and resolves style names by scanning the list backwards,
# so a style of ours called "Default" could lose to libass's Arial fallback.
STYLE_NAME = "He"


# =============================================================================
# words -> cues
# =============================================================================
def _normalize_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy/clean the input word list: strip padding, coerce times, drop empties.

    Accepts the compact ``{"s","e","w"}`` shape and Whisper's own
    ``{"start","end","word"}`` shape.

    Negative timestamps are clamped to 0 *here*, before anything downstream can
    reason about them: a negative start would make every later duration/gap
    calculation meaningless and can only come from broken ASR output.
    """
    cleaned: list[dict[str, Any]] = []
    for raw in words or []:
        text = str(raw.get("w", raw.get("word", "")) or "").strip()
        if not text:
            continue
        try:
            start = float(raw.get("s", raw.get("start", 0.0)))
            end = float(raw.get("e", raw.get("end", start)))
        except (TypeError, ValueError):
            logger.debug("subtitle_engine: skipping unparsable word %r", raw)
            continue
        if start != start or end != end:  # NaN: every comparison below would be False
            logger.debug("subtitle_engine: skipping word with NaN timing %r", raw)
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        cleaned.append({"s": start, "e": max(end, start), "w": text})
    # Stable sort: well-formed input is already monotonic, garbage gets repaired.
    cleaned.sort(key=lambda w: w["s"])
    return cleaned


def _text_of(group: list[dict[str, Any]]) -> str:
    """Assemble a cue's text from its words."""
    return " ".join(w["w"] for w in group)


def _span_of(group: list[dict[str, Any]]) -> float:
    """Wall-clock span covered by a group of words."""
    return group[-1]["e"] - group[0]["s"]


def _sentence_count(text: str) -> int:
    """How many sentences the text already holds."""
    return len(_SENTENCE_END_RE.findall(text))


def _split_long_group(
    group: list[dict[str, Any]], max_chars: int
) -> list[list[dict[str, Any]]]:
    """Recursively split an over-long group at the comma nearest its middle.

    Falls back to the middle word when the group holds no comma, and gives up
    (returns the group as-is) when it cannot be split any further.
    """
    if len(group) < 2 or len(_text_of(group)) <= max_chars:
        return [group]

    commas = [i for i, w in enumerate(group[:-1]) if w["w"].endswith(",")]
    middle = len(group) // 2
    cut = min(commas, key=lambda i: abs(i - middle)) + 1 if commas else middle
    cut = max(1, min(cut, len(group) - 1))  # never produce an empty half
    return _split_long_group(group[:cut], max_chars) + _split_long_group(
        group[cut:], max_chars
    )


def words_to_cues(
    words: list[dict[str, Any]],
    *,
    max_line: int = MAX_LINE_CHARS,
    max_lines: int = MAX_LINES,
    min_dur: float = MIN_CUE_DUR,
    max_dur: float = MAX_CUE_DUR,
    min_gap: float = MIN_CUE_GAP,
) -> list[dict[str, Any]]:
    """Re-spot word timestamps into readable subtitle cues.

    Pipeline:
      1. **Sentences** — group words, closing a group on a terminal ``. ! ?``.
      2. **Split** — a sentence longer than ``max_line * max_lines`` characters is
         split at the comma nearest its middle, recursively.
      3. **Merge** — a group shorter than ``min_dur`` is folded into the previous
         cue only if the result still fits ``max_line * max_lines`` characters,
         spans at most ``max_dur``, the silence between the two is at most
         :data:`MERGE_MAX_GAP`, the previous cue does not end with ``?`` and the
         previous cue holds fewer than 2 sentences. The question rule keeps a
         question and its answer apart: it is the cheapest available proxy for a
         speaker turn, and merging across one is the defect this engine exists to
         remove. The gap rule stops "Hello." at 0.0-0.5s being glued to "Bye." at
         5.5-6.0s just because the second one is short.
      4. **Lead-out** — each cue's end grows into the dead air ahead of it, up to
         ``next_start - min_gap`` and never past ``start + max_dur``; when cues
         collide the end shrinks instead, but not below ``start + min_dur``.
         This is what buys the CPS headroom (~17 cps target).

    Args:
        words: word timestamps, ``{"s": float, "e": float, "w": str}``; ``w`` may
            carry leading/trailing whitespace.
        max_line: characters per line.
        max_lines: lines per cue (``max_line * max_lines`` is the cue budget).
        min_dur: shortest readable cue, seconds.
        max_dur: longest cue, seconds.
        min_gap: gap left between consecutive cues, seconds.

    Returns:
        ``[{"start": float, "end": float, "text": str}, ...]`` sorted by start,
        non-overlapping, and every cue with a strictly positive duration. Word
        order and wording are preserved exactly.
    """
    clean = _normalize_words(words)
    if not clean:
        return []

    max_chars = max(1, int(max_line) * int(max_lines))

    # Fallback for unpunctuated ASR output (the known large-v3 failure mode):
    # with NO terminal punctuation at all there are no sentences to find, so
    # speech pauses become the sentence boundaries instead.
    #
    # The trigger is `terminals == 0`, not a ratio. A ratio silently changes the
    # spotting of ordinary transcripts — one long unpunctuated stretch inside an
    # otherwise well-punctuated file would drag the whole file onto the pause
    # heuristic. Zero terminals is the only state in which the punctuation-based
    # splitter is guaranteed to produce nothing at all, and it is exactly the
    # state large-v3 lands in. A transcript with even one "." keeps the normal
    # path untouched.
    terminals = sum(1 for w in clean if _TERMINAL_RE.search(w["w"]))
    sparse_punctuation = len(clean) >= 8 and terminals == 0
    if sparse_punctuation:
        logger.info(
            "spotting: no terminal punctuation in %d words — "
            "pause-based sentence fallback engaged",
            len(clean),
        )

    # 1. sentences
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for i, word in enumerate(clean):
        current.append(word)
        boundary = bool(_TERMINAL_RE.search(word["w"]))
        if not boundary and sparse_punctuation and i + 1 < len(clean):
            boundary = clean[i + 1]["s"] - word["e"] >= PAUSE_SPLIT_GAP
        if boundary:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    # 2. split over-long sentences
    groups = [part for s in sentences for part in _split_long_group(s, max_chars)]

    # 3. merge short fragments into the previous cue when it is safe to do so
    merged: list[list[dict[str, Any]]] = []
    for group in groups:
        if merged and _span_of(group) < min_dur:
            previous = merged[-1]
            prev_text = _text_of(previous)
            fits = len(prev_text) + 1 + len(_text_of(group)) <= max_chars
            span_ok = (group[-1]["e"] - previous[0]["s"]) <= max_dur
            gap_ok = (group[0]["s"] - previous[-1]["e"]) <= MERGE_MAX_GAP
            ends_question = prev_text.rstrip().endswith("?")
            room_for_sentence = _sentence_count(prev_text) < 2
            if fits and span_ok and gap_ok and not ends_question and room_for_sentence:
                merged[-1] = previous + group
                continue
        merged.append(group)

    # 3b. degenerate-input repair. Two groups that begin at (effectively) the same
    # instant cannot both be shown, and step 4's anti-overlap clamp would resolve
    # that by pushing the first cue's end back onto its own start — a zero-length
    # event. Fold them into one cue instead: deterministic, and no text is lost.
    # Only broken word timestamps (all-identical, or an ASR hiccup) get here.
    collapsed: list[list[dict[str, Any]]] = []
    for group in merged:
        if collapsed and (group[0]["s"] - collapsed[-1][0]["s"]) < MIN_VISIBLE_DUR:
            logger.warning(
                "spotting: cues at the same instant (%.3fs) — folding %r into the "
                "previous cue; the word timestamps are degenerate",
                group[0]["s"],
                _text_of(group)[:40],
            )
            collapsed[-1] = collapsed[-1] + group
            continue
        collapsed.append(group)

    cues = [
        {"start": g[0]["s"], "end": max(g[-1]["e"], g[0]["s"]), "text": _text_of(g)}
        for g in collapsed
    ]

    # 4. lead-out / anti-overlap
    for i, cue in enumerate(cues):
        next_start = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if next_start is None:
            limit = cue["end"] + min_dur  # last cue: a plain lead-out
        else:
            limit = next_start - min_gap
        if limit < cue["end"]:
            cue["end"] = max(limit, cue["start"] + min_dur)  # shrink, but stay readable
        else:
            cue["end"] = min(limit, cue["start"] + max_dur)  # extend into dead air
        if next_start is not None and cue["end"] > next_start:
            cue["end"] = next_start  # hard guarantee: never overlap
        # Step 3b guarantees next_start - start >= MIN_VISIBLE_DUR, so the clamp
        # above can never land on or before the start. Kept as a belt-and-braces
        # assertion of the contract rather than as live arithmetic.
        cue["end"] = max(cue["end"], cue["start"] + MIN_VISIBLE_DUR)
        cue["start"] = round(cue["start"], 3)
        cue["end"] = round(cue["end"], 3)

    logger.debug(
        "subtitle_engine: %d words -> %d sentences -> %d cues",
        len(clean),
        len(sentences),
        len(cues),
    )
    return cues


# =============================================================================
# line wrapping
# =============================================================================
def _hard_wrap(text: str, max_line: int) -> list[str]:
    """Last-resort wrap that is *guaranteed* to emit no line longer than ``max_line``.

    Breaks at the last space inside the budget when there is one, and mid-token when
    there is not (a 100-character URL has no other option). May return more than two
    lines: an over-long line is drawn straight off the edge of the frame by libass —
    ``WrapStyle: 2`` disables its own wrapping — so a third line is strictly less bad
    than losing text off-screen. ``translation_v2`` caps cues at 84 chars = 2 x 42, so
    in practice only pathological input reaches this.
    """
    lines: list[str] = []
    rest = text
    while len(rest) > max_line:
        cut = rest.rfind(" ", 0, max_line + 1)
        if cut <= 0:
            cut = max_line
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        lines.append(rest)
    return lines or [""]


def wrap_two_lines(text: str, max_line: int = MAX_LINE_CHARS) -> list[str]:
    """Wrap a cue into one line, or the best two-line split.

    A single line is always preferred. Otherwise every word boundary that leaves
    both halves within ``max_line`` is scored and the cheapest wins:

      * length imbalance (lower is better),
      * a small penalty when the top line is the longer one — subtitles read
        better bottom-heavy,
      * a strong bonus for breaking after ``.``/``?``/``!``,
      * a smaller bonus for breaking after a comma.

    When no word boundary can satisfy ``max_line`` — one over-long token, or simply
    more text than two lines can hold — it falls back to :func:`_hard_wrap`, which
    never returns an over-long line.

    Postcondition (relied on by :func:`build_ass`): ``max(len(l) for l in result)
    <= max_line`` for every input.
    """
    text = " ".join((text or "").split())
    if len(text) <= max_line:
        return [text]

    words = text.split(" ")
    best: tuple[float, list[str]] | None = None
    for i in range(1, len(words)):
        top = " ".join(words[:i])
        bottom = " ".join(words[i:])
        if len(top) > max_line or len(bottom) > max_line:
            continue
        score = float(abs(len(top) - len(bottom)))
        if len(top) > len(bottom):
            score += _PENALTY_TOP_HEAVY
        stripped = top.rstrip()
        if stripped.endswith((".", "?", "!")):
            score -= _BONUS_SENTENCE_BREAK
        elif stripped.endswith(","):
            score -= _BONUS_CLAUSE_BREAK
        if best is None or score < best[0]:
            best = (score, [top, bottom])

    if best is not None:
        return best[1]

    lines = _hard_wrap(text, max_line)
    logger.warning(
        "subtitle_engine: no two-line split fits %d chars for a %d-char cue — "
        "hard-wrapped into %d lines",
        max_line,
        len(text),
        len(lines),
    )
    return lines


# =============================================================================
# post-translation reflow
# =============================================================================
#: Single-letter Hebrew prefixes/conjunctions. They are *bound* morphemes — they
#: are written attached to the next word (ו + הלך = והלך), so one standing alone at
#: the end of a cue is always a cue boundary that fell in the wrong place.
DANGLING_PREFIXES = frozenset("ובלמשכה")

#: Longest silence (seconds) a dangling connector may be moved across. Past this the
#: two cues are not one phrase and moving the letter would put it under the wrong
#: moment of speech.
REFLOW_MAX_GAP = 1.5

#: A cue that would grow past this many characters is left alone: 2 lines x 42 is the
#: Netflix budget the rest of this module is built around.
REFLOW_MAX_CHARS = MAX_LINE_CHARS * MAX_LINES

_DANGLING_RE = re.compile(f"(?:^|\\s)([{''.join(DANGLING_PREFIXES)}])-?$")


def reflow_dangling_connectors(
    cues: list[dict[str, Any]],
    *,
    key: str = "translated_text",
    max_gap: float = REFLOW_MAX_GAP,
    max_chars: int = REFLOW_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Move a stranded single-letter Hebrew connector onto the next cue.

    Translation is done per cue, so a cue can end on a bound prefix whose word lives
    in the next cue — real output from this pipeline: ``"...לנחות ו"`` / ``"...שמכירה
    ב"``. On screen that is a lone letter hanging off the end of a line, and the word
    it belongs to appears a second later without it.

    This is a **pure text pass**: timings are never touched, so it cannot desync a
    cue from its audio. It is deliberately conservative — at most ONE token moves,
    never across a silence longer than ``max_gap``, and never into a cue that would
    end up longer than ``max_chars``.

    Args:
        cues: pipeline cue dicts. Not mutated; copies are returned.
        key: which field to reflow (``translated_text`` for the v2 path).

    Returns:
        A new list of cue copies, same length and same timings.
    """
    out = [dict(cue) for cue in (cues or [])]
    moved = 0
    for index in range(len(out) - 1):
        current, following = out[index], out[index + 1]
        text = str(current.get(key) or "").rstrip()
        next_text = str(following.get(key) or "").lstrip()
        if not text or not next_text:
            continue

        match = _DANGLING_RE.search(text)
        if not match:
            continue

        try:
            gap = float(following.get("start", 0)) - float(current.get("end", 0))
        except (TypeError, ValueError):
            continue
        if gap > max_gap:
            continue

        token = text[match.start(1):]  # the letter plus an optional trailing "-"
        remainder = text[: match.start(1)].rstrip()
        if not remainder:
            continue  # the whole cue is the connector; moving it would empty the cue

        # Hebrew prefixes are written attached ("ו" + "המטוס" = "והמטוס"), except
        # before a Latin word or a numeral, where the convention is a maqaf:
        # "ב-2026", "ה-iPhone".
        if not token.endswith("-") and unicodedata.bidirectional(next_text[0]) in (
            "L",
            "EN",
            "AN",
        ):
            token += "-"
        joined = token + next_text
        if len(joined) > max_chars:
            continue

        current[key] = remainder
        following[key] = joined
        moved += 1

    if moved:
        logger.info("reflow: moved %d dangling connector(s) to the next cue", moved)
    return out


# =============================================================================
# Hebrew typography + bidi
# =============================================================================
def gershayim(text: str) -> str:
    """Replace an ASCII ``"`` used as a Hebrew acronym mark with U+05F4 (״).

    Only a quote sitting *between two Hebrew letters* is converted, so English
    quotations (``he said "hi"``) are left untouched.
    """
    if not text:
        return text
    return _GERSHAYIM_RE.sub(GERSHAYIM, text)


def _ltr_runs(line: str) -> list[tuple[int, int]]:
    """Maximal ``[start, end]`` index pairs of stretches to keep left-to-right.

    A run is grown between *anchors* — characters the bidi algorithm calls strongly
    left-to-right or numeric (``L``/``EN``/``AN``: Latin letters, digits) — and is
    allowed to swallow anything between two anchors that is **not** strongly
    right-to-left. That is what keeps ``Microsoft Azure``, ``COVID-19``, ``3.5``,
    ``AT&T``, ``Boeing 737`` and ``2020 COVID-19`` intact as single left-to-right
    units. Adjacent numeric terminators (``ET``: ``$ € ₪ % ‰ °``) are absorbed at
    both edges so ``50%`` and ``$25`` stay glued to their number.

    The membership test is Unicode's own ``bidirectional`` category, deliberately
    not a hand-written character class: the class that preceded it forgot ``%``,
    ``$`` and ``&``, and rendered ``50%`` as ``%50``.
    """
    cats = [unicodedata.bidirectional(ch) for ch in line]
    anchors = [i for i, cat in enumerate(cats) if cat in _BIDI_LTR_ANCHOR]
    if not anchors:
        return []

    runs: list[list[int]] = [[anchors[0], anchors[0]]]
    for index in anchors[1:]:
        between = cats[runs[-1][1] + 1: index]
        if any(cat in _BIDI_STRONG_RTL for cat in between):
            runs.append([index, index])  # an RTL word closes the run
        else:
            runs[-1][1] = index

    for run in runs:
        while run[0] > 0 and cats[run[0] - 1] == _BIDI_NUMERIC_TERMINATOR:
            run[0] -= 1
        while run[1] + 1 < len(cats) and cats[run[1] + 1] == _BIDI_NUMERIC_TERMINATOR:
            run[1] += 1
    return [(start, end) for start, end in runs]


def bidi_isolate(line: str) -> str:
    """Prepare one rendered line for libass's bidi pass.

    Two facts, both established by rendering in this project's container and pinned
    by ``tests/integration/test_bidi_render.py``:

    1. **libass hard-defaults the paragraph direction to LTR** — it does not infer
       it from the first strong character. So an untouched Hebrew line comes out
       with its word order reversed. ``RLI ... PDI`` around the whole line is what
       tells libass the paragraph is right-to-left, and it is not optional: without
       it ``ב-ICC ... ב-2026`` renders reversed.
    2. **Latin/numeric stretches must be isolated as MAXIMAL runs, not per word.**
       Isolating each word separately makes N sibling isolates, which the bidi
       algorithm then lays out right-to-left *relative to each other* — the exact
       source of ``Microsoft Azure`` -> ``Azure Microsoft``, ``3.5`` -> ``5.3`` and
       ``COVID-19`` -> ``19-COVID``.

    A line with no strongly right-to-left character is returned untouched: libass's
    native LTR base is already correct for it, and wrapping it in RLI would kick its
    sentence-final period over to the left-hand edge.

    Never emits RLO (U+202E) — an override would fight the bidi algorithm rather
    than inform it.
    """
    if not line:
        return line
    if not any(unicodedata.bidirectional(ch) in _BIDI_STRONG_RTL for ch in line):
        return line

    pieces: list[str] = []
    cursor = 0
    for start, end in _ltr_runs(line):
        pieces.append(line[cursor:start])
        pieces.append(f"{LRI}{line[start:end + 1]}{PDI}")
        cursor = end + 1
    pieces.append(line[cursor:])
    return f"{RLI}{''.join(pieces)}{PDI}"


# =============================================================================
# ASS output
# =============================================================================
def _ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp: ``H:MM:SS.CC`` (centiseconds)."""
    centis = max(0, int(round(float(seconds) * 100)))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_safe(text: str) -> str:
    """Neutralize characters that would be parsed as ASS markup."""
    text = " ".join((text or "").split())  # also flattens stray newlines
    return text.replace("{", "(").replace("}", ")")


def build_ass(
    cues: list[dict[str, Any]],
    *,
    video_w: int,
    video_h: int,
    font_frac: float = 0.061,
    margin_v_frac: float = 0.12,
    rtl: bool = True,
) -> str:
    """Render cues as an ASS v4+ subtitle script.

    The style is the render-tested one: Noto Sans Hebrew, bold, white on a
    semi-transparent opaque box (``BorderStyle=4``, no drop shadow), bottom
    centre. Sizes are fractions of the video height so the result looks identical
    at 720p and 4K. ``PlayResX/Y`` match the video, so ASS units are pixels.

    Args:
        cues: ``[{"start","end","text"}, ...]`` as produced by :func:`words_to_cues`.
        video_w: video width in pixels.
        video_h: video height in pixels.
        font_frac: font size as a fraction of ``video_h``.
        margin_v_frac: bottom margin as a fraction of ``video_h``.
        rtl: apply :func:`bidi_isolate` to every line (Hebrew/Arabic targets).

    Returns:
        The complete ``.ass`` file content. Render it with FFmpeg's ``ass``
        filter (``ass=file.ass:shaping=complex``) — the ``subtitles`` filter has
        no ``shaping`` option and will hard-fail.
    """
    font_size = max(1, round(video_h * font_frac))
    margin_v = max(0, round(video_h * margin_v_frac))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {int(video_w)}\n"
        f"PlayResY: {int(video_h)}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {STYLE_NAME},Noto Sans Hebrew,{font_size},"
        "&H00FFFFFF,&H00FFFFFF,&H00000000,&H14000000,"
        "1,0,0,0,100,100,0,0,4,3,0,2,"
        f"{_MARGIN_H},{_MARGIN_H},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text\n"
    )

    events: list[str] = []
    for cue in cues or []:
        lines = wrap_two_lines(gershayim(_ass_safe(cue.get("text", ""))))
        if rtl:
            lines = [bidi_isolate(line) for line in lines]
        body = "\\N".join(lines)
        events.append(
            f"Dialogue: 0,{_ass_timestamp(cue['start'])},"
            f"{_ass_timestamp(cue['end'])},{STYLE_NAME},,0,0,0,,{body}"
        )

    logger.debug(
        "subtitle_engine: built ASS with %d events at %dx%d (font %d, margin_v %d)",
        len(events),
        video_w,
        video_h,
        font_size,
        margin_v,
    )
    return header + "\n".join(events) + "\n"
