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

Rendering notes (both empirically verified in this project's own container)
--------------------------------------------------------------------------
* FFmpeg's ``subtitles`` filter has **no** ``shaping`` option ("Option not
  found", hard failure). The ``ass`` filter does. That is why this engine emits
  ``.ass`` — it must be rendered with ``ass=...:shaping=complex``.
* Hebrew bidi: dropping every direction control **breaks** rendering (libass lays
  the runs out LTR by default). The clean, validated treatment is Unicode
  *isolates*: U+2067 (RLI) ... U+2069 (PDI) around the whole line and
  U+2066 (LRI) ... U+2069 (PDI) around each Latin/digit run. U+202E (RLO) — used
  by the legacy path in ``utils/rtl_utils.py`` — is deliberately never emitted:
  it is an override, it defeats the bidi algorithm instead of cooperating with it.

Every capability is an independent keyword toggle so the API/UI layer can expose
them one by one; nothing here reads config or touches the filesystem.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# --- Netflix Hebrew Timed Text limits (defaults, all overridable) -------------
MAX_LINE_CHARS = 42
MAX_LINES = 2
MIN_CUE_DUR = 1.2
MAX_CUE_DUR = 6.0
MIN_CUE_GAP = 0.08

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
# Latin words and digit groups get their own isolate so they read left-to-right.
_LTR_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
# ASCII '"' between two Hebrew letters is an acronym mark, not a quote.
_GERSHAYIM_RE = re.compile(f'(?<=[{_HEBREW_LETTERS}])"(?=[{_HEBREW_LETTERS}])')

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
         spans at most ``max_dur``, the previous cue does not end with ``?`` and
         the previous cue holds fewer than 2 sentences. The question rule keeps a
         question and its answer apart: it is the cheapest available proxy for a
         speaker turn, and merging across one is the defect this engine exists to
         remove.
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
        non-overlapping. Word order and wording are preserved exactly.
    """
    clean = _normalize_words(words)
    if not clean:
        return []

    max_chars = max(1, int(max_line) * int(max_lines))

    # 1. sentences
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in clean:
        current.append(word)
        if _TERMINAL_RE.search(word["w"]):
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
            ends_question = prev_text.rstrip().endswith("?")
            room_for_sentence = _sentence_count(prev_text) < 2
            if fits and span_ok and not ends_question and room_for_sentence:
                merged[-1] = previous + group
                continue
        merged.append(group)

    cues = [
        {"start": g[0]["s"], "end": g[-1]["e"], "text": _text_of(g)} for g in merged
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
def wrap_two_lines(text: str, max_line: int = MAX_LINE_CHARS) -> list[str]:
    """Wrap a cue into one line, or the best two-line split.

    A single line is always preferred. Otherwise every word boundary that leaves
    both halves within ``max_line`` is scored and the cheapest wins:

      * length imbalance (lower is better),
      * a small penalty when the top line is the longer one — subtitles read
        better bottom-heavy,
      * a strong bonus for breaking after ``.``/``?``/``!``,
      * a smaller bonus for breaking after a comma.

    Falls back to a hard character split only when no word boundary can satisfy
    ``max_line`` (a single word longer than the line budget).
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
    return [text[:max_line], text[max_line:]]


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


def bidi_isolate(line: str) -> str:
    """Wrap a line in Unicode bidi *isolates* for correct libass rendering.

    RLI ... PDI around the whole line establishes the right-to-left paragraph
    direction; LRI ... PDI around each Latin word or digit group keeps that run
    left-to-right inside it. Never emits RLO (U+202E) — an override would fight
    the bidi algorithm rather than inform it.
    """
    if not line:
        return line
    isolated = _LTR_RUN_RE.sub(lambda m: f"{LRI}{m.group(0)}{PDI}", line)
    return f"{RLI}{isolated}{PDI}"


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
