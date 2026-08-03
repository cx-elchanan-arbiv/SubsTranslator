"""
Unit tests for services/translation_v2 (broadcast-quality translation module).

The OpenAI client is mocked in full — no test here performs network I/O and no test
requires OPENAI_API_KEY. Every test asserts on either (a) the exact prompts that would
have been sent, or (b) the module's handling of a canned model response.
"""
import json
import os
import re
import sys

import pytest


def _find_backend_dir():
    """
    Locate the backend package root.

    Normally that is ``<repo>/backend`` (two levels up from this file). It is resolved by
    search rather than by a fixed relative path because docker-compose mounts
    ``./tests`` over ``/app/tests``, which shadows ``backend/tests`` inside the
    container — so this file also has to work when it is executed from elsewhere with
    ``/app`` as the working directory.
    """
    for seed in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = seed
        while True:
            if os.path.isfile(os.path.join(path, "services", "translation_v2.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise RuntimeError("could not locate the backend directory containing services/")


backend_dir = _find_backend_dir()
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.translation_v2 import (  # noqa: E402
    CONTEXT_MARKER,
    CPS_TOKENS_PER_CUE,
    DEFAULT_MAX_CHARS_PER_CUE,
    GERSHAYIM,
    LANGUAGE_NAMES,
    MAX_CUES_PER_CPS_REQUEST,
    MAX_CUES_PER_REQUEST,
    MIN_TAIL_CUES,
    OVERLAP_CUES,
    TranslationV2Error,
    build_system_prompt,
    enforce_cps,
    translate_cues,
)

APP_LANGUAGE_CODES = ["he", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "tr"]

_ID_LINE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


# --------------------------------------------------------------------------------------
# Fake OpenAI client
# --------------------------------------------------------------------------------------


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Response:
    def __init__(self, content, prompt_tokens=1000, completion_tokens=500):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _Completions:
    def __init__(self, owner, responder):
        self._owner = owner
        self._responder = responder

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        payload = self._responder(kwargs, len(self._owner.calls))
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return _Response(content)


class FakeClient:
    """Stands in for ``openai.OpenAI`` — records calls, returns canned JSON."""

    def __init__(self, responder):
        self.calls = []
        self.chat = type("_Chat", (), {})()
        self.chat.completions = _Completions(self, responder)

    # convenience accessors -------------------------------------------------------
    def system_prompt(self, call_index=0):
        return self.calls[call_index]["messages"][0]["content"]

    def user_prompt(self, call_index=0):
        return self.calls[call_index]["messages"][1]["content"]


def requested_ids(user_prompt):
    """Ids the model is asked to emit (i.e. lines NOT marked CONTEXT-ONLY)."""
    ids = []
    for line in user_prompt.splitlines():
        match = _ID_LINE.match(line)
        if match and CONTEXT_MARKER not in match.group(2):
            ids.append(int(match.group(1)))
    return ids


def context_ids(user_prompt):
    """Ids present purely as read-only context."""
    ids = []
    for line in user_prompt.splitlines():
        match = _ID_LINE.match(line)
        if match and CONTEXT_MARKER in match.group(2):
            ids.append(int(match.group(1)))
    return ids


def echo_responder(prefix="תרגום"):
    """Returns a well-formed translation for every requested id."""

    def _responder(kwargs, call_no):
        ids = requested_ids(kwargs["messages"][1]["content"])
        return {"cues": [{"id": i, "t": f"{prefix} {i}."} for i in ids]}

    return _responder


def make_cues(count, text="Hello there, this is a sentence.", dur=3.0):
    return [
        {"start": i * dur, "end": (i + 1) * dur, "text": f"{text} ({i + 1})"}
        for i in range(count)
    ]


#: A word ``enforce_cps``'s validator treats as filler, so a condensation may delete it.
_FILLER_WORD = "מאוד"
#: The word every cue below ends on. Keeping it is what makes a shorter cue LEGAL:
#: a condensation that drops the last content word is refused (``_cps_rejection``).
_TAIL_WORD = "סוף."


def cue_text(chars, tail=_TAIL_WORD):
    """Exactly ``chars`` characters of well-formed Hebrew, ending in ``tail``.

    The body is repeated filler, so any shorter version of it that keeps ``tail`` is a
    legal condensation. Tests that are about LENGTHS and PLUMBING need text the content
    validator will not veto for reasons those tests are not about; tests that are about
    the validator use real sentences instead.
    """
    words = []
    length = len(tail)
    while length + 1 + len(_FILLER_WORD) <= chars:
        words.append(_FILLER_WORD)
        length += len(_FILLER_WORD) + 1
    assert words, f"cue_text needs at least {len(tail) + len(_FILLER_WORD) + 1} chars"
    text = " ".join(words + [tail])
    if len(text) < chars:  # pad one filler word out to land on the exact length
        words[0] += "א" * (chars - len(text))
        text = " ".join(words + [tail])
    assert len(text) == chars
    return text


# --------------------------------------------------------------------------------------
# Language names / prompt content
# --------------------------------------------------------------------------------------


@pytest.mark.unit
class TestLanguageNames:
    def test_all_app_codes_have_full_english_names(self):
        for code in APP_LANGUAGE_CODES:
            assert code in LANGUAGE_NAMES, f"missing language code {code}"
            name = LANGUAGE_NAMES[code]
            assert isinstance(name, str) and len(name) > 2
            assert name != code
            assert name[0].isupper()

        assert LANGUAGE_NAMES["he"] == "Hebrew"
        assert LANGUAGE_NAMES["zh"] == "Chinese"

    def test_prompt_uses_full_language_name_never_the_code(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client)

        system = client.system_prompt()
        user = client.user_prompt()

        assert "Hebrew" in system
        assert "Hebrew" in user
        # The raw ISO code must never be handed to the model as the target language.
        assert '"he"' not in system
        assert "translate to he\n" not in system.lower()
        assert "into he\n" not in system.lower()
        assert "into he." not in user.lower()

    def test_other_languages_resolve_to_their_name(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(2), "es", client=client)
        assert "Spanish" in client.system_prompt()

    def test_unknown_language_raises_instead_of_leaking_code(self):
        client = FakeClient(echo_responder())
        with pytest.raises(TranslationV2Error):
            translate_cues(make_cues(2), "xx", client=client)
        assert client.calls == []


@pytest.mark.unit
class TestPromptHardRules:
    def test_punctuation_and_char_limit_rules_present(self):
        system = build_system_prompt("he")
        assert "PRESERVE sentence punctuation" in system
        assert "?" in system and "!" in system
        assert str(DEFAULT_MAX_CHARS_PER_CUE) in system
        assert "Condense" in system

    def test_hebrew_gershayim_rule_present(self):
        system = build_system_prompt("he")
        assert GERSHAYIM in system
        assert GERSHAYIM == "״"
        assert "U+05F4" in system
        assert "gershayim" in system.lower()
        # And it is Hebrew-specific, not boilerplate on every language.
        assert GERSHAYIM not in build_system_prompt("es")

    def test_cross_cue_consistency_and_speaker_rules_present(self):
        system = build_system_prompt("he")
        lowered = system.lower()
        assert "one continuous conversation" in lowered
        assert "across cues" in lowered
        assert "different" in lowered and "speakers" in lowered
        assert "numbers one to ten" in lowered
        assert "ICC" in system  # keep known Latin acronyms

    def test_json_response_shape_is_specified(self):
        system = build_system_prompt("he")
        assert '"id"' in system and '"t"' in system

    def test_context_note_is_injected(self):
        system = build_system_prompt(
            "he", context_note="An interview between a host and a prime minister."
        )
        assert "An interview between a host and a prime minister." in system

    def test_response_format_and_temperature_sent(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client)
        kwargs = client.calls[0]
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.2
        assert kwargs["model"] == "gpt-4o"


@pytest.mark.unit
class TestStyleIsAUserChoice:
    def test_clean_and_faithful_prompts_differ(self):
        clean = build_system_prompt("he", "clean")
        faithful = build_system_prompt("he", "faithful")

        assert clean != faithful
        assert "REMOVE spoken disfluencies" in clean
        assert "REMOVE spoken disfluencies" not in faithful
        assert "KEEP spoken disfluencies" in faithful
        assert "KEEP spoken disfluencies" not in clean
        # The filler examples the user was promised control over:
        for filler in ('"uh"', '"you know"', '"listen"'):
            assert filler in clean
            assert filler in faithful

    def test_style_reaches_the_api_call(self):
        clean_client = FakeClient(echo_responder())
        faithful_client = FakeClient(echo_responder())
        translate_cues(make_cues(2), "he", style="clean", client=clean_client)
        translate_cues(make_cues(2), "he", style="faithful", client=faithful_client)

        assert "REMOVE spoken disfluencies" in clean_client.system_prompt()
        assert "KEEP spoken disfluencies" in faithful_client.system_prompt()

    def test_default_style_is_clean(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(2), "he", client=client)
        assert client.system_prompt() == build_system_prompt("he", "clean")

    def test_invalid_style_rejected(self):
        client = FakeClient(echo_responder())
        with pytest.raises(ValueError):
            translate_cues(make_cues(2), "he", style="verbatim-ish", client=client)


# --------------------------------------------------------------------------------------
# Happy path / batching
# --------------------------------------------------------------------------------------


@pytest.mark.unit
class TestTranslateCues:
    def test_single_request_for_scene_within_limit(self):
        client = FakeClient(echo_responder())
        cues = make_cues(MAX_CUES_PER_REQUEST)
        result = translate_cues(cues, "he", client=client)

        assert len(client.calls) == 1, "<=40 cues must be one whole-scene request"
        assert CONTEXT_MARKER not in client.user_prompt()
        assert requested_ids(client.user_prompt()) == list(range(1, MAX_CUES_PER_REQUEST + 1))
        assert all(cue["translated"] for cue in result)

    def test_input_is_not_mutated_and_keys_preserved(self):
        client = FakeClient(echo_responder())
        cues = make_cues(3)
        cues[0]["speaker"] = "HOST"
        result = translate_cues(cues, "he", client=client)

        assert "translated" not in cues[0], "input cues must not be mutated"
        assert result[0]["speaker"] == "HOST"
        assert result[0]["start"] == cues[0]["start"]
        assert result[0]["text"] == cues[0]["text"]

    def test_empty_input_makes_no_request(self):
        client = FakeClient(echo_responder())
        assert list(translate_cues([], "he", client=client)) == []
        assert client.calls == []

    def test_blank_cue_text_is_not_sent_to_the_model(self):
        client = FakeClient(echo_responder())
        cues = make_cues(3)
        cues[1]["text"] = "   "
        result = translate_cues(cues, "he", client=client)

        assert requested_ids(client.user_prompt()) == [1, 3]
        assert result[1]["translated"] == ""

    def test_usage_and_cost_exposed(self):
        client = FakeClient(echo_responder())
        result = translate_cues(make_cues(5), "he", client=client)

        assert result.usage.prompt_tokens == 1000
        assert result.usage.completion_tokens == 500
        assert result.usage.requests == 1
        # gpt-4o: 1000/1M*2.50 + 500/1M*10.00
        assert result.usage.cost_usd == pytest.approx(0.0075, rel=1e-6)
        assert result.usage.as_dict()["total_tokens"] == 1500

    def test_chunking_beyond_limit_uses_read_only_overlap(self):
        client = FakeClient(echo_responder())
        total = MAX_CUES_PER_REQUEST + MIN_TAIL_CUES  # 50 -> chunks of 40 + 10
        result = translate_cues(make_cues(total), "he", client=client)

        assert len(client.calls) == 2

        first_user = client.user_prompt(0)
        second_user = client.user_prompt(1)

        # Each cue is emitted exactly once, across chunks, in order.
        assert requested_ids(first_user) == list(range(1, MAX_CUES_PER_REQUEST + 1))
        assert requested_ids(second_user) == list(range(MAX_CUES_PER_REQUEST + 1, total + 1))

        # ...and each chunk carries OVERLAP_CUES read-only cues on the available sides.
        assert context_ids(first_user) == [
            MAX_CUES_PER_REQUEST + 1,
            MAX_CUES_PER_REQUEST + 2,
            MAX_CUES_PER_REQUEST + 3,
        ]
        assert context_ids(second_user) == [
            MAX_CUES_PER_REQUEST - 2,
            MAX_CUES_PER_REQUEST - 1,
            MAX_CUES_PER_REQUEST,
        ]
        assert len(context_ids(first_user)) == OVERLAP_CUES

        # Overlap is context, never re-translated.
        assert not set(context_ids(second_user)) & set(requested_ids(second_user))
        assert CONTEXT_MARKER in first_user and CONTEXT_MARKER in second_user
        assert "do not return them" in second_user

        assert len(result) == total
        assert all(cue["translated"] for cue in result)
        assert result.usage.requests == 2

    def test_context_only_rule_explained_in_system_prompt(self):
        assert CONTEXT_MARKER in build_system_prompt("he")
        assert "never translate them" in build_system_prompt("he")


# --------------------------------------------------------------------------------------
# ID validation / retry / no silent source-text substitution
# --------------------------------------------------------------------------------------


@pytest.mark.unit
class TestIdValidation:
    def test_missing_id_recovered_by_single_targeted_retry(self):
        """Also the regression guard for v2's own retry-collection initialisation."""

        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            if call_no == 1:
                return {"cues": [{"id": i, "t": f"תרגום {i}."} for i in ids if i != 2]}
            return {"cues": [{"id": i, "t": "התאוששות."} for i in ids]}

        client = FakeClient(responder)
        result = translate_cues(make_cues(4), "he", client=client)

        assert len(client.calls) == 2, "exactly one retry"
        assert requested_ids(client.user_prompt(1)) == [2], "retry asks only for missing ids"
        # The rest of the chunk rides along as context so the retry stays consistent.
        assert context_ids(client.user_prompt(1)) == [1, 3, 4]
        assert result[1]["translated"] == "התאוששות."
        assert all(cue["translated"] for cue in result)

    def test_persistent_mismatch_raises_and_never_returns_source_text(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": f"תרגום {i}."} for i in ids if i != 3]}

        client = FakeClient(responder)
        cues = make_cues(5)

        with pytest.raises(TranslationV2Error) as excinfo:
            translate_cues(cues, "he", client=client)

        message = str(excinfo.value)
        assert "3" in message
        assert excinfo.value.missing_ids == [3]
        assert len(client.calls) == 2, "one retry, then fail — no infinite loop"
        # Nothing was mutated, and no source text was passed off as a translation.
        assert all("translated" not in cue for cue in cues)
        assert cues[2]["text"] not in message

    def test_extra_unrequested_ids_are_ignored(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            entries = [{"id": i, "t": f"תרגום {i}."} for i in ids]
            entries.append({"id": 999, "t": "לא ביקשנו."})
            return {"cues": entries}

        client = FakeClient(responder)
        result = translate_cues(make_cues(3), "he", client=client)

        assert len(client.calls) == 1
        assert len(result) == 3
        assert all(cue["translated"].startswith("תרגום") for cue in result)

    def test_empty_translation_string_counts_as_missing(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": "" if i == 1 else f"תרגום {i}."} for i in ids]}

        client = FakeClient(responder)
        with pytest.raises(TranslationV2Error) as excinfo:
            translate_cues(make_cues(3), "he", client=client)
        assert excinfo.value.missing_ids == [1]

    def test_invalid_json_raises_translation_error(self):
        client = FakeClient(lambda kwargs, call_no: "not json at all")
        with pytest.raises(TranslationV2Error):
            translate_cues(make_cues(2), "he", client=client)


# --------------------------------------------------------------------------------------
# CPS enforcement
# --------------------------------------------------------------------------------------


@pytest.mark.unit
class TestEnforceCps:
    def _cues(self):
        # cue 1: 2s * 17 CPS = 34 char budget, 60 chars -> violator
        # cue 2: 6s -> 84 char cap, 20 chars -> fine
        return [
            {"start": 0.0, "end": 2.0, "translated": cue_text(60)},
            {"start": 2.0, "end": 8.0, "translated": cue_text(20)},
        ]

    def test_no_request_when_everything_is_within_budget(self):
        client = FakeClient(echo_responder())
        cues = [{"start": 0.0, "end": 5.0, "translated": "שלום עולם."}]
        result = enforce_cps(cues, client=client)

        assert client.calls == []
        assert result[0]["translated"] == "שלום עולם."
        assert result.usage.requests == 0

    def test_only_violators_are_re_asked(self):
        """A legal reply at or under the limit is accepted at once."""
        fitting = cue_text(32)  # budget 34, floor ceil(34*0.85) = 29

        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": fitting} for i in ids]}

        client = FakeClient(responder)
        cues = self._cues()
        result = enforce_cps(cues, client=client)

        assert len(client.calls) == 1, "one batched request, no re-ask needed"
        assert requested_ids(client.user_prompt()) == [1]
        user = client.user_prompt()
        assert "max 34 chars" in user, "per-cue budget = floor(max_cps * duration)"
        assert "never below 29" in user, "the floor is stated to the model"
        assert result[0]["translated"] == fitting
        assert result[1]["translated"] == cue_text(20), "compliant cue untouched"
        assert cues[0]["translated"] == cue_text(60), "input not mutated"

    def test_an_over_condensed_but_legal_reply_is_shipped_not_re_asked(self):
        """R9: the 85% floor is GUIDANCE, and no longer a reason to send a second request.

        Measured over the eight-clip corpus: four cues entered the TOO SHORT re-ask,
        none came back better, and one came back destroyed. What the floor was a proxy
        FOR — a deleted content word — is now measured directly, so a short reply that
        kept everything is simply a good short reply.
        """
        crushed = "סוף."  # 4 chars against a 34-char budget, but nothing was lost

        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": crushed} for i in ids]}

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)

        assert len(client.calls) == 1, "a legal reply must not cost a second request"
        assert result[0]["translated"] == crushed

    def test_the_longest_fitting_candidate_wins(self):
        """Both attempts are legal; the one that threw away less content is the answer."""
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            # 50 is over the 34 budget AND over the 10% acceptance margin.
            text = cue_text(50) if call_no == 1 else cue_text(30)
            return {"cues": [{"id": i, "t": text} for i in ids]}

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == cue_text(30)

    def test_a_cue_barely_over_budget_is_left_alone(self):
        """The 10% trigger margin: a 0.1s reading-time overrun is not worth a rewrite."""
        from services.translation_v2 import CPS_TRIGGER_MARGIN

        client = FakeClient(echo_responder())
        # 6s cue -> 84-char cap; 88 chars is over the cap but inside the margin.
        assert 88 <= DEFAULT_MAX_CHARS_PER_CUE * CPS_TRIGGER_MARGIN
        cues = [{"start": 0.0, "end": 6.0, "translated": cue_text(88)}]
        result = enforce_cps(cues, client=client)
        assert client.calls == []
        assert result[0]["translated"] == cue_text(88)

    def test_char_cap_applies_to_long_duration_cues(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": "סוף."} for i in ids]}

        client = FakeClient(responder)
        # 30s duration -> CPS is fine, but 100 chars breaks the 84-char cue cap.
        cues = [{"start": 0.0, "end": 30.0, "translated": cue_text(100)}]
        result = enforce_cps(cues, client=client)

        assert requested_ids(client.user_prompt()) == [1]
        assert f"max {DEFAULT_MAX_CHARS_PER_CUE} chars" in client.user_prompt()
        assert result[0]["translated"] == "סוף."

    def test_still_violating_is_re_asked_once_then_the_shortest_is_kept(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": cue_text(40 if call_no == 1 else 38)}]}

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)

        assert len(client.calls) == 2, "exactly one re-ask, never a loop"
        assert "TOO LONG" in client.user_prompt(1)
        assert result[0]["translated"] == cue_text(38)

    def test_longer_reply_is_discarded_in_favour_of_original(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": cue_text(120)}]}

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == cue_text(60)

    def test_missing_replacement_keeps_original(self):
        client = FakeClient(lambda kwargs, call_no: {"cues": []})
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == cue_text(60)

    def test_request_failure_does_not_break_the_job(self):
        client = FakeClient(lambda kwargs, call_no: "}}} not json")
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == cue_text(60)

    def test_condensation_prompt_keeps_language_and_punctuation(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": "מקוצר."}]}

        client = FakeClient(responder)
        enforce_cps(self._cues(), client=client)

        system = client.system_prompt()
        assert "Do NOT translate" in system
        assert "PRESERVE sentence punctuation" in system
        assert GERSHAYIM in system
        assert client.calls[0]["response_format"] == {"type": "json_object"}


# --------------------------------------------------------------------------------------
# CPS batching, output budget, and partial failure
# --------------------------------------------------------------------------------------


_MARKER_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"


def marker(index):
    """A unique, digit-free Hebrew word, so no two fixture cues carry the same text.

    Digit-free on purpose: a token carrying a number may never be deleted by a
    condensation (``_cps_rejection``), and these fixtures need text the model is allowed
    to trim.
    """
    letters = _MARKER_LETTERS
    return letters[index // len(letters) % len(letters)] + letters[index % len(letters)]


def cps_violators(count, chars=90, dur=2.0):
    """``count`` cues that all break their reading-speed budget.

    2s at 17 CPS is a 34-character budget, so 90 characters violates it.

    90 and not 60, deliberately: 90 is also over the 84-character FRAME budget, which
    is the one thing :func:`apply_time_relief` cannot fix with time. These tests are
    about batching and about what happens to the TEXT, and a last cue that quietly
    solved itself by growing into the silence after it would make them measure the
    wrong thing.
    """
    return [
        {
            "start": i * dur,
            "end": (i + 1) * dur,
            "translated": f"{marker(i)} {cue_text(chars)}",
        }
        for i in range(count)
    ]


@pytest.mark.unit
class TestEnforceCpsBatching:
    """Condensation is batched. One unbounded request on a long video is a lost pass.

    The reply is JSON containing every cue, so an hour of subtitles in a single request
    runs past the completion budget and comes back truncated — which
    :func:`_parse_cue_map` rejects, discarding the whole pass rather than one batch of it.
    """

    def test_a_long_video_is_split_into_several_requests(self):
        fitting = cue_text(32)
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": fitting} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        violators = cps_violators(95)
        result = enforce_cps(violators, client=client)

        expected_batches = -(-95 // MAX_CUES_PER_CPS_REQUEST)  # ceil
        assert len(client.calls) == expected_batches == 3

        for index in range(len(client.calls)):
            asked = requested_ids(client.user_prompt(index))
            assert 0 < len(asked) <= MAX_CUES_PER_CPS_REQUEST, (
                f"batch {index} asked for {len(asked)} cues"
            )

        # Every cue is asked for exactly once — no gaps, no duplicates.
        asked_overall = [
            cue_id
            for index in range(len(client.calls))
            for cue_id in requested_ids(client.user_prompt(index))
        ]
        assert sorted(asked_overall) == list(range(1, 96))
        assert len(asked_overall) == len(set(asked_overall))
        assert all(cue["translated"] == fitting for cue in result)

    def test_exactly_one_batch_when_the_violators_fit(self):
        fitting = cue_text(32)
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": fitting} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        enforce_cps(cps_violators(MAX_CUES_PER_CPS_REQUEST), client=client)
        assert len(client.calls) == 1

    def test_every_request_declares_an_output_budget(self):
        """``max_tokens`` scaled to the batch, so a runaway reply is cut off, not billed.

        Each reply cue is by construction shorter than its input, which is itself capped
        at ``DEFAULT_MAX_CHARS_PER_CUE``, so the per-cue allowance is generous.
        """
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": "סוף."} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        enforce_cps(cps_violators(50), client=client)

        for index, call in enumerate(client.calls):
            batch_size = len(requested_ids(client.user_prompt(index)))
            assert call["max_tokens"] == batch_size * CPS_TOKENS_PER_CUE
            assert call["max_tokens"] > 0

    def test_translation_itself_is_not_capped(self):
        """``translate_cues`` must NOT inherit the ceiling: its replies are full sentences."""
        client = FakeClient(echo_responder())
        translate_cues(make_cues(5), "he", client=client)
        assert "max_tokens" not in client.calls[0]

    def test_one_failed_batch_does_not_cost_the_others(self):
        """A cosmetic pass must degrade per batch, not collapse entirely.

        Before batching there was one request, so any failure lost the whole pass. Now a
        bad reply must leave *its* cues untouched and let the rest through.
        """

        fitting = cue_text(32)

        def responder(kwargs, call_no):
            if call_no == 2:
                return "}}} truncated garbage"
            return {
                "cues": [
                    {"id": i, "t": fitting}
                    for i in requested_ids(kwargs["messages"][1]["content"])
                ]
            }

        client = FakeClient(responder)
        originals = cps_violators(95)
        result = enforce_cps(originals, client=client)

        assert len(client.calls) == 3, "a failed batch aborted the remaining ones"

        failed_ids = set(requested_ids(client.user_prompt(1)))
        for index, cue in enumerate(result, 1):
            if index in failed_ids:
                assert cue["translated"] == originals[index - 1]["translated"], (
                    f"cue {index} was in the failed batch but was modified"
                )
            else:
                assert cue["translated"] == fitting, (
                    f"cue {index} was in a successful batch but was not condensed"
                )

    def test_a_failure_in_every_batch_still_returns_the_input(self):
        client = FakeClient(lambda kwargs, call_no: "}}} not json")
        originals = cps_violators(50)
        result = enforce_cps(originals, client=client)
        assert [c["translated"] for c in result] == [
            c["translated"] for c in originals
        ]

    def test_keep_shorter_semantics_survive_batching(self):
        """When nothing fits, the shortest of {attempts, original} is kept — per batch.

        The re-ask repeats the same failure here on purpose: this test is about what
        happens when the model simply cannot deliver, across batch boundaries.
        """

        def responder(kwargs, call_no):
            out = []
            for cue_id in requested_ids(kwargs["messages"][1]["content"]):
                # odd ids: shorter but still over budget; even ids: longer than the input
                out.append({"id": cue_id, "t": cue_text(40 if cue_id % 2 else 200)})
            return {"cues": out}

        client = FakeClient(responder)
        originals = cps_violators(45)
        result = enforce_cps(originals, client=client)

        for index, cue in enumerate(result, 1):
            if index % 2:
                assert cue["translated"] == cue_text(40), "the shorter reply should be kept"
            else:
                assert cue["translated"] == originals[index - 1]["translated"], (
                    "a longer reply must be discarded in favour of the original"
                )


@pytest.mark.unit
class TestProgressReporting:
    """Translation is serial and can run for minutes; the UI must not sit frozen."""

    def test_translate_cues_reports_progress_per_chunk(self):
        client = FakeClient(echo_responder())
        seen = []
        cues = make_cues(MAX_CUES_PER_REQUEST * 2 + 5)

        translate_cues(
            cues, "he", client=client,
            progress_callback=lambda done, total, message: seen.append((done, total, message)),
        )

        assert seen, "no progress was reported at all"
        assert all(total == len(cues) for _done, total, _msg in seen)
        assert all(0 <= done <= len(cues) for done, _total, _msg in seen)
        assert [done for done, _t, _m in seen] == sorted(done for done, _t, _m in seen), (
            "progress went backwards"
        )
        assert seen[-1][0] == len(cues), "the final report is not 100%"
        assert all(isinstance(message, str) and message for _d, _t, message in seen)

    def test_enforce_cps_reports_progress_per_batch(self):
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": "קצר."} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        seen = []
        enforce_cps(
            cps_violators(95), client=client,
            progress_callback=lambda done, total, message: seen.append((done, total, message)),
        )
        assert len(seen) >= 3
        assert seen[-1][0] == seen[-1][1], "the reading-speed pass never reported done"

    def test_no_progress_is_reported_when_nothing_violates(self):
        """A pass that makes no request should not claim to be working."""
        client = FakeClient(echo_responder())
        seen = []
        enforce_cps(
            [{"start": 0.0, "end": 5.0, "translated": "שלום."}],
            client=client,
            progress_callback=lambda *args: seen.append(args),
        )
        assert client.calls == []
        assert seen == []

    def test_a_broken_callback_never_fails_the_translation(self):
        """Reporting progress is best-effort; it must not be able to fail a paid job."""

        def exploding(*_args):
            raise RuntimeError("the UI channel died")

        client = FakeClient(echo_responder())
        result = translate_cues(make_cues(3), "he", client=client, progress_callback=exploding)
        assert len(result) == 3
        assert all(cue["translated"] for cue in result)

    def test_a_broken_callback_never_fails_the_cps_pass(self):
        def exploding(*_args):
            raise RuntimeError("the UI channel died")

        fitting = cue_text(32)
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": fitting} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        result = enforce_cps(cps_violators(3), client=client, progress_callback=exploding)
        assert all(cue["translated"] == fitting for cue in result)

    def test_translation_works_with_no_callback_at_all(self):
        client = FakeClient(echo_responder())
        assert len(translate_cues(make_cues(3), "he", client=client)) == 3
        assert len(enforce_cps(cps_violators(3), client=client)) == 3


# --------------------------------------------------------------------------------------
# Regression guard for the legacy retry_translations NameError
# --------------------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyRetryTranslationsRegression:
    """
    The legacy recovery path in ``translation_services`` populated ``retry_translations``
    without ever creating it, so every attempt to recover missing cue ids died with a
    NameError that was swallowed by the surrounding ``except Exception``.

    Choice of seam: the buggy statement sits ~150 lines deep inside
    ``_make_openai_request_with_retries``, behind the module-level rate limiter, the
    token-budget acquisition loop and two nested retry loops. Driving it through the
    public API would require mocking most of ``openai_rate_limiter`` and would test the
    scaffolding rather than the fix. A source-level assertion pins the exact invariant
    that broke (initialisation must exist, at the right scope, before the first use) and
    cannot pass accidentally.
    """

    @staticmethod
    def _source():
        path = os.path.join(backend_dir, "services", "translation_services.py")
        with open(path, encoding="utf-8") as handle:
            return handle.read().splitlines()

    def test_retry_translations_is_initialized_before_use(self):
        lines = self._source()

        init_lines = [
            i for i, line in enumerate(lines) if re.match(r"\s*retry_translations\s*=\s*\{\s*\}", line)
        ]
        use_lines = [
            i for i, line in enumerate(lines) if "retry_translations[" in line or "in retry_translations" in line
        ]

        assert init_lines, "retry_translations is never initialized (NameError at runtime)"
        assert use_lines, "test is stale: retry_translations is no longer used"
        assert min(init_lines) < min(use_lines), (
            "retry_translations must be initialized before it is populated/read"
        )

    def test_initialization_is_at_the_scope_of_its_use(self):
        """Init indentation must not be deeper than the first use (same or outer block)."""
        lines = self._source()

        init_line = next(
            i for i, line in enumerate(lines) if re.match(r"\s*retry_translations\s*=\s*\{\s*\}", line)
        )
        first_use = next(
            i for i, line in enumerate(lines) if "retry_translations[" in line
        )

        def indent(text):
            return len(text) - len(text.lstrip())

        assert indent(lines[init_line]) <= indent(lines[first_use])
        assert first_use - init_line < 60, "init drifted far from the block it guards"


@pytest.mark.unit
class TestRobustnessRules:
    """Prompt rules added after the large-v3 unpunctuated-transcript incident."""

    def test_prompt_handles_unpunctuated_source(self):
        from services.translation_v2 import build_system_prompt
        prompt = build_system_prompt("he")
        assert "WITHOUT punctuation" in prompt
        assert "punctuate the translation correctly" in prompt.lower() or "punctuate the translation" in prompt

    def test_prompt_defaults_unknown_gender_to_masculine(self):
        """Masculine is the default with no textual evidence — see TestGenderInference."""
        from services.translation_v2 import build_system_prompt
        prompt = build_system_prompt("he")
        assert "WITH NO SUCH EVIDENCE, USE MASCULINE FORMS" in prompt
        assert "mid-conversation" in prompt


@pytest.mark.unit
class TestCpsReport:
    """``cps_report`` must measure the TRANSLATION, in whichever key it arrives under.

    The bug this class was written for: ``cps_report`` read ``translated`` only, but
    ``process_video_task`` calls it AFTER ``subtitle_pipeline.normalize_cues``, which
    has already renamed that key to ``translated_text``. So every reported
    ``cps_over_budget`` figure was measured on the untranslated English source. Caught
    by re-deriving the number from an archived research run (64-cue portrait job:
    the pipeline logged 39 cues over budget; the Hebrew it actually rendered had 2).
    """

    CUES = [
        # 20 Hebrew chars in 2s = 10 CPS, comfortably in budget...
        {"start": 0.0, "end": 2.0, "text": "a" * 60, "translated_text": "כן, אני חושב על זה."},
    ]

    def test_translated_text_is_preferred_over_the_source(self):
        from services.translation_v2 import cps_report

        report = cps_report(self.CUES)
        assert report[0]["chars"] == 19  # the Hebrew, not the 60-char source
        assert report[0]["ok"] is True

    def test_the_bug_would_have_reported_the_source_as_over_budget(self):
        """Guards the guard: measuring ``text`` here must give a different verdict."""
        from services.translation_v2 import cps_report

        assert cps_report(self.CUES, max_chars_per_cue=40)[0]["ok"] is True
        # ...whereas the 60-char source at 2s is 30 CPS and 60 chars: over on both counts.
        source_only = [{"start": 0.0, "end": 2.0, "text": "a" * 60}]
        assert cps_report(source_only, max_chars_per_cue=40)[0]["ok"] is False

    def test_translated_key_still_wins(self):
        """Pre-normalisation cues (straight out of ``translate_cues``) still work."""
        from services.translation_v2 import cps_report

        cues = [{"start": 0.0, "end": 2.0, "text": "x" * 60, "translated": "שלום."}]
        assert cps_report(cues)[0]["chars"] == len("שלום.")

    def test_a_blank_translation_falls_through_to_the_source(self):
        """An untranslated cue must be measured, not silently reported as 0 chars."""
        from services.translation_v2 import cps_report

        cues = [{"start": 0.0, "end": 2.0, "text": "hello there", "translated_text": ""}]
        assert cps_report(cues)[0]["chars"] == len("hello there")

    def test_the_frame_budget_is_honoured(self):
        """A portrait job measures against 66 chars, not the landscape 84."""
        from services.translation_v2 import cps_report

        cues = [{"start": 0.0, "end": 10.0, "text": "", "translated_text": "ש" * 70}]
        assert cps_report(cues, max_chars_per_cue=84)[0]["ok"] is True
        assert cps_report(cues, max_chars_per_cue=66)[0]["ok"] is False

    def test_zero_duration_is_never_in_budget(self):
        from services.translation_v2 import cps_report

        cues = [{"start": 1.0, "end": 1.0, "text": "", "translated_text": "שלום."}]
        report = cps_report(cues)
        assert report[0]["cps"] is None
        assert report[0]["ok"] is True  # length is fine; duration cannot be judged

    def test_empty_input(self):
        from services.translation_v2 import cps_report

        assert cps_report([]) == []
        assert cps_report(None) == []


# --------------------------------------------------------------------------------------
# P2 — terminology contrast + glossary
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestTerminologyContrastRule:
    """A clip arguing "it is not called Hebrew, it is called Ivrit" rendered BOTH names
    as עברית, so 36% of its cues asserted "it is not called X, it is called X".

    Collapsing two names for one referent is normally correct, which is exactly why the
    exception has to be stated rather than assumed.
    """

    def test_contrast_rule_is_present_for_every_language(self):
        for code in APP_LANGUAGE_CODES:
            prompt = build_system_prompt(code)
            assert "DISTINGUISHES" in prompt, code
            assert "TRANSLITERATE" in prompt, code

    def test_rule_names_the_failure_it_prevents(self):
        prompt = build_system_prompt("he")
        assert "it is not called A, it is called A" in prompt

    def test_rule_is_numbered_among_the_hard_rules(self):
        prompt = build_system_prompt("he")
        rules = prompt.split("HARD RULES\n", 1)[1]
        contrast = [l for l in rules.splitlines() if "DISTINGUISHES" in l]
        assert len(contrast) == 1
        assert re.match(r"^\d+\.\s", contrast[0])


@pytest.mark.unit
class TestGlossary:
    """An optional binding term list. Empty by default, so today's prompts are unchanged."""

    def test_absent_glossary_changes_nothing(self):
        base = build_system_prompt("he")
        assert build_system_prompt("he", glossary=None) == base
        assert build_system_prompt("he", glossary={}) == base
        assert "GLOSSARY" not in base

    def test_entries_are_rendered_verbatim_and_quoted(self):
        prompt = build_system_prompt("he", glossary={"Ivrit": "עִברית"})
        assert "GLOSSARY" in prompt
        assert '- "Ivrit" -> "עִברית"' in prompt

    def test_glossary_declares_itself_binding_over_the_rules(self):
        prompt = build_system_prompt("he", glossary={"Ivrit": "עִברית"})
        assert "overrides every rule above" in prompt

    def test_glossary_is_last_so_it_is_not_buried(self):
        prompt = build_system_prompt("he", glossary={"Ivrit": "עִברית"})
        assert prompt.index("GLOSSARY") > prompt.index("HARD RULES")

    def test_multiple_entries_keep_insertion_order(self):
        prompt = build_system_prompt(
            "he", glossary={"Ivrit": "עִברית", "Parsi": "פארסי"}
        )
        assert prompt.index('"Ivrit"') < prompt.index('"Parsi"')

    def test_translate_cues_threads_the_glossary_into_the_system_prompt(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client, glossary={"Ivrit": "עִברית"})
        assert '- "Ivrit" -> "עִברית"' in client.calls[0]["messages"][0]["content"]

    def test_translate_cues_without_glossary_sends_no_glossary_block(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client)
        assert "GLOSSARY" not in client.calls[0]["messages"][0]["content"]


@pytest.mark.unit
class TestNormalizeGlossary:
    """Untrusted input: it will come from a UI text box. It may never raise."""

    def test_none_and_empty(self):
        from services.translation_v2 import normalize_glossary

        assert normalize_glossary(None) == {}
        assert normalize_glossary({}) == {}

    def test_non_mapping_is_ignored(self):
        from services.translation_v2 import normalize_glossary

        assert normalize_glossary("Ivrit") == {}
        assert normalize_glossary(["Ivrit"]) == {}
        assert normalize_glossary(42) == {}

    def test_entries_are_stripped(self):
        from services.translation_v2 import normalize_glossary

        assert normalize_glossary({"  Ivrit  ": "  עִברית "}) == {"Ivrit": "עִברית"}

    def test_blank_and_non_string_entries_are_dropped(self):
        from services.translation_v2 import normalize_glossary

        out = normalize_glossary(
            {"Ivrit": "עִברית", "": "x", "y": "   ", 3: "z", "w": None}
        )
        assert out == {"Ivrit": "עִברית"}

    def test_overlong_glossary_is_truncated_not_rejected(self):
        from services.translation_v2 import MAX_GLOSSARY_ENTRIES, normalize_glossary

        big = {f"term{i}": f"t{i}" for i in range(MAX_GLOSSARY_ENTRIES + 10)}
        out = normalize_glossary(big)
        assert len(out) == MAX_GLOSSARY_ENTRIES


# --------------------------------------------------------------------------------------
# P5 — geresh and in-phrase gender agreement (prompt side)
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestHebrewTypographyAndAgreementRules:
    def test_geresh_rule_present_for_hebrew(self):
        from services.translation_v2 import GERESH

        prompt = build_system_prompt("he")
        assert GERESH in prompt
        assert "U+05F3" in prompt
        assert f"ג{GERESH}ורג{GERESH}" in prompt

    def test_geresh_rule_forbids_the_ascii_apostrophe(self):
        prompt = build_system_prompt("he")
        assert "never an" in prompt and "ASCII apostrophe" in prompt

    def test_gershayim_rule_still_present(self):
        """The new geresh rule must not have displaced the acronym rule."""
        prompt = build_system_prompt("he")
        assert GERSHAYIM in prompt
        assert "U+05F4" in prompt

    def test_noun_phrase_agreement_rule_present(self):
        prompt = build_system_prompt("he")
        assert "WITHIN each noun phrase" in prompt
        assert "אותה מחווה מדהימה" in prompt

    def test_agreement_rule_cites_the_observed_defect(self):
        """The engine produced `את אותו מחווה מדהימה` — masculine + feminine in one NP."""
        prompt = build_system_prompt("he")
        assert "את אותו מחווה מדהימה" in prompt

    def test_speaker_gender_rule_is_still_there_too(self):
        prompt = build_system_prompt("he")
        assert "EXPLICIT TEXTUAL EVIDENCE" in prompt
        assert "WITH NO SUCH EVIDENCE, USE MASCULINE FORMS" in prompt

    def test_hebrew_only_rules_absent_for_other_languages(self):
        prompt = build_system_prompt("es")
        assert "noun phrase" not in prompt
        assert "U+05F3" not in prompt


# --------------------------------------------------------------------------------------
# R1 — time relief: a cue is over budget because of a RATIO, and time is half of it
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestTimeRelief:
    """Extend first, condense only what time cannot fix."""

    def test_a_cue_with_silence_after_it_is_extended_not_shortened(self):
        from services.translation_v2 import apply_time_relief

        # 51 chars at 17 CPS needs 3.0s; the cue has 2.0s and 8 seconds of silence after.
        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 51},
                {"start": 10.0, "end": 12.0, "translated": "ב" * 10}]
        out, relieved = apply_time_relief(cues)
        assert relieved == 1
        assert out[0]["end"] == 3.0
        assert out[0]["translated"] == "א" * 51, "text untouched"
        assert cues[0]["end"] == 2.0, "input not mutated"

    def test_relief_never_overlaps_the_next_cue(self):
        from services.translation_v2 import CPS_MIN_CUE_GAP, apply_time_relief

        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 51},
                {"start": 2.5, "end": 4.0, "translated": "ב" * 10}]
        out, _relieved = apply_time_relief(cues)
        assert out[0]["end"] == round(2.5 - CPS_MIN_CUE_GAP, 3)

    def test_relief_never_exceeds_the_maximum_cue_duration(self):
        from services.translation_v2 import CPS_MAX_CUE_DUR, apply_time_relief

        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 200}]
        out, _relieved = apply_time_relief(cues, max_chars_per_cue=400)
        assert out[0]["end"] == CPS_MAX_CUE_DUR

    def test_relief_never_runs_past_the_end_of_the_video(self):
        from services.translation_v2 import apply_time_relief

        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 51}]
        out, _relieved = apply_time_relief(cues, video_duration=2.4)
        assert out[0]["end"] == 2.4

    def test_a_cue_over_the_FRAME_budget_is_not_relieved(self):
        """No amount of time makes 100 characters fit an 84-character frame."""
        from services.translation_v2 import apply_time_relief

        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 100}]
        out, relieved = apply_time_relief(cues)
        assert relieved == 0 and out[0]["end"] == 2.0

    def test_a_compliant_cue_is_never_touched(self):
        from services.translation_v2 import apply_time_relief

        cues = [{"start": 0.0, "end": 5.0, "translated": "שלום."}]
        out, relieved = apply_time_relief(cues)
        assert relieved == 0 and out == cues

    def test_enforce_cps_tries_time_before_it_spends_a_token(self):
        client = FakeClient(echo_responder())
        cues = [{"start": 0.0, "end": 2.0, "translated": "א" * 51},
                {"start": 10.0, "end": 12.0, "translated": "ב" * 10}]
        result = enforce_cps(cues, client=client)

        assert client.calls == [], "condensed a cue that only needed more time"
        assert result[0]["end"] == 3.0
        assert result[0]["translated"] == "א" * 51

    def test_time_relief_can_be_switched_off(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": cue_text(32)} for i in ids]}

        client = FakeClient(responder)
        cues = [{"start": 0.0, "end": 2.0, "translated": cue_text(51)},
                {"start": 10.0, "end": 12.0, "translated": cue_text(10)}]
        result = enforce_cps(cues, client=client, time_relief=False)
        assert len(client.calls) == 1
        assert result[0]["end"] == 2.0


# --------------------------------------------------------------------------------------
# R5 — gender is INFERRED; masculine is the last resort, not the first
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestGenderInference:
    """Gender comes from TEXT or it does not come at all.

    R5 asked the model to infer gender "from ALL the evidence in the scene", and the
    model duly inferred it from the atmosphere: an interview between two men (Hannity
    and Netanyahu, no gendered word anywhere in the transcript) came back with five of
    ten cues in the feminine, and a street respondent whose gender is stated nowhere
    became "מה את אוכלת?". There is no picture in the prompt — a model has nothing to
    infer FROM except the words, so "infer" without a source list is an invitation to
    guess, and a guess is right half the time and visibly wrong the other half.
    """

    def test_the_rule_names_the_only_admissible_evidence(self):
        prompt = build_system_prompt("he")
        assert "INFER each speaker's and each addressee's gender" in prompt
        assert "EXPLICIT TEXTUAL EVIDENCE" in prompt
        for admissible in ("a personal name", "my husband", "gendered pronoun"):
            assert admissible in prompt, admissible

    def test_atmosphere_is_explicitly_not_evidence(self):
        prompt = build_system_prompt("he")
        assert "are NOT evidence" in prompt
        for inadmissible in ("Tone, topic, politeness", "who is asking and who is answering"):
            assert inadmissible in prompt, inadmissible

    def test_masculine_is_the_default_and_it_is_mandatory(self):
        prompt = build_system_prompt("he")
        assert "WITH NO SUCH EVIDENCE, USE MASCULINE FORMS" in prompt
        assert "mandatory, not a preference" in prompt
        assert "Never invent a gender" in prompt

    def test_the_no_switching_guarantee_survives(self):
        prompt = build_system_prompt("he")
        assert "never switch it mid-conversation" in prompt

    def test_it_is_a_single_rule(self):
        rules = build_system_prompt("he").split("HARD RULES\n", 1)[1]
        gender = [line for line in rules.splitlines() if "INFER each speaker" in line]
        assert len(gender) == 1
        assert re.match(r"^\d+\.\s", gender[0])


# --------------------------------------------------------------------------------------
# R6 — filler list, terminology frame, dialogue dash
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestFillerAndDashRules:
    def test_sentence_initial_like_is_named_as_filler(self):
        prompt = build_system_prompt("he", "clean")
        assert 'sentence-initial "like"' in prompt

    def test_comparative_like_is_explicitly_protected(self):
        """It was rendered as the comparative כמו, which is ungrammatical there."""
        prompt = build_system_prompt("he", "clean")
        assert "comparative" in prompt
        assert "it moves like a train" in prompt

    def test_look_survives_in_both_styles(self):
        assert '"look"' in build_system_prompt("he", "clean")
        assert '"look"' in build_system_prompt("he", "faithful")

    def test_faithful_still_separates_interjection_from_comparison(self):
        prompt = build_system_prompt("he", "faithful")
        assert "KEEP spoken disfluencies" in prompt
        assert 'sentence-initial "like" is an' in prompt

    def test_the_dialogue_dash_rule_is_present(self):
        from services.translation_v2 import DIALOGUE_DASH

        prompt = build_system_prompt("he")
        assert DIALOGUE_DASH in prompt
        assert "MUST begin with" in prompt

    def test_the_cps_prompt_protects_the_dash_too(self):
        from services.translation_v2 import _CPS_SYSTEM_PROMPT, DIALOGUE_DASH

        assert DIALOGUE_DASH in _CPS_SYSTEM_PROMPT


@pytest.mark.unit
class TestTerminologySyntacticFrame:
    """The rule missed its own literal example, so it now names the FRAME."""

    def test_the_syntactic_frame_is_stated(self):
        prompt = build_system_prompt("he")
        assert "SYNTACTIC FRAME" in prompt
        assert "it wasn't called X, it was called Y" in prompt
        assert "they didn't call it X, they called it Y" in prompt

    def test_the_hebrew_ivrit_failure_is_quoted_verbatim(self):
        prompt = build_system_prompt("he")
        assert "לא קראו לזה עברית. קראו לזה עברית." in prompt

    def test_the_parsi_farsi_phonetic_contrast_is_an_example(self):
        prompt = build_system_prompt("he")
        assert "פארסי" in prompt and "פרסי" in prompt

    def test_transliteration_is_the_instruction(self):
        for code in ("he", "ar", "es"):
            assert "TRANSLITERATE the foreign term" in build_system_prompt(code), code


# --------------------------------------------------------------------------------------
# R7 — he->he is proofreading, not translating
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestProofreadMode:
    """23 of 24 cues came back byte-identical and were billed for. The same call,
    given the right contract, fixes verified ASR errors instead."""

    def test_same_language_detection(self):
        from services.translation_v2 import same_language

        assert same_language("he", "he") is True
        assert same_language("he-IL", "he") is True
        assert same_language("HE", "he") is True
        assert same_language("en", "he") is False
        assert same_language("auto", "he") is False
        assert same_language(None, "he") is False
        assert same_language("", "he") is False

    def test_the_prompt_is_swapped_when_source_equals_target(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client, source_lang="he")
        system = client.system_prompt()
        assert "proofreader" in system
        assert "This is NOT a translation task" in system

    def test_the_translation_prompt_is_used_when_the_languages_differ(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client, source_lang="en")
        assert "proofreader" not in client.system_prompt()
        assert "professional broadcast subtitler" in client.system_prompt()

    def test_auto_source_keeps_the_translation_prompt(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client, source_lang="auto")
        assert "proofreader" not in client.system_prompt()

    def test_no_source_language_keeps_todays_behaviour(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client)
        assert "proofreader" not in client.system_prompt()

    def test_the_result_reports_which_contract_ran(self):
        client = FakeClient(echo_responder())
        assert translate_cues(make_cues(2), "he", client=client, source_lang="he").mode == (
            "proofread"
        )
        assert translate_cues(make_cues(2), "he", client=client, source_lang="en").mode == (
            "translate"
        )

    def test_the_proofread_contract_states_its_rules(self):
        from services.translation_v2 import build_proofread_prompt

        prompt = build_proofread_prompt("he")
        assert "CHARACTER-FOR-CHARACTER unchanged" in prompt
        assert "garbled proper nouns" in prompt
        assert "LEAVE IT ALONE" in prompt
        assert "gershayim" in prompt

    def test_the_proofread_contract_keeps_the_same_json_shape(self):
        from services.translation_v2 import build_proofread_prompt

        prompt = build_proofread_prompt("he")
        assert '{"cues":[{"id":<int>,"t":' in prompt
        assert CONTEXT_MARKER in prompt

    def test_the_proofread_contract_honours_the_style_choice(self):
        from services.translation_v2 import build_proofread_prompt

        assert "REMOVE spoken disfluencies" in build_proofread_prompt("he", "clean")
        assert "KEEP spoken disfluencies" in build_proofread_prompt("he", "faithful")

    def test_the_user_message_says_proofread_not_translate(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(3), "he", client=client, source_lang="he")
        assert client.user_prompt().startswith("Proofread 3 Hebrew cues")

    def test_the_recorder_stage_names_the_mode(self):
        class Spy:
            def __init__(self):
                self.stages = []

            def record_llm(self, *, stage, system, user, response, meta):
                self.stages.append(stage)

        spy = Spy()
        client = FakeClient(echo_responder())
        translate_cues(make_cues(2), "he", client=client, source_lang="he", recorder=spy)
        assert spy.stages == ["proofread_chunk_1"]


# --------------------------------------------------------------------------------------
# R2 — dropped text still reaches the model, as context
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestContextOnlyPassthrough:
    """A dropped cue vanishing from the prompt is how GPT loses an antecedent."""

    @staticmethod
    def _cues():
        return [
            {"start": 0.0, "end": 2.0, "text": "First line."},
            {"start": 2.0, "end": 3.0, "text": "The dropped one.", "context_only": True},
            {"start": 3.0, "end": 5.0, "text": "It was terrible."},
        ]

    def test_a_context_only_cue_is_shown_but_never_requested(self):
        client = FakeClient(echo_responder())
        translate_cues(self._cues(), "he", client=client)
        user = client.user_prompt()
        assert requested_ids(user) == [1, 3]
        assert context_ids(user) == [2]
        assert "The dropped one." in user

    def test_a_context_only_cue_comes_back_untranslated(self):
        client = FakeClient(echo_responder())
        result = translate_cues(self._cues(), "he", client=client)
        assert result[1]["translated"] == ""
        assert result[0]["translated"] and result[2]["translated"]

    def test_a_missing_id_for_a_context_cue_is_never_an_error(self):
        client = FakeClient(echo_responder())
        result = translate_cues(self._cues(), "he", client=client)
        assert len(result) == 3

    def test_all_context_only_means_no_request_at_all(self):
        client = FakeClient(echo_responder())
        cues = [{"start": 0.0, "end": 1.0, "text": "only context", "context_only": True}]
        result = translate_cues(cues, "he", client=client)
        assert client.calls == []
        assert result[0]["translated"] == ""


# --------------------------------------------------------------------------------------
# R9 — a condensation is a SHORTER version of the cue, or it is not used
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestCondensationValidator:
    """Every case here is a verbatim model reply from the eight-clip corpus archive.

    The pass used to choose between candidates by LENGTH alone, which cannot tell a
    trimmed cue from an amputated one. These are the amputations it shipped.
    """

    def reject(self, original, candidate):
        from services.translation_v2 import _cps_rejection
        return _cps_rejection(original, candidate)

    # --- what must be refused -------------------------------------------------------
    def test_the_adjective_the_cue_turns_on(self):
        """eng_chyron 13. The NEXT cue is the one that says "private"."""
        assert self.reject(
            "אנחנו לא יודעים אם זה היה מטוס מסחרי.",
            "אנחנו לא יודעים אם זה היה מטוס.",
        )

    def test_a_modifier_left_with_nothing_to_modify(self):
        """babe 8: "they are considered very." is not a sentence."""
        assert self.reject(
            "לצה״ל יש מוניטין די קשוח. הם נחשבים מאוד קשוחים.",
            "לצה״ל יש מוניטין קשוח. הם נחשבים מאוד.",
        )

    def test_a_cue_truncated_mid_word_with_its_question_mark_gone(self):
        """trump 21: "איפה אתם?" -> "איפה את"."""
        assert self.reject("איפה אתם?", "איפה את")

    def test_a_question_turned_into_a_statement(self):
        """vertical 46: the cue ASKS "why?" and came back asserting."""
        assert self.reject(
            "אומרים ישו כי זה מה שמכירים. למה?",
            "אומרים ישו כי זה מה שמכירים.",
        )

    def test_the_punchline_of_the_clip(self):
        """vertical 29: the whole exchange exists to arrive at "so it became Parsi"."""
        assert self.reject(
            "כי הערבים לא יכלו להגות את ה-P, אז זה הפך לפארסי.",
            "הערבים לא יכלו להגות את ה-P.",
        )

    def test_the_verbs_reflexive_object(self):
        """street 3: "get yourselves ready" -> "get ready"."""
        assert self.reject(
            "הכינו את עצמכם לקליפים מצחיקים. אם אתם מוכנים, בואו נתחיל.",
            "הכינו לקליפים מצחיקים. אם מוכנים, נתחיל.",
        )

    def test_a_deleted_negation_says_the_opposite(self):
        assert self.reject("זה לא נכון בכלל.", "זה נכון.")

    def test_a_deleted_number_is_a_broadcast_correction(self):
        assert self.reject("שירתתי 15 שנים בצבא.", "שירתתי שנים בצבא.")

    def test_a_cue_ending_on_a_preposition(self):
        assert self.reject("הוא הלך אל הבית שלו.", "הוא הלך אל.")

    def test_a_lost_dialogue_dash(self):
        from services.translation_v2 import DIALOGUE_DASH
        assert self.reject(f"{DIALOGUE_DASH}כן, בוודאי.", "כן, בוודאי.")

    def test_an_empty_reply(self):
        assert self.reject("משהו כאן.", "   ")

    # --- what must be allowed -------------------------------------------------------
    def test_a_redundant_doublet_may_go(self):
        """podcast 5: "screamed and yelled" -> "screamed". Textbook condensation."""
        assert not self.reject(
            "והם צעקו וצרחו אחד על השני.", "והם צעקו אחד על השני."
        )

    def test_leading_filler_may_go(self):
        assert not self.reject("אה, אני שונא את הבחור הזה.", "אני שונא את הבחור הזה.")

    def test_an_intensifier_may_go(self):
        assert not self.reject("זה היה ממש מגעיל.", "זה היה מגעיל.")

    def test_a_trailing_tag_question_may_go(self):
        assert not self.reject(
            "אז אני מנסה לשמור על קור רוח. נכון.", "אני מנסה לשמור על קור רוח."
        )

    def test_a_repeated_interjection_may_go(self):
        assert not self.reject(
            "ניסיתי לגרום לו להכות אותי. היי, היי, היי.",
            "ניסיתי לגרום לו להכות אותי.",
        )

    def test_a_conjunctive_vav_is_not_a_deletion(self):
        """"והם" and "הם" are one word — Hebrew glues the "and" onto the next word."""
        assert not self.reject("וזה עובד, והם צועקים.", "זה עובד, הם צועקים.")

    def test_niqqud_is_not_part_of_the_word(self):
        """A subtitle font does not draw it, so it cannot decide whether a word survived."""
        assert not self.reject("כן, ויכולתי לומר עִברית.", "ויכולתי לומר עברית.")

    def test_an_ellipsis_may_become_a_full_stop(self):
        assert not self.reject("אז הוא אמר לי משהו…", "אז הוא אמר משהו.")


@pytest.mark.unit
class TestCondensationValidatorInThePass:
    """The validator has to reach the cues, not just exist."""

    def _cue(self, text, dur=2.0):
        return [{"start": 0.0, "end": dur, "translated": text}]

    def test_a_refused_batch_reply_is_re_asked_with_the_reason(self):
        replies = ["אנחנו לא יודעים אם זה היה מטוס.", "לא יודעים אם זה מטוס מסחרי."]

        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": replies[call_no - 1]}]}

        client = FakeClient(responder)
        # 1.85s -> floor(17 * 1.85) == 31, the budget this cue really had on the clip.
        original = "אנחנו לא יודעים אם זה היה מטוס מסחרי."
        result = enforce_cps(
            self._cue(original, dur=1.85), client=client, time_relief=False
        )

        assert len(client.calls) == 2
        assert "NOT A CONDENSATION" in client.user_prompt(1)
        assert "מסחרי" in client.user_prompt(1), "the reason names the deleted word"
        assert result[0]["translated"] == "לא יודעים אם זה מטוס מסחרי."

    def test_an_unfixable_cue_keeps_its_original_text(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": "אנחנו לא יודעים אם זה היה מטוס."}]}

        client = FakeClient(responder)
        original = "אנחנו לא יודעים אם זה היה מטוס מסחרי."
        result = enforce_cps(
            self._cue(original, dur=1.85), client=client, time_relief=False
        )

        assert len(client.calls) == 2, "one re-ask, never a loop"
        assert result[0]["translated"] == original, (
            "an over-budget correct cue beats an in-budget amputated one"
        )

    def test_a_reply_inside_the_acceptance_margin_is_not_re_asked(self):
        """The margin that decides whether to TOUCH a cue also decides when to stop.

        babe 8: 40 characters against a 38-character limit. Under R8 that 2-character
        overshoot bought a second request, which came back with "הם נחשבים מאוד."
        """
        from services.translation_v2 import CPS_TRIGGER_MARGIN

        original = "לצה״ל יש מוניטין די קשוח. הם נחשבים מאוד קשוחים."
        reply = "לצה״ל יש מוניטין קשוח. הם נחשבים קשוחים."
        # 2.28s -> floor(17 * 2.28) == 38
        client = FakeClient(lambda kwargs, call_no: {"cues": [{"id": 1, "t": reply}]})
        result = enforce_cps(self._cue(original, dur=2.28), client=client, time_relief=False)

        assert 38 < len(reply) <= 38 * CPS_TRIGGER_MARGIN
        assert len(client.calls) == 1, "a 2-character overshoot must not cost a re-ask"
        assert result[0]["translated"] == reply

    def test_a_tiny_budget_is_never_sent_to_the_model(self):
        """CPS_MIN_BUDGET: 5/5 corpus attempts under 20 characters destroyed content."""
        from services.translation_v2 import CPS_MIN_BUDGET

        client = FakeClient(lambda kwargs, call_no: {"cues": [{"id": 1, "t": "איפה?"}]})
        # 0.45s -> a 7-character budget.
        result = enforce_cps(self._cue("איפה אתם?", dur=0.45), client=client, time_relief=False)

        assert CPS_MIN_BUDGET == 20
        assert client.calls == [], "a sub-second cue has no filler to give"
        assert result[0]["translated"] == "איפה אתם?"

    def test_the_reask_prompt_no_longer_asks_anyone_to_pad_a_cue(self):
        """The TOO SHORT direction: 4 corpus cues in, 0 improvements, 1 destroyed."""
        from services.translation_v2 import _CPS_REASK_PROMPT

        assert "TOO SHORT" not in _CPS_REASK_PROMPT
        assert "TOO LONG" in _CPS_REASK_PROMPT
        assert "NOT A CONDENSATION" in _CPS_REASK_PROMPT

    def test_the_condensation_prompt_protects_the_last_word(self):
        from services.translation_v2 import _CPS_SYSTEM_PROMPT

        assert "NEVER delete the LAST meaningful word" in _CPS_SYSTEM_PROMPT


# --------------------------------------------------------------------------------------
# R9 — a one-cue tail chunk has no context and is not a chunk
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestTailChunkMerge:
    """A 41-cue podcast went out as 40 + 1, and cue 41 — alone, with three English
    context lines and no Hebrew at all — came back in the feminine for a male speaker."""

    def bounds(self, total):
        from services.translation_v2 import _chunk_bounds
        return _chunk_bounds(total)

    def test_a_single_cue_tail_is_folded_into_the_previous_chunk(self):
        assert self.bounds(MAX_CUES_PER_REQUEST + 1) == [(0, MAX_CUES_PER_REQUEST + 1)]

    def test_a_tail_just_under_the_floor_is_folded(self):
        total = MAX_CUES_PER_REQUEST + MIN_TAIL_CUES - 1
        assert self.bounds(total) == [(0, total)]

    def test_a_tail_at_the_floor_stays_its_own_request(self):
        total = MAX_CUES_PER_REQUEST + MIN_TAIL_CUES
        assert self.bounds(total) == [
            (0, MAX_CUES_PER_REQUEST),
            (MAX_CUES_PER_REQUEST, total),
        ]

    def test_the_merged_chunk_is_bounded(self):
        """Merging up is safe; the cap it may reach is stated, not accidental."""
        for total in range(MAX_CUES_PER_REQUEST + 1, MAX_CUES_PER_REQUEST * 4):
            for start, end in self.bounds(total):
                assert end - start <= MAX_CUES_PER_REQUEST + MIN_TAIL_CUES - 1

    def test_every_cue_is_still_covered_exactly_once(self):
        for total in list(range(1, 130)):
            covered = [i for start, end in self.bounds(total) for i in range(start, end)]
            assert covered == list(range(total)), total

    def test_a_short_video_is_still_one_request(self):
        assert self.bounds(MAX_CUES_PER_REQUEST) == [(0, MAX_CUES_PER_REQUEST)]
        assert self.bounds(1) == [(0, 1)]

    def test_the_lone_tail_cue_never_reaches_the_model_alone(self):
        client = FakeClient(echo_responder())
        translate_cues(make_cues(MAX_CUES_PER_REQUEST + 1), "he", client=client)
        assert len(client.calls) == 1
        assert requested_ids(client.user_prompt()) == list(
            range(1, MAX_CUES_PER_REQUEST + 2)
        )


# --------------------------------------------------------------------------------------
# R9 — terminology when transliteration cannot carry the contrast
# --------------------------------------------------------------------------------------
@pytest.mark.unit
class TestTerminologyCollapse:
    """R5's own worked example was self-defeating on the clip it was written for.

    It told the model to transliterate "Ivrit" as עִברית against "Hebrew" as עברית — two
    spellings that a subtitle font renders identically, so the cue still read "they
    didn't call it X, they called it X" on screen. v1 kept "Hebrew" in Latin and was
    right.
    """

    def test_the_collapse_case_has_its_own_instruction(self):
        prompt = build_system_prompt("he")
        assert "WHEN TRANSLITERATION CANNOT CARRY THE CONTRAST" in prompt
        assert "KEEP THE FOREIGN TERM IN LATIN SCRIPT" in prompt

    def test_niqqud_is_named_as_not_a_contrast(self):
        prompt = build_system_prompt("he")
        assert "subtitle fonts do not draw Hebrew niqqud" in prompt
        assert "לא קראו לזה Hebrew. קראו לזה עברית." in prompt

    def test_the_tautology_is_quoted_as_the_failure(self):
        prompt = build_system_prompt("he")
        assert "מה המילה העברית לעברית?" in prompt
        assert "מה המילה העברית ל-Hebrew?" in prompt

    def test_the_phonetic_contrast_that_works_is_untouched(self):
        """פרסי vs פארסי differ without diacritics, so transliteration still carries it."""
        prompt = build_system_prompt("he")
        assert "פארסי vs פרסי" in prompt
        assert "TRANSLITERATE the foreign term" in prompt

    def test_it_applies_to_every_language(self):
        for code in ("he", "ar", "es"):
            assert "KEEP THE FOREIGN TERM IN LATIN SCRIPT" in build_system_prompt(code)


@pytest.mark.unit
class TestProgrammeTitles:
    """"Fear Factor" came back as "פחד גורם" — a programme that does not exist."""

    def test_titles_are_translated_only_when_a_real_one_exists(self):
        prompt = build_system_prompt("he")
        assert "LEAVE THE ORIGINAL TITLE IN LATIN SCRIPT" in prompt
        assert "פחד גורם" in prompt

    def test_the_established_titles_are_given_as_the_positive_example(self):
        prompt = build_system_prompt("he")
        assert "הישרדות" in prompt and "המרוץ למיליון" in prompt


@pytest.mark.unit
class TestFillerYouKnow:
    """"You know, he really got hurt." came back as "אתה יודע, הוא באמת נפגע." in
    CLEAN style, where the whole point is that filler is deleted."""

    def test_the_rule_is_mechanical_not_a_judgement_call(self):
        """The judgement-shaped version of this rule was measured and did not bind."""
        prompt = build_system_prompt("he", "clean")
        assert "MECHANICAL RULE, no judgement required" in prompt
        assert "your translation starts AFTER it" in prompt

    def test_the_opening_phrases_are_listed_with_their_comma(self):
        prompt = build_system_prompt("he", "clean")
        for phrase in ('"you know,"', '"I mean,"', '"like,"', '"well,"', '"listen,"'):
            assert phrase in prompt, phrase

    def test_the_measured_cue_is_the_worked_example(self):
        prompt = build_system_prompt("he", "clean")
        assert '"You know, like, he really got hurt."' in prompt
        assert "אתה יודע, הוא באמת נפגע." in prompt

    def test_the_real_question_keeps_its_verb(self):
        prompt = build_system_prompt("he", "clean")
        assert '"Do you know what that means?" keeps it' in prompt
