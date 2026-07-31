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
        total = MAX_CUES_PER_REQUEST + 5  # 45 -> chunks of 40 + 5
        result = translate_cues(make_cues(total), "he", client=client)

        assert len(client.calls) == 2

        first_user = client.user_prompt(0)
        second_user = client.user_prompt(1)

        # Each cue is emitted exactly once, across chunks, in order.
        assert requested_ids(first_user) == list(range(1, MAX_CUES_PER_REQUEST + 1))
        assert requested_ids(second_user) == list(range(MAX_CUES_PER_REQUEST + 1, total + 1))

        # ...and each chunk carries OVERLAP_CUES read-only cues on the available sides.
        assert context_ids(first_user) == [41, 42, 43]
        assert context_ids(second_user) == [38, 39, 40]
        assert len(context_ids(first_user)) == OVERLAP_CUES

        # Overlap is context, never re-translated.
        assert not set(context_ids(second_user)) & set(requested_ids(second_user))
        assert CONTEXT_MARKER in first_user and CONTEXT_MARKER in second_user
        assert "do not translate them" in second_user

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
            {"start": 0.0, "end": 2.0, "translated": "א" * 60},
            {"start": 2.0, "end": 8.0, "translated": "ב" * 20},
        ]

    def test_no_request_when_everything_is_within_budget(self):
        client = FakeClient(echo_responder())
        cues = [{"start": 0.0, "end": 5.0, "translated": "שלום עולם."}]
        result = enforce_cps(cues, client=client)

        assert client.calls == []
        assert result[0]["translated"] == "שלום עולם."
        assert result.usage.requests == 0

    def test_only_violators_are_re_asked(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": "מקוצר."} for i in ids]}

        client = FakeClient(responder)
        cues = self._cues()
        result = enforce_cps(cues, client=client)

        assert len(client.calls) == 1, "one batched request, no loop"
        assert requested_ids(client.user_prompt()) == [1]
        user = client.user_prompt()
        assert "max 34 chars" in user, "per-cue budget = floor(max_cps * duration)"
        assert result[0]["translated"] == "מקוצר."
        assert result[1]["translated"] == "ב" * 20, "compliant cue untouched"
        assert cues[0]["translated"] == "א" * 60, "input not mutated"

    def test_char_cap_applies_to_long_duration_cues(self):
        def responder(kwargs, call_no):
            ids = requested_ids(kwargs["messages"][1]["content"])
            return {"cues": [{"id": i, "t": "קצר."} for i in ids]}

        client = FakeClient(responder)
        # 30s duration -> CPS is fine, but 100 chars breaks the 84-char cue cap.
        cues = [{"start": 0.0, "end": 30.0, "translated": "ג" * 100}]
        result = enforce_cps(cues, client=client)

        assert requested_ids(client.user_prompt()) == [1]
        assert f"max {DEFAULT_MAX_CHARS_PER_CUE} chars" in client.user_prompt()
        assert result[0]["translated"] == "קצר."

    def test_still_violating_keeps_the_shorter_version(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": "ד" * 40}]}  # shorter than 60, still > 34

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)

        assert len(client.calls) == 1, "no second attempt"
        assert result[0]["translated"] == "ד" * 40

    def test_longer_reply_is_discarded_in_favour_of_original(self):
        def responder(kwargs, call_no):
            return {"cues": [{"id": 1, "t": "ה" * 120}]}

        client = FakeClient(responder)
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == "א" * 60

    def test_missing_replacement_keeps_original(self):
        client = FakeClient(lambda kwargs, call_no: {"cues": []})
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == "א" * 60

    def test_request_failure_does_not_break_the_job(self):
        client = FakeClient(lambda kwargs, call_no: "}}} not json")
        result = enforce_cps(self._cues(), client=client)
        assert result[0]["translated"] == "א" * 60

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


def cps_violators(count, chars=60, dur=2.0):
    """``count`` cues that all break their reading-speed budget.

    2s at 17 CPS is a 34-character budget, so 60 characters violates it.
    """
    return [
        {"start": i * dur, "end": (i + 1) * dur, "translated": f"{'א' * chars}{i}"}
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
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": "מקוצר."} for i in requested_ids(kwargs["messages"][1]["content"])]
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
        assert all(cue["translated"] == "מקוצר." for cue in result)

    def test_exactly_one_batch_when_the_violators_fit(self):
        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": "קצר."} for i in requested_ids(kwargs["messages"][1]["content"])]
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
                "cues": [{"id": i, "t": "קצר."} for i in requested_ids(kwargs["messages"][1]["content"])]
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

        def responder(kwargs, call_no):
            if call_no == 2:
                return "}}} truncated garbage"
            return {
                "cues": [
                    {"id": i, "t": "מקוצר."}
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
                assert cue["translated"] == "מקוצר.", (
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
        """An over-budget reply is kept only when it is shorter than the original."""

        def responder(kwargs, call_no):
            out = []
            for cue_id in requested_ids(kwargs["messages"][1]["content"]):
                # odd ids: shorter but still over budget; even ids: longer than the input
                out.append({"id": cue_id, "t": "ד" * (40 if cue_id % 2 else 200)})
            return {"cues": out}

        client = FakeClient(responder)
        originals = cps_violators(45)
        result = enforce_cps(originals, client=client)

        for index, cue in enumerate(result, 1):
            if index % 2:
                assert cue["translated"] == "ד" * 40, "the shorter reply should be kept"
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

        client = FakeClient(
            lambda kwargs, call_no: {
                "cues": [{"id": i, "t": "קצר."} for i in requested_ids(kwargs["messages"][1]["content"])]
            }
        )
        result = enforce_cps(cps_violators(3), client=client, progress_callback=exploding)
        assert all(cue["translated"] == "קצר." for cue in result)

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
        from services.translation_v2 import build_system_prompt
        prompt = build_system_prompt("he")
        assert "MASCULINE" in prompt
        assert "mid-conversation" in prompt
