"""
The memory probe that decides whether ``large`` may run, and the record of what ran.

Why this file exists
--------------------
The guard read the wrong number, in two different ways, and both were measured:

* Inside this project's worker container the cgroup limit is 8.00 GiB against 2.98 GiB
  in use — 5.02 GiB of real headroom, below the guard's own 6.0 GiB bar — while
  ``/proc/meminfo`` reported 14.97 GiB ``MemAvailable``, because ``/proc`` describes the
  host VM and not the container the model has to fit inside. So the guard waved ``large``
  through exactly when it should not have.
* On native macOS ``/proc/meminfo`` does not exist at all, so the probe raised
  ``FileNotFoundError`` on every call and its ``except`` branch turned EVERY ``large``
  request into ``medium``, silently. Verified on this machine:
  ``os.path.exists('/proc/meminfo')`` is False.

Downgrades were also invisible to everything downstream: ``model_used`` was computed and
never read by anything, and the stats writer recorded the model that was REQUESTED —
which is why Redis holds 104 records under ``stats:index:model:large`` that cannot be
trusted to describe runs of ``large``.
"""

import os
import sys

import pytest

os.environ["TESTING"] = "true"


def _find_backend_dir():
    for seed in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        path = seed
        while True:
            if os.path.isfile(os.path.join(path, "services", "whisper_smart.py")):
                return path
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    raise RuntimeError("could not locate the backend directory containing services/")


backend_dir = _find_backend_dir()
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np  # noqa: E402

from services import whisper_smart  # noqa: E402
from services.whisper_smart import (  # noqa: E402
    LARGE_MODEL_MIN_FREE_GB,
    SmartWhisperManager,
    _cgroup_memory,
)

GIB = 1024**3


def write_cgroup(tmp_path, limit, current, inactive_file=0, name="v2"):
    """Lay out a fake cgroup directory and return its (limit, current, stat) paths."""
    root = tmp_path / name
    root.mkdir()
    (root / "limit").write_text(str(limit))
    (root / "current").write_text(str(current))
    (root / "stat").write_text(
        f"anon {max(current - inactive_file, 0)}\ninactive_file {inactive_file}\n"
    )
    return str(root / "limit"), str(root / "current"), str(root / "stat")


class FakeInfo:
    language = "en"


class FakeModel:
    def transcribe(self, audio, **kwargs):
        return iter([]), FakeInfo()


@pytest.fixture()
def manager(monkeypatch):
    """A manager whose model loading is stubbed — nothing is downloaded or decoded."""
    instance = SmartWhisperManager()
    loaded = []

    def fake_load_model(model_name):
        loaded.append(model_name)
        instance.last_model_used = model_name
        return FakeModel()

    monkeypatch.setattr(instance, "load_model", fake_load_model)
    instance.loaded = loaded
    return instance


def transcribe(manager, **kwargs):
    """One second of silence through ``transcribe_smart``; returns its result dict."""
    return manager.transcribe_smart(
        np.zeros(16000, dtype=np.float32), language="en", duration=1.0, **kwargs
    )


# ==================================================================================
# 1. the probe reads the container, not the host
# ==================================================================================
@pytest.mark.unit
class TestTheMemoryProbeReadsTheRightNumber:
    def test_a_cgroup_limit_is_read_as_limit_minus_working_set(self, tmp_path):
        paths = write_cgroup(tmp_path, limit=8 * GIB, current=3 * GIB)

        limit_gb, used_gb = _cgroup_memory(*paths)

        assert limit_gb == pytest.approx(8.0)
        assert used_gb == pytest.approx(3.0)

    def test_reclaimable_page_cache_is_not_counted_as_used(self, tmp_path):
        """Measured in the idle worker: 0.39 GiB current, 0.17 GiB of it file cache.

        The kernel hands that back under pressure, so counting it as "in use" would
        understate free memory by nearly half and downgrade `large` for no reason.
        """
        paths = write_cgroup(
            tmp_path,
            limit=8 * GIB,
            current=3 * GIB,
            inactive_file=1 * GIB,
        )

        _limit_gb, used_gb = _cgroup_memory(*paths)

        assert used_gb == pytest.approx(2.0)

    def test_an_unlimited_cgroup_is_not_a_limit(self, tmp_path):
        """cgroup v1 spells "no limit" as a near-2**63 sentinel; v2 writes "max"."""
        v1 = write_cgroup(
            tmp_path, limit=9223372036854771712, current=3 * GIB, name="v1"
        )
        assert _cgroup_memory(*v1) is None

        root = tmp_path / "v2max"
        root.mkdir()
        (root / "limit").write_text("max\n")
        (root / "current").write_text(str(3 * GIB))
        (root / "stat").write_text("")
        assert (
            _cgroup_memory(
                str(root / "limit"), str(root / "current"), str(root / "stat")
            )
            is None
        )

    def test_missing_cgroup_files_are_not_an_error(self, tmp_path):
        assert (
            _cgroup_memory(
                str(tmp_path / "nope"), str(tmp_path / "nope2"), str(tmp_path / "nope3")
            )
            is None
        )

    def test_the_probe_answers_unknown_rather_than_guessing(self, monkeypatch):
        """The macOS case: no cgroup files, no /proc/meminfo, nothing to read."""
        monkeypatch.setattr(whisper_smart, "_cgroup_memory", lambda *_a: None)

        def no_proc(path, *args, **kwargs):
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", no_proc)

        headroom, source = whisper_smart.memory_headroom_gb()

        assert headroom is None
        assert "no cgroup" in source and "meminfo" in source


# ==================================================================================
# 2. what the guard does with that number
# ==================================================================================
@pytest.mark.unit
class TestTheLargeModelGuard:
    def test_the_measured_container_case_now_downgrades(self, manager, monkeypatch):
        """8.00 GiB limit, 2.98 GiB in use = 5.02 GiB free, under the 6.0 GiB bar.

        This is the exact state the worker was measured in while ``/proc/meminfo`` was
        reporting 14.97 GiB and the guard was letting `large` through.
        """
        monkeypatch.setattr(
            whisper_smart,
            "memory_headroom_gb",
            lambda: (8.00 - 2.98, "cgroup limit 8.00 GiB, 2.98 GiB in use"),
        )

        result = transcribe(manager, model_preference="large")

        assert result["model_used"] == "medium"
        assert manager.loaded == ["medium"]

    def test_enough_headroom_runs_large(self, manager, monkeypatch):
        monkeypatch.setattr(
            whisper_smart,
            "memory_headroom_gb",
            lambda: (LARGE_MODEL_MIN_FREE_GB + 0.5, "cgroup"),
        )

        assert transcribe(manager, model_preference="large")["model_used"] == "large"

    def test_unknown_headroom_honours_the_request(self, manager, monkeypatch, caplog):
        """UNKNOWN is not LOW.

        This is the branch that fired on every native-macOS run and silently returned
        `medium` output for a `large` request.
        """
        monkeypatch.setattr(
            whisper_smart,
            "memory_headroom_gb",
            lambda: (None, "no cgroup memory limit and no readable /proc/meminfo"),
        )

        with caplog.at_level("WARNING", logger="services.whisper_smart"):
            result = transcribe(manager, model_preference="large")

        assert result["model_used"] == "large"
        assert any(
            "Cannot measure free memory" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        ), f"the guard gave up silently: {[r.message for r in caplog.records]}"

    def test_low_memory_downgrade_is_a_warning_naming_both_models(
        self, manager, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            whisper_smart, "memory_headroom_gb", lambda: (1.0, "cgroup")
        )

        with caplog.at_level("WARNING", logger="services.whisper_smart"):
            transcribe(manager, model_preference="large")

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "requested 'large'" in message and "running 'medium'" in message
            for message in warnings
        ), f"a downgrade nobody can see in the logs: {warnings}"


# ==================================================================================
# 3. every downgrade is loud, and every one of them is visible downstream
# ==================================================================================
@pytest.mark.unit
class TestDowngradesAreLoudAndRecorded:
    def test_small_is_mapped_to_medium_with_a_warning(self, manager, caplog):
        with caplog.at_level("WARNING", logger="services.whisper_smart"):
            result = transcribe(manager, model_preference="small")

        assert result["model_used"] == "medium"
        assert any(
            "requested 'small'" in r.message and "running 'medium'" in r.message
            for r in caplog.records
            if r.levelname == "WARNING"
        )

    def test_a_failed_transcription_falls_back_to_base_with_a_warning(
        self, monkeypatch, caplog
    ):
        instance = SmartWhisperManager()
        attempts = []

        class ExplodingModel:
            def transcribe(self, audio, **kwargs):
                raise RuntimeError("ct2 allocation failed")

        def fake_load_model(model_name):
            attempts.append(model_name)
            instance.last_model_used = model_name
            return ExplodingModel() if model_name != "base" else FakeModel()

        monkeypatch.setattr(instance, "load_model", fake_load_model)
        monkeypatch.setattr(whisper_smart, "memory_headroom_gb", lambda: (99.0, "test"))

        with caplog.at_level("WARNING", logger="services.whisper_smart"):
            result = transcribe(instance, model_preference="large")

        assert attempts == ["large", "base"]
        assert result["model_used"] == "base"
        assert any(
            "requested 'large'" in r.message and "running 'base'" in r.message
            for r in caplog.records
            if r.levelname == "WARNING"
        )

    def test_the_manager_publishes_the_model_that_actually_ran(
        self, manager, monkeypatch
    ):
        """``last_model_used`` is what the stats writer records instead of the request.

        Before this existed, ``model_used`` was computed in the result dict and read by
        nothing at all (grep: one hit, its own assignment), and the two entry points in
        transcription_service that call ``load_model`` directly never produced it.
        """
        monkeypatch.setattr(
            whisper_smart, "memory_headroom_gb", lambda: (1.0, "cgroup")
        )

        transcribe(manager, model_preference="large")

        assert manager.last_model_used == "medium"

    def test_loading_a_model_is_what_sets_it(self, monkeypatch):
        """The choke point both transcription_service paths go through."""
        instance = SmartWhisperManager()
        monkeypatch.setattr(whisper_smart, "WhisperModel", lambda *a, **k: FakeModel())

        assert instance.last_model_used is None
        instance.load_model("medium")
        assert instance.last_model_used == "medium"
