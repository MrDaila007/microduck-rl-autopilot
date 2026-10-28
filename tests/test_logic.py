"""Autopilot decisions that must hold without a GPU or a live training run."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autopilot


def _cfg() -> autopilot.Config:
    cfg = autopilot.Config()
    cfg.safety.allow = [
        "env.rewards.*.weight",
        "agent.max-iterations",
        "agent.algorithm.learning-rate",
        "agent.algorithm.entropy-coef",
        "env.scene.num-envs",
    ]
    cfg.safety.ranges = {
        "env.rewards.*.weight": [-10.0, 10.0],
        "agent.max-iterations": [1, 20000],
        "agent.algorithm.learning-rate": [1e-6, 0.01],
        "agent.algorithm.entropy-coef": [0.0, 0.5],
        "env.scene.num-envs": [64, 32768],
    }
    cfg.monitor.confidence_min = 0.6
    cfg.monitor.cooldown_iters = 10
    cfg.monitor.max_interventions = 6
    cfg.train.smoke_timeout_s = 1200
    cfg.train.scenarios = {
        "velocity": {"task": "Mjlab-Velocity-Flat-MicroDuck"},
    }
    cfg.train.scenario_args = ["--env.scene.num-envs", "64"]
    return cfg


def _adjust(value: float, key: str = "agent.max-iterations") -> dict:
    return {
        "verdict": "adjust",
        "confidence": 0.9,
        "changes": [{"key": key, "value": value, "reason": "test"}],
    }


def _metrics(**series: list[tuple[int, float]]) -> autopilot.Metrics:
    return autopilot.Metrics(tags=series)


def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(autopilot, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(autopilot, "RECS_DIR", tmp_path / "recs")
    monkeypatch.setattr(autopilot, "MJLAB_REPO", tmp_path)


# ---------------------------------------------------------------- extract_json


def test_extract_json_fenced_and_prose() -> None:
    raw = 'sure\n```json\n{"verdict": "keep", "confidence": 1}\n```\n'
    assert autopilot.extract_json(raw) == {"verdict": "keep", "confidence": 1}
    assert autopilot.extract_json("no object here") is None
    assert autopilot.extract_json("{not json} {\"verdict\": \"abort\"}")["verdict"] == "abort"


# ------------------------------------------------------------------ validate


def test_validate_rejects_shorter_horizon() -> None:
    ok, errors = autopilot.validate_rec(_adjust(2000), _cfg(), target_iterations=50000)
    assert not ok
    assert any("shorten" in err for err in errors)


def test_validate_accepts_longer_horizon_inside_range() -> None:
    ok, errors = autopilot.validate_rec(_adjust(8000), _cfg(), target_iterations=2000)
    assert ok, errors


def test_validate_still_enforces_range() -> None:
    ok, errors = autopilot.validate_rec(_adjust(50000), _cfg(), target_iterations=2000)
    assert not ok
    assert any("outside" in err for err in errors)


# -------------------------------------------------------------------- rules


def test_run_rules_nan_is_critical() -> None:
    m = _metrics(**{
        "Episode_Termination/nan_state": [(3, 1.0)],
        "Train/mean_reward": [(3, 1.0)],
        "Train/mean_episode_length": [(3, 20.0)],
    })
    findings = autopilot.run_rules(m, {}, _cfg())
    assert any(f.code == "nan_state" and f.severity == "critical" for f in findings)


def test_run_rules_entropy_collapse() -> None:
    entropy = [(100 + i, 10.0 if i < 5 else 0.1) for i in range(20)]
    m = _metrics(**{
        "Loss/entropy": entropy,
        "Train/mean_reward": [(119, 1.0)],
        "Train/mean_episode_length": [(119, 20.0)],
    })
    findings = autopilot.run_rules(m, {}, _cfg())
    assert any(f.code == "entropy_collapse" for f in findings)


def test_should_consult_llm_only_for_findings_plateau_or_post_run() -> None:
    quiet = autopilot.Finding("unused", "medium", "x")
    assert autopilot.should_consult_llm("monitoring", [], {"flat": False}) is False
    assert autopilot.should_consult_llm("monitoring", [], {"flat": None}) is False
    assert autopilot.should_consult_llm("monitoring", [], {"flat": True}) is True
    assert autopilot.should_consult_llm("monitoring", [quiet], {"flat": False}) is True
    assert autopilot.should_consult_llm("post_run", [], {"flat": False}) is True


def test_rules_keep_does_not_ask_for_changes() -> None:
    m = _metrics(**{"Train/mean_reward": [(i, float(i)) for i in range(1, 6)]})
    rec = autopilot.rules_keep(m, _cfg(), 5)
    assert rec["verdict"] == "keep"
    assert rec["changes"] == []
    assert rec["source"] == "rules"


# ---------------------------------------------------------------------- argv


def test_known_argv_prefers_stored_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    run = autopilot.Run(tmp_path / "run", "velocity", "run")
    state = {"experiments": {"velocity": {"last_argv": ["stored"]}}, "last_train_argv": ["other"]}
    assert autopilot._known_argv(state, run, _cfg()) == ["stored"]


def test_known_argv_from_wandb_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    run_dir = tmp_path / "logs" / "rsl_rl" / "velocity" / "2026-01-01_00-00-00_velocity"
    run_dir.mkdir(parents=True)
    meta = tmp_path / "wandb" / "run-abc" / "files"
    meta.mkdir(parents=True)
    (meta / "output.log").write_text(f"diff in {run_dir.name}/git\n")
    (meta / "wandb-metadata.json").write_text(json.dumps({
        "program": "/venv/bin/train",
        "executable": "/venv/bin/python",
        "args": ["Mjlab-Velocity-Flat-MicroDuck", "--env.scene.num-envs", "4096"],
    }))
    run = autopilot.Run(run_dir, "velocity", run_dir.name)
    state: dict = {"experiments": {}}
    argv = autopilot._known_argv(state, run, _cfg())
    assert argv[:2] == ["/venv/bin/python", "/venv/bin/train"]
    assert "Mjlab-Velocity-Flat-MicroDuck" in argv
    assert state["experiments"]["velocity"]["last_argv"] == argv


def test_known_argv_falls_back_to_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    run = autopilot.Run(tmp_path / "missing", "velocity", "missing")
    state = {"experiments": {}, "last_train_argv": ["train", "Some-Other-Task"]}
    argv = autopilot._known_argv(state, run, _cfg())
    assert argv is not None
    assert "Mjlab-Velocity-Flat-MicroDuck" in argv
    assert "Some-Other-Task" not in argv


def test_wandb_metadata_for_another_run_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    run_dir = tmp_path / "logs" / "rsl_rl" / "velocity" / "this-run"
    run_dir.mkdir(parents=True)
    meta = tmp_path / "wandb" / "run-other" / "files"
    meta.mkdir(parents=True)
    (meta / "output.log").write_text("diff in some-other-run/git\n")
    (meta / "wandb-metadata.json").write_text(json.dumps({
        "program": "/venv/bin/train",
        "executable": "/venv/bin/python",
        "args": ["Other-Task"],
    }))
    run = autopilot.Run(run_dir, "velocity", run_dir.name)
    argv = autopilot._known_argv({"experiments": {}}, run, _cfg())
    assert "Other-Task" not in (argv or [])
    assert "Mjlab-Velocity-Flat-MicroDuck" in argv


# -------------------------------------------------------------------- smoke


def test_fresh_intervening_flag_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    state = {
        "intervening": True,
        "intervening_since": datetime.now(timezone.utc).isoformat(),
    }
    assert autopilot.release_stale_intervening(_cfg(), state) is None
    assert state["intervening"] is True


def test_stale_intervening_closes_smoking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    recs = tmp_path / "recs"
    recs.mkdir()
    path = recs / "open.json"
    path.write_text(json.dumps({"status": "smoking", "verdict": "adjust"}))
    old = (datetime.now(timezone.utc) - timedelta(seconds=5000)).isoformat()
    state = {"intervening": True, "intervening_since": old}
    reason = autopilot.release_stale_intervening(_cfg(), state)
    assert reason and "past smoke_timeout" in reason
    assert state["intervening"] is False
    assert json.loads(path.read_text())["status"] == "smoke_failed"


def test_missing_timestamp_is_treated_as_stuck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    state = {"intervening": True}
    reason = autopilot.release_stale_intervening(_cfg(), state)
    assert reason and "no timestamp" in reason
    assert state["intervening"] is False


def test_settle_closes_old_smoking_without_a_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    recs = tmp_path / "recs"
    recs.mkdir()
    old = recs / "old.json"
    fresh = recs / "fresh.json"
    old.write_text(json.dumps({"status": "smoking"}))
    fresh.write_text(json.dumps({"status": "smoking"}))
    ago = datetime.now().timestamp() - 5000
    old.touch()
    os_utime = __import__("os").utime
    os_utime(old, (ago, ago))
    reason = autopilot.settle_smoke(_cfg(), {"intervening": False})
    assert reason and "closed" in reason
    assert json.loads(old.read_text())["status"] == "smoke_failed"
    assert json.loads(fresh.read_text())["status"] == "smoking"


# -------------------------------------------------------------------- revert


def test_window_worse_requires_both_falls_and_reward() -> None:
    m = _metrics(**{
        "Episode_Termination/fell_over": [(1, 20.0), (2, 20.0)],
        "Train/mean_reward": [(1, 1.0), (2, 1.0)],
    })
    assert autopilot.window_worse(m, {"fell_over": 5.0, "mean_reward": 4.0}, 1, 2)
    assert not autopilot.window_worse(m, {"fell_over": 5.0, "mean_reward": 0.5}, 1, 2)
    assert not autopilot.window_worse(m, {"fell_over": 30.0, "mean_reward": 4.0}, 1, 2)


def test_revertable_changes_drop_a_shorter_horizon() -> None:
    previous = [
        {"key": "agent.algorithm.entropy-coef", "value": 0.01},
        {"key": "agent.max-iterations", "value": 2000},
    ]
    kept = autopilot.revertable_changes(previous, 50000)
    assert kept == [{"key": "agent.algorithm.entropy-coef", "value": 0.01}]


def test_previous_values_read_reward_and_entropy() -> None:
    env = {"rewards": {"track_linear_velocity": {"weight": 2.0}}, "scene": {"num_envs": 4096}}
    agent = {"max_iterations": 50000, "algorithm": {"entropy_coef": 0.01, "learning_rate": 0.001}}
    changes = [
        {"key": "env.rewards.track_linear_velocity.weight", "value": 1.0},
        {"key": "agent.algorithm.entropy-coef", "value": 0.02},
        {"key": "env.observations.foo.scale", "value": 1.0},
    ]
    prev = autopilot.previous_values(changes, env, agent)
    assert {"key": "env.rewards.track_linear_velocity.weight", "value": 2.0} in prev
    assert {"key": "agent.algorithm.entropy-coef", "value": 0.01} in prev
    assert all(row["key"] != "env.observations.foo.scale" for row in prev)


def _pending(iter_at: int, fell: float, reward: float, previous: list[dict] | None = None) -> dict:
    return {
        "kind": "apply",
        "changes": [{"key": "agent.algorithm.entropy-coef", "value": 0.05}],
        "previous": previous if previous is not None else [
            {"key": "agent.algorithm.entropy-coef", "value": 0.01},
        ],
        "snapshot": {"iter": iter_at, "fell_over": fell, "mean_reward": reward},
        "iter": iter_at,
        "outcome": "pending",
    }


def test_revert_marks_ok_when_the_window_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, fell=10.0, reward=1.0)], "budget_used": 0}
    m = _metrics(**{
        "Episode_Termination/fell_over": [(i, 2.0) for i in range(1, 12)],
        "Train/mean_reward": [(i, 5.0) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {"experiments": {}}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is None
    assert exp["interventions"][0]["outcome"] == "ok"


def test_revert_recommends_when_auto_apply_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, fell=1.0, reward=5.0)]}
    m = _metrics(**{
        "Episode_Termination/fell_over": [(i, 9.0) for i in range(1, 12)],
        "Train/mean_reward": [(i, 0.2) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=False)
    assert out is not None
    assert out["status"] == "revert_recommended"
    assert exp["interventions"][0]["outcome"] == "pending"


def test_revert_flags_human_when_nothing_can_be_undone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    inv = _pending(0, fell=1.0, reward=5.0, previous=[
        {"key": "agent.max-iterations", "value": 1000},
    ])
    exp = {"interventions": [inv], "target_iterations": 50000, "budget_used": 0}
    m = _metrics(**{
        "Episode_Termination/fell_over": [(i, 9.0) for i in range(1, 12)],
        "Train/mean_reward": [(i, 0.2) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is not None
    assert out["status"] == "needs_human"
    assert exp["needs_human"] is True
    assert inv["outcome"] == "revert_failed"


def test_second_revert_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    inv = _pending(0, fell=1.0, reward=5.0)
    inv["outcome"] = "reverted"
    inv["reverted_at_iter"] = 0
    exp = {"interventions": [inv]}
    m = _metrics(**{
        "Episode_Termination/fell_over": [(i, 9.0) for i in range(1, 12)],
        "Train/mean_reward": [(i, 0.2) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is not None
    assert out["status"] == "needs_human"
    assert "second" not in out["diagnosis"]
    assert "human" in out["diagnosis"]
