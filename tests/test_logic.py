"""Autopilot decisions that must hold without a GPU or a live training run."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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


def test_window_worse_judges_survival_not_reward() -> None:
    m = _metrics(**{
        "Train/mean_episode_length": [(1, 150.0), (2, 150.0)],
        "Episode_Termination/fell_over": [(1, 20.0), (2, 20.0)],
        "Episode_Termination/time_out": [(1, 1.0), (2, 1.0)],
        "Train/mean_reward": [(1, 9.0), (2, 9.0)],
    })
    # shorter episodes are worse even though the (rescaled) reward went up
    assert autopilot.window_worse(m, {"ep_len_w": 200.0, "fall_share_w": 0.9, "mean_reward": 1.0}, 1, 2)
    # a 5% drop is noise
    assert not autopilot.window_worse(m, {"ep_len_w": 158.0}, 1, 2)
    # fewer falls excuse shorter episodes (more time-outs / resets)
    assert not autopilot.window_worse(m, {"ep_len_w": 200.0, "fall_share_w": 0.99}, 1, 2)
    # old snapshots without window means fall back to the point value
    assert autopilot.window_worse(m, {"mean_episode_length": 200.0}, 1, 2)


def test_fall_share_is_a_ratio_of_counts() -> None:
    m = _metrics(**{
        "Episode_Termination/fell_over": [(1, 20.76)],
        "Episode_Termination/time_out": [(1, 0.39)],
        "Episode_Termination/nan_state": [(1, 0.0)],
    })
    assert autopilot.fall_share(m, 0, 1) == pytest.approx(0.9816, abs=1e-3)


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


def _pending(iter_at: int, ep_len: float, previous: list[dict] | None = None) -> dict:
    return {
        "kind": "apply",
        "changes": [{"key": "agent.algorithm.entropy-coef", "value": 0.05}],
        "previous": previous if previous is not None else [
            {"key": "agent.algorithm.entropy-coef", "value": 0.01},
        ],
        "snapshot": {"iter": iter_at, "ep_len_w": ep_len},
        "iter": iter_at,
        "outcome": "pending",
    }


def test_revert_marks_ok_when_the_window_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, ep_len=100.0)], "budget_used": 0}
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 120.0) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {"experiments": {}}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is None
    assert exp["interventions"][0]["outcome"] == "ok"


def test_revert_recommends_when_auto_apply_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, ep_len=200.0)]}
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 120.0) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=False)
    assert out is not None
    assert out["status"] == "revert_recommended"
    assert exp["interventions"][0]["outcome"] == "pending"


def test_revert_flags_human_when_nothing_can_be_undone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    inv = _pending(0, ep_len=200.0, previous=[
        {"key": "agent.max-iterations", "value": 1000},
    ])
    exp = {"interventions": [inv], "target_iterations": 50000, "budget_used": 0}
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 120.0) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is not None
    assert out["status"] == "needs_human"
    assert exp["needs_human"] is True
    assert inv["outcome"] == "revert_failed"


def _relaunch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                      ckpt: str | None) -> tuple[autopilot.Run, dict, dict]:
    _isolate_state(monkeypatch, tmp_path)
    monkeypatch.setattr(autopilot, "STATE_DIR", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if ckpt:
        (run_dir / ckpt).write_bytes(b"ckpt")
    run = autopilot.Run(run_dir, "velocity", "run")
    m = _metrics(**{
        "Episode_Termination/fell_over": [(0, 40.0), (240, 5.0)],
        "Train/mean_reward": [(0, -2.0), (240, 3.0)],
    })
    monkeypatch.setattr(autopilot.Metrics, "load", lambda _run: m)
    monkeypatch.setattr(autopilot, "find_runs", lambda: [run])
    monkeypatch.setattr(autopilot, "spawn", lambda *a, **k: 4242)
    state: dict = {"experiments": {"velocity": {"target_iterations": 5000}}}
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "phase": "monitoring",
        "snapshot": autopilot.metric_snapshot(m, 240),
    }
    return run, state, record


def test_relaunch_baseline_is_the_resumed_checkpoint(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    run, state, record = _relaunch_fixture(tmp_path, monkeypatch, "model_0.pt")
    changes = [{"key": "agent.algorithm.entropy-coef", "value": 0.05}]
    autopilot._relaunch(run, changes, _cfg(), state, record, tmp_path / "rec.json",
                        {"last_checkpoint_iter": 0}, ["train", "Mjlab-Velocity-Flat-MicroDuck"])
    inv = state["experiments"]["velocity"]["interventions"][-1]
    assert inv["iter"] == 0
    assert inv["stopped_at_iter"] == 240
    assert inv["snapshot"]["iter"] == 0
    assert inv["snapshot"]["fell_over"] == 40.0
    assert inv["snapshot"]["mean_reward"] == -2.0


def test_relaunch_without_checkpoint_has_no_baseline(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    run, state, record = _relaunch_fixture(tmp_path, monkeypatch, None)
    changes = [{"key": "agent.algorithm.entropy-coef", "value": 0.05}]
    autopilot._relaunch(run, changes, _cfg(), state, record, tmp_path / "rec.json",
                        {"last_checkpoint_iter": None}, ["train", "Mjlab-Velocity-Flat-MicroDuck"])
    inv = state["experiments"]["velocity"]["interventions"][-1]
    assert inv["snapshot"] is None
    assert inv["stopped_at_iter"] == 240


def test_second_revert_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    inv = _pending(0, ep_len=200.0)
    inv["outcome"] = "reverted"
    inv["reverted_at_iter"] = 0
    exp = {"interventions": [inv]}
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 120.0) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is not None
    assert out["status"] == "needs_human"
    assert "second" not in out["diagnosis"]
    assert "human" in out["diagnosis"]


# --------------------------------------------------------------------- procs

TRAIN = "/home/u/microduck_rl/.venv/bin/train"

# pid, ppid, cmdline, started. systemd --user -> autopilot daemon -> uv -> trainer -> worker
PROC_TABLE = [
    (100, 1, ["/lib/systemd/systemd", "--user"], 1),
    (200, 100, ["python", "autopilot.py", "run"], 2),
    (300, 200, ["uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck"], 3),
    (400, 300, ["python", TRAIN, "Mjlab-Velocity-Flat-MicroDuck"], 4),
    (500, 400, ["python", "-c", "worker"], 5),
]


def _fake_procs(monkeypatch: pytest.MonkeyPatch, table: list[tuple]) -> None:
    rows = [(pid, ppid, cmd, "/home/u/microduck_rl", started) for pid, ppid, cmd, started in table]
    monkeypatch.setattr(autopilot, "_proc_iter", lambda: iter(rows))


def test_stop_training_interrupts_only_the_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_procs(monkeypatch, PROC_TABLE)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(autopilot.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(autopilot, "_pid_alive", lambda pid: False)
    autopilot.stop_training({"pid": 400}, _cfg(), {})
    assert sent == [(400, autopilot.signal.SIGINT)]


def test_stuck_trainer_escalates_on_its_tree_never_ancestors(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_procs(monkeypatch, PROC_TABLE)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(autopilot.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(autopilot, "_pid_alive", lambda pid: True)
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(autopilot, "time", SimpleNamespace(time=lambda: next(clock),
                                                           sleep=lambda _s: None))
    cfg = _cfg()
    autopilot.stop_training({"pid": 400}, cfg, {})
    assert {pid for pid, _ in sent} == {400, 500}
    assert (500, autopilot.signal.SIGKILL) in sent


def test_find_train_proc_skips_the_smoke_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    table = PROC_TABLE + [
        (600, 200, ["uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck"], 0),
        (601, 600, ["python", TRAIN, "Mjlab-Velocity-Flat-MicroDuck"], 0),
    ]
    _fake_procs(monkeypatch, table)
    proc = autopilot.find_train_proc({"smoke_pids": [600]})
    assert proc is not None and proc["pid"] == 400


# ------------------------------------------------------------- llm pausing


def test_llm_pause_reason() -> None:
    cfg = _cfg()
    assert autopilot.llm_pause_reason(cfg, {"budget_used": 5}) is None
    assert "budget exhausted" in autopilot.llm_pause_reason(cfg, {"budget_used": 6})
    assert "needs_human" in autopilot.llm_pause_reason(cfg, {"needs_human": True})
    cfg.monitor.min_iters_before_intervene = 500
    cfg.monitor.cooldown_iters = 300
    assert "too young" in autopilot.llm_pause_reason(cfg, {}, 100)
    assert "cooldown" in autopilot.llm_pause_reason(cfg, {"last_intervention_iter": 600}, 899)
    assert autopilot.llm_pause_reason(cfg, {"last_intervention_iter": 600}, 900) is None


def test_monitoring_skips_llm_when_budget_is_spent(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    entropy = [(100 + i, 10.0 if i < 5 else 0.1) for i in range(20)]
    m = _metrics(**{
        "Loss/entropy": entropy,
        "Train/mean_reward": [(119, 1.0)],
        "Train/mean_episode_length": [(119, 20.0)],
    })
    monkeypatch.setattr(autopilot.Metrics, "load", lambda _run: m)
    monkeypatch.setattr(autopilot, "run_cfg", lambda _run: ({}, {"max_iterations": 1000}))
    monkeypatch.setattr(autopilot, "find_train_proc", lambda state=None: None)

    def _no_llm(*_a, **_k):
        raise AssertionError("LLM must not be called with the budget spent")

    monkeypatch.setattr(autopilot, "llm_analyze", _no_llm)
    state = {"experiments": {"velocity": {"budget_used": 6}}}
    out = autopilot.monitoring_phase(_cfg(), state, autopilot.Run(tmp_path, "velocity", "r"),
                                     force=False, approve=True)
    assert out["status"] == "llm_paused"
    assert "entropy_collapse" in out["findings"]
    assert out["llm_s"] == 0

    autopilot.cmd_reset(_cfg(), "velocity", queue=False)
    exp = autopilot.load_state()["experiments"]["velocity"]
    assert exp["budget_used"] == 0
    assert "llm_paused" not in exp


# --------------------------------------------------------- revert x curriculum


def test_revert_is_inconclusive_when_curriculum_steps_early(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, ep_len=200.0)], "budget_used": 0}
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 120.0) for i in range(1, 12)],
        "Curriculum/com_range": [(i, 0.005 if i < 3 else 0.01) for i in range(1, 12)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 11, apply=True)
    assert out is None
    assert exp["interventions"][0]["outcome"] == "inconclusive"
    assert "com_range" in exp["interventions"][0]["inconclusive_reason"]


def test_revert_judges_only_the_stretch_before_a_curriculum_step(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_state(monkeypatch, tmp_path)
    exp = {"interventions": [_pending(0, ep_len=200.0)], "budget_used": 0}
    # resume warm-up reads short, then healthy until the stage change at iter 20,
    # where the harder stage shortens episodes
    m = _metrics(**{
        "Train/mean_episode_length": [(i, 40.0 if i < 5 else 210.0 if i < 20 else 50.0)
                                      for i in range(1, 31)],
        "Curriculum/com_range": [(i, 0.005 if i < 20 else 0.01) for i in range(1, 31)],
    })
    out = autopilot.maybe_revert(_cfg(), {}, autopilot.Run(tmp_path, "velocity", "r"),
                                 m, exp, 30, apply=True)
    assert out is None
    assert exp["interventions"][0]["outcome"] == "ok"
