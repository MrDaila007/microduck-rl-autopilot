"""Dashboard inbox, config patch, and human apply — no GPU and no Ollama."""

from __future__ import annotations

import json
import sys
import tomllib
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autopilot
import dashboard


def _cfg() -> autopilot.Config:
    cfg = autopilot.Config()
    cfg.auto.enabled = False
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
    cfg.safety.max_changes = 3
    cfg.monitor.confidence_min = 0.6
    cfg.train.scenarios = {"velocity": {"task": "Mjlab-Velocity-Flat-MicroDuck"}}
    cfg.train.default_scenario = "velocity"
    cfg.dashboard.host = "127.0.0.1"
    cfg.dashboard.port = 0
    return cfg


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(autopilot, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(autopilot, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(autopilot, "RECS_DIR", tmp_path / "recs")
    monkeypatch.setattr(autopilot, "STATE_DIR", tmp_path)
    monkeypatch.setattr(autopilot, "MJLAB_REPO", tmp_path)
    monkeypatch.setattr(autopilot, "LOG_ROOT", tmp_path / "rsl")


def _run_dir(tmp_path: Path) -> autopilot.Run:
    run_dir = tmp_path / "rsl" / "velocity" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "events.out.tfevents.1").write_bytes(b"")
    (run_dir / "model_10.pt").write_bytes(b"ckpt")
    return autopilot.Run(run_dir, "velocity", "run1")


def _state() -> dict:
    return {
        "experiments": {
            "velocity": {
                "budget_used": 2,
                "target_iterations": 5000,
                "interventions": [],
                "post_runs": [],
                "completed_runs": [],
                "last_argv": ["uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck"],
            }
        },
        "last_train_argv": ["uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck"],
        "intervening": False,
    }


def test_overview_from_status_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    autopilot.save_json(autopilot.STATUS_FILE, {
        "phase": "monitoring",
        "status": "rules_keep",
        "verdict": "keep",
        "diagnosis": "quiet",
        "eta_s": 12,
        "sec_per_iter": 1.5,
    })
    autopilot.save_state({"experiments": {}, "intervening": False})
    monkeypatch.setattr(dashboard, "ollama_status", lambda cfg: {"ok": True, "detail": "test"})
    monkeypatch.setattr(autopilot, "find_train_proc", lambda state=None: None)
    rt = dashboard.attach(_cfg())
    try:
        dashboard.serve(rt)
        port = rt.bound_port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/overview", timeout=3) as resp:
            body = json.load(resp)
        assert body["phase"] == "monitoring"
        assert body["verdict"] == "keep"
        assert body["eta_s"] == 12
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as resp:
            page = resp.read().decode()
        assert resp.status == 200
        assert "Автопилот" in page
    finally:
        dashboard.stop_server(rt)
        dashboard.detach()


def test_manual_changes_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _run_dir(tmp_path)
    autopilot.save_state(_state())
    rt = dashboard.attach(_cfg())
    try:
        code, body = dashboard.submit(rt, {
            "type": "manual_adjust",
            "changes": [{"key": "env.observations.foo.scale", "value": 1}],
        })
        assert code == 400
        assert "allowlist" in body["error"]

        code, body = dashboard.submit(rt, {
            "type": "manual_adjust",
            "changes": [{"key": "agent.algorithm.entropy-coef", "value": 9}],
        })
        assert code == 400
        assert "outside" in body["error"]

        code, body = dashboard.submit(rt, {
            "type": "manual_adjust",
            "changes": [{"key": "agent.max-iterations", "value": 100}],
        })
        assert code == 400
        assert "shorten" in body["error"]
        assert len(rt.inbox) == 0
    finally:
        dashboard.detach()


def test_force_apply_relaunches_when_auto_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _run_dir(tmp_path)
    cfg = _cfg()
    cfg.auto.enabled = False
    state = _state()
    spawned: dict = {}

    monkeypatch.setattr(autopilot.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot, "smoke_test", lambda *args, **kwargs: (True, "smoke ok"))
    monkeypatch.setattr(autopilot, "stop_training", lambda *args, **kwargs: None)
    monkeypatch.setattr(autopilot.Metrics, "load", staticmethod(lambda run: None))
    monkeypatch.setattr(
        autopilot, "find_train_proc",
        lambda state=None: {"pid": 4242, "cmdline": state["last_train_argv"] if state else []},
    )

    def fake_spawn(argv, cwd, log_file, extra_env):
        spawned["argv"] = argv
        return 5150

    monkeypatch.setattr(autopilot, "spawn", fake_spawn)
    blocked = autopilot.do_intervention(
        {
            "verdict": "adjust",
            "confidence": 1.0,
            "changes": [{"key": "agent.algorithm.entropy-coef", "value": 0.02}],
            "kind": "manual",
        },
        autopilot.Run(tmp_path / "rsl" / "velocity" / "run1", "velocity", "run1"),
        cfg,
        state,
        phase="monitoring",
        force_apply=False,
    )
    assert blocked["status"] == "ready_to_apply"
    assert "argv" not in spawned

    rt = dashboard.attach(cfg)
    try:
        out = dashboard.execute_manual(
            cfg, state, [{"key": "agent.algorithm.entropy-coef", "value": 0.02}],
        )
    finally:
        dashboard.detach()
    assert out["status"] == "applied"
    assert out["relaunch_pid"] == 5150
    assert spawned["argv"]
    assert state["experiments"]["velocity"]["budget_used"] == 2


def test_toml_patch_keeps_scenario_comment_and_task() -> None:
    real = Path(__file__).resolve().parents[1] / "autopilot.toml"
    text = real.read_text()
    updated = dashboard.apply_toml_updates(text, {
        "auto.enabled": False,
        "train.queue": ["velocity"],
        "monitor.min_sleep_s": 25,
        "monitor.max_sleep_s": 800,
        "safety.ranges": {"env.rewards.*.weight": [-5.0, 5.0]},
    })
    assert "# Locomotion" in updated
    assert 'task = "Mjlab-Velocity-Flat-MicroDuck"' in updated
    loaded = tomllib.loads(updated)
    assert loaded["auto"]["enabled"] is False
    assert loaded["train"]["queue"] == ["velocity"]
    assert loaded["train"]["scenarios"]["velocity"]["task"] == "Mjlab-Velocity-Flat-MicroDuck"
    assert loaded["monitor"]["min_sleep_s"] == 25
    assert loaded["monitor"]["max_sleep_s"] == 800
    assert loaded["safety"]["ranges"]["env.rewards.*.weight"] == [-5.0, 5.0]
    assert "env.rewards.*.weight" in loaded["safety"]["allow"]
    with pytest.raises(ValueError, match="unknown safety range"):
        dashboard.apply_toml_updates(text, {"safety.ranges": {"not.a.key": [0, 1]}})


def test_inbox_drains_one_action_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    cfg = _cfg()
    rt = dashboard.attach(cfg)
    state = {
        "experiments": {
            "exp": {
                "budget_used": 3,
                "post_runs": ["r"],
                "completed_runs": ["r"],
                "interventions": [],
            }
        }
    }
    try:
        code, _body = dashboard.submit(rt, {"type": "reset", "queue": True})
        assert code == 202
        again, busy = dashboard.submit(rt, {"type": "reset", "queue": True})
        assert again == 409
        assert "busy" in busy["error"]

        second = {"id": "second", "type": "reset", "status": "queued", "experiment": "exp", "queue": False}
        rt.actions["second"] = second
        rt.inbox.append(second)
        # the 409 guard is for HTTP; drain_one itself takes a single item
        first = dashboard.drain_one(rt, state)
        assert first is not None
        assert len(rt.inbox) == 1
        assert rt.inbox[0]["id"] == "second"
        second_out = dashboard.drain_one(rt, state)
        assert second_out is not None
        assert rt.inbox == dashboard.deque() or len(rt.inbox) == 0
        assert dashboard.drain_one(rt, state) is None
        assert state["experiments"]["exp"]["budget_used"] == 0
    finally:
        dashboard.detach()


def test_non_loopback_host_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback"):
        dashboard.check_bind_host("0.0.0.0")
    assert dashboard.check_bind_host("localhost") == "127.0.0.1"
