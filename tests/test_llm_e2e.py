"""End-to-end: the local LLM reads a real analysis payload and names the metrics.

Does not smoke-test, stop, or relaunch training. Run with the mjlab venv so
tensorboard can read tfevents:

    ../microduck_rl/.venv/bin/python tests/test_llm_e2e.py

Skipped by the default unit run (`pytest -m 'not e2e'`).
"""

from __future__ import annotations

import copy
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autopilot

try:
    import pytest
except ImportError:  # the mjlab venv has tensorboard but not pytest
    class _Mark:
        def e2e(self, fn):
            return fn

    class pytest:  # type: ignore
        mark = _Mark()


E2E_DIR = autopilot.STATE_DIR / "e2e"


def _ollama_up(cfg: autopilot.Config) -> bool:
    try:
        with urllib.request.urlopen(cfg.llm.base_url.rstrip("/") + "/api/version", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _blob(rec: dict) -> str:
    parts = [str(rec.get("diagnosis") or ""), str(rec.get("notes") or "")]
    evidence = rec.get("evidence") or []
    if isinstance(evidence, list):
        parts.extend(str(item) for item in evidence)
    for ch in rec.get("changes") or []:
        if isinstance(ch, dict):
            parts.append(str(ch.get("key") or ""))
            parts.append(str(ch.get("reason") or ""))
    return "\n".join(parts).lower()


def _assert_schema(rec: dict) -> None:
    assert isinstance(rec, dict), rec
    assert rec.get("verdict") in autopilot.VERDICTS, rec
    conf = rec.get("confidence")
    assert isinstance(conf, (int, float)) and not isinstance(conf, bool), rec
    assert 0.0 <= float(conf) <= 1.0, rec
    assert isinstance(rec.get("diagnosis"), str) and rec["diagnosis"].strip(), rec
    assert isinstance(rec.get("evidence"), list) and rec["evidence"], rec
    assert all(isinstance(item, str) and item.strip() for item in rec["evidence"])
    changes = rec.get("changes")
    assert isinstance(changes, list), rec


def _assert_changes_allowed(rec: dict, cfg: autopilot.Config, target: int | None) -> None:
    if rec.get("verdict") != "adjust":
        return
    ok, errors = autopilot.validate_rec(rec, cfg, target_iterations=target)
    assert ok, errors


def _save(tag: str, payload: dict, rec: dict) -> Path:
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    path = E2E_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{tag}.json"
    autopilot.save_json(path, {"tag": tag, "payload": payload, "rec": rec})
    return path


def _active_run(cfg: autopilot.Config) -> autopilot.Run:
    state = autopilot.load_state()
    runs = [r for r in autopilot.find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    assert runs, "no training run with tfevents"
    current = state.get("current_run")
    if current:
        for run in runs:
            if str(run.dir) == current:
                return run
    fresh = [r for r in runs if r.age_s <= cfg.monitor.active_after_s]
    return fresh[-1] if fresh else runs[-1]


def _sick_payload(cfg: autopilot.Config) -> dict:
    """A finished-looking velocity window with one sign bug and a farmed pose term.

    Numbers are invented on purpose. The model must name these terms, not the
    live run's.
    """
    return {
        "phase": "monitoring",
        "task_context": {
            "experiment": "velocity",
            "run_name": "e2e-sick-fixture",
            "num_envs": 4096,
            "max_iterations_in_cfg": 50000,
            "experiment_target_iterations": 50000,
        },
        "analysis": {
            "phase": "monitoring",
            "iterations": {"first": 0, "last": 5000, "count": 5001},
            "plateau": {
                "window": [4901, 5000], "n": 100, "first": 2.1, "last": 2.12,
                "mean": 2.11, "slope": 0.0001, "rel_gain": 0.009, "flat": True,
            },
            "trends_last_vs_prev_window": {
                "Train/mean_reward": {"mean": 2.11, "last": 2.12, "slope": 0.0001, "rel_vs_prev_window": 0.005},
                "Episode_Termination/fell_over": {"mean": 40.0, "last": 42.0, "slope": 0.2, "rel_vs_prev_window": 0.05},
                "Train/mean_episode_length": {"mean": 70.0, "last": 68.0, "slope": -0.1},
                "Loss/entropy": {"mean": 12.0, "last": 11.8, "slope": -0.01},
            },
            "penalty_audit": [{
                "term": "action_rate_l2",
                "weight": -0.05,
                "mean": 0.42,
                "max": 0.83,
                "status": "VIOLATION",
            }],
            "reward_mass": {
                "total_positive": 10.0,
                "top": [
                    {"term": "upright", "weight": 1.0, "mean": 6.0, "share": 0.6},
                    {"term": "head_pose_tracking", "weight": 1.0, "mean": 3.0, "share": 0.3},
                    {"term": "track_linear_velocity", "weight": 2.0, "mean": 1.0, "share": 0.1},
                ],
            },
            "curriculum": {"stage_changes": [], "pacing_warnings": []},
            "health": {
                "window": [4950, 5000],
                "fell_over": 40.0,
                "mean_episode_length": 70.0,
                "entropy": 12.0,
                "entropy_frac_of_max": 0.7,
            },
            "progress": {
                "target": 50000,
                "checkpoints": [4950],
                "last_checkpoint_iter": 4950,
                "remaining_to_target": 45050,
            },
            "recent_points": {},
        },
        "findings": [
            {
                "code": "penalty_sign",
                "severity": "high",
                "detail": "Episode_Reward/action_rate_l2 max=0.83 (weight=-0.05) > 0.01",
            },
            {
                "code": "stagnation",
                "severity": "medium",
                "detail": "mean_reward 2.1 -> 2.12 over 100 (rel gain 0.009)",
            },
        ],
        "rewards_cfg": [
            {"term": "track_linear_velocity", "weight": 2.0, "func": "track_linear_velocity"},
            {"term": "upright", "weight": 1.0, "func": "upright"},
            {"term": "head_pose_tracking", "weight": 1.0, "func": "head_pose_tracking"},
            {"term": "action_rate_l2", "weight": -0.05, "func": "action_rate_l2"},
        ],
        "curriculum_terms": [],
        "intervention_history": [],
        "allowlist": {
            "keys": cfg.safety.allow,
            "ranges": cfg.safety.ranges,
            "max_changes": cfg.safety.max_changes,
        },
    }


@pytest.mark.e2e
def test_llm_names_the_broken_metrics() -> None:
    cfg = autopilot.load_config()
    if not _ollama_up(cfg):
        raise AssertionError(f"ollama unreachable at {cfg.llm.base_url}")
    payload = _sick_payload(cfg)
    rec = autopilot.llm_analyze(cfg, payload)
    assert rec is not None, "LLM returned nothing parseable"
    _assert_schema(rec)
    _assert_changes_allowed(rec, cfg, target=50000)
    text = _blob(rec)
    assert "action_rate_l2" in text, rec
    assert any(term in text for term in ("upright", "track_linear_velocity", "fell_over")), rec
    if rec["verdict"] == "adjust":
        keys = [autopilot._change_key(ch) for ch in rec["changes"]]
        assert keys, rec
        assert any(
            key.startswith("env.rewards.") and any(
                term in key for term in ("action_rate_l2", "upright", "track_linear_velocity", "head_pose_tracking")
            )
            for key in keys
        ), rec
    path = _save("sick", payload, rec)
    print(f"sick rec -> {path}")
    print(json.dumps({k: rec.get(k) for k in ("verdict", "confidence", "diagnosis", "evidence", "changes", "notes")}, indent=2))


@pytest.mark.e2e
def test_llm_reads_the_live_training_log() -> None:
    cfg = autopilot.load_config()
    if not _ollama_up(cfg):
        raise AssertionError(f"ollama unreachable at {cfg.llm.base_url}")
    try:
        import tensorboard  # noqa: F401  — EventAccumulator, imported by Metrics.load
    except ImportError:
        if hasattr(pytest, "importorskip"):
            pytest.importorskip("tensorboard")
        raise AssertionError("tensorboard is required to read the training log")
    run = _active_run(cfg)
    state = copy.deepcopy(autopilot.load_state())
    env_cfg, agent_cfg = autopilot.run_cfg(run)
    metrics = autopilot.Metrics.load(run)
    assert metrics is not None and metrics.iters >= 1, run.dir
    findings = autopilot.run_rules(metrics, env_cfg, cfg)
    payload = autopilot.build_payload(
        run, metrics, env_cfg, agent_cfg, findings, state, cfg, "monitoring",
    )
    rec = autopilot.llm_analyze(cfg, payload)
    assert rec is not None, "LLM returned nothing parseable"
    _assert_schema(rec)
    _assert_changes_allowed(rec, cfg, target=state.get("experiments", {}).get(run.experiment, {}).get("target_iterations"))
    text = _blob(rec)
    names = ["mean_reward", "fell_over", "entropy", "episode_length"]
    names += [row["term"] for row in (payload["analysis"]["reward_mass"].get("top") or [])]
    assert any(name.lower() in text for name in names), rec
    if not any(f.severity == "critical" for f in findings):
        assert rec["verdict"] != "abort", rec
    path = _save("live", payload, rec)
    print(f"live run={run.dir.name} iter={metrics.iters} findings={[f.code for f in findings]}")
    print(f"live rec -> {path}")
    print(json.dumps({k: rec.get(k) for k in ("verdict", "confidence", "diagnosis", "evidence", "changes", "notes")}, indent=2))


if __name__ == "__main__":
    failed = 0
    for fn in (test_llm_names_the_broken_metrics, test_llm_reads_the_live_training_log):
        print(f"\n=== {fn.__name__} ===")
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
