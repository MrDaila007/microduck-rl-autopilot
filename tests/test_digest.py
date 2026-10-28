"""LLM digest: units, lineage, curriculum-managed weights. No GPU, no Ollama."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autopilot
import digest

ENV = {
    "decimation": 4,
    "sim": {"mujoco": {"timestep": 0.005}},
    "episode_length_s": 20.0,
    "rewards": {
        "track_linear_velocity": {"func": "mdp.track_linear_velocity", "weight": 2.0},
        "action_rate_l2": {"func": "mdp.action_rate_l2", "weight": -0.05},
    },
    "curriculum": {
        "action_rate_weight": {"func": "mdp.reward_weight",
                               "params": {"reward_name": "action_rate_l2"}},
    },
}


def _m(lo: int, hi: int, ep_len: float, rate_w: float = -0.8) -> autopilot.Metrics:
    its = range(lo, hi + 1)
    return autopilot.Metrics(tags={
        "Train/mean_reward": [(i, 1.0) for i in its],
        "Train/mean_episode_length": [(i, ep_len) for i in its],
        "Episode_Termination/fell_over": [(i, 20.0) for i in its],
        "Episode_Termination/time_out": [(i, 5.0) for i in its],
        # logged = per-episode sum / 20 s: a 4 s episode at kernel 0.8, weight 2
        "Episode_Reward/track_linear_velocity": [(i, 2.0 * 0.8 * 4.0 / 20.0) for i in its],
        "Episode_Reward/action_rate_l2": [(i, -0.1) for i in its],
        "Curriculum/action_rate_weight": [(i, -0.6 if i < hi - 10 else rate_w) for i in its],
    })


def test_units_are_ratios_and_seconds() -> None:
    d = digest.build_digest([_m(0, 100, ep_len=200.0)], ENV, {"target_iterations": 1000}, "monitoring")
    ep = d["episode_now"]
    assert ep["mean_s"] == 4.0
    assert ep["pct_of_horizon"] == 20.0
    assert ep["episodes_ending_in_fall_pct"] == 80.0
    track = next(r for r in d["reward_terms_now"] if r["term"] == "track_linear_velocity")
    assert track["raw_per_s_alive"] == pytest.approx(0.8)


def test_curriculum_weight_is_the_one_in_force() -> None:
    d = digest.build_digest([_m(0, 100, ep_len=200.0)], ENV, {}, "monitoring")
    rate = next(r for r in d["reward_terms_now"] if r["term"] == "action_rate_l2")
    assert rate["weight_cfg"] == -0.05
    assert rate["weight_now"] == pytest.approx(-0.8)
    assert rate["weight_set_by_curriculum"] is True
    assert d["curriculum_events"][-1]["reward_weights_changed"] == ["action_rate_l2"]
    assert any("not comparable" in f for f in d["facts"])


def test_validate_rejects_a_curriculum_managed_weight() -> None:
    cfg = autopilot.Config()
    cfg.safety.allow = ["env.rewards.*.weight"]
    rec = {"verdict": "adjust", "confidence": 0.9,
           "changes": [{"key": "env.rewards.action_rate_l2.weight", "value": -0.05}]}
    ok, errors = autopilot.validate_rec(rec, cfg, frozen_keys=digest.curriculum_managed_keys(ENV))
    assert not ok and any("curriculum" in e for e in errors)


def test_lineage_drops_the_resume_warmup_and_reads_the_old_stretch() -> None:
    old = _m(0, 300, ep_len=400.0)
    new = _m(250, 270, ep_len=50.0)      # 20 iters after resume: still warming up
    merged, resumes = digest.merge_lineage([old, new])
    assert resumes == [250]
    steps = [s for s, _ in merged.series("Train/mean_episode_length")]
    assert max(steps) == 249              # old run past the checkpoint is discarded
    d = digest.build_digest([old, new], ENV, {}, "monitoring")
    assert d["run"]["resume_warmup"] is True
    assert d["episode_now"]["mean_s"] == 8.0   # 400 steps * 0.02 s, from before the resume


def test_run_lineage_follows_pinned_resumes(tmp_path: Path) -> None:
    exp = tmp_path / "velocity"
    for name, agent in (("a", "resume: false\n"),
                        ("b", "resume: true\nload_run: a\n"),
                        ("c", "resume: true\nload_run: b\n")):
        (exp / name / "params").mkdir(parents=True)
        (exp / name / "params" / "agent.yaml").write_text(agent)
    run = autopilot.Run(exp / "c", "velocity", "c")
    assert [r.name for r in autopilot.run_lineage(run)] == ["a", "b", "c"]
