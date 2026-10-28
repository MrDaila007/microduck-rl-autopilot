#!/usr/bin/env python3
"""Microduck RL training autopilot.

A wake/sleep cycle around a local mjlab/rsl_rl run:

  wake -> is a run active?
    yes -> read tfevents, run the precomputed analytics. A healthy run is
           recorded as keep by the rules engine (no LLM). Findings, a plateau,
           or a finished run ask the local LLM (Ollama):
                     keep?   -> sleep
                     adjust? -> smoke-test the change, stop, resume from
                                the latest checkpoint with it, sleep.
                                One later window can revert that change.
                     abort?  -> stop for human review
    no  -> run ended: analyze the finished run, plan the next step with the
           LLM (done / continue / continue-with-changes), relaunch if it says
           so, then sleep. When nothing has ever run, optionally launch the
           configured training command.

The heavy lifting is done by the analytics block below so the local LLM only
has to judge and plan, not compute. See docs/autopilot.md.

The [train.scenarios] table in autopilot.toml maps friendly names to any
task in the mjlab registry (velocity, velstand, standup, sitstand,
groundpick, ballkick, rollers, swizzle, roller_*, spin, roulade + their
-Backlash- twins). auto.start_if_idle walks the optional [train] queue
through them, one completed scenario at a time.

Commands:
  status                     detected runs, state, queue, last recommendation
  scenarios                  list configured training scenarios + queue
  suggest [--force] [--approve]   one-shot analysis of the ACTIVE run
  plan [--force]             one-shot post-run planning for the newest ended run
  run                        daemon loop (systemd: microduck-rl-autopilot)
  apply REC.json             validate + smoke-test + restart for a saved rec
  launch [SCENARIO]          start a named scenario (or default/train.command)
  reset [--experiment X] [--queue]  clear budget/completed marks (+ queue)
  dashboard                  local web UI (default http://127.0.0.1:8765)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This repo (autopilot code + its own state/reports). The training project is
# [paths].mjlab_repo: tfevents are written there and spawned training runs
# with that cwd.
AUTOPILOT_DIR = Path(__file__).resolve().parent
STATE_DIR = AUTOPILOT_DIR / "logs"
RECS_DIR = STATE_DIR / "recommendations"
STATE_FILE = STATE_DIR / "state.json"
STATUS_FILE = STATE_DIR / "status.json"
CONFIG_FILE = AUTOPILOT_DIR / "autopilot.toml"
REPORTS_DIR = AUTOPILOT_DIR / "reports"

# mjlab repo the autopilot supervises; load_config() redirects both when
# [paths].mjlab_repo is set in autopilot.toml.
MJLAB_REPO = AUTOPILOT_DIR.parent / "microduck_rl"
LOG_ROOT = MJLAB_REPO / "logs" / "rsl_rl"

log = logging.getLogger("autopilot")

VERDICTS = ("keep", "adjust", "abort")


# --------------------------------------------------------------------------- config


@dataclass
class LlmCfg:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:35b-a3b-q4_K_M"
    max_tokens: int = 3000
    timeout_s: int = 300
    temperature: float = 0.2


@dataclass
class MonitorCfg:
    tick_s: int = 60
    analyze_every_iters: int = 5
    active_after_s: int = 300
    min_iters_before_intervene: int = 50
    stagnation_window: int = 100
    stagnation_min_iters: int = 200
    stagnation_rel_gain: float = 0.02
    cooldown_iters: int = 10
    max_interventions: int = 6
    confidence_min: float = 0.6
    post_run_log_lines: int = 30
    # Wake scheduling: while training runs, sleep is computed from the measured
    # iteration rate -> next check phase (ETA-aware), clamped to [min, max].
    min_sleep_s: int = 20
    max_sleep_s: int = 900
    # Try to schedule a check right after a checkpoint save (iter % N == 0) so
    # an intervention loses fewer iterations (rsl_rl only saves every N).
    align_checks_to_checkpoints: bool = True


@dataclass
class SafetyCfg:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    ranges: dict[str, list[float]] = field(default_factory=dict)
    max_changes: int = 3
    penalty_eps: float = 0.01


@dataclass
class TrainCfg_:
    command: str = ""
    default_scenario: str = ""
    scenario_args: list[str] = field(default_factory=lambda: ["--env.scene.num-envs", "4096"])
    scenarios: dict[str, dict] = field(default_factory=dict)
    queue: list[str] = field(default_factory=list)
    smoke_envs: int = 64
    smoke_iters: int = 5
    smoke_timeout_s: int = 1200
    stop_grace_s: int = 180
    save_interval: int = 50
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class AutoCfg:
    enabled: bool = True
    start_if_idle: bool = False


@dataclass
class PathsCfg:
    mjlab_repo: str = ""


@dataclass
class DashboardCfg:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class Config:
    llm: LlmCfg = field(default_factory=LlmCfg)
    monitor: MonitorCfg = field(default_factory=MonitorCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    train: TrainCfg_ = field(default_factory=TrainCfg_)
    auto: AutoCfg = field(default_factory=AutoCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    dashboard: DashboardCfg = field(default_factory=DashboardCfg)


def load_config(path: Path = CONFIG_FILE) -> Config:
    global MJLAB_REPO, LOG_ROOT
    cfg = Config()
    if not path.exists():
        return cfg
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    for section, cls in (
        ("llm", LlmCfg),
        ("monitor", MonitorCfg),
        ("safety", SafetyCfg),
        ("train", TrainCfg_),
        ("auto", AutoCfg),
        ("paths", PathsCfg),
        ("dashboard", DashboardCfg),
    ):
        if section in raw:
            base = getattr(cfg, section)
            for key, val in raw[section].items():
                if hasattr(base, key):
                    setattr(base, key, val)
    if cfg.paths.mjlab_repo:
        MJLAB_REPO = Path(cfg.paths.mjlab_repo).expanduser()
        LOG_ROOT = MJLAB_REPO / "logs" / "rsl_rl"
    return cfg


# --------------------------------------------------------------------------- state


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def load_state() -> dict:
    return load_json(
        STATE_FILE,
        {
            "experiments": {},
            "current_run": None,
            "ignore_runs": [],
            "intervening": False,
            "intervening_since": None,
            "last_train_argv": None,
        },
    )


def save_state(state: dict) -> None:
    save_json(STATE_FILE, state)


def exp_state(state: dict, experiment: str) -> dict:
    exp = state.setdefault("experiments", {}).setdefault(
        experiment, {"interventions": [], "post_runs": [], "completed_runs": []}
    )
    exp.setdefault("interventions", [])
    exp.setdefault("post_runs", [])
    exp.setdefault("completed_runs", [])
    return exp


def wandb_has_creds() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return "api.wandb.ai" in (Path.home() / ".netrc").read_text()
    except OSError:
        return False


def train_env(cfg: Config) -> dict[str, str]:
    """Env for spawned training. Offline only when wandb has no credentials —
    with ~/.netrc present (stats dashboard), runs log online."""
    env = dict(cfg.train.env)
    if "WANDB_MODE" not in env and not wandb_has_creds():
        env["WANDB_MODE"] = "offline"
    return env


# ----------------------------------------------------------------------- scenarios
#
# The mjlab registry holds the whole policy family (velocity, velstand,
# standup, sitstand, groundpick, ballkick, rollers, swizzle, roller_*,
# spin, roulade + their -Backlash- twins). autopilot.toml maps friendly
# names to task ids so ONE autopilot can supervise any of them and walk a
# training queue (velocity -> velstand -> ...) while idle.


def scenario_spec(cfg: Config, name: str) -> dict | None:
    return (cfg.train.scenarios or {}).get(name)


def scenario_launch(cfg: Config, name: str) -> tuple[list[str], dict[str, str]] | None:
    """argv + extra env for a named scenario, or None if unknown/broken."""
    spec = scenario_spec(cfg, name)
    if not spec or not spec.get("task"):
        return None
    args = [str(a) for a in (spec.get("args") or cfg.train.scenario_args)]
    argv = ["uv", "run", "train", str(spec["task"]), *args]
    if spec.get("max_iterations"):
        argv += ["--agent.max-iterations", str(int(spec["max_iterations"]))]
    env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
    warm = spec.get("warm_start_from")
    if warm:
        # Warm start != resume: load weights/normalizer/optimizer but restart
        # step-based curricula (AGENTS.md) — no --agent.resume, and
        # MICRODUCK_WARM_START=1 tells mjlab to zero the step counter.
        run_dir = Path(warm)
        if not run_dir.is_absolute():
            run_dir = MJLAB_REPO / run_dir
        ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            log.error("scenario %s: no checkpoint in %s", name, run_dir)
            return None
        argv += ["--agent.load-run", str(run_dir),
                 "--agent.load-checkpoint", ckpts[-1].name]
        env.setdefault("MICRODUCK_WARM_START", "1")
    return argv, env


def queue_next(cfg: Config, state: dict) -> str | None:
    """First queued scenario that has not completed (and was not aborted)."""
    q = state.get("queue") or {}
    done = set(q.get("completed", []))
    aborted = set(q.get("aborted", []))
    for name in cfg.train.queue or []:
        if name not in done and name not in aborted and scenario_spec(cfg, name):
            return name
    return None


def queue_note_abort(cfg: Config, state: dict, argv: list[str] | None) -> None:
    """An abort verdict on a queued scenario: park it (human decides)."""
    if not argv or not cfg.train.queue:
        return
    for name in cfg.train.queue:
        spec = scenario_spec(cfg, name)
        if spec and spec.get("task") and str(spec["task"]) in argv:
            q = state.setdefault("queue", {"completed": []})
            q.setdefault("aborted", [])
            if name not in q["aborted"]:
                q["aborted"].append(name)
            if q.get("current") == name:
                q["current"] = None
            save_state(state)
            log.warning("queue: scenario %s aborted — will not auto-relaunch "
                        "(reset --queue to re-enable)", name)
            return


def queue_note_completion(cfg: Config, state: dict, argv: list[str] | None) -> None:
    """A run finished — if it was launched for a queued scenario, mark it done."""
    if not argv or not cfg.train.queue:
        return
    for name in cfg.train.queue:
        spec = scenario_spec(cfg, name)
        if spec and spec.get("task") and str(spec["task"]) in argv:
            q = state.setdefault("queue", {"completed": []})
            if name not in q.setdefault("completed", []):
                q["completed"].append(name)
            if q.get("current") == name:
                q["current"] = None
            save_state(state)
            log.info("queue: scenario %s completed (%d/%d done)",
                     name, len(q["completed"]), len(cfg.train.queue))
            return


# --------------------------------------------------------------------------- runs


@dataclass
class Run:
    dir: Path
    experiment: str
    name: str

    @property
    def tfevents(self) -> list[Path]:
        return sorted(self.dir.glob("events.out.tfevents.*"))

    @property
    def newest_tfevents(self) -> Path | None:
        ev = self.tfevents
        return ev[-1] if ev else None

    @property
    def age_s(self) -> float:
        ev = self.newest_tfevents
        return time.time() - (ev.stat().st_mtime if ev else self.dir.stat().st_mtime)

    def checkpoint_iter(self) -> int | None:
        best = None
        for p in self.dir.glob("model_*.pt"):
            m = re.fullmatch(r"model_(\d+)\.pt", p.name)
            if m:
                n = int(m.group(1))
                best = n if best is None else max(best, n)
        return best

    def latest_checkpoint(self) -> Path | None:
        n = self.checkpoint_iter()
        if n is None:
            return None
        p = self.dir / f"model_{n}.pt"
        return p if p.exists() else None


def find_runs() -> list[Run]:
    runs: list[Run] = []
    if not LOG_ROOT.exists():
        return runs
    for exp_dir in LOG_ROOT.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not list(run_dir.glob("events.out.tfevents.*")):
                continue
            runs.append(Run(run_dir, exp_dir.name, run_dir.name))
    runs.sort(key=lambda r: r.newest_tfevents.stat().st_mtime if r.newest_tfevents else 0)
    return runs


def load_yaml(path: Path) -> dict:
    """Load mjlab's yaml dumps safely despite their python-specific tags.

    `!!python/name:pkg.mod.func` keeps the function path (needed to classify
    penalty functions), `!!python/object/apply:...` / `!!python/tuple` become
    inert placeholders. Nothing is imported, nothing raises.
    """
    import yaml

    class TolerantLoader(yaml.SafeLoader):
        pass

    def _python_tag(loader: yaml.Loader, suffix: str, node: yaml.Node) -> Any:
        if suffix.startswith("name:"):
            return suffix[len("name:"):]
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return f"<{suffix}>"

    TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)
    data = yaml.load(path.read_text(), Loader=TolerantLoader)
    return data if isinstance(data, dict) else {}


def run_cfg(run: Run) -> tuple[dict, dict]:
    env_cfg: dict = {}
    agent_cfg: dict = {}
    # agent.yaml first: it drives target/iteration logic and must survive an
    # env.yaml surprise (and vice versa).
    for filename, dest in (("agent.yaml", "agent"), ("env.yaml", "env")):
        p = run.dir / "params" / filename
        if not p.exists():
            continue
        try:
            loaded = load_yaml(p)
        except Exception as exc:  # a yaml surprise must not kill the daemon
            log.warning("cfg parse failed for %s: %s", p, exc)
            continue
        if dest == "agent":
            agent_cfg = loaded
        else:
            env_cfg = loaded
    return env_cfg, agent_cfg


def reward_table(env_cfg: dict) -> list[dict]:
    out = []
    for term, cfg in (env_cfg.get("rewards") or {}).items():
        if not isinstance(cfg, dict):
            continue
        func = str(cfg.get("func", ""))
        leaf = func.rsplit(".", 1)[-1].strip("'") if func else ""
        out.append({"term": term, "weight": cfg.get("weight"), "func": leaf})
    return out


PENALTY_FUNC_RE = re.compile(r"penalty|_l1$|_l1_|^cost|impact|violation", re.I)


def penalty_expect_le_zero(row: dict) -> bool:
    w = row["weight"]
    if isinstance(w, (int, float)) and w < 0:
        return True
    return bool(PENALTY_FUNC_RE.search(row["func"] or ""))


# --------------------------------------------------------------------------- metrics


@dataclass
class Metrics:
    tags: dict[str, list[tuple[int, float]]]
    walls: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    @classmethod
    def load(cls, run: Run) -> "Metrics | None":
        ev = run.newest_tfevents
        if ev is None:
            return None
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()
        tags: dict[str, list[tuple[int, float]]] = {}
        walls: dict[str, list[tuple[int, float]]] = {}
        for t in ea.Tags().get("scalars", []):
            events = ea.Scalars(t)
            tags[t] = [(s.step, s.value) for s in events]
            walls[t] = [(s.step, s.wall_time) for s in events]
        return cls(tags, walls)

    @property
    def iters(self) -> int:
        best = 0
        for series in self.tags.values():
            if series:
                best = max(best, series[-1][0])
        return best

    def series(self, tag: str) -> list[tuple[int, float]]:
        return self.tags.get(tag, [])

    def last(self, tag: str, default: float = float("nan")) -> float:
        s = self.series(tag)
        return s[-1][1] if s else default

    def window(self, tag: str, lo: int, hi: int) -> list[tuple[int, float]]:
        return [(st, v) for st, v in self.series(tag) if lo <= st <= hi]

    def stats(self, tag: str, lo: int, hi: int) -> dict:
        pts = [(st, v) for st, v in self.window(tag, lo, hi) if math.isfinite(v)]
        if not pts:
            return {"n": 0}
        xs = [st for st, _ in pts]
        ys = [v for _, v in pts]
        slope = 0.0
        if len(pts) >= 4:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            if den > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        return {
            "n": len(pts),
            "first": round(ys[0], 5),
            "last": round(ys[-1], 5),
            "min": round(min(ys), 5),
            "max": round(max(ys), 5),
            "mean": round(sum(ys) / len(ys), 5),
            "slope": round(slope, 7),
        }

    def samples(self, tag: str, lo: int, hi: int, n: int = 12) -> list[list[float]]:
        pts = [(st, v) for st, v in self.window(tag, lo, hi) if math.isfinite(v)]
        if len(pts) > n:
            stride = math.ceil(len(pts) / n)
            pts = pts[::stride]
        return [[st, round(v, 5)] for st, v in pts]

    # -- timing: tfevents carry wall_time, so iteration speed and ETA are
    #    measured locally, exactly like the metrics.

    @classmethod
    def _key(cls) -> str:
        return "Train/mean_reward"

    def sec_per_iter(self) -> float | None:
        """Measured seconds per training iteration (median of recent deltas)."""
        pts = self.walls.get(self._key(), [])
        if len(pts) < 3:
            return None
        deltas = []
        for (s0, w0), (s1, w1) in zip(pts[-25:], pts[-24:]):
            if s1 > s0 and w1 > w0:
                deltas.append((w1 - w0) / (s1 - s0))
        if not deltas:
            return None
        deltas.sort()
        return deltas[len(deltas) // 2]


# --------------------------------------------------------------------- analytics
#
# Everything here is deterministic and precomputed so the local LLM receives a
# finished, numbers-only state summary and only has to judge + plan.


ANALYSIS_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/entropy",
    "Loss/learning_rate",
    "Loss/value",
    "Loss/surrogate",
    "Policy/mean_std",
    "Perf/total_fps",
    "Episode_Termination/fell_over",
    "Episode_Termination/time_out",
    "Episode_Termination/nan_state",
    "Episode_Termination/out_of_terrain_bounds",
    "Metrics/twist/error_vel_xy",
    "Metrics/twist/error_vel_yaw",
    "Metrics/peak_height_mean",
    "Metrics/slip_velocity_mean",
    "Metrics/angular_momentum_mean",
]

CHANGE_MONEY_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/fell_over",
]


def plateau_report(m: Metrics, cfg: Config) -> dict:
    it = m.iters
    lo = max(0, it - cfg.monitor.stagnation_window + 1)
    st = m.stats("Train/mean_reward", lo, it)
    out: dict = {"window": [lo, it], **{k: st.get(k) for k in ("n", "first", "last", "mean", "slope")}}
    if st.get("n", 0) >= 10:
        span = max(abs(st["mean"]), 1e-6)
        rel = (st["last"] - st["first"]) / span
        out["rel_gain"] = round(rel, 4)
        out["flat"] = bool(rel < cfg.monitor.stagnation_rel_gain and it >= cfg.monitor.stagnation_min_iters)
    else:
        out["flat"] = None
    return out


def window_compare(m: Metrics, it: int, w: int) -> dict:
    """mean of last window vs mean of the window before it, per tag."""
    out = {}
    hi1, lo1 = it, max(0, it - w + 1)
    hi0, lo0 = lo1 - 1, max(0, lo1 - w)
    for tag in ANALYSIS_TAGS + sorted(t for t in m.tags if t.startswith("Episode_Reward/")):
        s1 = m.stats(tag, lo1, hi1)
        s0 = m.stats(tag, lo0, hi0)
        if s1.get("n", 0) == 0:
            continue
        row = {"mean": s1["mean"], "last": s1["last"], "slope": s1["slope"]}
        if s0.get("n", 0) >= 3 and abs(s0["mean"]) > 1e-9:
            row["rel_vs_prev_window"] = round((s1["mean"] - s0["mean"]) / abs(s0["mean"]), 3)
        out[tag] = row
    return out


def penalty_audit(m: Metrics, env_cfg: dict, cfg: Config, it: int) -> list[dict]:
    rows = []
    lo = max(0, it - 10)
    for r in reward_table(env_cfg):
        if not penalty_expect_le_zero(r):
            continue
        tag = f"Episode_Reward/{r['term']}"
        pts = m.window(tag, lo, it)
        vals = [v for _, v in pts if math.isfinite(v)]
        if not vals:
            continue
        mx = max(vals)
        rows.append(
            {
                "term": r["term"],
                "weight": r["weight"],
                "mean": round(sum(vals) / len(vals), 5),
                "max": round(mx, 5),
                "status": "VIOLATION" if mx > cfg.safety.penalty_eps else "ok",
            }
        )
    return rows


def reward_mass(m: Metrics, env_cfg: dict, it: int, w: int) -> dict:
    """Share of each positive reward term in the positive reward sum — the
    'mass vs weight' view AGENTS.md asks for."""
    lo = max(0, it - w + 1)
    rows = []
    total = 0.0
    for r in reward_table(env_cfg):
        vals = [v for _, v in m.window(f"Episode_Reward/{r['term']}", lo, it) if math.isfinite(v)]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if mean > 0:
            rows.append({"term": r["term"], "weight": r["weight"], "mean": round(mean, 4)})
            total += mean
    rows.sort(key=lambda x: -x["mean"])
    for row in rows:
        row["share"] = round(row["mean"] / total, 3) if total > 0 else None
    return {"total_positive": round(total, 4), "top": rows[:8]}


def curriculum_pacing(m: Metrics, it: int) -> dict:
    """Stage changes of Curriculum/* terms, plus metric jumps right at them."""
    changes = []
    for tag in sorted(t for t in m.tags if t.startswith("Curriculum/")):
        pts = m.series(tag)
        prev = None
        for st, v in pts:
            if prev is not None and abs(v - prev) > 1e-9 and st > 0:
                changes.append({"tag": tag.replace("Curriculum/", ""), "iter": st,
                                "from": round(prev, 4), "to": round(v, 4)})
            prev = v
    warnings = []
    for ch in changes[-10:]:
        for key in CHANGE_MONEY_TAGS:
            around = [(s, v) for s, v in m.window(key, ch["iter"] - 1, ch["iter"] + 1) if math.isfinite(v)]
            if len(around) < 2:
                continue
            (s0, v0), (s1, v1) = around[0], around[-1]
            if s0 == s1 or abs(v0) < 1e-9:
                continue
            rel = (v1 - v0) / abs(v0)
            if rel < -0.10 and key != "Train/mean_episode_length" or (rel < -0.10):
                warnings.append({"stage": ch, "metric": key,
                                 "step": round(rel, 3), "at_iter": s1})
                break
    return {"stage_changes": changes[-15:], "pacing_warnings": warnings}


def health_report(m: Metrics, it: int) -> dict:
    w = max(10, min(50, it // 4 or 1))
    lo = max(0, it - w + 1)
    out = {"window": [lo, it]}
    for tag in ("Episode_Termination/fell_over", "Episode_Termination/time_out",
                "Episode_Termination/nan_state", "Perf/total_fps",
                "Loss/entropy", "Policy/mean_std", "Train/mean_episode_length",
                "Metrics/peak_height_mean"):
        st = m.stats(tag, lo, it)
        if st.get("n"):
            out[tag.split("/", 1)[-1]] = st["mean"]
    # entropy collapse flag against the whole run
    ent = m.stats("Loss/entropy", 0, it)
    if ent.get("n", 0) and ent.get("max"):
        out["entropy_frac_of_max"] = round(m.stats("Loss/entropy", lo, it)["mean"] / ent["max"], 3)
    return out


def checkpoints_report(run: Run, target: int | None) -> dict:
    nums = []
    for p in run.dir.glob("model_*.pt"):
        mm = re.fullmatch(r"model_(\d+)\.pt", p.name)
        if mm:
            nums.append(int(mm.group(1)))
    nums.sort()
    done = (nums[-1] + 1) if nums else 0
    return {
        "target": target,
        "checkpoints": nums[-8:],
        "last_checkpoint_iter": nums[-1] if nums else None,
        "remaining_to_target": max(int(target) - done, 0) if target else None,
    }


def sleep_plan(cfg: Config, exp: dict, agent_cfg: dict, m: Metrics,
               it: int, last_analyzed: int) -> dict:
    """When to wake next, derived from the MEASURED iteration rate:

    - next analysis cadence phase (it + remaining slots of analyze_every_iters),
    - optionally nudged to land just after a checkpoint save when a save
      boundary is close (an intervention then loses ~0 iterations),
    - clamped to [min_sleep_s, max_sleep_s]; ETA to the plan's end included.
    """
    out: dict = {"sleep_s": cfg.monitor.tick_s, "eta_s": None,
                 "next_check_iter": None, "sec_per_iter": None}
    spi = m.sec_per_iter()
    if spi is None or not math.isfinite(spi) or spi <= 0:
        return out
    missing = cfg.monitor.analyze_every_iters - max(0, it - last_analyzed)
    next_iter = it + max(missing, 1)
    if cfg.monitor.align_checks_to_checkpoints:
        si = int(agent_cfg.get("save_interval") or 0)
        if si > 0:
            save_after = ((it + si) // si) * si
            if save_after > it and save_after - it <= max(missing, 1) * 2:
                next_iter = save_after + 1  # wake just after the save
    sleep = (next_iter - it) * spi
    sleep = max(cfg.monitor.min_sleep_s, min(cfg.monitor.max_sleep_s, sleep))
    out.update({"sleep_s": round(sleep), "next_check_iter": next_iter,
                "sec_per_iter": round(spi, 3)})
    target = exp.get("target_iterations")
    if target:
        out["eta_s"] = round(max(0, int(target) - it) * spi)
    return out


def build_analysis(run: Run, m: Metrics, env_cfg: dict, agent_cfg: dict,
                   cfg: Config, state: dict, phase: str) -> dict:
    it = m.iters
    w = cfg.monitor.stagnation_window
    exp = exp_state(state, run.experiment)
    target = exp.get("target_iterations")
    recent_tags = ["Train/mean_reward", "Train/mean_episode_length",
                   "Episode_Termination/fell_over", "Loss/entropy"]
    return {
        "phase": phase,
        "iterations": {"first": 0, "last": it, "count": it + 1},
        "plateau": plateau_report(m, cfg),
        "trends_last_vs_prev_window": window_compare(m, it, w),
        "penalty_audit": penalty_audit(m, env_cfg, cfg, it),
        "reward_mass": reward_mass(m, env_cfg, it, w),
        "curriculum": curriculum_pacing(m, it),
        "health": health_report(m, it),
        "progress": checkpoints_report(run, target),
        "recent_points": {t: m.samples(t, max(0, it - 12), it) for t in recent_tags},
    }


# --------------------------------------------------------------------------- rules


@dataclass
class Finding:
    code: str
    severity: str  # critical | high | medium | info
    detail: str


def run_rules(m: Metrics, env_cfg: dict, cfg: Config) -> list[Finding]:
    findings: list[Finding] = []
    it = m.iters
    lo = max(0, it - cfg.monitor.stagnation_window + 1)

    nan = m.window("Episode_Termination/nan_state", max(0, it - 20), it)
    if any(v > 0 for _, v in nan):
        findings.append(Finding("nan_state", "critical", "Episode_Termination/nan_state > 0"))
    for tag in ("Train/mean_reward", "Train/mean_episode_length"):
        for st, v in m.window(tag, max(0, it - 20), it):
            if not math.isfinite(v):
                findings.append(Finding("nonfinite_metric", "critical", f"{tag} at iter {st} = {v}"))

    for row in reward_audit_violations(m, env_cfg, cfg, it):
        findings.append(Finding("penalty_sign", "high", row))

    plat = plateau_report(m, cfg)
    if plat.get("flat"):
        findings.append(
            Finding("stagnation", "medium",
                    f"mean_reward {plat['first']} -> {plat['last']} over {plat['window']} "
                    f"(rel gain {plat.get('rel_gain')})")
        )

    ent = m.stats("Loss/entropy", lo, it)
    if ent.get("n", 0) >= 10 and ent["last"] < 0.15 * max(ent["max"], 1e-9):
        findings.append(Finding("entropy_collapse", "medium",
                                f"Loss/entropy {ent['max']} -> {ent['last']} in {lo}..{it}"))

    el = m.stats("Train/mean_episode_length", lo, it)
    if el.get("n", 0) >= 10 and el["mean"] > 0 and el["last"] < 0.5 * el["mean"] \
            and it >= cfg.monitor.stagnation_min_iters:
        findings.append(Finding("episode_len_collapse", "medium",
                                f"mean_episode_length {el['mean']:.1f} -> {el['last']:.1f}"))
    return findings


def reward_audit_violations(m: Metrics, env_cfg: dict, cfg: Config, it: int) -> list[str]:
    return [
        f"Episode_Reward/{r['term']} max={r['max']} (weight={r['weight']}) > {cfg.safety.penalty_eps}"
        for r in penalty_audit(m, env_cfg, cfg, it)
        if r["status"] == "VIOLATION"
    ]


# --------------------------------------------------------------------------- llm


SYSTEM_PROMPT = """You are the training supervisor for mjlab/PPO locomotion policies of the
Microduck robot (14 XL330 servos, 25 cm, 800 g, 50 Hz, deployed via ONNX on the
real robot). You receive a PRECOMPUTED analysis block from the local monitor —
trends, plateau detection, penalty audit, reward mass, curriculum pacing,
health and progress are already computed for you. Trust the numbers, quote them
as evidence, do not recompute or invent metrics.

Phases:
- "monitoring": a run is active.
    keep   = training is progressing; let it run.
    adjust = plateau/harmful config proven; propose allowlisted changes.
    abort  = critical invariant broken; stop for human review.
- "post_run": the run ENDED (target reached, crash or kill).
    keep   = if progress.remaining is 0 or null the plan is COMPLETE (done);
             if the run died early with remaining > 0, restart the SAME config
             from the latest checkpoint to finish the plan.
    adjust = next phase needs config changes; if the plan is exhausted and you
             want more training, include "agent.max-iterations" as the NEW
             ABSOLUTE total (e.g. current done + 2000) in changes.
    abort  = do not restart; needs a human.

Repository rules that constrain you (AGENTS.md — hard lessons):
- SIGN CONVENTION: Episode_Reward/<term> logs the WEIGHTED value. mjlab-base
  cost functions return >= 0 -> NEGATIVE weight; self-negating microduck
  functions (*_penalty, *_l1) return <= 0 -> POSITIVE weight. Every penalty
  term MUST log <= 0 — the penalty_audit block already flags violations
  (double negation: the policy farms the "penalty"). Fix the weight sign.
- RL optimizes the letter of the reward: under-specified DOFs get exploited
  (ballistic whip, butt-hopping, crash-sits). You can only reweight — if the
  cure is a gate, say so in notes, do not fake it with weight nudges.
- Never gate a positive reward on a bad state. If a positive term grows while
  health.fell_over is high, suspect farming.
- Regularizers: motion-blockers (body_ang_vel, angular_momentum, pose std)
  suppress dynamic motion — keep LOW for dynamic tasks. Smoothness
  (action_rate) is safe but heavy smoothness before skill discovery makes
  "do nothing" win.
- Compare reward MASS (reward_mass block), not weights: PPO sees relative
  advantage; the same weight is 4x weaker under a 4x-larger positive stack.
- Curricula must phase-align with learning; curriculum.pacing_warnings marks
  metrics stepping DOWN at stage boundaries. You may NOT edit curricula (not
  allowlisted) — put the recommendation in notes.
- Main task term must grow (total reward can rise on regularizers alone);
  episode length must behave as the task demands; typical gaits need
  4000-6000 iterations — early runs are too early to judge.
- Honest reporting: if unsure, keep. Never invent numbers.
- Changes: only keys in the allowlist, only numbers inside the given ranges,
  at most max_changes. Everything else goes to notes.

Output STRICT JSON only (no prose outside it):
{
  "verdict": "keep" | "adjust" | "abort",
  "confidence": 0.0..1.0,
  "diagnosis": "short, evidence-based",
  "evidence": ["metric refs with numbers", ...],
  "changes": [{"key": "<allowlisted-cli-flag>", "value": <number>, "reason": "..."}],
  "notes": "what a human should look at / ideas not expressible as a change"
}"""


def llm_analyze(cfg: Config, payload: dict) -> dict | None:
    body = json.dumps(
        {
            "model": cfg.llm.model,
            "max_tokens": cfg.llm.max_tokens,
            "temperature": cfg.llm.temperature,
            # qwen3.5's thinking block otherwise eats max_tokens and the reply
            # never reaches the JSON (stop_reason=max_tokens, empty text).
            "thinking": {"type": "disabled"},
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(payload, default=str)}],
        }
    ).encode()
    req = urllib.request.Request(
        cfg.llm.base_url.rstrip("/") + "/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": "ollama",
            "anthropic-version": "2023-06-01",
        },
    )
    reminder = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=cfg.llm.timeout_s) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            log.warning("LLM call failed (attempt %d): %s", attempt, exc)
            if attempt == 2:
                return None
            continue
        text = "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict))
        rec = extract_json(text)
        if rec is not None:
            return rec
        log.warning("LLM returned unparseable output (attempt %d): %.400s", attempt, text)
        reminder = "REMINDER: output ONLY a single valid JSON object."
        if reminder:
            body = json.dumps(
                {
                    "model": cfg.llm.model,
                    "max_tokens": cfg.llm.max_tokens,
                    "temperature": cfg.llm.temperature,
            # qwen3.5's thinking block otherwise eats max_tokens and the reply
            # never reaches the JSON (stop_reason=max_tokens, empty text).
            "thinking": {"type": "disabled"},
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": json.dumps(payload, default=str)},
                        {"role": "assistant", "content": text or "(empty)"},
                        {"role": "user", "content": reminder},
                    ],
                }
            ).encode()
            req = urllib.request.Request(
                cfg.llm.base_url.rstrip("/") + "/v1/messages",
                data=body,
                headers={"content-type": "application/json", "x-api-key": "ollama",
                         "anthropic-version": "2023-06-01"},
            )
    return None


def extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    dec = json.JSONDecoder()
    while start != -1:
        try:
            obj, _ = dec.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- proc


def _proc_iter():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cmdline = [c.decode(errors="replace") for c in (entry / "cmdline").read_bytes().split(b"\0") if c]
            stat = (entry / "stat").read_text()
            ppid = int(stat.split(") ", 1)[1].split()[1])
            cwd = os.readlink(entry / "cwd")
            started = (entry / "stat").stat().st_mtime
        except (OSError, ValueError, IndexError):
            continue
        yield pid, ppid, cmdline, cwd, started


TRAIN_MARKERS = (".venv/bin/train", "mjlab.scripts.train", "scripts/train.py")


def find_train_proc(state: dict | None = None) -> dict | None:
    own = os.getpid()
    smoke_pids = set(state.get("smoke_pids", [])) if state else set()
    best = None
    for pid, ppid, cmdline, cwd, started in _proc_iter():
        if pid in smoke_pids or pid == own:
            continue
        joined = " ".join(cmdline)
        if any(m in joined for m in TRAIN_MARKERS) and not any("autopilot" in c for c in cmdline):
            if best is None or started < best["started"]:
                best = {"pid": pid, "ppid": ppid, "cmdline": cmdline, "cwd": cwd, "started": started}
    return best


def proc_tree_ancestors(pid: int) -> list[int]:
    out, seen, cur = [], set(), pid
    while cur > 1 and cur not in seen:
        seen.add(cur)
        try:
            stat = Path(f"/proc/{cur}/stat").read_text()
            cur = int(stat.split(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if cur == os.getpid() or cur <= 1:
            break
        out.append(cur)
    return out


# Children we spawned (Popen handles). Needed so liveness checks don't mistake
# a zombie for a live process: our direct children are never reaped by init
# while we're alive, /proc/<pid> stays present until we poll()/wait() them.
_CHILDREN: dict[int, "subprocess.Popen"] = {}


def _pid_alive(pid: int) -> bool:
    p = _CHILDREN.get(pid)
    if p is not None:
        return p.poll() is None  # also reaps the zombie
    return Path(f"/proc/{pid}").exists()


def _reap_children() -> None:
    """Reap finished children (Popen.poll) so zombies never accumulate."""
    for p in list(_CHILDREN.values()):
        p.poll()
    for pid in [pid for pid, p in _CHILDREN.items() if p.returncode is not None]:
        _CHILDREN.pop(pid, None)


def stop_training(proc: dict, cfg: Config, state: dict) -> None:
    """SIGINT the training process tree (rsl_rl saves nothing on KeyboardInterrupt,
    so this can lose up to save_interval iterations — relaunches use a smaller
    save-interval to bound that), escalate SIGTERM/SIGKILL."""
    pid = proc["pid"]
    pids = [pid] + proc_tree_ancestors(pid)
    log.info("stopping training pid=%s (tree: %s)", pid, pids)
    for p in pids:
        try:
            os.kill(p, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + cfg.train.stop_grace_s
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(2)
    if not _pid_alive(pid):
        return
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + 30
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(1)
    if _pid_alive(pid):
        log.warning("training pid=%s survived SIGTERM; SIGKILL", pid)
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


# --------------------------------------------------------------------------- validate


def _change_key(change: dict) -> str:
    return str(change.get("key", "")).lstrip("-")


def validate_rec(rec: dict, cfg: Config, require_target_extension: bool = False,
                 target_iterations: int | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    verdict = rec.get("verdict")
    if verdict not in VERDICTS:
        return False, [f"bad verdict {verdict!r}"]
    if verdict in ("keep", "abort"):
        return True, []
    changes = rec.get("changes")
    if not isinstance(changes, list) or not changes:
        return False, ["verdict=adjust but changes[] is empty"]
    if len(changes) > cfg.safety.max_changes:
        errors.append(f"{len(changes)} changes > max {cfg.safety.max_changes}")
    conf = rec.get("confidence")
    if not isinstance(conf, (int, float)) or conf < cfg.monitor.confidence_min:
        errors.append(f"confidence {conf!r} < {cfg.monitor.confidence_min}")
    keys = set()
    for ch in changes:
        # allowlist/ranges are written without `--`; accept either spelling
        key = _change_key(ch)
        keys.add(key)
        value = ch.get("value")
        if not any(fnmatch.fnmatch(key, pat) for pat in cfg.safety.allow):
            errors.append(f"key {key!r} not in allowlist")
            continue
        if any(fnmatch.fnmatch(key, pat) for pat in cfg.safety.deny):
            errors.append(f"key {key!r} is denylisted")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"value for {key} is not numeric: {value!r}")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"value for {key} is not finite")
            continue
        for pat, bounds in cfg.safety.ranges.items():
            if fnmatch.fnmatch(key, pat):
                lo, hi = bounds
                if not (lo <= float(value) <= hi):
                    errors.append(f"value {value} for {key} outside [{lo}, {hi}]")
        # A running plan's horizon only grows. Cutting it short is a hand edit:
        # the model otherwise replaces a multi-day target with a short one.
        if (
            key == "agent.max-iterations"
            and target_iterations is not None
            and float(value) < float(target_iterations)
        ):
            errors.append(
                f"agent.max-iterations {value} would shorten target {target_iterations} "
                "(shortening is manual)"
            )
    if require_target_extension and "agent.max-iterations" not in keys:
        errors.append("plan exhausted: adjust must include agent.max-iterations (new absolute total)")
    return not errors, errors


# --------------------------------------------------------------------------- cmdline


def _flag(key: str) -> str:
    """LLM keys arrive as `agent.max-iterations` (JSON format); tyro needs `--`."""
    k = str(key)
    return k if k.startswith("--") else "--" + k


def strip_flags(argv: list[str], keys: set[str]) -> list[str]:
    # tyro accepts both spellings (--agent.max-iterations and --agent.max_iterations),
    # so a managed flag must be stripped in either form.
    flags = {_flag(k) for k in keys}
    # also the underscore spelling tyro accepts: --agent.max_iterations
    flags |= {"--" + f.removeprefix("--").replace("-", "_") for f in flags}
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and a in flags:
            i += 1
            if i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(a)
        i += 1
    return out


def change_flags(changes: list[dict]) -> list[str]:
    out = []
    for ch in changes:
        v = ch["value"]
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        out += [_flag(ch["key"]), str(v)]
    return out


MANAGED_FLAGS = {"--agent.load-checkpoint", "--agent.resume", "--agent.load-run",
                 "--agent.max-iterations", "--agent.save-interval"}


def relaunch_argv(orig_argv: list[str], changes: list[dict], ckpt: Path | None,
                  run_name: str, iterations: int, cfg: Config) -> list[str]:
    argv = strip_flags(orig_argv, MANAGED_FLAGS | {str(c["key"]) for c in changes})
    if ckpt is not None:
        # load-run pins to THIS run dir: the default load_run regex (".*")
        # would accept an older run with a model_<same N>.pt.
        argv += ["--agent.load-run", run_name, "--agent.load-checkpoint", ckpt.name,
                 "--agent.resume", "True"]
    argv += ["--agent.max-iterations", str(iterations)]
    argv += ["--agent.save-interval", str(cfg.train.save_interval)]
    argv += change_flags(changes)
    return argv


def spawn(argv: list[str], cwd: Path, log_file: Path, extra_env: dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(extra_env)
    with open(log_file, "ab") as fh:
        fh.write(f"\n=== {datetime.now().isoformat()} spawn: {' '.join(argv)}\n".encode())
    p = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=open(log_file, "ab"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    _CHILDREN[p.pid] = p
    log.info("spawned pid=%d cmd=%s log=%s", p.pid, " ".join(argv[:6]) + " ...", log_file)
    return p.pid


# --------------------------------------------------------------------------- smoke


def smoke_test(orig_argv: list[str], changes: list[dict], cfg: Config,
               state: dict, tag: str) -> tuple[bool, str]:
    """AGENTS.md smoke test (64 envs, 5 iters) with the proposed changes."""
    managed = MANAGED_FLAGS | {"--env.scene.num-envs", "--env.scene.num_envs"} | {
        str(c["key"]) for c in changes
    }
    argv = strip_flags(orig_argv, managed)
    argv += change_flags(changes)
    argv += ["--env.scene.num-envs", str(cfg.train.smoke_envs),
             "--agent.max-iterations", str(cfg.train.smoke_iters)]

    started = time.time()
    t0_dirs = {d.name for d in LOG_ROOT.glob("*/*")}
    log_file = STATE_DIR / f"smoke-{tag}.log"
    pid = spawn(argv, MJLAB_REPO, log_file, train_env(cfg))
    state.setdefault("smoke_pids", []).append(pid)
    save_state(state)

    ok = False
    try:
        while time.time() - started < cfg.train.smoke_timeout_s:
            if not _pid_alive(pid):
                ok = True
                break
            time.sleep(5)
    finally:
        state["smoke_pids"] = [p for p in state.get("smoke_pids", []) if p != pid]
        save_state(state)

    try:
        log_text = log_file.read_text(errors="replace")
    except OSError:
        log_text = ""
    if not ok:
        return False, f"smoke test timed out after {cfg.train.smoke_timeout_s}s"
    if "Traceback (most recent call last)" in log_text:
        return False, "smoke test raised:\n" + "\n".join(log_text.splitlines()[-30:])

    smoke_run = None
    for name in sorted({d.name for d in LOG_ROOT.glob("*/*")} - t0_dirs, reverse=True):
        cand = next((r for r in find_runs() if r.dir.name == name), None)
        if cand:
            smoke_run = cand
            break
    if smoke_run is None:
        for r in reversed(find_runs()):
            if r.newest_tfevents and r.newest_tfevents.stat().st_mtime >= started:
                smoke_run = r
                break
    if smoke_run is None:
        return False, "smoke test produced no run directory"
    if smoke_run.dir.name not in state.get("ignore_runs", []):
        state["ignore_runs"].append(smoke_run.dir.name)
        save_state(state)
    m = Metrics.load(smoke_run)
    if m is None or m.iters < 1:
        return False, f"smoke run {smoke_run.dir.name} logged no iterations"
    nanv = m.last("Episode_Termination/nan_state", 0.0)
    if nanv and nanv > 0:
        return False, f"smoke test logged nan_state={nanv}"
    rew = m.last("Train/mean_reward")
    if not math.isfinite(rew):
        return False, f"smoke test mean_reward={rew}"
    return True, f"smoke ok: {m.iters + 1} iters, mean_reward={rew:.3f}"


# --------------------------------------------------------------------------- payload


def build_payload(run: Run, m: Metrics, env_cfg: dict, agent_cfg: dict,
                  findings: list[Finding], state: dict, cfg: Config, phase: str) -> dict:
    exp = exp_state(state, run.experiment)
    analysis = build_analysis(run, m, env_cfg, agent_cfg, cfg, state, phase)
    payload: dict = {
        "phase": phase,
        "task_context": {
            "experiment": run.experiment,
            "run_name": run.name,
            "num_envs": (env_cfg.get("scene") or {}).get("num_envs"),
            "max_iterations_in_cfg": agent_cfg.get("max_iterations"),
            "experiment_target_iterations": exp.get("target_iterations"),
        },
        "analysis": analysis,
        "findings": [f.__dict__ for f in findings],
        "rewards_cfg": reward_table(env_cfg),
        "curriculum_terms": sorted((env_cfg.get("events") or {}).keys())[:40],
        "intervention_history": exp.get("interventions", [])[-6:],
        "allowlist": {
            "keys": cfg.safety.allow,
            "ranges": cfg.safety.ranges,
            "max_changes": cfg.safety.max_changes,
        },
    }
    if phase == "post_run":
        argv = _known_argv(state, run, cfg)
        payload["task_context"]["original_command"] = " ".join(argv) if argv else None
        tail = _log_tail(cfg)
        payload["process_log_tail"] = tail
    return payload


def _log_tail(cfg: Config) -> str | None:
    """Last lines of the daemon-owned training log (crash context), if any."""
    logs = sorted(STATE_DIR.glob("train-*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    try:
        lines = logs[-1].read_text(errors="replace").splitlines()
    except OSError:
        return None
    return "\n".join(lines[-cfg.monitor.post_run_log_lines:]) or None


# --------------------------------------------------------------------------- actions


def write_rec(record: dict, tag: str) -> Path:
    path = RECS_DIR / f"{tag}.json"
    save_json(path, record)
    return path


def lookup_cfg_value(key: str, env_cfg: dict, agent_cfg: dict) -> Any:
    """Current numeric value of an allowlisted key, used as the revert target."""
    if key.startswith("env.rewards.") and key.endswith(".weight"):
        term = key[len("env.rewards."):-len(".weight")]
        row = (env_cfg.get("rewards") or {}).get(term)
        if isinstance(row, dict):
            return row.get("weight")
        return None
    if key == "agent.max-iterations":
        return agent_cfg.get("max_iterations")
    algorithm = agent_cfg.get("algorithm") or {}
    if key == "agent.algorithm.learning-rate":
        return algorithm.get("learning_rate")
    if key == "agent.algorithm.entropy-coef":
        return algorithm.get("entropy_coef")
    if key == "env.scene.num-envs":
        return (env_cfg.get("scene") or {}).get("num_envs")
    return None


def previous_values(changes: list[dict], env_cfg: dict, agent_cfg: dict) -> list[dict]:
    out = []
    for ch in changes:
        key = _change_key(ch)
        old = lookup_cfg_value(key, env_cfg, agent_cfg)
        if isinstance(old, bool) or not isinstance(old, (int, float)):
            continue
        if not math.isfinite(float(old)):
            continue
        out.append({"key": key, "value": old})
    return out


def metric_snapshot(m: Metrics, it: int) -> dict:
    """Point values at `it` (last sample at or before it) for the revert check."""
    snap: dict[str, Any] = {"iter": it}
    for name, tag in (
        ("mean_reward", "Train/mean_reward"),
        ("mean_episode_length", "Train/mean_episode_length"),
        ("fell_over", "Episode_Termination/fell_over"),
        ("entropy", "Loss/entropy"),
    ):
        pts = [(step, v) for step, v in m.series(tag) if step <= it and math.isfinite(v)]
        snap[name] = pts[-1][1] if pts else None
    return snap


def window_worse(m: Metrics, snap: dict, lo: int, hi: int) -> bool:
    """True when the post-change window fell more AND scored less than the snapshot."""
    base_fell = snap.get("fell_over")
    base_rew = snap.get("mean_reward")
    if isinstance(base_fell, bool) or isinstance(base_rew, bool):
        return False
    if not isinstance(base_fell, (int, float)) or not isinstance(base_rew, (int, float)):
        return False
    fell = m.stats("Episode_Termination/fell_over", lo, hi)
    rew = m.stats("Train/mean_reward", lo, hi)
    if fell.get("n", 0) < 1 or rew.get("n", 0) < 1:
        return False
    return fell["mean"] > float(base_fell) and rew["mean"] < float(base_rew)


def should_consult_llm(phase: str, findings: list[Finding], plateau: dict | None) -> bool:
    """The local model is for judgment calls, not for a healthy keep.

    post_run always plans. monitoring calls only when a rule already fired
    or the plateau detector says the reward is flat. A quiet run records keep
    without touching the GPU the trainer is using.
    """
    if phase == "post_run":
        return True
    if findings:
        return True
    return bool(plateau and plateau.get("flat") is True)


def rules_keep(m: Metrics, cfg: Config, it: int) -> dict:
    plat = plateau_report(m, cfg)
    lo = max(0, it - cfg.monitor.stagnation_window + 1)
    fell = m.stats("Episode_Termination/fell_over", lo, it)
    slope = plat.get("slope")
    parts = ["rules keep: no findings and no plateau"]
    if slope is not None:
        parts.append(f"mean_reward slope={slope}")
    if fell.get("n"):
        parts.append(f"fell_over mean={fell.get('mean')}")
    return {
        "verdict": "keep",
        "confidence": 1.0,
        "diagnosis": "; ".join(parts),
        "evidence": [f"iterations={it}", f"plateau.flat={plat.get('flat')}"],
        "changes": [],
        "notes": "rules engine; LLM not consulted",
        "source": "rules",
    }


def _mark_intervening(state: dict) -> None:
    state["intervening"] = True
    state["intervening_since"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


def _clear_intervening(state: dict) -> None:
    state["intervening"] = False
    state["intervening_since"] = None
    save_state(state)


def _close_smoking_recs(reason: str, older_than_s: float | None = None) -> int:
    if not RECS_DIR.is_dir():
        return 0
    closed = 0
    now = time.time()
    for path in RECS_DIR.glob("*.json"):
        if older_than_s is not None:
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if age < older_than_s:
                continue
        rec = load_json(path, None)
        if not isinstance(rec, dict) or rec.get("status") != "smoking":
            continue
        rec["status"] = "smoke_failed"
        rec["smoke"] = reason
        save_json(path, rec)
        closed += 1
    return closed


def settle_smoke(cfg: Config, state: dict) -> str | None:
    """Unstick a dead smoke: expired intervening flag, then old smoking files.

    A recommendation still marked smoking and younger than smoke_timeout_s is
    left alone, so a live smoke is not closed out from under itself.
    """
    reason = release_stale_intervening(cfg, state)
    closed = _close_smoking_recs(
        f"smoking record older than smoke_timeout_s={cfg.train.smoke_timeout_s}",
        older_than_s=cfg.train.smoke_timeout_s,
    )
    if closed and not reason:
        reason = f"closed {closed} smoking recs older than smoke_timeout_s"
        log.error(reason)
    return reason


def release_stale_intervening(cfg: Config, state: dict) -> str | None:
    """Drop a stuck intervening flag once the smoke timeout has elapsed.

    A fresh flag (younger than smoke_timeout_s) stays set so a second smoke
    cannot start. A flag with no timestamp is the pre-fix stuck state and is
    cleared. Open recommendation files still marked smoking are closed.
    Returns the reason when the flag was cleared, else None.
    """
    if not state.get("intervening"):
        return None
    since = state.get("intervening_since")
    if not since:
        reason = "intervening flag had no timestamp"
    else:
        try:
            started = datetime.fromisoformat(str(since))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
        except ValueError:
            started = None
        if started is None:
            reason = f"intervening_since unparsable: {since!r}"
        else:
            age = (datetime.now(timezone.utc) - started).total_seconds()
            if age < cfg.train.smoke_timeout_s:
                return None
            reason = (
                f"intervening for {int(age)}s "
                f"past smoke_timeout_s={cfg.train.smoke_timeout_s}"
            )
    state["intervening"] = False
    state["intervening_since"] = None
    closed = _close_smoking_recs(reason)
    save_state(state)
    log.error("cleared stale intervening flag (%s); closed %d smoking recs", reason, closed)
    return reason


def _smoke_then_relaunch(run: Run, changes: list[dict], cfg: Config, state: dict,
                         record: dict, rec_path: Path, argv: list[str],
                         proc: dict | None, force_apply: bool = False) -> dict:
    """Smoke, then relaunch. intervening is always cleared, including on crash.

    force_apply is an explicit human apply (CLI `apply` or the dashboard).
    It relaunches after a passing smoke even when auto.enabled is false.
    The auto flag still gates the daemon's own unattended decisions.
    """
    record["status"] = "smoking"
    save_json(rec_path, record)
    _mark_intervening(state)
    try:
        ok_smoke, why = smoke_test(argv, changes, cfg, state, rec_path.stem)
        record["smoke"] = why
        if not ok_smoke:
            record["status"] = "smoke_failed"
            save_json(rec_path, record)
            log.error("smoke test failed: %s", why)
            return record
        if not cfg.auto.enabled and not force_apply:
            record["status"] = "ready_to_apply"
            save_json(rec_path, record)
            log.info("smoke passed; auto disabled — run: autopilot apply %s", rec_path)
            return record
        if proc is not None:
            stop_training(proc, cfg, state)
            time.sleep(3)
        exp = exp_state(state, run.experiment)
        return _relaunch(
            run, changes, cfg, state, record, rec_path,
            checkpoints_report(run, exp.get("target_iterations")), argv,
        )
    except Exception as exc:
        record["status"] = "smoke_failed"
        record["smoke"] = f"{type(exc).__name__}: {exc}"
        save_json(rec_path, record)
        log.exception("intervention failed during smoke/relaunch")
        return record
    finally:
        _clear_intervening(state)


def do_intervention(rec: dict, run: Run, cfg: Config, state: dict,
                    phase: str = "monitoring", force_apply: bool = False) -> dict:
    """Validate -> (smoke) -> stop/relaunch, or record the decision.

    force_apply skips the auto.enabled gate (a person asked). It does not
    spend the intervention budget; callers that auto-apply still do that.
    """
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp = exp_state(state, run.experiment)
    record = dict(rec)
    record.update({"run": str(run.dir), "phase": phase,
                   "time": datetime.now(timezone.utc).isoformat(), "status": "proposed"})
    rec_path = write_rec(record, tag)

    exhausted = False
    if phase == "post_run":
        prog = checkpoints_report(run, exp.get("target_iterations"))
        exhausted = not prog.get("remaining_to_target")

    ok, errors = validate_rec(
        rec, cfg, require_target_extension=(phase == "post_run" and exhausted),
        target_iterations=exp.get("target_iterations"),
    )
    if not ok:
        record.update({"status": "rejected", "errors": errors})
        save_json(rec_path, record)
        log.error("recommendation rejected: %s", errors)
        return record

    if rec["verdict"] == "keep":
        if phase == "monitoring":
            record["status"] = "keep"
            save_json(rec_path, record)
            return record
        # post_run + keep: plan complete -> done; died early -> restart same cfg
        prog = checkpoints_report(run, exp.get("target_iterations"))
        if not prog.get("remaining_to_target"):
            record["status"] = "plan_completed"
            exp["completed_runs"].append(run.dir.name)
            queue_note_completion(cfg, state,
                                  exp.get("last_argv") or state.get("last_train_argv"))
            save_json(rec_path, record)
            log.info("run %s: plan completed (%s)", run.dir.name, rec.get("diagnosis", "")[:200])
            return record
        if not cfg.auto.enabled and not force_apply:
            record["status"] = "ready_to_relaunch"
            save_json(rec_path, record)
            return record
        return _relaunch(run, [], cfg, state, record, rec_path, prog)

    if rec["verdict"] == "abort":
        proc = find_train_proc(state)
        if (cfg.auto.enabled or force_apply) and proc:
            stop_training(proc, cfg, state)
            record["status"] = "run_stopped"
        else:
            record["status"] = "abort_recommended"
        exp["last_verdict"] = "abort"
        queue_note_abort(cfg, state, exp.get("last_argv") or state.get("last_train_argv"))
        save_json(rec_path, record)
        log.warning("ABORT verdict: %s", rec.get("diagnosis"))
        return record

    # verdict == adjust
    changes = rec["changes"]
    argv = _known_argv(state, run, cfg)
    if argv is None:
        record["status"] = "no_argv"
        save_json(rec_path, record)
        log.error("cannot intervene: original launch command unknown (daemon saw no live process)")
        return record

    env_cfg, agent_cfg = run_cfg(run)
    record["previous"] = previous_values(changes, env_cfg, agent_cfg)
    record["kind"] = rec.get("kind") or "apply"
    if record["kind"] != "revert" and not record.get("snapshot"):
        measured = Metrics.load(run)
        if measured is not None and measured.iters >= 1:
            record["snapshot"] = metric_snapshot(measured, measured.iters)

    proc = find_train_proc(state)
    live = proc is not None and phase == "monitoring"
    if not live and not cfg.auto.enabled and not force_apply:
        record["status"] = "ready_to_apply"
        save_json(rec_path, record)
        return record
    return _smoke_then_relaunch(
        run, changes, cfg, state, record, rec_path, argv, proc if live else None,
        force_apply=force_apply,
    )


def _argv_matches_task(argv: list[str], task: str | None) -> bool:
    if not task:
        return False
    return str(task) in argv


def _argv_from_wandb_meta(meta: Path) -> list[str] | None:
    try:
        data = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    args = data.get("args")
    if not isinstance(args, list) or not args:
        return None
    program = data.get("program")
    exe = data.get("executable")
    if exe and program:
        return [str(exe), str(program), *[str(a) for a in args]]
    if program:
        return [str(program), *[str(a) for a in args]]
    return None


def _wandb_meta_matches_run(meta: Path, run_name: str) -> bool:
    """True when this wandb run's stdout names the rsl_rl run directory."""
    log_path = meta.parent / "output.log"
    try:
        with log_path.open("r", errors="replace") as fh:
            head = fh.read(8192)
    except OSError:
        return False
    return run_name in head


def argv_from_wandb(run: Run) -> list[str] | None:
    """Launch argv recorded by wandb for this run, if the log names the run dir."""
    candidates: list[Path] = []
    if run.dir.is_dir():
        candidates.extend(run.dir.rglob("wandb-metadata.json"))
    wandb_root = MJLAB_REPO / "wandb"
    if wandb_root.is_dir():
        candidates.extend(wandb_root.glob("run-*/files/wandb-metadata.json"))
    for meta in candidates:
        if not _wandb_meta_matches_run(meta, run.dir.name):
            continue
        argv = _argv_from_wandb_meta(meta)
        if argv:
            return argv
    return None


def _store_argv(state: dict, run: Run, argv: list[str]) -> list[str]:
    remembered = list(argv)
    exp_state(state, run.experiment)["last_argv"] = remembered
    state["last_train_argv"] = list(remembered)
    return remembered


def _known_argv(state: dict, run: Run, cfg: Config) -> list[str] | None:
    """Original launch command: state, then wandb metadata, then the scenario catalog.

    A dead run often has no `last_argv` (the daemon never saw the process).
    wandb-metadata.json still has program+args, and the experiment folder name
    is a scenario key, so a relaunch does not die with status no_argv.
    """
    exp = (state.get("experiments") or {}).get(run.experiment) or {}
    argv = exp.get("last_argv")
    if argv:
        return list(argv)
    spec = scenario_spec(cfg, run.experiment)
    task = str(spec["task"]) if spec and spec.get("task") else None
    global_argv = state.get("last_train_argv")
    if global_argv and _argv_matches_task(list(global_argv), task):
        return _store_argv(state, run, list(global_argv))
    recovered = argv_from_wandb(run)
    if recovered:
        log.info("recovered argv for %s from wandb metadata", run.dir.name)
        return _store_argv(state, run, recovered)
    launch = scenario_launch(cfg, run.experiment)
    if launch is not None:
        log.info("recovered argv for %s from scenario catalog", run.experiment)
        return _store_argv(state, run, launch[0])
    return None


def _relaunch(run: Run, changes: list[dict], cfg: Config, state: dict, record: dict,
              rec_path: Path, prog: dict, argv: list[str] | None = None) -> dict:
    exp = exp_state(state, run.experiment)
    argv = argv or _known_argv(state, run, cfg)
    if argv is None:
        record["status"] = "no_argv"
        save_json(rec_path, record)
        return record
    run2 = next((r for r in find_runs() if r.dir == run.dir), run)
    ckpt = run2.latest_checkpoint()
    # A plan-extending change carries the NEW absolute total. Make it the
    # target and drop it from changes — relaunch_argv appends
    # --agent.max-iterations itself (having it twice lets tyro pick either).
    change_list = list(changes)
    for ch in list(change_list):
        if str(ch.get("key", "")).lstrip("-") == "agent.max-iterations":
            exp["target_iterations"] = int(ch["value"])
    change_list = [c for c in change_list
                   if str(c.get("key", "")).lstrip("-") != "agent.max-iterations"]
    target = exp.get("target_iterations")
    done = prog.get("last_checkpoint_iter")
    if target and done is not None:
        remaining = max(int(target) - int(done), cfg.train.smoke_iters)
    else:
        remaining = max(m_value(changes), 1000)
    new_argv = relaunch_argv(argv, change_list, ckpt, run.dir.name, remaining, cfg)
    log_file = STATE_DIR / f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    new_pid = spawn(new_argv, MJLAB_REPO, log_file, train_env(cfg))
    state["last_train_argv"] = new_argv
    exp["last_argv"] = new_argv
    record.update({"status": "applied", "relaunch_pid": new_pid, "relaunch_argv": new_argv,
                   "checkpoint": ckpt.name if ckpt else None,
                   "iterations": remaining})
    save_json(rec_path, record)
    snap = record.get("snapshot") if isinstance(record.get("snapshot"), dict) else None
    iter_at = done if done is not None else (snap or {}).get("iter")
    entry = {
        "time": record["time"],
        "phase": record.get("phase"),
        "changes": changes,
        "previous": record.get("previous") or [],
        "snapshot": snap,
        "iter": iter_at,
        "rec": str(rec_path),
        "smoke": record.get("smoke"),
        "kind": record.get("kind") or "apply",
    }
    if changes:
        entry["outcome"] = "pending"
    exp["interventions"].append(entry)
    exp["last_verdict"] = "adjust"
    if done is not None:
        # keep cooldown bookkeeping on the iteration axis of the new run
        exp["last_intervention_iter"] = done
    log.info("relaunched pid=%d from %s (%s, %d iters, changes=%d)",
             new_pid, ckpt.name if ckpt else "fresh", run.dir.name, remaining, len(changes))
    return record


def m_value(changes: list[dict]) -> int:
    for ch in changes:
        if _change_key(ch) == "agent.max-iterations":
            return int(ch["value"])
    return 1000


def revertable_changes(previous: list[dict], target_iterations: int | None) -> list[dict]:
    """Previous values that are safe to put back. A shorter horizon is not."""
    kept: list[dict] = []
    for ch in previous:
        if not isinstance(ch, dict):
            continue
        key = _change_key(ch)
        value = ch.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if (
            key == "agent.max-iterations"
            and target_iterations is not None
            and float(value) < float(target_iterations)
        ):
            continue
        kept.append({"key": key, "value": value})
    return kept


def maybe_revert(cfg: Config, state: dict, run: Run, m: Metrics, exp: dict,
                 it: int, apply: bool) -> dict | None:
    """One automatic undo if the post-change window is worse on falls and reward.

    The decision is rules-only. A second undo of the same change is refused and
    the experiment is marked needs_human. Revert uses one budget slot.
    """
    if exp.get("needs_human"):
        return None
    originals = [
        inv for inv in (exp.get("interventions") or [])
        if inv.get("kind") != "revert" and inv.get("snapshot") and inv.get("changes")
    ]
    if not originals:
        return None
    inv = originals[-1]
    outcome = inv.get("outcome")
    if outcome in ("ok", "revert_failed"):
        return None
    base_iter = inv.get("iter")
    if not isinstance(base_iter, int):
        return None
    snap = inv.get("snapshot")
    if not isinstance(snap, dict):
        return None

    if outcome == "reverted":
        rev_iter = inv.get("reverted_at_iter")
        if not isinstance(rev_iter, int) or it - rev_iter < cfg.monitor.cooldown_iters:
            return None
        if window_worse(m, snap, rev_iter + 1, it):
            exp["needs_human"] = True
            exp["needs_human_reason"] = (
                "revert did not recover fell_over and mean_reward; "
                "leaving the run for a human"
            )
            inv["post_revert_judged"] = True
            save_state(state)
            log.error("%s", exp["needs_human_reason"])
            return {
                "status": "needs_human",
                "verdict": "keep",
                "diagnosis": exp["needs_human_reason"],
            }
        inv["post_revert_judged"] = True
        inv["outcome"] = "ok"
        save_state(state)
        return None

    if outcome not in (None, "pending"):
        return None
    if it - base_iter < cfg.monitor.cooldown_iters:
        return None
    if not window_worse(m, snap, base_iter + 1, it):
        inv["outcome"] = "ok"
        save_state(state)
        return None

    diagnosis = (
        f"reverting changes from iter {base_iter}: fell_over mean rose and "
        "mean_reward fell versus the pre-change snapshot"
    )
    if not apply:
        return {"status": "revert_recommended", "verdict": "adjust", "diagnosis": diagnosis}
    if exp.get("budget_used", 0) >= cfg.monitor.max_interventions:
        exp["needs_human"] = True
        exp["needs_human_reason"] = "revert needed but intervention budget is exhausted"
        inv["outcome"] = "revert_failed"
        save_state(state)
        return {
            "status": "needs_human",
            "verdict": "adjust",
            "diagnosis": exp["needs_human_reason"],
        }
    kept = revertable_changes(inv.get("previous") or [], exp.get("target_iterations"))
    if not kept:
        exp["needs_human"] = True
        exp["needs_human_reason"] = "regression after a change that cannot be auto-reverted"
        inv["outcome"] = "revert_failed"
        save_state(state)
        return {
            "status": "needs_human",
            "verdict": "adjust",
            "diagnosis": exp["needs_human_reason"],
        }
    rec = {
        "verdict": "adjust",
        "confidence": 1.0,
        "diagnosis": diagnosis,
        "evidence": [
            f"snapshot iter={snap.get('iter')} reward={snap.get('mean_reward')} "
            f"fell_over={snap.get('fell_over')}",
        ],
        "changes": kept,
        "notes": "rules revert; a second revert will not be attempted",
        "kind": "revert",
    }
    applied = do_intervention(rec, run, cfg, state, phase="monitoring")
    if applied.get("status") in ("applied", "ready_to_apply"):
        inv["outcome"] = "reverted"
        inv["reverted"] = True
        inv["reverted_at_iter"] = it
        exp["budget_used"] = exp.get("budget_used", 0) + 1
        exp["last_intervention_iter"] = it
        save_state(state)
        return {"status": "reverted", "verdict": "adjust", "diagnosis": diagnosis, "rec": applied}
    inv["outcome"] = "revert_failed"
    exp["needs_human"] = True
    exp["needs_human_reason"] = f"revert did not apply ({applied.get('status')})"
    save_state(state)
    return {
        "status": "needs_human",
        "verdict": "adjust",
        "diagnosis": exp["needs_human_reason"],
        "rec": applied,
    }


# --------------------------------------------------------------------------- phases


def monitoring_phase(cfg: Config, state: dict, run: Run, force: bool, approve: bool | None) -> dict:
    apply = cfg.auto.enabled if approve is None else approve
    env_cfg, agent_cfg = run_cfg(run)
    exp = exp_state(state, run.experiment)
    if not exp.get("interventions") and agent_cfg.get("max_iterations"):
        exp["target_iterations"] = agent_cfg["max_iterations"]
    else:
        exp.setdefault("target_iterations", agent_cfg.get("max_iterations"))

    m = Metrics.load(run)
    if m is None or m.iters < 1:
        return {"status": "no_metrics"}

    if settle_smoke(cfg, state):
        log.warning("stale intervening flag cleared; continuing this cycle")
    if state.get("intervening"):
        return {"status": "intervening"}
    it = m.iters
    last = exp.get("last_analyzed_iter", -10**9)
    if it < last:
        last = -10**9  # a new run replaced the old one (iter counter reset)

    def sched(res: dict) -> dict:
        res.update(sleep_plan(cfg, exp, agent_cfg, m, it, last))
        return res

    if not force and it - last < cfg.monitor.analyze_every_iters:
        return sched({"status": "waiting", "iters": it, "since_last": it - last})
    exp["last_analyzed_iter"] = it

    state["current_run"] = str(run.dir)
    state["current_experiment"] = run.experiment
    proc = find_train_proc(state)
    if proc:
        state["last_train_argv"] = proc["cmdline"]
        exp["last_argv"] = proc["cmdline"]

    findings = run_rules(m, env_cfg, cfg)
    log.info("monitoring run=%s iter=%d findings=%s", run.dir.name, it,
             [(f.code, f.severity) for f in findings])

    critical = [f for f in findings if f.severity == "critical"]
    if critical and apply:
        fake = {"verdict": "abort", "confidence": 1.0,
                "diagnosis": "; ".join(f.detail for f in critical),
                "evidence": [f.detail for f in critical], "changes": [], "notes": "rules engine"}
        rec = do_intervention(fake, run, cfg, state, phase="monitoring")
        exp["last_verdict"] = "abort"
        exp["budget_used"] = exp.get("budget_used", 0) + 1
        save_state(state)
        return {"status": "critical_stopped", "rec": rec}

    revert = maybe_revert(cfg, state, run, m, exp, it, apply)
    if revert is not None:
        save_state(state)
        return sched(revert)

    plateau = plateau_report(m, cfg)
    if not should_consult_llm("monitoring", findings, plateau):
        rec = rules_keep(m, cfg, it)
        exp["last_verdict"] = "keep"
        exp["last_diagnosis"] = rec["diagnosis"]
        exp.pop("last_error", None)
        write_rec(dict(rec, run=str(run.dir), iter=it,
                       time=datetime.now(timezone.utc).isoformat()),
                  datetime.now().strftime("%Y%m%d-%H%M%S") + "-keep")
        save_state(state)
        log.info("rules keep run=%s iter=%d (LLM skipped)", run.dir.name, it)
        return sched({
            "status": "rules_keep",
            "verdict": "keep",
            "diagnosis": rec["diagnosis"],
            "llm_s": 0,
        })

    payload = build_payload(run, m, env_cfg, agent_cfg, findings, state, cfg, "monitoring")
    t0 = time.perf_counter()
    rec = llm_analyze(cfg, payload)
    llm_s = round(time.perf_counter() - t0, 3)
    log.info("llm_analyze took %.1fs", llm_s)
    if rec is None:
        exp["last_error"] = "llm_unavailable"
        save_state(state)
        return {"status": "llm_unavailable", "llm_s": llm_s}
    exp.pop("last_error", None)
    rec.setdefault("verdict", "keep")
    exp["last_verdict"] = rec["verdict"]
    exp["last_diagnosis"] = rec.get("diagnosis")
    save_state(state)

    ok, errors = validate_rec(rec, cfg, target_iterations=exp.get("target_iterations"))
    result: dict = {"status": "analyzed", "verdict": rec["verdict"],
                    "diagnosis": rec.get("diagnosis"), "errors": errors, "llm_s": llm_s}

    if exp.get("needs_human") and rec.get("verdict") == "adjust" and apply:
        result["status"] = "needs_human"
        write_rec(dict(rec, errors=errors, run=str(run.dir), iter=it,
                       time=datetime.now(timezone.utc).isoformat(), status="needs_human"),
                  datetime.now().strftime("%Y%m%d-%H%M%S") + "-human")
        save_state(state)
        log.error("adjust suppressed: %s", exp.get("needs_human_reason"))
        return sched(result)

    if rec["verdict"] == "abort" and apply:
        result["rec"] = do_intervention(rec, run, cfg, state, phase="monitoring")
        exp["budget_used"] = exp.get("budget_used", 0) + 1
        save_state(state)
        return result

    if rec["verdict"] == "adjust" and apply and ok:
        budget = exp.get("budget_used", 0)
        since = it - exp.get("last_intervention_iter", -10**9)
        if budget >= cfg.monitor.max_interventions:
            result["status"] = "budget_exhausted"
            log.error("intervention budget exhausted for %s (autopilot reset to re-enable)",
                      run.experiment)
        elif it < cfg.monitor.min_iters_before_intervene and not force:
            result["status"] = "too_early"
        elif since < cfg.monitor.cooldown_iters:
            result["status"] = "cooldown"
        else:
            result["rec"] = do_intervention(rec, run, cfg, state, phase="monitoring")
            if result["rec"].get("status") in ("applied", "ready_to_apply"):
                exp["budget_used"] = budget + 1
                exp["last_intervention_iter"] = it
            save_state(state)
    elif rec["verdict"] == "adjust":
        result["status"] = "adjust_invalid"
        write_rec(dict(rec, errors=errors), datetime.now().strftime("%Y%m%d-%H%M%S") + "-invalid")
    else:
        write_rec(dict(rec, run=str(run.dir), iter=it,
                       time=datetime.now(timezone.utc).isoformat()),
                  datetime.now().strftime("%Y%m%d-%H%M%S") + "-keep")

    save_state(state)
    return sched(result)


def post_run_phase(cfg: Config, state: dict, run: Run, force: bool = False,
                   approve: bool | None = None) -> dict:
    """Run ended: analyze the whole thing, plan the next step, continue.

    approve=None -> follow cfg.auto.enabled; False -> plan only (the rec is
    saved for `autopilot apply`, nothing is spawned or stopped).
    """
    exp = exp_state(state, run.experiment)
    if run.dir.name in exp["post_runs"] and not force:
        return {"status": "already_planned"}
    if run.dir.name in exp["completed_runs"]:
        return {"status": "completed"}
    if settle_smoke(cfg, state):
        log.warning("stale intervening flag cleared before post_run")
    if state.get("intervening"):
        return {"status": "intervening"}

    env_cfg, agent_cfg = run_cfg(run)
    if not exp.get("interventions") and agent_cfg.get("max_iterations"):
        exp["target_iterations"] = agent_cfg["max_iterations"]
    else:
        exp.setdefault("target_iterations", agent_cfg.get("max_iterations"))

    m = Metrics.load(run)
    if m is None or m.iters < 1:
        return {"status": "no_metrics"}

    findings = run_rules(m, env_cfg, cfg)
    log.info("post_run run=%s iter=%d findings=%s", run.dir.name, m.iters,
             [(f.code, f.severity) for f in findings])
    payload = build_payload(run, m, env_cfg, agent_cfg, findings, state, cfg, "post_run")

    # the run is over — never "stop" it, only plan. post_run always consults
    # the model: the rules cannot decide done vs continue vs change.
    t0 = time.perf_counter()
    rec = llm_analyze(cfg, payload)
    llm_s = round(time.perf_counter() - t0, 3)
    log.info("llm_analyze took %.1fs", llm_s)
    if rec is None:
        exp["last_error"] = "llm_unavailable"
        save_state(state)
        return {"status": "llm_unavailable", "llm_s": llm_s}
    rec.setdefault("verdict", "keep")
    exp["last_verdict"] = rec["verdict"]
    exp["last_diagnosis"] = rec.get("diagnosis")
    exp["post_runs"].append(run.dir.name)
    save_state(state)

    result: dict = {"status": "planned", "verdict": rec["verdict"],
                    "diagnosis": rec.get("diagnosis"), "llm_s": llm_s}
    if approve is False:
        # plan-only mode: persist the recommendation for `autopilot apply`
        rec_path = write_rec(dict(rec, run=str(run.dir), phase="post_run",
                                  time=datetime.now(timezone.utc).isoformat(),
                                  status="recommended"),
                             datetime.now().strftime("%Y%m%d-%H%M%S") + "-plan")
        result["rec"] = {"status": "recommended", "path": str(rec_path)}
        save_state(state)
        return result
    rec2 = do_intervention(rec, run, cfg, state, phase="post_run")
    result["rec"] = rec2
    if rec2.get("status") in ("applied", "plan_completed", "ready_to_relaunch"):
        exp["budget_used"] = exp.get("budget_used", 0) + (1 if rec2.get("status") == "applied" else 0)
    save_state(state)
    return result


def analyze_once(cfg: Config, state: dict, force: bool = False, apply: bool | None = None) -> dict:
    runs = [r for r in find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    if not runs:
        return {"status": "no_runs"}
    run = runs[-1]
    if run.age_s > cfg.monitor.active_after_s:
        return {"status": "no_active_run", "newest": run.dir.name}
    out = monitoring_phase(cfg, state, run, force, apply)
    save_state(state)
    return out


# --------------------------------------------------------------------------- daemon


_QUIET_STATUSES = {"waiting", "no_active_run", "no_runs", "already_planned",
                   "completed", "no_metrics", "intervening", "idle"}


class SessionReport:
    """Append-only markdown journal of one daemon session (reports/)."""

    def __init__(self, cfg: Config) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = REPORTS_DIR / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        self.started = time.time()
        self.counts: dict[str, int] = {}
        self.llm_s = 0.0
        self._last_hb = time.time()
        self._write(
            f"# Autopilot session {datetime.now().isoformat(timespec='seconds')}\n\n"
            f"- auto={cfg.auto.enabled} model={cfg.llm.model} "
            f"tick={cfg.monitor.tick_s}s max_interventions={cfg.monitor.max_interventions}\n"
            f"- mjlab repo: `{MJLAB_REPO}`\n"
            f"- wandb: {'online (dashboard)' if wandb_has_creds() else 'offline -> tfevents only'}\n\n"
            "## Events\n\n"
        )

    def _write(self, text: str) -> None:
        with self.path.open("a") as fh:
            fh.write(text)

    def event(self, kind: str, text: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ts = datetime.now().strftime("%H:%M:%S")
        self._write(f"- `{ts}` **{kind}**: {text}\n")

    def note_cycle(self, status: dict) -> None:
        """Record a wake result; skip routine ones, keep an hourly heartbeat."""
        llm_s = status.get("llm_s")
        if isinstance(llm_s, (int, float)) and not isinstance(llm_s, bool) and llm_s:
            self.llm_s += float(llm_s)
        if status.get("intervening_cleared"):
            self.event("intervening_cleared", str(status["intervening_cleared"]))
        st = status.get("status") or status.get("phase") or "?"
        if st in _QUIET_STATUSES:
            if time.time() - self._last_hb >= 3600 and status.get("sec_per_iter"):
                self._last_hb = time.time()
                self.event(
                    "heartbeat",
                    f"iter={status.get('iters')} {status.get('sec_per_iter'):.1f}s/iter "
                    f"eta={status.get('eta_s')}s sleep={status.get('sleep_s')}s",
                )
            return
        parts = [f"phase={status.get('phase', '?')}"]
        if status.get("scenario"):
            parts.append(f"scenario={status['scenario']}")
        if status.get("run"):
            parts.append(f"run={status['run']}")
        if status.get("argv"):
            parts.append(f"argv: {status['argv']}")
        if status.get("verdict"):
            parts.append(f"verdict={status['verdict']}")
        if status.get("diagnosis"):
            parts.append(f"dx: {str(status['diagnosis'])[:300]}")
        rec = status.get("rec")
        if isinstance(rec, dict):
            parts.append(f"rec={rec.get('status')}")
            if rec.get("checkpoint"):
                parts.append(f"from {rec['checkpoint']}")
            keys = [c.get("key") for c in (rec.get("changes") or []) if isinstance(c, dict)]
            if keys:
                parts.append(f"changes={keys}")
            if rec.get("relaunch_pid"):
                parts.append(f"pid={rec['relaunch_pid']}")
        if status.get("sec_per_iter"):
            parts.append(f"{status['sec_per_iter']:.1f}s/iter")
        if status.get("eta_s") is not None:
            parts.append(f"eta={status['eta_s']}s")
        if status.get("sleep_s") is not None:
            parts.append(f"sleep={status['sleep_s']}s")
        if isinstance(llm_s, (int, float)) and not isinstance(llm_s, bool):
            parts.append(f"llm={float(llm_s):.1f}s")
        self.event(st, " ".join(str(p) for p in parts))

    def finish(self, stop_reason: str) -> None:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())) or "none"
        self._write(
            f"\n## Summary\n\n"
            f"- duration: {(time.time() - self.started) / 3600:.2f} h, stopped: {stop_reason}\n"
            f"- events: {counts}\n"
            f"- llm time: {self.llm_s:.1f}s\n"
        )


def cmd_run(cfg: Config) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(STATE_DIR / "autopilot.log")],
    )
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        log.info("shutting down (training is left running)")

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    log.info("autopilot cycle started (auto=%s, model=%s, tick=%ss, wandb_creds=%s)",
             cfg.auto.enabled, cfg.llm.model, cfg.monitor.tick_s, wandb_has_creds())
    report = SessionReport(cfg)
    log.info("session report: %s", report.path)
    import dashboard as dash
    rt = None
    if cfg.dashboard.enabled:
        rt = dash.attach(cfg)
        try:
            dash.serve(rt)
            log.info("dashboard %s", dash.url(rt.cfg, rt.bound_port))
        except Exception as exc:
            log.error("dashboard not started: %s", exc)
            dash.detach()
            rt = None

    while not stop["flag"]:
        status: dict = {}
        _reap_children()
        try:
            if rt is not None:
                dash.reload_if_stale(rt)
                cfg = rt.cfg
            state = load_state()
            status = {"time": datetime.now(timezone.utc).isoformat()}
            cleared = settle_smoke(cfg, state)
            if cleared:
                status["intervening_cleared"] = cleared
            if state.get("intervening"):
                status["phase"] = "intervening"
                status["status"] = "intervening"
            else:
                status.update(_cycle_once(cfg, state) or {})
            save_json(STATUS_FILE, status)
            report.note_cycle(status)
        except Exception as exc:
            log.exception("cycle failed")
            report.event("error", f"{type(exc).__name__}: {exc}")
        # ETA-aware wake: the cycle tells us when the next check phase is due;
        # fall back to the plain tick when nothing is being measured.
        sleep_s = int(status.get("sleep_s") or cfg.monitor.tick_s)
        sleep_s = max(1, min(sleep_s, cfg.monitor.max_sleep_s))
        log.debug("sleeping %ss (eta=%s)", sleep_s, status.get("eta_s"))
        if rt is not None:
            dash.wait_wake(rt, sleep_s, stop)
        else:
            for _ in range(sleep_s):
                if stop["flag"]:
                    break
                time.sleep(1)
    if rt is not None and rt.httpd is not None:
        dash.stop_server(rt)
    report.finish("signal")
    return 0


def _cycle_once(cfg: Config, state: dict) -> dict:
    """One wake: dashboard action | monitoring | post_run | idle.

    A dashboard action runs instead of the monitoring pass so a smoke test
    and a wake never overlap. The next wake resumes the normal cycle.
    """
    import dashboard as dash

    rt = dash.runtime()
    if rt is not None:
        dash.reload_if_stale(rt)
        cfg = rt.cfg
        acted = dash.drain_one(rt, state)
        if acted is not None:
            return acted
    # remember the live launch command every wake, so a later post-run
    # relaunch knows the original argv even if we never analyzed mid-run
    proc = find_train_proc(state)
    if proc:
        runs_all = find_runs()
        exp_name = runs_all[-1].experiment if runs_all else "_unknown"
        state["last_train_argv"] = proc["cmdline"]
        exp_state(state, exp_name)["last_argv"] = proc["cmdline"]
        save_state(state)

    runs = [r for r in find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    if runs:
        newest = runs[-1]
        if newest.age_s <= cfg.monitor.active_after_s or (proc and runs_all_is_newest(newest)):
            out = monitoring_phase(cfg, state, newest, force=False, approve=None)
            out["phase"] = "monitoring"
            out["run"] = newest.dir.name
            return out
        if newest.dir.name not in exp_state(state, newest.experiment)["post_runs"] and \
                newest.dir.name not in exp_state(state, newest.experiment)["completed_runs"]:
            out = post_run_phase(cfg, state, newest)
            out["phase"] = "post_run"
            out["run"] = newest.dir.name
            return out

    if cfg.auto.start_if_idle and (cfg.train.queue or cfg.train.scenarios or cfg.train.command):
        import shlex

        if not find_train_proc(state):
            # queue mode: walk the scenario list; single mode: default
            # scenario, else the raw command (back-compat).
            scenario = queue_next(cfg, state)
            extra_env: dict[str, str] = {}
            if scenario:
                launch = scenario_launch(cfg, scenario)
                if launch is None:
                    log.error("queue: scenario %s misconfigured (no task?)", scenario)
                    return {"phase": "idle", "run": None}
                argv, extra_env = launch
            else:
                name = cfg.train.default_scenario
                launch = scenario_launch(cfg, name) if name else None
                if launch is not None:
                    argv, extra_env = launch
                elif cfg.train.command:
                    argv = shlex.split(cfg.train.command)
                else:
                    return {"phase": "idle", "run": None}
            if argv:
                log_file = STATE_DIR / f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
                pid = spawn(argv, MJLAB_REPO, log_file,
                            {**train_env(cfg), **extra_env})
                state["last_train_argv"] = argv
                if scenario:
                    q = state.setdefault("queue", {"completed": []})
                    q["current"] = scenario
                save_state(state)
                log.info("launched%s pid=%d: %s",
                         f" scenario={scenario}" if scenario else "", pid, " ".join(argv))
                return {"phase": "idle", "status": "launched", "pid": pid,
                        "scenario": scenario or cfg.train.default_scenario or None,
                        "argv": " ".join(argv)}
    return {"phase": "idle", "run": None}


def runs_all_is_newest(run: Run) -> bool:
    runs = find_runs()
    return bool(runs) and runs[-1].dir == run.dir


# --------------------------------------------------------------------------- commands


def cmd_status(cfg: Config) -> int:
    state = load_state()
    runs = [r for r in find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    print(f"config: {CONFIG_FILE} (exists={CONFIG_FILE.exists()})")
    print(f"auto={cfg.auto.enabled} model={cfg.llm.model} base_url={cfg.llm.base_url}")
    print(f"wandb credentials: {'online (dashboard)' if wandb_has_creds() else 'none -> offline runs'}")
    try:
        with urllib.request.urlopen(cfg.llm.base_url.rstrip("/") + "/api/version", timeout=3) as r:
            print(f"ollama: OK {json.loads(r.read())}")
    except Exception as exc:
        print(f"ollama: UNREACHABLE ({exc})")
    proc = find_train_proc(state)
    print(f"training process: {proc['pid'] if proc else 'none'}")
    print(f"intervening: {state.get('intervening')}")
    q = state.get("queue") or {}
    if cfg.train.queue:
        done = set(q.get("completed", []))
        aborted = set(q.get("aborted", []))
        nxt = queue_next(cfg, state)
        print(f"scenario queue ({len(done)}/{len(cfg.train.queue)} done): "
              f"current={q.get('current')} next={nxt}"
              + (f" aborted={sorted(aborted)}" if aborted else ""))
    if not runs:
        print("runs: none")
    for r in runs[-6:]:
        try:
            m = Metrics.load(r)
        except Exception:
            m = None
        it = m.iters if m else "?"
        rew = m.last("Train/mean_reward") if m else float("nan")
        phase = "ACTIVE" if r.age_s <= cfg.monitor.active_after_s else f"ended {r.age_s / 3600:.1f}h ago"
        print(f"  {r.experiment}/{r.name}: iter={it} mean_reward={rew:.3f} [{phase}]")
        exp = exp_state(state, r.experiment)
        marks = (" [planned]" if r.dir.name in exp["post_runs"] else "") + \
                (" [completed]" if r.dir.name in exp["completed_runs"] else "")
        if exp.get("interventions") or exp.get("target_iterations"):
            print(f"    target={exp.get('target_iterations')} interventions="
                  f"{len(exp['interventions'])} budget={exp.get('budget_used', 0)} "
                  f"last_verdict={exp.get('last_verdict')}{marks}")
        if exp.get("last_diagnosis"):
            print(f"    diagnosis: {str(exp['last_diagnosis'])[:300]}")
    if STATUS_FILE.exists():
        st = load_json(STATUS_FILE, {})
        eta = f"{st['eta_s']}s" if st.get("eta_s") is not None else "?"
        nxt = st.get("next_check_iter", "?")
        sleep_v = st.get("sleep_s", "?")
        print(f"cycle: phase={st.get('phase')} status={st.get('status')} "
              f"sec/iter={st.get('sec_per_iter')} eta={eta} "
              f"next_check_iter={nxt} sleep={sleep_v}s")
    recs = sorted(RECS_DIR.glob("*.json")) if RECS_DIR.exists() else []
    if recs:
        last_rec = load_json(recs[-1], {})
        print(f"last recommendation: {recs[-1].name} status={last_rec.get('status')} "
              f"verdict={last_rec.get('verdict')}")
    return 0


def cmd_suggest(cfg: Config, force: bool, approve: bool | None) -> int:
    state = load_state()
    print(json.dumps(analyze_once(cfg, state, force=force, apply=approve),
                     indent=2, default=str))
    return 0


def cmd_plan(cfg: Config, force: bool, approve: bool | None) -> int:
    state = load_state()
    runs = [r for r in find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    if not runs:
        print('{"status": "no_runs"}')
        return 2
    run = runs[-1]
    if approve is None:
        approve = cfg.auto.enabled
    # force clears the planned mark so `plan` can re-decide
    if force:
        exp = exp_state(state, run.experiment)
        if run.dir.name in exp["post_runs"]:
            exp["post_runs"].remove(run.dir.name)
        save_state(state)
    print(json.dumps(post_run_phase(cfg, state, run, force=force, approve=approve),
                     indent=2, default=str))
    return 0


def cmd_apply(cfg: Config, rec_path: Path) -> int:
    state = load_state()
    rec = load_json(rec_path, {})
    if not rec:
        print(f"cannot read {rec_path}", file=sys.stderr)
        return 2
    runs = [r for r in find_runs() if r.dir.name not in state.get("ignore_runs", [])]
    if not runs:
        print("no runs", file=sys.stderr)
        return 2
    out = do_intervention(rec, runs[-1], cfg, state, phase=rec.get("phase", "monitoring"),
                          force_apply=True)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("status") in ("applied", "keep", "plan_completed",
                                      "ready_to_apply", "ready_to_relaunch") else 1


def cmd_launch(cfg: Config, scenario: str | None) -> int:
    state = load_state()
    import shlex

    if find_train_proc(state):
        print("a training process is already running")
        return 1
    extra_env: dict[str, str] = {}
    if scenario:
        launch = scenario_launch(cfg, scenario)
        if launch is None:
            known = ", ".join(sorted(cfg.train.scenarios)) or "(none configured)"
            print(f"unknown scenario {scenario!r}; known: {known}", file=sys.stderr)
            return 2
        argv, extra_env = launch
    elif cfg.train.default_scenario or cfg.train.scenarios:
        name = cfg.train.default_scenario or next(iter(cfg.train.scenarios), "")
        launch = scenario_launch(cfg, name) if name else None
        if launch is None:
            if not cfg.train.command:
                print(f"no default scenario and no train.command", file=sys.stderr)
                return 2
            argv = shlex.split(cfg.train.command)
        else:
            argv, extra_env = launch
    elif cfg.train.command:
        argv = shlex.split(cfg.train.command)
    else:
        print("no scenario given and train.command is empty in autopilot.toml",
              file=sys.stderr)
        return 2
    log_file = STATE_DIR / f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    pid = spawn(argv, MJLAB_REPO, log_file, {**train_env(cfg), **extra_env})
    state["last_train_argv"] = argv
    save_state(state)
    print(f"launched pid={pid}: {' '.join(argv)}")
    return 0


def cmd_scenarios(cfg: Config) -> int:
    state = load_state()
    q = state.get("queue") or {}
    done = set(q.get("completed", []))
    if not cfg.train.scenarios and not cfg.train.command:
        print("no scenarios configured (autopilot.toml [train.scenarios])")
        return 0
    print(f"default: {cfg.train.default_scenario or '(train.command)'}")
    if cfg.train.queue:
        print(f"queue ({len(done)}/{len(cfg.train.queue)} done):")
        for name in cfg.train.queue:
            spec = scenario_spec(cfg, name) or {}
            mark = "done" if name in done else ("running" if q.get("current") == name else "pending")
            print(f"  [{mark:>7}] {name}: {spec.get('task', '???')}")
    print("scenarios:")
    for name, spec in sorted(cfg.train.scenarios.items()):
        warm = f" (warm start from {spec['warm_start_from']})" if spec.get("warm_start_from") else ""
        it = f", {spec['max_iterations']} iters" if spec.get("max_iterations") else ""
        print(f"  {name}: {spec.get('task', '???')}{it}{warm}")
    return 0


def cmd_dashboard(cfg: Config) -> int:
    """Serve the local UI. If the daemon already holds the port, just print the URL.

    Without a daemon this process also drains the action inbox, so launch /
    stop / config still work. It does not run the monitoring cycle.
    """
    import errno

    import dashboard as dash

    rt = dash.attach(cfg)
    try:
        dash.serve(rt)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"dashboard already running at {dash.url(cfg)}")
            return 0
        raise
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(dash.url(rt.cfg, rt.bound_port))
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        log.info("dashboard shutting down")

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    while not stop["flag"]:
        try:
            dash.reload_if_stale(rt)
            dash.drain_one(rt, load_state())
        except Exception:
            log.exception("dashboard action failed")
        dash.wait_wake(rt, 1, stop)
    dash.stop_server(rt)
    return 0


def cmd_reset(cfg: Config, experiment: str | None, queue: bool) -> int:
    state = load_state()
    targets = [experiment] if experiment else list(state.get("experiments", {}))
    for name in targets:
        exp = exp_state(state, name)
        exp["budget_used"] = 0
        exp["last_verdict"] = None
        exp["post_runs"] = []
        exp["completed_runs"] = []
    if queue:
        state["queue"] = {"completed": [], "current": None}
    save_state(state)
    print(f"reset: {targets}" + (" + queue" if queue else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_FILE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p_sig = sub.add_parser("suggest")
    p_sig.add_argument("--force", action="store_true")
    p_sig.add_argument("--approve", action="store_true")
    p_sig.add_argument("--recommend-only", action="store_true")
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--force", action="store_true", help="re-plan even if already planned")
    p_plan.add_argument("--approve", action="store_true")
    p_plan.add_argument("--recommend-only", action="store_true")
    sub.add_parser("run")
    p_app = sub.add_parser("apply")
    p_app.add_argument("rec")
    p_lch = sub.add_parser("launch")
    p_lch.add_argument("scenario", nargs="?", default=None,
                       help="scenario name from autopilot.toml [train.scenarios]")
    sub.add_parser("scenarios")
    sub.add_parser("dashboard")
    p_res = sub.add_parser("reset")
    p_res.add_argument("--experiment", default=None)
    p_res.add_argument("--queue", action="store_true", help="clear scenario queue progress")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path(args.config))

    if args.cmd == "status":
        return cmd_status(cfg)
    if args.cmd == "suggest":
        approve = True if args.approve else (False if args.recommend_only else None)
        return cmd_suggest(cfg, args.force, approve)
    if args.cmd == "plan":
        approve = True if args.approve else (False if args.recommend_only else None)
        return cmd_plan(cfg, args.force, approve)
    if args.cmd == "run":
        return cmd_run(cfg)
    if args.cmd == "apply":
        return cmd_apply(cfg, Path(args.rec))
    if args.cmd == "launch":
        return cmd_launch(cfg, args.scenario)
    if args.cmd == "scenarios":
        return cmd_scenarios(cfg)
    if args.cmd == "dashboard":
        return cmd_dashboard(cfg)
    if args.cmd == "reset":
        return cmd_reset(cfg, args.experiment, args.queue)
    return 2


if __name__ == "__main__":
    sys.exit(main())
