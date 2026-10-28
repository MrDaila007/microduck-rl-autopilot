"""Deterministic training-log analysis and recommendation checks.

No process control and no LLM call. `autopilot.py` re-exports these names so
the daemon, the dashboard, and the tests keep using `autopilot.<name>`.
"""

from __future__ import annotations

import fnmatch
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERDICTS = ("keep", "adjust", "abort")


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
    exp = (state.get("experiments") or {}).get(getattr(run, "experiment", ""), {}) or {}
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


# --------------------------------------------------------------------------- validate


def _change_key(change: dict) -> str:
    return str(change.get("key", "")).lstrip("-")


def validate_rec(rec: dict, cfg: Config, require_target_extension: bool = False,
                 target_iterations: int | None = None,
                 frozen_keys: set[str] | None = None) -> tuple[bool, list[str]]:
    """frozen_keys: allowlisted keys a curriculum overwrites at runtime."""
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
        if frozen_keys and key in frozen_keys:
            errors.append(f"key {key!r} is set by a curriculum at runtime; changing it has no effect")
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


SNAPSHOT_WINDOW = 25        # iterations averaged into the pre-change baseline
# A relaunch resets every env: until one full episode has elapsed (1000 steps
# / 24 steps per iter ~ 42 iters) only short episodes complete, so episode
# length and per-episode rewards read low right after a resume.
RESUME_WARMUP_ITERS = 50
EP_LEN_DROP = 0.10          # relative episode-length loss that counts as worse
FALL_SHARE_SLACK = 0.01     # fall share must improve by more than this to excuse it


def fall_share(m: Metrics, lo: int, hi: int) -> float | None:
    """Fraction of episode endings that are falls in [lo, hi].

    Episode_Termination/<term> is a COUNT of envs ended by that term per env
    step (mjlab termination_manager), not a rate: only the ratio is readable.
    """
    fell = m.stats("Episode_Termination/fell_over", lo, hi)
    if not fell.get("n"):
        return None
    total = 0.0
    for tag in m.tags:
        if tag.startswith("Episode_Termination/"):
            st = m.stats(tag, lo, hi)
            if st.get("n"):
                total += max(st["mean"], 0.0)
    return round(fell["mean"] / total, 4) if total > 0 else None


def _window_mean(m: Metrics, tag: str, lo: int, hi: int) -> float | None:
    st = m.stats(tag, lo, hi)
    return st["mean"] if st.get("n") else None


def metric_snapshot(m: Metrics, it: int) -> dict:
    """Baseline at `it` for the revert check and the intervention ledger.

    Point values (last sample at or before it) plus SNAPSHOT_WINDOW means of
    the weight-invariant metrics. mean_reward is kept for the record only: a
    reward-weight change rescales it, so it cannot judge that change.
    """
    snap: dict[str, Any] = {"iter": it}
    for name, tag in (
        ("mean_reward", "Train/mean_reward"),
        ("mean_episode_length", "Train/mean_episode_length"),
        ("fell_over", "Episode_Termination/fell_over"),
        ("entropy", "Loss/entropy"),
    ):
        pts = [(step, v) for step, v in m.series(tag) if step <= it and math.isfinite(v)]
        snap[name] = pts[-1][1] if pts else None
    lo = max(0, it - SNAPSHOT_WINDOW + 1)
    snap["ep_len_w"] = _window_mean(m, "Train/mean_episode_length", lo, it)
    snap["fall_share_w"] = fall_share(m, lo, it)
    snap["error_vel_xy_w"] = _window_mean(m, "Metrics/twist/error_vel_xy", lo, it)
    return snap


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return None
    return float(v)


def window_worse(m: Metrics, snap: dict, lo: int, hi: int) -> bool:
    """True when the policy survives clearly shorter than at the baseline.

    Judged on weight-invariant metrics only: episode length must drop by
    EP_LEN_DROP and, when both sides know it, the fall share must not have
    improved. Older snapshots without window means use their point values.
    """
    base_len = _num(snap.get("ep_len_w")) or _num(snap.get("mean_episode_length"))
    post_len = _window_mean(m, "Train/mean_episode_length", lo, hi)
    if base_len is None or post_len is None or base_len <= 0:
        return False
    shorter = post_len < (1.0 - EP_LEN_DROP) * base_len
    base_fs = _num(snap.get("fall_share_w"))
    post_fs = fall_share(m, lo, hi)
    if base_fs is not None and post_fs is not None:
        return shorter and post_fs >= base_fs - FALL_SHARE_SLACK
    return shorter


def curriculum_changes_in(m: Metrics, lo: int, hi: int) -> dict[str, int]:
    """First iteration inside (lo, hi] at which each Curriculum/* term changed."""
    changed: dict[str, int] = {}
    for tag in sorted(t for t in m.tags if t.startswith("Curriculum/")):
        prev = None
        for st, v in m.series(tag):
            if prev is not None and lo < st <= hi and abs(v - prev) > 1e-9:
                changed[tag.removeprefix("Curriculum/")] = st
                break
            prev = v
    return changed


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


