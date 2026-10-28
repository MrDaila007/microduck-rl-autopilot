"""LLM-facing digest: the run's state as interpreted facts with units.

The raw tfevents mix scales that mislead a reader who does not know mjlab:
termination tags are COUNTS of envs per step, Episode_Reward/* is a per-episode
sum divided by the 20 s horizon (short episodes shrink every term), and
reward-weight curricula rescale mean_reward at their stage boundaries. This
module turns those into ratios, per-second rates while alive, effective
weights, grouped curriculum events and noise-aware trends, plus short English
facts. Deterministic; no process control, no LLM call.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from analysis import (
    RESUME_WARMUP_ITERS,
    Metrics,
    fall_share,
    penalty_expect_le_zero,
    reward_table,
)

DIGEST_WINDOW = 50          # iterations summarised as "now"
EVENT_WINDOW = 25           # iterations on each side of a curriculum step
TREND_Z = 2.0               # |delta| above this many standard errors = real

TASK_ERROR_TAGS = {
    "Metrics/twist/error_vel_xy": "lin_vel_error_m_s",
    "Metrics/twist/error_vel_yaw": "yaw_vel_error_rad_s",
}


# --------------------------------------------------------------------- config


def step_dt(env_cfg: dict) -> float | None:
    try:
        dt = float(env_cfg["decimation"]) * float(env_cfg["sim"]["mujoco"]["timestep"])
    except (KeyError, TypeError, ValueError):
        return None
    return dt if dt > 0 else None


def episode_horizon_s(env_cfg: dict) -> float | None:
    v = env_cfg.get("episode_length_s")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def curriculum_reward_terms(env_cfg: dict) -> dict[str, str]:
    """Curriculum term -> reward term whose weight it drives (reward_weight curricula)."""
    out: dict[str, str] = {}
    for name, c in (env_cfg.get("curriculum") or {}).items():
        params = (c or {}).get("params") if isinstance(c, dict) else None
        if isinstance(params, dict) and params.get("reward_name"):
            out[name] = str(params["reward_name"])
    return out


def curriculum_managed_keys(env_cfg: dict) -> set[str]:
    """CLI keys a curriculum overwrites at runtime: changing them is a no-op."""
    return {f"env.rewards.{term}.weight" for term in curriculum_reward_terms(env_cfg).values()}


def effective_weights(m: Metrics, env_cfg: dict) -> dict[str, float]:
    weights = {r["term"]: r["weight"] for r in reward_table(env_cfg)
               if isinstance(r["weight"], (int, float))}
    for cur, term in curriculum_reward_terms(env_cfg).items():
        s = m.series(f"Curriculum/{cur}")
        if s and math.isfinite(s[-1][1]):
            weights[term] = s[-1][1]
    return weights


# -------------------------------------------------------------------- lineage

# Tags whose value is a per-episode statistic: biased low right after a resume.
_EPISODIC_PREFIXES = ("Train/mean_episode_length", "Train/mean_reward",
                      "Episode_Reward/", "Episode_Termination/")


def merge_lineage(segments: list[Metrics]) -> tuple[Metrics, list[int]]:
    """One series per tag across resumed segments, oldest first.

    Each newer segment replaces the older one from its first iteration on
    (the old run's iterations past the checkpoint belong to a discarded
    policy). Its first RESUME_WARMUP_ITERS episodic points are dropped.
    Returns the merged metrics and the iterations where segments resumed.
    """
    if len(segments) == 1:
        return segments[0], []
    tags: dict[str, list[tuple[int, float]]] = {}
    walls: dict[str, list[tuple[int, float]]] = {}
    resumes: list[int] = []
    for idx, seg in enumerate(segments):
        start = _first_iter(seg)
        if idx > 0:
            resumes.append(start)
        for tag, series in seg.tags.items():
            old = [p for p in tags.get(tag, []) if p[0] < start]
            new = series
            if idx > 0 and tag.startswith(_EPISODIC_PREFIXES):
                new = [p for p in series if p[0] >= start + RESUME_WARMUP_ITERS]
            tags[tag] = old + list(new)
        for tag, series in seg.walls.items():
            walls[tag] = [p for p in walls.get(tag, []) if p[0] < start] + list(series)
    return Metrics(tags, walls), resumes


# --------------------------------------------------------------------- trends


def robust_trend(m: Metrics, tag: str, lo: int, hi: int,
                 higher_is_better: bool | None) -> dict | None:
    """Median of the first vs last quarter of [lo, hi], judged against noise."""
    ys = [v for _, v in m.window(tag, lo, hi) if math.isfinite(v)]
    if len(ys) < 8:
        return None
    q = max(2, len(ys) // 4)
    a, b = statistics.median(ys[:q]), statistics.median(ys[-q:])
    med = statistics.median(ys)
    sigma = 1.4826 * statistics.median(abs(y - med) for y in ys)
    delta = b - a
    se = sigma * math.sqrt(2.0 / q) if sigma > 0 else 0.0
    significant = abs(delta) > TREND_Z * se if se > 0 else abs(delta) > 0
    if not significant:
        direction = "flat"
    elif higher_is_better is None:
        direction = "rising" if delta > 0 else "falling"
    else:
        direction = "improving" if (delta > 0) == higher_is_better else "worsening"
    out = {"from": round(a, 4), "to": round(b, 4), "direction": direction,
           "window": [lo, hi]}
    if abs(a) > 1e-9:
        out["rel_change"] = round(delta / abs(a), 3)
    return out


# --------------------------------------------------------------------- blocks


def _first_iter(m: Metrics) -> int:
    s = m.series("Train/mean_reward")
    return s[0][0] if s else 0


def _mean(m: Metrics, tag: str, lo: int, hi: int) -> float | None:
    st = m.stats(tag, lo, hi)
    return st["mean"] if st.get("n") else None


def episode_block(m: Metrics, env_cfg: dict, lo: int, hi: int) -> dict:
    dt = step_dt(env_cfg)
    horizon = episode_horizon_s(env_cfg)
    steps = _mean(m, "Train/mean_episode_length", lo, hi)
    out: dict[str, Any] = {"mean_steps": round(steps, 1) if steps is not None else None}
    if steps is not None and dt:
        out["mean_s"] = round(steps * dt, 2)
        if horizon:
            out["horizon_s"] = horizon
            out["pct_of_horizon"] = round(100 * steps * dt / horizon, 1)
    fs = fall_share(m, lo, hi)
    if fs is not None:
        out["episodes_ending_in_fall_pct"] = round(100 * fs, 1)
    ends: dict[str, float] = {}
    for tag in m.tags:
        if tag.startswith("Episode_Termination/"):
            v = _mean(m, tag, lo, hi)
            if v is not None:
                ends[tag.removeprefix("Episode_Termination/")] = v
    total = sum(max(v, 0.0) for v in ends.values())
    if total > 0:
        out["termination_mix_pct"] = {k: round(100 * max(v, 0.0) / total, 1)
                                      for k, v in sorted(ends.items(), key=lambda kv: -kv[1])}
    return out


def reward_terms_block(m: Metrics, env_cfg: dict, lo: int, hi: int) -> list[dict]:
    """Per term: logged value, effective weight, and the UNWEIGHTED rate per
    second while alive (logged * horizon / episode_s / weight). For bounded
    kernels (tracking terms) that rate reads as quality in [0, 1]."""
    dt = step_dt(env_cfg)
    horizon = episode_horizon_s(env_cfg)
    steps = _mean(m, "Train/mean_episode_length", lo, hi)
    ep_s = steps * dt if steps and dt else None
    weights = effective_weights(m, env_cfg)
    managed = set(curriculum_reward_terms(env_cfg).values())
    rows = []
    total_abs = 0.0
    for r in reward_table(env_cfg):
        logged = _mean(m, f"Episode_Reward/{r['term']}", lo, hi)
        if logged is None:
            continue
        w = weights.get(r["term"])
        row: dict[str, Any] = {"term": r["term"], "func": r["func"],
                               "weight_cfg": r["weight"], "weight_now": w,
                               "logged": round(logged, 4)}
        if r["term"] in managed:
            row["weight_set_by_curriculum"] = True
        if ep_s and horizon:
            per_s = logged * horizon / ep_s
            row["weighted_per_s_alive"] = round(per_s, 4)
            if isinstance(w, (int, float)) and abs(w) > 1e-12:
                row["raw_per_s_alive"] = round(per_s / w, 4)
        row["kind"] = "penalty" if penalty_expect_le_zero(r) else "reward"
        total_abs += abs(logged)
        rows.append(row)
    for row in rows:
        row["share_of_abs_total_pct"] = round(100 * abs(row["logged"]) / total_abs, 1) if total_abs else None
    rows.sort(key=lambda x: -abs(x["logged"]))
    return rows


def curriculum_events(m: Metrics, env_cfg: dict, it: int, keep: int = 4) -> list[dict]:
    """Stage changes grouped by iteration, with weight-invariant metrics around them."""
    weight_terms = curriculum_reward_terms(env_cfg)
    by_iter: dict[int, dict[str, list[float]]] = {}
    for tag in sorted(t for t in m.tags if t.startswith("Curriculum/")):
        prev = None
        for st, v in m.series(tag):
            if prev is not None and abs(v - prev) > 1e-9:
                by_iter.setdefault(st, {})[tag.removeprefix("Curriculum/")] = [round(prev, 4), round(v, 4)]
            prev = v
    # adjacent iterations (a ramp logged over 2 steps) are one event
    merged: list[tuple[int, dict]] = []
    for st in sorted(by_iter):
        if merged and st - merged[-1][0] <= 2:
            for k, ft in by_iter[st].items():
                if k in merged[-1][1]:
                    merged[-1][1][k][1] = ft[1]
                else:
                    merged[-1][1][k] = ft
        else:
            merged.append((st, dict(by_iter[st])))
    out = []
    for st, changes in merged[-keep:]:
        before = (max(0, st - EVENT_WINDOW), st - 1)
        after = (st + 1, min(it, st + EVENT_WINDOW))
        ev: dict[str, Any] = {"iter": st, "changes": changes}
        rescaled = sorted(weight_terms[k] for k in changes if k in weight_terms)
        if rescaled:
            ev["reward_weights_changed"] = rescaled
        around = {}
        for name, fn in (
            ("episode_s", lambda lo, hi: _ep_s(m, env_cfg, lo, hi)),
            ("fall_pct", lambda lo, hi: _pct(fall_share(m, lo, hi))),
            ("lin_vel_error", lambda lo, hi: _mean(m, "Metrics/twist/error_vel_xy", lo, hi)),
        ):
            b, a = fn(*before), (fn(*after) if after[1] >= after[0] else None)
            if b is not None or a is not None:
                around[name] = [_r(b), _r(a)]
        ev["before_after"] = around
        ev["iters_ago"] = it - st
        out.append(ev)
    return out


def _ep_s(m: Metrics, env_cfg: dict, lo: int, hi: int) -> float | None:
    steps = _mean(m, "Train/mean_episode_length", lo, hi)
    dt = step_dt(env_cfg)
    return steps * dt if steps is not None and dt else None


def _pct(v: float | None) -> float | None:
    return 100 * v if v is not None else None


def _r(v: float | None, nd: int = 3) -> float | None:
    return round(v, nd) if v is not None else None


def interventions_ledger(exp: dict, keep: int = 5) -> list[dict]:
    rows = []
    for inv in (exp.get("interventions") or [])[-keep:]:
        snap = inv.get("snapshot") or {}
        rows.append({
            "iter": inv.get("iter"),
            "kind": inv.get("kind"),
            "changes": [{"key": c.get("key"), "value": c.get("value")}
                        for c in inv.get("changes") or [] if isinstance(c, dict)],
            "previous": inv.get("previous"),
            "baseline": {k: _r(snap.get(k)) for k in ("ep_len_w", "fall_share_w", "error_vel_xy_w")
                         if snap.get(k) is not None} or None,
            "outcome": inv.get("outcome"),
            "inconclusive_reason": inv.get("inconclusive_reason"),
        })
    return rows


# --------------------------------------------------------------------- digest


def build_digest(segments: list[Metrics], env_cfg: dict, exp: dict, phase: str,
                 sec_per_iter: float | None = None) -> dict:
    """segments: the run's resume chain, oldest first (the live one last)."""
    current = segments[-1]
    m, resumes = merge_lineage(segments)
    it = current.iters
    start = _first_iter(current)
    warmup = bool(start > 0 and it - start < RESUME_WARMUP_ITERS)
    # During warm-up the episodic "now" comes from the stretch before the resume.
    now_hi = it
    if warmup and resumes:
        eps = [s for s, _ in m.series("Train/mean_episode_length")]
        now_hi = max((s for s in eps if s <= it), default=it)
    lo = max(0, now_hi - DIGEST_WINDOW + 1)
    trend_lo = max(0, it - 4 * DIGEST_WINDOW + 1)
    target = exp.get("target_iterations")
    last_inv = exp.get("last_intervention_iter")
    run = {
        "phase": phase,
        "iter": it,
        "segment_started_at_iter": start,
        "resumed_at_iters": resumes,
        "resume_warmup": warmup,
        "episodic_now_window": [lo, now_hi],
        "plan_total_iters": target,
        "plan_done_pct": round(100 * it / target, 1) if target else None,
        "iters_since_last_intervention": (it - last_inv) if isinstance(last_inv, int) else None,
        "sec_per_iter": round(sec_per_iter, 2) if sec_per_iter else None,
    }
    trends = {}
    for name, tag, better in (
        ("episode_length", "Train/mean_episode_length", True),
        ("lin_vel_error", "Metrics/twist/error_vel_xy", False),
        ("yaw_vel_error", "Metrics/twist/error_vel_yaw", False),
        ("policy_std", "Policy/mean_std", None),
        ("mean_reward_scale_dependent", "Train/mean_reward", True),
    ):
        tr = robust_trend(m, tag, trend_lo, it, better)
        if tr:
            trends[name] = tr
    fall_series_lo, fall_series_hi = trend_lo, it
    q = max(2, (fall_series_hi - fall_series_lo + 1) // 4)
    fs_a = fall_share(m, fall_series_lo, fall_series_lo + q - 1)
    fs_b = fall_share(m, fall_series_hi - q + 1, fall_series_hi)
    if fs_a is not None and fs_b is not None:
        trends["fall_pct"] = {"from": round(100 * fs_a, 1), "to": round(100 * fs_b, 1)}
    task = {label: _r(_mean(m, tag, lo, now_hi)) for tag, label in TASK_ERROR_TAGS.items()
            if _mean(m, tag, lo, now_hi) is not None}
    digest = {
        "run": run,
        "episode_now": episode_block(m, env_cfg, lo, now_hi),
        "task_errors_now": task,
        "trends": trends,
        "reward_terms_now": reward_terms_block(m, env_cfg, lo, now_hi),
        "curriculum_events": curriculum_events(m, env_cfg, it),
        "curriculum_managed_weights": sorted(curriculum_managed_keys(env_cfg)),
        "interventions": interventions_ledger(exp),
    }
    digest["facts"] = facts(digest)
    return digest


def facts(d: dict) -> list[str]:
    """Short sentences a small model reads before the numbers."""
    out: list[str] = []
    run, ep = d["run"], d["episode_now"]
    if run.get("plan_done_pct") is not None:
        out.append(f"Iteration {run['iter']} of {run['plan_total_iters']} "
                   f"({run['plan_done_pct']}% of the plan).")
    if run.get("resume_warmup"):
        lo, hi = run["episodic_now_window"]
        out.append(f"Training resumed at iter {run['segment_started_at_iter']} less than "
                   f"{RESUME_WARMUP_ITERS} iters ago: episode length and Episode_Reward of the new "
                   f"segment are not readable yet, so the episode numbers below are from iters "
                   f"{lo}..{hi}, before the resume.")
    if ep.get("mean_s") is not None:
        out.append(f"Episodes last {ep['mean_s']} s on average"
                   + (f" of a {ep['horizon_s']} s horizon ({ep['pct_of_horizon']}%)." if ep.get("horizon_s") else "."))
    if ep.get("episodes_ending_in_fall_pct") is not None:
        out.append(f"{ep['episodes_ending_in_fall_pct']}% of episode endings are falls "
                   "(the rest are time-outs or other terminations).")
    for name, tr in d["trends"].items():
        if name == "fall_pct" or name == "mean_reward_scale_dependent":
            continue
        out.append(f"{name}: {tr['from']} -> {tr['to']} over iters {tr['window'][0]}..{tr['window'][1]}, "
                   f"{tr['direction']} (noise-aware).")
    track = [r for r in d["reward_terms_now"] if r["term"].startswith("track_") and "raw_per_s_alive" in r]
    for r in track:
        out.append(f"{r['term']} kernel averages {r['raw_per_s_alive']} per second alive "
                   "(1.0 = perfect tracking).")
    for ev in d["curriculum_events"]:
        if ev.get("reward_weights_changed"):
            out.append(f"Curriculum at iter {ev['iter']} changed reward weights of "
                       f"{ev['reward_weights_changed']}: mean_reward is not comparable across it.")
    if d["curriculum_managed_weights"]:
        out.append(f"Weights set by a curriculum (changing them has no effect): "
                   f"{d['curriculum_managed_weights']}.")
    last = d["interventions"][-1] if d["interventions"] else None
    if last and last.get("outcome") in (None, "pending"):
        out.append(f"The last change (iter {last['iter']}: {last['changes']}) is still being "
                   "evaluated; give it time before stacking another change.")
    return out
