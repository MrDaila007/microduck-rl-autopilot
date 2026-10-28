#!/usr/bin/env python3
"""Local dashboard for the training autopilot.

The HTTP thread only reads status files and enqueues actions. The daemon
(or `autopilot dashboard` when the daemon is down) drains the inbox on its
own thread, so a smoke test never races a wake cycle.

Bind address is loopback only. Static files live in ./dashboard/.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import autopilot

WEB_DIR = Path(__file__).resolve().parent / "dashboard"
CHART_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/fell_over",
    "Loss/entropy",
    "Perf/total_fps",
)
APPLYABLE_STATUS = {"ready_to_apply", "ready_to_relaunch", "recommended"}
ACTION_TYPES = {
    "launch", "stop", "apply_rec", "manual_adjust", "reset",
    "set_queue", "set_config", "suggest", "plan",
}

# Dotted keys the settings form may write. Everything else stays in the file.
CONFIG_KIND = {
    "auto.enabled": "bool",
    "auto.start_if_idle": "bool",
    "monitor.tick_s": "int",
    "monitor.analyze_every_iters": "int",
    "monitor.min_iters_before_intervene": "int",
    "monitor.stagnation_window": "int",
    "monitor.stagnation_min_iters": "int",
    "monitor.stagnation_rel_gain": "float",
    "monitor.cooldown_iters": "int",
    "monitor.max_interventions": "int",
    "monitor.confidence_min": "float",
    "monitor.min_sleep_s": "int",
    "monitor.max_sleep_s": "int",
    "train.default_scenario": "str",
    "train.queue": "queue",
    "safety.max_changes": "int",
    "safety.ranges": "ranges",
    "llm.model": "str",
    "llm.temperature": "float",
    "llm.timeout_s": "int",
}
INT_BOUNDS = {
    "monitor.tick_s": (1, 86400),
    "monitor.analyze_every_iters": (1, 100_000),
    "monitor.min_iters_before_intervene": (0, 1_000_000),
    "monitor.stagnation_window": (2, 1_000_000),
    "monitor.stagnation_min_iters": (0, 1_000_000),
    "monitor.cooldown_iters": (0, 1_000_000),
    "monitor.max_interventions": (0, 1000),
    "monitor.min_sleep_s": (1, 86400),
    "monitor.max_sleep_s": (1, 86400),
    "safety.max_changes": (1, 20),
    "llm.timeout_s": (1, 3600),
}
FLOAT_BOUNDS = {
    "monitor.stagnation_rel_gain": (0.0, 10.0),
    "monitor.confidence_min": (0.0, 1.0),
    "llm.temperature": (0.0, 2.0),
}
FIXED_KNOBS = (
    ("agent.algorithm.learning-rate", "learning rate"),
    ("agent.algorithm.entropy-coef", "entropy coef"),
    ("env.scene.num-envs", "num envs"),
    ("agent.max-iterations", "max iterations"),
)

_SECTION = re.compile(r"^\[([^\]\s]+)\]\s*(?:#.*)?$")
_KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_RANGE_LINE = re.compile(r'^(\s*)("(?:\\.|[^"\\])*")\s*=\s*(.*)$')

_RUNTIME: "Runtime | None" = None
_METRIC_CACHE: dict[str, tuple[float, int, autopilot.Metrics | None]] = {}
_OLLAMA_CACHE: dict[str, Any] = {"t": 0.0, "v": {"ok": False, "detail": ""}}


@dataclass
class Runtime:
    cfg: autopilot.Config
    wake: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    inbox: deque = field(default_factory=deque)
    actions: dict[str, dict] = field(default_factory=dict)
    busy: bool = False
    config_mtime: float | None = None
    httpd: ThreadingHTTPServer | None = None
    bound_port: int | None = None


def runtime() -> Runtime | None:
    return _RUNTIME


def sync_paths(cfg: autopilot.Config) -> None:
    """Point the imported autopilot module at the repo this config names.

    `python autopilot.py` loads this file as a second module, so its
    MJLAB_REPO global is not the one main() already set.
    """
    if cfg.paths.mjlab_repo:
        autopilot.MJLAB_REPO = Path(cfg.paths.mjlab_repo).expanduser()
        autopilot.LOG_ROOT = autopilot.MJLAB_REPO / "logs" / "rsl_rl"


def attach(cfg: autopilot.Config) -> Runtime:
    global _RUNTIME
    sync_paths(cfg)
    rt = Runtime(cfg=cfg)
    try:
        rt.config_mtime = autopilot.CONFIG_FILE.stat().st_mtime
    except OSError:
        rt.config_mtime = None
    _RUNTIME = rt
    return rt


def detach() -> None:
    global _RUNTIME
    _RUNTIME = None


def check_bind_host(host: str) -> str:
    """Dashboard listens on loopback only. Anything else is a footgun on a training box."""
    cleaned = (host or "").strip().lower()
    if cleaned == "localhost":
        return "127.0.0.1"
    if cleaned in ("127.0.0.1", "::1"):
        return cleaned
    raise ValueError(f"dashboard host must be loopback, got {host!r}")


def url(cfg: autopilot.Config, bound_port: int | None = None) -> str:
    host = check_bind_host(cfg.dashboard.host)
    port = int(bound_port if bound_port else cfg.dashboard.port)
    return f"http://{host}:{port}"


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(rt: Runtime) -> None:
    host = check_bind_host(rt.cfg.dashboard.host)
    port = int(rt.cfg.dashboard.port)
    if port != 0 and not 1 <= port <= 65535:
        raise ValueError(f"bad dashboard port {port}")
    Handler.runtime = rt
    httpd = _Server((host, port), Handler)
    rt.httpd = httpd
    rt.bound_port = int(httpd.server_address[1])
    threading.Thread(target=httpd.serve_forever, name="dashboard", daemon=True).start()


def stop_server(rt: Runtime) -> None:
    httpd = rt.httpd
    if httpd is None:
        return
    httpd.shutdown()
    httpd.server_close()
    rt.httpd = None


def wait_wake(rt: Runtime, seconds: int, stop: dict) -> None:
    """Sleep in 1s steps so a posted action wakes the daemon immediately."""
    for _ in range(max(1, int(seconds))):
        if stop.get("flag") or rt.wake.is_set():
            break
        time.sleep(1)
    rt.wake.clear()


def reload_if_stale(rt: Runtime) -> None:
    path = autopilot.CONFIG_FILE
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    if rt.config_mtime is None:
        rt.config_mtime = mtime
        return
    if mtime != rt.config_mtime:
        rt.cfg = autopilot.load_config(path)
        rt.config_mtime = mtime
        autopilot.log.info("reloaded %s", path)


# ------------------------------------------------------------------- toml patch


def _split_comment(rest: str) -> tuple[str, str]:
    in_str = False
    quote = ""
    escape = False
    for i, ch in enumerate(rest):
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "#":
            return rest[:i].rstrip(), rest[i:]
    return rest.rstrip(), ""


def fmt_toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".8g")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(fmt_toml(item) for item in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to toml")


def _replace_key_line(line: str, formatted: str) -> str:
    match = _KEY_LINE.match(line.rstrip("\n"))
    if match is None:
        return line
    _value, comment = _split_comment(match.group(3))
    suffix = f"  {comment}" if comment else ""
    return f"{match.group(1)}{match.group(2)} = {formatted}{suffix}"


def _insert_pending(lines: list[str], pending: dict[str, list[str]]) -> list[str]:
    """Append missing keys at the end of their existing section."""
    out: list[str] = []
    section = ""
    for line in lines:
        match = _SECTION.match(line)
        if match:
            extra = pending.pop(section, None)
            if extra:
                out.extend(extra)
            section = match.group(1)
        out.append(line)
    extra = pending.pop(section, None)
    if extra:
        out.extend(extra)
    for sec, rows in pending.items():
        if out and out[-1] != "":
            out.append("")
        out.append(f"[{sec}]")
        out.extend(rows)
    return out


def apply_toml_updates(text: str, updates: dict[str, Any]) -> str:
    """Replace allowlisted assignments in place. Comments and other tables stay.

    A key that is not in the file yet is inserted at the end of its section.
    A safety.ranges key that is not already in the file is an error — the
    dashboard must not grow the allowlist of bounds.
    """
    simple: dict[tuple[str, str], str] = {}
    ranges = updates.get("safety.ranges") or {}
    if ranges and not isinstance(ranges, dict):
        raise ValueError("safety.ranges must be a table")
    for dotted, value in updates.items():
        if dotted == "safety.ranges":
            continue
        if dotted not in CONFIG_KIND:
            raise ValueError(f"key {dotted!r} is not editable")
        section, key = dotted.split(".", 1)
        simple[(section, key)] = fmt_toml(value)

    seen_simple: set[tuple[str, str]] = set()
    seen_ranges: set[str] = set()
    out: list[str] = []
    section = ""
    for line in text.splitlines():
        header = _SECTION.match(line)
        if header:
            section = header.group(1)
            out.append(line)
            continue
        if section == "safety.ranges":
            range_line = _RANGE_LINE.match(line)
            if range_line:
                try:
                    range_key = json.loads(range_line.group(2))
                except json.JSONDecodeError:
                    range_key = None
                if isinstance(range_key, str):
                    seen_ranges.add(range_key)
                    if range_key in ranges:
                        _value, comment = _split_comment(range_line.group(3))
                        suffix = f"  {comment}" if comment else ""
                        formatted = fmt_toml(ranges[range_key])
                        out.append(
                            f"{range_line.group(1)}{json.dumps(range_key)} = {formatted}{suffix}"
                        )
                        continue
        key_line = _KEY_LINE.match(line)
        if key_line and (section, key_line.group(2)) in simple:
            out.append(_replace_key_line(line, simple[(section, key_line.group(2))]))
            seen_simple.add((section, key_line.group(2)))
            continue
        out.append(line)

    missing_ranges = [key for key in ranges if key not in seen_ranges]
    if missing_ranges:
        raise ValueError("unknown safety range: " + ", ".join(missing_ranges))

    pending: dict[str, list[str]] = {}
    for (sec, key), formatted in simple.items():
        if (sec, key) not in seen_simple:
            pending.setdefault(sec, []).append(f"{key} = {formatted}")
    if pending:
        out = _insert_pending(out, pending)
    body = "\n".join(out)
    if not text or text.endswith("\n"):
        body += "\n"
    return body


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def config_errors(cfg: autopilot.Config, updates: dict) -> list[str]:
    """Reject a settings payload before it touches autopilot.toml."""
    if not isinstance(updates, dict) or not updates:
        return ["empty config update"]
    errors: list[str] = []
    cleaned: dict[str, Any] = {}
    for key, value in updates.items():
        kind = CONFIG_KIND.get(str(key))
        if kind is None:
            errors.append(f"key {key!r} is not editable")
            continue
        if kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"{key} must be true or false")
            else:
                cleaned[key] = value
        elif kind == "int":
            number = _coerce_int(value)
            lo, hi = INT_BOUNDS[key]
            if number is None or not lo <= number <= hi:
                errors.append(f"{key} outside [{lo}, {hi}]")
            else:
                cleaned[key] = number
        elif kind == "float":
            number = _coerce_float(value)
            lo, hi = FLOAT_BOUNDS[key]
            if number is None or not lo <= number <= hi:
                errors.append(f"{key} outside [{lo}, {hi}]")
            else:
                cleaned[key] = number
        elif kind == "str":
            if not isinstance(value, str):
                errors.append(f"{key} must be a string")
                continue
            text = value.strip()
            if key == "llm.model" and (not text or len(text) > 200 or "\n" in text):
                errors.append("llm.model must be a non-empty single line")
            elif key == "train.default_scenario" and text and text not in (cfg.train.scenarios or {}):
                errors.append(f"unknown scenario {text!r}")
            else:
                cleaned[key] = text
        elif kind == "queue":
            if not isinstance(value, list) or len(value) > 64:
                errors.append("train.queue must be a list of at most 64 scenarios")
                continue
            names = []
            seen: set[str] = set()
            for item in value:
                if not isinstance(item, str) or item not in (cfg.train.scenarios or {}):
                    errors.append(f"unknown scenario {item!r}")
                    continue
                if item in seen:
                    errors.append(f"duplicate scenario {item!r}")
                    continue
                seen.add(item)
                names.append(item)
            cleaned[key] = names
        elif kind == "ranges":
            if not isinstance(value, dict):
                errors.append("safety.ranges must be an object")
                continue
            known = cfg.safety.ranges or {}
            parsed: dict[str, list[float]] = {}
            for range_key, bounds in value.items():
                if range_key not in known:
                    errors.append(f"unknown safety range {range_key!r}")
                    continue
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bounds)
                ):
                    errors.append(f"range {range_key} must be [lo, hi]")
                    continue
                lo, hi = float(bounds[0]), float(bounds[1])
                if not (lo <= hi) or lo != lo or hi != hi:
                    errors.append(f"range {range_key} has lo > hi")
                    continue
                parsed[str(range_key)] = [lo, hi]
            cleaned[key] = parsed
    min_sleep = cleaned.get("monitor.min_sleep_s", cfg.monitor.min_sleep_s)
    max_sleep = cleaned.get("monitor.max_sleep_s", cfg.monitor.max_sleep_s)
    if _coerce_int(min_sleep) is not None and _coerce_int(max_sleep) is not None:
        if int(min_sleep) > int(max_sleep):
            errors.append("min_sleep_s is greater than max_sleep_s")
    return errors


def patch_config_file(path: Path, updates: dict[str, Any]) -> None:
    text = path.read_text() if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = apply_toml_updates(text, updates)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new)
    tmp.replace(path)


# ------------------------------------------------------------------------ reads


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def visible_runs(state: dict) -> list[autopilot.Run]:
    ignore = set(state.get("ignore_runs") or [])
    return [run for run in autopilot.find_runs() if run.dir.name not in ignore]


def load_metrics(run: autopilot.Run) -> autopilot.Metrics | None:
    events = run.newest_tfevents
    if events is None:
        return None
    try:
        stat = events.stat()
    except OSError:
        return None
    key = str(events)
    hit = _METRIC_CACHE.get(key)
    if hit and hit[0] == stat.st_mtime and hit[1] == stat.st_size:
        return hit[2]
    try:
        loaded = autopilot.Metrics.load(run)
    except Exception as exc:
        autopilot.log.warning("metrics load failed for %s: %s", run.dir.name, exc)
        _METRIC_CACHE[key] = (stat.st_mtime, stat.st_size, None)
        return None
    if loaded is not None:
        _METRIC_CACHE[key] = (stat.st_mtime, stat.st_size, loaded)
    return loaded


def ollama_status(cfg: autopilot.Config) -> dict:
    now = time.time()
    if now - float(_OLLAMA_CACHE["t"]) < 15:
        return dict(_OLLAMA_CACHE["v"])
    detail = "unreachable"
    ok = False
    try:
        with urllib.request.urlopen(cfg.llm.base_url.rstrip("/") + "/api/version", timeout=1) as resp:
            detail = resp.read().decode(errors="replace")[:200]
            ok = True
    except Exception as exc:
        detail = str(exc)
    value = {"ok": ok, "detail": detail}
    _OLLAMA_CACHE["t"] = now
    _OLLAMA_CACHE["v"] = value
    return dict(value)


def _current_action(rt: Runtime) -> dict | None:
    with rt.lock:
        for action in rt.actions.values():
            if action.get("status") == "running":
                return {"id": action["id"], "type": action["type"], "status": "running"}
        if rt.inbox:
            action = rt.inbox[0]
            return {"id": action["id"], "type": action["type"], "status": action.get("status")}
    return None


def build_overview(rt: Runtime) -> dict:
    state = autopilot.load_state()
    status = autopilot.load_json(autopilot.STATUS_FILE, {})
    if not isinstance(status, dict):
        status = {}
    proc = autopilot.find_train_proc(state)
    exp_name = state.get("current_experiment")
    exp = (state.get("experiments") or {}).get(exp_name) or {} if exp_name else {}
    if not isinstance(exp, dict):
        exp = {}
    return _sanitize({
        "phase": status.get("phase"),
        "status": status.get("status"),
        "verdict": status.get("verdict") or exp.get("last_verdict"),
        "diagnosis": status.get("diagnosis") or exp.get("last_diagnosis"),
        "sec_per_iter": status.get("sec_per_iter"),
        "eta_s": status.get("eta_s"),
        "next_check_iter": status.get("next_check_iter"),
        "sleep_s": status.get("sleep_s"),
        "iters": status.get("iters"),
        "run": status.get("run"),
        "experiment": exp_name,
        "auto": {"enabled": bool(rt.cfg.auto.enabled), "start_if_idle": bool(rt.cfg.auto.start_if_idle)},
        "intervening": bool(state.get("intervening")),
        "intervening_since": state.get("intervening_since"),
        "needs_human": bool(exp.get("needs_human")),
        "needs_human_reason": exp.get("needs_human_reason"),
        "budget": {
            "used": exp.get("budget_used", 0),
            "max": rt.cfg.monitor.max_interventions,
        },
        "training": {"pid": proc["pid"]} if proc else None,
        "ollama": ollama_status(rt.cfg),
        "busy": bool(rt.busy or rt.inbox),
        "action": _current_action(rt),
        "llm_model": rt.cfg.llm.model,
        "time": status.get("time"),
    })


def _allowed(cfg: autopilot.Config, key: str) -> bool:
    if not any(autopilot.fnmatch.fnmatch(key, pat) for pat in cfg.safety.allow):
        return False
    if any(autopilot.fnmatch.fnmatch(key, pat) for pat in cfg.safety.deny):
        return False
    return True


def _bounds(cfg: autopilot.Config, key: str) -> list[float] | None:
    for pattern, bounds in (cfg.safety.ranges or {}).items():
        if autopilot.fnmatch.fnmatch(key, pattern) and isinstance(bounds, list) and len(bounds) == 2:
            return [float(bounds[0]), float(bounds[1])]
    return None


def _knob(cfg: autopilot.Config, key: str, label: str, value: Any) -> dict:
    return {
        "key": key,
        "label": label,
        "value": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        "bounds": _bounds(cfg, key),
    }


def knobs(cfg: autopilot.Config, env_cfg: dict, agent_cfg: dict) -> list[dict]:
    rows = []
    for reward in autopilot.reward_table(env_cfg):
        key = f"env.rewards.{reward['term']}.weight"
        if _allowed(cfg, key):
            rows.append(_knob(cfg, key, reward["term"], reward.get("weight")))
    for key, label in FIXED_KNOBS:
        if _allowed(cfg, key):
            rows.append(_knob(cfg, key, label, autopilot.lookup_cfg_value(key, env_cfg, agent_cfg)))
    return rows


def build_run(rt: Runtime) -> dict:
    state = autopilot.load_state()
    runs = visible_runs(state)
    if not runs:
        return {"status": "no_runs"}
    run = runs[-1]
    env_cfg, agent_cfg = autopilot.run_cfg(run)
    exp = (state.get("experiments") or {}).get(run.experiment) or {}
    if not isinstance(exp, dict):
        exp = {}
    target = exp.get("target_iterations") or agent_cfg.get("max_iterations")
    payload: dict[str, Any] = {
        "status": "ok",
        "name": run.name,
        "experiment": run.experiment,
        "age_s": round(run.age_s, 1),
        "active": run.age_s <= rt.cfg.monitor.active_after_s,
        "target_iterations": target,
        "knobs": knobs(rt.cfg, env_cfg, agent_cfg),
        "max_changes": rt.cfg.safety.max_changes,
    }
    metrics = load_metrics(run)
    if metrics is None or metrics.iters < 1:
        payload["status"] = "no_metrics"
        payload["iters"] = 0
        return _sanitize(payload)
    it = metrics.iters
    payload.update({
        "iters": it,
        "mean_reward": metrics.last("Train/mean_reward"),
        "health": autopilot.health_report(metrics, it),
        "plateau": autopilot.plateau_report(metrics, rt.cfg),
        "findings": [
            {"code": item.code, "severity": item.severity, "detail": item.detail}
            for item in autopilot.run_rules(metrics, env_cfg, rt.cfg)
        ],
        "penalty_audit": autopilot.penalty_audit(metrics, env_cfg, rt.cfg, it),
        "reward_mass": autopilot.reward_mass(metrics, env_cfg, it, rt.cfg.monitor.stagnation_window),
        "progress": autopilot.checkpoints_report(run, target if isinstance(target, int) else None),
        "curriculum": autopilot.curriculum_pacing(metrics, it),
    })
    return _sanitize(payload)


def build_series(rt: Runtime) -> dict:
    state = autopilot.load_state()
    runs = visible_runs(state)
    if not runs:
        return {"status": "no_runs", "series": {}}
    run = runs[-1]
    metrics = load_metrics(run)
    if metrics is None:
        return {"status": "no_metrics", "run": run.name, "series": {}}
    it = metrics.iters
    return _sanitize({
        "status": "ok",
        "run": run.name,
        "iters": it,
        "series": {tag: metrics.samples(tag, 0, it, 200) for tag in CHART_TAGS},
    })


def build_queue(rt: Runtime) -> dict:
    state = autopilot.load_state()
    queue = state.get("queue") or {}
    done = set(queue.get("completed") or [])
    aborted = set(queue.get("aborted") or [])
    current = queue.get("current")
    items = []
    for name in rt.cfg.train.queue or []:
        spec = autopilot.scenario_spec(rt.cfg, name) or {}
        if name in done:
            mark = "done"
        elif name in aborted:
            mark = "aborted"
        elif current == name:
            mark = "running"
        else:
            mark = "pending"
        items.append({"name": name, "task": spec.get("task"), "status": mark})
    scenarios = []
    for name, spec in sorted((rt.cfg.train.scenarios or {}).items()):
        scenarios.append({
            "name": name,
            "task": spec.get("task"),
            "max_iterations": spec.get("max_iterations"),
            "warm_start_from": spec.get("warm_start_from"),
        })
    return {
        "default_scenario": rt.cfg.train.default_scenario,
        "queue": items,
        "scenarios": scenarios,
        "current": current,
        "next": autopilot.queue_next(rt.cfg, state),
    }


def _applyable(rec: dict) -> bool:
    status = rec.get("status")
    if status in APPLYABLE_STATUS:
        return True
    return status == "needs_human" and rec.get("verdict") == "adjust"


def build_recs(limit: int = 20) -> dict:
    directory = autopilot.RECS_DIR
    if not directory.exists():
        return {"recs": []}
    files = sorted(directory.glob("*.json"), key=lambda path: path.name, reverse=True)[:limit]
    rows = []
    for path in files:
        rec = autopilot.load_json(path, {})
        if not isinstance(rec, dict):
            continue
        changes = rec.get("changes") if isinstance(rec.get("changes"), list) else []
        errors = rec.get("errors") if isinstance(rec.get("errors"), list) else []
        rows.append({
            "file": path.name,
            "time": rec.get("time"),
            "status": rec.get("status"),
            "verdict": rec.get("verdict"),
            "phase": rec.get("phase"),
            "kind": rec.get("kind"),
            "diagnosis": str(rec.get("diagnosis") or "")[:400],
            "changes": changes,
            "errors": errors,
            "smoke": rec.get("smoke"),
            "applyable": _applyable(rec),
        })
    return _sanitize({"recs": rows})


def build_config(rt: Runtime) -> dict:
    cfg = rt.cfg
    return {
        "auto": {"enabled": bool(cfg.auto.enabled), "start_if_idle": bool(cfg.auto.start_if_idle)},
        "monitor": {key.split(".", 1)[1]: getattr(cfg.monitor, key.split(".", 1)[1])
                    for key, kind in CONFIG_KIND.items()
                    if key.startswith("monitor.") and kind in ("int", "float")},
        "train": {
            "default_scenario": cfg.train.default_scenario,
            "queue": list(cfg.train.queue or []),
        },
        "safety": {
            "max_changes": cfg.safety.max_changes,
            "ranges": {key: [float(bounds[0]), float(bounds[1])]
                       for key, bounds in (cfg.safety.ranges or {}).items()
                       if isinstance(bounds, list) and len(bounds) == 2},
        },
        "llm": {
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "timeout_s": cfg.llm.timeout_s,
        },
        "scenarios": sorted((cfg.train.scenarios or {}).keys()),
    }


def _tail(path: Path | None, lines: int) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    chunk = data[-65536:] if len(data) > 65536 else data
    text = chunk.decode(errors="replace").splitlines()
    return "\n".join(text[-lines:])


def build_log(which: str, tail: int) -> dict:
    tail = max(1, min(int(tail), 500))
    if which == "train":
        files = sorted(autopilot.STATE_DIR.glob("train-*.log"))
        path = files[-1] if files else None
    else:
        which = "autopilot"
        path = autopilot.STATE_DIR / "autopilot.log"
    return {"which": which, "path": str(path) if path else "", "text": _tail(path, tail)}


# ----------------------------------------------------------------------- actions


def _new_action(kind: str, payload: dict) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    action = dict(payload)
    action.update({
        "id": f"{stamp}-{uuid.uuid4().hex[:4]}",
        "type": kind,
        "status": "queued",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result": None,
        "error": None,
    })
    return action


def normalize_changes(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("changes must be a non-empty list")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("change must be an object")
        key = str(item.get("key", "")).lstrip("-").strip()
        if not key:
            raise ValueError("change is missing a key")
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"value for {key} is not numeric")
        if isinstance(value, str):
            try:
                value = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError as exc:
                raise ValueError(f"value for {key} is not numeric") from exc
        if isinstance(value, float) and value.is_integer() and key in (
            "agent.max-iterations", "env.scene.num-envs",
        ):
            value = int(value)
        out.append({"key": key, "value": value})
    return out


def manual_errors(cfg: autopilot.Config, state: dict, changes: list[dict]) -> list[str]:
    """Same gate the daemon uses: allowlist, ranges, max_changes, no shorter horizon."""
    runs = visible_runs(state)
    target = None
    if runs:
        exp = (state.get("experiments") or {}).get(runs[-1].experiment) or {}
        if isinstance(exp, dict):
            target = exp.get("target_iterations")
    rec = {
        "verdict": "adjust",
        "confidence": 1.0,
        "changes": changes,
        "kind": "manual",
    }
    _ok, errors = autopilot.validate_rec(rec, cfg, target_iterations=target)
    return errors


def _safe_rec_name(name: str) -> Path:
    filename = Path(str(name)).name
    if not filename or filename != str(name) or not filename.endswith(".json"):
        raise ValueError("recommendation file must be a name inside logs/recommendations")
    path = (autopilot.RECS_DIR / filename).resolve()
    if path.parent != autopilot.RECS_DIR.resolve() or not path.is_file():
        raise ValueError(f"no recommendation {filename}")
    return path


def submit(rt: Runtime, body: Any) -> tuple[int, dict]:
    """Queue one action. 400 is a bad payload, 409 means a smoke or another action owns the loop."""
    if not isinstance(body, dict):
        return 400, {"error": "body must be an object"}
    kind = body.get("type")
    if kind not in ACTION_TYPES:
        return 400, {"error": f"unknown action {kind!r}"}
    try:
        payload = _validate_submit(rt, kind, body)
    except ValueError as exc:
        return 400, {"error": str(exc), "errors": [str(exc)]}
    state = autopilot.load_state()
    with rt.lock:
        if state.get("intervening") or rt.busy or rt.inbox:
            current = _current_action_locked(rt)
            return 409, {"error": "autopilot is busy", "action": current}
        action = _new_action(kind, payload)
        rt.actions[action["id"]] = action
        rt.inbox.append(action)
    rt.wake.set()
    autopilot.log.info("dashboard queued %s %s", kind, action["id"])
    return 202, {"id": action["id"], "status": "queued", "type": kind}


def _current_action_locked(rt: Runtime) -> dict | None:
    for action in rt.actions.values():
        if action.get("status") == "running":
            return {"id": action["id"], "type": action["type"], "status": "running"}
    if rt.inbox:
        action = rt.inbox[0]
        return {"id": action["id"], "type": action["type"], "status": action.get("status")}
    return None


def _validate_submit(rt: Runtime, kind: str, body: dict) -> dict:
    cfg = rt.cfg
    if kind == "launch":
        scenario = body.get("scenario")
        if scenario in ("", None):
            return {"scenario": None}
        if not isinstance(scenario, str) or scenario not in (cfg.train.scenarios or {}):
            raise ValueError(f"unknown scenario {scenario!r}")
        return {"scenario": scenario}
    if kind == "stop":
        return {"park": bool(body.get("park"))}
    if kind == "apply_rec":
        name = body.get("file") or body.get("rec")
        if not isinstance(name, str):
            raise ValueError("file is required")
        _safe_rec_name(name)
        return {"file": Path(name).name}
    if kind == "manual_adjust":
        changes = normalize_changes(body.get("changes"))
        errors = manual_errors(cfg, autopilot.load_state(), changes)
        if errors:
            raise ValueError("; ".join(errors))
        return {"changes": changes}
    if kind == "reset":
        experiment = body.get("experiment")
        if experiment is not None and not isinstance(experiment, str):
            raise ValueError("experiment must be a string")
        return {"experiment": experiment or None, "queue": bool(body.get("queue"))}
    if kind == "set_queue":
        updates: dict[str, Any] = {"train.queue": body.get("queue") if isinstance(body.get("queue"), list) else []}
        if body.get("default_scenario") is not None:
            updates["train.default_scenario"] = body.get("default_scenario")
        errors = config_errors(cfg, updates)
        if errors:
            raise ValueError("; ".join(errors))
        return {"values": updates}
    if kind == "set_config":
        values = body.get("values")
        if not isinstance(values, dict):
            raise ValueError("values must be an object")
        errors = config_errors(cfg, values)
        if errors:
            raise ValueError("; ".join(errors))
        return {"values": values}
    if kind in ("suggest", "plan"):
        return {}
    raise ValueError(f"unknown action {kind!r}")


def drain_one(rt: Runtime, state: dict) -> dict | None:
    """Run the oldest queued action. A second call is required for the next one.

    While a smoke test holds `intervening`, the action stays queued.
    """
    if state.get("intervening"):
        return None
    with rt.lock:
        if not rt.inbox:
            return None
        action = rt.inbox.popleft()
        rt.busy = True
        action["status"] = "running"
    try:
        result = execute(rt, state, action)
        action["status"] = "done"
        action["result"] = _sanitize(result)
        status = result.get("status") if isinstance(result, dict) else None
        return {
            "phase": "dashboard",
            "status": f"action_{action['type']}",
            "action": action["id"],
            "result_status": status,
        }
    except Exception as exc:
        action["status"] = "failed"
        action["error"] = f"{type(exc).__name__}: {exc}"
        autopilot.log.exception("dashboard action %s failed", action.get("id"))
        return {"phase": "dashboard", "status": "action_failed", "error": action["error"]}
    finally:
        with rt.lock:
            rt.busy = False


def execute(rt: Runtime, state: dict, action: dict) -> dict:
    kind = action.get("type")
    cfg = rt.cfg
    if kind == "launch":
        return execute_launch(cfg, state, action.get("scenario"))
    if kind == "stop":
        return execute_stop(cfg, state, bool(action.get("park")))
    if kind == "apply_rec":
        return execute_apply(cfg, state, str(action.get("file") or ""))
    if kind == "manual_adjust":
        return execute_manual(cfg, state, list(action.get("changes") or []))
    if kind == "reset":
        return execute_reset(state, action.get("experiment"), bool(action.get("queue")))
    if kind in ("set_queue", "set_config"):
        return execute_set_config(rt, dict(action.get("values") or {}))
    if kind == "suggest":
        return autopilot.analyze_once(cfg, state, force=True, apply=None)
    if kind == "plan":
        return execute_plan(cfg, state)
    return {"status": "unknown_action", "type": kind}


def execute_launch(cfg: autopilot.Config, state: dict, scenario: str | None) -> dict:
    import shlex

    if autopilot.find_train_proc(state):
        return {"status": "already_running"}
    extra_env: dict[str, str] = {}
    name = scenario
    if name:
        launch = autopilot.scenario_launch(cfg, name)
        if launch is None:
            return {"status": "unknown_scenario", "scenario": name}
        argv, extra_env = launch
    elif cfg.train.default_scenario or cfg.train.scenarios:
        name = cfg.train.default_scenario or next(iter(cfg.train.scenarios), "")
        launch = autopilot.scenario_launch(cfg, name) if name else None
        if launch is None:
            if not cfg.train.command:
                return {"status": "no_scenario"}
            argv = shlex.split(cfg.train.command)
            name = None
        else:
            argv, extra_env = launch
    elif cfg.train.command:
        argv = shlex.split(cfg.train.command)
        name = None
    else:
        return {"status": "no_scenario"}
    log_file = autopilot.STATE_DIR / f"train-{time.strftime('%Y%m%d-%H%M%S')}.log"
    repo = Path(cfg.paths.mjlab_repo).expanduser() if cfg.paths.mjlab_repo else autopilot.MJLAB_REPO
    pid = autopilot.spawn(argv, repo, log_file, {**autopilot.train_env(cfg), **extra_env})
    state["last_train_argv"] = argv
    if name and name in (cfg.train.queue or []):
        queue = state.setdefault("queue", {"completed": []})
        queue["current"] = name
    autopilot.save_state(state)
    autopilot.log.info("dashboard launched pid=%s scenario=%s", pid, name)
    return {"status": "launched", "pid": pid, "scenario": name, "argv": " ".join(argv)}


def execute_stop(cfg: autopilot.Config, state: dict, park: bool) -> dict:
    proc = autopilot.find_train_proc(state)
    if not proc:
        return {"status": "no_process"}
    argv = list(proc.get("cmdline") or [])
    before = list((state.get("queue") or {}).get("aborted") or [])
    autopilot.stop_training(proc, cfg, state)
    parked = False
    if park:
        autopilot.queue_note_abort(cfg, state, argv)
        after = list((state.get("queue") or {}).get("aborted") or [])
        parked = after != before
    return {"status": "stopped", "parked": parked}


def execute_apply(cfg: autopilot.Config, state: dict, name: str) -> dict:
    path = _safe_rec_name(name)
    rec = autopilot.load_json(path, {})
    if not isinstance(rec, dict) or not rec:
        return {"status": "no_rec"}
    runs = visible_runs(state)
    if not runs:
        return {"status": "no_runs"}
    # Explicit apply relaunches after smoke even when auto mode is off, and
    # does not spend the unattended intervention budget.
    out = autopilot.do_intervention(
        rec, runs[-1], cfg, state, phase=rec.get("phase") or "monitoring", force_apply=True,
    )
    autopilot.save_state(state)
    return out


def execute_manual(cfg: autopilot.Config, state: dict, changes: list[dict]) -> dict:
    errors = manual_errors(cfg, state, changes)
    if errors:
        return {"status": "rejected", "errors": errors}
    runs = visible_runs(state)
    if not runs:
        return {"status": "no_runs"}
    run = runs[-1]
    before = (autopilot.exp_state(state, run.experiment).get("budget_used", 0))
    proc = autopilot.find_train_proc(state)
    phase = "monitoring" if proc or run.age_s <= cfg.monitor.active_after_s else "post_run"
    rec = {
        "verdict": "adjust",
        "confidence": 1.0,
        "diagnosis": "manual dashboard adjust",
        "changes": changes,
        "kind": "manual",
        "notes": "dashboard",
    }
    out = autopilot.do_intervention(rec, run, cfg, state, phase=phase, force_apply=True)
    exp = autopilot.exp_state(state, run.experiment)
    if exp.get("budget_used", 0) != before:
        exp["budget_used"] = before
    autopilot.save_state(state)
    return out


def execute_reset(state: dict, experiment: str | None, queue: bool) -> dict:
    targets = [experiment] if experiment else list(state.get("experiments", {}))
    for name in targets:
        exp = autopilot.exp_state(state, name)
        exp["budget_used"] = 0
        exp["last_verdict"] = None
        exp["post_runs"] = []
        exp["completed_runs"] = []
    if queue:
        state["queue"] = {"completed": [], "current": None}
    autopilot.save_state(state)
    return {"status": "reset", "experiments": targets, "queue": bool(queue)}


def execute_set_config(rt: Runtime, updates: dict) -> dict:
    errors = config_errors(rt.cfg, updates)
    if errors:
        return {"status": "rejected", "errors": errors}
    try:
        patch_config_file(autopilot.CONFIG_FILE, updates)
    except ValueError as exc:
        return {"status": "rejected", "errors": [str(exc)]}
    rt.cfg = autopilot.load_config(autopilot.CONFIG_FILE)
    try:
        rt.config_mtime = autopilot.CONFIG_FILE.stat().st_mtime
    except OSError:
        rt.config_mtime = None
    autopilot.log.info("dashboard wrote config keys: %s", ", ".join(sorted(updates)))
    return {"status": "saved", "keys": sorted(updates)}


def execute_plan(cfg: autopilot.Config, state: dict) -> dict:
    runs = visible_runs(state)
    if not runs:
        return {"status": "no_runs"}
    run = runs[-1]
    exp = autopilot.exp_state(state, run.experiment)
    if run.dir.name in exp["post_runs"]:
        exp["post_runs"].remove(run.dir.name)
        autopilot.save_state(state)
    approve = True if cfg.auto.enabled else False
    return autopilot.post_run_phase(cfg, state, run, force=True, approve=approve)


# -------------------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    runtime: Runtime
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def log_message(self, fmt: str, *args: Any) -> None:
        autopilot.log.debug("dashboard " + fmt, *args)

    def _route(self, method: str) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            if method == "GET":
                self._get(path, query)
            elif method == "POST" and path == "/api/actions":
                length = int(self.headers.get("Content-Length") or 0)
                if length < 0 or length > 1_000_000:
                    self._send(400, {"error": "body too large"})
                    return
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": "invalid json"})
                    return
                code, payload = submit(self.runtime, body)
                self._send(code, payload)
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            autopilot.log.exception("dashboard request failed")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _get(self, path: str, query: dict[str, list[str]]) -> None:
        rt = self.runtime
        if path in ("/", "/index.html"):
            self._file("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._file("app.js", "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self._file("style.css", "text/css; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, {})
        elif path == "/api/overview":
            self._send(200, build_overview(rt))
        elif path == "/api/run":
            self._send(200, build_run(rt))
        elif path == "/api/series":
            self._send(200, build_series(rt))
        elif path == "/api/queue":
            self._send(200, build_queue(rt))
        elif path == "/api/recs":
            limit = _query_int(query, "limit", 20, 1, 100)
            self._send(200, build_recs(limit))
        elif path == "/api/config":
            self._send(200, build_config(rt))
        elif path == "/api/log":
            which = (query.get("which") or ["autopilot"])[0]
            tail = _query_int(query, "tail", 200, 1, 500)
            self._send(200, build_log(which, tail))
        elif path == "/api/actions":
            with rt.lock:
                rows = [
                    {"id": item["id"], "type": item["type"], "status": item["status"]}
                    for item in list(rt.actions.values())[-20:]
                ]
            self._send(200, {"actions": rows})
        elif path.startswith("/api/actions/"):
            action_id = path[len("/api/actions/"):]
            with rt.lock:
                action = rt.actions.get(action_id)
            if action is None:
                self._send(404, {"error": "no such action"})
            else:
                self._send(200, _sanitize({
                    "id": action["id"],
                    "type": action["type"],
                    "status": action["status"],
                    "error": action.get("error"),
                    "result": action.get("result"),
                }))
        else:
            self._send(404, {"error": "not found"})

    def _file(self, name: str, content_type: str) -> None:
        path = WEB_DIR / name
        if not path.is_file():
            self._send(404, {"error": f"missing {name}"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send(self, code: int, payload: dict) -> None:
        if code == 204:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = json.dumps(_sanitize(payload), ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def _query_int(query: dict[str, list[str]], key: str, default: int, lo: int, hi: int) -> int:
    raw = (query.get(key) or [None])[0]
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))
