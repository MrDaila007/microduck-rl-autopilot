# microduck training autopilot

Wake/sleep supervisor for the local mjlab/rsl_rl training runs of the
microduck robot: monitors tfevents, precomputes all analytics, asks the local
LLM (Ollama, `qwen3.5:35b-a3b-q4_K_M` on this machine) only to judge + plan,
and in auto mode applies allowlisted hyperparameter changes by itself —
validate → 64-env/5-iter smoke test → SIGINT stop → resume from the latest
checkpoint.

- Full reference: [docs/autopilot.md](docs/autopilot.md)
- The supervised training repo: `../microduck_rl` (`[paths].mjlab_repo`)
- Per-daemon-session journals: `reports/session-*.md`
- Runs as the systemd user unit `microduck-rl-autopilot` (Linger enabled)

## Quick start

```bash
autopilot status                      # what is running, queue, last verdict
autopilot scenarios                   # the whole training-scenario catalog
autopilot launch velstand             # start any registered scenario
autopilot plan --force --recommend-only   # dry-plan an ended run
autopilot dashboard                   # http://127.0.0.1:8765
apup                                  # start the daemon (alias for systemctl --user)
```

A wrapper lives at `~/.local/bin/autopilot` (on PATH); bash aliases (`ap`,
`aps`, `apl`, `apup`, `apdown`, `apst`, `aplog`, `aplaunch`) are in
`~/.bash_aliases`.

## Training scenarios

The mjlab registry holds the whole policy family. `autopilot scenarios` lists
33 of them by name — locomotion (`velocity`, `velstand` + `-rough`),
posture/episodic (`standup`, `sitstand`), tricks (`groundpick`, `ballkick`,
`roulade`), the roller-skate family (`rollers`, `swizzle`, `roller-crouch`,
`roller-slope`, `roller-standup`, `spin`) and every `-Backlash-` A/B twin.
Each can carry its own args / max-iterations / env / warm-start.

Chain a training program and the daemon walks it while idle:

```toml
[train]
default_scenario = "velocity"
queue = ["velocity", "velstand", "standup", "sitstand"]  # + auto.start_if_idle
```

A queued scenario is marked done when its plan completes; an `abort` verdict
parks it (never auto-relaunched). `autopilot reset --queue` re-enables.

## Layout

| path | what |
| - | - |
| `autopilot.py` | supervisor loop, smoke, relaunch, CLI. Re-exports `analysis` |
| `analysis.py` | metrics, reward audit, rules, recommendation checks |
| `autopilot.toml` | config: paths, scenario catalog + queue, safety allowlist, LLM, monitor cadence |
| `docs/autopilot.md` | reference: cycle, scenarios, safety model, verdict semantics |
| `reports/` | one markdown journal per daemon session |
| `logs/` | state.json, status.json, recs, spawned-run stdout (gitignored) |
