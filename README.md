# microduck training autopilot

Wake/sleep supervisor for the local mjlab/rsl_rl training run of the
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
PY=../microduck_rl/.venv/bin/python
$PY autopilot.py status                      # what is running, last verdict
$PY autopilot.py plan --force --recommend-only   # dry-plan an ended run
systemctl --user restart microduck-rl-autopilot  # daemon (creates reports/session-*.md)
```

## Layout

| path | what |
| - | - |
| `autopilot.py` | the whole autopilot (stdlib + tensorboard/yaml) |
| `autopilot.toml` | config: paths, safety allowlist, LLM, monitor cadence |
| `docs/autopilot.md` | reference: cycle, safety model, verdict semantics |
| `reports/` | one markdown journal per daemon session |
| `logs/` | state.json, status.json, recs, spawned-run stdout (gitignored) |
