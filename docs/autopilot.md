# Training autopilot — wake/sleep loop around a local rl run

`autopilot.py` (this repo) supervises a mjlab/rsl_rl training run in another
local repo (`[paths].mjlab_repo` in `autopilot.toml`, default
`../microduck_rl`): it watches that repo's tfevents, precomputes ALL the
analytics, asks the local LLM (Ollama, `qwen3.5:35b-a3b-q4_K_M`) only for
judgment + plan, and (auto mode) applies allowlisted overrides by itself —
smoke test → stop → resume from the latest checkpoint. Runs as the systemd
user unit `microduck-rl-autopilot`.

## The cycle

```
wake
 ├─ training process alive / run dir fresh?
 │    YES → monitoring phase
 │          read tfevents → analytics → rules engine
 │          → LLM verdict: keep | adjust | abort
 │          keep   → record, sleep until the next check phase
 │          adjust → validate → smoke test → SIGINT stop → relaunch from ckpt
 │          abort  → stop for human review (or rules-engine kill on critical)
 │    NO  → run ended?
 │          not planned yet → post_run phase: analyze the WHOLE run,
 │          LLM plans (done | restart same cfg | continue-with-changes),
 │          relaunch if the plan says so
 │          planned → idle (launch train.command only if auto.start_if_idle)
 └─ sleep: ETA-aware (see below), then wake again
```

The daemon never exits on analysis errors (one bad cycle logs a traceback and
the loop continues); it never kills training on shutdown, and it reaps its own
children so zombies can't fake a live process.

## Sleep planning (ETA-aware)

`monitor.min_sleep_s`..`max_sleep_s` bound the sleep; the target instant is
derived from the **measured** iteration rate (tfevents carry `wall_time`):

- `sec_per_iter` = median of the last ~25 iteration deltas,
- next wake lands on the analysis cadence (`analyze_every_iters`), nudged to
  fire just after a checkpoint save when a save boundary is close — an
  intervention then loses ~0 iterations (rsl_rl only writes `model_N.pt` every
  `save_interval`),
- `eta_s` = time from the current iteration to the experiment's
  `target_iterations` (the plan's end), reported in `status.json`.

`status.json` (one object per cycle): `phase`, `status`, `verdict`, `sec_per_iter`,
`eta_s`, `next_check_iter`, `sleep_s`.

## What the LLM gets (and what it does NOT)

Everything deterministic is computed in `build_analysis()`:

| block | contents |
| - | - |
| `plateau` | window slope, relative gain, flat flag |
| `trends_last_vs_prev_window` | per-tag mean/last/slope + rel change vs previous window (rewards, terminations, entropy, fps, curriculum, task errors) |
| `penalty_audit` | every penalty term: weight, mean, max, `VIOLATION` if it logs > `penalty_eps` (double-negation detector) |
| `reward_mass` | positive-reward terms by MASS (share of positive sum), not weight |
| `curriculum` | stage changes + `pacing_warnings` (metric stepping DOWN at a boundary) |
| `health` | fell_over/time_out/nan_state, entropy vs run max, episode length, fps |
| `progress` | target, last checkpoint, `remaining_to_target` |
| `findings` | rules-engine output: `nan_state`, `nonfinite_metric` (critical → auto-abort), `penalty_sign`, `stagnation`, `entropy_collapse`, `episode_len_collapse` |

The LLM receives this + the reward table + allowlist/ranges + intervention
history and must return STRICT JSON: `verdict`, `confidence`, `diagnosis`,
`evidence`, `changes[]`, `notes`. Thinking is disabled for the local model —
its reasoning block otherwise consumed `max_tokens` and the JSON never
arrived. On unparseable output it is re-asked once with a reminder.

## Safety (what the LLM can and cannot do)

- `safety.allow` (fnmatch) is the ONLY set of changeable keys — currently
  reward weights, `agent.max-iterations`, learning rate, entropy coef, num
  envs. Obs terms, robot/actuator config and curricula are NOT allowlisted —
  those are `AGENTS.md` invariants; ideas about them go into `notes`.
- `safety.ranges` bounds each value; `max_changes` caps the count;
  `confidence_min` gates auto-apply. Rejected recs are stored with `errors`.
- Every change is **smoke-tested first** (64 envs, 5 iters, AGENTS.md rule):
  traceback / nan_state / non-finite reward / timeout → `smoke_failed`, the
  live run is untouched.
- Intervention budget per experiment (`max_interventions`), a cooldown of
  `cooldown_iters` between interventions, and `min_iters_before_intervene`
  (never touch a young run).
- Stop = SIGINT to the process tree (rsl_rl saves nothing on KeyboardInterrupt
  → up to `save_interval` iterations can be lost), escalating
  SIGTERM → SIGKILL. Relaunches always pass `--agent.save-interval 50` to
  bound that loss.
- Resume always pins `--agent.load-run <dir> --agent.load-checkpoint
  model_N.pt --agent.resume True` — the default `load_run: .*` regex would
  otherwise accept an older run's checkpoint with the same number.

## Verdict semantics by phase

| phase | keep | adjust | abort |
| - | - | - | - |
| `monitoring` (run alive) | let it run | validate → smoke → stop → relaunch with changes | stop the run |
| `post_run` (run ended) | plan complete → mark `completed`; OR died early → restart same cfg from last ckpt | next step needs changes; if the plan is EXHAUSTED, must include `agent.max-iterations` as the NEW ABSOLUTE total (validated) | do not restart, human review |

## Commands

`PY=../microduck_rl/.venv/bin/python` (mjlab venv —
supplies tensorboard/yaml for the analytics):

```bash
$PY autopilot.py status          # runs, cycle, last rec
$PY autopilot.py suggest --force             # one-shot monitoring analysis
$PY autopilot.py plan --force --recommend-only  # plan an ended run, DO NOT apply
$PY autopilot.py plan --force                # plan + apply (auto mode)
$PY autopilot.py apply REC.json               # validate+smoke+stop+relaunch for a saved rec
$PY autopilot.py launch                       # start train.command
$PY autopilot.py reset [--experiment NAME]    # clear budget/completed marks
$PY autopilot.py run                          # daemon (systemd runs this)
```

Notes:

- `--recommend-only` writes the rec to `logs/recommendations/` and
  marks the run planned without spawning anything; review it, then
  `autopilot apply <rec>` (or `plan --approve`) when you like it.
- Before enabling the daemon on a machine with existing runs, run
  `plan --force --recommend-only` once so the newest run is marked planned —
  otherwise the daemon will plan (and possibly relaunch) it on first start.
- wandb: runs go ONLINE when `~/.netrc` / `WANDB_API_KEY` exists (stats
  dashboard); offline only without credentials. Analysis always reads the
  LOCAL copies: tfevents under `<mjlab_repo>/logs/rsl_rl/` + stdout under
  `logs/train-*.log` — the dashboard is a mirror, not a dependency.

## Files

```
autopilot/
  autopilot.py, autopilot.toml   code + config
  docs/autopilot.md              this reference
  reports/
    session-*.md                 per-daemon-session journal: verdicts,
                                 interventions, smoke results, sleeps, ETA,
                                 hourly heartbeats, session summary
  logs/
    status.json                  last cycle (phase, verdict, sec_per_iter, eta, sleep)
    state.json                   per-experiment: target, budget, interventions, planned marks
    recommendations/*.json       every verdict (proposed/applied/rejected/keep/plan...)
    autopilot.log                daemon log
    train-*.log, smoke-*.log     stdout of spawned training / smoke runs
```

Config: `autopilot.toml` (sections `[paths] [auto] [llm] [monitor] [train]
[safety]`); `[paths].mjlab_repo` points at the supervised training repo (its
tfevents are read from `<mjlab_repo>/logs/rsl_rl/`, spawned training runs
with `<mjlab_repo>` as cwd).
Unit: `~/.config/systemd/user/microduck-rl-autopilot.service`
(`systemctl --user {start,stop,status} microduck-rl-autopilot`).
Linger is enabled, so the unit survives logout.

## Known operational caveats

- A daemon restart mid-intervention leaves `intervening: true` in state if it
  was SIGKILLed during a smoke test; clear it (`state.json` →
  `intervening: false`) or the daemon parks in `phase: intervening`.
- `exp.target_iterations` is set once (first analysis) and only rewritten by a
  plan-extending change; hand-editing `state.json` while a cycle is mid-flight
  can be overwritten by that cycle's `save_state`.
