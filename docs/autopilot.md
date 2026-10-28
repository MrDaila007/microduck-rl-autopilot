# Training autopilot — wake/sleep loop around a local rl run

`autopilot.py` supervises a mjlab/rsl_rl training run in another local repo
(`[paths].mjlab_repo` in `autopilot.toml`, default `../microduck_rl`): it
watches that repo's tfevents, precomputes the analytics, asks the local LLM
(Ollama, `qwen3.5:35b-a3b-q4_K_M`) only for judgment + plan, and (auto mode)
applies allowlisted overrides by itself — smoke test → stop → resume from the
latest checkpoint. The log math (trends, penalty audit, reward mass, rules,
recommendation checks) lives in `analysis.py` and is re-exported from
`autopilot`, so the dashboard and the tests keep the same names. Runs as the
systemd user unit `microduck-rl-autopilot`.

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
`evidence`, `changes[]`, `notes`. On a live run the model is asked only when
the rules engine already raised a finding or the plateau detector fired;
a quiet run records `keep` without a model call (`status: rules_keep`,
`llm_s: 0`). `post_run` always asks. Thinking is disabled for the local model —
its reasoning block otherwise consumed `max_tokens` and the JSON never
arrived. On unparseable output it is re-asked once with a reminder. Session
reports accumulate seconds spent inside `llm_analyze`.

## Safety (what the LLM can and cannot do)

- `safety.allow` (fnmatch) is the ONLY set of changeable keys — currently
  reward weights, `agent.max-iterations`, learning rate, entropy coef, num
  envs. Obs terms, robot/actuator config and curricula are NOT allowlisted —
  those are `AGENTS.md` invariants; ideas about them go into `notes`.
- `safety.ranges` bounds each value; `max_changes` caps the count;
  `confidence_min` gates auto-apply. Rejected recs are stored with `errors`.
  `agent.max-iterations` may not go below the experiment's recorded
  `target_iterations` — shortening a plan is a hand edit.
- Every change is **smoke-tested first** (64 envs, 5 iters, AGENTS.md rule):
  traceback / nan_state / non-finite reward / timeout → `smoke_failed`, the
  live run is untouched. The pre-change point (`mean_reward`, episode length,
  `fell_over`, entropy) is stored on the intervention. After `cooldown_iters`,
  if the following window has both more falls and a lower reward, the rules
  engine reverts the previous allowlisted values once (same smoke → resume,
  one budget slot). A second revert is not attempted: the experiment is
  marked `needs_human` and further auto-applies are suppressed.
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
- The launch command used for that resume is recovered, in order, from the
  experiment's `last_argv`, from `wandb-metadata.json` whose `output.log`
  names the run directory, then from the scenario catalog keyed by the
  experiment folder name. A dead run the daemon never saw can still be
  relaunched.

## Verdict semantics by phase

| phase | keep | adjust | abort |
| - | - | - | - |
| `monitoring` (run alive) | let it run | validate → smoke → stop → relaunch with changes | stop the run |
| `post_run` (run ended) | plan complete → mark `completed`; OR died early → restart same cfg from last ckpt | next step needs changes; if the plan is EXHAUSTED, must include `agent.max-iterations` as the NEW ABSOLUTE total (validated) | do not restart, human review |

## Commands

There is a wrapper on PATH (`~/.local/bin/autopilot`) so you can call
`autopilot <cmd>` from anywhere; bash aliases (`ap`, `aps`, `apl`, `apup`,
`apdown`, `apst`, `aplog`, `aplaunch`) live in `~/.bash_aliases`. Under the
hood it runs the mjlab venv python (supplies tensorboard/yaml for the
analytics), i.e. `PY=../microduck_rl/.venv/bin/python`:

```bash
autopilot status               # runs, queue, cycle, last rec
autopilot scenarios            # list the scenario catalog + queue state
autopilot suggest --force      # one-shot monitoring analysis
autopilot plan --force --recommend-only  # plan an ended run, DO NOT apply
autopilot plan --force         # plan + apply (auto mode)
autopilot apply REC.json       # validate+smoke+stop+relaunch for a saved rec
autopilot launch [SCENARIO]    # start a named scenario (or default/train.command)
autopilot reset [--experiment NAME] [--queue]  # clear budget/completed marks (+ queue)
autopilot run                  # daemon (systemd runs this)
autopilot dashboard            # local UI, http://127.0.0.1:8765
```

## Dashboard

`autopilot dashboard` opens the page from `[dashboard]` in `autopilot.toml`
(`http://127.0.0.1:8765`). The daemon serves the
same page while it runs; if the port is already taken the command only prints
the URL.

The page reads the files the cycle already writes (status, state,
recommendations, tfevents) and shows phase, ETA, reward / falls / entropy,
the scenario queue and the last verdicts. Buttons do not call smoke or
`stop` on the HTTP thread. They enqueue one action; the daemon drains it at
the start of the next wake (the sleep is interrupted within a second). A
second click while a smoke test or another action is in progress gets HTTP
409.

Manual hyperparameter edits and `autopilot apply` are explicit human applies:
they still go through `validate_rec` (allowlist, ranges, `max_changes`, no
shorter `agent.max-iterations`) and the 64-env / 5-iter smoke test. The live
run is stopped only after smoke passes, even when `auto.enabled` is false.
That path does not spend the unattended intervention budget.

Settings write a whitelist of keys back into `autopilot.toml` (auto flags,
monitor cadence, queue, default scenario, existing safety ranges, LLM model).
Comments and `[train.scenarios.*]` are left as they are. The daemon reloads
the file when its mtime changes. `dashboard.host` is loopback or one private
address already on this machine. `0.0.0.0` is
rejected, so the page is not published on every interface.

## Training scenarios (the whole policy family)

`autopilot scenarios` lists every training scenario the autopilot can run.
Each maps a friendly name to a task id in the mjlab registry (`uv run
list-envs`): locomotion (`velocity`, `velstand` + `-rough`), posture/episodic
(`standup`, `sitstand`), tricks (`groundpick`, `ballkick`, `roulade`), the
roller-skate family (`rollers`, `swizzle`, `roller-crouch`, `roller-slope`,
`roller-standup`, `spin`), and every `-Backlash-` A/B twin. 33 scenarios by
default, all verified against the live registry.

Each scenario supports:

- `task` (required) — the registered task id.
- `args` — overrides `train.scenario_args` (default `--env.scene.num-envs 4096`).
- `max_iterations` — pins the plan length (adds `--agent.max-iterations`).
- `env` — extra environment variables for that scenario only.
- `warm_start_from` — a run dir: loads its LATEST checkpoint with
  `MICRODUCK_WARM_START=1` (weights/normalizer/optimizer kept, step-based
  curricula reset — the AGENTS.md warm-start ≠ resume rule).

Launch one by hand with `autopilot launch velstand`, or chain a training
program with the queue:

```toml
[train]
default_scenario = "velocity"
queue = ["velocity", "velstand", "standup", "sitstand"]  # walked when idle
```

With `auto.start_if_idle = true`, the daemon walks `queue` one scenario at a
time: it launches the first not-yet-done scenario, supervises it through the
normal monitoring/post-run cycle, and when the plan completes it marks the
scenario done and moves to the next. An `abort` verdict parks that scenario
(so it is NOT auto-relaunched); `autopilot reset --queue` clears the
progress. The queue state lives in `state.json` under `queue`
(`completed` / `current` / `aborted`) and is shown in `status`.

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
  autopilot.py, analysis.py, autopilot.toml   supervisor, log analysis, config
  dashboard.py, dashboard/       local web UI (stdlib HTTP + one static page)
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

- A smoke test sets `intervening: true` with `intervening_since`. The flag
  is cleared when the smoke returns, including on an exception, and any
  recommendation left at `smoking` is closed. If the daemon is killed
  mid-smoke, the next cycle keeps the flag until `smoke_timeout_s` and then
  clears it, marking those recommendations `smoke_failed`. Smoking files older
  than that timeout are closed on the next cycle even when the flag is already
  down. A fresh flag still parks the cycle so a second smoke cannot start.
- `exp.target_iterations` is set once (first analysis) and only rewritten by
  a plan-extending change that does not shorten it; hand-editing `state.json`
  while a cycle is mid-flight can be overwritten by that cycle's `save_state`.
