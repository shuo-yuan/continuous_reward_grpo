# Plan: adaptive reward thresholds for GRPO

Status (2026-10-31): phase 1 is planned and nothing has been launched. All phase-1 code is written and
CPU-tested. To start on a new machine, follow [Setting up a new machine](#setting-up-a-new-machine).

## Goal

**Find an adaptive thresholding schedule for the judge's 0–100 reward that clearly beats the
baselines on validation, and refine it based on experiment results.**

A *thresholding schedule* turns each rollout's judge score into the per-rollout signal that GRPO trains
on, by comparing the score with a threshold. *Adaptive* means the threshold changes during training in
response to the scores seen so far (per problem, across all problems, or per group), instead of being
fixed in advance.

The work has three phases. It is open-ended research, not a single pre-registered test.

1. **Phase 1: baselines and a first idea.** Run the five runs in [Phase 1: the runs](#phase-1-the-runs):
   the three baselines below, and the user's first adaptive idea at two settings.
   - Purpose: a rough sense of where each baseline ends up and how it trains, and a first look at the
     idea. Phase 1 is not expected to settle the question.
   - All five finish before any phase-2 GPU run, because runs 1, 4 and 5 are the baselines for the bar.
   - CPU work for phase 2 can start once run 2 has data: offline replays, new plugin code, tests.
2. **Phase 2: design.** Using the phase-1 results, design an adaptive schedule meant to clearly beat all
   three baselines.
   - The rule in [The first idea](#the-first-idea-a-per-problem-rising-threshold) is a starting point,
     not a commitment. Tune it, change it, or replace it with a better schedule.
   - Base design choices on evidence from the runs, for example: validation curves, training reward per
     problem, pass rates, how thresholds moved, the share of groups with mixed labels.
3. **Phase 3: refine.** Change the schedule, test it, compare, and repeat. Use cheap checks before full
   runs (see [Phases 2 and 3](#phases-2-and-3-designing-and-refining)).
   - Stop when a schedule passes the confirmation in
     [What "clearly beats the baselines" means](#what-clearly-beats-the-baselines-means), or when the
     user says to stop.
   - Then write up what was learned, including what did not work.

### Working with the user

- **Phase 1:** start when the user says go. After that the five runs follow one another in order.
- **Before every later GPU run** (a screening run, a full run or a seed repeat):
  - write the proposal in [Design notes](#design-notes): the exact schedule, its offline-replay
    statistics, and why it should beat the baselines;
  - then ask the user and wait.
- **No fixed compute budget.** The user decides at each check-in. Check in after phase 1, and after
  each finished run in phases 2 and 3, with that run's row in the [Results log](#results-log).
- **Repeated failures:** if a run fails twice at the same point, or its supervisor gives up, stop.
  Do not start the next run and do not work around the failure. Report it and ask.
- **Design questions go to the user.** Examples: changing anything on the fixed list below, tuning a
  baseline, reordering or dropping phase-1 runs, editing verl.

### Baselines

All three run with dynamic sampling and the same settings as every other run.

| baseline | what GRPO trains on | phase-1 run | what it tells us |
|---|---|---|---|
| regular GRPO | the raw score r (0–100) | 1 | the main reference |
| mean-quantized | 1[r > group mean] | 4 | whether a schedule beats splitting each group at its own mean |
| fixed threshold, c = 50 | 1[r ≥ 50 and r > 0], one bar for all problems | 5 | whether adaptivity beats a fixed absolute bar |

- **Baselines are not tuned.** c = 50 is one fixed bar, not the best one.
  - If a schedule's win over run 5 might come only from the extra tuning the schedule got, propose a
    sweep of c to the user.
- **The winner must be adaptive.** If a non-adaptive variant turns out best (for example another
  fixed c), report it as a finding. It then becomes a fourth baseline that an adaptive schedule must
  also beat.
- **The finished runs are not baselines.** The two runs without dynamic sampling
  ([Where we are](#where-we-are)) are references only (see
  [Reading the phase-1 results](#reading-the-phase-1-results)).

### What "clearly beats the baselines" means

**The metric.** Compute it from the validation dumps, not from wandb.
- **Dumps:** `outputs/validation_data_<run>/<step>.jsonl` (gzipped once the next one lands). Each row
  is one sample: the problem id is in `gts`, the 0–100 score in `score`, and there is a `judge_status`.
- **Final window:** validation steps 260, 280 and 300.
- **Per problem:** average `score` over the problem's 15 samples in the final window.
- **Which problems:** 170 of them. `full.parquet` has 172, but verl drops problems 156 and 157
  because their prompts exceed `max_prompt_length` (8192 tokens). So every validation dump has 850 rows
  over the same 170 ids. Pair over those 170.
- **Gap:** the mean over the 170 problems of (schedule − baseline).
- **95% CI:** resample the 170 problems with replacement 10,000 times; take the 2.5 and 97.5
  percentiles of the mean gap.
  - Check: this recipe reproduces the finished runs' −0.59 [−1.59, +0.44] (mean-quantized vs
    regular), using `data/reference/finished_runs_val_per_problem.csv`.

**Selection, then confirmation.**
- **Selection (seed 1):** a candidate qualifies when its CI's lower end is above 0 against each of
  runs 1, 4 and 5.
- **Confirmation (the deciding test):** rerun the candidate and the strongest baseline with a second
  seed. The strongest baseline is whichever of runs 1, 4 and 5 has the highest seed-1 final-window
  Avg@5.
  - It passes if, on the seed-2 pair alone, the CI's lower end is above 0;
  - and if, averaged over the two seeds, the gap against the other two baselines is still positive.
- **Why confirmation is needed:** many variants get compared on the same 170 problems. That makes the
  validation set a development set, where the best of many looks better than it really is. The seed-2
  pair is the only fresh test.
- **Second seed:** see [Reading the phase-1 results](#reading-the-phase-1-results) for what it needs.

**How big a gain is needed.**
- **Paired SE:** a final-window gap has a paired SE of about 0.52 points.
  - This covers only problem and sampling noise, not run-to-run (training seed) variation, which
    phase 1 does not measure. The real uncertainty is larger.
- **Chance of clearing the selection bar:**

  | true gain | against one baseline | against all three at once |
  |---|---|---|
  | 1 point | 48% | 24% |
  | 1.5 points | 82% | 65% |
  | 2 points | 97% | 93% |

  The "all three" column assumes an SE of 0.52 for each gap and a correlation of 0.5 between the
  three gaps.
- **So:** gains under about 1.5 points will rarely pass. Aim for 2 points or more.
- **No extra minimum gap:** passing the selection and the confirmation is what "clearly beats" means.

**Also report:**
- **Learning speed:** Avg@5 at steps 20–100, per training step and per generated rollout.
- **Best@5 over the final window:** for each problem, take the max score over its 5 samples at each
  step; average over the problems and the three steps. Do not use wandb's best@5, which is a bootstrap
  with replacement.
- **The validation failure split** (see [Reading the phase-1 results](#reading-the-phase-1-results)).

### Rules for fair comparisons

- **Keep fixed across all runs:**
  - model: Qwen3.5-9B, starting from the base model;
  - training data: the 200 training problems;
  - validation: 170 problems × 5 samples, 32k tokens, every 20 steps;
  - batch: 16 problems × 16 rollouts per step, 32k-token responses;
  - optimization: lr 1e-6, KL 0.001, temperature 1.0, 300 steps;
  - the advantage: vanilla GRPO on the per-rollout signal (group mean and std,
    `norm_adv_by_std_in_grpo=True`), with verl's loss aggregation as configured;
  - dynamic sampling, with the same settings;
  - the judge, and the machine and software stack.
- **Free to change:** only the mapping from a rollout's score (plus the schedule's state) to its
  per-rollout signal. For example:
  - where the threshold starts;
  - how and when it moves: up, down, how fast, on what trigger;
  - whether it applies per problem, across all problems, or per group;
  - what counts as a pass;
  - how a score's position relative to the threshold becomes the signal (the first idea uses 0/1).
- **Prefer simple schedules:** few settings, no special cases, and state saved with every checkpoint
  so a run resumes exactly.
- **Compute is not equal, even with the same dynamic-sampling settings.** A schedule that leaves fewer
  groups with mixed labels generates more rollouts per step, uses up the training data faster, and
  sends more judge submissions.
  - Record each run's total generated rollouts, wall time and judge-outage rate in the Results log.
  - Also compare at matched generated rollouts: take the schedule's validation step where it has
    generated as many rollouts as the baseline had by step 300.
  - If a schedule wins per step but loses at matched rollouts, report that.
  - Total generated rollouts = `dynamic_sampling/batches_consumed_total` × 256, at the run's last step
    (the metric is cumulative and survives resumes).
  - Do not sum the per-step `dynamic_sampling/rollouts_generated` from wandb: steps that are redone
    after a restart get logged twice.
- **Screening runs** (shorter runs, see [Phases 2 and 3](#phases-2-and-3-designing-and-refining)) are
  exempt from the 300-step rule. They can never pass the bar.
- **Anything else that has to change** is a question for the user. If it is approved, rerun the
  affected baselines with the same change before comparing.
- **Record every run** in the [Results log](#results-log), including screens and dropped variants.

## Where we are

All runs so far (one node, 8× H100 80GB): Qwen3.5-9B, the 200 FrontierSmith synthetic training
problems, 16 problems per step × 16 rollouts, 32k-token responses, lr 1e-6, 300 steps. Validation is
the 170 Frontier-CS problems × 5 samples, every 20 steps.

| run | reward GRPO sees | Avg@5 step 200 | Avg@5 step 300 | window {260, 280, 300} vs regular |
|---|---|---|---|---|
| base model (step 0) | — | 3.49 | — | — |
| regular GRPO (`qwen35_9b_grpo_16x16_32k`) | r (0–100) | 14.15 | 14.02 | — |
| mean-quantized (`qwen35_9b_grpo_quantized_16x16_32k`) | 1[r > group mean] | 12.32 | 13.57 | −0.59, 95% CI [−1.59, +0.43] |

Takeaways:
- **Regular GRPO plateaus.** It stops improving at about step 200, near 14 Avg@5.
- **Mean-quantization is no better.**
  - It trailed regular GRPO at every validation step from 40 on.
  - At steps 60–200 the gap was 0.7–2.1 points (mean 1.6).
  - It then narrowed to about 0.6 points, which is not significant at the end.
- **Splitting each group at its own mean throws away the absolute level.** A group whose scores all sit around 90 still has about half its rollouts labeled failures. A group whose scores all sit around 5 still has about half labeled passes.
- **The first steps barely train.**
  - The base model hits the 32k cap on most answers (86–94% of training rollouts in steps 1–4).
  - So about 70% of rollouts get zero advantage in steps 1–4, and 7–25 rollouts per step set the direction of each update.
  - Truncated rollouts are not masked from the loss.

## The first idea: a per-problem rising threshold

This is the user's first idea, and it runs in phase 1 at x = 0.75 and x = 1.0. Later phases may
change or replace it (see [Goal](#goal)).

Binarize each rollout against a **per-problem threshold that rises as the policy gets better on
that problem**, then run vanilla GRPO on the 0/1 labels. The threshold works as a curriculum:
- **Where the bar sits.** A problem's first group sets the bar at its own mean. After that, the bar is the highest mean of any group that cleared it, meaning at least x of the group's rollouts reached the bar at the time.
- **What a pass means.** In an offline replay on the logged rollouts, about half of rollouts pass after step 25 (48–55% at x = 0.75, 59–68% at x = 1.0).

### The rule (`grpo_adaptive_threshold`, `scripts/adv_adaptive_threshold.py`)

For a group of n = 16 rollouts with scores r_i on problem p:

```
first group of p:   tau_p = mean of that group
r_i reaches t   iff r_i >= t (up to float rounding) and r_i > 0
if at least ceil(x * n) rollouts reach tau_p:
    tau_p = max(tau_p, mean of the group)
b_i = 1 if r_i reaches tau_p (as just updated), else 0
A_i = (b_i - mean(b)) / (std(b) + eps)          # vanilla GRPO on b
```

Notes on the rule:
- **x is the only setting.**
  - At x = 0.75, a group in which at least 12 of its 16 rollouts reach the bar moves the bar up to the group mean, if that mean is higher. At x = 1.0 all 16 must reach it.
  - In the replay, x = 0.75 raises the bar in 10–33% of groups, depending on the phase. x = 1.0 raises it in 3–11%.
- **Raised bars apply at once.** A group that raises the bar is scored against the raised bar. So a group where everyone passes is split at its own mean instead of being thrown away.
- **A group where nobody reaches tau_p** gets b = 0 everywhere, so zero advantage, and tau_p stays where it is.
- **Thresholds are saved and restored.** They go into `adaptive_threshold_state.json` with each checkpoint and are read back on resume.
- **How far it is from mean-quantization.**
  - A problem's first group, and every group that raises the bar, is split at its own mean, the same as mean-quantization.
  - In the replay at x = 0.75, the adaptive labels match mean-quantization's in 90% of groups in steps 1–25, 69% in 26–100, 52% in 101–200 and 45% in 201–300. Only 2–17% of rollouts get a different label.
  - At x = 1.0 the labels match in 37–76% of groups, and 6–18% of rollouts differ.
  - So the difference from mean-quantization grows over training and is larger at x = 1.0. Mean-quantization finished 0.59 points below GRPO, so the adaptive gain has to come from the relabeled minority.
  - That makes mean-quantized with dynamic sampling (run 4) the closest baseline for "does the adaptive bar matter".

## Dynamic sampling in every run

Every run, in every phase, uses DAPO-style dynamic sampling (`scripts/dynamic_sampling.py`) so that early steps
are not decided by a handful of rollouts:
- **Which groups count.** After each generation round, only groups whose advantages are not all zero are kept. "All zero" is judged under each run's own reward: raw scores for GRPO, the 0/1 labels for the others.
- **Topping up.** More batches are drawn from the training data stream until 16 such groups are collected. Each extra round is sized by `DS_BATCHES_PER_ROUND=auto`: enough whole batches for the current keep rate.
- **Cap.** At most 6 generation batches per step. If still short, the batch is filled with groups whose advantages are all zero.
- **Always the same training batch.** Every step trains on exactly 16 groups × 16 rollouts = 256 rollouts.
- **Adaptive thresholds and dropped groups.** For the adaptive arms, every generated group updates the thresholds, kept or dropped, in generation order.
- **Judge outages.**
  - Pulling more batches stops when more than half of a round's rollouts timed out waiting for the judge (600 s) or could not reach it (`judge_status` −1/−2).
  - Compile errors (−3) and missing code (0) are real zeros and do not count.
- **Data order differs across arms.**
  - The finished runs trained on the same 16 problems at every step. Here each run pulls a different number of extra batches from the shared stream, so runs soon train on different problems at the same step.
  - Step-to-step training reward (SD about 5 points) is almost all problem mix. So compare training reward per problem over windows (steps 21–100, 101–200, 201–300), not step by step.
  - Validation is unaffected: same 170 problems for every arm.
- **Which training-reward metric.** `critic/score` now describes the filtered batch. Use `dynamic_sampling/score_mean_round1` for the first batch of each step.

## Phase 1: the runs

Same shape for all five phase-1 runs (and for later runs, see [Rules for fair comparisons](#rules-for-fair-comparisons)):
- 16 problems × 16 rollouts per step, one optimizer update per step (PPO mini-batch 16);
- 32,768-token responses;
- vLLM stops at 32,768 tokens, for training, the extra dynamic-sampling rounds and validation;
- 300 steps;
- validation every 20 steps with 5 samples per problem.

The remaining settings are unchanged from the finished runs (lr 1e-6 constant, no warmup, KL loss
0.001, entropy 0, temperature 1.0, Triton GDN prefill). One exception: the finished regular run
used vLLM's default FlashInfer GDN prefill for steps 1–20 and Triton from step 21. Step-0 validation
is skipped because all runs start from the same base model, which is already evaluated.

| # | arm | run name | time vs a finished run |
|---|---|---|---|
| 1 | regular GRPO | `qwen35_9b_grpo_dynsamp_16x16_32k` | about 1.1–1.2× |
| 2 | adaptive, x = 0.75 | `qwen35_9b_grpo_adaptive_threshold75_dynsamp_16x16_32k` | about 1.1–1.2× |
| 3 | adaptive, x = 1.0 | `qwen35_9b_grpo_adaptive_threshold100_dynsamp_16x16_32k` | about 1.1–1.2× |
| 4 | mean-quantized | `qwen35_9b_grpo_quantized_dynsamp_16x16_32k` | about 1.1–1.2× |
| 5 | fixed threshold, c = 50 | `qwen35_9b_grpo_fixed_threshold50_dynsamp_16x16_32k` | about 1.6–1.8× |

Order (chosen by the user):
- **Run 1 first.** Regular GRPO with dynamic sampling is the main reference.
- **Then the first idea.** Adaptive x = 0.75 (run 2). Together with run 1 it gives a first look at the
  idea after two runs.
- **Then the rest.** Adaptive x = 1.0 (run 3), then mean-quantized (run 4), then the fixed threshold
  (run 5), which is the slowest.
- **The full bar needs run 5.** Runs 1, 4 and 5 are all only available after run 5, so no schedule can
  be called a winner before then.
- **Total:** about 6–6.6× the wall time of one finished run, plus restarts.

Time per arm:
- **Base.** Each finished run took 65.8 h on 8× H100 80GB, validation included. Scale by the new machine's speed.
- **Runs 1–4.** Dynamic sampling needs about 2 generation batches per step in steps 1–25 and about 1.2–1.4 later, giving about 1.1–1.2× a finished run.
- **Fixed threshold.** At c = 50 only about a third of groups have mixed labels in steps 1–25, and about 55% later. So it needs about 4 generation batches per step early and about 2.4 later, giving about 1.6–1.8×.

The batch counts come from replays and simulations on the logged rollouts; they come from other policies, so on-policy rates may differ.

### Commands

Run one at a time, each after the previous run has finished, including its final validation.
Each command starts a supervisor in its own tmux session named after the run. When the training
container dies (stopped or crashed), the supervisor waits until the GPUs are idle and resumes from the
latest checkpoint. Where checkpoints and dumps go is set as described in
[Machine requirements](#machine-requirements).

```bash
cd FrontierSmith   # the repo root
export TRAIN_BATCH_SIZE=16 ROLLOUT_N=16 PPO_MINI_BATCH_SIZE=16 MAX_RESPONSE_LENGTH=32768 \
       USE_FUSED_KERNELS=True ROLLOUT_TP=1 TOTAL_STEPS=300 VAL_FREQ=20 VAL_N=5 \
       VAL_BEFORE_TRAIN=False CKPT_POLICY=full SAVE_FREQ=5 JUDGE_MAX_WAIT=600
# dynamic-sampling defaults: DS_MAX_GEN_BATCHES=6 DS_BATCHES_PER_ROUND=auto DS_OUTAGE_FRAC=0.5

# 1. regular GRPO
ESTIMATOR=grpo RUN_NAME=qwen35_9b_grpo_dynsamp_16x16_32k bash scripts/launch_dynamic_sampling_grpo.sh
# 2. adaptive, x = 0.75
ESTIMATOR=grpo_adaptive_threshold ADAPTIVE_FRAC=0.75 \
  RUN_NAME=qwen35_9b_grpo_adaptive_threshold75_dynsamp_16x16_32k bash scripts/launch_dynamic_sampling_grpo.sh
# 3. adaptive, x = 1.0
ESTIMATOR=grpo_adaptive_threshold ADAPTIVE_FRAC=1.0 \
  RUN_NAME=qwen35_9b_grpo_adaptive_threshold100_dynsamp_16x16_32k bash scripts/launch_dynamic_sampling_grpo.sh
# 4. mean-quantized
ESTIMATOR=grpo_quantized RUN_NAME=qwen35_9b_grpo_quantized_dynsamp_16x16_32k bash scripts/launch_dynamic_sampling_grpo.sh
# 5. fixed threshold, c = 50
ESTIMATOR=grpo_fixed_threshold FIXED_THRESHOLD=50 \
  RUN_NAME=qwen35_9b_grpo_fixed_threshold50_dynsamp_16x16_32k bash scripts/launch_dynamic_sampling_grpo.sh
```

The launcher ignores `RUN_TAG`, so the batch and length settings must be exported as above.

### Pre-flight before each launch

- **Settings:** the same command with `PRINT_ONLY=1` prints the settings without starting anything.
- **Judge:** `curl -s http://localhost:8082/health` returns `{"ok":true}`, and a test submission comes
  back judged:
  ```bash
  .venv/bin/python -c "
  import importlib.util
  s = importlib.util.spec_from_file_location('fcs', 'verl/verl/utils/reward_score/frontiercs.py')
  m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
  f = chr(96) * 3
  print(m.compute_score('frontiercs', f + 'cpp\nint main(){return 0;}\n' + f, 'frontiersmith_0'))"
  ```
  The expected output is `{'score': 0.0, 'acc': 0.0, 'judge_status': 1.0}`: judged, as a wrong
  answer. A `judge_status` of −1 or −2 means the judge timed out or could not be reached.
- **Data:**
  - `data/frontiercs/full.parquet` exists and has 172 rows;
  - `data/alebench/val.parquet` does not exist;
  - after the first validation (step 20), `outputs/validation_data_<run>/20.jsonl` has 850 rows.
- **Checkpoint storage:** at least 212 GB free where `CKPT_DIR` points. Check this by hand before a
  fresh launch: the supervisor's own check (`SHM_MIN_FREE_GB`, default 120) only covers resumes, when
  one checkpoint already exists.
- **Disk:** room for `outputs/` and the judge's log, about 1–1.5 GB per run.
- **GPUs:** all 8 are idle, and no `frontiersmith-train` container is running.

## Setting up a new machine

In order:
1. **Clone the repo under `$HOME`.**
   - The training container mounts only `$HOME` (and `/dev/shm`, through `--ipc host`). So the repo,
     the venv, `CKPT_DIR`, and any target of the rollout-dump symlink must all be under `$HOME` or
     `/dev/shm`.
   - Anything the container writes elsewhere is lost when it exits. A checkpoint written there would
     be gone, so every resume would silently start again from step 0.
2. **Build the Python environment.** Install [uv](https://github.com/astral-sh/uv), then run
   `source setup-env.sh`.
   - It builds `.venv` with a uv-managed Python 3.11 under `~/.local/share/uv`. The container image has
     no Python of its own. Without uv, `setup-env.sh` falls back to the host's `python3`, which the
     container cannot use.
   - Then run `pip install flash-linear-attention==0.5.2`, because `setup-env.sh` does not pin it.
   - Check the versions listed under [Machine requirements](#machine-requirements).
3. **Secrets:** put `WANDB_API_KEY=...` in `.env` at the repo root (it is git-ignored). Run
   `hf auth login` if the model download asks for it.
4. **Validation data:** copy the Frontier-CS problems and build `full.parquet` (see
   [Machine requirements](#machine-requirements), "Validation problems"). Check that it has 172 rows.
5. **Judge:**
   - `docker build -t frontiercs-judge Frontier-CS/algorithmic`, once (`run_judge.sh` does not build the
     image);
   - then `bash Frontier-CS/algorithmic/run_judge.sh`;
   - consider adding log rotation to its `docker run` line
     (`--log-opt max-size=500m --log-opt max-file=2`);
   - run the judge test in [Pre-flight](#pre-flight-before-each-launch).
6. **Storage:** choose `CKPT_DIR` (about 210 GB free at peak) and where the rollout dumps go. Both must
   be under `$HOME` or `/dev/shm`.
7. **Training image:** `scripts/run_in_container.sh` builds it automatically on first use, from
   `docker/Dockerfile.train`.
8. **Dry check:** run the run-1 command with `PRINT_ONLY=1`, then the full pre-flight.

**Code status.**
- All phase-1 code is written, including the fixed-threshold estimator for run 5
  (`scripts/adv_fixed_threshold.py`, `grpo_fixed_threshold`).
  - The fixed threshold computes b = 1 if r ≥ c (and r > 0), then vanilla GRPO on b.
  - "Reaches c" uses the adaptive estimator's own test and the same GRPO code. So run 5 differs from
    the adaptive runs only in where the bar comes from.
  - Its threshold is saved with every checkpoint (`fixed_threshold_state.json`). A resume with a
    missing file or a different threshold is refused.
- On the original machine, the code passed unit suites and an end-to-end suite that runs verl's `fit()`
  on CPU, including killed-and-resumed runs that match uninterrupted ones exactly. Those test suites
  are not in this repo.

## Machine requirements

What the scripts assume. Defaults can be changed with the environment variables named.

- **GPUs:** one node with 8 GPUs (`launch_common.sh` checks that each mini-batch splits over 8; the run
  script's `NGPU` defaults to 8). The memory settings were tuned on 80 GB H100s: one sequence per GPU per
  micro-batch, no parameter offload, vLLM `gpu_memory_utilization` 0.6, `ROLLOUT_TP=1`, fused log-prob
  kernels for the 32k responses. GPUs with less memory need retuning.
- **Software:**
  - docker with GPU access, and tmux;
  - training runs in the `frontiersmith-train` image, built from `docker/Dockerfile.train` on first use
    by `scripts/run_in_container.sh`. It mounts only `$HOME` (and `/dev/shm`) and reuses the project's
    `.venv`, so the venv has to be rebuilt on the new machine (see [Setting up a new machine](#setting-up-a-new-machine));
  - the finished runs used Python 3.11, vllm 0.20.0+cu129, torch 2.11.0+cu129, transformers 5.7.0,
    flash-linear-attention 0.5.2 (not pinned by `setup-env.sh`), ray 2.55.1 and the vendored verl.
    Keep these for the runs to be comparable with the finished runs;
  - the Frontier-CS judge in its own container on port 8082: build the image from
    `Frontier-CS/algorithmic`, then start it with `Frontier-CS/algorithmic/run_judge.sh`;
  - `WANDB_API_KEY` in the git-ignored `.env`.
- **Data:** `data/frontiercs/train_synthetic_200.parquet` (training, in the repo) and
  `data/frontiercs/full.parquet` (validation, not in the repo). Without `full.parquet` the run script
  quietly validates on the training problems. If `data/alebench/val.parquet` exists it is added to
  validation; the finished runs had none. The model `Qwen/Qwen3.5-9B` is downloaded to
  `.cache/huggingface` on first use.
- **Validation problems:** the Frontier-CS algorithmic problems are not redistributed (about 3.6 GB).
  Copy them from the [official repo](https://github.com/FrontierCS/Frontier-CS) into
  `Frontier-CS/algorithmic/problems/<numeric id>/`; the judge needs them too. Then build the validation
  file with
  `python scripts/prepare_frontiercs_parquet.py --full-for-both --ids data/sample_lists/frontiercs_val_172.json`.
  On 2026-10-30 this rebuilt a `full.parquet` identical to the one the finished runs used. It has
  172 rows; verl drops problems 156 and 157 (prompts over 8192 tokens), leaving the 170 used everywhere.
- **Checkpoint storage**, `CKPT_DIR` (default `/dev/shm/frontiersmith_ckpt/<run>`, which is RAM). It must be
  under `$HOME` or `/dev/shm`, the only places the training container can write to the host:
  - a full checkpoint (weights + Adam) is about 103 GB. A save briefly holds two, so a run needs about
    210 GB free at its peak;
  - a finished run keeps about 36 GB once its final optimizer state is deleted (about 68 GB,
    `$CKPT_DIR/global_step_300/actor/optim_*`);
  - the supervisor starts an attempt only while `/dev/shm` has `SHM_MIN_FREE_GB` free (default 120). It
    checks `/dev/shm` even when `CKPT_DIR` points elsewhere; set `SHM_MIN_FREE_GB=0` then.
- **Rollout dumps**, about 18 MB per step (about 5.5 GB per run): unless `outputs/rollout_data_<run>`
  already exists, the launcher links it to `/dev/shm/frontiersmith_rollouts/<run>`. To put them
  elsewhere, create that symlink yourself first (or a real directory, with `ALLOW_ROOT_ROLLOUT_DIR=1`);
  the target must be under `$HOME` or `/dev/shm`.
- **Under `outputs/`:** validation dumps (about 0.25 GB per run once gzipped) and the plugins'
  per-step JSON logs (about 30 MB per run).
- **Judge log:** the judge container's docker log grows about 2 MB per training step: about 0.75 GB per
  run for runs 1–4 and 1.3 GB for the fixed-threshold run, unless docker caps its logs.

## Reading the phase-1 results

Phase 1 gives a rough picture, so there is no single deciding test.

**Comparisons.**
- **Against the baselines:** compare each run with each of runs 1, 4 and 5, using the recipe in
  [What "clearly beats the baselines" means](#what-clearly-beats-the-baselines-means). Run 1 is the
  main reference.
- **Ties:** a CI that includes 0 is *inconclusive*, not "no difference": the CI is about ±1 point wide.
  Read smaller differences from the mechanism signals below instead.
- **Against the finished runs (no dynamic sampling):** both finished runs' per-problem validation
  scores are in `data/reference/finished_runs_val_per_problem.csv`. It covers every validation step,
  with each problem's mean and max score and its judge-failure counts, so paired comparisons work on
  any machine.
  - These comparisons cross machines and software, so they are context only.
  - Run 4 against the finished mean-quantized run is the cleanest check of what dynamic sampling does.
  - Run 1 against the finished regular run also differs in the GDN kernel during steps 1–20. That
    changes the weights for the rest of training, so their validation comparison mixes dynamic
    sampling, the kernel and the machine. "Compare from step 21" applies only to per-step training
    metrics.
  - If run 1 is clearly below the finished regular run (final-window paired CI upper end below 0; the
    finished run's window mean is 13.85), stop and ask the user whether to keep dynamic sampling.

**Second seed.** All phase-1 runs share one seed setup.
- **Data order:** pass `data.seed=<n>` as an extra argument to the launcher, which forwards extra
  arguments to Hydra. With `data.seed` unset, every run shuffles the data the same way.
- **vLLM sampling seed:** this needs a code change, so ask the user first (it edits verl).
  - `vllm_async_server.py` seeds each replica with `replica_rank + rollout.seed`, but `RolloutConfig`
    has no `seed` field.
  - Add the field, and use an offset larger than the number of replicas (for example 1000), so the
    replicas' seeds do not overlap with those of the first seed.
- **Not reproducible anyway:** generation is asynchronous, so even same-seed reruns differ.

**Also look at:**
- **Best@5:** as defined in the Goal.
- **Speed:** Avg@5 at steps 20–100, both per step and per generated rollout. For the cumulative count
  use `dynamic_sampling/batches_consumed_total` × 256.
- **Validation failure split:** from `judge_status` in the dumps.
  - 1: judged (a score of 0 is a wrong answer);
  - 0: no code block;
  - −3: judge error, such as a compile failure (a real 0);
  - −1: timed out waiting for the judge;
  - −2: could not reach the judge.
  - Report −1 and −2 separately: they are infrastructure, not the policy. The finished runs had 5–16
    such rows out of 850 per validation.
  - Truncation at 32k is not in the dump. Count it by re-tokenizing `output`, or add it to the dump.
- **Training side:**
  - informative groups per step;
  - dynamic-sampling rounds, cap hits and outage stops;
  - training reward per problem over step windows: use `dynamic_sampling/score_mean_round1` or the
    per-step dumps, not `critic/score`, which only covers the kept groups;
  - gradient norm, response length and truncation in steps 1–20: use
    `dynamic_sampling/response_length_mean_all` and `dynamic_sampling/clip_ratio_all`, which cover
    every generated rollout. `response_length/*` covers only the kept groups.
- **Thresholds:** how tau moves (`adaptive_threshold/*`: median, share raised, pass rate), compared
  with the offline replay.

**What the outcomes suggest for phase 2:**
- **Adaptive clears the bar against all three baselines:** promising. Refine it, then run the
  confirmation. Beating run 1 alone is not enough.
- **Adaptive is clearly worse than run 1** (CI upper end below 0): rethink the rule. Run 4 shows
  whether binarizing itself is the cost.
- **Adaptive vs run 4 is inconclusive:** there is no evidence yet that carrying a bar per problem helps
  over splitting at the group mean. Look at the groups where their labels differ.
- **Adaptive beats run 5 but not run 1:** adaptivity may help against a fixed bar but not against raw
  scores. Try how fast the bar rises, where it starts, or whether it should ever come down.
- **Run 5 matches or beats adaptive:** the adaptive part may not be what matters.
- **x = 0.75 vs 1.0:** only 6–18% of their labels differ, so validation will probably not separate
  them. Compare the mechanism signals instead: tau trajectories, pass rates, the share of mixed
  groups, and training reward per problem.

## Phases 2 and 3: designing and refining

**Cheap checks before full runs.**
- **Offline replay.** Each run writes `outputs/dynamic_sampling_<run>/<step>.json`. It holds every
  group the run generated, kept or dropped, with its problem id and all 16 scores and response lengths.
  - Any schedule that depends only on scores can be replayed on these logs. Replay shows the labels it
    would give: pass rates, how often and how far thresholds move, and the share of groups with mixed
    labels (which sets the dynamic-sampling cost).
  - Replay cannot show the training effect, because the logged rollouts come from other policies.
  - Adaptive runs also write `outputs/adaptive_threshold_<run>/<step>.json`: for each group, the
    threshold it found and the one it was scored against, plus every problem's threshold after each
    step.
- **Screening runs**, for example `TOTAL_STEPS=100`.
  - The learning rate is constant, so a 100-step run trains like the first 100 steps of a 300-step run.
  - Score a screen on the window {60, 80, 100} against the baselines' same steps, with the same paired
    recipe, and log it as phase "screen".
  - A screen can never pass the bar. Propose to the user which screens to promote to full runs, with
    their results.
  - Early gaps may not last. Mean-quantized trailed regular GRPO by 0.7–2.1 points over steps 60–200,
    but by only 0.59 over the final window (0.45 at step 300).
- **Cost:** a full run takes about 1.1–1.8× a finished run (65.8 h on 8× H100 80GB). Most of that
  spread comes from how many groups the schedule leaves with mixed labels.

**Adding a schedule (code).** A new schedule is a new estimator plugin.
1. **Copy and rename.** Copy `scripts/adv_fixed_threshold.py` (no state) or
   `scripts/adv_adaptive_threshold.py` (state per problem).
   - Rename `NAME`, the setting keys, `STATE_FILE`, `METRIC_PREFIX`, `_RUNTIME_ATTR` and the
     `outputs/<prefix>_<run>` dump directory.
   - If `_RUNTIME_ATTR` keeps the original name, `_install()` finds the original plugin's runtime and
     returns it, so the new estimator never gets registered.
   - Keep `_EVALUATORS_ATTR = "_frontiersmith_group_evaluators"`: that registry is shared.
2. **Register it** with verl's `core_algos.register_adv_est`.
   - The problem id is `reward_model["ground_truth"]` (for example `frontiersmith_37`), not `uid`. verl
     draws a new uid for every prompt of every batch.
   - Registered estimators only receive uid. `adv_adaptive_threshold.py` gets the problem id by wrapping
     `ray_trainer.compute_advantage`.
3. **Register a group evaluator for dynamic sampling** in `ray_trainer._frontiersmith_group_evaluators`.
   - Signature: `evaluate(gen_in, scores, step, algorithm_config) -> {uid: informative}`.
   - It is called once per generated group, in generation order, including groups that are then
     dropped.
   - Cache what it decided for each uid: n, number of passes, and any threshold used.
4. **In `compute_advantage`, label each kept group from that cache.**
   - balance_batch may have reordered the rows, so look groups up by uid.
   - Check that each group's pass count is unchanged. Never update the state a second time. Then clear
     the cache.
   - See `binarize_sampled` in `adv_adaptive_threshold.py`, and the matching check in
     `adv_fixed_threshold.py`.
5. **Decide which groups update the state.** The first idea updates on every generated group, kept or
   dropped. A new schedule may update only on kept groups instead. Either way, the evaluator and
   `compute_advantage` must agree, and the choice goes in the Design notes.
6. **State and startup.**
   - Save the state with each checkpoint by wrapping `RayPPOTrainer._save_checkpoint`.
   - Restore it in a wrapper of `RayPPOTrainer._load_checkpoint`. That method runs on every start,
     fresh starts included, and the existing plugins register the trainer and reset their state there.
   - Refuse to resume when the state file is missing or was written with other settings.
7. **Metrics:** wrap `ray_trainer.compute_data_metrics` and merge in a pending
   `{"<prefix>/...": value}` dict, as both plugins do. Without this nothing reaches wandb.
8. **Loading:** `scripts/main_ppo_dynamic.py` loads only `scripts/dynamic_sampling.py`. So in
   `dynamic_sampling.py`:
   - add a `_load_*` helper with its own `sys.modules` name, and call it at module top level;
   - add the estimator to `ESTIMATORS`;
   - dispatch it in `classify()`.
9. **Settings:** in `scripts/launch_dynamic_sampling_grpo.sh`, add the estimator to the `ESTIMATOR`
   case. For each setting, validate an environment variable and append `+algorithm.<key>=<value>` to
   `ALGO_OVERRIDES`, as `ADAPTIVE_FRAC` and `FIXED_THRESHOLD` do.
   - The plugin reads the setting from `config.algorithm`.
   - Environment variables do not reach the training container: `launch_common.sh` forwards only a
     fixed list.
10. **Test on CPU before any GPU run.** The phase-1 test suites are not in this repo, so write
    equivalent tests:
    - unit tests of the labels and the state;
    - a short run through verl's `fit()` on CPU that is killed and resumed, and must then match the
      uninterrupted run.

**Background:** `analysis/unbiased_continuous_advantage_memo.md` is an earlier analysis of advantage
estimators for the continuous reward: the bias of GRPO's per-group std, and the length bias of the
loss aggregation. It used an older setup (8 × 8 rollouts, 16k tokens).

## Monitoring

- **Watch:** the run's log, its supervisor and its tmux session (there is no queue for these runs),
  the judge, and vLLM hangs (GPUs idle in the middle of generation).
- **Finished or gave up:** the supervisor writes to `outputs/logs/<run>_supervisor.log`.
  - Done: it logs `training finished (exit 0)`. Then check that
    `outputs/validation_data_<run>/300.jsonl` has 850 rows.
  - Gave up: it logs `giving up after 3 consecutive errors`. Then stop and ask (see
    [Working with the user](#working-with-the-user)).
- **After any restart:** check that the judge is still up and grading.
- **Judge outages during training:**
  - If a step logs `dynamic_sampling/outage_stop` = 1 or a high `dynamic_sampling/judge_outage_frac_all`,
    those rollouts were trained on as zeros. For adaptive schedules they also updated the thresholds.
  - Stop and ask the user whether to roll back to the checkpoint before the outage.
- **Validation:**
  - After each validation, count the dump's rows with `judge_status` −1 or −2 (usually 5–16 of 850).
  - If there are far more, tell the user at once: that step's checkpoint is deleted at the next save, 5
    steps later.
  - Gzip each dump once the next one lands.
- **Dynamic-sampling health:** rounds per step, cap hits, outage stops, the share of informative
  groups, and `judge_outage_frac_all`.
- **Thresholds:** under dynamic sampling, dropped groups also update tau, so tau may climb faster than
  the replay suggests. Watch `adaptive_threshold/tau_median` and `frac_revisits_raised`.
- **Fixed threshold:** watch `fixed_threshold/pass_rate` and `frac_groups_mixed` (over every generated
  group). If few groups are mixed, dynamic sampling hits its 6-batch cap and steps slow down.
- **After each run:** delete its final optimizer state (about 68 GB,
  `$CKPT_DIR/global_step_300/actor/optim_*`) if checkpoint storage is short.
  - Nothing in this plan needs a run's checkpoint after its final validation.
  - `/dev/shm` is cleared on reboot, so copy the final weights elsewhere if you want to keep them.

## Results log

One row per run, in every phase, including screens and seed repeats. Each gap is against that baseline
run, with its 95% CI, on the final window (screens: the {60, 80, 100} window).

| run name | phase | schedule and settings | Avg@5 (window) | vs run 1 | vs run 4 | vs run 5 | rollouts generated | wall time | notes (restarts, judge outages) |
|---|---|---|---|---|---|---|---|---|---|
| (none yet) | | | | | | | | | |

## Design notes

Before each GPU run in phases 2 and 3, add an entry here:
- the date;
- the exact schedule and its settings;
- its offline-replay statistics;
- why it should beat the baselines;
- what result would change the plan.

Also note variants that were dropped without a run, and why.

(none yet)

## Files

- `scripts/adv_adaptive_threshold.py`: the adaptive-threshold estimator and its saved state.
- `scripts/dynamic_sampling.py`, `scripts/main_ppo_dynamic.py`: the dynamic-sampling plugin and its entry point.
- `scripts/adv_fixed_threshold.py`: the fixed-threshold estimator (run 5).
- `scripts/launch_dynamic_sampling_grpo.sh`: the launcher for every run.
- `verl/verl/experimental/agent_loop/single_turn_agent_loop.py`: vLLM `max_tokens` = response length (the 32k stop).
- `analysis/`: earlier notes and simulations on continuous-reward advantages (background).
- `data/reference/finished_runs_val_per_problem.csv`: per-problem validation scores of the two finished runs
  (every validation step), for paired comparisons on any machine.
