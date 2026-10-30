# Plan: an adaptive reward threshold for GRPO

Status (2026-10-30): planned, nothing launched. All the code is written and CPU-tested. What a
machine needs to run it is under [Machine requirements](#machine-requirements).

## Goal

Design a clean, adaptive thresholding schedule for the judge's 0–100 reward that makes GRPO
training do better than plain GRPO on held-out validation (Frontier-CS, Avg@5).

"Clean" means:
- **One knob.** The only setting is the trigger fraction x.
- **Per problem, and it only goes up.** The threshold never comes back down.
- **Zero never passes.** A rollout that scored 0 is never labeled a pass.
- **No warmup, no special cases, no flags.**
- **Resumable.** The thresholds are saved with every checkpoint and restored on resume.

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

## The idea

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
  - That makes mean-quantized with dynamic sampling (arm 4) the closest control for "does the adaptive bar matter".

### Comparison arms

| arm | reward GRPO sees | what it tests |
|---|---|---|
| regular GRPO | r | **The baseline the goal is measured against.** |
| mean-quantized | 1[r > group mean] | Does carrying a per-problem bar add anything over splitting at the group mean? |
| fixed threshold, c = 50 | 1[r ≥ 50], one bar for all problems | Is adaptivity needed, or is any absolute bar enough? |

## Dynamic sampling in every run

All five runs use DAPO-style dynamic sampling (`scripts/dynamic_sampling.py`) so that early steps
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
  - The finished runs trained on the same 16 problems at every step. Here each arm pulls a different number of extra batches from the shared stream, so the arms soon train on different problems at the same step.
  - Step-to-step training reward (SD about 5 points) is almost all problem mix. So compare training reward per problem over windows (steps 21–100, 101–200, 201–300), not step by step.
  - Validation is unaffected: same 170 problems for every arm.
- **Which training-reward metric.** `critic/score` now describes the filtered batch. Use `dynamic_sampling/score_mean_round1` for the first batch of each step.

## The runs

Same shape for all five:
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

Order:
- **The baseline comes first.** Regular GRPO with dynamic sampling (run 1) is the bar every other arm is measured against.
- **Then the goal's test.** Adaptive x = 0.75 (run 2); with run 1 it gives the primary comparison after two runs.
- **Then the other arms.** Adaptive x = 1.0 (run 3), then mean-quantized (run 4), the closest control for the adaptive part.
- **Fixed threshold goes last.** It is the slowest run.
- **Total:** about 6–6.6× the wall time of one finished run, plus restarts.

Time per arm:
- **Base.** Each finished run took 65.8 h on 8× H100 80GB, validation included. Scale by the new machine's speed.
- **Arms 1–4.** Dynamic sampling needs about 2 generation batches per step in steps 1–25 and about 1.2–1.4 later, giving about 1.1–1.2× a finished run.
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
- **Judge:** the container (`Competitive-Programming`) is up and grades a test submission.
- **Checkpoint storage:** at least 212 GB free where `CKPT_DIR` points.
- **Disk:** room for `outputs/` and the judge's log, about 1–1.5 GB per run.
- **GPUs:** all 8 idle, and no `frontiersmith-train` container running.

## Before the first launch

1. **The fixed-threshold arm (run 5): written (2026-10-29).**
   - `scripts/adv_fixed_threshold.py` registers `grpo_fixed_threshold`: b = 1 if r ≥ c (and r > 0), then vanilla GRPO on b. It requires `+algorithm.fixed_threshold=<c>`.
     - "Reaches c" is the adaptive estimator's own test (same float tolerance, zero never passes), and b goes through the same GRPO code. So arm 5 differs from the adaptive arms only in where the bar comes from.
     - The threshold is saved with every checkpoint (`fixed_threshold_state.json`). A resume with a missing file or a different threshold is refused, so a run never mixes thresholds. The threshold is also checked before any generation, with dynamic sampling on or off.
     - Logged as `fixed_threshold/*` (pass rate, groups by outcome; with dynamic sampling over every generated group).
   - `scripts/dynamic_sampling.py` accepts the estimator, loads it, refuses a missing or bad threshold before generating anything, and counts a group as useful when its b is mixed.
   - `scripts/launch_dynamic_sampling_grpo.sh` takes `ESTIMATOR=grpo_fixed_threshold` and `FIXED_THRESHOLD` (default 50; a plain number in [0, 100]).
   - CPU tests (2026-10-29), all passing on the current code:
     - unit suites t1, t2, t5, t6 and t7 (t7 is the fixed-threshold suite, 239 tests);
     - end-to-end suite, 65/65 scenarios on the real verl `fit()` on CPU: the fixed arm with and without dynamic sampling, kill/resume runs that match the uninterrupted run exactly, refused resumes (other threshold, missing state file), and the startup refusal of a missing threshold.

## Machine requirements

What the scripts assume. Defaults can be changed with the environment variables named.

- **GPUs:** one node with 8 GPUs (`launch_common.sh` checks that each mini-batch splits over 8; the run
  script's `NGPU` defaults to 8). The memory settings were tuned on 80 GB H100s: one sequence per GPU per
  micro-batch, no parameter offload, vLLM `gpu_memory_utilization` 0.6, `ROLLOUT_TP=1`, fused log-prob
  kernels for the 32k responses. GPUs with less memory need retuning.
- **Software:**
  - docker with GPU access, and tmux;
  - training runs in the `frontiersmith-train` image, built from `docker/Dockerfile.train` on first use
    by `scripts/run_in_container.sh`. It mounts `$HOME` and reuses the project's `.venv`, so the venv has
    to be rebuilt on the new machine;
  - the finished runs used Python 3.11, vllm 0.20.0+cu129, torch 2.11.0+cu129, transformers 5.7.0,
    flash-linear-attention 0.5.2, ray 2.55.1 and the vendored verl. Keep these for the arms to be
    comparable with the finished runs;
  - the Frontier-CS judge in its own container on port 8082 (`Frontier-CS/algorithmic/run_judge.sh`);
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
  On 2026-10-30 this rebuilt a `full.parquet` identical to the one the finished runs used.
- **Checkpoint storage**, `CKPT_DIR` (default `/dev/shm/frontiersmith_ckpt/<run>`, which is RAM):
  - a full checkpoint (weights + Adam) is about 103 GB. A save briefly holds two, so a run needs about
    210 GB free at its peak;
  - a finished run keeps about 36 GB once its final optimizer state is deleted (about 68 GB,
    `$CKPT_DIR/global_step_300/actor/optim_*`);
  - the supervisor starts an attempt only while `/dev/shm` has `SHM_MIN_FREE_GB` free (default 120). It
    checks `/dev/shm` even when `CKPT_DIR` points elsewhere; set `SHM_MIN_FREE_GB=0` then.
- **Rollout dumps**, about 18 MB per step (about 5.5 GB per run): unless `outputs/rollout_data_<run>`
  already exists, the launcher links it to `/dev/shm/frontiersmith_rollouts/<run>`. To put them
  elsewhere, create that symlink yourself first (or a real directory, with `ALLOW_ROOT_ROLLOUT_DIR=1`).
- **Under `outputs/`:** validation dumps (about 0.25 GB per run once gzipped) and the plugins'
  per-step JSON logs (about 30 MB per run).
- **Judge log:** the judge container's docker log grows about 2 MB per training step: about 0.75 GB per
  run for arms 1–4 and 1.3 GB for the fixed-threshold run, unless docker caps its logs.

## How we judge the result

**Primary test, fixed before launch: arm 2 (adaptive, x = 0.75) against arm 1 (GRPO with dynamic sampling).**
- **Metric:** Avg@5 averaged over validation steps 260, 280 and 300, paired by problem (170 problems), with a paired bootstrap 95% CI.
- **Provisional success:** the CI's lower end is above 0.
- **Confirmed success:** a repeat of both arms with a second seed also shows a positive gap. The CI covers problem and sampling noise only, not seed-to-seed variation.
- **Detection limit:**
  - The paired SE of the window gap is about 0.5.
  - A true 1-point gain clears the bar only about half the time; about 1.5 points is needed for 80% power.
  - Single checkpoints are not compared; they move by about 1 point with response-length swings.
- **The x = 1.0 arm:** arm 3 against arm 1 is secondary. If only x = 1.0 clears the bar, it is a lead for the second-seed repeat, not the goal met.
- **Reference:** every arm is also reported against the finished regular GRPO run (no dynamic sampling, window mean 13.85). Those runs were on another machine, so all five arms must run on the same machine, and comparisons with the finished runs also include any hardware or software difference.
- **Seed knob:** the launcher has none, and `data.seed` is unset, so a repeat needs one added first (vLLM sampling seed, and possibly the data order).

**Secondary.**
- **Best-of-5** over the same window.
- **Speed:** Avg@5 at steps 20–100, both per training step and per generated rollout (cumulative `dynamic_sampling/rollouts_generated`). The fixed-threshold arm samples about twice as much per step early.
- **Failure split** on validation: cut off at 32k, rejected by the judge, no code.
- **Training side:**
  - informative groups per step;
  - dynamic-sampling rounds, cap hits and outage stops;
  - training reward per problem over step windows;
  - gradient norm and response length in steps 1–20.
- **Thresholds:** how tau evolves (`adaptive_threshold/*`: median, raised share, pass rate), compared with the replay.
- **Effect of dynamic sampling alone:**
  - Arm 4 vs the finished mean-quantized run is the cleanest check (it also crosses machines).
  - Arm 1 vs the finished regular run also differs in the GDN kernel for steps 1–20, so compare them from step 21.

**What each outcome means.**
- **Adaptive beats arm 1:** repeat both with a second seed before calling the goal met.
- **Adaptive is worse than arm 1** (CI's upper end below 0): drop this rule. Arm 4 shows whether binarizing itself is the cost.
- **Arm 1 ends below the finished regular run** (window 13.85): dynamic sampling hurt GRPO. Then adaptive must also beat 13.85 before claiming the goal.
- **Adaptive ties arm 4:** carrying a per-problem bar adds nothing over splitting at the group mean.
- **Adaptive ties arm 1 but beats arm 5 (fixed 50):** adaptivity helps relative to a fixed bar but not relative to raw rewards. Next, try how fast the bar rises (x) or where it starts.
- **Fixed 50 matches adaptive:** the adaptive part is not what matters.
- **x = 0.75 vs 1.0:** shows whether a faster-rising bar helps or hurts.

## Monitoring

- **Watch:** the run's log, its supervisor and its tmux session (there is no queue for these runs), the judge, and vLLM hangs (GPUs idle in the middle of generation).
- **After any restart:** check that the judge is still up and grading.
- **Validation:** each dump is gzipped by hand once the next one lands, and the arms are compared at every validation step.
- **Dynamic-sampling health:** rounds per step, cap hits, outage stops, the share of informative groups, and `judge_outage_frac_all`.
- **Thresholds:** dropped groups also update tau under dynamic sampling, so tau may climb faster than the replay suggests. Watch `adaptive_threshold/tau_median` and `frac_revisits_raised`.
- **Fixed threshold:** `fixed_threshold/pass_rate` and `frac_groups_mixed` (over every generated group). If few groups are mixed, dynamic sampling hits its 6-batch cap and steps slow down.
- **After each run:** delete its final optimizer state (about 68 GB, `$CKPT_DIR/global_step_300/actor/optim_*`) if checkpoint storage is short.

## Files

- `scripts/adv_adaptive_threshold.py`: the adaptive-threshold estimator and its saved state.
- `scripts/dynamic_sampling.py`, `scripts/main_ppo_dynamic.py`: the dynamic-sampling plugin and its entry point.
- `scripts/adv_fixed_threshold.py`: the fixed-threshold estimator (run 5).
- `scripts/launch_dynamic_sampling_grpo.sh`: the launcher for all five runs.
- `verl/verl/experimental/agent_loop/single_turn_agent_loop.py`: vLLM `max_tokens` = response length (the 32k stop).
