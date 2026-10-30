# Memo: Unbiased advantage estimators for continuous (partial-credit) rewards in the FrontierSmith GRPO pipeline

Setting: Qwen3.5-9B, 200 competitive-programming prompts, deterministic judge r in [0,100], 8 prompts x n=8 per step, ppo_epochs=1, one mini-batch per step (ratio == 1, clipfrac = 0), AdamW lr 1e-6, grad_clip 1.0, KL coef 0.001. Evidence: a code audit of the vendored verl, one adversarial math referee per idea (exact-enumeration toys), and two independent Monte-Carlo simulations with closed-form true gradients (`analysis/toy_unbiasedness_sim_a.py`, `analysis/toy_unbiasedness_sim_b.py`). Reward noise is out of scope.

## 1. Definition, and why the current pipeline is biased

**Definition used throughout.** J(theta) = E_{x~D} E_{y~pi_theta(.|x)}[r(x,y)]. The applied update is g_hat = (aggregation of) A_i * sum_t grad log pi(y_it | x_i, y_i<t). An estimator is *unbiased for J* if E[g_hat | theta] = kappa * grad J with kappa > 0 not depending on the current samples (it may depend on theta, past batches, fixed hyperparameters). It is *unbiased for a reweighted J_w = E_x[w(x) J_x]* only if w is a fixed function of x or of past data. Baselines may depend on x, on the other samples, on past data, or on the prefix y_<t, never on the sample's own reward or own future tokens. Everything else is "biased", and we distinguish *direction* bias (not fixable by rescaling) from *scale-only* factors.

**Why the running GRPO pipeline is biased (all items verified in the fork and reproduced in both sims):**

1. **Per-group std from the current samples (direction bias + sample-dependent cross-prompt weight).** A_i = (r_i - mean_g)/(std_g + 1e-6) is a bounded nonlinear transform of r_i (|A| <= (n-1)/sqrt(n) = 2.47), so E[A_i S_i] is the gradient of a sample-dependent transform of the reward, not of r. Measured: 4% of non-flat groups span < 2 points yet get full-size advantages ([50,...,50.15,50] -> A = +2.47 for the 50.15); in the toys that near-flat prompt receives 326-384x its J-share of gradient and within-prompt cosines are 0.94-0.998. Not a legal J_w either: the intended 1/sigma_x weight is estimated from the same 8 samples that form the numerator (27% relative error, correlated with r_i).
2. **Aggregation actually running is per-sequence token-mean, not batch token-mean.** ppo_micro_batch_size_per_gpu=1 with the legacy worker gives agg_loss('token-mean') a one-sequence micro-batch, i.e. weight 1/L_i per sequence (confirmed by actor/pg_loss ~ 0 at step 81, which a batch token-mean cannot produce). E[(r-b) S / L] = grad E[(r-b)/L]: the baseline term -b grad E[1/L] does not vanish, pushing r > b responses shorter and r < b longer (Dr.GRPO's length bias, biting through the measured within-group corr(r,L) = -0.14). Toy: RLOO under this aggregation has cos 0.965-0.968 and 26-27% direction residual; GRPO under it has cos 0.56-0.86.
3. **Scale-only, harmless:** self-inclusion in the group mean ((n-1)/n), eps=1e-6, inactive clipping (ratio == 1), grad_clip inactive at baseline norms 0.1-0.5.
4. **Not currently running but relevant:** a global batch token-mean denominator (new-engine path) is a ratio estimator with 0.5-1% direction residual (5x the MC floor, z = 3.6; the audit's "cos > 0.99999" was slightly optimistic); `reinforce_plus_plus_baseline` whitening is biased (token-weighted batch mean lifts every advantage, 5% measured sign flips, nonzero expected gradient on an all-equal prompt in the toy); verl's shipped `opo` is biased 2.4-3.9% (own L_i, r_i enter the baseline denominator).

Net: the two direction biases (1) and (2) are independent; fixing only one leaves cos 0.93-0.99 in the toys. Any fix needs a baseline independent of y_i, no current-sample normalizer, and a constant loss denominator.

**Implementation fact that governs every "constant" below:** on the legacy worker path (`trainer.use_legacy_worker_impl=auto`, which all 9B scripts use), `config.global_batch_info` is never filled, so with `loss_agg_mode=seq-mean-token-sum-norm` agg_loss divides each sequence's token sum by seq_mask.sum()=1 and by `loss_mask.shape[-1]` = padded `rollout.response_length` = `data.max_response_length` = **16000** in every `scripts/run_verl_grpo_frontiercs_qwen35_9b*.sh` (one referee quoted 32000; the scripts say 16000). `actor.loss_scale_factor` is silently ignored on this path. S is constant (good) but not configurable there; fold any desired constant into the advantage and into kl_loss_coef.

## 2. Ranked table

Cos / bias = cosine to true gradient and relative residual after best scale, sim A / sim B (MC floors 0.1-0.2%). relVar = trace Var(g)/kappa^2 relative to RLOO under constant aggregation, sim A / sim B. "Confirmed" means proposer, referee, and both sims agree unless flagged.

| # | Estimator (formula) | Unbiased for | Cos / bias (A / B) | Continuous-reward behaviour (0.15-pt gap; flat group; cross-prompt) | relVar (A / B) | verl effort | Risks |
|---|---|---|---|---|---|---|---|
| 1 | **rloo-const**: A_i = (r_i - mean_{-i})/c, c = 18 fixed; seq-mean-token-sum-norm | **J**, kappa = 1/(S c). Confirmed (toy ratio identical on every coordinate; drgrpo centering = same x 7/8) | 1.000 / 1.000; 0.06% / 0.17% (at floor) | 0.15 pt -> 0.008 (GRPO 2.47); flat -> exactly 0; reward units everywhere (100-vs-0 -> 5.6, uncapped); high-spread prompts dominate (J-consistent) | 1.00 / 1.00 (reference) | Config + 3-line plugin (or config-only `rloo` with rescaled kl_coef/grad_clip) | Early all-flat phase gives tiny gradients, Adam beta2=0.999 does not re-normalize; long responses carry gradient share ~L_i/S (true REINFORCE profile); uncapped A up to 5.6 with S=16000 -> verify grad_clip stays inactive; KL per sequence re-weighted 1/L_i -> 1/S |
| 2 | **rloo-lagged-scale (EMA)**: c_t = max(sqrt(vbar_{t-1}), c_min); vbar updated AFTER the step | **J** conditionally on F_{t-1}, kappa_t = 1/(S c_t). Confirmed. Leak control (current step folded in before use): 0.7% bias, 2-2.5% cross-prompt reweighting; no-lag (batch RMS): 10.6% bias | 1.000 / 1.000; 0.07% / 0.17% | Same as #1 within a step; scale tracks the flat -> spread transition (replay on 105 saved steps: RMS(A) median 0.91, p90 1.26, max 1.78 vs GRPO's 0.90) | 1.007 / 1.006 | Plugin + 5 lines of persisted state | One-step lag -> 1.5-3x bursts after a spread jump; c_min load-bearing; must update exactly once per step, after advantages |
| 2b | leave-one-group-out c_{-g} (stateless) | Proposer: "J, negligible coupling". **Referee: under without-replacement sampling it is E_x[w_theta(x) grad J_x], w spread <= 1.5% on 200 prompts (12x in a 2-prompt toy).** Sims (with-replacement draws) show unbiased for J. Trust referee: prefer the EMA form | 1.000 / 1.000 | as #2 | **1.18 / 1.13** (7-group scale estimate is noisy; floor active when others are flat) | Plugin | 13-18% variance for no bias benefit |
| 3 | **is-prompt-sampling**: draw prompts from q_t(x) ~ s(x) (past-data spread, floored), A_i = (r_i - mean_{-i})/(|D| q_t(x)) | **J**, kappa = 1/S. Confirmed; control without the 1/q weight is unbiased for J_q only (19.6% bias vs J); without-replacement draw with u applied: biased (cos 0.85) | 1.000 / 1.000; 0.05% / 0.16% | Small gaps scaled by <= 1/s_min; flat prompts drawn at the floor rate and contribute 0; per-sampled-prompt advantages ~unit RMS yet the target stays uniform J | **0.49 / 0.69 (best)** | **Most engineering**: custom `AbstractCurriculumSampler` (needs `data.dataloader_num_workers=0`), estimator reads u(x), state for resume | q/u mismatch silently reweights (log mean(u q |D|) == 1); must sample WITH replacement; epoch accounting changes; starvation if s_min too small |
| 4 | **rloo-hist-baseline**: b_i = (7 mean_{-i} + k h(x))/(7+k), h = EMA of past visits' group means, k from past drift | **J**, kappa = 1/(S c). Confirmed. **Disagreement**: proposer said a current-group-mean leak into h "breaks" it; referee + sim A show a linear leak is scale-only (kappa x (1 - k beta/(n(n-1+k)))); only nonlinear current-sample dependence in k or c biases direction (cos 0.9989 when k is from the current group) | 1.000 / 1.000; 0.06% / 0.17% | Linear; flat group with history -> all 8 get the same small zero-mean A (bounded by k/(7+k)) | 0.966 / 0.967 (toy, stationary policy; real-data proxy 0.977 at k=1-2, 1.008 at k=4, 1.09 at k=8 under the measured 10-27 pt drift) | Plugin + stable prompt key (hash of un-padded prompt ids; `index` is not stable) + persisted history | Gain 1-3.5% only, 0 in epoch 1; k too large is worse than RLOO; stale history on a prompt that jumped 0 -> 100 |
| 5 | **lagged-gradnorm-scale**: A_i = a_t (r_i - mean_{-i}), a_t = g_target / max(Gbar_{t-1}, G_min), Gbar = EMA of past raw pre-clip grad norms | **J**, kappa_t = a_t/S. Confirmed; control normalizing by the current batch's own norm: 16.4% bias (= what frequent clipping degenerates into) | 1.000 / 1.000; 0.07% / 0.17% | As #2; invariant to reward units, S, and length drift (mean length went 14k -> 6.7k over 105 steps, a ~2x gradient-scale drift a reward-std EMA does not see) | 1.023 / 1.01 | Plugin + monkey-patch of `Tracking.log` (fragile) or driver-side proxy | Degenerate after all-flat runs (G_min + skip rule essential); norm includes KL term; "big-win" steps are not bigger steps |
| 6 | **opo-loo**: b_{-i} = sum_{j!=i} u_j^alpha r_j / sum_{j!=i} u_j^alpha, u = L (alpha fixed before the step; gate from past rho^2(r,L) > 1/7) | **J** for any pre-fixed alpha, kappa = 1/(S c) — ONLY under token-sum aggregation (under the running per-sequence token-mean it targets grad E[r/L] - E[b] grad E[1/L], neither J nor a J_w). Confirmed; shipped verl `opo` biased 2.4-3.9% (z = 9) | 1.000 / 1.000 | Linear; flat -> 0 | 1.017 / 1.019 (net loss, as predicted: measured |rho(r,L)| ~ 0.15 < 0.38) | Plugin | No gain at measured coupling; mildly pushes toward shorter responses (dynamics); keep only as the unbiased stand-in whenever OPO/OTB appears in an ablation |
| 6b | position-level LOO (survival W=1, or OTB-style cumulative W from others' tokens) | **J**. Confirmed; shipped verl OTB (own current token in W, own trajectory in group) biased 4.5% / 10-15% exact | 1.000 / 1.000 | Linear; (bs,T) tensor | 1.024 / 1.027 | Plugin; OTB-W needs `calculate_sum_pi_squared` forwarded | Fewer samples alive late -> worse than RLOO (real-data proxy 1.046) |
| 7 | **prefix-feature-baseline**: b_i(t) = mean_{-i} + clip(beta^T(phi(y_<t) - mean of others alive at t), s_cap), beta ridge-fit on past steps | **J**. Confirmed at machine precision in the referee's enumeration; illegal variants (own current token, own L_i, own full-response features) detected at 14-18% residual | 1.000 / 1.000 | Linear; flat groups contribute zero-mean noise of size |beta| spread(phi) (clip + ridge) | 0.996 / 0.993 (prefix features explained ~2% of within-group variance in the toys) | Plugin + forwarding old_log_probs/entropys + ring buffer; offline R^2 study needs a verl dump edit | A small critic in disguise; "safe but useless" unless offline within-group R^2 >~ 0.1; exclusive cumsum for log-prob features is load-bearing |
| 8 | **prefix-branching (as specified)**: m prefixes x k continuations; prefix-token A = (r_ab - b_{-a})/k, continuation A = r_ab - b_{a,-b} | **REFUTED (biased).** Proposer claimed J. Referee: E[g_x] = m P_x + m k C_x (prefix part under-weighted by k; sign can flip: toy batch E[g] = +21.5 vs grad J = -1.0). Sim A: 1.8-4.9% bias, 17-30x floor; sim B: cos 0.959, per-prompt z 60-109 | 0.9998 / 0.959; 1.8% / 29% | — | 0.45 / 1.70 | Rollout-path change (two-stage generation, parent ids, prefix_len) | Do not use as written |
| 8b | **prefix-branching, corrected**: no /k on prefix tokens (every copy carries the full r_ab - b_{-a}) | **J**, kappa = 1/(S c). Confirmed by referee enumeration and both sims (residual at floor; kappa equals rloo-const's) | 1.000 / 1.000 | Pairing is where continuous credit shines (siblings rarely tie) | 0.47 / 1.45 — contradictory because the toys cannot represent long shared prefixes (A's gain is driven by filler tokens; B has T0=1 and an m-1=3 prefix baseline) | Largest engineering lift of the set | KL on shared prefix tokens counted k times (divide by k); reduced prefix diversity; T0 is a new hyperparameter; do not copy TreeRL/Tree-GRPO formulas |
| 9 | **rloo-prompt-scale**: A_i = (r_i - mean_{-i})/s_x, s_x from PAST visits shrunk toward vbar, floored at s_min | **Not J. J_w^{(t)}** with w = 1/s_x from past data (moving target); frozen / task-known variant: fixed J_w. Confirmed: per-prompt direction exact (cos 1.000), realized cross-prompt weights equal claimed 1/s_x to 3 decimals. **Referee correction**: "per-prompt optima coincide with J's" is false under shared parameters + KL penalty (w(x) is a per-prompt inverse temperature of the KL-regularized optimum) | 0.907 / 0.956 vs J by construction | Small gaps -> <= 0.15/s_min; flat -> 0; equalizes prompts (GRPO's intent, legally), but floored prompts (46% at s_min=5) are up-weighted up to c/s_min = 3.6x: near-flat (solved/hopeless) prompts gain budget | 1.26 / 1.20 in J-units | Plugin + prompt key + `compute_advantage` wrapper | Up-weights the prompts with the least headroom; history predicts spread weakly (corr 0.45); first ~25 steps have no history |
| 9b | cross-fit variant: /(s_{-i} + eps0) (std of the other 7) | **Biased under the definition** (all three agree): exact per-prompt direction, but weight E[1/(s_{-i}+eps0)] depends on the current theta | 0.958 / 0.930 | advantages no longer sum to 0 per group; s_{-i}=0 whenever the other 7 tie | 1.26 / **2.16** | Plugin | Doubles variance; skip |

References (for calibration): GRPO|token-mean cos 0.806 / 0.914, relVar 1.46 / 1.26; GRPO|per-sequence token-mean (what runs today) cos 0.563 / 0.856; GRPO|ssn cos 0.805 / 0.914 (aggregation fix alone does nothing for the std defect); Dr.GRPO|ssn unbiased (kappa x 7/8); RLOO|batch token-mean 0.5-1% residual; verl OPO 2.4-3.9%; R++ baseline 10-11%, cos 0.994, nonzero gradient on an all-equal prompt; illegal own-length control detected at z = 220.

**Where I side when sources disagree.** Prefix-branching: the referee and both sims refute the proposer, the counting error is explicit (m prefix draws vs m k continuation draws), and the fix is verified — trust the refutation. Leave-one-group-out: the referee's without-replacement argument is the one that matches verl's `RandomSampler`; the sims drew with replacement and so could not see it — trust the referee, and the variance penalty settles it anyway. Hist-baseline leak: the referee and sim A both show the linear leak is scale-only; the proposer's update-order rule is still correct practice. kappa bookkeeping: proposers wrote 1/(G S c); referees show 1/(S c) (the 1/G and DP mean are the 1/N that cancels the N = 8n group terms) — scale-only. S: 16000, per the scripts.

## 3. Recommendation

Three training runs, in this order. All require `loss_agg_mode=seq-mean-token-sum-norm`, `ppo_mini_batch_size == train_batch_size` (ratio stays 1), and the plugin imported in the driver process.

**Run A — rloo_scaled (minimal fix; do this first).** Removes both direction biases with one estimator and one aggregation flag.

```
algorithm.adv_estimator=rloo_scaled \
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.grad_clip=1.0            # verify inactive; if it fires, raise ADV_SCALE (never add a sample-dependent normalizer)
# env: ADV_SCALE=18
```
Config-only fallback with no plugin: `algorithm.adv_estimator=rloo` + the same loss_agg_mode, with advantages in raw reward units (18x larger): set `kl_loss_coef=0.018` and `grad_clip=10` so clipping stays inactive. Ablation ladder for the write-up: (a) grpo + ssn (length fix only, still biased), (b) rloo_scaled + current token-mean (std fix only, 0.5-27% biased depending on path), (c) both.

**Run B — rloo_lagged_scale (EMA beta=0.9, c_min=3, s0=18).** Same as A plus a predetermined global scale that follows the 83%-flat -> spread transition; the only legal replacement for GRPO's step-size normalization. Optional cheap add-on once prompt keys exist: the hist baseline with k set from measured drift (k ~ 2; never 8).

**Run C — is-prompt-sampling** if a 30-50% variance reduction at equal rollout cost is worth building the sampler. Cheapest first step: `rloo_lagged_scale` plus a spread-proportional sampler and the 1/(|D| q) weight; log the invariant every step.

Plugin sketch (new file, e.g. `analysis/adv_plugins.py`; no edits under `verl/`):

```python
import os, json, math, numpy as np, torch
from verl.trainer.ppo.core_algos import register_adv_est, compute_rloo_vectorized_outcome_advantage as _rloo

_STATE = {"v_bar": float(os.environ.get("LAG_S0", 18.0)) ** 2}
_PATH = os.environ.get("LAG_STATE_PATH")
if _PATH and os.path.exists(_PATH):
    _STATE.update(json.load(open(_PATH)))

def _loo_dev_and_group_var(token_level_rewards, response_mask, index):
    adv, _ = _rloo(token_level_rewards, response_mask, index)      # (r_i - mean_{-i}) broadcast on tokens
    scores = token_level_rewards.sum(-1).cpu().numpy()
    inv = np.unique(index, return_inverse=True)[1]
    cnt = np.bincount(inv); s1 = np.bincount(inv, scores); s2 = np.bincount(inv, scores**2)
    var_g = np.where(cnt > 1, (s2 - s1**2 / cnt) / np.maximum(cnt - 1, 1), 0.0)   # ddof=1, flat -> 0
    return adv, float(var_g.mean())

@register_adv_est("rloo_scaled")
def rloo_scaled(token_level_rewards, response_mask, index, config=None, **kw):
    c = float(os.environ.get("ADV_SCALE", 18.0))
    adv, _ = _loo_dev_and_group_var(token_level_rewards, response_mask, index)
    return adv / c, adv / c

@register_adv_est("rloo_lagged_scale")
def rloo_lagged_scale(token_level_rewards, response_mask, index, config=None, **kw):
    beta = float(os.environ.get("LAG_BETA", 0.9)); c_min = float(os.environ.get("LAG_CMIN", 3.0))
    c_t = max(math.sqrt(_STATE["v_bar"]), c_min)                   # read BEFORE the update
    adv, v_now = _loo_dev_and_group_var(token_level_rewards, response_mask, index)
    _STATE["v_bar"] = beta * _STATE["v_bar"] + (1 - beta) * v_now  # update AFTER; exactly once per step
    _STATE["c_t"] = c_t
    if _PATH: json.dump(_STATE, open(_PATH, "w"))
    return adv / c_t, adv / c_t
```
Driver import: a thin launcher that subclasses `TaskRunner` (`verl/trainer/main_ppo.py:108`) with `def run(self, config): import analysis.adv_plugins; super().run(config)` and calls `run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(NoisyTaskRunner))`; any registered string in `algorithm.adv_estimator` falls through `compute_advantage`'s else-branch to `get_adv_estimator_fn`. Hyperparameters via env vars (`+algorithm.*` keys may be rejected by the structured config). The hist baseline and prompt-scale variants additionally need the `compute_advantage` wrapper that hashes `data.batch['prompts']` into a stable key (the per-step `uid` is a fresh uuid4; `non_tensor_batch['index']` is 0 for every row of the synthetic parquet).

**Diagnostics to watch (all runs).**
- `actor/grad_norm` (pre-clip): target median ~0.2-0.3 with >= 3x headroom to the clip; clip frequency must stay near 0 — a clipped step is a sample-dependent rescale. With S = 16000 and |A| up to 5.6 the proposer's "0.3x baseline" estimate is not guaranteed; raise c by a constant if needed.
- `actor/pg_loss` identity: must equal -(1/N) sum_i A_i L_i / S (log that quantity; under the baseline path it was ~0 because advantages sum to 0 per group).
- `actor/kl_loss`: KL is now token-sum-weighted per sequence; any constant folded into A must be folded into `kl_loss_coef` or the PG:KL ratio silently moves.
- Per-step advantage stats (`critic/advantages/*` plus RMS(A), max|A|, fraction of flat groups, and c_t for Run B): expect RMS ~1, max 3-6, flat groups exactly 0; Run B c_t should drift slowly (replay: 18 -> 14-16 over 105 steps).
- Response length and truncation rate at the 16000 budget: removing the 1/L_i weight removes the length penalty (Dr.GRPO's observation); long responses now carry ~L_i/S of the gradient.
- Run C only: mean(u(x) q_t(x) |D|) == 1 every step; realized visit frequencies vs s(x).

## 4. Scripts, reruns, open questions

Simulation A (autoregressive toy, 4 prompts, lagged state per chain):
`cd /home/shuoyuan/personal/noisy_grpo/FrontierSmith && .venv/bin/python analysis/toy_unbiasedness_sim_a.py --chains 6000 --steps 150 --json out_a.json` (~9 min CPU; defaults 3000 chains x 90 steps).

Simulation B (tabular autoregressive policy, 6 prompts incl. near-flat, all-equal, bimodal and length-penalized, 1M batches, two seeds):
`.venv/bin/python analysis/toy_unbiasedness_sim_b.py --nb 1000000 --warmup 5000 --seed 1 --json out_b1.json` (~4 min; repeat with `--seed 2`).

Offline replay of advantages on saved rollouts: `analysis/continuous_reward_advantages.py` (reads `outputs/rollout_data_synthetic/*.jsonl`). Referee enumeration scripts and sim logs/JSON live in the session scratchpad `/var/tmp/claude-1003/-home-shuoyuan-personal-noisy-grpo/83a962cf-965e-4f2a-bf5e-9369212c5246/scratchpad/` (referee_*.py, is_prompt_sampling_check.py, rloo_hist_check.py, prefix_branching_check.py, opo_loo_exact.py, prefix_baseline_check.py, final.log/json, sim_b_seed{1,2}.log/json, pf_diag.py); copy into `analysis/` if they should persist.

**What the toys cannot settle.**
1. Absolute gradient scale on the real model under token-sum aggregation (S = 16000, uncapped A, 1k-16k-token responses): whether grad_clip stays inactive, and whether long responses dominating the gradient (true REINFORCE variance) hurts in practice. Only a run shows this.
2. Learning speed under J-weighting vs GRPO's per-prompt equalization: unbiasedness says nothing about which objective reaches a better pass rate; the toys show the equalized objective (rloo-prompt-scale) costs 20-26% variance in J-units and up-weights solved/hopeless prompts, but not what the user's evaluation rewards.
3. Length dynamics once the 1/L_i weight is gone, and the re-weighted KL term.
4. The real variance benefit of the (corrected) tree estimator on long responses (sim A says 0.47, sim B says 1.45; neither models a long shared prefix); it needs the offline Var(r|x,z)/Var(r|x) measurement from truncated-prefix continuations before any rollout-path engineering.
5. The within-group R^2 of prefix log-prob/entropy features (prefix-feature baseline), which needs old_log_probs in the rollout dump.
6. History-based rows (hist baseline k, prompt-scale shrinkage) were simulated with a stationary policy; the measured 10-27-point drift between visits makes them less favourable than the toys suggest, and k must come from past-step drift, not the toy optimum.
7. Shared with every estimator and outside the definition: AdamW and grad_clip are sample-dependent nonlinearities; vLLM-vs-FSDP log-prob mismatch (rollout_correction off); without-replacement epoch sampling makes each step's conditional target the mean over not-yet-visited prompts (past-data reweighting, averages out over the epoch).

Nothing here is novel as an estimator: #1 is RLOO / Dr.GRPO with a constant denominator, #2/#5 are lagged normalizers, #3 is textbook importance sampling over prompts, #4 is a Bayesian-shrunk baseline, #7 is Greensmith-style state-dependent baselines, #8 is VinePPO/TreeRL with corrected accounting. The contribution is the audit of what the fork actually computes and the exact confirmation of which combinations are unbiased.