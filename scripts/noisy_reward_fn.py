"""Noisy reward for VERL custom_reward_function: the true task reward plus controlled noise.

Wraps VERL's default task reward (the Frontier-CS judge for data_source="frontiercs",
a 0-100 partial-credit score) and corrupts it ON TRAINING ROWS ONLY:

  noise_type="gaussian":  r + N(0, noise_std^2)          zero-mean; clipped only if clip=True
  noise_type="random":    with prob noise_p, r -> U[noise_low, noise_high]
                          (noise_p=1 is the paper's random-reward baseline)
  noise_type="subsample": the judge's score recomputed on a random subset of the problem's
                          test cases: 100 * mean(scoreRatio) over min(noise_k, n_cases) of
                          the n_cases cases (9-10 for the training problems), drawn without
                          replacement. Realistic "too few tests" noise: unbiased, bounded
                          like the clean score, variance s^2/k * (n-k)/(n-1) (s^2 = variance
                          of the per-case ratios, so 0 on all-pass / all-fail rows), and
                          noisy == clean bit for bit when noise_k >= n_cases.
  noise_type="none":      clean reward (sanity check that the wrapper changes nothing)
  noise_on_zero=False:    leave rows whose clean score is exactly 0 untouched, so all-zero
                          groups stay flat (zero advantage) as in the clean baseline

noise_type has no default: it must be set explicitly (no silent default noise model).

Subsample details: the judge's score is the unweighted mean over all cases of the per-case
scoreRatio (Frontier-CS/algorithmic/judge/src/judge_engine.js judgeDefault /
judgeInteractive: no subtask weights, no clamp; "score" = sum(scoreRatio) / n * 100), and
GET /result/<sid> returns the per-case list, which verl's frontiercs.compute_score drops.
So for training rows this wrapper submits to and polls the judge itself, replicating
frontiercs.compute_score (same helpers, endpoints, timeouts, retries, FRONTIERCS_MAX_WAIT
and judge_status codes; still one submission per row) and keeping res["cases"]. clean is
exactly what frontiercs.compute_score returns; the subset score is summed in the judge's
order and arithmetic. Whenever there are no usable per-case results (no code, timeout,
client/HTTP error, judge error, missing/empty/non-numeric cases) noisy = clean.
Validation rows still go through verl's default_compute_score unchanged.

Train vs validation: verl calls the same reward function for both and passes no split
flag, and both parquets use data_source="frontiercs" with no extra_info. The rows are told
apart by ground_truth: the FrontierSmith training problems are "frontiersmith_<n>", the
Frontier-CS validation problems are numeric ids. Rows whose ground_truth does not start
with noisy_gt_prefix get the judge's result dict unchanged, so validation metrics
(val-core/<ds>/acc, val-aux/<ds>/{reward,score,judge_status}) are exactly the baseline's.

The noise (Gaussian draw, replacement draw, or case subset) is a deterministic function of
(noise_seed, ground_truth, response), so it is independent across samples, identical
across the reward workers / threads and restarts, and reproducible.

Training rows return {"score": noisy (what GRPO trains on), "acc": clean,
"judge_status": passed through, "clean_score": clean, "noise": noisy - clean}. verl does
not turn these extra keys into per-step training metrics (critic/score/* is the NOISY
score); they are written to the rollout dumps (trainer.rollout_data_dir/<step>.jsonl).

Configure through Hydra (reward_kwargs is not in the base config, hence the "+"); use an
absolute path, the function is imported inside the Ray reward-loop workers:
  reward.custom_reward_function.path=$PROJECT_ROOT/scripts/noisy_reward_fn.py
  reward.custom_reward_function.name=compute_score
  +reward.custom_reward_function.reward_kwargs.noise_type=gaussian
  +reward.custom_reward_function.reward_kwargs.noise_std=10
or, for test-case subsampling:
  +reward.custom_reward_function.reward_kwargs.noise_type=subsample
  +reward.custom_reward_function.reward_kwargs.noise_k=5
Unknown reward_kwargs (typos) raise instead of being silently ignored.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import os
import time

import numpy as np
import requests

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import frontiercs as fcs

NOISE_TYPES = ("none", "gaussian", "random", "subsample")
# kwargs that verl's reward managers may add to the call; forwarded to the task scorer.
_PASSTHROUGH_KWARGS = frozenset({"reward_router_address", "reward_model_tokenizer"})
_announced = False
_subsample_fallback_warnings = 0


def _rng(seed: int, ground_truth, solution_str: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{ground_truth}|{solution_str}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def is_noisy_row(ground_truth, noisy_gt_prefix: str) -> bool:
    return str(ground_truth).startswith(noisy_gt_prefix)


def judge_with_cases(data_source, solution_str, ground_truth, extra_info=None):
    """frontiercs.compute_score (same requests, retries, timeouts, status codes and return
    value; called there with judge_url=None) that also returns the judge's per-case results.

    Returns (result, cases): result is exactly frontiercs.compute_score's dict, cases is
    res["cases"] of a "done" verdict (unvalidated) or None.
    """
    if data_source != "frontiercs":
        raise ValueError(f"noisy_reward_fn: noise_type='subsample' needs data_source 'frontiercs', got {data_source!r}")

    url = ((extra_info.get("judge_url") if extra_info else None) or fcs.DEFAULT_JUDGE_URL).rstrip("/")
    problem_id = str(ground_truth)
    code = fcs.extract_cpp(solution_str)

    if not code:
        return fcs._result(0.0, fcs.STATUS_NO_CODE), None

    def submit():
        r = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": problem_id, "lang": "cpp"},
            timeout=fcs.SUBMIT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("sid")

    def poll(sid):
        r2 = requests.get(f"{url}/result/{sid}", timeout=fcs.POLL_TIMEOUT)
        if r2.status_code == 404:
            return None  # not ready yet
        r2.raise_for_status()
        return r2.json()

    try:
        sid = fcs._with_retries(submit, "submit")
        if not sid:
            return fcs._result(0.0, fcs.STATUS_CLIENT_ERROR), None

        start = time.time()
        while time.time() - start < fcs.MAX_WAIT:
            res = fcs._with_retries(lambda: poll(sid), "poll")
            if res is not None:
                status = res.get("status")
                if status == "done":
                    raw_score = float(res.get("score", 0))
                    clean = fcs._result(fcs.normalize_frontiercs_score(raw_score, problem_id), fcs.STATUS_JUDGED)
                    return clean, res.get("cases")
                if status == "error":
                    return fcs._result(0.0, fcs.STATUS_JUDGE_ERROR), None
            time.sleep(fcs.POLL_INTERVAL)
        return fcs._result(0.0, fcs.STATUS_TIMEOUT), None
    except Exception:
        return fcs._result(0.0, fcs.STATUS_CLIENT_ERROR), None


def subsample_score(cases, k: int, rng: np.random.Generator, ground_truth) -> float | None:
    """100 * mean(scoreRatio) over min(k, n) of the n cases, drawn without replacement.

    Summed left to right in the judge's case order and divided as the judge does
    (sum / n * 100), so k >= n reproduces the judge's score bit for bit (an explicit loop:
    Python >= 3.12 sum() uses compensated summation, JavaScript does not). Returns None if
    the per-case results are missing, empty or non-numeric.
    """
    if not isinstance(cases, list) or not cases:
        return None
    try:
        ratios = [float(c["scoreRatio"]) for c in cases]
    except (TypeError, KeyError, ValueError):
        return None
    if not all(math.isfinite(x) for x in ratios):
        return None
    m = min(k, len(ratios))
    total = 0.0
    for i in np.sort(rng.choice(len(ratios), size=m, replace=False)):
        total += ratios[i]
    return fcs.normalize_frontiercs_score(total / m * 100, str(ground_truth))


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    noise_type: str | None = None,
    noise_std: float = 10.0,
    noise_p: float = 0.0,
    noise_low: float = 0.0,
    noise_high: float = 100.0,
    noise_k: int = 5,
    noise_seed: int = 0,
    clip: bool = False,
    clip_low: float = 0.0,
    clip_high: float = 100.0,
    noise_on_zero: bool = True,
    noisy_gt_prefix: str = "frontiersmith_",
    **kwargs,
):
    global _announced, _subsample_fallback_warnings
    unknown = set(kwargs) - _PASSTHROUGH_KWARGS
    if unknown:
        raise TypeError(f"noisy_reward_fn: unknown reward_kwargs {sorted(unknown)}")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noisy_reward_fn: set reward_kwargs.noise_type to one of {NOISE_TYPES}, got {noise_type!r}")
    if noise_type == "subsample" and (
        isinstance(noise_k, bool) or not isinstance(noise_k, numbers.Integral) or noise_k < 1
    ):
        raise ValueError(f"noisy_reward_fn: reward_kwargs.noise_k must be an int >= 1, got {noise_k!r}")
    if not _announced:
        _announced = True
        print(
            f"[noisy_reward_fn pid={os.getpid()}] noise_type={noise_type} noise_std={noise_std} noise_p={noise_p} "
            f"U[{noise_low},{noise_high}] noise_k={noise_k} seed={noise_seed} clip={clip}[{clip_low},{clip_high}] "
            f"noise_on_zero={noise_on_zero}; noisy rows: ground_truth startswith {noisy_gt_prefix!r}",
            flush=True,
        )

    noisy_row = is_noisy_row(ground_truth, noisy_gt_prefix)
    cases = None
    if noise_type == "subsample" and noisy_row:
        result, cases = judge_with_cases(data_source, solution_str, ground_truth, extra_info=extra_info)
    else:
        result = default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    if not noisy_row:
        return result  # validation (or any non-training row): the untouched judge result

    out = dict(result) if isinstance(result, dict) else {"score": result}
    clean = float(out["score"])
    out.setdefault("acc", clean)

    rng = _rng(noise_seed, ground_truth, solution_str)
    if noise_type == "none" or (clean == 0.0 and not noise_on_zero):
        noisy = clean
    elif noise_type == "gaussian":
        noisy = clean + float(rng.normal(0.0, noise_std))
    elif noise_type == "random":
        noisy = float(rng.uniform(noise_low, noise_high)) if rng.random() < noise_p else clean
    else:  # "subsample"
        sub = subsample_score(cases, noise_k, rng, ground_truth)
        if sub is None and out.get("judge_status") == fcs.STATUS_JUDGED and _subsample_fallback_warnings < 5:
            _subsample_fallback_warnings += 1
            print(f"[noisy_reward_fn pid={os.getpid()}] subsample: judged {ground_truth} without usable "
                  f"per-case results; noisy = clean", flush=True)
        noisy = clean if sub is None else sub
    if clip:
        noisy = min(max(noisy, clip_low), clip_high)

    out.update(score=noisy, clean_score=clean, noise=noisy - clean)
    return out
