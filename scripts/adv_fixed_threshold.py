"""Fixed-threshold quantized GRPO: ``algorithm.adv_estimator=grpo_fixed_threshold``.

Like grpo_adaptive_threshold (scripts/adv_adaptive_threshold.py), but every rollout of every
problem is binarized against one fixed threshold c instead of a per-problem threshold that rises:

    r_i reaches c  iff  r_i >= c (up to float rounding) and r_i > 0
    b_i = 1 if r_i reaches c, else 0
    A_i = GRPO(b)_i = (b_i - mean_g(b)) / (std_g(b) + eps)

c = algorithm.fixed_threshold (required, a finite number; pass it as +algorithm.fixed_threshold=<c>;
the launcher's FIXED_THRESHOLD, default 50). "Reaches" is the adaptive estimator's own test (same
float tolerance; a score of 0 never reaches a threshold, which only matters for c <= 1e-6, i.e. c
about 0) and the advantages come from the same code, so the two arms differ only in where the
threshold comes from. A group in which every rollout reaches c, or none does, gets A = 0.
r_i is the sequence reward (token_level_rewards summed over the response); epsilon (1e-6) and
algorithm.norm_adv_by_std_in_grpo (default True) are vanilla GRPO's.

The threshold is checked when fit() starts (RayPPOTrainer._load_checkpoint, before any validation
or generation) and written to <ckpt>/global_step_N/fixed_threshold_state.json right before verl
saves step N. Resuming from a checkpoint without that file, or with a different threshold than the
file was written with, is an error: a run never mixes thresholds (e.g. a relaunch under the same
RUN_NAME with another FIXED_THRESHOLD). Nothing else is carried across steps.

The estimator is registered in verl's registry (it only needs the uid groups). Dynamic sampling
(scripts/dynamic_sampling.py) classifies the groups it generates through evaluate_groups
(registered in ray_trainer._frontiersmith_group_evaluators): a group is informative iff its b is
mixed. The groups evaluated at a step are remembered (uid, size, passes) until the estimator runs on
the step's training batch, which checks that every trained group was evaluated and has the same size
and number of passes again, so the same mixed/flat outcome (otherwise the rewards changed in between,
and dynamic sampling chose groups by an outcome that the update does not see).

Logged every step (wandb and console, through compute_data_metrics), prefix fixed_threshold/:
    threshold                               c
    pass_rate                               fraction of rollouts with b = 1
    frac_groups_none_pass / _all_pass / _mixed
                                            groups by outcome of b (_mixed = the ones with nonzero advantages)
    num_groups_evaluated, frac_groups_kept  with dynamic sampling only: groups generated this step (the other
                                            metrics cover all of them) and the fraction trained on
Without dynamic sampling they cover the training batch.

Loaded by scripts/dynamic_sampling.py. It loads scripts/adv_adaptive_threshold.py (which loads
scripts/adv_estimators.py) for the shared test and GRPO code.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import torch

from verl.trainer.ppo import core_algos, ray_trainer

NAME = "grpo_fixed_threshold"
KEY = "fixed_threshold"
STATE_FILE = "fixed_threshold_state.json"
STATE_VERSION = 1
METRIC_PREFIX = "fixed_threshold/"
_RUNTIME_ATTR = "_frontiersmith_fixed_threshold_runtime"  # on ray_trainer: makes installing idempotent
_EVALUATORS_ATTR = "_frontiersmith_group_evaluators"  # on ray_trainer: {estimator: evaluate_groups}


def _load_adaptive_threshold():
    """scripts/adv_adaptive_threshold.py, imported once per process (through it, scripts/adv_estimators.py)."""
    name = "frontiersmith_adv_adaptive_threshold"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_adaptive_threshold.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_adaptive = _load_adaptive_threshold()
_base = _adaptive._base


def threshold_from(config) -> float:
    """algorithm.fixed_threshold, checked: a finite int or float (not a bool or a string)."""
    c = _base._config_get(config, KEY, None)
    if c is None:
        raise ValueError(f"{NAME} needs algorithm.{KEY} (pass +algorithm.{KEY}=<c>)")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
        raise ValueError(f"algorithm.{KEY} must be a finite number, got {c!r}")
    return float(c)


def fixed_binarize(scores: torch.Tensor, threshold: float, rel_tol: float = _adaptive.REL_TOL) -> torch.Tensor:
    """b_i = 1.0 if scores[i] reaches ``threshold`` (scores[i] > 0 too), else 0.0; same dtype/device as scores."""
    s = scores.detach().to(torch.float64).cpu()
    if not torch.isfinite(s).all():
        raise ValueError(f"non-finite rewards: {s[~torch.isfinite(s)].tolist()}")
    return _adaptive._reaches(s, threshold, rel_tol).to(dtype=scores.dtype, device=scores.device)


def group_records(binary: torch.Tensor, index) -> list:
    """One {"uid", "n", "n_pass"} per group (index = uid per row), in order of first appearance."""
    if len(index) != binary.shape[0]:
        raise ValueError(f"index has {len(index)} entries but there are {binary.shape[0]} rows")
    rows_of = {}
    for i, g in enumerate(index):
        rows_of.setdefault(g, []).append(i)
    b = binary.detach().cpu()
    return [{"uid": g, "n": len(rows), "n_pass": int(b[rows].sum().item())} for g, rows in rows_of.items()]


def compute_fixed_threshold_advantage(
    token_level_rewards,
    response_mask,
    index,
    threshold: float,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    """(advantages, returns, b): vanilla GRPO on b = 1[sequence reward reaches ``threshold``]."""
    binary = fixed_binarize(token_level_rewards.detach().sum(dim=-1), threshold)
    advantages, returns = _adaptive._grpo_on_binary(
        binary, token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo, config
    )
    return advantages, returns, binary


def step_metrics(records, threshold: float) -> dict:
    """Scalar metrics over one step's group records (see the module docstring)."""
    n_groups = len(records)
    n_rollouts = sum(r["n"] for r in records)
    m = {
        "threshold": threshold,
        "pass_rate": sum(r["n_pass"] for r in records) / n_rollouts,
        "frac_groups_none_pass": sum(r["n_pass"] == 0 for r in records) / n_groups,
        "frac_groups_all_pass": sum(r["n_pass"] == r["n"] for r in records) / n_groups,
        "frac_groups_mixed": sum(0 < r["n_pass"] < r["n"] for r in records) / n_groups,
    }
    return {METRIC_PREFIX + k: v for k, v in m.items()}


class _Runtime:
    """Per-process state: this step's metrics, and the groups dynamic sampling evaluated that the
    estimator has not seen yet."""

    def __init__(self):
        self.pending_metrics = None  # metrics of the last estimator call until compute_data_metrics picks them up
        self.reset_sampled()

    def reset_sampled(self) -> None:
        self.ds_step = None  # step whose groups evaluate_groups binarized, until the estimator uses them
        self.ds_groups: dict = {}  # uid -> {"uid", "n", "n_pass"}
        self.ds_records: list = []  # records of every group evaluated at ds_step, in generation order

    def evaluate_groups(self, data, scores: torch.Tensor, step: int, config) -> dict:
        """Binarize freshly generated groups for dynamic sampling; returns {uid: informative} in generation order
        (informative = some but not all of the group's rollouts get b = 1)."""
        threshold = threshold_from(config)
        if self.ds_step is None:
            self.ds_step = step
        elif self.ds_step != step:
            raise RuntimeError(f"{NAME}: groups evaluated at step {self.ds_step} were never used for advantages")
        uids = data.non_tensor_batch["uid"]
        again = set(uids) & self.ds_groups.keys()
        if again:
            raise RuntimeError(f"{NAME}: groups evaluated twice at step {step}: {sorted(map(str, again))[:3]}")
        records = group_records(fixed_binarize(scores, threshold), uids)
        for rec in records:
            self.ds_groups[rec["uid"]] = rec
        self.ds_records.extend(records)
        return {rec["uid"]: 0 < rec["n_pass"] < rec["n"] for rec in records}

    def sampled_metrics(self, records: list, threshold: float) -> dict:
        """Metrics of every group evaluated this step, after checking the training batch's groups (``records``)
        against them; forgets the evaluated groups."""
        for rec in records:
            cached = self.ds_groups.get(rec["uid"])
            if cached is None:
                raise RuntimeError(f"{NAME}: group {rec['uid']} was not evaluated when it was generated")
            if (rec["n"], rec["n_pass"]) != (cached["n"], cached["n_pass"]):
                raise RuntimeError(
                    f"{NAME}: group {rec['uid']}: {rec['n_pass']} of {rec['n']} rollouts reach the threshold now, "
                    f"{cached['n_pass']} of {cached['n']} did when it was generated (the rewards changed in between)"
                )
        evaluated = self.ds_records
        metrics = step_metrics(evaluated, threshold)
        metrics[METRIC_PREFIX + "num_groups_evaluated"] = len(evaluated)
        metrics[METRIC_PREFIX + "frac_groups_kept"] = len(records) / len(evaluated)
        self.reset_sampled()
        return metrics


def _install():
    existing = getattr(ray_trainer, _RUNTIME_ATTR, None)
    if existing is not None:
        return existing
    rt = _Runtime()

    @core_algos.register_adv_est(NAME)
    def _estimator(token_level_rewards, response_mask, index, config=None, **kwargs):
        threshold = threshold_from(config)
        norm_adv_by_std_in_grpo = bool(_base._config_get(config, "norm_adv_by_std_in_grpo", True))
        advantages, returns, binary = compute_fixed_threshold_advantage(
            token_level_rewards,
            response_mask,
            index,
            threshold,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )
        records = group_records(binary, index)
        if rt.ds_step is not None:  # dynamic sampling evaluated this step's groups
            rt.pending_metrics = rt.sampled_metrics(records, threshold)
        else:
            rt.pending_metrics = step_metrics(records, threshold)
        return advantages, returns

    orig_compute_data_metrics = ray_trainer.compute_data_metrics

    def compute_data_metrics(*args, **kwargs):
        metrics = orig_compute_data_metrics(*args, **kwargs)
        if rt.pending_metrics is not None:
            metrics = {**metrics, **rt.pending_metrics}
            rt.pending_metrics = None
        return metrics

    trainer_cls = ray_trainer.RayPPOTrainer
    orig_save = trainer_cls._save_checkpoint
    orig_load = trainer_cls._load_checkpoint

    def _active(trainer) -> bool:
        estimator = trainer.config.algorithm.adv_estimator
        return str(getattr(estimator, "value", estimator)) == NAME

    def _save_checkpoint(self, *args, **kwargs):
        # Written before verl's own save, so a checkpoint verl marks complete always has it.
        path = os.path.join(
            os.path.abspath(self.config.trainer.default_local_dir), f"global_step_{self.global_steps}", STATE_FILE
        )
        active = _active(self)
        if active:
            state = {
                "version": STATE_VERSION,
                "estimator": NAME,
                "step": self.global_steps,
                "threshold": threshold_from(self.config.algorithm),
            }
            _adaptive._write_json(path, state)
        result = orig_save(self, *args, **kwargs)
        if active and not os.path.exists(path):
            _adaptive._write_json(path, state)
        return result

    def _load_checkpoint(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        rt.pending_metrics = None
        rt.reset_sampled()
        if _active(self):
            threshold = threshold_from(self.config.algorithm)
            if self.global_steps > 0:
                path = os.path.join(_adaptive._resume_folder(self), STATE_FILE)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{NAME}: resuming from step {self.global_steps} but {path} is missing")
                with open(path) as f:
                    state = json.load(f)
                if state.get("version") != STATE_VERSION or state.get("estimator") != NAME:
                    found = f"version {state.get('version')}, estimator {state.get('estimator')}"
                    raise ValueError(f"{path}: not an {NAME} state file ({found})")
                if state.get("step") != self.global_steps:
                    found = state.get("step")
                    raise RuntimeError(f"{NAME}: {path} holds step {found}, resuming step {self.global_steps}")
                if state.get("threshold") != threshold:
                    raise ValueError(
                        f"{path}: written with {KEY}={state.get('threshold')!r}, but this run has {KEY}={threshold!r}"
                    )
            print(f"[{NAME}] {KEY}={threshold} at step {self.global_steps}", flush=True)
        return result

    ray_trainer.compute_data_metrics = compute_data_metrics
    trainer_cls._save_checkpoint = _save_checkpoint
    trainer_cls._load_checkpoint = _load_checkpoint
    setattr(ray_trainer, _RUNTIME_ATTR, rt)
    if not hasattr(ray_trainer, _EVALUATORS_ATTR):
        setattr(ray_trainer, _EVALUATORS_ATTR, {})
    getattr(ray_trainer, _EVALUATORS_ATTR)[NAME] = rt.evaluate_groups
    return rt


RUNTIME = _install()
