"""Adaptive-threshold quantized GRPO: ``algorithm.adv_estimator=grpo_adaptive_threshold``.

Like grpo_quantized (scripts/adv_estimators.py), but each rollout's reward is binarized
against a per-problem threshold that is carried across steps instead of against the mean of
its own group:

    first group of problem p:   tau_p = mean of that group
    every group of problem p:   r_i reaches t  iff  r_i >= t (up to float rounding) and r_i > 0
                                if at least a fraction x of the group reaches tau_p:
                                    tau_p = max(tau_p, mean_g(r))
                                b_i = 1 if r_i reaches tau_p (as just updated), else 0
                                A_i = GRPO(b)_i = (b_i - mean_g(b)) / (std_g(b) + eps)

x = algorithm.adaptive_threshold_frac (required, 0 < x <= 1; pass it as
+algorithm.adaptive_threshold_frac=<x>; the launcher's ADAPTIVE_FRAC, default 0.75). It is the
only setting: any other algorithm.adaptive_threshold_* key is refused. The threshold only goes
up. Whether a group raises it is decided against the threshold the group found; a group that
raises it is then binarized against the raised threshold. A group in which everybody reaches
tau_p is the special case where the whole group, not only a fraction x, does: it is binarized
against max(tau_p, its mean), i.e. split at its mean (flat only if its scores are all equal, up
to float rounding). A group in which nobody reaches tau_p gets b = 0 everywhere, so A = 0, and
tau_p stays where it is (no fallback). On a problem's first group tau_p is that group's mean
(the raise changes nothing), so b = 1[r >= mean]: the grpo_quantized split, except that a score
equal to the mean counts as reaching it. A score of 0 never reaches a threshold: a problem
whose first group scored all zeros has tau_p = 0, and its later groups get b = 1[r > 0]
(instead of all passing) until a fraction x of one group scores above 0 (that group is split at
its mean, the new tau_p).
r_i is the sequence reward (token_level_rewards summed over the response); epsilon and
norm_adv_by_std_in_grpo are vanilla GRPO's.

Problem identity is reward_model["ground_truth"] (the problem id, e.g. frontiersmith_199), not
uid, which verl draws fresh for every prompt of every batch. verl's registered estimators only
receive uid, so this module replaces ray_trainer.compute_advantage with a wrapper that handles
grpo_adaptive_threshold itself and passes every other estimator to the original. Groups (uid)
are processed in order of first appearance in the batch.

Dynamic sampling (scripts/dynamic_sampling.py) has to know which groups get nonzero advantages
while it is still generating. It calls evaluate_groups (registered in
ray_trainer._frontiersmith_group_evaluators) on every group it generates: the group is binarized
and tau_p updated right away, exactly once per group (also for groups it then drops), in
generation order, and the threshold the group was binarized against is remembered by uid.
compute_advantage then recomputes b of the kept groups from those thresholds (balance_batch may
have reordered the rows) without touching tau again; its metrics and dump cover every group
evaluated at that step. Groups generated during a judge outage (rollouts scored 0) update tau
like any other, as they do without dynamic sampling.

State ({problem id: tau}) lives in the TaskRunner process, where verl computes advantages, and
is checkpointed with the model: written to <ckpt>/global_step_N/adaptive_threshold_state.json
right before verl saves step N, and read back when verl resumes from step N. Resuming from a
checkpoint without that file is an error, not a silent reset of every threshold; so is resuming
with a different adaptive_threshold_frac than the file was written with, or from a version 1
file (written by an earlier version of this plugin: the original rule, which binarized every
group against the threshold from before its own raise, or the short-lived switch between that
rule and this one; the thresholds evolve identically under all of them, but the runs trained
on different b).

Logged every step (wandb and console, through compute_data_metrics), prefix adaptive_threshold/:
    tau_mean, tau_median, tau_min, tau_max  thresholds of all problems seen so far, after this step
    num_problems                            problems with a threshold
    batch_tau_mean                          mean threshold this step's groups found (before their update)
    batch_score_minus_tau                   mean over groups of (group mean score - threshold found)
    pass_rate                               fraction of rollouts with b = 1
    pass_rate_pre                           fraction of rollouts reaching the threshold their group found
    frac_groups_first_visit / _none_pass / _all_pass / _mixed
                                            groups by outcome of b (_mixed = the ones with nonzero advantages)
    frac_groups_all_pass_pre                groups in which everybody reached the threshold the group found
    frac_groups_raised_first                groups whose threshold went up (so b is against the raised one),
                                            over all groups: frac_revisits_raised x the share of revisits
    frac_revisits_raised, tau_raise_mean    among groups of already-seen problems: the fraction
                                            whose threshold went up, and the mean increase
    num_groups_evaluated, frac_groups_kept  with dynamic sampling only: groups evaluated this step
                                            (the other metrics cover all of them) and the fraction trained on
Per-group detail and every threshold are also written to
outputs/adaptive_threshold_<experiment_name>/<step>.json (next to trainer.rollout_data_dir); a
group's tau_bin (the threshold its b is computed against) always equals its tau_after.

Loaded by scripts/main_ppo_adaptive.py and scripts/dynamic_sampling.py (it also loads
scripts/adv_estimators.py, so grpo_quantized and drgrpo_scaled stay available).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys

import numpy as np
import torch

from verl.trainer.ppo import core_algos, ray_trainer

NAME = "grpo_adaptive_threshold"
FRAC_KEY = "adaptive_threshold_frac"
STATE_FILE = "adaptive_threshold_state.json"
STATE_VERSION = 2  # version 1: an earlier version of this plugin (same thresholds, different b)
METRIC_PREFIX = "adaptive_threshold/"
REL_TOL = 1e-6
_RUNTIME_ATTR = "_frontiersmith_adaptive_threshold_runtime"  # on ray_trainer: makes installing idempotent
_EVALUATORS_ATTR = "_frontiersmith_group_evaluators"  # on ray_trainer: {estimator: evaluate_groups}


def _load_base_estimators():
    """scripts/adv_estimators.py, imported once per process (it registers grpo_quantized, drgrpo_scaled)."""
    name = "frontiersmith_adv_estimators_base"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_estimators.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_base = _load_base_estimators()


def _reaches(r: torch.Tensor, threshold: float, rel_tol: float) -> torch.Tensor:
    """Which scores reach ``threshold`` (up to float rounding); a zero score never does."""
    return (r >= threshold - rel_tol * max(1.0, abs(threshold))) & (r > 0)


def adaptive_binarize(scores: torch.Tensor, group_ids, problem_ids, tau: dict, frac: float, rel_tol: float = REL_TOL):
    """b_i = 1.0 if scores[i] reaches its problem's threshold as its group updates it, else 0.0; updates ``tau``
    in place (see the module docstring).

    scores: (bs,) sequence rewards; group_ids: GRPO group (uid) per row; problem_ids: problem key
    per row (one problem per group); tau: {problem key: threshold}, modified in place.
    Returns (b with the dtype/device of scores, one record per group in processing order).
    """
    n_rows = scores.shape[0]
    if len(group_ids) != n_rows or len(problem_ids) != n_rows:
        raise ValueError(f"{n_rows} scores but {len(group_ids)} group ids and {len(problem_ids)} problem ids")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"{FRAC_KEY} must be in (0, 1], got {frac}")
    s = scores.detach().to(torch.float64).cpu()
    if not torch.isfinite(s).all():
        raise ValueError(f"non-finite rewards: {s[~torch.isfinite(s)].tolist()}")
    rows_of = {}
    for i, g in enumerate(group_ids):
        rows_of.setdefault(g, []).append(i)
    b = torch.zeros_like(s)
    records = []
    for g, rows in rows_of.items():
        keys = {problem_ids[i] for i in rows}
        if len(keys) != 1:
            raise ValueError(f"group {g} mixes problems {sorted(map(str, keys))}")
        key = keys.pop()
        r = s[rows]
        n = len(rows)
        mean = r.mean().item()
        first_visit = key not in tau
        if first_visit:
            tau[key] = mean
        tau_used = float(tau[key])
        reached = _reaches(r, tau_used, rel_tol)
        n_pass_pre = int(reached.sum().item())
        triggered = n_pass_pre >= math.ceil(frac * n - 1e-9)
        tau_after = max(tau_used, mean) if triggered else tau_used
        passed = _reaches(r, tau_after, rel_tol) if tau_after != tau_used else reached
        b[rows] = passed.to(torch.float64)
        tau[key] = tau_after
        records.append(
            {
                "uid": g,
                "problem": key,
                "n": n,
                "n_pass": int(passed.sum().item()),
                "n_pass_pre": n_pass_pre,
                "mean": mean,
                "tau_used": tau_used,
                "tau_bin": tau_after,
                "tau_after": tau_after,
                "first_visit": first_visit,
                "triggered": triggered,
            }
        )
    return b.to(dtype=scores.dtype, device=scores.device), records


def _grpo_on_binary(
    binary, token_level_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True, config=None
):
    """Vanilla GRPO (advantages, returns) of the 0/1 rewards ``binary``, put on each row's last valid token."""
    quantized_rewards = _base._put_on_last_valid_token(binary, response_mask, like=token_level_rewards.detach())
    return core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=quantized_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )


def compute_adaptive_threshold_advantage(
    token_level_rewards,
    response_mask,
    index,
    problem_ids,
    tau: dict,
    frac: float,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    """(advantages, returns, b, records); ``tau`` is updated in place."""
    binary, records = adaptive_binarize(token_level_rewards.detach().sum(dim=-1), index, problem_ids, tau, frac)
    advantages, returns = _grpo_on_binary(
        binary, token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo, config
    )
    return advantages, returns, binary, records


def step_metrics(records, *args) -> dict:
    """Scalar metrics over one step's group records (see the module docstring): step_metrics(records, tau), tau
    = the thresholds after the step. The older step_metrics(records, binary, tau) still works (binary is unused:
    pass_rate comes from the records)."""
    if len(args) not in (1, 2):
        raise TypeError(f"step_metrics(records, tau) takes 2 positional arguments, {1 + len(args)} were given")
    tau = args[-1]
    taus = np.array(list(tau.values()), dtype=np.float64)
    n_groups = len(records)
    n_rollouts = sum(r["n"] for r in records)
    revisits = [r for r in records if not r["first_visit"]]
    m = {
        "tau_mean": float(taus.mean()),
        "tau_median": float(np.median(taus)),
        "tau_min": float(taus.min()),
        "tau_max": float(taus.max()),
        "num_problems": len(tau),
        "batch_tau_mean": float(np.mean([r["tau_used"] for r in records])),
        "batch_score_minus_tau": float(np.mean([r["mean"] - r["tau_used"] for r in records])),
        "pass_rate": sum(r["n_pass"] for r in records) / n_rollouts,
        "pass_rate_pre": sum(r["n_pass_pre"] for r in records) / n_rollouts,
        "frac_groups_first_visit": sum(r["first_visit"] for r in records) / n_groups,
        "frac_groups_none_pass": sum(r["n_pass"] == 0 for r in records) / n_groups,
        "frac_groups_all_pass": sum(r["n_pass"] == r["n"] for r in records) / n_groups,
        "frac_groups_mixed": sum(0 < r["n_pass"] < r["n"] for r in records) / n_groups,
        "frac_groups_all_pass_pre": sum(r["n_pass_pre"] == r["n"] for r in records) / n_groups,
        "frac_groups_raised_first": sum(r["tau_bin"] > r["tau_used"] for r in records) / n_groups,
        "frac_revisits_raised": (
            sum(r["tau_after"] > r["tau_used"] for r in revisits) / len(revisits) if revisits else 0.0
        ),
        "tau_raise_mean": float(np.mean([r["tau_after"] - r["tau_used"] for r in revisits])) if revisits else 0.0,
    }
    return {METRIC_PREFIX + k: v for k, v in m.items()}


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


class _Runtime:
    """Per-process state: the thresholds, the trainer they belong to, this step's metrics, and the
    groups dynamic sampling evaluated that compute_advantage has not used yet."""

    def __init__(self):
        self.tau: dict = {}
        self.step = 0  # last global step whose groups updated tau
        self.trainer = None
        self.pending_metrics = None  # (global step, metrics) until compute_data_metrics picks it up
        self.reset_sampled()

    def reset_sampled(self) -> None:
        self.ds_step = None  # step whose groups evaluate_groups binarized, until compute_advantage uses them
        self.ds_groups: dict = {}  # uid -> {"tau_bin", "n", "n_pass", "problem"}
        self.ds_records: list = []  # records of every group evaluated at ds_step, in generation order
        self.ds_round = 0  # evaluate_groups calls at ds_step

    def state_dict(self, config) -> dict:
        return {
            "version": STATE_VERSION,
            "estimator": NAME,
            "step": self.step,
            "tau": self.tau,
            "frac": _frac_from(config),
        }

    def load_state_dict(self, d: dict, path: str, config) -> None:
        if d.get("version") not in (1, STATE_VERSION) or d.get("estimator") != NAME:
            found = f"version {d.get('version')}, estimator {d.get('estimator')}"
            raise ValueError(f"{path}: not an {NAME} state file ({found})")
        if d["version"] != STATE_VERSION:
            raise ValueError(
                f"{path}: a version {d['version']} state file, written by an earlier version of this plugin (the "
                "original rule, b against the threshold from before the group's raise, or the short-lived switch "
                "between rules); this code always binarizes a group that raises the threshold against the raised one "
                f"(version {STATE_VERSION}) and does not resume a run started before that"
            )
        frac = _frac_from(config)
        if "frac" not in d:
            raise ValueError(f"{path}: has no frac (version {STATE_VERSION} state files always store it)")
        if d["frac"] != frac:
            raise ValueError(f"{path}: written with frac={d['frac']!r}, but this run has frac={frac!r}")
        self.tau = {k: float(v) for k, v in d["tau"].items()}
        self.step = int(d["step"])

    def evaluate_groups(self, data, scores: torch.Tensor, step: int, config) -> dict:
        """Binarize freshly generated groups now, for dynamic sampling (see the module docstring).

        data: the generation input (uid and reward_model per row); scores: the sequence rewards of
        its rows, as compute_advantage will see them. Updates tau once per group and keeps the
        threshold each group was binarized against until compute_advantage. Returns
        {uid: informative} in generation order; informative = some but not all of the group's
        rollouts get b = 1 (nonzero advantages).
        """
        if self.trainer is None:
            raise RuntimeError(f"{NAME}: no trainer registered (RayPPOTrainer._load_checkpoint did not run)")
        frac = _frac_from(config)
        if step != self.trainer.global_steps:
            raise RuntimeError(f"{NAME}: groups evaluated for step {step}, trainer at step {self.trainer.global_steps}")
        if self.ds_step is None:
            if step <= self.step:
                raise RuntimeError(f"{NAME}: thresholds already updated at step {self.step}, evaluating step {step}")
            self.ds_step = step
        elif self.ds_step != step:
            raise RuntimeError(f"{NAME}: groups evaluated at step {self.ds_step} were never used for advantages")
        uids = data.non_tensor_batch["uid"]
        again = set(uids) & self.ds_groups.keys()
        if again:
            raise RuntimeError(f"{NAME}: groups evaluated twice at step {step}: {sorted(map(str, again))[:3]}")
        _, records = adaptive_binarize(scores, uids, _problem_ids(data), self.tau, frac)
        self.ds_round += 1
        for rec in records:
            rec["round"] = self.ds_round
            self.ds_groups[rec["uid"]] = {k: rec[k] for k in ("tau_bin", "n", "n_pass", "problem")}
        self.ds_records.extend(records)
        self.step = step  # tau holds this step's updates now (the checkpoint guard relies on it)
        return {rec["uid"]: 0 < rec["n_pass"] < rec["n"] for rec in records}

    def binarize_sampled(self, data, scores: torch.Tensor, step: int):
        """b of a batch of groups evaluate_groups saw at ``step``, against the thresholds they were binarized with.

        Leaves tau alone, marks every record of the step kept or not, and forgets the evaluated groups.
        Returns (b with the dtype/device of scores, the records of every group evaluated at ``step``).
        """
        if self.ds_step != step:
            raise RuntimeError(f"{NAME}: groups were evaluated at step {self.ds_step}, advantages asked at step {step}")
        s = scores.detach().to(torch.float64).cpu()
        problems = _problem_ids(data)
        rows_of = {}
        for i, g in enumerate(data.non_tensor_batch["uid"]):
            rows_of.setdefault(g, []).append(i)
        b = torch.zeros_like(s)
        for g, rows in rows_of.items():
            cached = self.ds_groups.get(g)
            if cached is None:
                raise RuntimeError(f"{NAME}: group {g} was not evaluated when it was generated")
            if len(rows) != cached["n"] or {problems[i] for i in rows} != {cached["problem"]}:
                raise RuntimeError(
                    f"{NAME}: group {g} has {len(rows)} rows of {sorted({problems[i] for i in rows})}, "
                    f"but was evaluated with {cached['n']} rows of {cached['problem']}"
                )
            passed = _reaches(s[rows], cached["tau_bin"], REL_TOL)
            n_pass = int(passed.sum().item())
            if n_pass != cached["n_pass"]:
                raise RuntimeError(
                    f"{NAME}: group {g}: {n_pass} rollouts reach its threshold now, {cached['n_pass']} did when it "
                    "was generated (the rewards changed in between)"
                )
            b[rows] = passed.to(torch.float64)
        records = self.ds_records
        for rec in records:
            rec["kept"] = rec["uid"] in rows_of
        self.reset_sampled()
        return b.to(dtype=scores.dtype, device=scores.device), records


def _frac_from(config) -> float:
    """algorithm.adaptive_threshold_frac, checked; refuses any other algorithm.adaptive_threshold_* key (a stray or
    leftover override, e.g. of a removed setting, would otherwise be ignored without a word)."""
    keys = config.keys() if hasattr(config, "keys") else config if hasattr(config, "__iter__") else ()
    unknown = sorted(k for k in keys if isinstance(k, str) and k.startswith("adaptive_threshold_") and k != FRAC_KEY)
    if unknown:
        raise ValueError(
            f"{NAME}: unknown setting(s) {', '.join('algorithm.' + k for k in unknown)}; algorithm.{FRAC_KEY} is the "
            "only one (a group that raises its threshold is always binarized against the raised one)"
        )
    frac = _base._config_get(config, FRAC_KEY, None)
    if frac is None:
        raise ValueError(f"{NAME} needs algorithm.{FRAC_KEY} (pass +algorithm.{FRAC_KEY}=<x>, 0 < x <= 1)")
    frac = float(frac)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"algorithm.{FRAC_KEY} must be in (0, 1], got {frac}")
    return frac


def _problem_ids(data) -> list:
    reward_model = data.non_tensor_batch.get("reward_model")
    if reward_model is None:
        raise KeyError(f"{NAME}: the batch has no reward_model column to take problem ids from")
    ids = []
    for i, rm in enumerate(reward_model):
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        if not isinstance(gt, str) or not gt:
            raise ValueError(f"{NAME}: row {i} has no string reward_model['ground_truth'] (got {rm!r:.200})")
        ids.append(gt)
    return ids


def _resume_folder(trainer) -> str:
    """The global_step_N folder verl's _load_checkpoint just resumed from (same rules as verl)."""
    tc = trainer.config.trainer
    if tc.resume_mode == "resume_path":
        return os.path.abspath(tc.resume_from_path)
    return os.path.join(os.path.abspath(tc.default_local_dir), f"global_step_{trainer.global_steps}")


def _install():
    existing = getattr(ray_trainer, _RUNTIME_ATTR, None)
    if existing is not None:
        return existing
    rt = _Runtime()

    def _registry_stub(*args, **kwargs):
        raise RuntimeError(f"{NAME} runs through the patched ray_trainer.compute_advantage (it needs problem ids)")

    core_algos.register_adv_est(NAME)(_registry_stub)

    orig_compute_advantage = ray_trainer.compute_advantage
    signature = inspect.signature(orig_compute_advantage)

    def compute_advantage(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if getattr(a["adv_estimator"], "value", a["adv_estimator"]) != NAME:
            return orig_compute_advantage(*args, **kwargs)
        trainer = rt.trainer
        if trainer is None:
            raise RuntimeError(f"{NAME}: no trainer registered (RayPPOTrainer._load_checkpoint did not run)")
        step = int(trainer.global_steps)
        sampled = rt.ds_step is not None  # dynamic sampling already binarized this step's groups
        if not sampled and step <= rt.step:
            raise RuntimeError(f"{NAME}: thresholds already updated at step {rt.step}, asked again at step {step}")
        data = a["data"]
        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = ray_trainer.compute_response_mask(data)
        if sampled:
            token_level_rewards = data.batch["token_level_rewards"]
            binary, records = rt.binarize_sampled(data, token_level_rewards.detach().sum(dim=-1), step)
            advantages, returns = _grpo_on_binary(
                binary,
                token_level_rewards,
                data.batch["response_mask"],
                data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=a["norm_adv_by_std_in_grpo"],
                config=a["config"],
            )
            metrics = step_metrics(records, rt.tau)
            metrics[METRIC_PREFIX + "num_groups_evaluated"] = len(records)
            metrics[METRIC_PREFIX + "frac_groups_kept"] = sum(r["kept"] for r in records) / len(records)
        else:
            advantages, returns, _, records = compute_adaptive_threshold_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                problem_ids=_problem_ids(data),
                tau=rt.tau,
                frac=_frac_from(a["config"]),
                norm_adv_by_std_in_grpo=a["norm_adv_by_std_in_grpo"],
                config=a["config"],
            )
            rt.step = step
            metrics = step_metrics(records, rt.tau)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        rt.pending_metrics = (step, metrics)
        rollout_dir = trainer.config.trainer.get("rollout_data_dir", None)
        if rollout_dir:
            out_dir = os.path.join(
                os.path.dirname(os.path.abspath(rollout_dir)),
                f"adaptive_threshold_{trainer.config.trainer.experiment_name}",
            )
            _write_json(os.path.join(out_dir, f"{step}.json"), {"step": step, "groups": records, "tau": rt.tau})
        return data

    orig_compute_data_metrics = ray_trainer.compute_data_metrics

    def compute_data_metrics(*args, **kwargs):
        metrics = orig_compute_data_metrics(*args, **kwargs)
        pending = rt.pending_metrics
        if pending is not None and rt.trainer is not None and pending[0] == rt.trainer.global_steps:
            metrics = {**metrics, **pending[1]}
            rt.pending_metrics = None
        return metrics

    trainer_cls = ray_trainer.RayPPOTrainer
    orig_save = trainer_cls._save_checkpoint
    orig_load = trainer_cls._load_checkpoint

    def _save_checkpoint(self, *args, **kwargs):
        # Written before verl's own save, so a checkpoint verl marks complete always has it.
        path = os.path.join(
            os.path.abspath(self.config.trainer.default_local_dir), f"global_step_{self.global_steps}", STATE_FILE
        )
        active = rt.trainer is self and self.config.algorithm.adv_estimator == NAME
        if active:
            if rt.ds_step is not None:
                raise RuntimeError(
                    f"{NAME}: saving step {self.global_steps} while groups evaluated at step {rt.ds_step} "
                    "were never used for advantages"
                )
            if rt.step != self.global_steps:
                raise RuntimeError(f"{NAME}: saving step {self.global_steps} but thresholds are from step {rt.step}")
            _write_json(path, rt.state_dict(self.config.algorithm))
        result = orig_save(self, *args, **kwargs)
        if active and not os.path.exists(path):
            _write_json(path, rt.state_dict(self.config.algorithm))
        return result

    def _load_checkpoint(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        rt.trainer = self
        rt.tau, rt.step, rt.pending_metrics = {}, 0, None
        rt.reset_sampled()
        config = self.config.algorithm
        if config.adv_estimator == NAME:
            if self.global_steps > 0:
                path = os.path.join(_resume_folder(self), STATE_FILE)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{NAME}: resuming from step {self.global_steps} but {path} is missing")
                with open(path) as f:
                    rt.load_state_dict(json.load(f), path, config)
                if rt.step != self.global_steps:
                    raise RuntimeError(f"{NAME}: {path} holds step {rt.step}, resuming step {self.global_steps}")
            taus = list(rt.tau.values())
            print(
                f"[{NAME}] thresholds at step {self.global_steps}: {len(taus)} problems"
                + (f", mean {np.mean(taus):.3f}, min {min(taus):.3f}, max {max(taus):.3f}" if taus else "")
                + f"; {FRAC_KEY}={_frac_from(config)}",
                flush=True,
            )
        return result

    ray_trainer.compute_advantage = compute_advantage
    ray_trainer.compute_data_metrics = compute_data_metrics
    trainer_cls._save_checkpoint = _save_checkpoint
    trainer_cls._load_checkpoint = _load_checkpoint
    setattr(ray_trainer, _RUNTIME_ATTR, rt)
    if not hasattr(ray_trainer, _EVALUATORS_ATTR):
        setattr(ray_trainer, _EVALUATORS_ATTR, {})
    getattr(ray_trainer, _EVALUATORS_ATTR)[NAME] = rt.evaluate_groups
    return rt


RUNTIME = _install()
