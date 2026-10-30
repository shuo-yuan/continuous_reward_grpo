"""Dynamic sampling (DAPO) for GRPO-style estimators: ``+algorithm.dynamic_sampling=True``.

A GRPO group (the n rollouts of one prompt) whose rollouts all end up with the same reward gets zero
advantages and only contributes the KL term. With dynamic sampling every training step keeps
generating until it has enough groups with nonzero advantages ("informative" groups):

    1. fit() pulls its dataloader batch (B = data.train_batch_size prompts) and generates n rollouts
       per prompt, as usual (round 1).
    2. Each group is classified the way the configured estimator will treat it:
           grpo, drgrpo_scaled       informative iff its scores are not all equal
           grpo_quantized            iff binarizing at the group mean (scripts/adv_estimators.py) gives mixed 0/1
           grpo_adaptive_threshold   iff some but not all rollouts get b = 1 against the per-problem threshold
                                     (evaluate_groups of scripts/adv_adaptive_threshold.py, which updates the
                                     thresholds once for every generated group, kept or not, and binarizes the
                                     group against its threshold as that update left it)
           grpo_fixed_threshold      iff some but not all rollouts get b = 1 against the fixed threshold
                                     (evaluate_groups of scripts/adv_fixed_threshold.py)
    3. While fewer than T groups are informative, fewer than M dataloader batches were used this step
       and no judge outage was seen, the next round pulls more batches from the same training
       dataloader stream (as in DAPO they are consumed, not resampled), builds them exactly as fit()
       builds its own and generates them all in one call.
    4. The step trains on exactly B groups: the first B informative groups in generation order and,
       if there are fewer, flat groups in generation order to fill up (they keep the batch shape;
       their gradient is the KL term only). Surplus groups are discarded.
fit() then runs unchanged on the B*n assembled rows (log-probs, advantages, update, dumps, metrics).

Config (flat under algorithm, pass as +algorithm.<key>=<value>; checked when fit() starts):
    dynamic_sampling                    false (default): nothing changes
    dynamic_sampling_max_gen_batches    M, dataloader batches per step including fit()'s own (default 6)
    dynamic_sampling_target_groups      T, 1 <= T <= B (default B)
    dynamic_sampling_batches_per_round  batches per extra round, or "auto" (default): enough to reach T at the
                                        informative rate seen so far this step (taken as at least 0.1)
    dynamic_sampling_outage_frac        stop pulling once more than this fraction of a round's rollouts got
                                        judge_status -1 or -2 (timed out, judge unreachable; default 0.5):
                                        their 0 scores make groups look flat
Refused when on: other estimators, grpo_fixed_threshold without a valid algorithm.fixed_threshold,
data.gen_batch_size != data.train_batch_size, rollout.skip_rollout, algorithm.use_kl_in_reward, a reward model
(scores must be the agent loop's rm_scores), rollout.n < 2.

Wiring (RayPPOTrainer.fit is patched): trainer.train_dataloader is wrapped in SharedLoader, one endless
iterator over the real loader's epochs that fit() and the extra rounds both pull from, so data.pt
checkpoints stay exact; fit() therefore stops only at trainer.total_training_steps (keep
trainer.total_epochs high). _get_gen_batch is patched to capture fit()'s prompt batch, and the rollout
manager's generate_sequences runs the round loop, which rewrites that prompt batch in place to the kept
groups (the generation output carries no uid: groups are matched through the generation input, and each
round checks that output row i is the rollout of input row i through the raw_prompt the agent loop copies
into its output). Validation generation passes through untouched. The number of dataloader batches consumed
is saved with each checkpoint in <ckpt>/global_step_N/dynamic_sampling_state.json (it only feeds metrics).

Logged every step, prefix dynamic_sampling/:
    rounds, batches_used, prompts_generated, rollouts_generated   this step's generation
    informative_groups_round1, frac_informative_round1           fit()'s own batch
    informative_groups_total, frac_informative_all               every generated group
    kept_informative, kept_flat, dropped_informative, dropped_flat
    cap_hit                              1 if pulling stopped at M batches
    outage_stop                          1 if a round had more than outage_frac judge failures (pulling stopped
                                         there, also when that round reached T)
    judge_outage_frac_all                fraction of the generated rollouts with judge_status -1/-2 (logged only
                                         when the rollouts carry judge_status)
    score_mean_round1, score_mean_all, score_mean_kept
    response_length_mean_all, clip_ratio_all, generated_tokens_total   every generated rollout
    gen_s_round1, gen_s_extra            generation wall time of round 1 / of the extra rounds
    batches_consumed_total, epoch        dataloader batches consumed by the run, and in epochs
Every generated group (round, problem, scores, response lengths, judge statuses, informative / kept /
filled; no texts) is written to outputs/dynamic_sampling_<experiment_name>/<step>.json, next to
trainer.rollout_data_dir.

Caveats:
  * critic/score/*, response_length/*, train-aux/* and the rollout dumps describe the trained (filtered)
    batch; compare with runs without dynamic sampling on dynamic_sampling/score_mean_round1.
    timing_per_token_ms/gen and perf/throughput are skewed (the tokens of dropped groups are not
    counted); timing_s/gen is the true wall time of all rounds.
  * training/epoch is wrong (fit() never leaves the epoch it started in); use dynamic_sampling/epoch.
  * every extra batch is n*B more judge submissions.
  * below outage_frac, a rollout the judge never scored (judge_status -1/-2) counts as a real 0: a group whose
    only differences are such failures is informative, and is preferred over flat groups. With
    grpo_adaptive_threshold, the groups of a round with a judge outage still update the thresholds (all-zero
    first visits set tau_p = 0), as they would without dynamic sampling. judge_outage_frac_all shows how
    often the judge failed.

Loaded by scripts/main_ppo_dynamic.py. It loads scripts/adv_adaptive_threshold.py (which loads
scripts/adv_estimators.py) and scripts/adv_fixed_threshold.py, so grpo_quantized, drgrpo_scaled,
grpo_adaptive_threshold and grpo_fixed_threshold are available with dynamic sampling on or off.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from typing import NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import ray_trainer

NAME = "dynamic_sampling"  # the switch (algorithm.dynamic_sampling), prefix of the other keys and of the metrics
STATE_FILE = "dynamic_sampling_state.json"
METRIC_PREFIX = "dynamic_sampling/"
ESTIMATORS = ("grpo", "grpo_quantized", "grpo_adaptive_threshold", "grpo_fixed_threshold", "drgrpo_scaled")
OUTAGE_STATUSES = (-1, -2)  # judge_status (verl/utils/reward_score/frontiercs.py): timed out, judge unreachable
_RUNTIME_ATTR = "_frontiersmith_dynamic_sampling_runtime"  # on ray_trainer: makes installing idempotent
_EVALUATORS_ATTR = "_frontiersmith_group_evaluators"  # on ray_trainer, filled by adv_*_threshold.py
_WRAPPED_ATTR = "_frontiersmith_dynamic_sampling"  # marks a wrapped generate_sequences


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


def _load_fixed_threshold():
    """scripts/adv_fixed_threshold.py, imported once per process."""
    name = "frontiersmith_adv_fixed_threshold"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_fixed_threshold.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_adaptive = _load_adaptive_threshold()
_fixed = _load_fixed_threshold()
_base = _adaptive._base


class Settings(NamedTuple):
    max_gen_batches: int
    target_groups: int
    batches_per_round: int | None  # None: "auto"
    outage_frac: float


def _estimator(config) -> str:
    estimator = config.algorithm.adv_estimator
    return str(getattr(estimator, "value", estimator))


def _positive_int(value, key: str, what: str = "a positive integer") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{NAME}: algorithm.{key} must be {what}, got {value!r}")
    return value


def settings_from(config) -> Settings | None:
    """The validated dynamic-sampling settings of a trainer config; None when dynamic sampling is off."""

    def get(key, default):
        return _base._config_get(config.algorithm, key, default)

    enabled = get(NAME, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{NAME}: algorithm.{NAME} must be true or false, got {enabled!r}")
    if not enabled:
        return None
    batch_size = int(config.data.train_batch_size)
    max_gen_batches = _positive_int(get("dynamic_sampling_max_gen_batches", 6), "dynamic_sampling_max_gen_batches")
    target_groups = _positive_int(get("dynamic_sampling_target_groups", batch_size), "dynamic_sampling_target_groups")
    if target_groups > batch_size:
        raise ValueError(
            f"{NAME}: algorithm.dynamic_sampling_target_groups must be at most data.train_batch_size "
            f"({batch_size}), got {target_groups}"
        )
    per_round = get("dynamic_sampling_batches_per_round", "auto")
    if per_round != "auto":
        per_round = _positive_int(per_round, "dynamic_sampling_batches_per_round", 'a positive integer or "auto"')
    outage_frac = get("dynamic_sampling_outage_frac", 0.5)
    if isinstance(outage_frac, bool) or not isinstance(outage_frac, (int, float)) or not 0 < outage_frac <= 1:
        raise ValueError(f"{NAME}: algorithm.dynamic_sampling_outage_frac must be in (0, 1], got {outage_frac!r}")
    estimator = _estimator(config)
    if estimator not in ESTIMATORS:
        raise ValueError(f"{NAME}: algorithm.adv_estimator must be one of {', '.join(ESTIMATORS)}, got {estimator}")
    if estimator == _fixed.NAME:
        _fixed.threshold_from(config.algorithm)  # also checked, DS on or off, by adv_fixed_threshold's _load_checkpoint
    gen_batch_size = config.data.get("gen_batch_size", None)
    if gen_batch_size is not None and gen_batch_size != batch_size:
        raise ValueError(
            f"{NAME}: data.gen_batch_size ({gen_batch_size}) must be unset or data.train_batch_size ({batch_size})"
        )
    rollout = config.actor_rollout_ref.rollout
    if rollout.get("skip_rollout", False):
        raise ValueError(f"{NAME}: does not work with actor_rollout_ref.rollout.skip_rollout")
    if rollout.n < 2:
        raise ValueError(f"{NAME}: needs actor_rollout_ref.rollout.n >= 2 (a group of one is never informative)")
    if get("use_kl_in_reward", False):
        raise ValueError(f"{NAME}: needs algorithm.use_kl_in_reward=False (groups are classified on the raw scores)")
    if OmegaConf.select(config, "reward.reward_model.enable", default=False):
        raise ValueError(
            f"{NAME}: needs reward.reward_model.enable=False (scores must come with the generation, as rm_scores)"
        )
    return Settings(max_gen_batches, target_groups, None if per_round == "auto" else per_round, float(outage_frac))


class SharedLoader:
    """trainer.train_dataloader under dynamic sampling: one endless stream of batches over the real
    loader's epochs, pulled by fit() (by iterating) and by the round loop (pull()).

    Only this wrapper calls iter() on the real loader, once per epoch: a second live iterator would
    start another epoch, and the loader's state_dict (saved as data.pt) would follow that one.
    len, state_dict, sampler, dataset, ... are the real loader's.
    """

    def __init__(self, loader):
        self.loader = loader
        self.batches_consumed_total = 0  # by the whole run; restored on resume
        self._it = None

    def pull(self) -> dict:
        """The next batch, starting the loader's next epoch when the current one is exhausted."""
        for _ in range(2):
            if self._it is None:
                self._it = iter(self.loader)
            try:
                batch = next(self._it)
            except StopIteration:
                self._it = None
                continue
            self.batches_consumed_total += 1
            return batch
        raise RuntimeError(f"{NAME}: the train dataloader yielded no batch in a fresh epoch")

    def __iter__(self):
        while True:
            yield self.pull()

    def __len__(self) -> int:
        return len(self.loader)

    def state_dict(self) -> dict:
        return self.loader.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self._it = None
        self.loader.load_state_dict(state_dict)

    def __getattr__(self, name):
        if name == "loader":  # only before __init__ ran (copy, unpickling)
            raise AttributeError(name)
        return getattr(self.loader, name)


class _Round(NamedTuple):
    index: int  # 1 = fit()'s own batch
    batches: int  # dataloader batches in this round
    prompt: DataProto  # prompt side, one row per group
    gen_in: DataProto  # generation input, n rows per group
    output: DataProto  # generation output (row-aligned with gen_in), its timing popped
    timing: dict
    gen_s: float
    scores: np.ndarray  # float64 per rollout
    lengths: np.ndarray  # response length per rollout
    informative: np.ndarray  # bool per group
    judge_status: np.ndarray | None  # per rollout, as generated (None: no such column)
    judge_bad_frac: float | None  # rollouts with an OUTAGE_STATUSES judge_status; None without that column


def classify(trainer, gen_in: DataProto, scores: torch.Tensor) -> np.ndarray:
    """Informative flag of each group of gen_in (runs of n rows), as the configured estimator will see it."""
    config = trainer.config
    n = config.actor_rollout_ref.rollout.n
    estimator = _estimator(config)
    if estimator in (_adaptive.NAME, _fixed.NAME):
        evaluate = getattr(ray_trainer, _EVALUATORS_ATTR, {}).get(estimator)
        if evaluate is None:
            raise RuntimeError(f"{NAME}: no group evaluator registered for {estimator}")
        verdict = evaluate(gen_in, scores, trainer.global_steps, config.algorithm)
        return np.array([verdict[uid] for uid in gen_in.non_tensor_batch["uid"][::n]], dtype=bool)
    if estimator == "grpo_quantized":
        passed = _base.quantize_by_group_mean(scores, gen_in.non_tensor_batch["uid"]).cpu().reshape(-1, n).sum(dim=1)
        return ((passed > 0) & (passed < n)).numpy()
    s = scores.detach().to(torch.float64).cpu().reshape(-1, n)
    return (s.amax(dim=1) != s.amin(dim=1)).numpy()


def _judge_failed(status) -> np.ndarray:
    """Per rollout: does its judge_status say the judge never scored it?"""
    return np.array([s in OUTAGE_STATUSES for s in status], dtype=bool)


def judge_outage_frac(output: DataProto) -> float | None:
    """Fraction of the rollouts whose judge_status says the judge never scored them; None without judge_status."""
    status = output.non_tensor_batch.get("judge_status")
    if status is None or len(status) == 0:
        return None
    return float(_judge_failed(status).mean())


def next_round_batches(settings: Settings, n_informative: int, n_prompts: int, batches_used: int, batch_size: int):
    """Dataloader batches for the next round, given what this step generated so far."""
    room = settings.max_gen_batches - batches_used
    if settings.batches_per_round is not None:
        return min(settings.batches_per_round, room)
    rate = max(n_informative / n_prompts, 0.1)
    deficit = settings.target_groups - n_informative
    return min(max(math.ceil(deficit / rate / batch_size), 1), room)


def select_groups(informative: list, batch_size: int) -> tuple:
    """Groups to train on: the first batch_size informative ones in generation order (rounds, then position),
    then flat ones in generation order up to batch_size. Returns (kept, filled) bool masks per round."""
    order = [(r, g) for r, flags in enumerate(informative) for g in range(len(flags))]
    chosen = [(r, g) for r, g in order if informative[r][g]][:batch_size]
    fill = [(r, g) for r, g in order if not informative[r][g]][: batch_size - len(chosen)]
    if len(chosen) + len(fill) != batch_size:
        raise RuntimeError(f"{NAME}: {len(order)} groups generated, fewer than the {batch_size} to train on")
    kept = [np.zeros(len(flags), dtype=bool) for flags in informative]
    filled = [np.zeros(len(flags), dtype=bool) for flags in informative]
    for r, g in chosen + fill:
        kept[r][g] = True
    for r, g in fill:
        filled[r][g] = True
    return kept, filled


def assemble(rounds: list, kept: list, n: int) -> tuple:
    """(prompt side, generation output) of the kept groups of all rounds, in generation order."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager  # loaded by init_workers already

    parts = [(rd, np.flatnonzero(mask)) for rd, mask in zip(rounds, kept, strict=True) if mask.any()]
    AgentLoopManager._align_outputs_for_concat([rd.output for rd, _ in parts])
    prompts, outputs, uids = [], [], []
    for rd, groups in parts:
        rows = (groups[:, None] * n + np.arange(n)).reshape(-1)
        prompts.append(rd.prompt.select_idxs(groups))
        outputs.append(rd.output.select_idxs(rows))
        uids.append(rd.gen_in.non_tensor_batch["uid"][rows])
    for part in prompts + outputs:
        part.meta_info = {}  # select_idxs shares the round's meta_info, which concat would compare
    prompt, output = DataProto.concat(prompts), DataProto.concat(outputs)
    if not np.array_equal(np.concatenate(uids), np.repeat(prompt.non_tensor_batch["uid"], n)):
        raise RuntimeError(f"{NAME}: the assembled rollouts are not aligned with their prompts")
    output.meta_info = {
        "reward_extra_keys": sorted({k for rd, _ in parts for k in rd.output.meta_info.get("reward_extra_keys", [])})
    }
    return prompt, output


def merge_timing(timings: list, rows: list) -> dict:
    """One agent-loop timing dict for several generate calls of ``rows`` rollouts each: min of the mins, max of
    the maxes, row-weighted mean of the means, agent_loop/slowest/* of the call with the slowest rollout,
    anything else from the first call that has it."""
    slowest_key = "agent_loop/slowest/generate_sequences"
    slowest = timings[max(range(len(timings)), key=lambda i: timings[i].get(slowest_key, -math.inf))]
    merged = {}
    for key in dict.fromkeys(k for timing in timings for k in timing):
        have = [(timing[key], w) for timing, w in zip(timings, rows, strict=True) if key in timing]
        if key.endswith("/min"):
            merged[key] = min(v for v, _ in have)
        elif key.endswith("/max"):
            merged[key] = max(v for v, _ in have)
        elif key.endswith("/mean"):
            merged[key] = sum(v * w for v, w in have) / sum(w for _, w in have)
        elif key.startswith("agent_loop/slowest/") and key in slowest:
            merged[key] = slowest[key]
        else:
            merged[key] = have[0][0]
    return merged


def _generate_round(trainer, generate, index: int, batches: int, prompt: DataProto, gen_in: DataProto) -> _Round:
    """Generate (and score) one round and classify its groups."""
    n = trainer.config.actor_rollout_ref.rollout.n
    expected = np.repeat(prompt.non_tensor_batch["uid"], n)
    if len(gen_in) != len(expected) or not np.array_equal(gen_in.non_tensor_batch["uid"], expected):
        raise RuntimeError(f"{NAME}: round {index}: the generation input is not its prompt batch repeated {n} times")
    padded, pad = pad_dataproto_to_divisor(gen_in, trainer.config.actor_rollout_ref.rollout.agent.num_workers)
    start = time.perf_counter()
    output = unpad_dataproto(generate(padded), pad)
    gen_s = time.perf_counter() - start
    if len(output) != len(gen_in):
        raise RuntimeError(f"{NAME}: round {index}: {len(gen_in)} rows in, {len(output)} rows generated")
    sent, back = gen_in.non_tensor_batch.get("raw_prompt"), output.non_tensor_batch.get("raw_prompt")
    if sent is not None and back is not None:  # str: prompts loaded from parquet may be numpy arrays
        wrong = [i for i, (a, b) in enumerate(zip(sent, back, strict=True)) if str(a) != str(b)]
        if wrong:
            raise RuntimeError(
                f"{NAME}: round {index}: generation output row {wrong[0]} is not the rollout of input row "
                f"{wrong[0]} (raw_prompt differs in {len(wrong)} of {len(sent)} rows)"
            )
    if "rm_scores" not in output.batch.keys():
        raise RuntimeError(f"{NAME}: the generation output has no rm_scores (rewards must come from the agent loop)")
    timing = output.meta_info.pop("timing", {})
    scores = output.batch["rm_scores"].sum(dim=-1)
    response_length = output.batch["responses"].shape[-1]
    lengths = output.batch["attention_mask"][:, -response_length:].sum(dim=-1)
    return _Round(
        index=index,
        batches=batches,
        prompt=prompt,
        gen_in=gen_in,
        output=output,
        timing=timing,
        gen_s=gen_s,
        scores=scores.detach().to(torch.float64).cpu().numpy(),
        lengths=lengths.cpu().numpy(),
        informative=classify(trainer, gen_in, scores),
        judge_status=output.non_tensor_batch.get("judge_status"),
        judge_bad_frac=judge_outage_frac(output),
    )


def _build_round(rt, trainer, batch_dicts: list) -> tuple:
    """(prompt side, generation input) of extra dataloader batches, built the way fit() builds its own."""
    config = trainer.config
    batch = DataProto.concat([DataProto.from_single_dict(d) for d in batch_dicts])
    batch.meta_info["temperature"] = config.actor_rollout_ref.rollout.temperature
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    rt.building = True
    try:
        gen_batch = trainer._get_gen_batch(batch)
    finally:
        rt.building = False
    gen_batch.meta_info["global_steps"] = trainer.global_steps
    return batch, gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)


def _step_metrics(rounds: list, kept: list, n: int, batches_used: int, stop, loader: SharedLoader) -> dict:
    """The dynamic_sampling/* metrics of one step, unprefixed (see the module docstring)."""
    first = rounds[0]
    informative = np.concatenate([rd.informative for rd in rounds])
    kept_groups = np.concatenate(kept)
    scores = np.concatenate([rd.scores for rd in rounds])
    lengths = np.concatenate([rd.lengths for rd in rounds])
    clipped = np.concatenate([rd.lengths == rd.output.batch["responses"].shape[-1] for rd in rounds])
    failed = [_judge_failed(rd.judge_status) for rd in rounds if rd.judge_status is not None]
    m = {
        "rounds": len(rounds),
        "batches_used": batches_used,
        "prompts_generated": len(informative),
        "rollouts_generated": len(scores),
        "informative_groups_round1": int(first.informative.sum()),
        "frac_informative_round1": float(first.informative.mean()),
        "informative_groups_total": int(informative.sum()),
        "frac_informative_all": float(informative.mean()),
        "kept_informative": int((kept_groups & informative).sum()),
        "kept_flat": int((kept_groups & ~informative).sum()),
        "dropped_informative": int((~kept_groups & informative).sum()),
        "dropped_flat": int((~kept_groups & ~informative).sum()),
        "cap_hit": int(stop == "cap"),
        "outage_stop": int(stop == "outage"),
        "score_mean_round1": float(first.scores.mean()),
        "score_mean_all": float(scores.mean()),
        "score_mean_kept": float(scores[np.repeat(kept_groups, n)].mean()),
        "response_length_mean_all": float(lengths.mean()),
        "clip_ratio_all": float(clipped.mean()),
        "generated_tokens_total": int(lengths.sum()),
        "gen_s_round1": first.gen_s,
        "gen_s_extra": float(sum(rd.gen_s for rd in rounds[1:])),
        "batches_consumed_total": loader.batches_consumed_total,
        "epoch": loader.batches_consumed_total / len(loader),
    }
    if failed:
        m["judge_outage_frac_all"] = float(np.concatenate(failed).mean())
    return m


def _status_counts(statuses) -> dict:
    """{judge_status: count}; "none" counts rows without a status (a round without the column dumps {})."""
    return dict(collections.Counter("none" if s is None else f"{float(s):g}" for s in statuses))


def _dump(trainer, step: int, rounds: list, kept: list, filled: list, n: int) -> None:
    """Per-group detail of every group generated at ``step`` (no texts), if trainer.rollout_data_dir is set."""
    rollout_dir = trainer.config.trainer.get("rollout_data_dir", None)
    if not rollout_dir:
        return
    groups = []
    for rd, kept_r, filled_r in zip(rounds, kept, filled, strict=True):
        status = rd.judge_status
        reward_models = rd.prompt.non_tensor_batch.get("reward_model", [None] * len(rd.prompt))
        for g, (uid, rm) in enumerate(zip(rd.prompt.non_tensor_batch["uid"], reward_models, strict=True)):
            rows = slice(g * n, (g + 1) * n)
            groups.append(
                {
                    "round": rd.index,
                    "uid": str(uid),
                    "problem": rm.get("ground_truth") if isinstance(rm, dict) else None,
                    "scores": np.round(rd.scores[rows], 4).tolist(),
                    "response_lengths": rd.lengths[rows].tolist(),
                    "judge_status": {} if status is None else _status_counts(status[rows]),
                    "informative": bool(rd.informative[g]),
                    "kept": bool(kept_r[g]),
                    "filled": bool(filled_r[g]),
                }
            )
    rounds_info = [
        {"round": rd.index, "batches": rd.batches, "prompts": len(rd.prompt), "gen_s": round(rd.gen_s, 3)}
        for rd in rounds
    ]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(rollout_dir)), f"{NAME}_{trainer.config.trainer.experiment_name}"
    )
    dump = {"step": step, "rounds": rounds_info, "groups": groups}
    _adaptive._write_json(os.path.join(out_dir, f"{step}.json"), dump)


def _sample_step(rt, trainer, generate, prompts: DataProto) -> DataProto:
    """fit()'s training generate call under dynamic sampling: the round loop, then the kept groups' output."""
    step = trainer.global_steps
    capture, rt.capture = rt.capture, None
    if capture is None or capture[0] != step:
        found = "nothing" if capture is None else f"step {capture[0]}"
        raise RuntimeError(f"{NAME}: training generate call at step {step}, but fit()'s prompt batch captured {found}")
    batch = capture[1]
    config = trainer.config
    n = config.actor_rollout_ref.rollout.n
    batch_size = config.data.train_batch_size
    if len(batch) != batch_size:
        raise RuntimeError(f"{NAME}: fit()'s batch has {len(batch)} prompts, data.train_batch_size is {batch_size}")
    settings, loader = rt.settings, trainer.train_dataloader
    # round 1 keeps its own view of fit()'s batch, which is rewritten in place to the kept groups below
    first = DataProto(batch=batch.batch, non_tensor_batch=batch.non_tensor_batch, meta_info=batch.meta_info)
    rounds = [_generate_round(trainer, generate, 1, 1, first, prompts)]
    batches_used, stop = 1, None
    while True:
        bad = rounds[-1].judge_bad_frac
        if bad is not None and bad > settings.outage_frac:
            print(
                f"[{NAME}] WARNING: JUDGE OUTAGE? step {step}, round {rounds[-1].index}: {bad:.0%} of the rollouts "
                "timed out or could not reach the judge (judge_status -1/-2); pulling no more batches this step",
                flush=True,
            )
            stop = "outage"
            break
        n_informative = sum(int(rd.informative.sum()) for rd in rounds)
        if n_informative >= settings.target_groups:
            break
        if batches_used >= settings.max_gen_batches:
            stop = "cap"
            break
        n_prompts = sum(len(rd.prompt) for rd in rounds)
        k = next_round_batches(settings, n_informative, n_prompts, batches_used, batch_size)
        prompt, gen_in = _build_round(rt, trainer, [loader.pull() for _ in range(k)])
        rounds.append(_generate_round(trainer, generate, len(rounds) + 1, k, prompt, gen_in))
        batches_used += k
    kept, filled = select_groups([rd.informative for rd in rounds], batch_size)
    kept_prompts, output = assemble(rounds, kept, n)
    batch.batch, batch.non_tensor_batch = kept_prompts.batch, kept_prompts.non_tensor_batch  # fit() repeats it next
    batch.check_consistency()
    output.meta_info["timing"] = merge_timing([rd.timing for rd in rounds], [len(rd.gen_in) for rd in rounds])
    m = _step_metrics(rounds, kept, n, batches_used, stop, loader)
    rt.pending_metrics = (step, {METRIC_PREFIX + key: value for key, value in m.items()})
    _dump(trainer, step, rounds, kept, filled, n)
    print(
        f"[{NAME}] step {step}: {m['rounds']} rounds, {m['batches_used']} batches, {m['informative_groups_total']}/"
        f"{m['prompts_generated']} groups informative; training on {m['kept_informative']} informative + "
        f"{m['kept_flat']} flat" + {None: "", "cap": " (batch cap hit)", "outage": " (judge outage)"}[stop],
        flush=True,
    )
    return output


def _wrap_generate(rt, manager):
    """manager.generate_sequences, with fit()'s training calls running the round loop."""
    generate = manager.generate_sequences

    def generate_sequences(prompts):
        trainer = rt.trainer
        if trainer is None or trainer.async_rollout_manager is not manager or prompts.meta_info.get("validate"):
            return generate(prompts)
        return _sample_step(rt, trainer, generate, prompts)

    setattr(generate_sequences, _WRAPPED_ATTR, True)
    return generate_sequences


def _consumed_before(trainer) -> int:
    """Dataloader batches the run consumed before the step fit() resumes at (state file of the checkpoint)."""
    step = trainer.global_steps
    if step == 0:
        return 0
    path = os.path.join(_adaptive._resume_folder(trainer), STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        if state.get("version") == 1 and state.get("step") == step:
            return int(state["batches_consumed_total"])
        problem = f"holds step {state.get('step')} (version {state.get('version')})"
    else:
        problem = "is missing"
    print(
        f"[{NAME}] WARNING: resuming from step {step} but {path} {problem}; counting one dataloader batch per step "
        "so far (only dynamic_sampling/batches_consumed_total and epoch use it)",
        flush=True,
    )
    return step


class _Runtime:
    """Per-process state: the trainer dynamic sampling is on for, its settings, this step's handoffs."""

    def __init__(self):
        self.trainer = None
        self.settings = None
        self.reset_step()

    def reset_step(self) -> None:
        self.capture = None  # (global step, fit()'s prompt batch) from fit()'s _get_gen_batch call
        self.building = False  # True while the round loop builds its own batches (they are not captured)
        self.pending_metrics = None  # (global step, metrics) until compute_data_metrics picks them up


def _install():
    existing = getattr(ray_trainer, _RUNTIME_ATTR, None)
    if existing is not None:
        return existing
    rt = _Runtime()
    trainer_cls = ray_trainer.RayPPOTrainer
    orig_fit = trainer_cls.fit
    orig_get_gen_batch = trainer_cls._get_gen_batch
    orig_save = trainer_cls._save_checkpoint
    orig_load = trainer_cls._load_checkpoint
    orig_compute_data_metrics = ray_trainer.compute_data_metrics

    def fit(self, *args, **kwargs):
        settings = settings_from(self.config)
        rt.trainer, rt.settings = (self, settings) if settings is not None else (None, None)
        rt.reset_step()
        if settings is not None:
            if not isinstance(self.train_dataloader, SharedLoader):
                self.train_dataloader = SharedLoader(self.train_dataloader)
            manager = self.async_rollout_manager
            if not getattr(manager.generate_sequences, _WRAPPED_ATTR, False):
                manager.generate_sequences = _wrap_generate(rt, manager)
        return orig_fit(self, *args, **kwargs)

    def _get_gen_batch(self, batch, *args, **kwargs):
        gen_batch = orig_get_gen_batch(self, batch, *args, **kwargs)
        # fit()'s call: its prompt batch carries the sampling temperature (_validate's meta_info is empty)
        if rt.trainer is self and not rt.building and "temperature" in batch.meta_info:
            rt.capture = (self.global_steps, batch)
        return gen_batch

    def compute_data_metrics(*args, **kwargs):
        metrics = orig_compute_data_metrics(*args, **kwargs)
        pending = rt.pending_metrics
        if pending is not None and rt.trainer is not None and pending[0] == rt.trainer.global_steps:
            metrics = {**metrics, **pending[1]}
            rt.pending_metrics = None
        return metrics

    def _save_checkpoint(self, *args, **kwargs):
        if rt.trainer is not self:
            return orig_save(self, *args, **kwargs)
        # Written before verl's own save, so a checkpoint verl marks complete always has it.
        path = os.path.join(
            os.path.abspath(self.config.trainer.default_local_dir), f"global_step_{self.global_steps}", STATE_FILE
        )
        state = {
            "version": 1,
            "step": self.global_steps,
            "batches_consumed_total": self.train_dataloader.batches_consumed_total,
        }
        _adaptive._write_json(path, state)
        result = orig_save(self, *args, **kwargs)
        if not os.path.exists(path):
            _adaptive._write_json(path, state)
        return result

    def _load_checkpoint(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        if rt.trainer is self:
            rt.reset_step()
            loader = self.train_dataloader
            loader.batches_consumed_total = _consumed_before(self)
            epoch, total_epochs = self.global_steps // len(loader), self.config.trainer.total_epochs
            if epoch >= total_epochs:
                raise ValueError(
                    f"{NAME}: fit() would resume in epoch {epoch} (step {self.global_steps}, {len(loader)} batches "
                    f"per epoch) of trainer.total_epochs={total_epochs} and stop at once; raise trainer.total_epochs"
                )
            s = rt.settings
            print(
                f"[{NAME}] on: max_gen_batches={s.max_gen_batches}, target_groups={s.target_groups}, "
                f"batches_per_round={s.batches_per_round or 'auto'}, outage_frac={s.outage_frac}; "
                f"{loader.batches_consumed_total} dataloader batches consumed before step {self.global_steps + 1}",
                flush=True,
            )
        return result

    trainer_cls.fit = fit
    trainer_cls._get_gen_batch = _get_gen_batch
    trainer_cls._save_checkpoint = _save_checkpoint
    trainer_cls._load_checkpoint = _load_checkpoint
    ray_trainer.compute_data_metrics = compute_data_metrics
    setattr(ray_trainer, _RUNTIME_ATTR, rt)
    return rt


RUNTIME = _install()
