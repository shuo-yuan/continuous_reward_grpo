"""Offline analysis: how different advantage estimators treat a continuous (0-100) reward.

Recomputes, from saved GRPO rollouts (trainer.rollout_data_dir/*.jsonl), the advantages that
verl's estimators would assign to the same responses, with verl's exact formulas:

  grpo        (r - mean_g) / (std_g + 1e-6)            std_g unbiased, per group of n
  drgrpo      (r - mean_g)                              grpo with norm_adv_by_std_in_grpo=False
  rloo        n/(n-1) * (r - mean_g)                    leave-one-out baseline
  rpp         masked_whiten(r - mean_g) over all response tokens of the step batch
              (reinforce_plus_plus_baseline: token-weighted batch mean/std, unbiased)
  binary      grpo on 1[r > 0]                         contrast with a pass/fail reward

Every response's advantage is broadcast to its tokens and the loss is token-mean, so a
response's share of the policy-gradient signal is ~ |A| * length; "weight" below means that.

Usage:
  python analysis/continuous_reward_advantages.py --rollout-dir outputs/rollout_data_synthetic \
      --steps 1-100 --tokenizer <hf model dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import numpy as np

EPS = 1e-6


def load_steps(rollout_dir: str, lo: int, hi: int):
    steps = {}
    for n in range(lo, hi + 1):
        path = os.path.join(rollout_dir, f"{n}.jsonl")
        if os.path.exists(path):
            steps[n] = [json.loads(line) for line in open(path)]
    return steps


def grpo_adv(r: np.ndarray) -> np.ndarray:
    if len(r) < 2:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std(ddof=1) + EPS)


def step_advantages(rows, lengths):
    """Return per-response arrays for one step batch."""
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["input"]].append(i)
    n_resp = len(rows)
    r = np.array([row["score"] for row in rows], dtype=np.float64)
    L = np.asarray(lengths, dtype=np.float64)
    out = {k: np.zeros(n_resp) for k in ("grpo", "drgrpo", "rloo", "binary")}
    gid = np.zeros(n_resp, dtype=int)
    gstd = np.zeros(n_resp)
    grange = np.zeros(n_resp)
    for g, idx in enumerate(groups.values()):
        idx = np.array(idx)
        rg = r[idx]
        n = len(idx)
        gid[idx] = g
        gstd[idx] = rg.std(ddof=1) if n > 1 else 0.0
        grange[idx] = rg.max() - rg.min()
        out["grpo"][idx] = grpo_adv(rg)
        out["drgrpo"][idx] = rg - rg.mean()
        out["rloo"][idx] = (rg - rg.mean()) * n / (n - 1) if n > 1 else 0.0
        out["binary"][idx] = grpo_adv((rg > 0).astype(np.float64))
    # reinforce_plus_plus_baseline: whiten (r - mean_g) over all tokens in the batch
    a = out["drgrpo"]
    mu = (a * L).sum() / L.sum()
    var = (L * (a - mu) ** 2).sum() / (L.sum() - 1)
    out["rpp"] = (a - mu) / math.sqrt(var + 1e-8)
    return r, L, gid, gstd, grange, out


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", default="outputs/rollout_data_synthetic")
    ap.add_argument("--steps", default="1-100")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir (token lengths); chars/3.3 if unset")
    ap.add_argument("--block", type=int, default=20)
    args = ap.parse_args()
    lo, hi = map(int, args.steps.split("-"))
    steps = load_steps(args.rollout_dir, lo, hi)
    print(f"loaded {len(steps)} steps ({lo}-{hi}), {sum(len(v) for v in steps.values())} responses")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Per-response records across all steps
    rec = collections.defaultdict(list)
    for n, rows in sorted(steps.items()):
        outs = [row["output"] for row in rows]
        if tok is not None:
            lengths = [len(x) for x in tok(outs, add_special_tokens=False)["input_ids"]]
        else:
            lengths = [max(1, int(len(o) / 3.3)) for o in outs]
        r, L, gid, gstd, grange, adv = step_advantages(rows, lengths)
        rec["step"] += [n] * len(r)
        rec["gkey"] += [f"{n}:{g}" for g in gid]
        rec["r"] += list(r)
        rec["L"] += list(L)
        rec["gstd"] += list(gstd)
        rec["grange"] += list(grange)
        for k, v in adv.items():
            rec[k] += list(v)
    R = {k: np.asarray(v) if k != "gkey" else np.asarray(v) for k, v in rec.items()}
    schemes = ["grpo", "drgrpo", "rloo", "rpp", "binary"]

    # One row per group for group-level stats
    _, first = np.unique(R["gkey"], return_index=True)
    G = {k: R[k][first] for k in ("step", "gstd", "grange")}
    flat = G["gstd"] == 0

    print("\n=== 1. Within-group reward spread (0-100 points), per block of steps")
    print(f"{'steps':>9} {'groups':>6} {'all-equal':>9} {'std<1':>7} {'std<5':>7} {'median std':>10} {'p90 std':>8}")
    for b in range(lo, hi + 1, args.block):
        m = (G["step"] >= b) & (G["step"] < b + args.block)
        nf = m & ~flat
        s = np.sort(G["gstd"][nf])
        print(f"{b:>4}-{min(b + args.block - 1, hi):<4} {m.sum():>6} {pct(flat[m].mean()):>9} "
              f"{pct((G['gstd'][nf] < 1).sum() / max(m.sum(), 1)):>7} {pct((G['gstd'][nf] < 5).sum() / max(m.sum(), 1)):>7} "
              f"{(s[len(s) // 2] if len(s) else 0):>10.1f} {(s[int(len(s) * .9)] if len(s) else 0):>8.1f}")

    print("\n=== 2. Share of the policy-gradient signal (sum |A|*len) by within-group spread quartile")
    print("    (non-flat groups; Q1 = smallest spread). Uniform = each quartile ~25%.")
    nf_resp = R["gstd"] > 0
    qs = np.quantile(G["gstd"][~flat], [0.25, 0.5, 0.75])
    edges = [0] + list(qs) + [np.inf]
    print(f"    spread quartile edges (std, points): {', '.join(f'{q:.1f}' for q in qs)}")
    print(f"{'scheme':>8} " + " ".join(f"{f'Q{i+1}':>7}" for i in range(4)))
    for k in schemes:
        w = np.abs(R[k]) * R["L"] * nf_resp
        tot = w.sum()
        shares = [w[(R["gstd"] > edges[i]) & (R["gstd"] <= edges[i + 1])].sum() / tot for i in range(4)]
        print(f"{k:>8} " + " ".join(f"{pct(s):>7}" for s in shares))

    print("\n=== 3. Tiny differences: non-flat groups whose scores span < 2 points")
    tiny = (R["grange"] > 0) & (R["grange"] < 2)
    ng_tiny = len(np.unique(R["gkey"][tiny]))
    print(f"    {ng_tiny} such groups ({pct(ng_tiny / max((~flat).sum(), 1))} of non-flat groups)")
    for k in ("grpo", "drgrpo", "rpp"):
        all_mean = np.abs(R[k][nf_resp]).mean()
        print(f"    {k:>7}: mean |A| in tiny-spread groups = {np.abs(R[k][tiny]).mean() / all_mean:5.2f}x its average |A|")
    ex = [g for g in np.unique(R["gkey"][tiny])][:3]
    for g in ex:
        m = R["gkey"] == g
        print(f"    e.g. step {g.split(':')[0]}: scores {np.round(R['r'][m], 2).tolist()} -> grpo A {np.round(R['grpo'][m], 2).tolist()}")

    print("\n=== 4. Agreement between schemes (all responses in non-flat groups)")
    def corr(a, b):
        return np.corrcoef(a, b)[0, 1]
    g, d, p = R["grpo"][nf_resp], R["drgrpo"][nf_resp], R["rpp"][nf_resp]
    print(f"    corr(grpo, drgrpo) = {corr(g, d):.3f}   (same sign always; differ only by a per-group 1/std factor)")
    print(f"    corr(grpo, rpp)    = {corr(g, p):.3f}")
    print(f"    corr(drgrpo, rpp)  = {corr(d, p):.3f}")
    nz = (g != 0) & (p != 0)
    print(f"    sign(rpp) != sign(grpo) for {pct((np.sign(g[nz]) != np.sign(p[nz])).mean())} of responses "
          f"(the token-weighted batch mean shift in rpp)")
    flat_resp = R["gstd"] == 0
    print(f"    responses in all-equal groups: {pct(flat_resp.mean())}; rpp gives them |A| mean "
          f"{np.abs(R['rpp'][flat_resp]).mean():.3f} (grpo/drgrpo give 0)")
    # length vs score inside groups
    rs = []
    for gk in np.unique(R["gkey"][nf_resp]):
        m = R["gkey"] == gk
        if R["L"][m].std() > 0:
            rs.append(corr(R["r"][m], R["L"][m]))
    print(f"    within-group corr(score, length): median {np.median(rs):+.2f}, mean {np.mean(rs):+.2f}")

    print("\n=== 5. Binarized reward (1[score > 0]) vs continuous")
    bflat = []
    for gk in np.unique(R["gkey"]):
        m = R["gkey"] == gk
        bflat.append(len(np.unique(R["r"][m] > 0)) == 1)
    bflat = np.array(bflat)
    print(f"    all-equal groups: continuous {pct(flat.mean())} -> binary {pct(bflat.mean())}")
    b, nzb = R["binary"], (R["binary"] != 0) & (g.size > 0)
    both = nf_resp & (R["binary"] != 0)
    print(f"    corr(grpo, binary) on responses where both nonzero: {corr(R['grpo'][both], R['binary'][both]):.3f}")

    print("\n=== 6. Dr. GRPO scale: divide rewards by c so its advantage magnitude matches GRPO's")
    print("    c = RMS(drgrpo A) / RMS(grpo A), token-weighted, over all responses (flat groups included)")
    print(f"{'steps':>9} {'RMS grpo':>9} {'RMS drgrpo':>10} {'c':>7}")
    cs = []
    for b in range(lo, hi + 1, args.block):
        m = (R["step"] >= b) & (R["step"] < b + args.block)
        w = R["L"][m]
        rg = math.sqrt((w * R["grpo"][m] ** 2).sum() / w.sum())
        rd = math.sqrt((w * R["drgrpo"][m] ** 2).sum() / w.sum())
        cs.append(rd / rg if rg > 0 else float("nan"))
        print(f"{b:>4}-{min(b + args.block - 1, hi):<4} {rg:>9.3f} {rd:>10.2f} {cs[-1]:>7.1f}")
    w = R["L"]
    c_all = math.sqrt((w * R["drgrpo"] ** 2).sum() / w.sum()) / math.sqrt((w * R["grpo"] ** 2).sum() / w.sum())
    print(f"    all steps: c = {c_all:.1f}")


if __name__ == "__main__":
    main()
