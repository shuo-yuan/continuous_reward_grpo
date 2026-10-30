#!/usr/bin/env python
"""
toy_unbiasedness_sim_b.py  --  variant-B toy harness for advantage-estimator unbiasedness.

Every proposed estimator (plus GRPO / Dr.GRPO / RLOO / OPO / REINFORCE++-baseline references) is run on the
SAME toy problem and compared with the EXACT policy gradient.

Toy problem
-----------
* K = 6 prompts.  A response is a sequence of 1-3 tokens from the vocabulary {0, 1, 2, EOS}:
      (EOS) [L=1],  (a,EOS) [L=2],  (a,b,EOS) [L=3],  (a,b,c) [L=3, truncated at TMAX=3 without EOS]
  -> 40 responses per prompt, 13 prefix states x 4 logits = 52 parameters per prompt, D = 312 parameters.
* Policy: pi(y_t | x, y_<t) = softmax(theta[x, state(y_<t), :]).  Tokens are modelled EXPLICITLY (choice stated):
  sum_t grad log pi(y_t | x, y_<t) is the sum of the per-token softmax scores (e_{y_t} - pi(.|state_t)), one
  parameter row per visited state.  This is NOT the "L identical tokens" shortcut, so position-dependent
  baselines are meaningful and lengths differ (L in {1,2,3}).
* Rewards r(x,y) in [0,100], deterministic in (x,y).  Prompt 0 near-flat (50 / 50.15 / 50.3 / 49.9),
  prompt 1 spread ~25, prompt 2 all-equal, prompt 3 spread ~13, prompt 4 bimodal (0 / 100 / small partial credit),
  prompt 5 spread ~18 with a strong length penalty.  Longer responses score slightly lower on prompts 1, 3, 5.
* A step: B = 8 prompts drawn uniformly WITH replacement from the K = 6 (so E_x is uniform over the dataset),
  n = 8 samples per prompt, N = 64 sequences.  Fully on-policy, ratio == 1, so the applied gradient is
      g_hat = sum_i sum_t w_i A_{i,t} grad log pi(y_it | x_i, y_i<t)
  with the aggregation weight w_i exactly as in verl's agg_loss (core_algos.py):
      token-mean              : w_i = 1 / sum_j L_j          (batch total tokens -> sample dependent)
      seq-mean-token-mean     : w_i = 1 / (N L_i)            (per-sequence token mean; the legacy 1-seq micro-batch path)
      seq-mean-token-sum-norm : w_i = 1 / (N S), S = TMAX    (constant denominator; loss_mask.shape[-1])
* The target direction is G = concat_x grad J_x = K * grad J  (J = (1/K) sum_x J_x); kappa absorbs the constant.

Monte Carlo
-----------
All randomness comes from the response / prompt sampling RNG streams (seed, seed+1 for the importance-sampled
prompt stream, seed+2 for the prefix-branching stream).  Lagged / EMA / per-prompt-history statistics are
computed from PREVIOUS batches only (a warm-up prefix of the same stream initialises them); the arrays holding
"state before step t" are built with exclusive cumulative sums / one-step-lagged recursions.

Rows: the 8 references (GRPO under three aggregations, Dr.GRPO raw, RLOO under two aggregations, verl OPO,
reinforce_plus_plus_baseline), the proposed ideas (rloo-const, rloo-lagged-scale [EMA and leave-one-group-out],
rloo-prompt-scale [lagged, frozen, stateless cross-fit], is-prompt-sampling, rloo-hist-baseline [k=2, 8],
lagged-gradnorm-scale, prefix-branching [as stated, and with the /k on prefix tokens removed], opo-loo,
otb-loo-position, prefix-feature-baseline) and one CONTROL row whose baseline illegally uses the sample's own
length, to show the harness detects a known bias.

Reported per estimator: fitted kappa = <E[g],G>/|G|^2, cos(E[g], G), relative bias after the best scale
(|E[g]/kappa - G| / |G| = tan(angle)), its MC noise floor, a z-statistic (~1 if unbiased), per-prompt block
ratios |E[g]_x| / (kappa |G_x|) (constant 1 = unbiased for J; varying = reweighting), per-prompt cosines,
and trace Var(g)/kappa^2 relative to RLOO under seq-mean-token-sum-norm.

Usage: .venv/bin/python analysis/toy_unbiasedness_sim_b.py --nb 200000 --warmup 2000 --seed 1 [--json out.json]
"""
import argparse
import json
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------- constants
K, VOC, EOS, TMAX = 6, 4, 3, 3
NSTATE = 1 + 3 + 9          # root, after (a), after (a,b)
PP = NSTATE * VOC           # params per prompt = 52
D = K * PP                  # 312
B, n = 8, 8                 # prompts per batch, samples per prompt
N = B * n                   # 64 sequences per batch
S_CONST = float(TMAX)       # verl: loss_mask.shape[-1] (max response length) for seq-mean-token-sum-norm
NF = 5                      # prefix features for the prefix-feature baseline

# hyper-parameters of the ideas (as proposed)
C_FIXED = 18.0              # rloo-const scale (measured RMS within-group std)
LAG_S0, LAG_BETA, LAG_CMIN = 18.0, 0.9, 3.0          # rloo-lagged-scale
PS_K0, PS_SMIN = 2.0, 5.0                            # rloo-prompt-scale shrinkage / floor
XF_EPS0 = 9.0                                        # cross-fit variant additive floor
IS_SMIN = 5.0                                        # is-prompt-sampling floor (uses PS_K0 shrinkage)
HIST_BETA, HIST_K_LIST = 0.5, (2.0, 8.0)             # rloo-hist-baseline
GN_TARGET, GN_BETA, GN_MIN, GN_INIT = 0.05, 0.9, 1e-3, 1.0   # lagged-gradnorm-scale
PB_M, PB_K, PB_T0 = 4, 2, 1                          # prefix-branching: m prefixes x k continuations, T0 = 1 token
PF_SCAP, PF_RIDGE = 10.0, 1.0                        # prefix-feature baseline clip / ridge


# ----------------------------------------------------------------------------- environment
def enumerate_responses():
    resp = [(EOS,)]
    for a in range(3):
        resp.append((a, EOS))
        for b in range(3):
            resp.append((a, b, EOS))
            for c in range(3):
                resp.append((a, b, c))
    return resp


def state_of(prefix):
    if len(prefix) == 0:
        return 0
    if len(prefix) == 1:
        return 1 + prefix[0]
    return 4 + 3 * prefix[0] + prefix[1]


def build_rewards(L, rng):
    R = len(L)
    Rtab = np.zeros((K, R))
    # 0: near-flat  [50, 50, ..., 50.15, 50.3, 49.9]
    r = np.full(R, 50.0)
    idx = rng.choice(R, 6, replace=False)
    r[idx[:3]] = 50.15
    r[idx[3]] = 50.3
    r[idx[4:6]] = 49.9
    Rtab[0] = r
    # 1: wide spread (~25) with mild length penalty
    Rtab[1] = np.clip(rng.uniform(5, 95, R) - 4.0 * (L - 2), 0, 100)
    # 2: all equal
    Rtab[2] = 50.0
    # 3: medium spread (~13) with mild length penalty
    Rtab[3] = np.clip(55 + rng.normal(0, 13, R) - 3.0 * (L - 2), 0, 100)
    # 4: bimodal: mostly 0, some 100, a few partial credits
    r = np.zeros(R)
    idx = rng.choice(R, 10, replace=False)
    r[idx[:6]] = 100.0
    r[idx[6:]] = rng.uniform(5, 30, 4)
    Rtab[4] = r
    # 5: strongly length-coupled (~18)
    Rtab[5] = np.clip(72 - 10.0 * (L - 1) + rng.normal(0, 10, R), 0, 100)
    return Rtab


class Env:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.resp = enumerate_responses()
        self.R = len(self.resp)
        self.L = np.array([len(y) for y in self.resp])
        self.first = np.array([y[0] for y in self.resp])
        self.states = np.full((self.R, TMAX), -1, int)
        self.tokens = np.full((self.R, TMAX), -1, int)
        for yi, y in enumerate(self.resp):
            for t in range(len(y)):
                self.states[yi, t] = state_of(y[:t])
                self.tokens[yi, t] = y[t]
        theta = rng.normal(0, 0.8, (K, NSTATE, VOC))
        theta[:, 0, EOS] -= 1.5          # empty responses rare
        self.theta = theta
        self.Rtab = build_rewards(self.L, rng)
        self.set_policy(theta)

    # --- policy-dependent tables
    def probs(self, theta):
        z = theta - theta.max(-1, keepdims=True)
        pi = np.exp(z)
        pi /= pi.sum(-1, keepdims=True)
        logpi = np.log(pi)
        logP = np.zeros((K, self.R))
        for yi in range(self.R):
            for t in range(self.L[yi]):
                logP[:, yi] += logpi[:, self.states[yi, t], self.tokens[yi, t]]
        return pi, logpi, np.exp(logP)

    def set_policy(self, theta):
        pi, logpi, P = self.probs(theta)
        self.pi, self.logpi, self.P = pi, logpi, P
        S_tok = np.zeros((K, self.R, TMAX, PP))
        for yi in range(self.R):
            for t in range(self.L[yi]):
                s, tok = self.states[yi, t], self.tokens[yi, t]
                S_tok[:, yi, t, s * VOC:(s + 1) * VOC] = -pi[:, s, :]
                S_tok[:, yi, t, s * VOC + tok] += 1.0
        self.S_tok = S_tok
        self.S_seq = S_tok.sum(2)
        self.cdf = np.cumsum(P, 1)
        self.cdf[:, -1] = 1.0
        ccdf = np.zeros((K, VOC, self.R))
        for z in range(VOC):
            pz = P * (self.first == z)[None, :]
            pz = pz / pz.sum(1, keepdims=True)
            ccdf[:, z] = np.cumsum(pz, 1)
            ccdf[:, z, -1] = 1.0
        self.ccdf = ccdf
        self.pi0cdf = np.cumsum(pi[:, 0, :], -1)
        self.pi0cdf[:, -1] = 1.0
        # exact gradient of J_x (one block per prompt) and of sum_x J_x
        self.G_blocks = np.einsum('xy,xy,xyk->xk', P, self.Rtab, self.S_seq)
        flat = (self.Rtab.max(1) - self.Rtab.min(1)) == 0          # all-equal prompt: grad J_x = 0 exactly
        self.G_blocks[flat] = 0.0
        self.G = self.G_blocks.reshape(D)
        self.J_x = (P * self.Rtab).sum(1)
        # per-prompt reward statistics under the policy
        mu = self.J_x
        self.sigma_x = np.sqrt((P * (self.Rtab - mu[:, None]) ** 2).sum(1))
        EL = (P * self.L[None, :]).sum(1)
        VL = (P * (self.L[None, :] - EL[:, None]) ** 2).sum(1)
        cov = (P * (self.Rtab - mu[:, None]) * (self.L[None, :] - EL[:, None])).sum(1)
        self.EL = EL
        self.corr_rL = cov / np.sqrt(np.maximum(self.sigma_x ** 2 * VL, 1e-30))
        # prefix features (legal: functions of x and y_<t only)
        H = -(pi * logpi).sum(-1)                       # (K, NSTATE) entropy of each state
        Lbar = EL.mean()
        PHI = np.zeros((K, self.R, TMAX, NF))
        for yi in range(self.R):
            for t in range(self.L[yi]):
                PHI[:, yi, t, 0] = t / Lbar
                PHI[:, yi, t, 2] = H[:, self.states[yi, t]]            # entropy of the current state (prefix-determined)
                if t > 0:
                    PHI[:, yi, t, 1] = np.mean([logpi[:, self.states[yi, s], self.tokens[yi, s]] for s in range(t)], axis=0)
                    PHI[:, yi, t, 3] = float(self.tokens[yi, 0] == 0)
                    PHI[:, yi, t, 4] = float(self.tokens[yi, 0] == 1)
        self.PHI = PHI

    def grad_check(self, rng, ncoord=12, eps=1e-5):
        """central finite differences of sum_x J_x against G"""
        errs = []
        for j in rng.choice(D, ncoord, replace=False):
            x, rem = divmod(j, PP)
            s, v = divmod(rem, VOC)
            tp = self.theta.copy(); tp[x, s, v] += eps
            tm = self.theta.copy(); tm[x, s, v] -= eps
            Jp = (self.probs(tp)[2] * self.Rtab).sum()
            Jm = (self.probs(tm)[2] * self.Rtab).sum()
            fd = (Jp - Jm) / (2 * eps)
            errs.append(abs(fd - self.G[j]) / (abs(self.G[j]) + 1e-9))
        return max(errs)


# ----------------------------------------------------------------------------- sampling helpers
def sample_from_cdf(cdf_rows, u):
    """cdf_rows (..., R) ; u (...) -> index"""
    return np.minimum((u[..., None] > cdf_rows).sum(-1), cdf_rows.shape[-1] - 1)


def sample_main_stream(env, rng, NBT, chunk=20000):
    x = rng.integers(0, K, (NBT, B))
    y = np.empty((NBT, B, n), np.int16)
    for c0 in range(0, NBT, chunk):
        sl = slice(c0, min(c0 + chunk, NBT))
        u = rng.random((sl.stop - sl.start, B, n))
        y[sl] = sample_from_cdf(env.cdf[x[sl]][:, :, None, :], u)
    return x, y


def onehot(x):
    return np.eye(K)[x]


def agg_w(mode, Lb):
    if mode == 'token-mean':
        return np.broadcast_to(1.0 / Lb.sum((1, 2))[:, None, None], Lb.shape).copy()
    if mode == 'seq-mean-token-mean':
        return 1.0 / (N * Lb)
    if mode == 'seq-mean-token-sum-norm':
        return np.full(Lb.shape, 1.0 / (N * S_CONST))
    if mode == 'seq-mean-token-sum':
        return np.full(Lb.shape, 1.0 / N)
    raise ValueError(mode)


def assemble_seq(Aw, S_seq_b, P1):
    C = np.einsum('bgi,bgik->bgk', Aw, S_seq_b)
    return np.einsum('bgk,bgx->bxk', C, P1).reshape(-1, D)


def assemble_tok(Aw, S_tok_b, P1):
    C = np.einsum('bgit,bgitk->bgk', Aw, S_tok_b)
    return np.einsum('bgk,bgx->bxk', C, P1).reshape(-1, D)


# ----------------------------------------------------------------------------- accumulators
class Acc:
    def __init__(self):
        self.s = np.zeros(D); self.s2 = np.zeros(D); self.cnt = 0
        self.p = 0.0; self.p2 = 0.0; self.extra = {}

    def add(self, g, G):
        self.s += g.sum(0); self.s2 += (g ** 2).sum(0); self.cnt += len(g)
        proj = g @ G / (G @ G)
        self.p += proj.sum(); self.p2 += (proj ** 2).sum()

    def summary(self, env):
        G, Gb = env.G, env.G_blocks
        m = self.s / self.cnt
        var = np.maximum(self.s2 / self.cnt - m ** 2, 0.0)
        se = np.sqrt(var / self.cnt)
        kappa = m @ G / (G @ G)
        kappa_se = np.sqrt(max(self.p2 / self.cnt - (self.p / self.cnt) ** 2, 0) / self.cnt)
        cos = m @ G / (np.linalg.norm(m) * np.linalg.norm(G))
        resid = np.linalg.norm(m - kappa * G) / (abs(kappa) * np.linalg.norm(G))
        floor = np.sqrt((se ** 2).sum()) / (abs(kappa) * np.linalg.norm(G))
        ok = se > 0
        z = np.sqrt(np.mean(((m - kappa * G)[ok] / se[ok]) ** 2))
        trvar_n = var.sum() / kappa ** 2
        blocks = []
        for x in range(K):
            sl = slice(x * PP, (x + 1) * PP)
            mb, seb, gb = m[sl], se[sl], Gb[x]
            gn = np.linalg.norm(gb)
            okb = seb > 0
            if gn == 0.0:   # flat prompt: G_x = 0 exactly; report whether E[g]_x = 0 within noise
                zb = float(np.sqrt(np.mean((mb[okb] / seb[okb]) ** 2))) if okb.any() else 0.0
                blocks.append(dict(ratio=None, cos=None, z=zb, ci=None,
                                   norm_over_kG=float(np.linalg.norm(mb) / (abs(kappa) * np.linalg.norm(G)))))
            else:
                kx = mb @ gb / (gn ** 2)
                blocks.append(dict(ratio=float(np.linalg.norm(mb) / (abs(kappa) * gn)),
                                   cos=float(mb @ gb / (np.linalg.norm(mb) * gn)),
                                   z=float(np.sqrt(np.mean(((mb - kx * gb)[okb] / seb[okb]) ** 2))),
                                   ci=float(1.96 * np.sqrt((seb ** 2).sum()) / (abs(kappa) * gn)),
                                   norm_over_kG=None))
        return dict(kappa=float(kappa), kappa_rel_ci=float(1.96 * kappa_se / abs(kappa)), cos=float(cos),
                    resid=float(resid), floor=float(floor), ci_halfwidth=float(1.96 * floor), z=float(z),
                    trvar_norm=float(trvar_n), blocks=blocks, nb=int(self.cnt), extra=self.extra)


# ----------------------------------------------------------------------------- group statistics
def group_stats(r):
    mean_g = r.mean(-1)
    var_g = r.var(-1, ddof=1)
    loo = (r.sum(-1, keepdims=True) - r) / (n - 1)
    return mean_g, var_g, loo


def ema_lagged(v_now, v0, beta):
    """returns the EMA value BEFORE step t (state built from steps < t)"""
    out = np.empty(len(v_now) + 1)
    out[0] = v0
    vn = v_now.tolist()
    cur = v0
    for t in range(len(vn)):
        cur = beta * cur + (1 - beta) * vn[t]
        out[t + 1] = cur
    return out[:-1]


def exclusive_prompt_cumsums(x, val):
    """per-prompt exclusive cumulative sums over batches: (NBT, K) sums of val over groups with that prompt
    in batches strictly before t, and the matching counts"""
    NBT = len(x)
    SS = np.zeros((NBT, K)); M = np.zeros((NBT, K))
    for p in range(K):
        mask = (x == p)
        cnt = mask.sum(1); ss = (mask * val).sum(1)
        M[:, p] = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        SS[:, p] = np.concatenate([[0], np.cumsum(ss)[:-1]])
    return SS, M


def hist_mean_before(x, mean_g, beta):
    """per-group: EMA (over previous visits of the same prompt, strictly earlier batches) of the group mean"""
    NBT = len(x)
    h = np.zeros((NBT, B)); has = np.zeros((NBT, B), bool)
    for p in range(K):
        tt, gg = np.nonzero(x == p)
        vals = mean_g[tt, gg].tolist()
        ema = np.empty(len(vals)); cur = None
        for i, v in enumerate(vals):
            cur = v if cur is None else beta * cur + (1 - beta) * v
            ema[i] = cur
        last = np.searchsorted(tt, tt, side='left') - 1
        ok = last >= 0
        h[tt[ok], gg[ok]] = ema[last[ok]]
        has[tt[ok], gg[ok]] = True
    return h, has


# ----------------------------------------------------------------------------- is-prompt-sampling stream
def sample_is_stream(env, rng, NBT):
    """spread-proportional prompt sampling: q_t(x) = s(x)/Z from PAST data; u(x) = 1/(K q_t(x))"""
    x_is = np.empty((NBT, B), np.int8); y_is = np.empty((NBT, B, n), np.int16); u_is = np.empty((NBT, B))
    SS = np.zeros(K); M = np.zeros(K); v_bar = LAG_S0 ** 2
    visits = np.zeros(K); inv_check = np.zeros(NBT)
    cdf = env.cdf; Rtab = env.Rtab
    for t in range(NBT):
        v_x = (PS_K0 * v_bar + SS) / (PS_K0 + M)
        s = np.maximum(np.sqrt(v_x), IS_SMIN)
        q = s / s.sum()
        qc = np.cumsum(q); qc[-1] = 1.0
        x = np.minimum(np.searchsorted(qc, rng.random(B), side='right'), K - 1)
        u = rng.random((B, n))
        y = sample_from_cdf(cdf[x][:, None, :], u)
        r = Rtab[x[:, None], y]
        var_g = r.var(-1, ddof=1)
        x_is[t] = x; y_is[t] = y; u_is[t] = 1.0 / (K * q[x])
        inv_check[t] = np.mean(u_is[t] * q[x] * K)
        np.add.at(visits, x, 1)
        # AFTER the step: update history
        np.add.at(SS, x, var_g); np.add.at(M, x, 1)
        v_bar = LAG_BETA * v_bar + (1 - LAG_BETA) * var_g.mean()
    return x_is, y_is, u_is, visits, inv_check


# ----------------------------------------------------------------------------- prefix-branching stream
def sample_pb_stream(env, rng, NB):
    x = rng.integers(0, K, (NB, B))
    uz = rng.random((NB, B, PB_M))
    z = sample_from_cdf(env.pi0cdf[x][:, :, None, :], uz)                     # (NB,B,m) first tokens
    uc = rng.random((NB, B, PB_M, PB_K))
    y = sample_from_cdf(env.ccdf[x[..., None], z][:, :, :, None, :], uc)      # (NB,B,m,k)
    return x, z, y


def pb_token_advantages(env, x, y, prefix_div=PB_K):
    """y (nb,B,m,k): per-token advantages (nb,B,m*k,T), /c applied by caller.
    prefix_div = k is the idea as stated (k copies sum to rbar_a - b_{-a});
    prefix_div = 1 gives every copy the full (r_ab - b_{-a}) so prefix and continuation parts carry the same
    weight n = m k in expectation."""
    r = env.Rtab[x[..., None, None], y]                                        # (nb,B,m,k)
    rbar = r.mean(-1)                                                          # (nb,B,m)
    b_prefix = (rbar.sum(-1, keepdims=True) - rbar) / (PB_M - 1)               # (nb,B,m)  other prefixes
    b_sib = (r.sum(-1, keepdims=True) - r) / (PB_K - 1)                        # (nb,B,m,k) sibling continuations
    A = np.zeros(r.shape + (TMAX,))
    A[..., :PB_T0] = ((r - b_prefix[..., None]) / prefix_div)[..., None]       # prefix tokens (t < T0)
    A[..., PB_T0:] = (r - b_sib)[..., None]                                    # continuation tokens
    nb = len(x)
    return A.reshape(nb, B, PB_M * PB_K, TMAX), r.reshape(nb, B, PB_M * PB_K), y.reshape(nb, B, PB_M * PB_K)


# ----------------------------------------------------------------------------- prefix-feature baseline fit
def fit_prefix_features(env, x, y, r, mean_g, chunk=2000):
    XtX = np.zeros((NF, NF)); Xty = np.zeros(NF); yy = 0.0; nrow = 0
    for c0 in range(0, len(x), chunk):
        sl = slice(c0, min(c0 + chunk, len(x)))
        xb, yb = x[sl], y[sl]
        PHI = env.PHI[xb[..., None], yb]                                       # (nb,B,n,T,F)
        alive = (np.arange(TMAX)[None, None, None, :] < env.L[yb][..., None])  # (nb,B,n,T)
        num = (PHI * alive[..., None]).sum(2, keepdims=True)
        cnt = alive.sum(2, keepdims=True)
        Xc = (PHI - num / np.maximum(cnt, 1)[..., None]) * alive[..., None]
        tgt = (r[sl] - mean_g[sl][..., None])[..., None] * alive
        X = Xc[alive]; Y = tgt[alive]
        XtX += X.T @ X; Xty += X.T @ Y; yy += (Y ** 2).sum(); nrow += len(Y)
    beta = np.linalg.solve(XtX + PF_RIDGE * np.eye(NF), Xty)
    r2 = (2 * beta @ Xty - beta @ XtX @ beta) / yy
    return beta, float(r2), nrow


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=int, default=200000, help='evaluation batches')
    ap.add_argument('--warmup', type=int, default=2000, help='warm-up batches (lagged statistics, ridge fit)')
    ap.add_argument('--seed', type=int, default=1, help='sampling seed (env seed is fixed at 0)')
    ap.add_argument('--env-seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()
    t0 = time.time()

    env = Env(args.env_seed)
    print(f'[env] K={K} prompts, {env.R} responses/prompt, D={D} params; |G|={np.linalg.norm(env.G):.4f}; '
          f'finite-difference max rel err = {env.grad_check(np.random.default_rng(123)):.2e}')
    for x in range(K):
        print(f'   prompt {x}: J_x={env.J_x[x]:6.2f}  sigma_x={env.sigma_x[x]:6.3f}  E[L]={env.EL[x]:.2f}  '
              f'corr(r,L)={env.corr_rL[x]:+.3f}  |G_x|={np.linalg.norm(env.G_blocks[x]):.4f}  '
              f'P(L=1,2,3)=({(env.P[x]*(env.L==1)).sum():.2f},{(env.P[x]*(env.L==2)).sum():.2f},{(env.P[x]*(env.L==3)).sum():.2f})')

    W, NB = args.warmup, args.nb
    NBT = W + NB
    rng = np.random.default_rng(args.seed)
    x_main, y_main = sample_main_stream(env, rng, NBT)
    r_main = env.Rtab[x_main[..., None], y_main]
    L_main = env.L[y_main]
    mean_g, var_g, loo_main = group_stats(r_main)
    nonflat = (var_g > 0).any(1)
    print(f'[sampling] {NBT} batches ({W} warm-up + {NB} eval) x {N} sequences; flat-group fraction '
          f'{(var_g == 0).mean():.3f}; {time.time()-t0:.1f}s')

    # ---- lagged global scale (rloo-lagged-scale, EMA variant): state before step t
    v_bar = ema_lagged(var_g.mean(1), LAG_S0 ** 2, LAG_BETA)
    c_lag = np.maximum(np.sqrt(v_bar), LAG_CMIN)
    # ---- per-prompt lagged scale (rloo-prompt-scale)
    SSx, Mx = exclusive_prompt_cumsums(x_main, var_g)
    v_x = (PS_K0 * v_bar[:, None] + SSx) / (PS_K0 + Mx)
    s_x = np.maximum(np.sqrt(v_x), PS_SMIN)
    W_ps = 1.0 / s_x                                                           # (NBT, K) weights before step t
    w_frozen = W_ps[W].copy()                                                  # frozen at the end of warm-up
    # ---- per-prompt historical mean baseline (rloo-hist-baseline)
    h_before, has_hist = hist_mean_before(x_main, mean_g, HIST_BETA)
    # ---- prefix-feature ridge fit on warm-up batches only
    beta_pf, r2_pf, nrow_pf = fit_prefix_features(env, x_main[:W], y_main[:W], r_main[:W], mean_g[:W])
    print(f'[prefix-feature] ridge beta={np.round(beta_pf, 3)} token-level within-group R^2={r2_pf:.3f} on {nrow_pf} rows')
    # ---- is-prompt-sampling stream (sequential: q_t depends on past draws)
    x_is, y_is, u_is, visits_is, inv_is = sample_is_stream(env, np.random.default_rng(args.seed + 1), NBT)
    print(f'[is-sampling] visit frequency per prompt (eval) = {np.round(visits_is / visits_is.sum(), 3)}, '
          f'invariant mean(u q K) = {inv_is[W:].mean():.6f}; {time.time()-t0:.1f}s')
    # ---- prefix-branching stream (stateless, eval batches only)
    x_pb, z_pb, y_pb = sample_pb_stream(env, np.random.default_rng(args.seed + 2), NB)
    # ---- lagged grad-norm scale: pass 1 computes the RAW rloo/smtsn gradient norm of every batch
    raw_norm = np.empty(NBT)
    for c0 in range(0, NBT, args.chunk):
        sl = slice(c0, min(c0 + args.chunk, NBT))
        Aw = (r_main[sl] - loo_main[sl]) * agg_w('seq-mean-token-sum-norm', L_main[sl])
        g = assemble_seq(Aw, env.S_seq[x_main[sl][..., None], y_main[sl]], onehot(x_main[sl]))
        raw_norm[sl] = np.linalg.norm(g, axis=1)
    Gbar = np.empty(NBT + 1); Gbar[0] = GN_INIT; a_gn = np.empty(NBT)
    rn = raw_norm.tolist(); nf = nonflat.tolist()
    for t in range(NBT):
        a_gn[t] = GN_TARGET / max(Gbar[t], GN_MIN)
        Gbar[t + 1] = GN_BETA * Gbar[t] + (1 - GN_BETA) * rn[t] if nf[t] else Gbar[t]
    print(f'[gradnorm] raw rloo norm median {np.median(raw_norm):.4f}, a_t range over eval '
          f'[{a_gn[W:].min():.4f}, {a_gn[W:].max():.4f}]; c_t (lagged EMA) range [{c_lag[W:].min():.2f}, {c_lag[W:].max():.2f}]; '
          f'{time.time()-t0:.1f}s')

    # ------------------------------------------------------------------ estimator table
    # (id, label, aggregation, stream, kind)
    specs = [
        ('ref-grpo-token-mean', 'GRPO (r-mean)/(std+1e-6) | token-mean', 'token-mean', 'main', 'seq'),
        ('ref-grpo-seq-token-mean', 'GRPO | seq-mean-token-mean (running legacy path)', 'seq-mean-token-mean', 'main', 'seq'),
        ('ref-grpo-smtsn', 'GRPO | seq-mean-token-sum-norm (ablation a)', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('ref-drgrpo-smtsn', 'Dr.GRPO raw (r-mean) | seq-mean-token-sum-norm', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('ref-rloo-token-mean', 'RLOO | token-mean (verl default agg)', 'token-mean', 'main', 'seq'),
        ('ref-rloo-seq-token-mean', 'RLOO | seq-mean-token-mean (ablation b)', 'seq-mean-token-mean', 'main', 'seq'),
        ('ref-opo-smtsn', 'OPO verl (self-inclusive L-weighted mean) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('ref-rpp-baseline-token-mean', 'reinforce_plus_plus_baseline (batch whitening) | token-mean', 'token-mean', 'main', 'seq'),
        ('rloo-const', 'RLOO / c=18 | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-lagged-scale', 'RLOO / c_t (lagged EMA of past within-group var) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-lagged-scale-logo', 'RLOO / c_{-g} (leave-one-group-out, stateless) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-prompt-scale', 'RLOO * w_t(x), lagged per-prompt scale (k0=2,s_min=5) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-prompt-scale-frozen', 'RLOO * w0(x), frozen per-prompt scale | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-prompt-scale-crossfit', 'RLOO / (s_{-i}+eps0=9), stateless cross-fit | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('is-prompt-sampling', 'x ~ q_t ∝ s(x) (past), A = u(x) RLOO, u=1/(K q_t) | smtsn', 'seq-mean-token-sum-norm', 'is', 'seq'),
        ('rloo-hist-baseline-k2', 'RLOO shrunk to past-visit EMA mean, k=2 | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('rloo-hist-baseline-k8', 'RLOO shrunk to past-visit EMA mean, k=8 | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('lagged-gradnorm-scale', 'a_t RLOO, a_t = g_target/EMA(past raw grad norms) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('prefix-branching', 'tree RLOO as stated: prefix tokens (r-b_{-a})/k, cont. tokens r-b_sib; m=4,k=2,T0=1 | smtsn', 'seq-mean-token-sum-norm', 'pb', 'tok'),
        ('prefix-branching-fixed', 'tree RLOO corrected: prefix tokens (r-b_{-a}) (no /k) | smtsn', 'seq-mean-token-sum-norm', 'pb', 'tok'),
        ('control-illegal-ownL', 'CONTROL (illegal): b_i = loo_i + 5*(L_i - mean_{-i} L_j) uses own length | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('opo-loo', 'LOO L-weighted baseline (u=L, alpha=1) | smtsn', 'seq-mean-token-sum-norm', 'main', 'seq'),
        ('otb-loo-position', 'position-level survival-conditioned LOO baseline (W=1) | smtsn', 'seq-mean-token-sum-norm', 'main', 'tok'),
        ('prefix-feature-baseline', 'LOO mean + ridge prefix-feature correction (past fit, clip 10) | smtsn', 'seq-mean-token-sum-norm', 'main', 'tok'),
    ]
    accs = {s[0]: Acc() for s in specs}
    xf_wsum = np.zeros(K); xf_cnt = np.zeros(K)       # cross-fit: realised E[1/(s_{-i}+eps0)] per prompt
    tarange = np.arange(TMAX)

    for c0 in range(W, NBT, args.chunk):
        sl = slice(c0, min(c0 + args.chunk, NBT))
        nb = sl.stop - sl.start
        x, y, r, Lb = x_main[sl], y_main[sl], r_main[sl], L_main[sl]
        mg, vg, loo = mean_g[sl], var_g[sl], loo_main[sl]
        sg = np.sqrt(vg)
        P1 = onehot(x)
        Sseq = env.S_seq[x[..., None], y]
        Stok = env.S_tok[x[..., None], y]
        Wtm, Wstm, Wsn = agg_w('token-mean', Lb), agg_w('seq-mean-token-mean', Lb), agg_w('seq-mean-token-sum-norm', Lb)
        A = {}
        # ---- references
        A['ref-grpo-token-mean'] = (r - mg[..., None]) / (sg[..., None] + 1e-6)
        A['ref-grpo-seq-token-mean'] = A['ref-grpo-token-mean']
        A['ref-grpo-smtsn'] = A['ref-grpo-token-mean']
        A['ref-drgrpo-smtsn'] = r - mg[..., None]
        rloo = r - loo
        A['ref-rloo-token-mean'] = rloo
        A['ref-rloo-seq-token-mean'] = rloo
        A['ref-opo-smtsn'] = r - ((Lb * r).sum(-1) / Lb.sum(-1))[..., None]
        a0 = r - mg[..., None]
        Ltot = Lb.sum((1, 2))
        mu = (a0 * Lb).sum((1, 2)) / Ltot
        var_w = ((a0 - mu[:, None, None]) ** 2 * Lb).sum((1, 2)) / Ltot * (Ltot / (Ltot - 1))
        A['ref-rpp-baseline-token-mean'] = (a0 - mu[:, None, None]) / np.sqrt(var_w + 1e-8)[:, None, None]
        # ---- ideas
        A['rloo-const'] = rloo / C_FIXED
        A['rloo-lagged-scale'] = rloo / c_lag[sl][:, None, None]
        SSg = vg * (n - 1)
        c_logo = np.maximum(np.sqrt((SSg.sum(1, keepdims=True) - SSg) / (N - n)), LAG_CMIN)
        A['rloo-lagged-scale-logo'] = rloo / c_logo[..., None]
        A['rloo-prompt-scale'] = rloo * W_ps[sl][np.arange(nb)[:, None], x][..., None]
        A['rloo-prompt-scale-frozen'] = rloo * w_frozen[x][..., None]
        S2 = (r ** 2).sum(-1, keepdims=True)
        var_loo = np.maximum(((S2 - r ** 2) - (n - 1) * loo ** 2) / (n - 2), 0.0)
        wxf = 1.0 / (np.sqrt(var_loo) + XF_EPS0)
        A['rloo-prompt-scale-crossfit'] = rloo * wxf
        np.add.at(xf_wsum, x.ravel(), wxf.sum(-1).ravel()); np.add.at(xf_cnt, x.ravel(), n)
        for kk in HIST_K_LIST:
            hb = h_before[sl]; hh = has_hist[sl]
            b = np.where(hh[..., None], ((n - 1) * loo + kk * hb[..., None]) / (n - 1 + kk), loo)
            A[f'rloo-hist-baseline-k{int(kk)}'] = (r - b) / C_FIXED
        A['lagged-gradnorm-scale'] = rloo * a_gn[sl][:, None, None]
        Lr = (Lb * r).sum(-1, keepdims=True)
        b_opo_loo = (Lr - Lb * r) / (Lb.sum(-1, keepdims=True) - Lb)
        A['opo-loo'] = (r - b_opo_loo) / C_FIXED
        # position-level survival-conditioned LOO baseline
        alive = (tarange[None, None, None, :] < Lb[..., None])                 # (nb,B,n,T)
        num = (alive * r[..., None]).sum(2, keepdims=True) - alive * r[..., None]
        den = alive.sum(2, keepdims=True) - alive
        b_pos = np.where(den > 0, num / np.maximum(den, 1), b_opo_loo[..., None])
        A['otb-loo-position'] = (r[..., None] - b_pos) / C_FIXED * alive
        # prefix-feature baseline
        PHI = env.PHI[x[..., None], y]                                         # (nb,B,n,T,F)
        numf = (PHI * alive[..., None]).sum(2, keepdims=True) - PHI * alive[..., None]
        cntf = den
        corr = np.clip((PHI - numf / np.maximum(cntf, 1)[..., None]) @ beta_pf, -PF_SCAP, PF_SCAP) * (cntf > 0)
        A['prefix-feature-baseline'] = (r[..., None] - (loo[..., None] + corr)) / C_FIXED * alive
        # ---- other streams
        xi, yi = x_is[sl].astype(int), y_is[sl]
        ri = env.Rtab[xi[..., None], yi]; Li = env.L[yi]
        _, _, looi = group_stats(ri)
        A['is-prompt-sampling'] = (ri - looi) * u_is[sl][..., None]
        sl_pb = slice(c0 - W, sl.stop - W)
        xp = x_pb[sl_pb]
        Apb, rp, yp = pb_token_advantages(env, xp, y_pb[sl_pb])
        A['prefix-branching'] = Apb / C_FIXED
        A['prefix-branching-fixed'] = pb_token_advantages(env, xp, y_pb[sl_pb], prefix_div=1)[0] / C_FIXED
        # control: a baseline that uses the sample's OWN length (future information) -> must show up as biased
        A['control-illegal-ownL'] = (r - (loo + 5.0 * (Lb - (Lb.sum(-1, keepdims=True) - Lb) / (n - 1)))) / C_FIXED

        for sid, label, mode, stream, kind in specs:
            if stream == 'main':
                Wm = {'token-mean': Wtm, 'seq-mean-token-mean': Wstm, 'seq-mean-token-sum-norm': Wsn}[mode]
                if kind == 'seq':
                    g = assemble_seq(A[sid] * Wm, Sseq, P1)
                else:
                    g = assemble_tok(A[sid] * Wm[..., None], Stok, P1)
            elif stream == 'is':
                Wm = agg_w(mode, Li)
                g = assemble_seq(A[sid] * Wm, env.S_seq[xi[..., None], yi], onehot(xi))
            else:
                Wm = agg_w(mode, env.L[yp])
                g = assemble_tok(A[sid] * Wm[..., None], env.S_tok[xp[..., None], yp], onehot(xp))
            accs[sid].add(g, env.G)
        if ((c0 - W) // args.chunk) % 50 == 0:
            print(f'   ... {sl.stop - W}/{NB} eval batches, {time.time()-t0:.0f}s', flush=True)

    # ------------------------------------------------------------------ claimed weights for reweighted ideas
    wbar_lag = W_ps[W:].mean(0)                       # average lagged per-prompt weight over eval steps
    wbar_xf = xf_wsum / np.maximum(xf_cnt, 1)         # realised cross-fit weight per prompt
    nonflat_x = np.linalg.norm(env.G_blocks, axis=1) > 0
    def _norm(w):
        w = np.where(nonflat_x, w, np.nan)
        return w / np.nanmean(w)
    claimed = {
        'rloo-prompt-scale': _norm(wbar_lag),
        'rloo-prompt-scale-frozen': _norm(w_frozen),
        'rloo-prompt-scale-crossfit': _norm(wbar_xf),
    }

    # ------------------------------------------------------------------ report
    res = {sid: accs[sid].summary(env) for sid, *_ in specs}
    ref_var = res['rloo-const']['trvar_norm']
    ref_var_tm = res['ref-rloo-token-mean']['trvar_norm']
    print('\n' + '=' * 160)
    print(f'RESULTS  (eval batches = {NB}, sampling seed = {args.seed}; |G| = {np.linalg.norm(env.G):.4f}; '
          f'relative variance is trace Var(g)/kappa^2 relative to RLOO|smtsn (= rloo-const))')
    print('=' * 160)
    hdr = f"{'id':30s} {'kappa':>10s} {'cos':>9s} {'rel.bias':>9s} {'floor':>8s} {'z':>7s} {'var/RLOO':>9s}  " \
          f"{'block ratio |E g_x|/(k|G_x|) per prompt (p2 flat: z)':58s} {'block cos':40s}"
    print(hdr)
    for sid, label, mode, stream, kind in specs:
        s = res[sid]
        br = ' '.join(('%6.3f' % b['ratio']) if b['ratio'] is not None else ('z%4.1f' % b['z']) for b in s['blocks'])
        bc = ' '.join(('%6.3f' % b['cos']) if b['cos'] is not None else '   -  ' for b in s['blocks'])
        print(f"{sid:30s} {s['kappa']:10.3e} {s['cos']:9.5f} {s['resid']:9.4f} {s['floor']:8.4f} {s['z']:7.2f} "
              f"{s['trvar_norm']/ref_var:9.3f}  {br:58s} {bc:40s}")
    print('\nper-prompt z of block direction (~1 if E[g]_x ∝ G_x):')
    for sid, *_ in specs:
        print(f"   {sid:30s} " + ' '.join('%6.2f' % b['z'] for b in res[sid]['blocks']))
    print('\nper-prompt 95% CI half-width of the block ratio (p2 flat: |E g_x| / (kappa |G|)):')
    for sid, *_ in specs:
        print(f"   {sid:30s} " + ' '.join(('%6.3f' % b['ci']) if b['ci'] is not None else ('%6.1e' % b['norm_over_kG']) for b in res[sid]['blocks']))
    print('\nclaimed w(x) (normalised over non-flat prompts) vs realised block ratio (normalised) for the reweighted ideas:')
    for sid, w in claimed.items():
        ratios = np.array([b['ratio'] if b['ratio'] is not None else np.nan for b in res[sid]['blocks']])
        rn_ = ratios / np.nanmean(ratios)
        print(f"   {sid:30s} claimed {np.round(w, 3)}  realised {np.round(rn_, 3)}  max|diff| {np.nanmax(np.abs(rn_ - w)):.3f}")
    ci_prompt0 = [res[sid]['blocks'][0]['ci'] for sid, *_ in specs]
    print(f"\nRLOO|token-mean normalised variance / RLOO|smtsn = {ref_var_tm/ref_var:.4f}")
    print(f"sigma_x = {np.round(env.sigma_x, 3)}; 1/max(sigma_x, s_min) normalised = "
          f"{np.round((1/np.maximum(env.sigma_x, PS_SMIN)) / (1/np.maximum(env.sigma_x, PS_SMIN)).mean(), 3)}")
    print(f'total time {time.time()-t0:.0f}s')

    if args.json:
        out = dict(nb=NB, warmup=W, seed=args.seed, env_seed=args.env_seed, G_norm=float(np.linalg.norm(env.G)),
                   sigma_x=env.sigma_x.tolist(), corr_rL=env.corr_rL.tolist(), EL=env.EL.tolist(), J_x=env.J_x.tolist(),
                   r2_prefix_features=r2_pf, beta_pf=beta_pf.tolist(),
                   claimed_w={k: v.tolist() for k, v in claimed.items()},
                   ref_var_rloo_smtsn=ref_var, ref_var_rloo_token_mean=ref_var_tm,
                   results={sid: dict(label=label, agg=mode, **res[sid]) for sid, label, mode, *_ in specs})
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
