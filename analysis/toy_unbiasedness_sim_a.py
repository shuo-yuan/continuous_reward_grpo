#!/usr/bin/env python
"""
toy_unbiasedness_sim_a.py -- Variant A toy harness: is a given advantage estimator
(+ verl loss aggregation) an unbiased score-function estimator of grad J?

TOY (explicit token model; K=4 prompts, M=6 "approaches" per prompt)
----------------------------------------------------------------------
Policy: for prompt x every token is drawn i.i.d. from pi_x = softmax(theta[x, :]) over
M=6 symbols (one parameter block per prompt, shared across all positions).
  token 1 = "approach" a  : fixes the base reward R(x,a) and the response length L(x,a)
                            (the stop rule is a function of the prefix, no parameters);
  token 2 = "detail"   b  : adds a small modifier D(x,a,b) (0 on the near-flat and the
                            all-equal prompt), so the reward is deterministic given the
                            response and the prefix-branching sibling baseline is non-trivial;
  tokens 3..L           : fillers; their score terms (e_y - pi) are zero-mean given the
                            prefix and act like the gradient noise of long responses.
Sequence log-prob = sum_t log pi_x(y_t); its gradient (the token-sum of scores that verl's
token-sum aggregations build) is  S = sum_t (e_{y_t} - pi_x) = count - L*pi_x   in block x.
J = (1/K) sum_x J_x,  J_x = sum_{a,b} pi(a) pi(b) [R(x,a)+D(x,a,b)]  (x uniform over K);
grad_c J_x = pi(c) [(Rbar1(c) - J_x) + (Rbar2(c) - J_x)]  (closed form, checked by FD).
Aggregations (verl agg_loss, ratio==1):
  seq-mean-token-sum-norm (ssn): g = sum_i A_i S_i / (N * S_CONST),  S_CONST = L_max (constant)
  token-mean (tm)              : g = sum_i A_i S_i / sum_i L_i          (sample-dependent denom.)
  seq-mean-token-mean (stm)    : g = (1/N) sum_i A_i S_i / L_i         (legacy micro-batch=1 path)
Monte Carlo: R independent chains x T steps (burn-in for lagged statistics), B=8 groups x
n=8 samples per step; prompts drawn uniformly with replacement (= E_x) except for the
importance-sampling idea.  E[g] is estimated with the filler tokens' zero-mean scores
integrated out analytically (Rao-Blackwell over tokens 3..L) wherever the estimator's weights
provably do not depend on the sample's own tokens beyond (y1, y2, L, r); the fully explicit
gradient g_full (all tokens) is used for the variance, for the gradient-norm feedback, for
the controls that depend on it, and as a cross-check printed for every estimator.
Variance is compared after dividing by each estimator's fitted kappa.
Finding worth flagging: the prefix-branching idea as written (prefix-token advantage divided by k)
is biased in this toy -- a tree has only m distinct prefix draws against m*k continuation draws,
so the 1/k under-weights the prefix gradient by exactly k; dropping the 1/k makes it unbiased.
"""
import argparse
import json
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------- toy spec
K, M = 4, 6
B, n = 8, 8
N = B * n
Lmax = 12
S_CONST = float(Lmax)
C_FIXED = 18.0

R_BASE = np.array(
    [
        [12.0, 30.0, 48.0, 60.0, 77.0, 95.0],  # spread ~25
        [40.0, 46.0, 52.0, 55.0, 62.0, 70.0],  # spread ~10
        [50.0, 50.0, 50.0, 50.15, 50.0, 50.0],  # near-flat
        [60.0] * 6,  # all-equal
    ]
)
L_MAT = np.array(
    [
        [9, 4, 12, 3, 7, 10],
        [5, 8, 3, 11, 6, 9],
        [6, 6, 10, 2, 8, 4],
        [3, 7, 5, 9, 11, 4],
    ]
)
DVEC = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
DCOEF = np.array([1.0, 0.5, 1.0, 0.8, 1.2, 0.6])
DSCALE = np.array([3.0, 2.0, 0.0, 0.0])
D_MOD = DSCALE[:, None, None] * DCOEF[None, :, None] * DVEC[None, None, :]  # [K, a, b]
R_TOT = R_BASE[:, :, None] + D_MOD  # [K, a, b]
assert R_TOT.min() >= 0 and R_TOT.max() <= 100
assert L_MAT.min() >= 2 and L_MAT.max() <= Lmax


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


THETA = np.random.default_rng(12345).normal(0.0, 0.8, (K, M))
PI = softmax(THETA)
LOGPI = np.log(PI)
PI_SQ = (PI**2).sum(-1)  # ||pi_x||^2


def exact_grad():
    """grad of J = (1/K) sum_x J_x, blocks [K, M]; also per-prompt J_x and reward std."""
    Rbar1 = (R_TOT * PI[:, None, :]).sum(-1)  # E[r | a]   [K, M]
    Rbar2 = (R_TOT * PI[:, :, None]).sum(1)  # E[r | b]   [K, M]
    Jx = (PI * Rbar1).sum(-1)  # [K]
    gJx = PI * ((Rbar1 - Jx[:, None]) + (Rbar2 - Jx[:, None]))
    var_x = (PI[:, :, None] * PI[:, None, :] * (R_TOT - Jx[:, None, None]) ** 2).sum((1, 2))
    return gJx / K, Jx, np.sqrt(var_x)


def fd_check(gJ, h=1e-6):
    def J(theta):
        p = softmax(theta)
        return np.mean((p[:, :, None] * p[:, None, :] * R_TOT).sum((1, 2)))

    num = np.zeros_like(THETA)
    for x in range(K):
        for a in range(M):
            e = np.zeros_like(THETA)
            e[x, a] = h
            num[x, a] = (J(THETA + e) - J(THETA - e)) / (2 * h)
    return np.abs(num - gJ).max() / np.abs(gJ).max()


GRAD_J, J_X, SIGMA_X = exact_grad()
GRAD_J_FLAT = GRAD_J.ravel()
NORM_GJ = np.linalg.norm(GRAD_J_FLAT)


# ----------------------------------------------------------------------------- sampling
def sample_cat(rng, p, extra=()):
    """p: [..., M] probabilities. Returns int indices of shape p.shape[:-1] + extra."""
    C = p.shape[-1]
    cdf = np.cumsum(p, -1)
    u = rng.random(p.shape[:-1] + tuple(extra))
    if extra:
        cdf = cdf.reshape(p.shape[:-1] + (1,) * len(extra) + (C,))
    idx = (u[..., None] > cdf).sum(-1)
    return np.minimum(idx, C - 1)


def onehot(y):
    return np.eye(M)[y]


def draw_batch(rng, xidx, branch=None):
    """xidx: [R, B] prompt ids. branch=(m, k): m prefixes (token 1) per group, each with k
    children (consecutive in the group) that share the prefix and draw their own token 2 + fillers."""
    Rn = xidx.shape[0]
    x = np.repeat(xidx, n, axis=1)  # [R, N]
    px = PI[x]  # [R, N, M]
    if branch is None:
        y1 = sample_cat(rng, px)
    else:
        m, k = branch
        y1p = sample_cat(rng, PI[np.repeat(xidx, m, axis=1)])  # [R, B*m]
        y1 = np.repeat(y1p, k, axis=1)  # [R, N]
    y2 = sample_cat(rng, px)
    L = L_MAT[x, y1]
    fill = sample_cat(rng, px, (Lmax - 2,))  # [R, N, Lmax-2]
    tok = np.concatenate([y1[..., None], y2[..., None], fill], -1)  # [R, N, Lmax]
    mask = np.arange(Lmax)[None, None, :] < L[..., None]
    r = R_TOT[x, y1, y2]
    count = np.stack([((tok == c) & mask).sum(-1) for c in range(M)], -1).astype(float)
    S_full = count - L[..., None] * px  # exact sequence score (all tokens)
    S_rb = onehot(y1) + onehot(y2) - 2.0 * px  # E[S_full | y1, y2]
    rg = r.reshape(Rn, B, n)
    Lg = L.reshape(Rn, B, n)
    mean_g = rg.mean(-1)
    loo = (rg.sum(-1, keepdims=True) - rg) / (n - 1)
    std_g = np.sqrt(((rg - mean_g[..., None]) ** 2).sum(-1) / (n - 1))
    return dict(
        R=Rn, xidx=xidx, x=x, y1=y1, y2=y2, L=L, tok=tok, mask=mask, r=r, count=count,
        S_full=S_full, S_rb=S_rb, rg=rg, Lg=Lg, mean_g=mean_g, loo=loo, std_g=std_g,
        flat=std_g < 1e-12, rows=np.arange(Rn)[:, None],
    )


# ----------------------------------------------------------------------------- gradient assembly
def g_from_seq(w, batch, S):
    """w: [R, N] per-sequence weight (advantage x aggregation). Returns [R, K, M]."""
    Rn = batch["R"]
    idx = (np.arange(Rn)[:, None] * K + batch["x"]).ravel()
    out = np.empty((Rn * K, M))
    for c in range(M):
        out[:, c] = np.bincount(idx, weights=(w * S[..., c]).ravel(), minlength=Rn * K)
    return out.reshape(Rn, K, M)


def g_from_tok(W, batch, rb):
    """W: [R, N, Lmax] per-token weight. sum_{i,t} W mask (e_{y_it} - pi_x). rb -> tokens 1,2 only."""
    Rn = batch["R"]
    Wm = W * batch["mask"]
    if rb:
        Wm = Wm.copy()
        Wm[:, :, 2:] = 0.0
    idx = ((np.arange(Rn)[:, None, None] * K + batch["x"][..., None]) * M + batch["tok"]).ravel()
    pos = np.bincount(idx, weights=Wm.ravel(), minlength=Rn * K * M).reshape(Rn, K, M)
    tot = Wm.sum(-1)
    idx2 = (np.arange(Rn)[:, None] * K + batch["x"]).ravel()
    sub = np.bincount(idx2, weights=tot.ravel(), minlength=Rn * K).reshape(Rn, K)
    return pos - sub[..., None] * PI[None]


def agg(A, batch, mode):
    """Advantage [R, N] -> per-sequence gradient weight under a verl loss_agg_mode."""
    L = batch["L"]
    if mode == "ssn":
        return A / (N * S_CONST)
    if mode == "tm":
        return A / L.sum(1, keepdims=True)
    if mode == "stm":
        return A / (N * L)
    if mode == "sts":
        return A / N
    raise ValueError(mode)


# ----------------------------------------------------------------------------- estimators
class Est:
    """weights(batch, t) -> (w_seq [R,N] | None, W_tok [R,N,Lmax] | None, tok_rb_ok)
    update(batch, g_full) runs AFTER the step's gradient is formed (lagged statistics)."""

    batch_kind = "U"  # U uniform prompts | Q importance-sampled | P42 / P24 prefix-branching
    claims = "J"  # J | Jw | none   (what the idea claims to be unbiased for)
    seq_rb_ok = True
    post_norm = None  # if set: gradient is rescaled by g_target/||g_full|| (NGD control)
    expect = "unbiased"

    def __init__(self, name, impl):
        self.name, self.impl = name, impl

    def init_state(self, Rn):
        pass

    def update(self, batch, g_full, t):
        pass

    def claimed_w(self):
        return None


class RefEst(Est):
    def __init__(self, name, kind, mode, impl, expect):
        super().__init__(name, impl)
        self.kind, self.mode, self.expect = kind, mode, expect
        self.claims = "J"

    def weights(self, batch, t):
        rg, Lg, mean_g, std_g, loo = batch["rg"], batch["Lg"], batch["mean_g"], batch["std_g"], batch["loo"]
        if self.kind == "grpo":
            A = (rg - mean_g[..., None]) / (std_g[..., None] + 1e-6)
        elif self.kind == "drgrpo":
            A = rg - mean_g[..., None]
        elif self.kind == "rloo":
            A = rg - loo  # = n/(n-1) (r - mean_g)
        elif self.kind == "opo":
            A = rg - ((Lg * rg).sum(-1) / Lg.sum(-1))[..., None]
        elif self.kind == "rpp":
            a = (rg - mean_g[..., None]).reshape(batch["R"], N)
            L = batch["L"].astype(float)
            T = L.sum(1, keepdims=True)
            mu = (L * a).sum(1, keepdims=True) / T
            var = (L * (a - mu) ** 2).sum(1, keepdims=True) / T * (T / (T - 1))
            A = (a - mu) / np.sqrt(var + 1e-8)
        else:
            raise ValueError(self.kind)
        return agg(A.reshape(batch["R"], N), batch, self.mode), None, True


class RlooConst(Est):
    def __init__(self, name, c=C_FIXED, centering="loo", mode="ssn"):
        super().__init__(name, f"A=(r-{'mean_-i' if centering == 'loo' else 'mean_g'})/{c}; {mode}")
        self.c, self.centering, self.mode = c, centering, mode

    def weights(self, batch, t):
        dev = batch["rg"] - (batch["loo"] if self.centering == "loo" else batch["mean_g"][..., None])
        return agg((dev / self.c).reshape(batch["R"], N), batch, self.mode), None, True


class LaggedScale(Est):
    """c_t = max(sqrt(v_bar_{t-1}), c_min); v_bar = EMA of past steps' mean s_g^2. leak=True
    updates v_bar with the CURRENT step before use (control). beta=0 with leak: current-batch std."""

    def __init__(self, name, beta=0.9, cmin=3.0, s0=C_FIXED, leak=False):
        super().__init__(name, f"A=(r-mean_-i)/c_t, c_t=max(sqrt(vbar),{cmin}), vbar EMA beta={beta} "
                               f"of mean_g s_g^2 ({'current step INCLUDED' if leak else 'past steps only'}); ssn")
        self.beta, self.cmin, self.s0, self.leak = beta, cmin, s0, leak
        self.expect = "biased (control)" if leak else "unbiased"

    def init_state(self, Rn):
        self.vbar = np.full(Rn, self.s0**2)

    def weights(self, batch, t):
        vnow = (batch["std_g"] ** 2).mean(-1)
        if self.leak:
            self.vbar = self.beta * self.vbar + (1 - self.beta) * vnow
        c = np.maximum(np.sqrt(self.vbar), self.cmin)
        self.c_t = c
        A = (batch["rg"] - batch["loo"]) / c[:, None, None]
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def update(self, batch, g_full, t):
        if not self.leak:
            vnow = (batch["std_g"] ** 2).mean(-1)
            self.vbar = self.beta * self.vbar + (1 - self.beta) * vnow


class LeaveOneGroupOutScale(Est):
    def __init__(self, name, cmin=3.0):
        super().__init__(name, f"A=(r-mean_-i)/c_-g, c_-g=max({cmin}, sqrt((SS_tot-SS_g)/(N-n))) from the OTHER 7 groups; ssn")
        self.cmin = cmin

    def weights(self, batch, t):
        SS = ((batch["rg"] - batch["mean_g"][..., None]) ** 2).sum(-1)  # [R, B]
        c2 = (SS.sum(-1, keepdims=True) - SS) / (N - n)
        c = np.maximum(self.cmin, np.sqrt(c2))
        A = (batch["rg"] - batch["loo"]) / c[..., None]
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True


class SpreadHistory:
    """Per-prompt history of past within-group variances + global lagged EMA (shared machinery of
    rloo-prompt-scale and is-prompt-sampling). s_x = max(sqrt((k0 vbar + sum s^2)/(k0+m_x)), s_min)."""

    def __init__(self, Rn, k0=2.0, smin=5.0, beta=0.9, s0=C_FIXED):
        self.k0, self.smin, self.beta = k0, smin, beta
        self.ssum = np.zeros((Rn, K))
        self.cnt = np.zeros((Rn, K))
        self.vbar = np.full(Rn, s0**2)

    def s_x(self):
        vx = (self.k0 * self.vbar[:, None] + self.ssum) / (self.k0 + self.cnt)
        return np.maximum(np.sqrt(vx), self.smin)  # [R, K]

    def update(self, batch):
        rows = batch["rows"]
        np.add.at(self.ssum, (np.broadcast_to(rows, batch["xidx"].shape), batch["xidx"]), batch["std_g"] ** 2)
        np.add.at(self.cnt, (np.broadcast_to(rows, batch["xidx"].shape), batch["xidx"]), 1.0)
        self.vbar = self.beta * self.vbar + (1 - self.beta) * (batch["std_g"] ** 2).mean(-1)


class PromptScaleLagged(Est):
    claims = "Jw"

    def __init__(self, name, k0=2.0, smin=5.0, beta=0.9):
        super().__init__(name, f"A=(r-mean_-i)/s_x, s_x=max(sqrt((k0 vbar+sum_past s_g^2)/(k0+m_x)),{smin}), k0={k0}, past visits only; ssn")
        self.k0, self.smin, self.beta = k0, smin, beta

    def init_state(self, Rn):
        self.hist = SpreadHistory(Rn, self.k0, self.smin, self.beta)
        self.wsum = np.zeros(K)
        self.wcnt = 0

    def weights(self, batch, t):
        w = 1.0 / self.hist.s_x()  # [R, K]
        self._w = w
        A = (batch["rg"] - batch["loo"]) * w[batch["rows"], batch["xidx"]][..., None]
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def update(self, batch, g_full, t):
        if t >= BURN:
            self.wsum += self._w.mean(0)
            self.wcnt += 1
        self.hist.update(batch)

    def claimed_w(self):
        return self.wsum / max(self.wcnt, 1)


class PromptScaleFrozen(Est):
    claims = "Jw"

    def __init__(self, name, smin=5.0):
        super().__init__(name, f"A=(r-mean_-i)*w(x), w(x)=1/max(sigma_x(base policy),{smin}) FIXED; ssn")
        self.w = 1.0 / np.maximum(SIGMA_X, smin)

    def weights(self, batch, t):
        A = (batch["rg"] - batch["loo"]) * self.w[batch["xidx"]][..., None]
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def claimed_w(self):
        return self.w


class CrossFitScale(Est):
    claims = "none"
    expect = "direction exact per prompt, theta-dependent cross-prompt weight"

    def __init__(self, name, eps0=5.0):
        super().__init__(name, f"A=(r_i-mean_-i)/(s_-i+{eps0}), s_-i = ddof=1 std of the OTHER 7 rewards; ssn")
        self.eps0 = eps0

    def init_state(self, Rn):
        self.wsum = np.zeros(K)
        self.wcnt = 0

    def weights(self, batch, t):
        rg, loo = batch["rg"], batch["loo"]
        S2 = (rg**2).sum(-1, keepdims=True)
        var = ((S2 - rg**2) - (n - 1) * loo**2) / (n - 2)
        s = np.sqrt(np.maximum(var, 0.0))
        wi = 1.0 / (s + self.eps0)
        self._wi = wi
        A = (rg - loo) * wi
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def update(self, batch, g_full, t):
        if t >= BURN:  # realized mean weight per prompt (the implied theta-dependent w(x))
            np.add.at(self.wsum, batch["xidx"].ravel(), self._wi.mean(-1).ravel())
            cnt = np.bincount(batch["xidx"].ravel(), minlength=K)
            self._cnt = getattr(self, "_cnt", np.zeros(K)) + cnt

    def claimed_w(self):
        return self.wsum / np.maximum(getattr(self, "_cnt", np.ones(K)), 1)


class ISSampler:
    """Draws the B prompts of a step from q_t(x) = s(x)/Z with s from PAST data; records u(x)=1/(K q)."""

    def __init__(self, Rn, k0=2.0, smin=5.0, beta=0.9):
        self.hist = SpreadHistory(Rn, k0, smin, beta)
        self.qsum = np.zeros(K)
        self.qcnt = 0

    def draw(self, rng, t):
        s = self.hist.s_x()
        q = s / s.sum(-1, keepdims=True)  # [R, K]
        xidx = sample_cat(rng, q, (B,))  # [R, B]
        if t >= BURN:
            self.qsum += q.mean(0)
            self.qcnt += 1
        return xidx, q

    def update(self, batch):
        self.hist.update(batch)


class ISEst(Est):
    batch_kind = "Q"

    def __init__(self, name, correct=True):
        super().__init__(name, "prompts ~ q_t(x) ∝ s(x) (past data, floored); A=" + ("u(x)(r-mean_-i), u=1/(K q_t(x))" if correct else "(r-mean_-i) [no 1/q correction]") + "; ssn")
        self.correct = correct
        self.claims = "J" if correct else "Jw"

    def weights(self, batch, t):
        dev = batch["rg"] - batch["loo"]
        if self.correct:
            u = 1.0 / (K * batch["q"][batch["rows"], batch["xidx"]])
            dev = dev * u[..., None]
        return agg(dev.reshape(batch["R"], N), batch, "ssn"), None, True

    def claimed_w(self):
        if self.correct:
            return np.ones(K)
        return K * SAMPLER.qsum / max(SAMPLER.qcnt, 1)


class HistBaseline(Est):
    """b_i = ((n-1) m_-i + k h(x)) / (n-1+k), h(x) = EMA(beta) of PAST visits' group means."""

    def __init__(self, name, k=2.0, beta=0.5, c=C_FIXED, adaptive=False, leak=False):
        kd = "k=sigma2_within/tau2_drift from past steps (clip [0.1,10])" if adaptive else f"k={k}"
        super().__init__(name, f"A=(r-b)/{c}, b=((n-1)mean_-i + k h(x))/(n-1+k), h=EMA({beta}) of {'PAST' if not leak else 'past+CURRENT (control)'} group means, {kd}; ssn")
        self.k, self.beta, self.c, self.adaptive, self.leak = k, beta, c, adaptive, leak
        self.expect = "unbiased" if not leak else "scale-only self term (control)"

    def init_state(self, Rn):
        self.h = np.zeros((Rn, K))
        self.cnt = np.zeros((Rn, K))
        self.ssig, self.nsig, self.stau, self.ntau = 0.0, 0.0, 0.0, 0.0

    def _k(self, Rn):
        if not self.adaptive or self.ntau == 0:
            return np.full(Rn, 2.0 if self.adaptive else self.k)
        return np.full(Rn, float(np.clip((self.ssig / self.nsig) / (self.stau / self.ntau), 0.1, 10.0)))

    def _apply_update(self, batch):
        rows = batch["rows"][:, 0]
        for g in range(B):
            xg = batch["xidx"][:, g]
            mg = batch["mean_g"][:, g]
            has = self.cnt[rows, xg] > 0
            if self.adaptive:
                self.stau += float(((mg - self.h[rows, xg]) ** 2)[has].sum())
                self.ntau += float(has.sum())
                self.ssig += float((batch["std_g"][:, g] ** 2).sum())
                self.nsig += len(rows)
            self.h[rows, xg] = np.where(has, self.beta * self.h[rows, xg] + (1 - self.beta) * mg, mg)
            self.cnt[rows, xg] += 1

    def weights(self, batch, t):
        if self.leak:
            self._apply_update(batch)
        rows, xidx = batch["rows"], batch["xidx"]
        hx = self.h[rows, xidx]
        has = self.cnt[rows, xidx] > 0
        kk = self._k(batch["R"])[:, None]
        self.k_used = kk
        b = np.where(has[..., None], ((n - 1) * batch["loo"] + kk[..., None] * hx[..., None]) / (n - 1 + kk[..., None]), batch["loo"])
        A = (batch["rg"] - b) / self.c
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def update(self, batch, g_full, t):
        if not self.leak:
            self._apply_update(batch)


class LaggedGradNorm(Est):
    def __init__(self, name, beta=0.9, g_target=0.25, g_min=1e-3, G0=0.4):
        super().__init__(name, f"A=a_t(r-mean_-i), a_t=g_target/max(Gbar_(t-1),{g_min}), Gbar=EMA({beta}) of PAST steps' raw ||g|| (=||g_t||/a_t, skipped on all-flat steps), g_target={g_target}; ssn")
        self.beta, self.g_target, self.g_min, self.G0 = beta, g_target, g_min, G0

    def init_state(self, Rn):
        self.Gbar = np.full(Rn, self.G0)

    def weights(self, batch, t):
        self.a_t = self.g_target / np.maximum(self.Gbar, self.g_min)
        A = self.a_t[:, None, None] * (batch["rg"] - batch["loo"])
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True

    def update(self, batch, g_full, t):
        raw = np.linalg.norm(g_full.reshape(batch["R"], -1), axis=1) / self.a_t
        nonflat = (~batch["flat"]).any(-1)
        self.Gbar = np.where(nonflat, self.beta * self.Gbar + (1 - self.beta) * raw, self.Gbar)


class CurrentBatchNGD(Est):
    """Control: normalized-gradient descent with the CURRENT batch's own norm (what frequent clipping becomes)."""
    seq_rb_ok = False
    claims = "none"
    expect = "biased (control)"

    def __init__(self, name, g_target=0.25):
        super().__init__(name, f"g = {g_target} * g_rloo / ||g_rloo|| (own-batch norm); ssn")
        self.post_norm = g_target

    def weights(self, batch, t):
        A = batch["rg"] - batch["loo"]
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True


class PrefixBranching(Est):
    def __init__(self, name, m, k, c=C_FIXED, fix=True):
        super().__init__(name, f"m={m} prefixes (token 1) x k={k} children; prefix-token A=(r_ab-b_-a){'/k' if fix else ' [no 1/k]'}, "
                               f"b_-a=mean of other prefixes' rbar; continuation-token A=r_ab-b_(a,-b) (sibling LOO); /{c}; ssn")
        self.m, self.k, self.c, self.fix = m, k, c, fix
        self.batch_kind = f"P{m}{k}"
        # A tree has m distinct prefix draws but m*k continuation draws: E[sum] = m*grad_prefix + m*k*grad_cont
        # with the 1/k (prefix part under-weighted by k), and m*k*(grad_prefix + grad_cont) = n*gradJ_x without it.
        self.expect = "biased (prefix part under-weighted by k)" if fix else "unbiased"

    def weights(self, batch, t):
        Rn = batch["R"]
        rg = batch["rg"].reshape(Rn, B, self.m, self.k)
        rbar = rg.mean(-1)  # [R, B, m]
        b_pre = (rbar.sum(-1, keepdims=True) - rbar) / (self.m - 1)
        b_sib = (rg.sum(-1, keepdims=True) - rg) / (self.k - 1)
        A_pre = (rg - b_pre[..., None]) / (self.k if self.fix else 1.0)
        A_con = rg - b_sib
        W = np.empty((Rn, N, Lmax))
        W[:, :, 0] = A_pre.reshape(Rn, N)
        W[:, :, 1:] = A_con.reshape(Rn, N)[..., None]
        W /= self.c * N * S_CONST
        return None, W, True


class OpoLoo(Est):
    def __init__(self, name, alpha=1.0, c=C_FIXED):
        super().__init__(name, f"A=(r_i-b_-i)/{c}, b_-i=sum_(j!=i) L_j^{alpha} r_j / sum_(j!=i) L_j^{alpha}; ssn")
        self.alpha, self.c = alpha, c

    def weights(self, batch, t):
        rg, u = batch["rg"], batch["Lg"].astype(float) ** self.alpha
        Sw = (u * rg).sum(-1, keepdims=True)
        U = u.sum(-1, keepdims=True)
        b = (Sw - u * rg) / (U - u)
        A = (rg - b) / self.c
        return agg(A.reshape(batch["R"], N), batch, "ssn"), None, True


class PositionLoo(Est):
    """Position-level LOO baseline. weight='one': survival-conditioned (W=1). weight='otb': cumulative
    token score-norm W (others' tokens only). own_included=True: verl-OTB style (own trajectory and own
    current token in W) -> control."""

    def __init__(self, name, weight="one", c=C_FIXED, own_included=False, zero_tail=False):
        desc = {"one": "b_-i,t = mean of r_j over OTHER samples alive at t (fallback mean_-i)",
                "otb": "b_-i,t = sum_(j!=i, alive) W_jt r_j / sum W_jt, W_jt = cumsum_s<=t ||e_y_js - pi||^2 (others' tokens; fallback mean_-i)"}[weight]
        if own_included:
            desc = "b_t = sum_j W_jt r_j / sum_j W_jt over ALL samples incl. own (own current token inside W_it)" + (", zero tail of the single longest" if zero_tail else "") + " [verl-OTB-style control]"
        super().__init__(name, f"A_it=(r_i-b_t)/{c}; " + desc + "; ssn")
        self.weight, self.c, self.own, self.zero_tail = weight, c, own_included, zero_tail
        self.expect = "biased (control)" if own_included else "unbiased"
        self.claims = "none" if own_included else "J"

    def weights(self, batch, t):
        Rn = batch["R"]
        mask = batch["mask"].astype(float)  # [R, N, Lmax]
        if self.weight == "one":
            Wc = mask
        else:
            wtok = 1.0 - 2.0 * PI[batch["x"][..., None], batch["tok"]] + PI_SQ[batch["x"]][..., None]
            Wc = np.cumsum(wtok * mask, -1) * mask
        r = batch["r"]
        Wg = Wc.reshape(Rn, B, n, Lmax)
        mg = mask.reshape(Rn, B, n, Lmax)
        rgg = batch["rg"][..., None]
        num = (Wg * rgg * mg).sum(2, keepdims=True)
        den = (Wg * mg).sum(2, keepdims=True)
        if self.own:
            b = num / (den + 1e-8)
            b = np.broadcast_to(b, Wg.shape).copy()
            if self.zero_tail:
                Lg = batch["Lg"]
                srt = np.sort(Lg, -1)
                mx, smx = srt[..., -1], srt[..., -2]
                is_max = (Lg == mx[..., None]) & ((Lg == mx[..., None]).sum(-1, keepdims=True) == 1) & (mx > smx)[..., None]
                tail = np.arange(Lmax)[None, None, None, :] >= smx[..., None, None]
                b[is_max[..., None] & tail] = 0.0
        else:
            num_l = num - Wg * rgg * mg
            den_l = den - Wg * mg
            fallback = np.broadcast_to(batch["loo"][..., None], Wg.shape)
            b = np.where(den_l > 1e-12, num_l / np.where(den_l > 1e-12, den_l, 1.0), fallback)
        A = (rgg - b) * mg / self.c
        W = A.reshape(Rn, N, Lmax) / (N * S_CONST)
        return None, W, (not self.own)


class PrefixFeatureBaseline(Est):
    """b_i(t) = mean_-i + clip(beta^T (phi_i(t) - mean_{j!=i alive at t} phi_j(t)), +-s_cap);
    phi(prefix) = [t/Lbar, mean log pi(y_<t), log pi(y_1) (0 at t=1)]; beta ridge-fitted on PAST steps."""

    def __init__(self, name, c=C_FIXED, lam=10.0, s_cap=10.0):
        super().__init__(name, f"A_it=(r_i-mean_-i-clip(beta^T dphi_i(t),±{s_cap}))/{c}, prefix features [t/Lbar, mean logpi(y<t), logpi(y1)], beta ridge(lam={lam}) on PAST steps; ssn")
        self.c, self.lam, self.s_cap = c, lam, s_cap
        self.p = 3

    def init_state(self, Rn):
        self.XtX = np.zeros((Rn, self.p, self.p))
        self.Xty = np.zeros((Rn, self.p))
        self.beta = np.zeros((Rn, self.p))

    def _features(self, batch):
        Rn = batch["R"]
        mask = batch["mask"].astype(float)
        logp = LOGPI[batch["x"][..., None], batch["tok"]] * mask  # [R, N, Lmax]
        cum = np.cumsum(logp, -1)
        j = np.arange(Lmax, dtype=float)
        mlp = np.zeros_like(logp)
        mlp[:, :, 1:] = cum[:, :, :-1] / j[1:]
        lp1 = np.zeros_like(logp)
        lp1[:, :, 1:] = logp[:, :, :1]
        pos = np.broadcast_to((j + 1) / L_MAT.mean(), logp.shape)
        phi = np.stack([pos, mlp, lp1], -1)  # [R, N, Lmax, p]
        phig = phi.reshape(Rn, B, n, Lmax, self.p)
        mg = mask.reshape(Rn, B, n, Lmax)
        gsum = (phig * mg[..., None]).sum(2, keepdims=True)
        gcnt = mg.sum(2, keepdims=True)
        return phi, phig, mg, gsum, gcnt

    def weights(self, batch, t):
        Rn = batch["R"]
        phi, phig, mg, gsum, gcnt = self._features(batch)
        self._cache = (phig, mg, gsum, gcnt)
        oth_cnt = gcnt - mg
        oth_mean = (gsum - phig * mg[..., None]) / np.where(oth_cnt > 0, oth_cnt, 1.0)[..., None]
        X_loo = (phig - oth_mean) * (oth_cnt > 0)[..., None] * mg[..., None]
        delta = np.clip(np.einsum("rbnlp,rp->rbnl", X_loo, self.beta), -self.s_cap, self.s_cap)
        W_tok = (-delta * mg / self.c).reshape(Rn, N, Lmax) / (N * S_CONST)
        w_seq = agg(((batch["rg"] - batch["loo"]) / self.c).reshape(Rn, N), batch, "ssn")
        return w_seq, W_tok, False

    def update(self, batch, g_full, t):
        phig, mg, gsum, gcnt = self._cache
        X = (phig - gsum / np.maximum(gcnt, 1.0)[..., None]) * mg[..., None]  # [R,B,n,L,p]
        y = (batch["rg"] - batch["mean_g"][..., None])[..., None] * mg  # [R,B,n,L]
        Xf = X.reshape(batch["R"], -1, self.p)
        yf = y.reshape(batch["R"], -1)
        self.XtX += np.einsum("rip,riq->rpq", Xf, Xf)
        self.Xty += np.einsum("rip,ri->rp", Xf, yf)
        self.beta = np.linalg.solve(self.XtX + self.lam * np.eye(self.p)[None], self.Xty[..., None])[..., 0]


# ----------------------------------------------------------------------------- harness
BURN = 30
SAMPLER = None


def build_estimators():
    E = []
    # references (verl estimators x verl aggregations)
    E += [RefEst("grpo | token-mean", "grpo", "tm", "A=(r-mean_g)/(std_g[ddof=1]+1e-6); token-mean (sum A_i S_i / sum L_i)", "biased (std, ratio)")]
    E += [RefEst("grpo | seq-mean-token-mean", "grpo", "stm", "A=(r-mean_g)/(std_g+1e-6); per-sequence token-mean (legacy micro-batch=1 path)", "biased (std + 1/L)")]
    E += [RefEst("grpo | seq-mean-token-sum-norm", "grpo", "ssn", "A=(r-mean_g)/(std_g+1e-6); ssn (ablation a)", "biased (std)")]
    E += [RefEst("drgrpo raw | ssn", "drgrpo", "ssn", "A=r-mean_g; ssn (Dr.GRPO native)", "unbiased")]
    E += [RefEst("drgrpo raw | token-mean", "drgrpo", "tm", "A=r-mean_g; token-mean", "ratio bias O(1/64)")]
    E += [RefEst("drgrpo raw | seq-mean-token-mean", "drgrpo", "stm", "A=r-mean_g; per-sequence token-mean (3a)", "biased (1/L)")]
    E += [RefEst("rloo (verl) | ssn", "rloo", "ssn", "A=n/(n-1)(r-mean_g)=r-mean_-i; ssn", "unbiased")]
    E += [RefEst("rloo (verl) | token-mean", "rloo", "tm", "A=r-mean_-i; token-mean (ablation b)", "ratio bias O(1/64)")]
    E += [RefEst("rloo (verl) | seq-mean-token-mean", "rloo", "stm", "A=r-mean_-i; per-sequence token-mean (3a)", "biased (1/L)")]
    E += [RefEst("opo (verl, own-incl.) | ssn", "opo", "ssn", "A=r-sum_j L_j r_j/sum_j L_j (own included); ssn", "biased O(1/n)")]
    E += [RefEst("opo (verl, own-incl.) | token-mean", "opo", "tm", "A=r-sum_j L_j r_j/sum_j L_j; token-mean", "biased")]
    E += [RefEst("reinforce++-baseline (verl) | token-mean", "rpp", "tm", "A=whiten_tokens(r-mean_g): (a-mu)/sqrt(var+1e-8), mu,var token-weighted over the batch (Bessel); token-mean", "biased (batch whitening)")]
    # ideas
    E += [RlooConst("rloo-const (/18, ssn)")]
    E += [RlooConst("rloo-const/drgrpo-centering (/18, ssn)", centering="mean")]
    E += [RlooConst("rloo-const | token-mean (ablation b)", mode="tm")]
    E += [LaggedScale("rloo-lagged-scale/ema beta=0.9")]
    E += [LaggedScale("rloo-lagged-scale/LEAK control (current step in EMA)", leak=True)]
    E += [LaggedScale("rloo/current-batch-std control (beta=0, no lag)", beta=0.0, leak=True)]
    E += [LeaveOneGroupOutScale("rloo-lagged-scale/leave-one-group-out")]
    E += [PromptScaleLagged("rloo-prompt-scale/lagged k0=2 smin=5")]
    E += [PromptScaleFrozen("rloo-prompt-scale/frozen w(x)=1/max(sigma_x,5)")]
    E += [CrossFitScale("rloo-prompt-scale/cross-fit s_-i+5 (stateless)")]
    E += [ISEst("is-prompt-sampling (q∝s(x) past, u=1/(Kq))")]
    E += [ISEst("is-prompt-sampling/no-u control (unbiased for J_q)", correct=False)]
    E += [HistBaseline("rloo-hist k=2 beta=0.5")]
    E += [HistBaseline("rloo-hist k adaptive (sigma2/tau2 from past)", adaptive=True)]
    E += [HistBaseline("rloo-hist/LEAK control (current mean in h)", leak=True)]
    E += [LaggedGradNorm("lagged-gradnorm-scale beta=0.9")]
    E += [CurrentBatchNGD("gradnorm/current-batch NGD control")]
    E += [PrefixBranching("prefix-branching m=4 k=2 (as specified: prefix A / k)", 4, 2)]
    E += [PrefixBranching("prefix-branching m=2 k=4 (as specified: prefix A / k)", 2, 4)]
    E += [PrefixBranching("prefix-branching m=4 k=2 / corrected (no 1/k on prefix)", 4, 2, fix=False)]
    E += [PrefixBranching("prefix-branching m=2 k=4 / corrected (no 1/k on prefix)", 2, 4, fix=False)]
    E += [OpoLoo("opo-loo alpha=1 (L-weighted LOO)")]
    E += [OpoLoo("opo-loo alpha=0.5", alpha=0.5)]
    E += [PositionLoo("opo-loo position-level W=1 (survival LOO)")]
    E += [PositionLoo("otb-loo position-level cumulative-W LOO", weight="otb")]
    E += [PositionLoo("otb own-included (verl-style, zero tail) control", weight="otb", own_included=True, zero_tail=True)]
    E += [PrefixFeatureBaseline("prefix-feature-baseline (ridge on past)")]
    return E


def run(R, T, seed, verbose=True):
    global SAMPLER
    rng = np.random.default_rng(seed)
    ests = build_estimators()
    for e in ests:
        e.init_state(R)
    SAMPLER = ISSampler(R)
    acc = {e.name: dict(se=np.zeros((R, K * M)), sf=np.zeros((R, K * M)), sq=np.zeros(R), cnt=0) for e in ests}
    t0 = time.time()
    for t in range(T):
        batches = {}
        batches["U"] = draw_batch(rng, rng.integers(0, K, (R, B)))
        xq, q = SAMPLER.draw(rng, t)
        batches["Q"] = draw_batch(rng, xq)
        batches["Q"]["q"] = q
        batches["P42"] = draw_batch(rng, rng.integers(0, K, (R, B)), branch=(4, 2))
        batches["P24"] = draw_batch(rng, rng.integers(0, K, (R, B)), branch=(2, 4))
        for e in ests:
            bt = batches[e.batch_kind]
            w, Wt, tok_rb = e.weights(bt, t)
            g_full = np.zeros((R, K, M))
            g_eval = np.zeros((R, K, M))
            if w is not None:
                g_full += g_from_seq(w, bt, bt["S_full"])
                g_eval += g_from_seq(w, bt, bt["S_rb"]) if e.seq_rb_ok else 0.0
            if Wt is not None:
                g_full += g_from_tok(Wt, bt, False)
                g_eval += g_from_tok(Wt, bt, tok_rb) if (tok_rb and e.seq_rb_ok) else g_from_tok(Wt, bt, False)
            if not e.seq_rb_ok:
                g_eval = g_full.copy()
            if e.post_norm is not None:
                nrm = np.linalg.norm(g_full.reshape(R, -1), axis=1)
                sc = np.where(nrm > 0, e.post_norm / np.where(nrm > 0, nrm, 1.0), 0.0)  # zero gradient stays zero
                g_full = g_full * sc[:, None, None]
                g_eval = g_full.copy()
            e.update(bt, g_full, t)
            if t >= BURN:
                a = acc[e.name]
                a["se"] += g_eval.reshape(R, -1)
                a["sf"] += g_full.reshape(R, -1)
                a["sq"] += (g_full.reshape(R, -1) ** 2).sum(1)
                a["cnt"] += 1
        SAMPLER.update(batches["Q"])
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"  step {t + 1}/{T}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return ests, acc


def analyze(ests, acc, R):
    out = []
    ref_name = "rloo-const (/18, ssn)"
    gJ = GRAD_J_FLAT
    blocks = [slice(x * M, (x + 1) * M) for x in range(K)]
    gJx_norm = np.array([np.linalg.norm(gJ[b]) for b in blocks])

    def summarize(e):
        a = acc[e.name]
        cnt = a["cnt"]
        chain_means = a["se"] / cnt  # [R, KM]
        Eg = chain_means.mean(0)
        se = chain_means.std(0, ddof=1) / np.sqrt(R)
        Egf = (a["sf"] / cnt).mean(0)
        Esq = a["sq"].sum() / (R * cnt)
        tr_var = Esq - (Egf**2).sum()
        kappa = Eg @ gJ / (gJ @ gJ)
        # all relative quantities are measured against ||kappa gradJ|| (= the scale of E[g]) so they are scale-free
        resid = np.linalg.norm(Eg - kappa * gJ) / (abs(kappa) * NORM_GJ)
        cos = Eg @ gJ / (np.linalg.norm(Eg) * NORM_GJ)
        floor = np.sqrt((se**2).sum()) / (abs(kappa) * NORM_GJ)  # residual expected from MC noise alone (upper bound)
        hw = 1.96 * se
        hw_max = hw.max() / (abs(kappa) * NORM_GJ)  # largest coordinate 95% CI half-width, relative to ||kappa gradJ||
        hw_norm = 1.96 * floor
        ratios, ratio_hw, bcos = [], [], []
        for x, b in enumerate(blocks):
            nx = np.linalg.norm(Eg[b])
            if gJx_norm[x] > 0:
                ratios.append(nx / gJx_norm[x] / kappa)
                ratio_hw.append(1.96 * np.sqrt((se[b] ** 2).sum()) / gJx_norm[x] / abs(kappa))
                bcos.append(Eg[b] @ gJ[b] / (nx * gJx_norm[x]) if nx > 0 else np.nan)
            else:
                ratios.append(np.nan)
                ratio_hw.append(np.nan)
                bcos.append(np.nan)
        # zero-gradient prompt: is E[g]_x zero within CI?
        b3 = blocks[3]
        zero_block = np.linalg.norm(Eg[b3]) / (abs(kappa) * NORM_GJ)
        zero_block_hw = 1.96 * np.sqrt((se[b3] ** 2).sum()) / (abs(kappa) * NORM_GJ)
        cross = np.linalg.norm(Egf - Eg) / (abs(kappa) * NORM_GJ)
        return dict(Eg=Eg, se=se, kappa=kappa, resid=resid, cos=cos, floor=floor, hw_max=hw_max, hw_norm=hw_norm,
                    ratios=np.array(ratios), ratio_hw=np.array(ratio_hw), bcos=np.array(bcos), tr_var=tr_var,
                    zero_block=zero_block, zero_block_hw=zero_block_hw, cross=cross)

    summ = {e.name: summarize(e) for e in ests}
    ref = summ[ref_name]
    ref_nvar = ref["tr_var"] / ref["kappa"] ** 2
    for e in ests:
        s = summ[e.name]
        rel_var = (s["tr_var"] / s["kappa"] ** 2) / ref_nvar
        r3 = s["ratios"][:3]
        hw3 = s["ratio_hw"][:3]
        spread = (r3.max() - r3.min()) / r3.mean()
        resolved = hw3 <= 0.10  # a block whose ratio CI is wider than +-0.10 cannot support a verdict
        # constant-ratio test on resolved blocks: ratio == 1 within 1.5 x CI (+0.5% slack)
        const_ok = np.all((np.abs(r3 - 1.0) <= 1.5 * hw3 + 0.005) | ~resolved)
        # within-prompt direction test on resolved blocks (block cosines ~ 1)
        dir_ok = np.all((s["bcos"][:3] > 0.9995) | ~resolved)
        # global bias detected beyond the MC noise floor?
        detected = s["resid"] > 3.0 * s["floor"] + 0.002
        if (not detected) and const_ok:
            verdict = "unbiased for J"
        elif dir_ok and e.claims == "Jw":
            verdict = "unbiased for stated reweighted objective"
        else:
            verdict = "biased"
        unres = [f"x{i}" for i in range(3) if not resolved[i]]
        cw = e.claimed_w()
        rew = ("constant" if const_ok else "varying") + f": ratios x0..x2 = [{r3[0]:.3f}, {r3[1]:.3f}, {r3[2]:.3f}] ±[{hw3[0]:.3f}, {hw3[1]:.3f}, {hw3[2]:.3f}]"
        rew += f"; within-prompt cos x0..x2 = [{s['bcos'][0]:.5f}, {s['bcos'][1]:.5f}, {s['bcos'][2]:.5f}]"
        if unres:
            rew += f"; unresolved blocks (CI>0.10): {','.join(unres)}"
        if cw is not None:
            cwn = cw[:3] / cw[:3].mean()
            rn = r3 / r3.mean()
            rew += f"; claimed w(x)/mean = [{cwn[0]:.3f}, {cwn[1]:.3f}, {cwn[2]:.3f}] vs measured ratio/mean = [{rn[0]:.3f}, {rn[1]:.3f}, {rn[2]:.3f}]"
        out.append(dict(name=e.name, impl=e.impl, expect=e.expect, claims=e.claims, cos=float(s["cos"]), kappa=float(s["kappa"]),
                        resid=float(s["resid"]), floor=float(s["floor"]), hw_max=float(s["hw_max"]), hw_norm=float(s["hw_norm"]),
                        ratios=[float(v) for v in s["ratios"]], ratio_hw=[float(v) for v in s["ratio_hw"]],
                        bcos=[float(v) for v in s["bcos"]], rel_var=float(rel_var), zero_block=float(s["zero_block"]),
                        zero_block_hw=float(s["zero_block_hw"]), cross=float(s["cross"]), verdict=verdict, reweighting=rew,
                        spread=float(spread), unresolved=unres))
    return out


def print_table(res):
    hdr = f"{'estimator':<58} {'cos':>9} {'bias%':>7} {'floor%':>7} {'CI%':>6} {'r_x0':>7} {'r_x1':>7} {'r_x2':>8} {'bcos0':>7} {'bcos1':>7} {'bcos2':>7} {'relVar':>7} {'x3|0':>5}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        print(f"{r['name']:<58} {r['cos']:>9.6f} {100 * r['resid']:>7.2f} {100 * r['floor']:>7.2f} {100 * r['hw_max']:>6.2f} "
              f"{r['ratios'][0]:>7.3f} {r['ratios'][1]:>7.3f} {r['ratios'][2]:>8.3f} {r['bcos'][0]:>7.4f} {r['bcos'][1]:>7.4f} {r['bcos'][2]:>7.4f} {r['rel_var']:>7.3f} "
              f"{'ok' if r['zero_block'] <= r['zero_block_hw'] + 1e-6 else 'BAD':>5}  {r['verdict']}")
    print("\ncos = cosine(E[g], gradJ); bias% = ||E[g]-kappa gradJ|| / ||kappa gradJ|| (kappa = best-fit scalar); floor% = upper bound on")
    print("the bias% that MC noise alone would produce; CI% = largest per-coordinate 95% half-width / ||kappa gradJ||;")
    print("r_x = ||E[g]_x|| / (kappa ||gradJ_x||) for prompts 0 (spread~25), 1 (spread~10), 2 (near-flat [50,50,50,50.15,50,50]);")
    print("bcos_x = within-prompt cosine(E[g]_x, gradJ_x); relVar = trace Var(g)/kappa^2 relative to rloo-const (/18, ssn);")
    print("x3|0: E[g] on the all-equal prompt (gradJ_x = 0) is zero within its CI.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    print(f"theta =\n{np.array2string(THETA, precision=3)}\npi =\n{np.array2string(PI, precision=3)}")
    print(f"J_x = {np.array2string(J_X, precision=3)}, within-prompt reward std sigma_x = {np.array2string(SIGMA_X, precision=3)}")
    print(f"||gradJ|| = {NORM_GJ:.5f}; per-prompt block norms = {np.array2string(np.linalg.norm(GRAD_J, axis=1), precision=5)}")
    print(f"finite-difference check of closed-form gradJ: max rel err = {fd_check(GRAD_J):.2e}")
    print(f"MC: {args.chains} chains x {args.steps} steps, burn-in {BURN} -> {args.chains * (args.steps - BURN)} evaluation batches of {N} samples")
    ests, acc = run(args.chains, args.steps, args.seed)
    res = analyze(ests, acc, args.chains)
    print()
    print_table(res)
    print("\nPer-prompt reweighting detail:")
    for r in res:
        print(f"  {r['name']}: {r['reweighting']}; g_full-vs-RB cross-check diff = {100 * r['cross']:.2f}% ; kappa = {r['kappa']:.4g}; implemented as: {r['impl']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
