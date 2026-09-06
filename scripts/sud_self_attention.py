#!/usr/bin/env python
"""attn_forward/attn_backward -- ONE lightweight, single-head scaled dot-product self-attention
block, as pure functions over explicit weight matrices (not a stateful object) -- so the SAME math
can be called from JointBiaffine's own forward/loss_and_backward on ITS OWN (h, h) representations,
not just from an external `SelfAttentionMixer` instance. `SelfAttentionMixer` (below) is a thin
stateful wrapper around them for standalone use/testing.

⚠ SUPERSEDED PLACEMENT, KEPT AS A DOCUMENTED NEGATIVE RESULT: this was originally wired into
`train_arcfactored.py` as `--attn`, applied to X BEFORE JointBiaffine's own Wh/Wd/Lh/Ld projections
-- and it REGRESSED (-0.87 LAS vs --sibling alone, even below plain baseline; see
NEGATIVE-RESULTS.md's "`--grandparent` and `--attn`" entry). The diagnosis, once pointed out
directly: Wh/Wd/Lh/Ld (and their ReLU) are exactly "the pieces that need INDIVIDUAL token
information" -- each needs X's own clean, unmixed per-token content to decide THAT token's own
arc/label-scoring activation pattern. Mixing X BEFORE those projections blurs the very identity
signal (agreement, lemma) the OTHER bias terms already read cleanly, before any per-token decision
gets to run on unmixed input.

THE FIX (`JointBiaffine`'s `use_attn_hd`, in `sud_joint_biaffine.py`): attention now runs AFTER
Wh/Wd's per-token projection and ReLU, refining H/D (the "am I a good candidate head/dependent"
representations) with sentence-wide context, rather than the raw shared input those individual
decisions are made from. Only the ARC-scoring pathway (H, D) is touched -- LH/LD (label-scoring)
stays untouched, since diagnosis found label accuracy given a correct head already close to the
transition parser's (labelling isn't the diagnosed problem; long-distance ATTACHMENT is). This
module still supplies the underlying math either way -- only WHERE it is inserted changed.

WITHOUT "TRANSFORMERS" AND WITHOUT BLOWING UP PARAMETERS, per direct instruction: ONE attention
block, not a multi-layer stack -- no positional encoding, no feed-forward sublayer, no layer norm,
SINGLE head. Four square (w, w) projections (Wq, Wk, Wv, Wo) -- at w=96, 36,864 parameters per
instance, smaller than JointBiaffine's own `V` tensor (nlab * h * h, ~479k for la's 52 labels) by
more than an order of magnitude. `use_attn_hd` uses TWO independent instances (head-side,
dependent-side, since H and D live in different learned subspaces) -- 73,728 params total, still
under a fifth of `V`.

A RESIDUAL connection (X_out = X + attention_output @ Wo) keeps the input as the base case, and `Wo`
is initialised to ZERO (unlike every other projection here) so this starts as an exact no-op: this
can only ADD capability, never regress below not having it, BY CONSTRUCTION -- the "cannot hurt
before it's learned anything" half of the claim held even for the refuted placement; only the OTHER
half ("what it eventually learns is useful") failed there, motivating the placement fix, not a
retreat from the safety property itself.

Attention is UNMASKED (no candidate-window restriction) and spans whichever set of representations
it is applied to -- escaping a limited receptive field is the entire point, so restricting attention's
own reach would defeat it. Sentences here are short, so the O(n^2) attention matrix is trivial.

⚠ GRADIENT-CHECKED BEFORE TRUSTING IT, the same discipline every term in `sud_joint_biaffine.py`
uses -- checked here as a general vector-Jacobian product against an ARBITRARY upstream gradient
(this module owns no loss of its own), not against a concrete loss function. Run this file directly
to gradient-check against finite differences. `use_attn_hd`'s OWN placement inside JointBiaffine gets
its own additional check in sud_joint_biaffine.py's `_numeric_grad_check` (a different claim: that
threading these same functions into ITS forward/backward, at a different point, is ALSO correct).
"""
import numpy as np


def attn_forward(X, Wq, Wk, Wv, Wo, window=None):
    """X: (n, w). Returns (X_out, cache); X_out = X + softmax(QK^T / sqrt(w)) @ V @ Wo.

    ⚠ `window` (an int, or None for the original unmasked behaviour) restricts attention to
    |i-j| <= window -- a DIFFERENT knob from JointBiaffine's own `--window` (which only restricts
    which ARCS are decoding CANDIDATES, never how far a representation can look). Built to test
    whether unmasked, whole-sentence mixing is itself the reason `--attn-hd` interacts badly when
    stacked with `--clausegap` (NEGATIVE-RESULTS.md), or whether restricting attention to a more
    local span changes its behaviour at all. Masking BEFORE the softmax (not after) is what makes
    this a real restriction rather than decoration: a masked position gets -1e9, so it receives
    (numerically) zero attention weight and, via the identical downstream softmax-Jacobian formula
    in `attn_backward`, zero gradient too -- no separate backward-side masking is needed, since it
    is already baked into the cached `A`."""
    w = Wq.shape[0]
    n = X.shape[0]
    Q = X @ Wq; K = X @ Wk; V = X @ Wv                                # (n, w) each
    scale = 1.0 / np.sqrt(w)
    raw = Q @ K.T                                                      # (n, n)
    scores = raw * scale
    if window is not None:
        idx = np.arange(n)
        m = np.abs(idx[:, None] - idx[None, :]) <= window
        scores = np.where(m, scores, -1e9)
    scores = scores - scores.max(1, keepdims=True)
    E = np.exp(scores); A = E / E.sum(1, keepdims=True)               # (n, n), row-stochastic
    ctx = A @ V                                                        # (n, w)
    out = ctx @ Wo                                                     # (n, w)
    X_out = X + out
    cache = dict(X=X, Q=Q, K=K, V=V, A=A, ctx=ctx)
    return X_out, cache


def attn_backward(dX_out, cache, Wq, Wk, Wv, Wo):
    """dX_out: (n, w) gradient wrt the block's OUTPUT. Returns (grads_dict, dX_in) -- grads_dict has
    keys Wq/Wk/Wv/Wo; dX_in is the gradient wrt the block's INPUT (needed to backprop further, e.g.
    into Wh/Wd's own ReLU gate when this sits after them, or into a --joint encoder when it doesn't)."""
    X, Q, K, V, A, ctx = (cache[k] for k in ("X", "Q", "K", "V", "A", "ctx"))
    w = Wq.shape[0]
    scale = 1.0 / np.sqrt(w)
    g = {}
    g["Wo"] = ctx.T @ dX_out                                            # (w, w)
    dctx = dX_out @ Wo.T                                                # (n, w)
    dA = dctx @ V.T                                                     # (n, n)
    dV = A.T @ dctx                                                     # (n, w)
    # softmax backward, per row: dscores[i,:] = A[i,:] * (dA[i,:] - sum_j A[i,j] dA[i,j])
    row_dot = np.sum(A * dA, axis=1, keepdims=True)
    dscores = A * (dA - row_dot)                                        # (n, n)
    draw = dscores * scale                                              # scores = raw * scale
    dQ = draw @ K                                                       # raw = Q @ K.T
    dK = draw.T @ Q
    g["Wq"] = X.T @ dQ
    g["Wk"] = X.T @ dK
    g["Wv"] = X.T @ dV
    # X feeds Q/K/V AND the residual path directly -- every path's contribution sums.
    dX_in = dX_out + dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return g, dX_in


class SelfAttentionMixer:
    """Thin stateful wrapper around attn_forward/attn_backward, for standalone use (this module's
    own gradient check) -- kept for the historical `--attn` placement (X, before Wh/Wd/Lh/Ld),
    documented as a negative result. `JointBiaffine`'s `use_attn_hd` calls the pure functions
    directly on its own p[...] keys instead of instantiating this class."""
    def __init__(self, w, seed=0):
        r = np.random.default_rng(seed)
        s = lambda *d: (r.normal(size=d) * (1.0 / np.sqrt(d[0]))).astype("float64")
        self.p = {"Wq": s(w, w), "Wk": s(w, w), "Wv": s(w, w),
                  # ⚠ Wo STARTS AT ZERO -- see the module note: makes this layer an exact identity
                  # at initialisation, so training can only add capability, never regress below
                  # today's baseline before it has learned anything.
                  "Wo": np.zeros((w, w))}
        self.w = w

    def forward(self, X):
        p = self.p
        return attn_forward(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"])

    def backward(self, dX_out, cache):
        p = self.p
        return attn_backward(dX_out, cache, p["Wq"], p["Wk"], p["Wv"], p["Wo"])


def _numeric_grad_check(seed=0):
    """No loss of its own to check against (unlike JointBiaffine), so this verifies backward() is
    the correct vector-Jacobian product for an ARBITRARY upstream gradient `dX_out`: for
    L(x) = <dX_out, forward(x)[0]> (dX_out held FIXED, not recomputed), dL/dx is EXACTLY what
    backward(dX_out, cache) must return, by the definition of a VJP -- the same contract
    JointBiaffine's own `dX` return value is held to, checked the same way finite-difference checks
    are always checked in this project: against every parameter AND against the input."""
    rng = np.random.default_rng(seed)
    n, w = 5, 6
    m = SelfAttentionMixer(w, seed=seed + 1)
    # ⚠ RANDOMISE EVERY PARAM BEFORE CHECKING, Wo included -- the same convention
    # sud_joint_biaffine.py's own check uses (overwriting even params whose PRODUCTION init is
    # special-cased, e.g. Wo's zero-init here), so the check exercises a genuinely nonzero Wo
    # rather than the trivial all-zero-gradient case its own init would otherwise give.
    for key in m.p:
        m.p[key] = rng.normal(size=m.p[key].shape) * 0.5
    X = rng.normal(size=(n, w))
    dX_out = rng.normal(size=(n, w))          # arbitrary, fixed upstream gradient
    _, cache0 = m.forward(X)
    g, dX_in = m.backward(dX_out, cache0)

    def f(Xp):
        Xo, _ = m.forward(Xp)
        return float(np.sum(Xo * dX_out))

    eps = 1e-5
    worst = 0.0
    for key in m.p:
        arr = m.p[key]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps; fp = f(X)
            arr[idx] = old - eps; fm = f(X)
            arr[idx] = old
            num = (fp - fm) / (2 * eps)
            ana = g[key][idx]
            denom = max(abs(num), abs(ana), 1e-6)
            worst = max(worst, abs(num - ana) / denom)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + eps; fp = f(X)
        X[idx] = old - eps; fm = f(X)
        X[idx] = old
        num = (fp - fm) / (2 * eps)
        ana = dX_in[idx]
        denom = max(abs(num), abs(ana), 1e-6)
        worst = max(worst, abs(num - ana) / denom)
    print(f"[self-attn seed={seed}] worst relative error: {worst:.2e}")
    assert worst < 5e-4, "gradient check FAILED"
    print(f"[self-attn seed={seed}] gradient check PASSED")


def _numeric_grad_check_windowed(window, seed=0):
    """Same VJP contract as `_numeric_grad_check`, but exercising `attn_forward`'s `window` mask --
    a genuinely new code path (the -1e9-then-softmax masking), not assumed correct just because the
    unmasked case already gradient-checks: a masked position's near-zero `A` entry could still leak
    a stray gradient if the masking were applied inconsistently between forward and backward."""
    rng = np.random.default_rng(seed)
    n, w = 7, 6
    Wq = rng.normal(size=(w, w)) * 0.5; Wk = rng.normal(size=(w, w)) * 0.5
    Wv = rng.normal(size=(w, w)) * 0.5; Wo = rng.normal(size=(w, w)) * 0.5  # nonzero, unlike prod init
    X = rng.normal(size=(n, w))
    dX_out = rng.normal(size=(n, w))

    def f(Xp):
        Xo, _ = attn_forward(Xp, Wq, Wk, Wv, Wo, window=window)
        return float(np.sum(Xo * dX_out))

    _, cache0 = attn_forward(X, Wq, Wk, Wv, Wo, window=window)
    g, dX_in = attn_backward(dX_out, cache0, Wq, Wk, Wv, Wo)

    eps = 1e-5
    worst = 0.0
    params = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo}
    for key, arr in params.items():
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps; fp = f(X)
            arr[idx] = old - eps; fm = f(X)
            arr[idx] = old
            num = (fp - fm) / (2 * eps)
            ana = g[key][idx]
            denom = max(abs(num), abs(ana), 1e-6)
            worst = max(worst, abs(num - ana) / denom)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + eps; fp = f(X)
        X[idx] = old - eps; fm = f(X)
        X[idx] = old
        num = (fp - fm) / (2 * eps)
        ana = dX_in[idx]
        denom = max(abs(num), abs(ana), 1e-6)
        worst = max(worst, abs(num - ana) / denom)
    print(f"[self-attn windowed window={window} seed={seed}] worst relative error: {worst:.2e}")
    assert worst < 5e-4, "gradient check FAILED"
    print(f"[self-attn windowed window={window} seed={seed}] gradient check PASSED")


if __name__ == "__main__":
    for s in range(6):
        _numeric_grad_check(seed=s)
    for window in (0, 1, 3):
        for s in range(3):
            _numeric_grad_check_windowed(window, seed=s)
