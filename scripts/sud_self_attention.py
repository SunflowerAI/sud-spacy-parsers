#!/usr/bin/env python
"""SelfAttentionMixer -- ONE lightweight, single-head scaled dot-product self-attention layer,
inserted between an encoder's output and JointBiaffine's own projections.

WHY. Diagnosing la_frozen_sib_s0 found the LAS gap monotonic in arc length (dist1 -5.25 -> dist5
-17.36, plateauing ~-16..-17 through dist9) despite the biaffine's own candidate window (`--window`,
50) never excluding those arcs as candidates -- the deficiency is in the RICHNESS of the token
representations X being scored, not in which candidates get considered. `la_frozen`'s X comes from
the transition parser's own tok2vec, almost certainly a shallow CNN-style encoder
(`MaxoutWindowEncoder`: narrow window x fixed depth) whose receptive field may not reach the
distances where the gap plateaus -- and no scoring formula sitting on top of X (first-order,
--sibling, --grandparent, or any further hand-picked higher-order term) can recover context that was
never encoded into X to begin with.

THE ALTERNATIVE TO --sibling/--grandparent's CHAIN. Each of those is, structurally, "attend to
exactly ONE specific other token, chosen by a hand-written rule, after decoding a tree once" -- a
chain that only grows (great-grandparent, a coordination-specific term, ...) each time a new
diagnosis motivates one, and each new order costs its own gradient-checked bucket function for a
shrinking slice of the error. Self-attention is the GENERAL form of that: a LEARNED, SOFT,
whole-sentence mixing that can attend to WHATEVER other tokens help a given decision, discovered by
gradient descent rather than hypothesised by a human -- no two-pass decode needed, since it doesn't
need a tree to know where to look.

WITHOUT "TRANSFORMERS" AND WITHOUT BLOWING UP PARAMETERS, per direct instruction: this is ONE
attention block, not a multi-layer Transformer stack -- no positional encoding (the frozen encoder's
own features and `dist`/`direction`'s bucket terms already carry position information the biaffine
reads directly), no feed-forward sublayer, no layer norm, SINGLE head. Four square (w, w) projection
matrices (Wq, Wk, Wv, Wo) -- with w=96 (this project's standing hidden width), that is 4*96*96 =
36,864 parameters, smaller than JointBiaffine's own `V` tensor (nlab * h * h, ~479k for la's 52
labels) by more than an order of magnitude.

A RESIDUAL connection (X_out = X + attention_output @ Wo) keeps the well-trained frozen features as
the base case, and `Wo` is initialised to ZERO (unlike every other projection here, which uses the
usual random-normal scaling) so this layer starts as an exact no-op: X_out == X at step 0. If
attention never learns anything useful, gradient descent simply leaves Wo near zero -- this can only
ADD capability relative to today's `la_frozen` baseline, never actively hurt it by construction
(the same "additive, gated, cannot make things worse before it learns anything" discipline
`sud_joint_biaffine.py`'s bias terms already follow, applied to an architecture layer instead of a
scalar bias table).

Attention is UNMASKED and spans the WHOLE sentence (unlike JointBiaffine's own `window_mask`, which
only restricts which ARCS are decoding candidates, never how far a representation can look) --
escaping a limited receptive field is the entire point, so restricting attention's own reach would
defeat it. Sentences here are short (rarely 100+ tokens), so the O(n^2) attention matrix is trivial.

⚠ GRADIENT-CHECKED BEFORE TRUSTING IT, the same discipline every term in `sud_joint_biaffine.py`
uses -- checked here as a general vector-Jacobian product against an ARBITRARY upstream gradient
(this module owns no loss of its own, unlike JointBiaffine's own loss_and_backward), not against a
concrete loss function. Run this file directly to gradient-check against finite differences.
"""
import numpy as np


class SelfAttentionMixer:
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
        """X: (n, w), any dtype forward() is called with (float32 in production, float64 in this
        module's own gradient check -- same convention JointBiaffine's forward/loss_and_backward
        follow). Returns (X_out, cache); X_out = X + softmax(QK^T / sqrt(w)) @ V @ Wo."""
        p = self.p
        Q = X @ p["Wq"]; K = X @ p["Wk"]; V = X @ p["Wv"]            # (n, w) each
        scale = 1.0 / np.sqrt(self.w)
        raw = Q @ K.T                                                 # (n, n)
        scores = raw * scale
        scores = scores - scores.max(1, keepdims=True)
        E = np.exp(scores); A = E / E.sum(1, keepdims=True)          # (n, n), row-stochastic
        ctx = A @ V                                                   # (n, w)
        out = ctx @ p["Wo"]                                            # (n, w)
        X_out = X + out
        cache = dict(X=X, Q=Q, K=K, V=V, A=A, ctx=ctx)
        return X_out, cache

    def backward(self, dX_out, cache):
        """dX_out: (n, w) gradient wrt the layer's OUTPUT (handed in by whatever consumed X_out --
        in practice JointBiaffine.loss_and_backward's own `dX` return value). Returns (grads_dict,
        dX_in): dX_in is the gradient wrt this layer's INPUT, needed to backprop further into a
        --joint encoder (harmless to compute and discard in frozen mode, where X_in has no further
        backward path)."""
        p = self.p
        X, Q, K, V, A, ctx = (cache[k] for k in ("X", "Q", "K", "V", "A", "ctx"))
        scale = 1.0 / np.sqrt(self.w)
        g = {}
        g["Wo"] = ctx.T @ dX_out                                       # (w, w)
        dctx = dX_out @ p["Wo"].T                                      # (n, w)
        dA = dctx @ V.T                                                # (n, n)
        dV = A.T @ dctx                                                # (n, w)
        # softmax backward, per row: dscores[i,:] = A[i,:] * (dA[i,:] - sum_j A[i,j] dA[i,j])
        row_dot = np.sum(A * dA, axis=1, keepdims=True)
        dscores = A * (dA - row_dot)                                   # (n, n)
        draw = dscores * scale                                         # scores = raw * scale
        dQ = draw @ K                                                  # raw = Q @ K.T
        dK = draw.T @ Q
        g["Wq"] = X.T @ dQ
        g["Wk"] = X.T @ dK
        g["Wv"] = X.T @ dV
        # X feeds Q/K/V AND the residual path directly -- every path's contribution sums.
        dX_in = dX_out + dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
        return g, dX_in


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


if __name__ == "__main__":
    for s in range(6):
        _numeric_grad_check(seed=s)
