#!/usr/bin/env python
"""Chu-Liu/Edmonds maximum spanning arborescence, for arc-factored (graph-based) parsing.

WHY IT IS HERE. Latin's non-projective headroom has survived three interventions (more actions,
upsampling, beam search) and `NEGATIVE-RESULTS.md` indicts the PSEUDO-PROJECTIVE REPRESENTATION
rather than any decoder over it. Two costs were then measured directly on gold trees:

  * a HARD CEILING of 0.72 UAS -- round-tripping gold Latin through projectivize/deprojectivize
    fails to return 395 of 54 897 heads, because deprojectivization is a heuristic breadth-first
    search for a token bearing the right label and it picks the wrong one;
  * 200 distinct decorated label types over 2 731 tokens, 78 of them occurring exactly ONCE.

An arc-factored decoder has neither: no decorated labels, so no sparsity; no round trip, so no
ceiling; and the score of a tree IS the objective, so unlike a transition beam there is no payoff
hidden behind a post-processing step.

⚠ MULTI-ROOT BY DESIGN. Node 0 is a VIRTUAL ROOT. Every token may attach to it, so the result is a
FOREST over the doc and the tokens attaching to node 0 are the sentence roots. This project's
parsers double as sentencisers (ArcEager's BREAK), and a decoder that forced a single root would
silently give that up.

⚠ THE SCORE MATRIX IS WINDOWED, so this must tolerate -inf. Latin arcs are short: at k=50, 99.99 %
of all arcs and 100 % of CROSSING arcs are within the window, which makes scoring O(n*k) rather
than O(n^2) -- standing hazard 10 is about costs that grow with the length of the CALL, and whole
multi-sentence docs are the call here.
"""
from typing import List

import numpy as np

NEG = -np.inf


def mst(scores: np.ndarray) -> np.ndarray:
    """Maximum spanning arborescence rooted at node 0.

    `scores[h, d]` is the score of head h -> dependent d. Node 0 is the virtual root and never
    takes a head. Returns `heads`, length n, with `heads[0] == 0`.
    """
    n = scores.shape[0]
    s = scores.astype("float64", copy=True)
    np.fill_diagonal(s, NEG)
    s[:, 0] = NEG                       # the virtual root never takes a head
    heads = np.zeros(n, dtype="int64")
    heads[1:] = np.argmax(s[:, 1:], axis=0)
    cyc = _find_cycle(heads, n)
    if cyc is None:
        return heads
    return _contract(s, heads, cyc, n)


def _find_cycle(heads: np.ndarray, n: int):
    colour = np.zeros(n, dtype="int8")          # 0 unvisited, 1 on stack, 2 done
    for start in range(1, n):
        if colour[start]:
            continue
        path = []
        v = start
        while colour[v] == 0:
            colour[v] = 1
            path.append(v)
            v = heads[v]
            if v == 0:
                break
        if v != 0 and colour[v] == 1:
            return path[path.index(v):]
        for u in path:
            colour[u] = 2
    return None


def _contract(s, heads, cyc, n):
    """Standard CLE contraction: collapse the cycle, solve, then expand by breaking one cycle arc."""
    cyc_set = set(cyc)
    outside = [v for v in range(n) if v not in cyc_set]
    idx = {v: i for i, v in enumerate(outside)}
    m = len(outside)
    cyc_score = sum(s[heads[v], v] for v in cyc)
    S = np.full((m + 1, m + 1), NEG)
    C = m                                       # index of the contracted node
    for a in outside:
        for b in outside:
            if a != b:
                S[idx[a], idx[b]] = s[a, b]
    # entering the cycle: best (head outside -> v in cycle), discounting v's current in-cycle arc
    best_in = {}
    for a in outside:
        best, arg = NEG, None
        for v in cyc:
            val = s[a, v] - s[heads[v], v]
            if val > best:
                best, arg = val, v
        S[idx[a], C] = cyc_score + best if best > NEG else NEG
        best_in[a] = arg
    # leaving the cycle: best (v in cycle -> b outside)
    best_out = {}
    for b in outside:
        best, arg = NEG, None
        for v in cyc:
            if s[v, b] > best:
                best, arg = s[v, b], v
        S[C, idx[b]] = best
        best_out[b] = arg
    sub = mst(S)
    heads_out = np.zeros(n, dtype="int64")
    for b in outside:
        if b == 0:
            continue
        h = sub[idx[b]]
        heads_out[b] = best_out[b] if h == C else outside[h]
    entry_head = outside[sub[C]]                # who enters the cycle from outside
    broken = best_in[entry_head]                # and at which cycle node
    for v in cyc:
        heads_out[v] = entry_head if v == broken else heads[v]
    return heads_out
