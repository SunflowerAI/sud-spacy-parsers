#!/usr/bin/env python
"""SECOND-ORDER (consecutive-sibling) scoring for the arc-factored decoder, via local-search
reranking on top of the existing first-order Chu-Liu/Edmonds decode.

WHY LOCAL-SEARCH RERANKING, NOT EXACT SECOND-ORDER MST. Exact maximum-weight decoding with
second-order (sibling) factors is NP-hard for general (non-projective) graphs -- the Matrix-Tree
theorem that makes exact FIRST-ORDER non-projective decoding tractable does not extend to
second-order factors. This project uses CLE specifically because Latin is 37% non-projective
(docs/latin.md) and a projective-only algorithm (Eisner's O(n^3) DP, which DOES have an exact
second-order extension: McDonald & Pereira 2006) would give up exactly the non-projective coverage
CLE exists for. Local-search reranking -- decode first-order-optimal via CLE as always, then
greedily re-attach dependents where a first-order-score loss is outweighed by a sibling-score gain
-- is the standard tractable compromise real systems use when exact higher-order decoding is out of
reach, and it costs nothing at TRAINING time: the sibling table here is estimated directly from gold
corpus counts (a smoothed PMI table, not a gradient-trained parameter), so this can be measured
against an ALREADY-TRAINED checkpoint (`la_frozen_full`) with no retraining at all.

MOTIVATION (scripts run one-off, not checked in): `diagnose_la_deprel_errors.py`-style analysis of
`la_frozen_full` (the RELIABLE, deterministic arc-factored checkpoint -- frozen encoder, no seed
variance) found the LAS gap to the transition parser concentrated almost entirely in long-distance
attachment (arc-length gap grows from -5.75 at distance 1 to -20+ at distance 5-8) and in
`conj:coord`/`subj` specifically, while label accuracy GIVEN a correct head was nearly identical
(91.14 vs 92.46) -- an attachment problem, not a labelling one, and one arc-factored (first-order)
scoring has no mechanism to address: each candidate arc is scored independently of which OTHER
dependents its head already has, so nothing rewards "these three tokens form a coordination chain"
as a unit. Checking la's gold training trees directly confirms real, strong sibling structure a
first-order model cannot see: `conj:coord -> punct` (a coordinated conjunct followed by punctuation,
both dependents of the same head) at P=0.985 over 2,580 occurrences; `punct -> cc` at lift 9.07 over
5,681 occurrences; `subj@pass -> comp:aux@pass` (passive-construction siblings) at lift 8.82.

⚠ "CONSECUTIVE" MEANS ADJACENT IN SORTED ABSOLUTE POSITION, not adjacent on one side of the head --
matching exactly how the motivating corpus statistics were computed (sort a head's dependents by
token index, look at consecutive pairs), so the table's own meaning and the reranker's use of it
stay consistent with each other.
"""
import collections

import numpy as np


def build_sibling_table(gold_docs, labels, alpha=1.0, clip=3.0):
    """(nlab, nlab) PMI table: `table[l1, l2] = log( P(l2 | l1, consecutive) / P(l2) )`, estimated
    from GOLD dependency trees (a corpus statistic, not a gradient-trained parameter -- this can be
    built once from `tr.load(lang, 'train', ...)` and reused for any checkpoint of that language).

    `labels` is the checkpoint's own `meta["labels"]` (the sorted deprel list), so table indices
    line up with whatever `li = {l: i for i, l in enumerate(labels)}` the caller already uses.

    Additive (Laplace) smoothing with `alpha` avoids -inf for unseen pairs; `clip` bounds the
    resulting bonus to a modest range relative to the biaffine's own raw score scale (seen
    elsewhere in this project's checkpoints to run roughly -20..+30), so an unusual pair can nudge a
    decoding decision without being able to overrule a strong arc-factored signal outright.
    """
    li = {l: i for i, l in enumerate(labels)}
    nlab = len(labels)
    pair_ct = np.zeros((nlab, nlab), dtype="float64")
    l1_ct = np.zeros(nlab, dtype="float64")
    l2_ct = np.zeros(nlab, dtype="float64")
    for d in gold_docs:
        by_head = collections.defaultdict(list)
        for t in d:
            if t.head.i != t.i and t.dep_ in li:
                by_head[t.head.i].append((t.i, li[t.dep_]))
        for h, deps in by_head.items():
            deps.sort()
            for i in range(len(deps) - 1):
                l1, l2 = deps[i][1], deps[i + 1][1]
                pair_ct[l1, l2] += 1
                l1_ct[l1] += 1
                l2_ct[l2] += 1
    total = l2_ct.sum()
    p_l2 = (l2_ct + alpha) / (total + alpha * nlab)
    # P(l2 | l1) with the SAME smoothing on the conditional
    p_l2_given_l1 = (pair_ct + alpha) / (l1_ct[:, None] + alpha * nlab)
    table = np.log(p_l2_given_l1) - np.log(p_l2)[None, :]
    return np.clip(table, -clip, clip)


def _is_ancestor(heads, start_token, target_token):
    """True if `target_token` lies on `start_token`'s path up to the root, walking the CURRENT
    tree. Both arguments are TOKEN indices (0..n-1) -- `heads[i]` is `i`'s head in HEAD-SPACE
    (0 = virtual root, 1..n = token 0..n-1 shifted by one), so each step must convert back to a
    token index (`v - 1`) before indexing `heads[]` again.

    Used to reject a re-attachment that would create a cycle: re-parenting dependent `d` under
    `h_new` is only safe if token `h_new - 1` is NOT currently a descendant of `d`."""
    v = heads[start_token]                     # head-space: 0 or 1..n
    seen = 0
    while v != 0 and seen <= len(heads):
        tok = v - 1                            # token-space
        if tok == target_token:
            return True
        v = heads[tok]
        seen += 1
    return False


def rerank_with_siblings(S, chosen, heads, sib_table, k=5, max_passes=3):
    """Local-search refinement of a first-order CLE tree using a second-order sibling bonus.

    `S`: (n+1, n) arc scores (head x dependent, virtual root at index 0) -- the SAME matrix already
    used for first-order decoding (`combined.max(-1)` in every prediction path this project has).
    `chosen`: (n+1, n) label id that scored best for each (head, dependent) cell -- `combined.argmax(-1)`.
    `heads`: (n,) the first-order-optimal parent of each dependent (1-indexed into the (n+1)-space,
    0 = virtual root), e.g. `mst(Sq)[1:]` -- MODIFIED IN PLACE and also returned.
    `sib_table`: (nlab, nlab) from `build_sibling_table`.

    Returns `(heads, labels)`, `labels` being the (possibly revised) label choice per dependent --
    revising a dependent's HEAD also revises its label to whatever scored best under the new head
    (`chosen[new_head, d]`), since a different head can favour a different label.

    ⚠ GREEDY, NOT OPTIMAL. Each pass considers one dependent's candidate heads at a time, holding
    every other dependent's current attachment fixed, and accepts the first improving move found --
    the same "coordinate ascent" approximation local search always makes for an intractable joint
    objective. `max_passes` bounds the number of full sweeps; the loop also exits early once a full
    sweep makes no change.
    """
    n = S.shape[1]
    heads = heads.copy()
    labels = np.array([chosen[heads[d], d] for d in range(n)])

    def sib_positions(h):
        """Dependents currently attached to head h, sorted by position (their token index)."""
        return sorted(d for d in range(n) if heads[d] == h)

    def local_sib_bonus(h, deps_sorted, pos, lab_at):
        """Sum of sib_table[l_i, l_{i+1}] over the (at most two) pairs touching `pos` within
        `deps_sorted` (h's dependents, sorted) -- the only pairs affected by changing the label or
        presence of the dependent AT `pos`. `lab_at(d)` looks up d's current label."""
        total = 0.0
        idx = deps_sorted.index(pos) if pos in deps_sorted else None
        if idx is None:
            return 0.0
        if idx > 0:
            total += sib_table[lab_at(deps_sorted[idx - 1]), lab_at(pos)]
        if idx < len(deps_sorted) - 1:
            total += sib_table[lab_at(pos), lab_at(deps_sorted[idx + 1])]
        return total

    lab_at = lambda d: labels[d]

    for _pass in range(max_passes):
        changed = False
        for d in range(n):
            old_h = heads[d]
            old_l = labels[d]
            old_deps = sib_positions(old_h)
            old_arc = S[old_h, d]
            old_sib = local_sib_bonus(old_h, old_deps, d, lab_at)
            old_total = old_arc + old_sib

            best_h, best_l, best_total = old_h, old_l, old_total
            # candidate heads: the top-k by raw arc score (h_new is in the SAME head-space as
            # `heads[]`'s own values -- 0 = virtual root, 1..n = token h_new-1), excluding the
            # current head, self-attachment (h_new-1 == d), and anything that would create a cycle
            # (the token at h_new-1 must not currently be a descendant of d).
            cand_order = np.argsort(-S[:, d])
            tried = 0
            for h_new in cand_order:
                if h_new == old_h or (h_new != 0 and h_new - 1 == d):
                    continue
                if h_new != 0 and _is_ancestor(heads, h_new - 1, d):
                    continue
                tried += 1
                if tried > k:
                    break
                l_new = int(chosen[h_new, d])
                new_deps = sib_positions(h_new) + [d]
                new_deps.sort()

                def lab_at_new(x):
                    return l_new if x == d else labels[x]

                new_arc = S[h_new, d]
                new_sib = local_sib_bonus(h_new, new_deps, d, lab_at_new)
                new_total = new_arc + new_sib
                if new_total > best_total:
                    best_h, best_l, best_total = int(h_new), l_new, new_total

            if best_h != old_h:
                heads[d] = best_h
                labels[d] = best_l
                changed = True
        if not changed:
            break
    return heads, labels
