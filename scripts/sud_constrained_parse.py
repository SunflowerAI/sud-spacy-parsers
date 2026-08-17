#!/usr/bin/env python3
"""Re-decode a sentence under a ROOT constraint, letting the parser choose everything else.

WHY NOT POST-HOC SURGERY. Reassigning the root by hand leaves the rest of the tree incoherent: the
old root needs a new head and a new relation, and so on outward. The tree has to be re-derived, not
patched.

WHY NOT N-BEST SELECTION. spaCy exposes `beam_parse` + `get_beam_parses`, and the beam DOES contain
a finite-verb-rooted tree 89.7 % of the time (against a 93.0 % ceiling from the treebank). But this
arm was trained with `beam_update_prob = 0.0`, i.e. greedily, so its action scores are not calibrated
as SEQUENCE scores and the beam's ranking is close to meaningless. Measured on 3 295 test tokens:

    greedy                              UAS 0.6850   LAS 0.5475
    beam rank-0 (width 16)              UAS 0.5269   LAS 0.4206
    beam + finite-verb-root selection   UAS 0.5293   LAS 0.4219

A sanity check confirms the machinery, not the usage, is the issue: at `beam_width=1` rank-0
reproduces greedy on 99.6 % of heads, and agreement decays to 84.7 / 73.9 / 56.3 % at widths
2 / 4 / 16. So the beam walks away from the greedy tree and takes 13 LAS with it. Selecting from an
uncalibrated beam is not a route; beam TRAINING would be, at the cost of a retrain.

WHAT THIS DOES INSTEAD. It re-runs the ordinary greedy decode with ONE action masked out: the
designated root may not receive a head. Everything else is the parser's own highest-scoring valid
action at every step, so the result is a coherent tree the parser derived itself — the constraint
only removes derivations, it never invents an arc.

In spaCy's ArcEager a LEFT-ARC attaches S(0) to B(0) and a RIGHT-ARC attaches B(0) to S(0), so
"never give token R a head" is exactly: forbid L-* when S(0) == R, and forbid R-* when B(0) == R.
A token that never receives a head is rendered as its own head with the root label, which is what
`set_annotations` does anyway.

⚠ THE CONSTRAINT MUST BE SATISFIABLE. If masking leaves no valid action the decode would deadlock,
so the mask is dropped for that step rather than allowed to hang (and the caller is told). Shift is
essentially always available, so this is a guard rather than a common path.
"""
from typing import Optional

import numpy


def parse_with_root(parser, doc, root_idx: int):
    """Re-decode `doc` so token `root_idx` receives no head; return a NEW doc, or None on failure.

    `doc` must already have been through everything the parser depends on (tok2vec, and any
    `annotating_components`). The input doc is not modified.
    """
    moves = parser.moves
    names = [moves.get_class_name(j) for j in range(moves.n_moves)]
    is_left = [n.startswith("L-") for n in names]
    is_right = [n.startswith("R-") for n in names]

    out = doc.copy()
    states = moves.init_batch([out])
    step = parser.model.predict([out])
    limit = 8 * (len(out) + 1) + 32     # ArcEager is linear in tokens; this only catches a pathology
    guard = 0
    final = None
    while states and guard < limit:
        guard += 1
        scores = step.predict(states)
        alive = []
        for state, row in zip(states, scores):
            s0, b0 = state.S(0), state.B(0)
            order = [int(j) for j in numpy.argsort(-numpy.asarray(row))]
            chosen = next(
                (j for j in order
                 if not ((is_left[j] and s0 == root_idx) or (is_right[j] and b0 == root_idx))
                 and moves.is_valid(state, names[j])),
                None)
            if chosen is None:
                # The constraint is unsatisfiable at this step. Drop it rather than deadlock —
                # a decode that hangs is worse than one that occasionally fails to constrain.
                chosen = next((j for j in order if moves.is_valid(state, names[j])), None)
            if chosen is None:
                return None
            moves.transition(state, names[chosen])
            if state.is_final():
                final = state
            else:
                alive.append(state)
        states = alive
    if final is None:
        return None
    moves.set_annotations(final, out)
    return out
