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


# ---------------------------------------------------------------------------------------------
# AGREEMENT-CONSTRAINED DECODING
#
# A Sanskrit adjective or participle agrees with the noun it modifies in case, number AND gender.
# Measured on the Vedic test with GOLD morphology, an `ADJ/participle --mod--> NOUN` arc that
# violates that agreement is a wrong ATTACHMENT 95.1 % of the time (137/144) — the parser produces
# such arcs on 31.3 % of the population where gold does so on 3.0 %, a 10x gap. That makes the
# violation a near-certain error DETECTOR, which is what the earlier post-processing rules all
# lacked: forcing `subj` on nominatives, or the root onto a finite verb, overrode a parser that was
# usually already right, and both lost.
#
# ⚠ GENITIVE CHILDREN ARE EXEMPT AND THE RULE IS WRONG WITHOUT THAT. Genitive attribution
# (`suptasya karṇam`, "the sleeping one's ear") is precisely a modifier that does NOT agree in case
# — that is what the genitive is for. Excluding Gen children halves the gold violation rate,
# 8.0 % -> 3.2 %: more than half of the apparent violations were the rule being wrong, not the tree.
#
# ⚠ IT IS A STRONG PRIOR, NOT A LAW. Gold still violates it ~3 % of the time (gender mismatches on
# gender-ambiguous forms, and some annotation noise), so a hard prohibition is wrong ~3 % of the
# time by construction. And forbidding an arc does not say what to put instead: the parser
# re-decodes and may attach elsewhere, equally wrongly. Whether it NETS positive is an empirical
# question — `eval_agreement_constraint.py` answers it; do not assume from the 95 % figure.
#
# ⚠ MORPHOLOGY ERROR IS PART OF THE SIGNAL. On PREDICTED morphology the violation rate is 41.4 %
# against 31.3 % on gold, so roughly a third of what this fires on is the morphologiser being
# wrong rather than the parser. At inference only predicted features exist, so that is the regime
# the constraint actually runs in.

_ADJ_VERBFORMS = ("Part", "Gdv")


def _feat(token, key):
    v = token.morph.get(key)
    return v[0] if v else None


def is_adjlike(token) -> bool:
    vf = token.morph.get("VerbForm")
    return token.pos_ == "ADJ" or (bool(vf) and vf[0] in _ADJ_VERBFORMS)


def agreement_violation(child, head) -> bool:
    """True iff a `mod` arc from `child` to `head` would break adjectival agreement."""
    if not is_adjlike(child) or head.pos_ != "NOUN":
        return False
    c_case, h_case = _feat(child, "Case"), _feat(head, "Case")
    if c_case is None or h_case is None or c_case == "Gen":
        return False
    return any(_feat(child, k) != _feat(head, k) for k in ("Case", "Number", "Gender"))


def parse_with_agreement(parser, doc):
    """Re-decode `doc`, forbidding `mod` arcs that break adjectival agreement.

    Returns a NEW doc (the input is untouched), or None if the decode could not complete.
    Only `L-mod` / `R-mod` are masked: the parser keeps every other option at every step, so it
    chooses the best tree it can build WITHOUT the offending arc, rather than being patched.
    """
    moves = parser.moves
    names = [moves.get_class_name(j) for j in range(moves.n_moves)]
    # In spaCy's ArcEager a LEFT arc makes S(0) the child of B(0); a RIGHT arc makes B(0) the child
    # of S(0). So the (child, head) pair a candidate action would create depends on its direction.
    left_mod = [j for j, n in enumerate(names) if n == "L-mod"]
    right_mod = [j for j, n in enumerate(names) if n == "R-mod"]

    out = doc.copy()
    states = moves.init_batch([out])
    step = parser.model.predict([out])
    limit, guard, final = 8 * (len(out) + 1) + 32, 0, None
    while states and guard < limit:
        guard += 1
        scores = step.predict(states)
        alive = []
        for state, row in zip(states, scores):
            s0, b0 = state.S(0), state.B(0)
            banned = set()
            if 0 <= s0 < len(out) and 0 <= b0 < len(out):
                if agreement_violation(out[s0], out[b0]):
                    banned.update(left_mod)      # L: child S(0), head B(0)
                if agreement_violation(out[b0], out[s0]):
                    banned.update(right_mod)     # R: child B(0), head S(0)
            order = [int(j) for j in numpy.argsort(-numpy.asarray(row))]
            chosen = next((j for j in order
                           if j not in banned and moves.is_valid(state, names[j])), None)
            if chosen is None:
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

# The relations the CLAUSE MERGE actually introduced, counted on the training split rather than
# assumed: conj:coord 1065, parataxis 852, comp:obj 221, subj 110, mod 103, comp:obl 87. `punct` is
# not one of them but must always be permitted — a mark attaches to the root of the unit on its
# left, and a sentence-final mark therefore reaches back across every earlier mark.
MERGE_RELATIONS = {"conj:coord", "parataxis", "comp:obj", "subj", "mod", "comp:obl"}
ALWAYS_ALLOWED = {"punct"}


def parse_with_clause_bounds(parser, doc, allowed, marks=("\u2016",)):
    """Re-decode `doc`, forbidding arcs that cross a clause mark unless their label is in `allowed`.

    Returns a NEW doc (the input is untouched), or None if the decode could not complete. As in
    `parse_with_agreement`, only the offending ACTIONS are masked: the parser still chooses the best
    tree it can build without them, rather than having its output patched afterwards.

    ⚠ `marks` DEFAULTS TO THE DOUBLE DAṆḌA ALONE, because a single daṇḍa is not a clause boundary.
    It marks a half-verse and sits INSIDE one of the treebank's own clause units (`Punctuation=comma`
    is unit-medial 8 129 times), so arcs legitimately cross it: on the merged test 2 126 gold arcs
    cross a `|`, only 55 % of them coordination, parataxis or punct. Even restricting to `‖`, 41 % of
    the 1 837 crossing arcs are something else — `comp:obj` 231, `flat` 201, `mod@ccomp` 98 — so a
    strict {conj:coord, parataxis} set forbids 758 gold test arcs outright. Pass `allowed` to say
    which trade is wanted; `MERGE_RELATIONS` is the empirically attested set.

    Pseudo-projective composites are matched on the BASE label (the part before `||`), since that is
    the relation the arc actually bears.
    """
    allowed = set(allowed) | ALWAYS_ALLOWED
    moves = parser.moves
    names = [moves.get_class_name(j) for j in range(moves.n_moves)]
    banned_actions = [j for j, n in enumerate(names)
                      if n[:2] in ("L-", "R-") and n[2:].split("||")[0] not in allowed]
    if not banned_actions:
        return doc.copy()

    out = doc.copy()
    mark_at = [t.text in marks for t in out]
    states = moves.init_batch([out])
    step = parser.model.predict([out])
    limit, guard, final = 8 * (len(out) + 1) + 32, 0, None
    while states and guard < limit:
        guard += 1
        scores = step.predict(states)
        alive = []
        for state, row in zip(states, scores):
            s0, b0 = state.S(0), state.B(0)
            banned = set()
            if 0 <= s0 < len(out) and 0 <= b0 < len(out):
                lo, hi = (s0, b0) if s0 < b0 else (b0, s0)
                if any(mark_at[k] for k in range(lo + 1, hi)):
                    banned = set(banned_actions)   # the span is the same either direction
            order = [int(j) for j in numpy.argsort(-numpy.asarray(row))]
            chosen = next((j for j in order
                           if j not in banned and moves.is_valid(state, names[j])), None)
            if chosen is None:
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


def _units(doc, marks):
    """Unit index per token. A mark belongs to the unit on its LEFT, which is where it attaches."""
    unit, u = [], 0
    for t in doc:
        unit.append(u)
        if t.text in marks:
            u += 1
    return unit


def _decode(parser, doc, ban_for):
    """Shared masked decode. `ban_for(s0, b0)` -> (ban_left, ban_right) for the arc each direction
    would create; a banned action is skipped unless nothing else is valid."""
    moves = parser.moves
    names = [moves.get_class_name(j) for j in range(moves.n_moves)]
    lefts = [j for j, n in enumerate(names) if n.startswith("L-")]
    rights = [j for j, n in enumerate(names) if n.startswith("R-")]

    out = doc.copy()
    states = moves.init_batch([out])
    step = parser.model.predict([out])
    limit, guard, final = 8 * (len(out) + 1) + 32, 0, None
    while states and guard < limit:
        guard += 1
        scores = step.predict(states)
        alive = []
        for state, row in zip(states, scores):
            s0, b0 = state.S(0), state.B(0)
            banned = set()
            if 0 <= s0 < len(out) and 0 <= b0 < len(out):
                bl, br = ban_for(s0, b0)
                if bl:
                    banned.update(lefts)
                if br:
                    banned.update(rights)
            order = [int(j) for j in numpy.argsort(-numpy.asarray(row))]
            chosen = next((j for j in order
                           if j not in banned and moves.is_valid(state, names[j])), None)
            if chosen is None:
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


def parse_with_unit_roots(parser, doc, marks=("\u2016",)):
    """Two-pass decode: a unit-crossing arc may only have a UNIT ROOT as its dependent.

    The rule the clause merge itself obeys — an introduced edge always REPLACES a root edge, so the
    dependent of a cross-unit arc is the thing that headed its whole unit. Checked against the gold
    rather than assumed: on the merged test 81.5 %% of unit-crossing content arcs have a dependent
    whose subtree covers its entire unit (train 78.8 %%), against 59 %% for the label-based version
    in `parse_with_clause_bounds`, which is why that one had nothing to trade.

    ⚠ WHY TWO PASSES. The test is not decidable incrementally in ArcEager. A LEFT arc pops its
    child, so the child's subtree IS final at that moment — but a RIGHT arc pushes it, and the child
    can still collect dependents to its right. Cross-unit arcs are overwhelmingly RIGHT arcs (unit
    n's root attaching back into unit n-1), so an incremental test would reject exactly the case
    that matters. Pass 1 therefore forbids EVERY crossing arc, which yields one tree per unit and
    names each unit's root; pass 2 re-decodes allowing a crossing arc only from those roots. The
    second pass is a full re-decode, not a patch, so unit-internal structure may still be revised in
    light of the attachment.

    `marks` is the double daṇḍa alone: a single daṇḍa is a half-verse break INSIDE a unit, not a
    unit boundary (see `parse_with_clause_bounds`).
    """
    unit = _units(doc, marks)
    first = _decode(parser, doc, lambda s0, b0: (unit[s0] != unit[b0],) * 2)
    if first is None:
        return None
    roots = {t.i for t in first if t.head.i == t.i or unit[t.head.i] != unit[t.i]}

    def ban_for(s0, b0):
        if unit[s0] == unit[b0]:
            return False, False
        return s0 not in roots, b0 not in roots      # L: child s0; R: child b0

    return _decode(parser, doc, ban_for)
