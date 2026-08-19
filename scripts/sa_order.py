#!/usr/bin/env python3
"""Constrained word-order permutation for Sanskrit training data.

Sanskrit's constituent order is famously free, the syntax half is only 163 308 tokens, and a
permutation preserves the tree exactly — so permuting is a way to multiply the data without
inventing annotation. But "randomised word order" taken literally produces strings no one would
write, and the parser is measurably order-sensitive: re-parsing a sentence under a projectivising
permutation costs 3.63 LAS against its OWN reordered gold. So the permutation has to stay inside
what Sanskrit actually does. Every constraint below is measured on corpus_sa_mwt_rl2_norm/train.

GAP DEGREE <= 2. Attested: 77.5 % of sentences projective, 20.4 % gap degree 1, 2.0 % degree 2,
0.1 % beyond. Uniform permutation takes non-projective arcs from 6.7 % to 70.5 % — which is not
augmentation but corruption, since spaCy's parser handles bounded non-projectivity through
pseudo-projective encoding and nothing beyond it.

WACKERNAGEL CLITICS MOVE WITH THEIR HOST. `ca vā hi u eva api tu khalu nu sma vai ha iva` are
11 031 tokens (6.7 % of the corpus) and stand sentence-initial 54 times in 11 031. They are
attached to the preceding unit rather than permuted independently.

`iti` IS RIGHT-ANCHORED: 41.2 % clause-final, 0.7 % clause-initial, following its head 81.3 % of
the time. It closes quoted material, so it keeps the right edge of what it scopes over.

SCONJ IS LEFT-ANCHORED: 75.5 % precede their head, 34.6 % clause-initial, 0.4 % clause-final.

CCONJ PINS BY ADJACENCY, not position: 84.6 % adjacent to their head against 59.7 % for NOUN.
It travels with its conjunct.

COMPOUNDS ARE ATOMIC: members carry `Compound=Yes` and a compound is one orthographic word.

SANDHI IS REGENERATED at the new junctions from the padapāṭha in NORM, using the same engine the
representation already rests on. `scripts/validate_sandhi_dcs.py` measures that engine against DCS's
real editorial text at 91.8 % sentence-exact on attested forms (>98 % per junction), which is what
makes regeneration safe enough to do rather than skip.
"""
from __future__ import annotations

import collections
import random
from typing import List, Optional

CLITICS = {"ca", "vā", "hi", "u", "eva", "api", "tu", "khalu", "nu", "sma", "vai", "ha", "iva"}
MAX_GAP_DEGREE = 2


def _gap_degree(heads: List[int]) -> int:
    kids = collections.defaultdict(list)
    for c, h in enumerate(heads):
        if c != h:
            kids[h].append(c)

    def desc(i):
        out, stack = set(), [i]
        while stack:
            x = stack.pop()
            out.add(x)
            stack += kids[x]
        return out

    worst = 0
    for i in range(len(heads)):
        span = sorted(desc(i))
        worst = max(worst, sum(1 for a, b in zip(span, span[1:]) if b - a > 1))
    return worst


def build_units(toks) -> List[List[int]]:
    """Group token indices into atoms that move together: compounds, clitic+host, CCONJ+conjunct."""
    n = len(toks)
    unit_of = list(range(n))

    def merge(a, b):
        ra, rb = unit_of[a], unit_of[b]
        if ra == rb:
            return
        lo, hi = min(ra, rb), max(ra, rb)
        for i in range(n):
            if unit_of[i] == hi:
                unit_of[i] = lo

    for i, t in enumerate(toks):
        if t["compound"] and i + 1 < n:
            merge(i, i + 1)                      # a compound member binds to what follows it
        if i > 0 and (t["lemma"] in CLITICS or t["norm"] in CLITICS):
            merge(i, i - 1)                      # Wackernagel: attach to the preceding unit
        if t["upos"] == "CCONJ" and i + 1 < n:
            merge(i, i + 1)                      # coordinator travels with its conjunct
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[unit_of[i]].append(i)
    return [groups[k] for k in sorted(groups, key=lambda k: min(groups[k]))]


def reorder(toks, heads, rng: random.Random, tries: int = 8) -> Optional[List[int]]:
    """Return a permutation of token indices, or None if no acceptable one was found.

    The gap degree is SAMPLED to the attested distribution (77.5 / 20.4 / 2.0 %), not merely capped:
    a projective order is built constructively, then perturbed by lifting one subtree when a
    discontinuous target is drawn.
    """
    if len(toks) < 4:
        return None
    r = rng.random()
    want = 0 if r < 0.775 else (1 if r < 0.979 else 2)
    for _ in range(tries):
        base = _projective_order(heads, rng)
        if base is None:
            return None
        if want == 0:
            if base != list(range(len(toks))) and _units_ok(base, toks):
                return base
            continue
        cand = _perturb(base, heads, rng, want, toks)
        if cand is not None and cand != list(range(len(toks))):
            return cand
    return None


def _children(heads):
    """-> (children map, ALL roots). A sentence can have several self-headed tokens: `clause_parser`
    keeps every root a sub-parse produced, and the corpus carries such sentences. Emitting from one
    root silently dropped the others' tokens, which showed up as a KeyError rather than as a
    quietly truncated sentence — lucky, but not something to rely on."""
    kids = collections.defaultdict(list)
    roots = []
    for c, h in enumerate(heads):
        if c == h:
            roots.append(c)
        else:
            kids[h].append(c)
    return kids, roots


def _projective_order(heads, rng) -> Optional[List[int]]:
    """Build a PROJECTIVE linearisation directly, instead of shuffling and hoping.

    Each node emits its subtree as one contiguous block, itself placed at a random position among
    its children's blocks, so projectivity holds by construction and gap degree 0 is reached
    whenever the tree admits it. Shuffling did not manage that: capping at <= 2 gave 27/42/31 % and
    target-sampling 46/29/24 %, against an attested 77.5/20.4/2.0 — degree 2 over-represented
    twelvefold, i.e. a corpus far more discontinuous than the language, which is exactly the
    corruption this constraint set exists to prevent.
    """
    kids, roots = _children(heads)
    if not roots:
        return None

    def emit(i):
        blocks = [[i]] + [emit(c) for c in kids[i]]
        rng.shuffle(blocks)
        return [x for b in blocks for x in b]

    # Root order is NOT shuffled: multiple roots are consecutive clauses, and swapping them would
    # reorder clauses rather than words — a different operation from the one being sampled.
    order = [x for r in roots for x in emit(r)]
    return order if len(order) == len(heads) else None


def _units_ok(order, toks) -> bool:
    """Compounds contiguous and in order; no clitic or `iti` sentence-initial."""
    pos = {t: i for i, t in enumerate(order)}
    for i, t in enumerate(toks):
        if t["compound"] and i + 1 < len(toks) and pos.get(i + 1, -99) != pos.get(i, -1) + 1:
            return False
    first = order[0]
    if toks[first]["lemma"] in CLITICS or toks[first]["norm"] in CLITICS:
        return False
    return toks[first]["lemma"] != "iti"


def _perturb(order, heads, rng, want, toks, tries: int = 30) -> Optional[List[int]]:
    """Lift one subtree out of a projective order and reinsert it, to reach gap degree `want`."""
    kids, _roots = _children(heads)

    def subtree(i):
        out, st = set(), [i]
        while st:
            x = st.pop()
            out.add(x)
            st += kids[x]
        return out

    for _ in range(tries):
        blk = subtree(order[rng.randrange(len(order))])
        rest = [t for t in order if t not in blk]
        piece = [t for t in order if t in blk]
        if not rest or not piece:
            continue
        at = rng.randrange(len(rest) + 1)
        cand = rest[:at] + piece + rest[at:]
        where = {o: n for n, o in enumerate(cand)}
        if _gap_degree([where[heads[o]] for o in cand]) == want and _units_ok(cand, toks):
            return cand
    return None
