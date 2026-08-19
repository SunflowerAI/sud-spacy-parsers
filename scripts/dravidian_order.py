#!/usr/bin/env python3
"""Word-order variation for Tamil and Telugu: a fresh linearisation of the same tree.

Same contract as ``scripts/la_order.py`` — every head, deprel, lemma, tag and feature stays exactly
where it was, and only the string moves — but the constraints are almost the opposite ones, and
that is the whole point of a separate module rather than a parameter on the Latin one.

WHAT THE TREEBANKS ACTUALLY SAY. Measured on the training corpora (828 Tamil TTB+MWTT sentences,
1 051 Telugu MTG sentences), as the share of dependents that PRECEDE their head:

    deprel        ta        te
    subj       99.0 %    99.9 %
    comp       99.7 %    98.6 %
    udep       99.8 %    99.1 %
    mod        94.0 %    96.9 %
    det        98.3 %    96.1 %
    compound   10.0 %   100.0 %
    conj        0.0 %     0.0 %      a conjunct FOLLOWS what it is coordinated with
    root is the last non-punctuation token in 93.7 % / 93.2 % of sentences

So both languages are rigidly HEAD-FINAL, and Latin's central move — generating hyperbaton because
a projective re-linearisation would hand the model a corpus without discontinuity — mostly does not
apply. **Telugu is 99.9 % projective: one crossing sentence in 1 051.** Tamil has 18.0 %. Inventing
displacement in Telugu would be inventing a construction the language does not have, so
``p_hyperbaton`` defaults to 0 there and is nonzero only for Tamil.

WHAT IS ACTUALLY FREE, and it is the thing worth augmenting. Not the position of the head — that is
a real fact about Dravidian and a parser SHOULD learn it. What is free is the ORDER OF SIBLINGS in
the preverbal field, and it is genuinely free rather than merely variable:

    subject before object    ta 157 / 56 (26 % OSV)      te 198 / 59 (23 % OSV)
    mod before udep          ta  69 / 133                te  32 / 25

With 400-1 000 training sentences a parser will memorise the particular orders it happened to see —
that a subject precedes its object, that a temporal adjunct precedes an instrumental — as though
they were rules. This module takes that away and leaves head-finality intact.

**THE SIDE OF THE HEAD IS READ OFF THE DATA, NEVER ASSIGNED.** A child that was originally before
its head stays before it; one that was after stays after. Head-finality therefore falls out of the
corpus rather than being encoded as a rule, and the post-head phenomena the treebanks DO have —
``conj``, right-dislocation, the quotative — survive without needing to be listed. This is the one
design decision that keeps the transform from inventing Dravidian.

**Enclitics.** Tamil's ``-um``/``-ē``/``-ā`` are split off as their own tokens (``mod@emph``, 134 in
train) and are phonologically bound to the word in front. In 130 of those 134 the word in front IS
the syntactic head, but in 4 it is not, so the constraint cannot be expressed on the tree. They
travel as RIDERS on whatever token they originally followed, exactly as Latin's ``-que`` does.

**Coordination is a sequence, not a set.** ``conj`` children keep their original relative order:
permuting conjuncts would change what the sentence means while leaving the tree well-formed, which
is the one class of error no metric here would report.

**Punctuation is not shuffled**: each mark is re-attached to the edge of its head's new span that it
was originally on, so a comma that closed a clause still closes it and the full stop stays last.

⚠ ``clause_only`` DEFAULTS TO TRUE AND IS A GENUINELY OPEN QUESTION. Scrambling in Dravidian is
described as a clausal phenomenon, and prenominal order (DEM > NUM > ADJ > N) is usually called
rigid. The corpora are less clear-cut than that: among nominal heads with two or more pre-head
children, 98 % (ta) / 71 % (te) sit in a label-set the corpus attests in more than one order. That
measure is coarse — it pools across heads and strips subtypes — so it licenses an experiment, not a
conclusion. Hence the knob, and hence both settings being trained rather than one being chosen here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

#: Deprels whose token is a bound enclitic: it rides the token it originally followed, whatever its
#: head is. Tamil splits `-um`/`-e`/`-a` off as `mod@emph`; Telugu's post-verbal `andi`-type
#: particles are annotated `discourse`.
ENCLITIC_DEPRELS = frozenset({"mod@emph", "discourse"})

#: Kept in their original relative order rather than shuffled. A coordination is a sequence.
SEQUENCE_DEPRELS = ("conj", "flat", "list", "compound", "parataxis")

#: Heads whose pre-head children may be scrambled when `clause_only` is set.
CLAUSE_UPOS = frozenset({"VERB", "AUX"})


@dataclass(frozen=True)
class Tok:
    """The only fields the linearisation reads. Everything else rides along untouched."""
    form: str
    lemma: str
    upos: str
    deprel: str
    head: int              # 0-based index into the sentence; -1 for the root
    feats: str = ""
    space_after: bool = True

    @property
    def base_deprel(self) -> str:
        return self.deprel.split("@")[0].split(":")[0]


@dataclass(frozen=True)
class OrderPolicy:
    """How hard the shuffle pushes, and which constraints it honours.

    ``p_sentence`` is deliberately not 1.0, for the reason `la_order.OrderPolicy` gives: real input
    has the order its author chose, and a model that has never seen the treebank's own
    linearisation has been taught that position carries no information at all — which is false,
    just much weaker than an un-augmented arm believes.
    """
    p_sentence: float = 0.5        # sentences re-linearised at all
    p_hyperbaton: float = 0.0      # per free constituent, chance it rises out of its parent's span
    min_len: int = 3               # shorter sentences have nothing to shuffle
    clause_only: bool = True       # scramble only under VERB/AUX heads (see the module docstring)
    respect_enclitics: bool = True
    respect_sequence: bool = True  # conj/flat/list keep their original relative order
    respect_punct: bool = True


#: Tamil has 18.0 % of training sentences carrying a crossing arc, so a purely projective
#: re-linearisation would under-produce discontinuity; Telugu has 0.1 % and must not be given any.
#: `scripts/calibrate_ta_order.py` sets the Tamil rate against the corpus's own statistics.
POLICIES = {
    "ta": OrderPolicy(p_hyperbaton=0.08),
    "te": OrderPolicy(p_hyperbaton=0.0),
}


@dataclass
class Reordered:
    """The permutation, plus the two things that are properties of the ORDER, not of the token."""
    order: list[int]               # new sequence, as indices into the original sentence
    spaces: list[bool]             # space-after in the new order

    @property
    def changed(self) -> bool:
        return self.order != sorted(self.order)


class _Linearizer:
    def __init__(self, toks: list[Tok], rng: random.Random, policy: OrderPolicy):
        self.toks = toks
        self.rng = rng
        self.policy = policy
        self.n = len(toks)
        self.rider_of: dict[int, int] = {}      # enclitic index -> the token it rides
        self.punct_of: dict[int, tuple[int, bool]] = {}   # punct index -> (head, was_after)
        self.kids: list[list[int]] = [[] for _ in range(self.n)]
        self.roots: list[int] = []

    # -- setup -------------------------------------------------------------------------------
    def mark_riders(self) -> None:
        """An enclitic rides the token it originally followed; punctuation rides its head's edge."""
        if self.policy.respect_enclitics:
            for i, tok in enumerate(self.toks):
                if tok.deprel in ENCLITIC_DEPRELS and i > 0:
                    host = i - 1
                    # Chains (`X um e`) collapse onto the first non-enclitic host, so a rider is
                    # never itself a host -- otherwise seating order would depend on dict order.
                    while host in self.rider_of:
                        host = self.rider_of[host]
                    self.rider_of[i] = host
        if self.policy.respect_punct:
            for i, tok in enumerate(self.toks):
                if tok.upos == "PUNCT" and 0 <= tok.head < self.n and tok.head not in self.rider_of:
                    self.punct_of[i] = (tok.head, i > tok.head)

    def build_tree(self) -> bool:
        """Children lists over the tokens that are neither riders nor punctuation.

        Returns False if the sentence is not a usable tree (a cycle, or a head out of range), in
        which case the caller leaves it alone. A permutation built from a broken tree does not
        raise — it yields a well-formed Example with a DIFFERENT sentence — so this is checked.
        """
        skip = set(self.rider_of) | set(self.punct_of)
        for i, tok in enumerate(self.toks):
            if i in skip:
                continue
            h = tok.head
            if h == -1 or h >= self.n or h < -1:
                self.roots.append(i)
            elif h in skip or h == i:
                return False                     # a real head must not be a rider or itself
            else:
                self.kids[h].append(i)
        if not self.roots:
            return False
        # reachability: every non-skipped token must be emitted exactly once
        seen: set[int] = set()
        stack = list(self.roots)
        while stack:
            v = stack.pop()
            if v in seen:
                return False                     # a cycle
            seen.add(v)
            stack.extend(self.kids[v])
        return len(seen) == self.n - len(skip)

    # -- linearisation -----------------------------------------------------------------------
    def scramblable(self, v: int) -> bool:
        if not self.policy.clause_only:
            return True
        return self.toks[v].upos in CLAUSE_UPOS or self.toks[v].head == -1

    def emit(self, v: int) -> tuple[list[int], list[list[int]]]:
        """Linearise the subtree at `v`. Returns (sequence, blocks that RISE to the caller).

        A risen block is one that detached from this subtree and must be re-inserted inside an
        ancestor's span — which is how a projective recursion produces a crossing arc at all.
        """
        pre, post, risen = [], [], []
        for c in self.kids[v]:
            seq, up = self.emit(c)
            risen.extend(up)
            (pre if c < v else post).append((c, seq))

        # Sequence-preserving children keep their slot; the rest of the PRE block is shuffled.
        if self.scramblable(v) and len(pre) > 1:
            fixed = [k for k, (c, _) in enumerate(pre)
                     if self.policy.respect_sequence
                     and self.toks[c].base_deprel in SEQUENCE_DEPRELS]
            movable = [k for k in range(len(pre)) if k not in set(fixed)]
            shuffled = movable[:]
            self.rng.shuffle(shuffled)
            remap = dict(zip(movable, shuffled))
            pre = [pre[remap.get(k, k)] for k in range(len(pre))]

        # Hyperbaton: a free pre-head block may detach and be re-seated inside an ancestor's span.
        # Never a sequence child, and never the only child -- a subtree that loses everything
        # cannot be discontinuous with itself.
        keep = []
        for c, seq in pre:
            if (self.policy.p_hyperbaton > 0 and len(pre) > 1
                    and self.toks[c].base_deprel not in SEQUENCE_DEPRELS
                    and self.rng.random() < self.policy.p_hyperbaton):
                risen.append(seq)
            else:
                keep.append((c, seq))
        pre = keep       # if everything rose, `keep` is empty; the blocks land further up instead

        out: list[int] = []
        for _, seq in pre:
            out.extend(seq)
        out.append(v)
        for _, seq in post:
            out.extend(seq)

        # Re-seat whatever rose from BELOW inside this span, if this is a place it can go: after
        # the first token, so the block sits strictly inside and the parent arc crosses it.
        if risen and len(out) > 1 and self.scramblable(v):
            take, risen = risen, []
            cut = self.rng.randrange(1, len(out))
            for block in take:
                out = out[:cut] + block + out[cut:]
        return out, risen

    def seat_riders(self, seq: list[int]) -> list[int]:
        """Put each enclitic immediately after its host, and each mark back on its head's edge."""
        riders: dict[int, list[int]] = {}
        for i, host in sorted(self.rider_of.items()):
            riders.setdefault(host, []).append(i)
        out: list[int] = []
        for i in seq:
            out.append(i)
            out.extend(riders.get(i, ()))

        for i, (head, was_after) in sorted(self.punct_of.items()):
            # the span of `head` is contiguous by construction, so its edges are well defined
            members = {head} | self._descendants(head) | {
                r for r, h in self.rider_of.items() if h in ({head} | self._descendants(head))}
            positions = [k for k, t in enumerate(out) if t in members]
            if not positions:
                out.append(i)
                continue
            out.insert(positions[-1] + 1 if was_after else positions[0], i)
        return out

    def _descendants(self, v: int) -> set[int]:
        out: set[int] = set()
        stack = list(self.kids[v])
        while stack:
            x = stack.pop()
            out.add(x)
            stack.extend(self.kids[x])
        return out


def reorder_sentence(toks: list[Tok], rng: random.Random,
                     policy: OrderPolicy = OrderPolicy()) -> Reordered:
    """Re-linearise one sentence. Falls back to the identity on anything it cannot parse."""
    n = len(toks)
    identity = Reordered(list(range(n)), [t.space_after for t in toks])
    if n < policy.min_len or rng.random() >= policy.p_sentence:
        return identity
    lin = _Linearizer(toks, rng, policy)
    lin.mark_riders()
    if not lin.build_tree():
        return identity
    seq: list[int] = []
    for root in lin.roots:
        part, risen = lin.emit(root)
        for block in risen:                      # nothing above the root to rise into
            part = part + block
        seq.extend(part)
    seq = lin.seat_riders(seq)
    if sorted(seq) != list(range(n)):
        return identity                          # a permutation bug must not reach the trainer
    return Reordered(seq, _spaces_for(seq, toks))


def _spaces_for(order: list[int], toks: list[Tok]) -> list[bool]:
    """Space-after belongs to the POSITION, not to the token: the last token of a sentence keeps
    the original last token's spacing, and everything else is separated normally. A token that was
    written joined to its neighbour (`SpaceAfter=No`) keeps that only if the neighbour still
    follows it."""
    out = []
    for k, i in enumerate(order):
        if k + 1 == len(order):
            out.append(toks[order[-1]].space_after)
        elif order[k + 1] == i + 1:
            out.append(toks[i].space_after)
        else:
            out.append(True)
    return out


def reorder_doc(toks: list[Tok], sent_starts: list[int], rng: random.Random,
                policy: OrderPolicy = OrderPolicy()) -> Reordered:
    """Re-linearise each sentence of a multi-sentence document independently.

    Sentence boundaries do not move: a permutation inside a sentence leaves every sentence
    occupying the positions it already did, which is what keeps ``SENT_START`` valid.
    """
    bounds = list(sent_starts) + [len(toks)]
    order: list[int] = []
    spaces: list[bool] = []
    for a, b in zip(bounds, bounds[1:]):
        chunk = [Tok(t.form, t.lemma, t.upos, t.deprel,
                     t.head - a if a <= t.head < b else -1, t.feats, t.space_after)
                 for t in toks[a:b]]
        r = reorder_sentence(chunk, rng, policy)
        order.extend(i + a for i in r.order)
        spaces.extend(r.spaces)
    return Reordered(order, spaces)


def crossing_arcs(heads: list[int]) -> int:
    """Number of crossing arc PAIRS in a 0-based head list (-1 for the root). For calibration."""
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads) if h >= 0 and h != i]
    arcs.sort()
    return sum(1 for a, b in arcs for c, d in arcs if a < c < b < d)
