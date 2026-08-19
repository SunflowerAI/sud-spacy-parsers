#!/usr/bin/env python3
"""Latin word-order variation: a fresh linearisation of the same tree.

Latin word order is famously free, and the Latin arm's three treebanks are not free in the same
way — ITTB is scholastic prose with a strong verb-final habit, PROIEL is narrative, Perseus is
classical poetry with heavy discontinuity. A parser trained on that mixture still learns a
POSITIONAL prior it has no right to: that a subject precedes its verb, that a genitive follows its
noun. This module is the transform that takes it away — it re-linearises the sentence, **leaving
every head, deprel, lemma, tag and feature exactly where it was**, so the gold tree is untouched
and only the string moves.

``scripts/la_augment.py`` samples from it at training time (composed with the orthographic
augmenter, order first so the orthography pass re-cases the new sentence opening);
``scripts/make_la_scrambled_conllu.py`` applies it deterministically to a test set, which is how an
arm gets scored on an order it has never seen.

Free is not the same as random, and three things would be destroyed by a uniform shuffle.

**Wackernagel.** A handful of particles are enclitic on the first word of their clause and occur
essentially nowhere else. Measured on the 40 305 training sentences, as the particle's rank inside
the contiguous range of its own head's yield:

    lemma        n     pos0   pos1   pos2   pos3   pos4+
    autem     6 523     1.7   76.4   20.7    0.6     0.6
    enim      3 599     0.2   85.0   14.1    0.3     0.4
    igitur    2 875     3.6   79.2   16.8    0.1     0.3
    uero        638     0.6   70.5   24.3    2.7     1.9
    quoque       69     1.4   79.7    7.2    7.2     4.3
    ---------------------------------------------------- the ones that only LOOK like the above
    ergo      1 251    26.2   60.0   12.6    0.0     1.2
    quidem      821     1.3   50.7   30.7    4.8    12.5
    tamen     1 009     9.8   39.3   15.1    1.5    34.3

The first five are re-seated at position 2 of their clause after the shuffle. ``ergo``, ``quidem``
and ``tamen`` are NOT: they look like Wackernagel particles and are described as such, but in this
data they are mobile, and a rule at 39 % would be inventing an order rather than preserving one.
They go through the ordinary shuffle.

**Enclitics proper.** ``-que`` and ``-ve`` are split off as their own tokens (1 517 in train, every
one of them a childless ``cc``), and they are phonologically bound to the word in front — which is
NOT their syntactic head in 43 % of cases, so the tree cannot express the constraint. They travel
as riders on the token they originally followed, keeping the spacing they had (only 152 of the
1 517 are written joined; the rest the treebanks space out, and either way the pair is preserved).

**Prepositions, subordinators, relativisers, coordinators.** SUD makes the adposition the HEAD of
its complement, so "prepositions come first" is just "the head slot goes first in its own subtree" —
one line, and it falls out projectively. Same for SCONJ, which heads its clause. Relativisers pull
their block to the left edge of the clause they open (stopping at the next clause boundary, so a
relative inside an embedded clause does not drag the whole thing forward), and ``cc`` goes to the
left edge of the conjunct it marks, which reproduces the corpus's own 81 % / 93 % / 97 % initial
rate for ``et`` / ``atque`` / ``at``.

**Hyperbaton is generated, not merely tolerated.** 37.75 % of the training sentences contain a
crossing arc. A recursive re-linearisation is projective BY CONSTRUCTION, so shuffling without
displacement would hand the model a corpus in which discontinuity does not exist, while 38 % of the
test set has it. So a constituent may be detached from its parent's block and re-inserted inside an
ancestor's span (``p_hyperbaton``, with ``p_rise`` deciding how far up it travels) — never breaking
a glue pair, so ``in`` keeps its complement and a host keeps its enclitic.
``scripts/calibrate_la_order.py`` sets the rate against the corpus's own crossing statistics.

Punctuation is not shuffled: each mark is re-attached to the edge of its head's new span that it
was originally on, so a comma that closed a clause still closes it and the full stop stays last.
"""
from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

MACRON = "̄"
BREVE = "̆"


def fold(s: str) -> str:
    """Lowercase and drop length marks, so a lemma list matches a macronised treebank too."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if c not in (MACRON, BREVE))


#: Split-off enclitics. Bound to the word in front, whatever their head is.
#:
#: ``ne`` is NOT one of them, and assuming it was cost 535 detached clitics before the full corpus
#: was looked at. Every ``ne`` token in train is the ADV/SCONJ ``nē`` "lest, not" (deprel ``mod``,
#: ``comp:obj``, never ``cc``), and 499 of them are the FIRST half of ``neque``, split as
#: ``nē`` + ``que``. Treating it as an enclitic made it ride the word in front of it and tore
#: ``neque`` in two. The interrogative ``-ne`` is simply not split anywhere in these treebanks.
ENCLITICS = frozenset({"que", "ue", "ve"})

#: Second position in their clause in >=70 % of occurrences (table in the module docstring).
WACKERNAGEL = frozenset({"autem", "enim", "uero", "vero", "igitur", "quoque"})

#: Connectives that open their clause and are not attached as ``cc`` (which is handled generically).
CONNECTIVE_INITIAL = frozenset({"nam", "namque", "at", "atqui", "itaque", "siquidem", "porro",
                                "enimuero", "enimvero", "ceterum", "caeterum"})

#: The adpositions Latin puts AFTER their complement. Everything else goes in front.
POSTPOSITIONS = frozenset({"causa", "gratia", "tenus", "instar", "ergo", "versus", "uersus"})

#: A relativiser stops pulling leftward at one of these: the next clause down is not its clause.
_FINITE = "VerbForm=Fin"

#: Where each unit sits in its parent's ordering. Only units of EQUAL rank are shuffled against
#: each other, so the scale is what keeps the closed-class constraints from colliding: a
#: coordinator precedes a preposition (``et in urbe``, never ``in et urbe``), and a preposition
#: precedes the relative it pied-pipes (``de quo``, from a child that would otherwise front).
RANK_CC = 0          # `et`, `atque`, `nam` — left edge of the conjunct they mark
RANK_HEAD_FIRST = 1  # the ADP / SCONJ / relativiser that heads this subtree
RANK_REL = 2         # a child block that opens a relative clause
RANK_FREE = 3        # everything else, including an ordinary head slot — shuffled
RANK_CONJ = 4        # a conjunct follows what it is coordinated with
RANK_HEAD_LAST = 5   # a postposition


@dataclass(frozen=True)
class Tok:
    """The only fields the linearisation reads. Everything else rides along untouched."""
    form: str
    lemma: str
    upos: str
    deprel: str
    head: int          # 0-based index into the sentence; -1 for the root
    feats: str = ""
    space_after: bool = True

    @property
    def is_rel(self) -> bool:
        return "PronType=Rel" in self.feats

    @property
    def is_finite(self) -> bool:
        return _FINITE in self.feats

    @property
    def base_deprel(self) -> str:
        """``comp:obl@agent`` -> ``comp``; the subtype never changes a placement decision."""
        return self.deprel.split("@", 1)[0].split(":", 1)[0]


@dataclass(frozen=True)
class OrderPolicy:
    """How hard the shuffle pushes, and which closed-class constraints it honours.

    ``p_sentence`` is deliberately not 1.0. Real Latin input has the order its author chose, and a
    model that has never seen the treebank's own linearisation has been taught that position
    carries no information at all — which is false, just much weaker than an un-augmented arm
    believes. Half verbatim, half re-linearised.
    """
    p_sentence: float = 0.5        # sentences re-linearised at all
    p_hyperbaton: float = 0.12     # per free constituent, chance it detaches from its parent
    p_rise: float = 0.4            # a detached block travels one level further up
    min_len: int = 3               # shorter sentences have nothing to shuffle
    respect_adp: bool = True
    respect_sconj: bool = True
    respect_rel: bool = True
    respect_cc: bool = True
    respect_conj: bool = True      # a conjunct follows what it is coordinated with
    respect_wackernagel: bool = True
    respect_enclitics: bool = True
    respect_punct: bool = True
    recase: bool = True            # move the sentence-initial capital onto the new first word


@dataclass
class Reordered:
    """The permutation, plus the two things that are properties of the ORDER rather than the token."""
    order: list[int]               # new sequence, as indices into the original sentence
    forms: list[str]               # forms in the new order, re-cased
    spaces: list[bool]             # space-after in the new order

    @property
    def changed(self) -> bool:
        return self.order != sorted(self.order)


# ---------------------------------------------------------------- spacing

#: Which side of a boundary owns a missing space. A punctuation mark or an enclitic owns it (they
#: attach to their neighbour); anything else and the RIGHT token owns it, which is the enclitic
#: case again with the list unread.
def _attachments(toks: list[Tok]) -> tuple[list[bool], list[bool]]:
    n = len(toks)
    left = [False] * n           # attaches to whatever precedes it, with no space
    right = [False] * n          # whatever follows it attaches, with no space
    for i in range(n - 1):
        if toks[i].space_after:
            continue
        nxt = toks[i + 1]
        if nxt.upos == "PUNCT" or fold(nxt.form) in ENCLITICS:
            left[i + 1] = True
        elif toks[i].upos == "PUNCT":
            right[i] = True
        else:
            left[i + 1] = True
    return left, right


def _spaces_for(order: list[int], toks: list[Tok]) -> list[bool]:
    left, right = _attachments(toks)
    out = []
    for k, i in enumerate(order):
        if k == len(order) - 1:
            # The trailing space belongs to the sentence BOUNDARY, not to whichever token now
            # happens to sit last, so it keeps the value the original last token carried.
            out.append(toks[-1].space_after)
        else:
            j = order[k + 1]
            out.append(not (right[i] or left[j]))
    return out


# ---------------------------------------------------------------- the linearisation


class _Linearizer:
    def __init__(self, toks: list[Tok], rng: random.Random, policy: OrderPolicy):
        self.toks = toks
        self.rng = rng
        self.pol = policy
        self.n = len(toks)
        self.riders: dict[int, str] = {}          # index -> rider kind
        self.no_break: set[int] = set()           # nothing may be inserted right after these
        self.skip_for_wackernagel: set[int] = set()   # not a "first word" for Wackernagel counting
        self.kids: dict[int, list[int]] = {}
        self.root = 0

    # -- riders ---------------------------------------------------------

    def _child_count(self) -> list[int]:
        counts = [0] * self.n
        for i, t in enumerate(self.toks):
            if 0 <= t.head < self.n and t.head != i:
                counts[t.head] += 1
        return counts

    def mark_riders(self) -> None:
        """Tokens taken out of the shuffle and re-seated afterwards.

        Only CHILDLESS tokens may ride: a rider contributes no block, so one with dependants would
        take them out of the linearisation with it.
        """
        counts = self._child_count()
        pol = self.pol
        for i, t in enumerate(self.toks):
            if counts[i]:
                continue
            if pol.respect_punct and t.upos == "PUNCT":
                self.riders[i] = "punct"
            elif (pol.respect_enclitics and i > 0 and fold(t.form) in ENCLITICS
                  and t.upos == "CCONJ" and t.base_deprel == "cc"
                  and self.toks[i - 1].upos != "PUNCT"):
                self.riders[i] = "enclitic"
            elif pol.respect_wackernagel and fold(t.lemma) in WACKERNAGEL:
                self.riders[i] = "wackernagel"

    # -- the skeleton tree ----------------------------------------------

    def build_tree(self) -> bool:
        """Children lists over the non-rider tokens, with heads re-pointed past any rider.

        Returns False if there is no usable skeleton (everything rode, or the root itself did).
        """
        skeleton = [i for i in range(self.n) if i not in self.riders]
        if len(skeleton) < 2:
            return False

        def lift(i: int) -> int:
            """Nearest non-rider ancestor of ``i``, or -1."""
            seen = set()
            h = self.toks[i].head
            while 0 <= h < self.n and h not in seen:
                if h not in self.riders:
                    return h
                seen.add(h)
                h = self.toks[h].head
            return -1

        self.kids = {i: [] for i in skeleton}
        roots = []
        for i in skeleton:
            h = lift(i)
            if h < 0 or h == i:
                roots.append(i)
            else:
                self.kids[h].append(i)
        if len(roots) != 1:
            return False                          # a forest: leave the sentence alone
        self.root = roots[0]
        return True

    # -- placement classes ----------------------------------------------

    def rel_pull(self, i: int, memo: dict[int, bool]) -> bool:
        """Does ``i``'s block open a relative clause, so that it belongs at the clause's left edge?

        Stops at the next clause down (a finite verb or an SCONJ): a relative pronoun inside an
        embedded clause is that clause's business, not this one's.
        """
        if i in memo:
            return memo[i]
        t = self.toks[i]
        memo[i] = True if t.is_rel else False
        if not memo[i]:
            for c in self.kids[i]:
                ct = self.toks[c]
                if ct.upos == "SCONJ" or ct.is_finite:
                    continue
                if self.rel_pull(c, memo):
                    memo[i] = True
                    break
        return memo[i]

    def head_slot_rank(self, i: int) -> int:
        """Where the head's own token sits among its children's blocks; see the rank scale above."""
        t = self.toks[i]
        pol = self.pol
        if t.upos == "ADP" and pol.respect_adp:
            return RANK_HEAD_LAST if fold(t.lemma) in POSTPOSITIONS else RANK_HEAD_FIRST
        if t.upos == "SCONJ" and pol.respect_sconj:
            return RANK_HEAD_FIRST
        if pol.respect_rel and t.is_rel:
            return RANK_HEAD_FIRST
        if pol.respect_cc and (t.base_deprel == "cc" or fold(t.lemma) in CONNECTIVE_INITIAL):
            return RANK_HEAD_FIRST
        return RANK_FREE

    def child_rank(self, c: int, memo: dict[int, bool]) -> int:
        """Where a child's whole block sits; see the rank scale above."""
        t = self.toks[c]
        pol = self.pol
        if pol.respect_cc and (t.base_deprel == "cc" or fold(t.lemma) in CONNECTIVE_INITIAL):
            return RANK_CC
        if pol.respect_rel and self.rel_pull(c, memo):
            return RANK_REL
        if pol.respect_conj and t.base_deprel == "conj":
            return RANK_CONJ
        return RANK_FREE

    # -- recursion --------------------------------------------------------

    def emit(self, v: int, memo: dict[int, bool], is_root: bool) -> tuple[list[int], list[list[int]]]:
        """Linearise ``v``'s subtree. Returns (sequence, blocks asking to be placed higher up).

        The two lists are kept apart on purpose: ``incoming`` is what the children sent up and may
        be re-seated HERE, ``rising`` is what leaves for the parent. Merging them lets this node's
        own detached constituent be re-inserted into the very block it was detached from, which is
        a no-op that looks like hyperbaton in the rate counter and is not.
        """
        rng, pol = self.rng, self.pol
        incoming: list[list[int]] = []
        units: list[tuple[int, list[int], bool]] = []     # (rank, tokens, floatable)

        hrank = self.head_slot_rank(v)
        if hrank == RANK_HEAD_FIRST:
            self.no_break.add(v)                          # in/ut/et keep whatever follows them
            if self.toks[v].upos == "ADP":
                self.skip_for_wackernagel.add(v)
        units.append((hrank, [v], False))

        for c in self.kids[v]:
            seq, cf = self.emit(c, memo, is_root=False)
            incoming.extend(cf)
            rank = self.child_rank(c, memo)
            if rank == RANK_CC:
                self.skip_for_wackernagel.add(c)
            units.append((rank, seq, rank in (RANK_FREE, RANK_CONJ)))

        ordered: list[tuple[int, list[int], bool]] = []
        for rank in (RANK_CC, RANK_HEAD_FIRST, RANK_REL, RANK_FREE, RANK_CONJ, RANK_HEAD_LAST):
            group = [u for u in units if u[0] == rank]
            rng.shuffle(group)                            # only equal ranks compete
            ordered.extend(group)

        # Hyperbaton: detach one free constituent and let an ancestor re-seat it.
        rising: list[list[int]] = []
        if not is_root and len(ordered) > 1 and rng.random() < pol.p_hyperbaton:
            candidates = [k for k, u in enumerate(ordered) if u[2]]
            if candidates:
                rising.append(ordered.pop(rng.choice(candidates))[1])

        seq = [t for _, tokens, _ in ordered for t in tokens]

        # Re-seat whatever the children sent up, unless it is still climbing. The root places
        # everything: there is nowhere further to go.
        for block in incoming:
            if not is_root and rng.random() < pol.p_rise:
                rising.append(block)
                continue
            slots = [k for k in range(1, len(seq)) if seq[k - 1] not in self.no_break]
            if slots:
                k = rng.choice(slots)
                seq[k:k] = block
            else:
                seq.extend(block)
        return seq, rising

    # -- re-seating the riders --------------------------------------------

    def span_of(self, i: int, pos: dict[int, int]) -> tuple[int, int]:
        """The new first and last position of ``i``'s subtree, over the skeleton only."""
        stack, lo, hi = [i], self.n, -1
        while stack:
            v = stack.pop()
            if v in pos:
                lo, hi = min(lo, pos[v]), max(hi, pos[v])
            stack.extend(self.kids.get(v, ()))
        return (lo, hi) if hi >= 0 else (0, len(pos) - 1)

    def _past_enclitics(self, seq: list[int], slot: int) -> int:
        """Push a candidate slot past any enclitic already seated there.

        Enclitics go in first and are invisible to ``span_of`` (they are riders, so they are not in
        the children map), so a later insertion computed off a subtree edge can land BETWEEN a host
        and its clitic -- which is how ``solidābit.que`` came out of an earlier draft.
        """
        while slot < len(seq) and self.riders.get(seq[slot]) == "enclitic":
            slot += 1
        return slot

    @staticmethod
    def _splice(seq: list[int], pending: list[tuple[int, int]]) -> list[int]:
        """Insert right-to-left so earlier slots stay valid; one slot keeps the original order."""
        out = list(seq)
        for slot, i in sorted(pending, key=lambda p: (-p[0], -p[1])):
            out.insert(min(max(slot, 0), len(out)), i)
        return out

    def seat_riders(self, seq: list[int]) -> list[int]:
        """Put the riders back, TIGHTEST BINDING FIRST.

        The three classes are seated in three passes rather than one, because they are not
        independent: an enclitic is part of the word in front of it, so it is already there when
        Wackernagel counts to position two, and punctuation is not a word at all, so it is seated
        last and never displaces a particle that was counting words.
        """
        # 1. enclitics -- part of the host's phonological word
        pos = {t: k for k, t in enumerate(seq)}
        pending = []
        for i in sorted(k for k, v in self.riders.items() if v == "enclitic"):
            slot = pos.get(i - 1)                     # the token it originally followed
            if slot is None:                          # the host rode too: use its syntactic head
                slot = pos.get(self.toks[i].head)
            pending.append((len(seq) if slot is None else slot + 1, i))
        seq = self._splice(seq, pending)

        # 2. Wackernagel particles -- second WORD of their clause
        pos = {t: k for k, t in enumerate(seq)}
        pending = []
        for i in sorted(k for k, v in self.riders.items() if v == "wackernagel"):
            h = self.toks[i].head
            lo, hi = self.span_of(h, pos) if h in pos else (0, len(seq) - 1)
            slot = lo + 1
            # A preposition or a coordinator is not the first word of the clause for this purpose:
            # `in autem urbe` and `et autem` do not occur, where `cum autem` and `quī autem` are
            # the ordinary order -- which is why only those two classes are skipped over.
            while slot <= hi and seq[slot - 1] in self.skip_for_wackernagel:
                slot += 1
            pending.append((self._past_enclitics(seq, min(slot, len(seq))), i))
        seq = self._splice(seq, pending)

        # 3. punctuation -- to the edge of its head's span that it was already on
        pos = {t: k for k, t in enumerate(seq)}
        pending = []
        for i in sorted(k for k, v in self.riders.items() if v == "punct"):
            h = self.toks[i].head
            if h in pos:
                lo, hi = self.span_of(h, pos)
                pending.append((self._past_enclitics(seq, lo if i < h else hi + 1), i))
            else:
                # Its head rode too (punctuation on punctuation). Keep it next to the token it
                # originally followed, which is always defined.
                prev = next((j for j in range(i - 1, -1, -1) if j in pos), None)
                pending.append((0 if prev is None
                                else self._past_enclitics(seq, pos[prev] + 1), i))
        return self._splice(seq, pending)


# ---------------------------------------------------------------- entry points


def reorder_sentence(toks: list[Tok], rng: random.Random,
                     policy: OrderPolicy = OrderPolicy()) -> Reordered:
    """Re-linearise ONE sentence. The tree is not touched — only the order of the tokens."""
    n = len(toks)
    identity = Reordered(list(range(n)), [t.form for t in toks], [t.space_after for t in toks])
    if n < policy.min_len or rng.random() >= policy.p_sentence:
        return identity

    lin = _Linearizer(toks, rng, policy)
    lin.mark_riders()
    if not lin.build_tree():
        return identity
    seq, leftover = lin.emit(lin.root, {}, is_root=True)
    for block in leftover:                            # the root places everything; belt and braces
        seq.extend(block)
    order = lin.seat_riders(seq)
    if len(order) != n or len(set(order)) != n:       # a bug here would silently drop tokens
        raise AssertionError(f"la_order lost tokens: {len(set(order))} of {n}")
    if order == identity.order:
        return identity

    forms = [toks[i].form for i in order]
    if policy.recase:
        forms = _recase(order, toks, forms)
    return Reordered(order, forms, _spaces_for(order, toks))


def _recase(order: list[int], toks: list[Tok], forms: list[str]) -> list[str]:
    """Move the sentence's opening capital onto whatever now opens it.

    A proper noun keeps its capital wherever it lands; a word that is all upper case is a numeral
    or a siglum, not a capitalised word, and is left alone.
    """
    old_first, new_first = 0, order[0]
    if old_first == new_first:
        return forms
    was_capital = toks[old_first].form[:1].isupper()
    out = list(forms)
    k_old = order.index(old_first)
    if was_capital and toks[old_first].upos != "PROPN" and any(c.islower() for c in out[k_old]):
        out[k_old] = out[k_old][:1].lower() + out[k_old][1:]
    if was_capital and out[0][:1].isalpha():
        out[0] = out[0][:1].upper() + out[0][1:]
    return out


def reorder_doc(toks: list[Tok], sent_starts: list[int], rng: random.Random,
                policy: OrderPolicy = OrderPolicy()) -> Reordered:
    """Re-linearise each sentence of a multi-sentence document independently.

    Sentence boundaries do not move: a permutation inside a sentence leaves every sentence
    occupying the positions it already did, which is what keeps ``SENT_START`` valid.
    """
    bounds = list(sent_starts) + [len(toks)]
    order: list[int] = []
    forms: list[str] = []
    spaces: list[bool] = []
    for a, b in zip(bounds, bounds[1:]):
        chunk = [Tok(t.form, t.lemma, t.upos, t.deprel,
                     t.head - a if a <= t.head < b else -1, t.feats, t.space_after)
                 for t in toks[a:b]]
        r = reorder_sentence(chunk, rng, policy)
        order.extend(i + a for i in r.order)
        forms.extend(r.forms)
        spaces.extend(r.spaces)
    return Reordered(order, forms, spaces)


def crossing_arcs(heads: list[int]) -> int:
    """Number of crossing arc PAIRS in a 0-based head list (-1 for the root). For calibration."""
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads) if h >= 0 and h != i]
    arcs.sort()
    return sum(1 for a, b in arcs for c, d in arcs if a < c < b < d)
