#!/usr/bin/env python3
"""`sud.ko_order_variants.v1` — constrained scrambling for Korean training data.

WHAT KOREAN'S FREEDOM ACTUALLY IS, measured by `scripts/calibrate_ko_order.py` on the training
treebank. Korean is rigidly head-final — `mod` 96.6 %, `subj` 98.4 % and `comp:obj` 97.1 % of
dependents precede their head — and the relations that do not (`flat`, `conj:coord`, `conj:appos`)
are head-initial because SUD chains coordination, not because Korean puts them there. So the
vertical dimension is fixed and there is exactly one degree of freedom worth augmenting: **the order
of a head's pre-head dependents among themselves**, available at 27.1 % of heads.

⚠ AND IT IS NOT UNIFORM. That is the whole design:

    subj  before comp:obj    96.1 %          mod before subj      47.7 %
    mod   before comp:obj    78.3 %          mod before udep      59.1 %

A uniform shuffle would teach `comp:obj` before `subj` at 50 % against an attested 3.9 %, i.e. would
spend most of its augmented data on orders Korean does not use. This augmenter SAMPLES from the
corpus's own bigram distribution over sibling relations (`scripts/ko_order_bigrams.json`) instead,
so a permutation is drawn in proportion to how attested that ordering of relations is.

⚠ NON-PROJECTIVE SENTENCES ARE LEFT ALONE. Rebuilding the string from the tree projectivises it,
which would change which arcs cross — and a crossing arc is what spaCy's pseudo-projective encoding
turns into a `||`-suffixed label, so projectivising the input silently rewrites the LABELS the
parser is trained on. 11.75 % of Korean sentences contain one; they pass through in their attested
order, and only the projective remainder is permuted.

⚠ PUNCTUATION NEVER MOVES, and neither do the head-initial relations. Marks keep their slots, which
is also what makes the SPACING correct without recomputing it: `SpaceAfter=No` in this treebank
means "the next token is a mark", and if the mark has not moved the property still holds of that
position. `--check` asserts the string it produces re-tokenises to the same token count.

WHAT IT IS FOR, and the honest prior. Latin's word-order augmentation collapsed the LAS spread
across word orders from 17.44 to 8.38 and bought **+0.13 on natural order**; Sanskrit's bought
+1.70 over three seeds, most plausibly as small-data regularisation. Korean's measured
order-sensitivity is −5.1 LAS under full sibling scrambling (`scripts/eval_ko_scramble.py`), a third
of Latin's, so the robustness argument is weak here — but ko trains on 56 687 tokens, the smallest
treebank in the set, so the regularisation argument is the strongest here. Which of those dominates
is a measurement, not a prediction.
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import Dict, Iterator, List, Optional, Tuple

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import registry

BOS = "<s>"
#: Head-initial by SUD's own convention rather than by Korean word order — permuting them would
#: produce a different ANNOTATION, not a different sentence.
FIXED_DEPS = {"flat", "conj", "conj:coord", "conj:appos", "conj:dicto", "punct", "cc"}
SMOOTH = 0.5


def load_table(path) -> Tuple[Dict[str, float], Dict[str, float]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise ValueError(
            f"sud.ko_order_variants.v1: {p} not found. Build it with "
            f"scripts/calibrate_ko_order.py. Refusing to fall back to a uniform shuffle, which "
            f"would teach `comp:obj` before `subj` at 50 % against an attested 3.9 %.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw["bigrams"], raw["contexts"]


def _nonprojective(heads: List[int]) -> bool:
    """An arc h->d is non-projective iff some token strictly between them is not a descendant of h.
    Computed here rather than imported so the gold side and this side are measured by one code."""
    kids: Dict[int, List[int]] = {}
    for c, h in enumerate(heads):
        if c != h:
            kids.setdefault(h, []).append(c)

    def descendants(i):
        out, stack = set(), [i]
        while stack:
            x = stack.pop()
            out.add(x)
            stack.extend(kids.get(x, ()))
        return out

    for d, h in enumerate(heads):
        if d == h:
            continue
        lo, hi = (h, d) if h < d else (d, h)
        if hi - lo < 2:
            continue
        desc = descendants(h)
        if any(k not in desc for k in range(lo + 1, hi)):
            return True
    return False


def _sample_order(deps: List[str], rng: random.Random, bigrams, contexts) -> List[int]:
    """Order the indices of `deps` by drawing each next relation from P(d | previous relation).

    `bigrams = None` is the UNIFORM sampler, which is not an augmentation recipe but the worst case
    an evaluation needs: `eval_ko_scramble.py` reports both, because how far an arm falls under
    orders Korean does not use is a different question from how far it falls under orders it does.
    """
    if bigrams is None:
        perm = list(range(len(deps)))
        rng.shuffle(perm)
        return perm
    remaining = list(range(len(deps)))
    out: List[int] = []
    prev = BOS
    while remaining:
        weights = []
        for i in remaining:
            n = bigrams.get(f"{prev}\t{deps[i]}", 0)
            weights.append(n + SMOOTH)
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        pick = remaining[-1]
        for i, w in zip(remaining, weights):
            acc += w
            if r <= acc:
                pick = i
                break
        out.append(pick)
        remaining.remove(pick)
        prev = deps[pick]
    return out


def _relinearise(sent, rng: random.Random, p_head: float, bigrams, contexts) -> Optional[List[int]]:
    """Doc-global indices in their new order, or None if this sentence is left alone."""
    start = sent.start
    heads = [t.head.i - start for t in sent]
    if _nonprojective(heads):
        return None
    kids: Dict[int, List[int]] = {}
    for c, h in enumerate(heads):
        if c != h:
            kids.setdefault(h, []).append(c)

    # ⚠ NO MARK MAY CHANGE ITS ABSOLUTE POSITION, and keeping each one in its sibling SLOT is not
    # enough to achieve that: swapping two subtrees of unequal length shifts everything between
    # them. A quotation mark is the case that shows why it matters — `'정정당당한 야구'다` opens a
    # bracket, and a modifier permuted out of it strands the quote against a word it does not
    # belong to. So permutation is confined to a punct-FREE region: a head may have its pre-head
    # dependents resampled only when the whole span from its leftmost pre-head descendant to itself
    # contains no mark. `check_ko_order.py` asserts on the marks and on the spacing.
    lo = list(range(len(sent)))                      # leftmost descendant, in the ATTESTED order

    def bounds(i: int) -> int:
        for k in kids.get(i, ()):
            lo[i] = min(lo[i], bounds(k))
        return lo[i]

    for r in (i for i, h in enumerate(heads) if h == i):
        bounds(r)
    n_punct = [0] * (len(sent) + 1)                  # prefix sums, so the span test is O(1)
    for i, t in enumerate(sent):
        n_punct[i + 1] = n_punct[i] + (t.pos_ == "PUNCT")

    touched = False

    def order(i: int) -> List[int]:
        nonlocal touched
        ks = sorted(kids.get(i, ()))
        pre, post = [k for k in ks if k < i], [k for k in ks if k > i]
        movable = [k for k in pre if sent[k].dep_.split("@")[0] not in FIXED_DEPS]
        if movable and n_punct[i] - n_punct[min(lo[k] for k in pre)] > 0:
            movable = []                             # a mark stands in the region: leave it alone
        if len(movable) > 1 and rng.random() < p_head:
            deps = [sent[k].dep_.split("@")[0] for k in movable]
            perm = _sample_order(deps, rng, bigrams, contexts)
            reordered = [movable[j] for j in perm]
            if reordered != movable:
                touched = True
            slots = {old: new for old, new in zip(movable, reordered)}
            pre = [slots.get(k, k) for k in pre]
        out: List[int] = []
        for k in pre:
            out.extend(order(k))
        out.append(i)
        for k in post:
            out.extend(order(k))
        return out

    roots = [i for i, h in enumerate(heads) if h == i]
    seq: List[int] = []
    for r in sorted(roots):
        seq.extend(order(r))
    if len(seq) != len(sent) or sorted(seq) != list(range(len(sent))):
        return None                       # a malformed tree: leave it exactly as it is
    if not touched:
        return None
    return [start + i for i in seq]


def order_example(nlp: Language, example: Example, rng: random.Random, p_head: float,
                  bigrams, contexts) -> Example:
    """Re-linearise the projective sentences of one example. The TREE is untouched — heads are
    re-indexed through the permutation, so every arc still joins the same two words — and so is
    every annotation on every token. Only the string moves."""
    ref = example.reference
    n = len(ref)
    order = list(range(n))
    changed = False
    for sent in ref.sents:
        seq = _relinearise(sent, rng, p_head, bigrams, contexts)
        if seq is None:
            continue
        order[sent.start:sent.end] = seq
        changed = True
    if not changed:
        return example

    where = {old: new for new, old in enumerate(order)}
    data = example.to_dict()
    ta = data["token_annotation"]
    for key in ("ORTH", "LEMMA", "POS", "TAG", "MORPH", "DEP"):
        if key in ta:
            ta[key] = [ta[key][i] for i in order]
    ta["HEAD"] = [where[ta["HEAD"][i]] for i in order]
    # SPACY is a property of the POSITION, not of the token: `SpaceAfter=No` here means "a mark
    # follows", and marks never move. SENT_START is untouched for the same reason — a permutation
    # never crosses a sentence.
    ents = data["doc_annotation"]["entities"]
    data["doc_annotation"]["entities"] = [ents[i] for i in order]

    # ⚠ SPACING IS CARRIED BY THE TOKEN, NOT THE POSITION. `SpaceAfter=No` looks positional — it
    # nearly always means "a mark follows" — but a permutation that swaps two subtrees of unequal
    # length moves the mark's absolute index while leaving it in the same sibling slot, and a
    # positional spacing array would then put a space before it. So each token records whether it is
    # GLUED to what precedes it, and the array is rebuilt from that.
    glued = [False] * n
    for i in range(1, n):
        glued[i] = not ref[i - 1].whitespace_
    words = [ref[i].text for i in order]
    spaces = [not glued[order[k + 1]] for k in range(n - 1)] + [bool(ref[order[-1]].whitespace_)]
    ta["ORTH"] = words
    ta["SPACY"] = spaces
    predicted = Doc(nlp.vocab, words=words, spaces=spaces)
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.ko_order_variants.v1")
def create_ko_order_augmenter(p_order: float = 0.5, p_head: float = 0.5,
                              table: str = "scripts/ko_order_bigrams.json", seed: int = 0):
    """`p_order` is the chance an example is re-linearised at all, `p_head` the chance a given head
    has its pre-head dependents resampled. The rest pass through in their attested order, so the
    model never stops seeing the Korean people actually write."""
    rng = random.Random(seed)
    bigrams, contexts = load_table(table)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if rng.random() < p_order:
            yield order_example(nlp, example, rng, p_head, bigrams, contexts)
        else:
            yield example

    return augmenter
