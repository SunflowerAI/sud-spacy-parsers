#!/usr/bin/env python3
"""Arabic orthographic variation: how much of the vocalisation is written, and how.

The Arabic counterpart of `la_orth.py`, and it rests on the same trick. Latin trains on the
MACRONISED copy and derives the plain spelling by stripping, because stripping is exact -- so the
marked copy is a strict superset and the unmarked one never has to be stored. Arabic has the same
relation between `Vform` and the surface: **removing diacritics is exact and monotone**, so a
corpus built with FORM = Vform contains every lighter spelling as a derivation.

    fold(strip(Vform)) == fold(FORM) on 97.50 % of 223 881 train tokens

⚠ The two folds are not tidying, they are the other two axes, and both are real variation rather
than noise:

  * **hamza.** PADT's raw FORM writes `اميركية` where its Vform writes `أَمِيرِكِيَّةٍ` -- the
    vocalised column RESTORES the hamza the running text omits. Both spellings are ordinary
    written Arabic, so which one appears is a style, not an error.
  * **digits.** `Vform` uses Arabic-Indic `١٥` where FORM uses `15`, on 9 765 train tokens. Again
    both are current.

The residual 2.5 % is genuine orthographic divergence between PADT's raw text and its normalised
vocalisation. It is not lost: those are alternative spellings and the sampling shows the model
both neighbourhoods anyway.

VOCALISATION IS NOT A BINARY, which is the whole point of doing this by sampling rather than by
training on two copies. Real Arabic runs the entire ladder -- bare newswire, fully pointed
scripture and children's books, and in between a great deal of prose that writes a shadda here and
a case ending there, marking only what would otherwise be ambiguous. `Style.mode` covers the
attested partial regimes:

    full      everything
    internal  the stem marked, the case ending (iʿrāb) left to the reader -- very common
    final     only the case ending
    shadda    only the shadda, the single most frequently written mark
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# ⚠ NAMED ONE BY ONE, NEVER unpacked from a single string literal. These are combining marks that
# render on a dotted circle in most editors, so `FATHA, DAMMA, KASRA, SUKUN, SHADDA = "..."` is
# unreadable AND was wrong: the literal's 4th character is U+0651 SHADDA and its 5th U+0652 SUKUN,
# so the last two names were bound to each other's marks. Same defect, same line, as the one this
# module's sibling `ar_vocalise.py` shipped with. The fa modules name each mark and were fine.
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
DIAC = set(FATHA + DAMMA + KASRA + SUKUN + SHADDA + FATHATAN + DAMMATAN + KASRATAN + "ٰٕٔٓ")
HAMZA_FOLD = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
MODES = ("full", "internal", "final", "shadda")


def strip_diac(s: str) -> str:
    return "".join(c for c in s if c not in DIAC)


def fold_hamza(s: str) -> str:
    return "".join(HAMZA_FOLD.get(c, c) for c in s)


def fold_digits(s: str) -> str:
    return s.translate(AR_DIGITS)


def split_ending(v: str) -> tuple[str, str]:
    """(stem, case ending). The ending is the run of diacritics after the LAST consonant, which is
    where iʿrāb lives -- so `internal` and `final` are complements of each other by construction."""
    idx = max((i for i, c in enumerate(v) if c not in DIAC), default=-1)
    return v[:idx + 1], v[idx + 1:]


def write_at(v: str, mode: str) -> str:
    """Render the fully vocalised form `v` at one partial-vocalisation regime."""
    if mode == "full":
        return v
    if mode == "shadda":
        return "".join(c for c in v if c not in DIAC or c == SHADDA)
    stem, ending = split_ending(v)
    if mode == "final":
        return strip_diac(stem) + ending
    if mode == "internal":
        return stem
    raise ValueError(mode)


@dataclass(frozen=True)
class Style:
    """One document's writing convention."""
    rate: float          # share of tokens that carry any diacritic at all
    mode: str            # which marks a vocalised token keeps
    hamza: bool          # fold أ إ آ -> ا, i.e. write the hamza-less running-text spelling
    digits: bool         # fold ١٥ -> 15


@dataclass
class OrthPolicy:
    # BARE MUST DOMINATE. Undiacritised text is the overwhelming majority of written Arabic and is
    # what every published figure for this arm is measured on, so the augmentation has to widen the
    # model's range without moving its centre of mass off the spelling it is actually judged on.
    p_bare: float = 0.40
    p_full: float = 0.15
    min_rate: float = 0.05
    max_rate: float = 0.95
    mode_weights: tuple = (0.45, 0.25, 0.15, 0.15)   # full, internal, final, shadda
    p_hamza_fold: float = 0.5
    p_digit_fold: float = 0.5


def sample_style(rng: random.Random, policy: OrthPolicy) -> Style:
    r = rng.random()
    if r < policy.p_bare:
        rate = 0.0
    elif r < policy.p_bare + policy.p_full:
        rate = 1.0
    else:
        rate = rng.uniform(policy.min_rate, policy.max_rate)
    return Style(rate=rate,
                 mode=rng.choices(MODES, weights=policy.mode_weights)[0],
                 hamza=rng.random() < policy.p_hamza_fold,
                 digits=rng.random() < policy.p_digit_fold)


def vary_word(vocalised: str, style: Style, rng: random.Random) -> str:
    """`vocalised` is the token as the corpus stores it: fully pointed. Everything lighter is a
    derivation, so this only ever REMOVES marks (or folds a carrier), never invents one."""
    out = write_at(vocalised, style.mode) if rng.random() < style.rate else strip_diac(vocalised)
    if style.hamza:
        out = fold_hamza(out)
    if style.digits:
        out = fold_digits(out)
    return out
