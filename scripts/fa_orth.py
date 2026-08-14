#!/usr/bin/env python3
"""Persian orthographic variation: vocalisation, ezāfe, letterforms, ZWNJ.

The Persian counterpart of `ar_orth.py`, and it works in the OPPOSITE DIRECTION, which is forced by
the data. Arabic stores the corpus fully pointed and only ever removes marks, because PADT ships
the gold `Vform`. Persian has no vocalised gold anywhere, so the corpus stays as the treebank
writes it and the augmenter ADDS marks, from the same reconstructed table `fa_vocalise` ships
against (`build_fa_vocalise_lut.py`) and the same syntactically-derived ezāfe rules
(`build_fa_ezafe_rules.py`).

⚠ That asymmetry has a consequence worth stating: the Arabic augmentation is exact -- every
spelling it produces is a real subset of a gold annotation -- while the Persian one is only as good
as the reconstruction. A wrong vowel in the table becomes a wrong vowel the parser is trained to
expect. The mitigation is that the parser is being taught to IGNORE these marks, not to predict
them: the target is robustness, so an occasional wrong vowel is noise in the input, not a corrupted
label.

Four axes, all attested in ordinary Persian text:

  * **vocalisation** -- rare in running prose, routine in dictionaries, textbooks, children's books,
    poetry and scripture, and used sporadically anywhere a word would otherwise be ambiguous.
  * **ezāfe** -- the kasra linking head to modifier, written when a writer wants to disambiguate.
    Supplied from the parse, since it is syntactic; consonant-final hosts only, because on a
    vowel-final host the ezāfe is a LETTER and adding one would change the tokenisation.
  * **Arabic vs Persian letterforms** -- `ی`/`ي` and `ک`/`ك` are the single most common real-world
    variation in Persian text, because an Arabic keyboard produces the Arabic codepoints. A model
    that has only seen the Persian ones treats half the internet as out-of-vocabulary.
  * **ZWNJ** -- `می‌رود` is also written `میرود`. Only the JOINING direction is sampled: replacing a
    ZWNJ with a space would split the token and change the tree, which this must never do.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FATHA = "َ"
KASRA = "ِ"
DAMMA = "ُ"
SHADDA = "ّ"
SUKUN = "ْ"
DIAC = set(FATHA + KASRA + DAMMA + SHADDA + SUKUN + "ًٌٍٰ")
ZWNJ = "‌"
PERSIAN_TO_ARABIC = {"ی": "ي", "ک": "ك"}
VOWEL_FINAL = "اوهیآ"
MODES = ("full", "shadda", "sparse")


def strip_diac(s: str) -> str:
    return "".join(c for c in s if c not in DIAC)


def to_arabic_letters(s: str) -> str:
    return "".join(PERSIAN_TO_ARABIC.get(c, c) for c in s)


def drop_zwnj(s: str) -> str:
    return s.replace(ZWNJ, "")


def write_at(v: str, mode: str, rng: random.Random) -> str:
    """Render a vocalised form at one partial regime. `sparse` keeps each mark independently, which
    is what a writer pointing only the ambiguous syllable actually produces -- unlike Arabic, where
    the partial regimes are principled (stem vs case ending), Persian pointing is ad hoc."""
    if mode == "full":
        return v
    if mode == "shadda":
        return "".join(c for c in v if c not in DIAC or c == SHADDA)
    if mode == "sparse":
        return "".join(c for c in v if c not in DIAC or rng.random() < 0.5)
    raise ValueError(mode)


@dataclass(frozen=True)
class Style:
    rate: float          # share of tokens that get any diacritic
    mode: str
    ezafe: bool          # write the ezāfe kasra where the parse licenses it
    arabic: bool         # ی -> ي, ک -> ك
    zwnj: bool           # drop the ZWNJ


@dataclass
class OrthPolicy:
    # As in Arabic, BARE MUST DOMINATE: unpointed text is what Persian normally is and what every
    # published figure for this arm is measured on.
    p_bare: float = 0.45
    p_full: float = 0.10
    min_rate: float = 0.05
    max_rate: float = 0.90
    mode_weights: tuple = (0.4, 0.2, 0.4)      # full, shadda, sparse
    p_ezafe: float = 0.35
    p_arabic: float = 0.20
    p_zwnj: float = 0.20


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
                 ezafe=rng.random() < policy.p_ezafe,
                 arabic=rng.random() < policy.p_arabic,
                 zwnj=rng.random() < policy.p_zwnj)


def vary_word(form: str, vocalised: str | None, add_ezafe: bool,
              style: Style, rng: random.Random) -> str:
    out = form
    if vocalised and rng.random() < style.rate:
        out = write_at(vocalised, style.mode, rng)
    if add_ezafe and style.ezafe and not out.rstrip(ZWNJ).endswith(tuple(VOWEL_FINAL)):
        out = out + KASRA
    if style.arabic:
        out = to_arabic_letters(out)
    if style.zwnj:
        out = drop_zwnj(out)
    return out
