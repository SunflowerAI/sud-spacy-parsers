#!/usr/bin/env python3
"""Latin orthographic variation, as functions on a word form.

Latin is printed in more than one orthography, and the differences are systematic rather than
random: an edition either marks vowel length or does not, either distinguishes the glides as
``j``/``v`` or writes them ``i``/``u``, either ligates ``ae``/``oe`` or leaves them apart, and
either capitalises the first word of a sentence or does not. A parser trained on one convention has
no reason to handle another, and the SUD Latin treebanks do not agree among themselves: **ITTB
writes ``u`` throughout and capitalises no sentence at all**, PROIEL and Perseus write ``v`` and
capitalise 16 % / 28 % of sentence openings, and nothing anywhere writes ``j`` or a ligature.

This module is the transform half; ``scripts/la_augment.py`` is the training-time half that samples
from it, and ``scripts/make_la_variant_conllu.py`` the evaluation half that applies it
deterministically to a test set.

The four axes, and what each one rests on:

``macrons``   Removable but not inventable, so the macronised treebank
              (``*.macron.conllu``, Alatius via ``macronise_la.py``) is the source and the plain
              spelling is derived. Exact: ``strip_length`` reproduces the plain FORM on 586 604 /
              586 604 training tokens.
``breves``    A breve marks a SHORT vowel, so the candidates are exactly the vowels Alatius left
              unmarked -- minus the members of a diphthong, which have no independent length, and
              minus the glides, which are not vowels at all.
``u``/``v``   Needs to know which ``u`` is the consonant, which is lexical (``silua`` -> *silva*
              but ``minuere`` stays vocalic). Harvested from the v-writing treebanks by
              ``build_la_glide_lut.py``; see there for the held-out numbers.
``i``/``j``   Has NO in-corpus evidence -- not one ``j`` in 586 604 tokens -- so this axis alone is
              a rule: ``i`` is the consonant when it opens the word or follows a syllabic vowel and
              a vowel follows it. Three things keep it honest: nothing precedes ``i``, since Latin
              has no ``ji`` (which is what saves ``iis``, ``ii``, ``iit``); the ``u`` of ``qu``/
              ``ngu`` and a glide ``u`` are not syllabic vowels (which is what saves ``quia`` and
              ``uiam``, i.e. *viam*); and the ``eo`` "go" family is excluded by LEMMA, because
              ``iens``/``ierunt`` are shaped exactly like ``iecit`` and only the lexeme tells them
              apart.
``ae``/``oe`` Ligated only where the pair is a real diphthong, which the macrons already record:
              Alatius writes hiatus ``āēr``, ``āeris``, ``poētae`` but diphthongal ``aere``,
              ``caelum``, so an unmarked literal ``ae`` is the diphthong and a marked one is not.

Every function preserves case and is a no-op on a form with no lowercase letter, which is what
keeps Roman numerals (``V``, ``XIV``, ``I``) out of the glide transforms.
"""
from __future__ import annotations

import gzip
import json
import random
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

MACRON = "̄"
BREVE = "̆"

_LONG = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "y": "ȳ"}
_SHORT = {"a": "ă", "e": "ĕ", "i": "ĭ", "o": "ŏ", "u": "ŭ"}
LONG = {**_LONG, **{k.upper(): v.upper() for k, v in _LONG.items()}}
SHORT = {**_SHORT, **{k.upper(): v.upper() for k, v in _SHORT.items()}}

LIGATURE = {"ae": "æ", "oe": "œ", "Ae": "Æ", "Oe": "Œ", "AE": "Æ", "OE": "Œ"}
UNLIGATURE = {"æ": "ae", "œ": "oe", "Æ": "Ae", "Œ": "Oe"}

VOWEL_BASES = set("aeiouy")
#: pairs that are a single vowel nucleus, so neither half takes a breve of its own
DIPHTHONGS = ("ae", "oe", "au")
#: only these are ligated -- ``au`` has no ligature and ``eu`` is hiatus in most Latin words
LIGATABLE = ("ae", "oe")

#: forms of ``eo`` "go" and its compounds: the one place the i/j rule needs a lexical exception,
#: since ``iens``/``ierunt``/``ieram`` are shaped like ``iecit``/``iacere`` but keep a vocalic i.
_EO_STEMS = ("eo", "ire", "queo")
_EO_PREFIXES = ("", "ab", "ad", "ambi", "ante", "circum", "co", "com", "de", "dis", "ex", "in",
                "inter", "intro", "ne", "ob", "per", "prae", "praeter", "pro", "red", "re", "sub",
                "subter", "super", "trans", "ue", "ve")
EO_LEMMAS = frozenset(p + s for p in _EO_PREFIXES for s in _EO_STEMS)

_LUT_PATH = Path(__file__).resolve().parent / "la_glide_lut.json.gz"
_lut: dict | None = None


def _load_lut() -> dict:
    """The glide tables, loaded once.

    Without them every ``u`` is left vocalic, so the u/v axis quietly stops varying anything and an
    augmented run trains on less than it says it does. It is gitignored (derived from the treebanks,
    rebuilt in seconds), so its absence is a NORMAL state for a fresh clone and has to be loud --
    the rest of this project has been bitten too often by an input that went missing without saying
    so. Degrade, but say it once.
    """
    global _lut
    if _lut is None:
        try:
            with gzip.open(_LUT_PATH, "rt", encoding="utf8") as fh:
                _lut = dict(json.load(fh))
        except OSError:
            warnings.warn(
                f"{_LUT_PATH.name} is missing, so no `u` will be treated as the consonant and the "
                "u/v axis of the augmenter is inert. Build it with "
                "`.venv/bin/python scripts/build_la_glide_lut.py`.",
                RuntimeWarning, stacklevel=2)
            _lut = {"lex": {}, "ctx": {}}
    return _lut


# ---------------------------------------------------------------- length marks


def _decompose(s: str) -> list[tuple[str, str]]:
    """Each character as (base, marks), so length marks can be read and rewritten positionally."""
    out: list[tuple[str, str]] = []
    for ch in s:
        d = unicodedata.normalize("NFD", ch)
        out.append((d[0], "".join(c for c in d[1:] if c in (MACRON, BREVE))))
    return out


def strip_length(s: str) -> str:
    """Drop macrons and breves, keeping every other diacritic (Greek accents in the treebank)."""
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in (MACRON, BREVE)))


def is_marked(ch: str) -> bool:
    return any(c in (MACRON, BREVE) for c in unicodedata.normalize("NFD", ch)[1:])


def base_of(ch: str) -> str:
    return unicodedata.normalize("NFD", ch)[0]


def has_macron(s: str) -> bool:
    return MACRON in unicodedata.normalize("NFD", s)


def _put(word: str, i: int, ch: str) -> str:
    return word[:i] + ch + word[i + 1:]


# ---------------------------------------------------------------- glides


def _key(word: str) -> str:
    return strip_length(word).lower().replace("v", "u")


def _context(word: str, i: int) -> str:
    prv = word[i - 1] if i > 0 else "^"
    nxt = word[i + 1] if i + 1 < len(word) else "$"
    prv2 = (word[i - 2] if i > 1 else "^") if prv == "u" else "."
    return f"{prv2}{prv}{nxt}"


def glide_u_positions(word: str) -> set[int]:
    """Indices of ``u``/``v`` in ``word`` that spell the CONSONANT.

    A ``v`` already in the form is taken at face value -- the writer marked it. Otherwise the
    harvested lexicon answers if it has the word, and the context rule if it does not.
    """
    if not any(c.islower() for c in word):
        return set()
    key = _key(word)
    positions = [i for i, c in enumerate(key) if c == "u"]
    if not positions:
        return set()
    if "v" in word.lower():
        return {i for i, c in enumerate(word.lower()) if c == "v"}
    lut = _load_lut()
    mask = lut["lex"].get(key)
    if mask is not None and len(mask) == len(positions):
        return {p for p, m in zip(positions, mask) if m == "1"}
    ctx = lut["ctx"]
    return {p for p in positions if ctx.get(_context(key, p), 0)}


def _cluster_u(key: str, i: int) -> bool:
    """``u`` written after ``q`` (``quia``) or ``ngu`` (``lingua``): neither glide nor syllable."""
    prv = key[i - 1] if i > 0 else ""
    prv2 = key[i - 2] if i > 1 else ""
    return prv == "q" or (prv == "g" and prv2 == "n")


def consonantal_i_positions(word: str, lemma: str = "") -> set[int]:
    """Indices of ``i`` in ``word`` that spell the CONSONANT (would be written ``j``)."""
    if not any(c.islower() for c in word):
        return set()
    if lemma and _key(lemma) in EO_LEMMAS:
        return set()
    key = _key(word)
    chars = _decompose(key)
    glides = glide_u_positions(word)
    raw = _decompose(word)

    def syllabic(i: int) -> bool:
        b = chars[i][0]
        if b not in VOWEL_BASES:
            return False
        if b == "u":
            return i not in glides and not _cluster_u(key, i)
        return True

    out = set()
    for i, (b, _) in enumerate(chars):
        if b != "i" or raw[i][1]:            # a length-marked i is a vowel by definition
            continue
        if i + 1 >= len(chars) or chars[i + 1][0] not in VOWEL_BASES:
            continue
        if chars[i + 1][0] == "i":           # Latin has no "ji"
            continue
        if i == 0:
            out.add(i)
        elif chars[i - 1][0] != "i" and syllabic(i - 1):
            out.add(i)
    return out


# ---------------------------------------------------------------- the four axes


def set_macrons(word: str, keep: bool) -> str:
    return word if keep else strip_length(word)


def diphthong_spans(word: str) -> list[tuple[int, str]]:
    """Start index and text of each real diphthong: an UNMARKED pair, since Alatius marks the
    hiatus (``āēr``, ``poētae``) and leaves the genuine diphthong bare (``caelum``, ``aere``)."""
    out = []
    n = len(word)
    for i in range(n - 1):
        if is_marked(word[i]) or is_marked(word[i + 1]):
            continue
        pair = (base_of(word[i]) + base_of(word[i + 1])).lower()
        if pair in DIPHTHONGS:
            out.append((i, pair))
    return out


def breve_candidates(word: str, lemma: str = "") -> list[int]:
    """Positions a breve could go: an unmarked vowel that is neither half of a diphthong nor a
    glide. ``y`` is left out for want of a precomposed breve."""
    if not any(c.islower() for c in word):
        return []
    skip = set()
    for i, _ in diphthong_spans(word):
        skip.update((i, i + 1))
    skip |= glide_u_positions(word) | consonantal_i_positions(word, lemma)
    key = _key(word)
    skip |= {i for i in range(len(key)) if key[i] == "u" and _cluster_u(key, i)}
    return [i for i, ch in enumerate(word)
            if i not in skip and not is_marked(ch) and base_of(ch).lower() in SHORT]


def add_breve(word: str, rng: random.Random, lemma: str = "") -> str:
    cands = breve_candidates(word, lemma)
    if not cands:
        return word
    i = rng.choice(cands)
    return _put(word, i, SHORT[base_of(word[i])])


def set_uv(word: str, use_v: bool | None) -> str:
    """Write the consonantal ``u`` as ``v`` or as ``u``, leaving the vocalic ones alone.

    ``None`` leaves the word's own choice standing, which is what makes a "macrons only" or
    "capitals only" style genuinely single-axis over a corpus that is not itself uniform.
    """
    if use_v is None or not any(c.islower() for c in word):
        return word
    glides = glide_u_positions(word)
    if not glides:
        return word
    out = word
    for i in glides:
        ch = out[i]
        if use_v:
            out = _put(out, i, "V" if ch.isupper() else "v")
        elif base_of(ch).lower() == "v":
            out = _put(out, i, "U" if ch.isupper() else "u")
    return out


def set_ij(word: str, use_j: bool | None, lemma: str = "") -> str:
    """Write the consonantal ``i`` as ``j``. ``False`` and ``None`` are the same instruction here:
    no source in this project writes ``j``, so there is never one to undo."""
    if not use_j or not any(c.islower() for c in word):
        return word
    out = word
    for i in consonantal_i_positions(word, lemma):
        out = _put(out, i, "J" if out[i].isupper() else "j")
    return out


def set_ligature(word: str, use_lig: bool | None) -> str:
    if use_lig is None:
        return word
    if use_lig:
        out = []
        i = 0
        spans = {s for s, p in diphthong_spans(word) if p in LIGATABLE}
        while i < len(word):
            lig = LIGATURE.get(word[i:i + 2]) if i in spans else None
            if lig:
                out.append(lig)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return "".join(out)
    return "".join(UNLIGATURE.get(ch, ch) for ch in word)


def set_initial_case(word: str, upper: bool | None) -> str:
    """Capitalise or lowercase the first letter only -- ``AE`` stays ``AE``, ``V`` stays ``V``."""
    if upper is None or not word or not word[0].isalpha():
        return word
    if upper:
        return word[0].upper() + word[1:]
    if not any(c.islower() for c in word):     # all-caps: a numeral or a siglum, not a word
        return word
    return word[0].lower() + word[1:]


# ---------------------------------------------------------------- sampling


@dataclass(frozen=True)
class Style:
    """One edition's spelling habits, held constant over a document.

    The glide and ligature choices are genuinely all-or-nothing -- no edition writes *vita* on one
    line and *uita* on the next. Length marking is not: a school text marks every long vowel, a
    commentary marks only the vowels that disambiguate a form, and most print marks none, so the
    rate is a number rather than a flag.
    """
    use_v: bool | None = None
    use_j: bool | None = None
    use_lig: bool | None = None
    macron_rate: float = 0.0       # share of words written with their length marks
    breve_rate: float = 0.0        # share of those that also carry a breve on a short vowel
    capitalise: bool | None = None  # is the first word of a sentence capitalised

    #: ``None`` on any of the three flags means "leave the source's own spelling", which is how a
    #: single-axis evaluation style is expressed; ``sample_style`` never produces one.


@dataclass(frozen=True)
class OrthPolicy:
    """How ``sample_style`` draws a document's ``Style``.

    ``p_uniform_length`` is the load-bearing one. Drawing the macron rate uniformly would mean the
    model almost never sees a page that is *consistently* marked or *consistently* unmarked -- which
    is what nearly all real input is, and what the two-copy union arm trained on exclusively. So
    half the documents get an all-or-nothing rate and the rest a mixture. ``p_capital`` sits at
    0.5 against the treebanks' own 0 % (ITTB) / 16 % (PROIEL) / 28 % (Perseus); the ligature is
    rarer than the glide conventions because it is rarer in print.
    """
    p_v: float = 0.5
    p_j: float = 0.5
    p_lig: float = 0.25
    p_capital: float = 0.5
    p_length: float = 0.5           # documents that mark vowel length at all
    p_uniform_length: float = 0.5   # of those, the share that mark it uniformly
    p_breve_doc: float = 0.3        # length-marking documents that use breves at all
    max_breve_rate: float = 0.5
    protect_propn: bool = True      # never lowercase a gold PROPN at a sentence opening


def sample_style(rng: random.Random, policy: OrthPolicy = OrthPolicy()) -> Style:
    if rng.random() >= policy.p_length:
        macron_rate = 0.0
    elif rng.random() < policy.p_uniform_length:
        macron_rate = 1.0
    else:
        macron_rate = rng.random()
    breve_rate = (rng.random() * policy.max_breve_rate
                  if macron_rate and rng.random() < policy.p_breve_doc else 0.0)
    return Style(use_v=rng.random() < policy.p_v,
                 use_j=rng.random() < policy.p_j,
                 use_lig=rng.random() < policy.p_lig,
                 macron_rate=macron_rate,
                 breve_rate=breve_rate,
                 capitalise=rng.random() < policy.p_capital)


def vary_word(word: str, style: Style, rng: random.Random, lemma: str = "") -> str:
    """One word through every axis except capitalisation, which is a property of the sentence.

    Order matters: the glide decisions read the unligated spelling, the breve must not land on a
    diphthong the ligature is about to fuse, and the ligature changes the string's length, so it
    goes last.
    """
    marks_length = rng.random() < style.macron_rate
    out = set_macrons(word, marks_length)
    out = set_uv(out, style.use_v)
    out = set_ij(out, style.use_j, lemma)
    # A breve only belongs in a text that marks length at all, so it rides on the same decision --
    # including for a word that happens to have no long vowel of its own to show for it.
    if marks_length and rng.random() < style.breve_rate:
        out = add_breve(out, rng, lemma)
    return set_ligature(out, style.use_lig)
