#!/usr/bin/env python3
"""A Korean morphological analyser behind one function, for use as a PARSER INPUT.

WHY KOREAN NEEDS ONE. `ko_sud_gsd` is tokenised by eojeol — a stem with its particles and endings
fused into one whitespace-delimited token — so the same lexeme appears as a fresh string under every
particle stack it takes. Measured on the released arm (`scripts/eval_ko_oov.py`):

    seen  7 645 tok (65.5 %)   UAS 75.95   LAS 71.84
    OOV   4 032 tok (34.5 %)   UAS 52.16   LAS 38.10

**A third of test tokens are strings the parser has never met, and they parse 33.7 LAS below the
rest.** That is not a vocabulary shortage — 64.2 % of the STEMS inside those unseen eojeol are in
the training data already, sitting there unreachable behind a particle. Cutting each token to its
stem drops the out-of-vocabulary rate from 34.5 % to 12.4 %.

WHY AN ANALYSER RATHER THAN THE ARM'S OWN LEMMATISER. The lemmatiser recovers the stem on 97.8 % of
seen tokens and **52.6 % of unseen ones** — it fails exactly where it would be needed, because an
edit-tree lemmatiser trained on 56 687 tokens has no more evidence about an unseen eojeol than the
parser does. A dictionary-driven analyser does: the stems are in its dictionary and the particle and
ending inventory is closed, so it segments a form it has never seen in a corpus.

⚠ THE OBJECTION THIS LAYER HAS TO ANSWER. `sud_lex_embed.py` records the reason a per-form table is
usually worthless: a table keyed on the form is a FUNCTION of the form, so conditioning on
(form, f(form)) is conditioning on the form, which the parser already reads. That argument is
airtight for a form the model has a trained representation of, and it is exactly what fails for an
OOV eojeol — an unseen string hashes to an untrained row and carries nothing the model can use,
while its stem is a DIFFERENT symbol with a trained one. So the prediction is specific and
falsifiable: any gain must sit on the OOV tokens. `eval_ko_oov.py` reports the split per arm, and
if the gain is spread evenly across seen and unseen tokens then this channel is not doing what it
was built to do, whatever the headline says.

WHY IT MUST BE A RUNTIME CALL, NOT A SHIPPED TABLE. `sud_analyser_embed.py` reached the same
conclusion for Sanskrit by a different route (a frozen extract missed 6.5 % of test tokens whose
forms the analyser recognised perfectly well, and widening it 4x barely moved that). Here the
argument is sharper still: the tokens this channel exists for are BY DEFINITION the ones absent
from any corpus-derived key set, so a frozen table would answer for every token except the ones
that need answering. It would also load cleanly and score like its own capacity control.

⚠ RECORD THE REGIME AND READ IT BACK (CLAUDE.md hazard 10). Two analysers do not agree on
segmentation, so a model trained against one and deployed against another is being fed a channel it
never saw. `fingerprint()` returns the backend and its tagset, `sud.KoAnalyserEmbed.v1` stores it in
the model bytes at training time, and the forward pass refuses on a mismatch rather than parsing
quietly worse.

NO JACKKNIFING, unlike `sud_lex_embed.py`'s corpus lexicon: this table comes from outside the
treebank, so a form's training-time and inference-time answers are the same answer and there is no
leakage to fold away.

Backends, tried in this order:

  `natto-py` + mecab-ko  the development machine (MECAB_PATH=/opt/homebrew/lib/libmecab.dylib);
                         Homebrew mecab-ko, which conflicts with the Japanese mecab
  `python-mecab-ko`      the SHIPPABLE route — a pip wheel that vendors mecab-ko and mecab-ko-dic,
                         so a user needs no Homebrew and no MECAB_PATH

Both are mecab-ko-dic, whose tagset IS the Sejong tagset the treebank annotates in, so no tag
mapping stands between the analyser and the XPOS column: `잡스는` comes back `잡스/NNP + 는/JX`
against a gold `잡스+는` / `NNP+JX`.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

# form -> [(morpheme, tag), ...]. Module level rather than a Model attr: thinc would try to
# serialise a Model attr, and this is a pure function of its key plus the backend.
_CACHE: dict = {}
_BACKEND = None
_BACKEND_NAME = None

# The mecab-ko-dic feature CSV, by position:
#   0 tag  1 semantic class  2 final-consonant  3 reading  4 type  5 first tag  6 last tag
#   7 expression
_TAG, _READING, _TYPE, _EXPR = 0, 3, 4, 7

# `Inflect` (설계해내지만 -> 해 = 하/XSV + 아/EC), `Compound` and `Preanalysis` all put their
# decomposition in the expression field. Expanding it is what makes the analyser's output line up
# with the treebank's own `+`-joined analysis instead of merely resembling it.
_EXPANDABLE = {"Inflect", "Compound", "Preanalysis"}


class AnalyserUnavailable(RuntimeError):
    pass


#: `KO_ANALYSER_BACKEND=python-mecab-ko` (or `natto-py`) pins the choice. It exists for the
#: verification in `scripts/check_ko_backends.py` — a machine with both installed must be able to
#: run each in turn — and for a user who has both and wants the one their model was trained against.
_ENV = "KO_ANALYSER_BACKEND"


def _load_backend():
    """The first backend that imports wins. Never returns a do-nothing analyser: a silent backend
    would make every token read 'unanalysed' and the arm would score like its capacity control."""
    global _BACKEND, _BACKEND_NAME
    if _BACKEND is not None:
        return _BACKEND
    want = os.environ.get(_ENV)
    tried = []
    try:
        if want and want != "natto-py":
            raise ImportError(f"{_ENV}={want}")
        from natto import MeCab  # type: ignore

        tagger = MeCab()

        def _natto(form: str):
            out = []
            for node in tagger.parse(form, as_nodes=True):
                # `is_nor()` alone drops the UNKNOWN nodes, and mecab-ko-dic treats bare digit runs
                # as unknown: `8년을` came back `년 + 을` with the `8` silently gone, which is worse
                # than no analysis because the stem then names the classifier.
                if node.is_bos() or node.is_eos():
                    continue
                out.append((node.surface, node.feature.split(",")))
            return out

        # natto's __del__ runs after cffi has torn its type cache down, so every process that
        # touches it ends with a TypeError traceback on a successful run. Dropping the tagger at
        # exit keeps a driver's log honest — a spurious traceback is one a reader learns to ignore.
        import atexit

        atexit.register(lambda: globals().update(_BACKEND=None) or tagger.__dict__.clear())
        _BACKEND, _BACKEND_NAME = _natto, "natto-py/mecab-ko-dic"
        return _BACKEND
    except Exception as e:  # natto raises for a missing libmecab, not only ImportError
        tried.append(f"natto-py ({type(e).__name__}: {e})")
    try:
        if want and want != "python-mecab-ko":
            raise ImportError(f"{_ENV}={want}")
        import mecab  # type: ignore

        tagger = mecab.MeCab()

        def _pymecab(form: str):
            out = []
            for tok in tagger.parse(form):
                surface = getattr(tok, "surface", None) or tok[0]
                feature = getattr(tok, "feature", None) or tok[1]
                if not isinstance(feature, str):
                    # python-mecab-ko hands back a Feature dataclass; its str() is the CSV.
                    feature = str(feature)
                out.append((surface, feature.split(",")))
            return out

        _BACKEND, _BACKEND_NAME = _pymecab, "python-mecab-ko/mecab-ko-dic"
        return _BACKEND
    except Exception as e:
        tried.append(f"python-mecab-ko ({type(e).__name__}: {e})")
    raise AnalyserUnavailable(
        "ko_analyser: no Korean morphological analyser is available, and this channel will NOT "
        "fall back to leaving every token unanalysed — that loads cleanly and parses worse instead "
        "of failing. Install one:\n"
        "  pip install python-mecab-ko          # vendors mecab-ko + mecab-ko-dic, no Homebrew\n"
        "  brew install mecab-ko mecab-ko-dic && export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib"
        "\nTried: " + "; ".join(tried))


def backend_name() -> str:
    _load_backend()
    return _BACKEND_NAME or "?"


def fingerprint() -> str:
    """What `sud.KoAnalyserEmbed.v1` stamps into the model. Records the BINDING and the DICTIONARY,
    `<binding>/<dictionary>`, so a model says exactly what produced its channel.

    ⚠ It is checked on only the DICTIONARY half (`dictionary()`), and that is a measured decision,
    not a convenience. `scripts/check_ko_backends.py` ran all 31 532 distinct eojeol of the treebank
    through both bindings: the tag sequence — the whole multi-hot block — is identical on
    **100.00 %** of them, and the first-morpheme key on **99.99 %** (3 forms differ, all of them a
    dictionary-edition difference rather than a segmentation policy, e.g. Homebrew's mecab-ko-dic
    returning `않` where the vendored one returns the correct `하`). Refusing a model because it was
    trained through `natto-py` and run through `python-mecab-ko` would reject an install that gives
    it the same channel to four decimal places. Swapping mecab-ko-dic for a different analyser still
    refuses, which is what the guard is for."""
    return backend_name()


def dictionary() -> str:
    """The half of the fingerprint that is actually the channel."""
    return backend_name().rsplit("/", 1)[-1]


def analyse(form: str) -> List[Tuple[str, str]]:
    """One eojeol -> its morphemes as (morpheme, Sejong tag), left to right.

    Returns [] when the analyser has nothing to say, which the layer encodes as its own value —
    an unanalysed token and a token analysed as a bare noun must not be the same input, for the
    reason an unset MORPH and an empty one must not be (CLAUDE.md; it cost sa 6.8 LAS)."""
    got = _CACHE.get(form)
    if got is not None:
        return got
    backend = _load_backend()
    out: List[Tuple[str, str]] = []
    for surface, f in backend(form):
        tag = f[_TAG] if f else ""
        expr = f[_EXPR] if len(f) > _EXPR else "*"
        typ = f[_TYPE] if len(f) > _TYPE else "*"
        if typ in _EXPANDABLE and expr not in ("*", ""):
            for part in expr.split("+"):
                bits = part.split("/")
                if len(bits) >= 2 and bits[0] and bits[1]:
                    out.append((bits[0], bits[1]))
            continue
        if "+" in tag:
            # A fused tag with no expression to expand it: keep the pieces, drop the fusion, so the
            # tag inventory stays the treebank's 45 atoms rather than growing a tail of pairs.
            reading = f[_READING] if len(f) > _READING and f[_READING] != "*" else surface
            parts = tag.split("+")
            out.append((reading, parts[0]))
            out.extend((surface, t) for t in parts[1:])
            continue
        reading = f[_READING] if len(f) > _READING and f[_READING] != "*" else surface
        out.append((reading, tag))
    out = [(m, t) for m, t in out if t]
    _CACHE[form] = out
    return out


def stem(form: str) -> Optional[str]:
    """The first morpheme the analyser returns. None when it says nothing."""
    got = analyse(form)
    return got[0][0] if got else None


# 조사 (J*), 어미 (E*) and the derivational suffixes (XS*) are the closed functional classes; a
# morpheme with one of these tags is where the eojeol stops being lexical.
_FUNCTIONAL = ("J", "E", "XS")


def is_functional(tag: str) -> bool:
    return tag.startswith(_FUNCTIONAL)


def content(form: str) -> Optional[str]:
    """The LEXICAL part of the eojeol: every morpheme up to the first functional one, joined.

    Not `stem()`, because mecab-ko-dic splits compounds the treebank keeps whole — `환차익을` comes
    back `환 + 차익 + 을`, whose first morpheme `환` is a misleading symbol while the join `환차익`
    is the gold stem exactly. Falls back to the first morpheme when the eojeol opens with a
    functional one, so this never returns the empty string."""
    got = analyse(form)
    if not got:
        return None
    lead = []
    for m, t in got:
        if is_functional(t):
            break
        lead.append(m)
    return "".join(lead) if lead else got[0][0]


def tags(form: str) -> List[str]:
    return [t for _, t in analyse(form)]


def clear_cache() -> None:
    _CACHE.clear()


if __name__ == "__main__":
    import sys

    print(f"backend: {backend_name()}")
    for w in sys.argv[1:] or ["잡스는", "워즈니악에게", "설계해내지만", "뷁뷀하다"]:
        print(f"  {w:<14}" + "  ".join(f"{m}/{t}" for m, t in analyse(w)))
