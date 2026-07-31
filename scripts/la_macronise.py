#!/usr/bin/env python3
"""``la_macronise`` -- restore vowel-length macrons to parsed Latin.

This is the "rule + lexicon" half of the Alatius macroniser, re-hosted on top of THIS pipeline's
own morphology. Alatius works by tagging with RFTagger and then looking each (form, tag) up in a
Morpheus-derived lexicon of macronised forms. Our released Latin model already predicts UPOS,
full FEATS and a lemma, so the tagging half is already done -- and done by a tagger trained on the
same treebank the lexicon is keyed to, rather than by a separate one that sometimes disagrees with
it. What remains is the lookup, which is what this component is.

Purely additive: spaCy tokens are immutable, so nothing here touches ``token.text``. The macronised
form is exposed as extensions, and the original ``doc.text`` is untouched:

    token._.macron   -- the macronised word form (str; == token.text when nothing was added)
    token._.macron_level -- which backoff level fired ("L1"/"L2"/"L3"/"S4"/"S3"/None)
    doc._.macron     -- the macronised text, rebuilt with the doc's own whitespace

Backoff, most specific first (see build_la_macron_lut.py for how the table is harvested/pruned):

    L1  (form, upos, feats)  the morphologiser disambiguating genuine homographs
    L2  (form, upos)
    L3  (form)               a bare word list
    S4  (form[-4:], upos, feats)   ending-only, generalises to unseen forms
    S3  (form[-3:], upos, feats)
    --  otherwise the form is left bare (no macrons invented)

MEASURED (agreement with Alatius on the held-out ITTB+PROIEL+Perseus test split, gold morphology):
whole-token 94.32 %, per-vowel 97.34 %. The residue is overwhelmingly STEM length on words the
table has never seen: at the suffix levels the ENDING is 94.3 % right from morphology alone but the
STEM only 75.4 %, and 39.8 % of those tokens really do carry a stem macron. That split is the whole
story -- endings are a function of the paradigm, which we predict; stems are lexical, and covering
them for arbitrary vocabulary needs Morpheus itself, not a treebank-harvested table.

Two caveats worth stating plainly:
  * these numbers are AGREEMENT WITH ALATIUS, not gold vowel length. Alatius is ~98-99 % on vowels,
    so the ceiling here is its accuracy, not ours.
  * with PREDICTED rather than gold morphology, L1 fires on the morphologiser's output
    (``morph_acc`` ~0.83 on la dev), so real-world accuracy is below the figures above.
"""
import gzip
import json
import unicodedata
from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc, Token

LONG = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū", "y": "ȳ"}
LONG.update({k.upper(): v.upper() for k, v in list(LONG.items())})

for _ext, _target in (("macron", Token), ("macron_level", Token), ("macron", Doc)):
    if not _target.has_extension(_ext):
        _target.set_extension(_ext, default=None)


def strip_macron(s):
    n = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in n if c != "̄"))


# --- paradigm override -------------------------------------------------------------------------
# The lookup table memorises (form, morph) -> pattern pairs and CANNOT express a paradigm rule, so
# an unseen (form, morph) combination falls through to the form-only level, which is
# morphology-blind and can flatly contradict correctly-predicted morphology. That is how nominative
# `Gallia` came out `Galliā`: the treebank only ever attests the ablative, so the form-only
# majority is the ablative pattern, and it overrode a correct Case=Nom.
#
# These cells of the Latin paradigm fix the FINAL vowel's length absolutely, whatever the lexicon
# says. Keyed on (InflClass, Case, Number, final letter) -> is that vowel long.
#
# NB the harvested data DISAGREES with these rules on ~1500 training tokens (IndEurA/Nom/Sing/-a is
# marked long 12.9 % of the time; IndEurA/Abl/Sing/-a is marked long only 89.0 %). That is the
# Alatius macroniser's own RFTagger contradicting the treebank's gold morphology -- the rule is
# right and the data is wrong. Applying it therefore LOWERS measured agreement-with-Alatius while
# raising real accuracy; see scripts/eval_la_macronise.py --paradigm.
#
# Deliberately conservative: only cells that are exceptionless in classical Latin and that hinge on
# a final vowel. Third-declension ablative -e (short for consonant stems, -ī for i-stems) is left
# out precisely because it is NOT determined by InflClass alone.
_PARADIGM = {
    # a-stems (1st declension): nominative/vocative singular -a is short, ablative singular -ā long
    ("IndEurA", "Nom", "Sing", "a"): False,
    ("IndEurA", "Voc", "Sing", "a"): False,
    ("IndEurA", "Abl", "Sing", "a"): True,
    # o-stems (2nd declension): dative and ablative singular -ō are long
    ("IndEurO", "Dat", "Sing", "o"): True,
    ("IndEurO", "Abl", "Sing", "o"): True,
    # e-stems (5th declension): ablative singular -ē is long
    ("IndEurE", "Abl", "Sing", "e"): True,
}


def _feat(feats, key):
    """Read one feature out of a CoNLL-U FEATS string (also what str(token.morph) yields)."""
    for part in str(feats).split("|"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return ""


# When InflClass is absent -- routinely so for PROPN in ITTB/PROIEL -- the declension is still
# recoverable from the LEMMA's ending, which is how a Latinist reads it: a lemma in -a is an
# a-stem, one in -us/-um an o-stem. Restricted to nominals, and only used as a fallback.
_LEMMA_CLASS = (("a", "IndEurA"), ("us", "IndEurO"), ("um", "IndEurO"))


def _infl_class(feats, lemma, upos):
    ic = _feat(feats, "InflClass")
    if ic or upos not in ("NOUN", "PROPN", "ADJ"):
        return ic
    lem = strip_macron(str(lemma or "")).lower()
    for suf, cls in _LEMMA_CLASS:
        if lem.endswith(suf) and len(lem) > len(suf):
            return cls
    return ""


def paradigm_final(form, feats, lemma=None, upos=None):
    """Return True/False if the paradigm fixes the FINAL vowel's length, else None.

    Returns None whenever the cell is not covered -- including when InflClass is ABSENT, which is
    the common case for PROPN in ITTB/PROIEL. That is why nominative `Gallia` is still not fixed by
    this rule: it carries Case=Nom|Gender=Fem|Number=Sing and no InflClass at all, so there is
    nothing to key on. Inferring the declension from the lemma would be the next step, and is not
    attempted here.
    """
    if not form or form[-1] not in "aeiouyAEIOUY":
        return None
    return _PARADIGM.get((_infl_class(feats, lemma, upos), _feat(feats, "Case"),
                          _feat(feats, "Number"), form[-1].lower()))


def apply_mask(form, mask):
    """Lengthen the vowels whose bit is set, preserving the form's own case."""
    out = []
    for i, ch in enumerate(form):
        out.append(LONG.get(ch, ch) if (mask >> i) & 1 else ch)
    return "".join(out)


@Language.factory("la_macronise", default_config={"lut": None, "paradigm": True})
def make_la_macronise(nlp, name, lut, paradigm):
    return LaMacronise(lut, paradigm)


class LaMacronise:
    def __init__(self, lut=None, paradigm=True):
        self.paradigm = paradigm
        self.l1 = self.l2 = self.l3 = self.s4 = self.s3 = {}
        # `lut` is a BUILD-time convenience only. In a packaged model the table travels inside the
        # model directory and is restored by from_disk(), which runs after __init__ -- so a config
        # that still names a build-time path (or none at all) must not be fatal here, or the wheel
        # fails to load with FileNotFoundError before from_disk ever gets a chance.
        if lut and Path(lut).exists():
            self._load_blob(json.loads(gzip.open(lut, "rb").read().decode("utf-8")))

    def _load_blob(self, b):
        self.l1 = {(f, u, x): m for f, u, x, m in b["L1"]}
        self.l2 = {(f, u): m for f, u, m in b["L2"]}
        self.l3 = {f: m for f, m in b["L3"]}
        self.s4 = {(f, u, x): m for f, u, x, m in b["S4"]}
        self.s3 = {(f, u, x): m for f, u, x, m in b["S3"]}

    def _lookup(self, form, upos, feats):
        """Return (mask, level) for the lowercased form, or (None, None)."""
        n = len(form)
        m = self.l1.get((form, upos, feats))
        if m is not None:
            return m, "L1"
        m = self.l2.get((form, upos))
        if m is not None:
            return m, "L2"
        m = self.l3.get(form)
        if m is not None:
            return m, "L3"
        for k, tab, lvl in ((4, self.s4, "S4"), (3, self.s3, "S3")):
            m = tab.get((form[-k:], upos, feats))
            if m is not None:
                # the stored mask is indexed from the right; shift it back onto the form
                return sum(1 << (i + n - k) for i in range(k)
                           if (m >> i) & 1 and 0 <= i + n - k < n), lvl
        return None, None

    def resolve(self, form, upos, feats, lemma=None):
        """(macronised form, level) for one token -- the single path used by __call__ AND the
        evaluator, so a measurement can never silently miss the paradigm override."""
        if not any(c.isalpha() for c in form):
            return form, None
        mask, level = self._lookup(strip_macron(form).lower(), upos, feats)
        mask = mask or 0
        if self.paradigm:
            fixed = paradigm_final(form, feats, lemma, upos)
            if fixed is not None:
                bit = 1 << (len(form) - 1)
                new = (mask | bit) if fixed else (mask & ~bit)
                if new != mask:
                    level = f"{level or 'none'}+P"
                mask = new
        base = strip_macron(form)
        return (apply_mask(base, mask) if mask else base), level

    def __call__(self, doc):
        if not self.l3:
            raise RuntimeError(
                "la_macronise has no lookup table. It is not distributed -- the vowel-length data "
                "is derived from Morpheus (CC BY-SA 3.0) via the Alatius macroniser, and shipping "
                "it inside this CC BY-NC-SA model would add a restriction BY-SA forbids. "
                "Build it locally:  bash scripts/build_la_macron.sh  (see NOTICE.md)."
            )
        pieces = []
        for tok in doc:
            out, level = self.resolve(tok.text, tok.pos_, str(tok.morph) or "_",
                                       tok.lemma_)
            tok._.macron = out
            tok._.macron_level = level
            pieces.append(out + tok.whitespace_)
        doc._.macron = "".join(pieces)
        return doc

    # -- serialisation: the table travels inside the model directory ------------------
    def to_disk(self, path, exclude=tuple()):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {
            "L1": [[f, u, x, m] for (f, u, x), m in self.l1.items()],
            "L2": [[f, u, m] for (f, u), m in self.l2.items()],
            "L3": [[f, m] for f, m in self.l3.items()],
            "S4": [[f, u, x, m] for (f, u, x), m in self.s4.items()],
            "S3": [[f, u, x, m] for (f, u, x), m in self.s3.items()],
        }
        with gzip.open(path / "lut.json.gz", "wb", compresslevel=9) as fh:
            fh.write(json.dumps(blob, ensure_ascii=False).encode("utf-8"))

    def from_disk(self, path, exclude=tuple()):
        p = Path(path) / "lut.json.gz"
        if p.exists():
            self._load_blob(json.loads(gzip.open(p, "rb").read().decode("utf-8")))
        return self

    def to_bytes(self, exclude=tuple()):
        blob = {
            "L1": [[f, u, x, m] for (f, u, x), m in self.l1.items()],
            "L2": [[f, u, m] for (f, u), m in self.l2.items()],
            "L3": [[f, m] for f, m in self.l3.items()],
            "S4": [[f, u, x, m] for (f, u, x), m in self.s4.items()],
            "S3": [[f, u, x, m] for (f, u, x), m in self.s3.items()],
        }
        return gzip.compress(json.dumps(blob, ensure_ascii=False).encode("utf-8"), 9)

    def from_bytes(self, data, exclude=tuple()):
        self._load_blob(json.loads(gzip.decompress(data).decode("utf-8")))
        return self
