#!/usr/bin/env python3
"""``ar_vocalise`` -- restore the full vocalisation (tashkīl) to parsed Arabic.

The Arabic analogue of ``la_macronise``, and it rests on the same observation: written Arabic
omits the short vowels, restoring them is mostly a LEXICAL lookup, and the part that is not
lexical is a MORPHOLOGICAL question this pipeline already answers. `ar_sud_padt` predicts UPOS,
full FEATS (Case, Definite, Mood, Voice, …) and a lemma, so the disambiguating half is done; what
remains is the table.

That morphology is not a nicety here, it is the whole difficulty. Measured on SUD_Arabic-PADT, a
majority-class table keyed on the bare consonantal skeleton alone tops out at **84.92 %**, adding
UPOS reaches 86.90 %, and adding FEATS reaches **98.55 %** -- because the final short vowel of an
Arabic word is its case ending (iʿrāb), which is a syntactic fact and is exactly what a parser is
for. Two thirds of all tokens sit under a skeleton that is ambiguous; under (form, UPOS, FEATS)
that falls to 8 %.

Purely additive. spaCy tokens are immutable and, more to the point, the parser was trained on the
undiacritised orthography -- so nothing here touches ``token.text`` and ``doc.text`` is unchanged:

    token._.vocalised        -- the fully vocalised word form (str; == token.text when nothing
                                could be added)
    token._.vocalised_level  -- which rung answered ("L1"/"L2"/"L3"/"C"/"X"/None)
    doc._.vocalised          -- the vocalised text, rebuilt with the doc's own whitespace

THE TABLE IS HARVESTED FROM THE TREEBANK ITSELF. Every PADT token carries ``Vform=`` in MISC (all
223 881 train tokens), which is the gold vocalisation sitting beside the gold morphology -- the
same free pairing the Latin macron table was harvested from, except that here it is in the
treebank rather than needing a separate tool to generate it. ``build_ar_vocalise_lut.py`` reads it.

⚠ LICENCE. SUD_Arabic-PADT is **CC BY-NC-SA 3.0** (see its LICENSE.txt), so a table harvested from
it is NonCommercial and so, by this project's own reasoning about en_gum -- annotations are what a
trained model absorbs -- is any model trained on it. The released `ar_sud_padt` wheel currently
declares CC BY-SA 4.0, which looks wrong independently of this component. Nothing here is packaged
until that is settled; see the module notes in `build_ar_vocalise_lut.py`.

CAMEL TOOLS IS THE FALL-THROUGH, AND IT IS NOT BUNDLED. The residue the table cannot reach is
vocabulary, not morphology -- 6.07 % of test tokens have a skeleton train never saw, and that is
the whole remaining error budget. CAMeL Tools' `calima-msa` analyser answers those, and the ar
wheel already depends on camel-tools because `ar_tokenizer` needs it for clitic boundaries. The
database is **GPL v2** (Aramorph/BAMA-derived), so it is fetched by the user with ``camel_data``
and never redistributed here -- the same position `la_macronise` takes toward Morpheus, and for
the same reason: GPL restricts distribution, not use.

⚠ CAMEL AND PADT WRITE DIFFERENT CONVENTIONS, and comparing them naively costs 24 points. Four
differences, all mechanical, all handled by ``to_padt``/``canon``:

    sukūn          CAMeL  تَرْفُض      PADT  تَرفُض      PADT writes 39 sukūn in 223 881 tokens
    vowel + matres CAMeL  وِزارِيّ     PADT  وِزَارِيّ    PADT writes the fatḥa before a long ā
    tanwīn + alif  CAMeL  وَزِيراً     PADT  وَزِيرًا     the tanwīn sits the other side of the alif
    hamzat waṣl    CAMeL  الصِّناعَة   PADT  اَلصِّنَاعَة  PADT marks the article's alif

Held-out on PADT test, the analyser's oracle recall on out-of-table tokens is 13.63 % compared
raw and **37.68 %** compared after normalisation. So the normaliser is not tidying, it is most of
the fall-through's value.

WHAT IT SCORES. Whole-token exact match on PADT test, cascade L1→L2→L3, table only:

    with GOLD UPOS+FEATS        89.96 %      the ceiling the table itself imposes
    with PREDICTED UPOS+FEATS   86.06 %      <- deployment, released arm's own morphology

The 3.9-point gap is the morphologiser's Case errors landing on the final vowel, which is the
expected shape: it is the one position no lexicon can settle. NB a slice of the residue is gold
noise rather than error -- PADT leaves foreign/unanalysed `X` tokens bare (`دمشق` stays `دمشق`),
so where the tagger reads them as ordinary nouns the table vocalises them and is marked wrong for
producing the better answer. `X` is passed through unchanged here to match the treebank.
"""
import gzip
import json
import re
import warnings
from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc, Token

for _ext, _target in (("vocalised", Token), ("vocalised_level", Token), ("vocalised", Doc)):
    if not _target.has_extension(_ext):
        _target.set_extension(_ext, default=None)

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َُِّْ"
FATHATAN, DAMMATAN, KASRATAN = "ًٌٍ"
DAGGER_ALIF, TATWEEL = "ٰ", "ـ"
ALIF, WAW, YA, ALIF_MAQ = "اويى"
DIAC = set(FATHA + DAMMA + KASRA + SUKUN + SHADDA + FATHATAN + DAMMATAN + KASRATAN
           + DAGGER_ALIF + "ٕٔٓ")
HAMZA_ALIF = {"أ": ALIF, "إ": ALIF, "آ": ALIF, "ٱ": ALIF}


def strip_diac(s):
    """The consonantal skeleton -- the key everything is looked up under."""
    return "".join(c for c in s if c not in DIAC).replace(TATWEEL, "")


def fold_hamza(s):
    """Second rung of the key ladder: أ إ آ ٱ -> ا, ى -> ي, ة -> ه. Real text spells the hamza
    carriers inconsistently, and the treebank is not immune; folding lets a form still find its
    entry. Least-normalised first, as in `la_macronise.key_ladder` -- fold too early and distinct
    entries collide."""
    return "".join(HAMZA_ALIF.get(c, c) for c in s).replace(ALIF_MAQ, YA).replace("ة", "ه")


def canon(s):
    """Reduce a vocalised form to a convention-free shape, for COMPARISON only -- never for output.
    Everything removed here is predictable from what remains, so two spellings that differ only in
    these respects are the same answer written two ways."""
    if not s:
        return s
    s = s.replace("+", "").replace(TATWEEL, "").replace(SUKUN, "")
    s = s.replace(DAGGER_ALIF, ALIF)
    s = re.sub(FATHATAN + ALIF, ALIF + FATHATAN, s)
    s = re.sub(ALIF_MAQ + FATHATAN, FATHATAN + ALIF_MAQ, s)
    s = re.sub(FATHA + ALIF, ALIF, s)
    s = re.sub(KASRA + YA, YA, s)
    s = re.sub(DAMMA + WAW, WAW, s)
    s = re.sub(FATHA + ALIF_MAQ, ALIF_MAQ, s)
    return s


def to_padt(s):
    """Rewrite a CAMeL-convention vocalisation into PADT's. The inverse of the differences `canon`
    erases -- but this one has to COMMIT to a spelling, so it only makes the changes that are
    always right: drop the sukūn PADT does not write, and put the tanwīn back before its alif."""
    if not s:
        return s
    s = s.replace("+", "").replace(SUKUN, "")
    s = re.sub(ALIF + FATHATAN, FATHATAN + ALIF, s)
    return s


# UPOS -> the `pos` values calima-msa uses, for selecting among an analyser's readings.
_CAMEL_POS = {
    "NOUN": {"noun", "noun_num", "noun_quant"}, "PROPN": {"noun_prop"},
    "ADJ": {"adj", "adj_comp", "adj_num"}, "VERB": {"verb", "verb_pseudo"},
    "ADV": {"adv", "adv_interrog", "adv_rel"}, "ADP": {"prep"},
    "CCONJ": {"conj"}, "SCONJ": {"conj_sub"}, "PRON": {"pron", "pron_dem", "pron_rel",
                                                       "pron_interrog"},
    "DET": {"pron_dem", "part_det"}, "NUM": {"noun_num", "digit"}, "PART": {"part"},
    "AUX": {"verb", "verb_pseudo"}, "INTJ": {"interj"}, "PUNCT": {"punc"},
}
# Tokens PADT declines to vocalise at all. `X` is its foreign/unanalysable class and its Vform is
# the bare form on essentially all of them, so inventing vowels there would disagree with the gold
# by construction.
_PASS_THROUGH = {"X", "PUNCT", "SYM"}


@Language.factory("ar_vocalise",
                  default_config={"lut": None, "camel": True, "require_data": False})
def make_ar_vocalise(nlp, name, lut, camel, require_data):
    return ArVocalise(lut, camel, require_data)


class ArVocalise:
    def __init__(self, lut=None, camel=True, require_data=False):
        # Same degrade-don't-fail posture as `la_macronise`: a component sitting in a default
        # pipeline that RAISES on missing data would break every ordinary `nlp(text)` for users who
        # never asked for vocalisation. `require_data=True` restores a hard failure for a caller
        # who added the pipe on purpose.
        self.require_data = require_data
        self._warned = False
        self.l1 = self.l2 = self.l3 = {}
        # A build-time convenience only: in a packaged model the table travels in the model
        # directory and arrives via from_disk(), which runs AFTER __init__ -- so a config naming a
        # path that no longer exists must not be fatal, or the wheel fails to load.
        if lut and Path(lut).exists():
            self._load_blob(json.loads(gzip.open(lut, "rb").read().decode("utf-8")))
        self.camel = camel
        self._analyzer = None

    # --- the analyser fall-through (never bundled; GPL v2) ---------------------------------------
    def _get_analyzer(self):
        if self._analyzer is None and self.camel:
            try:
                from camel_tools.morphology.analyzer import Analyzer
                from camel_tools.morphology.database import MorphologyDB
                db = self.camel if isinstance(self.camel, str) else None
                self._analyzer = Analyzer(
                    MorphologyDB(db) if db else MorphologyDB.builtin_db(), "NOAN_PROP")
            except Exception:
                self._analyzer = False   # asked and unavailable; do not ask again per doc
        return self._analyzer or None

    def _from_camel(self, form, upos):
        an = self._get_analyzer()
        if an is None:
            return None
        try:
            reads = an.analyze(form)
        except Exception:
            return None
        if not reads:
            return None
        want = _CAMEL_POS.get(upos)
        pool = [a for a in reads if want and a.get("pos") in want] or reads
        diacs = [a["diac"] for a in pool if a.get("diac")]
        if not diacs:
            return None
        # calima returns readings in the database's own order with no frequency attached, so there
        # is nothing to rank on beyond the UPOS filter above; take the first surviving reading.
        return to_padt(diacs[0])

    # --- lookup ---------------------------------------------------------------------------------
    def lookup(self, form, upos, feats):
        """Return (vocalised, level). The key ladder is least-normalised first.

        The pass-through classes take a SHORTER ladder: the UPOS-keyed rungs, then the token
        unchanged. PADT leaves most foreign/unanalysable `X` bare but vocalises the ones it
        recognises (`برلين` -> `بَرلِين`), so neither answer alone is right -- table-then-identity
        scores 99.20 % on test `X` against bare identity's 90.43 %. They skip the skeleton-only
        rung, whose majority is taken over the other parts of speech, and skip the analyser, which
        has no useful opinion about a foreign string."""
        skel = strip_diac(form)
        passthrough = upos in _PASS_THROUGH
        for key in (skel, fold_hamza(skel)):
            rungs = [(self.l1, "L1", (key, upos, feats)), (self.l2, "L2", (key, upos))]
            if not passthrough:
                rungs.append((self.l3, "L3", key))
            for table, rung, k in rungs:
                hit = table.get(k)
                if hit is not None:
                    return hit, rung
        if passthrough:
            return form, "X"
        got = self._from_camel(form, upos)
        if got:
            return got, "C"
        return form, None

    def __call__(self, doc):
        if not (self.l1 or self.l2 or self.l3):
            if self.require_data:
                raise RuntimeError(
                    "ar_vocalise has no table. Build one with\n"
                    "  python scripts/build_ar_vocalise_lut.py\n"
                    "or set require_data=False to pass text through unchanged.")
            if not self._warned:
                warnings.warn(
                    "ar_vocalise has no vocalisation table, so every token is passing through "
                    "unchanged. The table is derived from SUD_Arabic-PADT, which is CC BY-NC-SA "
                    "3.0, so it is not bundled in this wheel. Two ways to get one:\n"
                    "  (1) clone https://github.com/surfacesyntacticud/SUD_Arabic-PADT and run\n"
                    "        python -m %s --train <path>/ar_padt-sud-train.conllu\n"
                    "      (the builder ships beside this module as build_ar_vocalise_lut.py)\n"
                    "  (2) pass an existing table's path as the `lut` config value.\n"
                    "Independently, `pip install camel-tools` plus `camel_data -i "
                    "morphology-db-msa-r13` enables the analyser fall-through."
                    % __name__.replace("ar_vocalise", "build_ar_vocalise_lut"),
                    RuntimeWarning, stacklevel=2)
                self._warned = True
            return doc
        out = []
        for tok in doc:
            v, lvl = self.lookup(tok.text, tok.pos_, str(tok.morph) or "_")
            tok._.vocalised, tok._.vocalised_level = v, lvl
            out.append(v + tok.whitespace_)
        doc._.vocalised = "".join(out)
        return doc

    # --- serialisation --------------------------------------------------------------------------
    def _blob(self):
        return {"L1": [[f, u, x, m] for (f, u, x), m in self.l1.items()],
                "L2": [[f, u, m] for (f, u), m in self.l2.items()],
                "L3": [[f, m] for f, m in self.l3.items()]}

    def _load_blob(self, blob):
        self.l1 = {(f, u, x): m for f, u, x, m in blob.get("L1", [])}
        self.l2 = {(f, u): m for f, u, m in blob.get("L2", [])}
        self.l3 = {f: m for f, m in blob.get("L3", [])}

    def to_disk(self, path, exclude=tuple()):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "lut.json.gz", "wb", compresslevel=9) as fh:
            fh.write(json.dumps(self._blob(), ensure_ascii=False).encode("utf-8"))

    def from_disk(self, path, exclude=tuple()):
        p = Path(path) / "lut.json.gz"
        if p.exists():
            self._load_blob(json.loads(gzip.open(p, "rb").read().decode("utf-8")))
        return self

    def to_bytes(self, exclude=tuple()):
        return gzip.compress(json.dumps(self._blob(), ensure_ascii=False).encode("utf-8"), 9)

    def from_bytes(self, data, exclude=tuple()):
        self._load_blob(json.loads(gzip.decompress(data).decode("utf-8")))
        return self
