#!/usr/bin/env python3
"""``fa_vocalise`` -- restore the unwritten short vowels to parsed Persian.

The Persian counterpart of `ar_vocalise`, and it is a WEAKER proposition, for a reason worth
stating plainly rather than discovering later.

Arabic works because SUD_Arabic-PADT ships the gold vocalisation in MISC beside the gold
morphology, so the table is a straight extract and the pipeline's predicted UPOS+FEATS chooses
between rival vocalisations of one spelling -- which is most of the task, since two thirds of
Arabic tokens sit under an ambiguous skeleton. Persian has none of that:

  * SUD_Persian-PerDT carries **no vocalisation at all**. Its `Translit=` is a mechanical
    romanisation of the consonantal spelling (`nšst` for نشست), so it records no short vowels;
    there is no `Ezafe` feature either. So there is nothing to harvest and nothing to score
    against -- the accuracy figures `ar_vocalise` quotes have no Persian equivalent, and this
    module does not invent one.
  * The lexicon it is built from (Tihu, via `build_fa_vocalise_lut.py`) holds ONE pronunciation
    per word, so the morphological disambiguation that justifies putting this in a pipeline has no
    alternatives to choose between. Persian's real homographs -- کرم kerm/karam/kerem, شکر
    šekar/šokr -- are absent or arbitrarily resolved in the source.

So: this is a lexicon lookup that happens to live in a pipeline, and the one thing the pipeline
genuinely contributes is the LEMMA, which extends coverage over inflected forms (see below).
Treat it as a convenience for readers and learners, not as a measured system.

Purely additive, exactly as `ar_vocalise` and `la_macronise` are -- `doc.text` is untouched:

    token._.vocalised        -- the vocalised word form (str; == token.text when nothing was added)
    token._.vocalised_level  -- which rung answered ("F"/"L"/"X"/None)
    doc._.vocalised          -- the vocalised text, rebuilt with the doc's own whitespace

EZĀFE IS THE ONE PART THE PIPELINE REALLY DECIDES. The ezāfe -e linking a head to its modifier is
a SYNTACTIC fact, so no dictionary can supply it and the parse can. No treebank here annotates it
(checked SUD PerDT, upstream UD PerDT and UD Seraji), so the rules are derived from the one place
Persian is forced to write it -- after a vowel-final stem, as ی or ٔ -- by
`build_fa_ezafe_rules.py`. On that observable subset `NOUN <-mod- ADJ` carries the ezāfe 92.6 % of
the time against a 12.5 % base rate, and `NOUN <-mod- VERB` (a relative clause) 1.9 %, which is the
distinction working. The kasra is added ONLY to a consonant-final host: on a vowel-final one the
mark is a LETTER (ی), and writing it would change the consonantal skeleton, which this component
never does.

RUNGS.
    P  (form, UPOS), for the ~129 homographs KaamelDict annotates with a usable POS.
    F  the form itself.
    L  the LEMMA's vocalisation transferred onto a form the lemma PREFIXES: نِشَست + ند ->
       نِشَستند. This is where the pipeline earns its place -- fa's lemmatiser scores 0.981 -- and
       it is deliberately conservative: marks are copied only over the stretch the lemma actually
       spells, and the affix is left bare rather than guessed. PerDT writes verb lemmas as
       `past#present` pairs (`شد#شو`), so each side is tried.

COVERAGE, on PerDT TEST, over the 89.9 % of tokens that are not PUNCT/SYM/X/NUM (21 698 of
24 133). The jump is the lexicon, not the method -- Tihu alone answered 72.80 %:

    P  POS-conditioned homograph        111
    F  the form itself               19 049
    L  lemma transfer                 1 302
                                     ------
       answered                      94.30 %      left bare 5.70 %

That is COVERAGE, not accuracy: on the tokens it answers, correctness is bounded by KaamelDict's
and Tihu's own and by the aligner's, and none of the three has been scored against Persian gold
because none exists here. The ezāfe rules ARE scored -- see above -- because orthography supplies
gold for the vowel-final subset.

The table is NOT bundled -- same footing as la's Morpheus table and ar's PADT table. Build one
with `build_fa_vocalise_lut.py`; until then every token passes through unchanged and the component
warns once.
"""
import gzip
import json
import warnings
from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc, Token

KASRA = "\u0650"
VOWEL_FINAL = "اوهیآ"

for _ext, _target in (("vocalised", Token), ("vocalised_level", Token), ("vocalised", Doc)):
    if not _target.has_extension(_ext):
        _target.set_extension(_ext, default=None)

DIAC = set("ًٌٍَُِّْٰ")
# Nothing to vocalise, and a lexicon hit on one would be a coincidence rather than an answer.
_PASS_THROUGH = {"PUNCT", "SYM", "X", "NUM"}


def strip_diac(s):
    return "".join(c for c in s if c not in DIAC)


@Language.factory("fa_vocalise",
                  default_config={"lut": None, "ezafe": None, "require_data": False})
def make_fa_vocalise(nlp, name, lut, ezafe, require_data):
    return FaVocalise(lut, ezafe, require_data)


class FaVocalise:
    def __init__(self, lut=None, ezafe=None, require_data=False):
        self.require_data = require_data
        self._warned = False
        self.forms = {}
        self.pos_forms = {}
        self.ezafe = {}
        if ezafe and Path(ezafe).exists():
            self.ezafe = json.loads(Path(ezafe).read_text(encoding="utf-8"))
        # Build-time convenience only; in a packaged model the table arrives via from_disk, which
        # runs after __init__ -- so a stale path must not be fatal or the wheel fails to load.
        if lut and Path(lut).exists():
            self._load_blob(json.loads(gzip.open(lut, "rb").read().decode("utf-8")))

    def lookup(self, form, lemma, upos):
        if upos in _PASS_THROUGH:
            return form, "X"
        key = strip_diac(form)
        hit = self.pos_forms.get("|".join((key, upos)))
        if hit is not None:
            return hit, "P"
        hit = self.forms.get(key)
        if hit is not None:
            return hit, "F"
        # Lemma transfer. PerDT writes verb lemmas as `past#present`, so try each side; require the
        # lemma to be a genuine PREFIX of the form, which is what makes copying its marks safe.
        for part in (lemma or "").split("#"):
            part = strip_diac(part)
            if part and key.startswith(part) and len(part) < len(key):
                lv = self.forms.get(part)
                if lv is not None:
                    return lv + key[len(part):], "L"
        return form, None

    def takes_ezafe(self, tok):
        """The next token is this token's own dependent, in a configuration the derived table
        keeps. `tok.nbor` raises at the edge, so the index check comes first."""
        if not self.ezafe or tok.i + 1 >= len(tok.doc):
            return False
        nxt = tok.doc[tok.i + 1]
        if nxt.head.i != tok.i:
            return False
        return "|".join((tok.pos_, nxt.dep_, nxt.pos_)) in self.ezafe

    def __call__(self, doc):
        # The ezafe rules alone are enough to do useful work: they ship in the wheel (PerDT-derived,
        # CC BY-SA 4.0) while the lexicon does not (KaamelDict is GPL), so the common installed
        # state is rules-without-lexicon and it must not be treated as "no data".
        if not (self.forms or self.ezafe):
            if self.require_data:
                raise RuntimeError(
                    "fa_vocalise has no table. Build one with\n"
                    "  python scripts/build_fa_vocalise_lut.py\n"
                    "or set require_data=False to pass text through unchanged.")
            if not self._warned:
                warnings.warn(
                    "fa_vocalise has no vocalisation table, so every token is passing through "
                    "unchanged. Build one with `pip install PersianG2p` followed by "
                    "`python scripts/build_fa_vocalise_lut.py` (the builder ships beside this "
                    "module), or pass an existing table's path as the `lut` config value.",
                    RuntimeWarning, stacklevel=2)
                self._warned = True
            return doc
        out = []
        for tok in doc:
            v, lvl = self.lookup(tok.text, tok.lemma_, tok.pos_)
            if self.takes_ezafe(tok) and not v.rstrip("\u200c").endswith(tuple(VOWEL_FINAL)):
                # consonant-final only: on a vowel-final host the ezāfe is the LETTER ی, and
                # adding a letter would change the skeleton this component promises not to touch.
                v = v + KASRA
                lvl = (lvl or "") + "+EZ"
            tok._.vocalised, tok._.vocalised_level = v, lvl
            out.append(v + tok.whitespace_)
        doc._.vocalised = "".join(out)
        return doc

    def _load_blob(self, blob):
        self.forms = dict(blob.get("F", []))
        self.pos_forms = dict(blob.get("P", []))

    def to_disk(self, path, exclude=tuple()):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with gzip.open(path / "lut.json.gz", "wb", compresslevel=9) as fh:
            fh.write(json.dumps({"F": sorted(self.forms.items()),
                                 "P": sorted(self.pos_forms.items())},
                                ensure_ascii=False).encode("utf-8"))
        (path / "ezafe.json").write_text(json.dumps(self.ezafe, ensure_ascii=False),
                                         encoding="utf-8")

    def from_disk(self, path, exclude=tuple()):
        p = Path(path) / "lut.json.gz"
        if p.exists():
            self._load_blob(json.loads(gzip.open(p, "rb").read().decode("utf-8")))
        e = Path(path) / "ezafe.json"
        if e.exists():
            self.ezafe = json.loads(e.read_text(encoding="utf-8"))
        return self

    def to_bytes(self, exclude=tuple()):
        return gzip.compress(json.dumps({"F": sorted(self.forms.items()),
                                         "P": sorted(self.pos_forms.items()),
                                         "EZ": self.ezafe}, ensure_ascii=False).encode("utf-8"), 9)

    def from_bytes(self, data, exclude=tuple()):
        blob = json.loads(gzip.decompress(data).decode("utf-8"))
        self._load_blob(blob)
        self.ezafe = blob.get("EZ", {})
        return self
