#!/usr/bin/env python3
"""`variant_norm`: route an 異體字 onto the treebank's own glyph through NORM.

⚠⚠ **SUPERSEDED BY `sud.CharSegTokenizer.v1`, WHICH APPLIES THE SAME MAP AT ORTH.** Kept because it
is the measurement that decided the question, and because it is the right tool if a caller needs
`token.text` to stay exactly as written. It is NOT registered by `seg_code.py` and no shipped arm
names it; importing this module is what turns the factory on.

WHY IT LOST. NORM is one of the encoder's FOUR channels, and PREFIX/SUFFIX/SHAPE are computed from
ORTH — for a single Han character all four are the same character, so setting NORM alone changes a
quarter of the signal. It also cannot reach the segmenter, which is itself a character model
trained on treebank orthography and meets the variant glyph before any pipe runs. On 54 018 tokens
of kanripo:

    baseline                            PROPN 5.52%    无 -> VERB:141 PROPN:98 NOUN:59
    this pipe (NORM only)               PROPN 5.48%    无 -> VERB:130 NOUN:75  PROPN:69
    the glyph rewritten at the tokeniser PROPN 5.17%   無 -> VERB:364 ADV:47   PROPN:0

It is not worthless — 乗 NOUN→VERB, 别 NOUN→VERB, 従 NOUN→VERB, 曽 NOUN→ADV all come right — but it
leaves most of the gain on the table. See docs/chinese-family.md.

WHY. Every encoder in the lzh arm reads NORM — the shared `tok2vec` the parser listens to
(`attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]`) and the morphologiser's own `HashEmbedCNN`. A
character the training split never showed therefore reaches both of them as an unknown hash row,
and the class prior waiting there is PROPN: 37.5 % of Kyoto's TYPES are PROPN against 8.5 % of its
tokens, 49.4 % of its hapax types are, and 73.0 % of its multi-character tokens are. Measured on
the shipped 0.2.0 wheel over the traditional test set, a token holding a character absent from
train is tagged at **51.79 %** UPOS accuracy with **39.13 %** PROPN precision, against 93.13 % and
93.79 % overall.

Most of those characters are not rare words. They are ordinary ones in a different graphic form —
无 = 無, 隂 = 陰, 徳 = 德, 逺 = 遠 — and between them the 311 such types that occur more than 500
times in kanripo carry 80 % of the whole out-of-treebank character mass. Mapping the glyph onto the
one the treebank uses hands the encoder a row it has actually learned.

This pipe does NOT touch ORTH, tokenisation or any output field: `token.text` still reads 无, and
`token.norm_` reads 無. PREFIX/SUFFIX/SHAPE are still computed from the original glyph, so the
change is confined to the one channel that has a learned row to offer.

⚠ IT REFUSES TO LOAD WITHOUT ITS TABLE. The table IS the component — a `from_disk` that shrugged
at a missing file would give a pipe that loads, runs, and is wrong on exactly the input it exists
for, which is the failure mode CLAUDE.md standing hazard 8 and 11 both describe. `to_disk` writes
the table into the model directory; `from_disk` raises if it is not there.

Build the table with `scripts/build_lzh_variant_norm.py` (Unihan first, SikuBERT's embedding table
for the residue that Unihan does not link).
"""
import json
import pathlib

from spacy.language import Language
from spacy.util import ensure_path


@Language.factory("variant_norm", default_config={"table": None})
def make_variant_norm(nlp, name, table):
    return VariantNorm(table)


class VariantNorm:
    def __init__(self, table=None):
        self.map = {}
        if table:
            self.load_table(table)

    def load_table(self, path):
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        # Accept both the builder's payload and a bare {variant: standard} dict.
        m = d.get("map", d) if isinstance(d, dict) else {}
        self.map = {k: v for k, v in m.items() if not k.startswith("__")}
        return self

    def __call__(self, doc):
        if not self.map:
            return doc
        for t in doc:
            n = "".join(self.map.get(c, c) for c in t.text)
            if n != t.text:
                t.norm_ = n
        return doc

    # --- serialisation. The table travels INSIDE the model directory; see the docstring.
    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "table.json").write_text(json.dumps(self.map, ensure_ascii=False),
                                         encoding="utf-8")

    def from_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        f = path / "table.json"
        if not f.exists():
            raise OSError(
                f"variant_norm: {f} is missing. This component IS its table — loading without it "
                f"would give a pipe that runs and silently does nothing on exactly the characters "
                f"it exists for. Rebuild with scripts/build_lzh_variant_norm.py.")
        self.map = json.loads(f.read_text(encoding="utf-8"))
        return self
