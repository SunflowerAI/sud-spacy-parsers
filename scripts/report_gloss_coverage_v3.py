#!/usr/bin/env python3
"""How much of each held-out language the gloss channel can actually fill, and by which key.

This is the ceiling on what the v3 lexical channel can do zero-shot, and it is a property of the
DICTIONARY, not of the model -- so it is worth knowing before any arm finishes. Four numbers per
language, because three different things can be missing and they have different fixes:

    dict_lemma / dict_form   the key was in the Wiktionary bag
    rows_lemma / rows_form   ... and at least one of its English words has a row in the table

If `rows` tracks `dict`, the English table is not the bottleneck and the dictionary is.

⚠ LEMMA IS AN UPPER BOUND AND FORM IS THE DEPLOYMENT NUMBER. The v2 contract declares UPOS and
`Doc._.tb_lang` as user inputs and says nothing about lemmas, so keying by the gold lemma column
adds an annotation layer the contract does not have. Wiktionary indexes lemmas, so the two diverge
most in the most inflected languages -- Greek 76.4 % by lemma against 44.5 % by form.

⚠ AND BOTH UNDERSTATE A REAL GLOSSING USER, which is the one direction of error worth being wrong
in. A fieldworker glossing a text glosses EVERY token; a dictionary only covers what it happens to
list. The three held-out languages that carry a real `Gloss=` column run 67-81 % fill, well above
what Wiktionary gives for anything here. The Wiktionary figure is the conservative one.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from spacy.tokens import DocBin                       # noqa: E402
from spacy.vocab import Vocab                         # noqa: E402
from sud_generic_embed_v3 import load_vectors         # noqa: E402

SPLIT = re.compile(r"[-.:=,;/\[\]()<>+~]+|_")


def resolves(table, gloss):
    """Does any piece of this gloss have an English row? All-caps pieces are Leipzig categories."""
    for p in (q for part in SPLIT.split(gloss.replace("_", " ")) for q in part.split()):
        if p.isalpha() and not (p.isupper() and len(p) > 1) and table.row("en", p) is not None:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="assets_vec/generic_vec_v3.npz")
    ap.add_argument("--gloss-dir", default="assets_vec/dict")
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", default="metrics/generic_v3/gloss_coverage.json")
    a = ap.parse_args()

    table = load_vectors(a.table)
    man = json.load(open(a.manifest, encoding="utf-8"))["languages"]
    langs = sorted(l for l, v in man.items() if v["pool"] == "test")

    rows, missing = [], []
    print(f"{'lang':5s} {'tokens':>7s} {'dict/lem':>9s} {'rows/lem':>9s} "
          f"{'dict/frm':>9s} {'rows/frm':>9s}")
    for lang in langs:
        gp = os.path.join(a.gloss_dir, f"{lang}-en.json")
        cp = os.path.join(a.corpus, f"{lang}-{a.split}.spacy")
        if not os.path.exists(gp):
            missing.append(lang)
            continue
        if not os.path.exists(cp):
            continue
        raw = json.load(open(gp, encoding="utf-8"))
        gd = {k: " ".join(list(v)[:a.top]) for k, v in raw.items() if v}
        docs = list(DocBin().from_disk(cp).get_docs(Vocab()))
        n = dl = df = rl = rf = 0
        for d in docs:
            for t in d:
                n += 1
                gl = gd.get(t.lemma_) or gd.get((t.lemma_ or "").lower())
                gf = gd.get(t.text) or gd.get(t.text.lower())
                if gl:
                    dl += 1
                    rl += resolves(table, gl)
                if gf:
                    df += 1
                    rf += resolves(table, gf)
        rec = dict(lang=lang, tokens=n, headwords=len(gd),
                   dict_lemma=round(dl / n, 4), rows_lemma=round(rl / n, 4),
                   dict_form=round(df / n, 4), rows_form=round(rf / n, 4),
                   genus=man[lang]["genus"])
        rows.append(rec)
        print(f"{lang:5s} {n:7d} {dl/n:9.1%} {rl/n:9.1%} {df/n:9.1%} {rf/n:9.1%}")

    if rows:
        ml = sum(r["rows_lemma"] for r in rows) / len(rows)
        mf = sum(r["rows_form"] for r in rows) / len(rows)
        print(f"\nmean fillable: {ml:.1%} by lemma (upper bound), {mf:.1%} by form (deployment), "
              f"over {len(rows)} languages")
    if missing:
        print(f"NO Wiktionary extract: {missing}  -- not scorable on the gloss fill by this route")

    out = pathlib.Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(table=a.table, split=a.split, top=a.top, languages=rows,
                   no_dictionary=missing), open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
