#!/usr/bin/env python3
"""Score Arabic/Persian arms across vocalisation variants of the SAME test set.

Every variant is the identical treebank with only the FORM column rewritten
(`make_ar_variant_conllu.py`), so the trees -- and therefore the gold -- are held constant and the
only thing moving between rows is how the text is written. The `la` counterpart of this is
`eval_la_variants.py`, and the question is the same: how much does this arm lose when it is handed
an orthography it was not trained on.

All evaluation is gold-preproc, as everywhere in this project outside English.

    python scripts/eval_ar_variants.py --model released=training_ar_sud_xpos/model-best \\
        --corpus-dir corpus_ar_variants --prefix ar --out metrics_ar_variants.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _m in ("ar_tokenizer", "sud_feats_embed", "sud_tagger", "sud_misc", "sud_idiom",
           "sud_shared_data", "sud_shared_rule", "sud_reported_data", "sud_reported_rule",
           "sud_subject_frames", "sud_subject_rule", "gold_tok_corpus", "ar_vocalise",
           "fa_vocalise", "fa_align"):
    try:
        __import__(_m)
    except Exception:
        pass

from spacy.cli.evaluate import evaluate   # noqa: E402

ORDER = ["bare", "shadda", "final", "internal", "p25", "p50", "p75", "full"]
METRICS = [("TAG", "tag_acc"), ("POS", "pos_acc"), ("LEMMA", "lemma_acc"),
           ("UAS", "dep_uas"), ("LAS", "dep_las")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--prefix", default="ar")
    ap.add_argument("--out")
    a = ap.parse_args()

    cdir = Path(a.corpus_dir)
    corpora = {}
    for p in sorted(cdir.glob(f"{a.prefix}_*.spacy")):
        corpora[p.stem[len(a.prefix) + 1:]] = p
    variants = [v for v in ORDER if v in corpora] + \
               [v for v in sorted(corpora) if v not in ORDER]

    allres = {}
    for spec in a.model:
        label, path = spec.split("=", 1)
        res = {}
        for v in variants:
            try:
                res[v] = evaluate(path, str(corpora[v]), gold_preproc=True, silent=True)
            except Exception as e:                     # a variant that cannot be scored is data
                print(f"  {label}/{v}: FAILED {e}")    # too, so record it rather than abort
                res[v] = None
        allres[label] = res
        print(f"\n{label} (gold-preproc; same trees, only the vocalisation differs)")
        print("  " + "variant".ljust(10) + "".join(n.rjust(8) for n, _ in METRICS))
        for v in variants:
            r = res[v]
            cells = "".join((f"{r[k] * 100:8.2f}" if r and r.get(k) is not None else "     n/a")
                            for _, k in METRICS)
            print("  " + v.ljust(10) + cells)
        base = res.get(variants[0])
        if base and base.get("dep_las") is not None:
            got = [r["dep_las"] for r in res.values() if r and r.get("dep_las") is not None]
            print(f"  LAS spread across orthographies: {(max(got) - min(got)) * 100:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
