#!/usr/bin/env python3
"""Score Latin arms across orthographic variants of the SAME test set.

Every variant is the identical treebank with only the FORM column rewritten
(``make_la_variant_conllu.py``), so the trees, and therefore the gold, are held constant: the only
thing that moves between rows is the spelling. That makes the columns directly comparable and the
question sharp -- how much LAS does this arm lose when the text is printed a different way.

    .venv/bin/python scripts/eval_la_variants.py \
        --model union=training_la_seg/model-best --model aug=training_la_aug/model-best \
        --corpus-dir corpus_la_variants --out metrics/la/metrics_la_variants.json

All evaluation is gold-preproc, as everywhere else in this project outside English.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacy.cli.evaluate import evaluate

#: order matters only for reading: the two the union arm was trained for come first.
VARIANT_ORDER = ["plain", "macron", "mixed", "breve", "v", "vj", "lig", "caps", "lower", "all"]
METRICS = [("TAG", "tag_acc"), ("POS", "pos_acc"), ("LEMMA", "lemma_acc"),
           ("UAS", "dep_uas"), ("LAS", "dep_las")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--metrics", default="LAS,UAS,TAG,LEMMA",
                    help="comma-separated subset of TAG,POS,LEMMA,UAS,LAS to tabulate")
    args = ap.parse_args()

    models = [tuple(m.split("=", 1)) for m in args.model]
    shown = [m for m in METRICS if m[0] in args.metrics.split(",")]
    corpora = {p.stem.split(".")[-1]: p for p in sorted(args.corpus_dir.glob("*.spacy"))}
    variants = [v for v in VARIANT_ORDER if v in corpora] + \
               [v for v in corpora if v not in VARIANT_ORDER]

    results: dict[str, dict[str, dict]] = {}
    for name, path in models:
        results[name] = {}
        for variant in variants:
            scores = evaluate(path, str(corpora[variant]), gold_preproc=True, silent=True)
            results[name][variant] = {k: scores.get(k) for _, k in METRICS}
            print(f"  scored {name} on {variant}", flush=True)

    for label, key in shown:
        print(f"\n{label} (gold-preproc, same trees, only the spelling differs)")
        print(f"  {'variant':<10}" + "".join(f"{n:>12}" for n, _ in models) +
              ("       Δ" if len(models) == 2 else ""))
        for variant in variants:
            row = [results[n][variant].get(key) for n, _ in models]
            cells = "".join(f"{100 * v:>12.2f}" if isinstance(v, float) else f"{'--':>12}"
                            for v in row)
            delta = ""
            if len(row) == 2 and all(isinstance(v, float) for v in row):
                delta = f"{100 * (row[1] - row[0]):>+8.2f}"
            print(f"  {variant:<10}{cells}{delta}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
