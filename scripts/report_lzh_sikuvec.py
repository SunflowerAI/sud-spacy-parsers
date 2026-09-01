#!/usr/bin/env python3
"""Report the SikuBERT-vector morphologiser arms against their shuffled control.

Two tables, and the second is the one that answers the question. The aggregate is what hid the
kanripo vectors' real behaviour (63.30 % probe accuracy, +0.04 LAS), so UPOS is also reported on
the population the shipped arm actually fails on: forms unseen in train, multi-character tokens,
and tokens holding a character absent from the treebank.
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_lzh_sikubert import blocks, report  # noqa: E402


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=["sikuvec", "sikuvec_ctl"])
    ap.add_argument("--baseline", default="training_lzh_seg_morph/model-best",
                    help="the same layer with no vector channel at all")
    a = ap.parse_args()

    print(f"{'arm':<16}{'seed':>6}{'POS_ACC':>10}{'MORPH_ACC':>12}{'TAG':>10}")
    agg = collections.defaultdict(list)
    for arm in a.arms:
        for s in a.seeds:
            p = pathlib.Path(f"metrics/lzh/metrics_lzh_{arm}_s{s}_gp.json")
            if not p.exists():
                print(f"{arm:<16}{s:>6}   (no metrics file)")
                continue
            m = json.loads(p.read_text())
            pos = (m.get("pos_acc") or 0) * 100
            mor = (m.get("morph_acc") or 0) * 100
            tag = (m.get("tag_acc") or 0) * 100
            agg[arm].append(pos)
            print(f"{arm:<16}{s:>6}{pos:>10.2f}{mor:>12.2f}{tag:>10.2f}")
    print()
    for arm, v in agg.items():
        if v:
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"  {arm:<16} POS_ACC mean {statistics.mean(v):6.2f}  sd {sd:5.2f}  n={len(v)}")
    if len(agg) == 2 and all(agg.values()):
        (a1, v1), (a2, v2) = list(agg.items())
        print(f"\n  {a1} - {a2}: {statistics.mean(v1) - statistics.mean(v2):+.2f} POS_ACC "
              f"(mean over {min(len(v1), len(v2))} seeds)")
        print("  ⚠ compare that difference against the sd above before calling it a result.")

    # --- the slice table, on seed 0 of each arm plus the no-vector baseline
    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc

    tr, te = list(blocks("train")), list(blocks("test"))
    trforms = collections.Counter(w for b in tr for w, _ in b)
    trchars = set("".join(trforms))
    yte = np.array([p for b in te for _, p in b])
    forms = np.array([w for b in te for w, _ in b], dtype=object)

    arms = [(f"{arm} s{a.seeds[0]}", f"training_lzh_{arm}_s{a.seeds[0]}/model-best")
            for arm in a.arms]
    if pathlib.Path(a.baseline).exists():
        arms.append(("no vector channel", a.baseline))
    for label, path in arms:
        if not pathlib.Path(path).exists():
            continue
        nlp = spacy.load(path)
        docs = [Doc(nlp.vocab, words=[w for w, _ in b], spaces=[False] * len(b)) for b in te]
        pred = []
        for d in nlp.pipe(docs, batch_size=64):
            pred += [t.pos_ for t in d]
        report(yte, np.array(pred), forms, trforms, trchars, label=label)


if __name__ == "__main__":
    main()
