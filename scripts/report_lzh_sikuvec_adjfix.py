#!/usr/bin/env python3
"""Re-measure the SikuBERT-vector morphologiser's failure-slice numbers on the CURRENT
(post-ADJ-recode, v0.3.2-generation) lzh arm. Adapted from scripts/report_lzh_sikuvec.py,
which reads the PRE-recode gold conllu (no .adjfix suffix) -- reusing it as-is against an
adjfix-trained model would score against the wrong UPOS inventory (VERB+Degree=Pos vs ADJ)
and silently corrupt every former-VERB+Degree=Pos token's accuracy. This script points
`blocks()` at the .adjfix gold instead and otherwise reuses probe_lzh_sikubert.report()
verbatim, which is suffix-agnostic.
"""
import collections
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_lzh_sikubert import report  # noqa: E402

D = "assets_lzh/SUD_Classical_Chinese-Kyoto"
SUFFIX = "relabeled_ext.udep_ruled.punct.rulemerged.adjfix.conllu"


def blocks(split):
    cur = []
    for line in open(f"{D}/lzh_kyoto-sud-{split}.{SUFFIX}", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                yield cur
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append((f[1], f[3]))
    if cur:
        yield cur


def main():
    seeds = [0, 1, 2]
    arms = ["morph_adjfix_siku", "morph_adjfix_siku_ctl"]
    baseline = "training_lzh_seg_morph_adjfix/model-best"

    import importlib.util

    def load_code(path):
        spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc

    tr, te = list(blocks("train")), list(blocks("test"))
    trforms = collections.Counter(w for b in tr for w, _ in b)
    trchars = set("".join(trforms))
    yte = np.array([p for b in te for _, p in b])
    forms = np.array([w for b in te for w, _ in b], dtype=object)
    print(f"train {sum(len(b) for b in tr)} tokens / {len(tr)} blocks;  "
          f"test {sum(len(b) for b in te)} tokens / {len(te)} blocks")
    print(f"trchars: {len(trchars)}")

    # report per seed, per arm, plus the no-vector baseline once
    slice_summary = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm in arms:
        for s in seeds:
            path = f"training_lzh_{arm}_s{s}/model-best"
            if not pathlib.Path(path).exists():
                print(f"MISSING: {path}")
                continue
            nlp = spacy.load(path)
            docs = [Doc(nlp.vocab, words=[w for w, _ in b], spaces=[False] * len(b)) for b in te]
            pred = []
            for d in nlp.pipe(docs, batch_size=64):
                pred += [t.pos_ for t in d]
            pred = np.array(pred)
            label = f"{arm} s{s}"
            report(yte, pred, forms, trforms, trchars, label=label)
            slices = {
                "ALL": np.ones(len(yte), bool),
                "form UNSEEN in train": np.array([f not in trforms for f in forms]),
                "char absent from the treebank": np.array(
                    [any(c not in trchars for c in f) for f in forms]),
            }
            for name, m in slices.items():
                if m.sum() == 0:
                    continue
                acc = (yte[m] == pred[m]).mean() * 100
                slice_summary[arm][name].append(acc)

    if pathlib.Path(baseline).exists():
        nlp = spacy.load(baseline)
        docs = [Doc(nlp.vocab, words=[w for w, _ in b], spaces=[False] * len(b)) for b in te]
        pred = []
        for d in nlp.pipe(docs, batch_size=64):
            pred += [t.pos_ for t in d]
        pred = np.array(pred)
        report(yte, pred, forms, trforms, trchars, label="no-vector baseline (production)")
        slices = {
            "ALL": np.ones(len(yte), bool),
            "form UNSEEN in train": np.array([f not in trforms for f in forms]),
            "char absent from the treebank": np.array(
                [any(c not in trchars for c in f) for f in forms]),
        }
        for name, m in slices.items():
            if m.sum() == 0:
                continue
            acc = (yte[m] == pred[m]).mean() * 100
            slice_summary["baseline"][name].append(acc)
    else:
        print(f"MISSING baseline: {baseline}")

    print("\n=== 3-seed summary per slice ===")
    import statistics
    for arm, sl in slice_summary.items():
        for name, vals in sl.items():
            if len(vals) > 1:
                sd = statistics.stdev(vals)
                print(f"{arm:24s} {name:32s} mean {statistics.mean(vals):6.2f}  sd {sd:5.2f}  "
                      f"per-seed {[f'{v:.2f}' for v in vals]}")
            else:
                print(f"{arm:24s} {name:32s} {vals}")


if __name__ == "__main__":
    main()
