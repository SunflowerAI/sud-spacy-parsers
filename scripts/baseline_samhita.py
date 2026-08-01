#!/usr/bin/env python3
"""Trivial baselines for the saṃhitā -> CSL task — the Phase 2 learnability gate.

Before writing a neural model, establish how much of this task is local. Three baselines, each
trained on the training pairs and scored on test with `scripts/eval_samhita.py`:

  allkeep   predict "=" everywhere. The floor: it scores 0 on every split metric while still
            getting ~84 % of characters right, which is why character accuracy is a useless metric
            here and the evaluation reports split F instead.
  unigram   per-character majority label.
  ngram     majority label per character n-gram window, backing off to shorter windows and finally
            to the character. This is the real gate: a window count model measures how much of the
            decision is visible in a fixed local context, which is exactly what a CNN tagger
            exploits. If a wide n-gram cannot get near the target, more layers will not either.

    baseline_samhita.py TRAIN.jsonl TEST.jsonl [--out-dir data_samhita] [--radius 1 2 3 4 5]
"""
import argparse
import json
import pathlib
from collections import Counter, defaultdict

PAD = "\x02"


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def window(text, i, r):
    lo, hi = i - r, i + r + 1
    return "".join(text[j] if 0 <= j < len(text) else PAD for j in range(lo, hi))


def fit_ngram(rows, radii):
    """radius -> {window: majority label}. Also returns the global majority label."""
    counts = {r: defaultdict(Counter) for r in radii}
    overall = Counter()
    for row in rows:
        s, labs = row["samhita"], row["labels"]
        for i, lab in enumerate(labs):
            overall[lab] += 1
            for r in radii:
                counts[r][window(s, i, r)][lab] += 1
    tables = {r: {w: c.most_common(1)[0][0] for w, c in counts[r].items()} for r in radii}
    return tables, overall.most_common(1)[0][0]


def predict(rows, tables, fallback, radii_desc):
    out = []
    for row in rows:
        s = row["samhita"]
        labs = []
        for i in range(len(s)):
            lab = None
            for r in radii_desc:                      # longest window first, back off
                lab = tables[r].get(window(s, i, r))
                if lab is not None:
                    break
            labs.append(lab if lab is not None else fallback)
        out.append({"sent_id": row["sent_id"], "labels": labs})
    return out


def write(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--out-dir", default="data_samhita")
    ap.add_argument("--radius", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()

    tr, te = load(a.train), load(a.test)
    out_dir = pathlib.Path(a.out_dir)
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from eval_samhita import score, report

    print(f"train {len(tr)} sentences, test {len(te)}\n")

    allkeep = [{"sent_id": r["sent_id"], "labels": ["="] * len(r["samhita"])} for r in te]
    report(score(te, {r["sent_id"]: r["labels"] for r in allkeep}), "allkeep (floor)", 3)
    write(allkeep, out_dir / "pred_test_allkeep.jsonl")

    radii = sorted(a.radius)
    tables, fallback = fit_ngram(tr, [0] + radii)
    print()
    uni = predict(te, tables, fallback, [0])
    report(score(te, {r["sent_id"]: r["labels"] for r in uni}), "unigram (character majority)", 5)
    write(uni, out_dir / "pred_test_unigram.jsonl")

    for r in radii:
        desc = sorted([x for x in radii if x <= r], reverse=True) + [0]
        pred = predict(te, tables, fallback, desc)
        print()
        report(score(te, {x["sent_id"]: x["labels"] for x in pred}),
               f"ngram radius {r} (window {2 * r + 1} chars, backing off)", 6)
        write(pred, out_dir / f"pred_test_ngram{r}.jsonl")


if __name__ == "__main__":
    main()
