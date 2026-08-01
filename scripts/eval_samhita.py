#!/usr/bin/env python3
"""Score saṃhitā -> CSL pre-tokeniser predictions against the gold labels.

Four metrics, in increasing strictness. The first and third are the ones comparable to the
literature (rcNN-SS 96.84 F / 87.08 sentence-PM, TransLIST 98.86 / 93.97, ByT5 93.83 PM) — but do
not treat those as a like-for-like target: they are measured on real editorial sandhi with 10-100x
more data, whereas this data is synthesised from a 20 k-sentence treebank.

  split-location  a break is inserted here (either kind). What a segmenter is usually judged on.
  split-type      the break is inserted AND is the right kind (word vs compound). Project-specific
                  and load-bearing: the compound divider is what `sa_tokenizer` reads to stamp
                  `Compound=Yes`, worth +1.30 LAS, so getting the type wrong costs real accuracy.
                  No published system reports this, because they conflate the two.
  full-label      the entire label matches, i.e. the coalescence is also resolved correctly. This
                  is "split-rule" in Hellwig & Nehrdich's terms.
  sentence PM     every label in the sentence is right.

Also breaks out per-label F for the hard classes. The literature's worst case is `ā-ā` (F 53-80
even for the state of the art); the analogue here is the `'`/`"` + `â`/`ā` family, since an `ā` on
the surface can be a plain `ā`, a word break, or either of two coalescences.

    eval_samhita.py GOLD.jsonl PRED.jsonl        # PRED has {"sent_id", "labels"}
"""
import argparse
import json
from collections import defaultdict

BREAKS = (" ", "-")


def has_break(label):
    return any(d in label for d in BREAKS)


def break_type(label):
    """'w', 'c', 'wc'... — the divider characters in the label, or '' if it inserts no break."""
    return "".join("w" if ch == " " else "c" for ch in label if ch in BREAKS)


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold_rows, pred_labels):
    """gold_rows: list of dicts with sent_id/labels. pred_labels: {sent_id: [labels]}."""
    loc = [0, 0, 0]          # tp, pred, gold
    typ = [0, 0, 0]
    full = [0, 0, 0]
    pm = n_sent = 0
    per_label = defaultdict(lambda: [0, 0, 0])
    skipped = 0
    for row in gold_rows:
        g = row["labels"]
        p = pred_labels.get(row["sent_id"])
        if p is None or len(p) != len(g):
            skipped += 1
            continue
        n_sent += 1
        pm += (g == p)
        for gl, pl in zip(g, p):
            if has_break(gl):
                loc[2] += 1
            if has_break(pl):
                loc[1] += 1
            if has_break(gl) and has_break(pl):
                loc[0] += 1
                if break_type(gl) == break_type(pl):
                    typ[0] += 1
            if break_type(gl):
                typ[2] += 1
            if break_type(pl):
                typ[1] += 1
            if gl != "=":
                full[2] += 1
                per_label[gl][2] += 1
            if pl != "=":
                full[1] += 1
                per_label[pl][1] += 1
            if gl != "=" and gl == pl:
                full[0] += 1
                per_label[gl][0] += 1
    return {
        "sentences": n_sent, "skipped": skipped,
        "split_location": prf(*loc), "split_type": prf(*typ), "full_label": prf(*full),
        "sentence_pm": pm / n_sent if n_sent else 0.0,
        "per_label": {k: (prf(*v), v[2]) for k, v in per_label.items()},
    }


def report(s, label="", per_label_top=10):
    if label:
        print(f"--- {label} ---")
    print(f"  sentences {s['sentences']}" + (f"  (skipped {s['skipped']})" if s["skipped"] else ""))
    for k in ("split_location", "split_type", "full_label"):
        p, r, f = s[k]
        print(f"  {k:15s} P {100 * p:6.2f}  R {100 * r:6.2f}  F {100 * f:6.2f}")
    print(f"  {'sentence_PM':15s} {100 * s['sentence_pm']:6.2f}")
    hard = sorted(s["per_label"].items(), key=lambda kv: -kv[1][1])[:per_label_top]
    print("  per-label F (by gold frequency):")
    for lab, ((p, r, f), n) in hard:
        print(f"    {lab!r:10s} n={n:6d}  P {100 * p:6.2f}  R {100 * r:6.2f}  F {100 * f:6.2f}")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    preds = {r["sent_id"]: r["labels"] for r in load(a.pred)}
    report(score(load(a.gold), preds), a.label)


if __name__ == "__main__":
    main()
