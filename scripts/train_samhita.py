#!/usr/bin/env python3
"""Train the saṃhitā -> CSL character tagger (`scripts/sa_presegment.py`).

A standalone Thinc loop rather than `spacy train`: this is not a Doc-level component, has no
Example/Scorer plumbing, and forcing it into a spaCy pipe would mean inventing a fake component
just to borrow a training loop.

Early stopping tracks **dev split-location F**, not character accuracy — 84 % of characters are
plain "keep", so a model that does nothing already scores 84 % on accuracy and 0 on every metric
that matters.

The keep class is deliberately NOT downweighted: the split metrics are computed separately anyway,
and reweighting mostly trades precision for recall on exactly the number being reported.

    train_samhita.py data_samhita/train.jsonl data_samhita/dev.jsonl models/sa_presegment \\
        [--width 64] [--depth 6] [--epochs 30] [--batch-size 32] [--lr 0.001]
"""
import argparse
import json
import pathlib
import random
import sys

from thinc.api import Adam, fix_random_seed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sa_presegment import Presegmenter, build_vocabs      # noqa: E402
from eval_samhita import score                            # noqa: E402


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def minibatch(rows, size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    # sort within a window by length so batches are roughly uniform, then shuffle the batches
    batches = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(batches)
    return [[rows[i] for i in b] for b in batches]


def dev_score(seg, dev):
    preds = seg.predict([r["samhita"] for r in dev])
    return score(dev, {r["sent_id"]: p for r, p in zip(dev, preds)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", default="", help="comma-separated sub-character channels, e.g. "
                                              "'radical,qieyun' (see scripts/lzh_char_channels.py)")
    a = ap.parse_args()

    fix_random_seed(a.seed)
    rng = random.Random(a.seed)
    tr, dev = load(a.train), load(a.dev)
    chars, labels = build_vocabs(tr)
    aux = None
    if a.aux:
        from lzh_char_channels import build as build_channels
        aux = build_channels([x for x in a.aux.split(",") if x])
    seg = Presegmenter(chars, labels, width=a.width, depth=a.depth, aux=aux)
    # Provenance, so a downstream bundler can REFUSE a segmenter trained on the wrong generation of
    # the data. lzh ships traditional-only; a both-scripts segmenter has a near-identical character
    # inventory and is indistinguishable by any check on the weights (measured: 1.6 % vs 1.5 % of
    # characters absent from the arm's vocab -- the overlap test cannot tell them apart).
    seg.corpus = str(pathlib.Path(a.train).parent)
    print(f"train {len(tr)}  dev {len(dev)}  chars {len(chars)}  labels {len(labels)}  "
          f"width {a.width} depth {a.depth} (receptive field +/-{a.depth})")

    X0 = [seg.encode_chars(r["samhita"]) for r in tr[:64]]
    seg.model.initialize(X=X0)
    n_params = sum(int(nd.get_param(k).size) for nd in seg.model.walk()
                   for k in nd.param_names if nd.has_param(k))
    print(f"parameters {n_params:,}  (~{n_params * 4 / 1e6:.2f} MB fp32)")

    opt = Adam(a.lr)
    ops = seg.model.ops
    best, best_epoch, since = -1.0, -1, 0
    out = pathlib.Path(a.out)
    for epoch in range(a.epochs):
        total = 0.0
        for batch in minibatch(tr, a.batch_size, rng):
            batch = [r for r in batch if r["samhita"]]
            if not batch:
                continue
            X = [seg.encode_chars(r["samhita"]) for r in batch]
            gold = [seg.encode_labels(r["labels"]) for r in batch]
            scores, backprop = seg.model.begin_update(X)
            grads, n = [], sum(len(g) for g in gold)
            for s, g in zip(scores, gold):
                d = s.copy()
                d[ops.xp.arange(len(g)), ops.asarray1i(g)] -= 1.0
                total += float(-ops.xp.log(
                    s[ops.xp.arange(len(g)), ops.asarray1i(g)] + 1e-9).sum())
                grads.append(d / n)
            backprop(grads)
            seg.model.finish_update(opt)
        s = dev_score(seg, dev)
        loc_f = s["split_location"][2]
        full_f = s["full_label"][2]
        flag = ""
        if loc_f > best:
            best, best_epoch, since = loc_f, epoch, 0
            seg.to_disk(out)
            flag = "  <- saved"
        else:
            since += 1
        print(f"  epoch {epoch:3d}  loss {total / max(1, len(tr)):8.3f}  "
              f"dev split-loc F {100 * loc_f:6.2f}  full-label F {100 * full_f:6.2f}  "
              f"PM {100 * s['sentence_pm']:6.2f}{flag}")
        if since >= a.patience:
            print(f"  early stop (no dev improvement for {a.patience} epochs)")
            break
    print(f"best dev split-location F {100 * best:.2f} at epoch {best_epoch} -> {out}")


if __name__ == "__main__":
    main()
