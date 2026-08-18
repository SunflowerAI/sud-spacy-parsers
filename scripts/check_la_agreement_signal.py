#!/usr/bin/env python3
"""Is Latin agreement DISCRIMINATIVE enough, under PREDICTED morphology, to be worth a channel?

The go/no-go that `configs/config_sa_multitask_agree.cfg` ran for Sanskrit before building anything:
gold ADJ--mod-->NOUN pairs came out 89.5 % case/number/gender-compatible against 65.4 % for random
nearby ADJ/NOUN non-arcs, and that 24-point gap is what justified the block.

Latin is asked the same question, with one difference that matters. Sanskrit reads CANDIDATE SETS
off the tokeniser, upstream of the encoder. Latin has no such analyser; the signal has to come from
the FROZEN MORPHOLOGISER that `config_la_lemvec.cfg` already runs in front of the parser, which sets
FEATS on only ~67 % of tokens and sets some of them wrong. So the contrast is measured twice:

    gold morph        the ceiling -- what the INFORMATION is worth (what analyse_la_agreement_errors
                      already used, and NOT what any shippable arm can read)
    predicted morph   what an actual channel would carry, morphologiser errors and gaps included

If the predicted-morph gap collapses towards zero the channel is not worth training, and this costs
two minutes rather than a night.

    .venv/bin/python scripts/check_la_agreement_signal.py \
        corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy \
        --morph training_la_aug_lemma/model-best
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy import util
from spacy.tokens import Doc, DocBin

sys.path.insert(0, str(Path(__file__).resolve().parent))

CNG = ("Case", "Number", "Gender")
AGREER = {"ADJ", "DET", "NUM"}
NOMINAL = {"NOUN", "PROPN", "PRON"}
#: the deprels an agreeing modifier actually carries in SUD
AGREE_DEPS = {"mod", "det"}


def compatible(a, b, keys) -> bool | None:
    """True/False if both sides declare every key, None if either does not (so it is not counted).

    Multi-valued features intersect rather than compare, which is what an underspecified form means.
    """
    out = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if not va or not vb:
            return None
        out.append(bool(set(va) & set(vb)))
    return all(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--morph", default="training_la_aug_lemma/model-best",
                    help="the arm whose PREDICTED FEATS a real channel would read")
    ap.add_argument("--code", type=Path, default=Path("scripts/seg_code.py"))
    ap.add_argument("--window", type=int, default=3, help="offsets each way for the negatives")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    blank = spacy.blank("la")
    gold_docs = list(DocBin().from_disk(args.corpus).get_docs(blank.vocab))
    nlp = spacy.load(args.morph)
    rng = random.Random(args.seed)

    c: Counter = Counter()
    for gold in gold_docs:
        pred = nlp(Doc(nlp.vocab, words=[t.text for t in gold],
                       spaces=[bool(t.whitespace_) for t in gold]))
        c["tok"] += len(gold)
        c["morph set"] += sum(1 for t in pred if str(t.morph))
        for t in gold:
            if t.pos_ not in AGREER:
                continue
            dep = t.dep_.split("@", 1)[0].split(":", 1)[0]
            h = t.head
            # POSITIVE: the real arc, when it goes to a nominal
            if dep in AGREE_DEPS and h.i != t.i and h.pos_ in NOMINAL:
                for src, tag in ((gold, "gold"), (pred, "pred")):
                    r = compatible(src[t.i].morph, src[h.i].morph, CNG)
                    if r is not None:
                        c[f"pos/{tag}/n"] += 1
                        c[f"pos/{tag}/ok"] += r
            # NEGATIVE: a nearby nominal that is NOT the head -- the distractor the parser must beat
            cands = [u for u in gold[max(0, t.i - args.window):t.i + args.window + 1]
                     if u.pos_ in NOMINAL and u.i != h.i and u.i != t.i]
            if not cands:
                continue
            u = rng.choice(cands)
            for src, tag in ((gold, "gold"), (pred, "pred")):
                r = compatible(src[t.i].morph, src[u.i].morph, CNG)
                if r is not None:
                    c[f"neg/{tag}/n"] += 1
                    c[f"neg/{tag}/ok"] += r

    print(f"{c['tok']} tokens; morphologiser set FEATS on {c['morph set'] / c['tok']:.2%}")
    print(f"  {'morphology':<10} {'arc pairs':>10} {'compatible':>11} "
          f"{'non-arc':>9} {'compatible':>11} {'gap':>7}")
    for tag in ("gold", "pred"):
        pn, po = c[f"pos/{tag}/n"], c[f"pos/{tag}/ok"]
        nn, no = c[f"neg/{tag}/n"], c[f"neg/{tag}/ok"]
        if not pn or not nn:
            print(f"  {tag:<10} no comparable pairs")
            continue
        p, n = po / pn, no / nn
        print(f"  {tag:<10} {pn:10d} {p:10.1%} {nn:9d} {n:10.1%} {p - n:+7.1%}")
    print("\n  'arc pairs' are gold ADJ/DET/NUM --mod|det--> NOUN/PROPN/PRON; 'non-arc' is a random "
          f"nominal within {args.window} tokens that is NOT the head.")
    print("  Coverage matters as much as the gap: a pair is counted only where BOTH tokens declare "
          "all of Case, Number and Gender, so `pred` having far fewer pairs is itself the finding.")


if __name__ == "__main__":
    main()
