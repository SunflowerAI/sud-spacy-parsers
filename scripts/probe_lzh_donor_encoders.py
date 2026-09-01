#!/usr/bin/env python3
"""Why does a BETTER donor encoder make a WORSE parser?

Handing the parser the morphologiser's UPOS-supervised encoder cost -0.54 LAS against the identical
architecture with a donor trained on shuffled vectors, on all three seeds (NEGATIVE-RESULTS.md).
The obvious hypothesis is that a representation optimised harder for UPOS has COLLAPSED the lexical
detail the parser needs: the better it gets at naming the category, the less of anything else
survives its 64 dimensions.

That is testable without training another parser. Take each donor's frozen 64-d output and fit a
linear probe on it for three targets:

  UPOS    what the donor was supervised on — the real-vector donor must win, or the premise is wrong
  DEPREL  what the parser needs — if the real-vector donor does NOT win here, the transfer has
          nothing to transfer
  FORM    lexical identity, as a direct measure of how much detail survived the compression

⚠ A probe on 64 dims is a low ceiling for FORM by construction; the number that matters is the
DIFFERENCE between the two donors, not its absolute level.
"""
import argparse
import collections
import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_lzh_sikubert import blocks  # noqa: E402


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def deprels(split, path_tpl):
    out = []
    for line in pathlib.Path(path_tpl.format(split=split)).open(encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        out.append(f[7])
    return out


def encode(donor, blks, vocab):
    from spacy.tokens import Doc
    enc = []
    for b in blks:
        doc = Doc(vocab, words=[w for w, _ in b], spaces=[False] * len(b))
        Y, _ = donor([doc], is_train=False)
        enc.append(np.asarray(Y[0]))
    return np.vstack(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="training_lzh_sikuvec_s0/model-best")
    ap.add_argument("--control", default="training_lzh_sikuvec_ctl_s0/model-best")
    ap.add_argument("--train-tokens", type=int, default=120000)
    ap.add_argument("--conllu", default="assets_lzh/SUD_Classical_Chinese-Kyoto/"
                    "lzh_kyoto-sud-{split}.relabeled_ext.udep_ruled.punct.rulemerged.conllu")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy
    from sklearn.linear_model import LogisticRegression
    from sud_frozen_pipe_tok2vec import load_encoder

    tr, te = list(blocks("train")), list(blocks("test"))
    keep, n = [], 0
    for b in tr:
        keep.append(b)
        n += len(b)
        if n >= a.train_tokens:
            break
    ntr = sum(len(b) for b in keep)

    y = {"UPOS": (np.array([p for b in keep for _, p in b]),
                  np.array([p for b in te for _, p in b])),
         "DEPREL": (np.array(deprels("train", a.conllu)[:ntr]),
                    np.array(deprels("test", a.conllu))),
         "FORM": (np.array([w for b in keep for w, _ in b]),
                  np.array([w for b in te for w, _ in b]))}
    print(f"train {ntr} tokens, test {sum(len(b) for b in te)}", flush=True)

    res = {}
    for name, path in (("real-vector donor", a.arm), ("shuffled-vector donor", a.control)):
        # ⚠ the host must hold the table the donor was TRAINED against — StaticVectors reads
        # doc.vocab.vectors at forward time, so the donor's own pipeline vocab is the right one.
        nlp = spacy.load(path)
        donor = load_encoder(path)
        Xtr, Xte = encode(donor, keep, nlp.vocab), encode(donor, te, nlp.vocab)
        res[name] = {}
        for tgt, (ytr, yte) in y.items():
            m = min(len(ytr), len(Xtr))
            clf = LogisticRegression(max_iter=1500).fit(Xtr[:m], ytr[:m])
            res[name][tgt] = 100 * (clf.predict(Xte) == yte[:len(Xte)]).mean()
            print(f"  {name:<24}{tgt:<8}{res[name][tgt]:6.2f}%", flush=True)

    print(f"\n{'target':<10}{'real donor':>14}{'shuffled donor':>18}{'Δ':>9}")
    for tgt in y:
        d = res["real-vector donor"][tgt] - res["shuffled-vector donor"][tgt]
        print(f"{tgt:<10}{res['real-vector donor'][tgt]:>14.2f}"
              f"{res['shuffled-vector donor'][tgt]:>18.2f}{d:>+9.2f}")


if __name__ == "__main__":
    main()
