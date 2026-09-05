#!/usr/bin/env python
"""Pretrain LANGS[lang]['pretrain_embed'] (char-hash + fastText lemma vectors, `sud.LemmaVecEmbed.v1`)
via a TAGGING objective (predict gold UPOS), so `train_arcfactored.py --joint --pretrain <ckpt>` can
start the arc-factored decoder's embed from something already mature, instead of purely random init.

WHY A TAGGING OBJECTIVE, AND WHY THIS EMBED SPECIFICALLY. The multi-seed sweep
(NEGATIVE-RESULTS.md) found `--joint`'s fresh embed+BiLSTM landing 11-15 LAS below the single
lucky-seed numbers this whole line of work was built on. `la_frozen` (the arc-factored decoder on
la's DEPLOYED, already-mature tok2vec) did better (~65 LAS) and far more stably -- but that encoder
was co-adapted to the TRANSITION parser's own objective, which is exactly the "borrowing the
competitor's own equipment" concern raised about using it for a fair comparison. A tagging-pretrained
encoder is trained via a task genuinely neutral between the two parsing architectures: neither
transition actions nor arc-factored biaffine scores ever touch it.

`pretrain_embed` (LANGS[lang], `sud.LemmaVecEmbed.v1`) is deliberately NOT the same architecture as
`joint_embed` (`sud.LemmaVecFeatsEmbed.v1` for la, plain `spacy.MultiHashEmbed.v2` for lzh): it has
NO per-feature morphology channel. Reading PREDICTED Case/Number/etc as an input to the very model
being trained to predict tags would be circular -- and the morphological signal the arc-factored
decoder needs still arrives at FINE-TUNING time via the existing, separately-verified LATE bias terms
(--agreement/--feat), which already read predicted morphology from a genuinely separate upstream
tagger. This script's embed reads only char-hash features (NORM/PREFIX/SUFFIX/SHAPE) and the fastText
lemma vector table -- nothing else.

⚠ INPUT REGIME MATCHES `--joint`'s OWN, NOT GOLD. `make_plain()` builds docs from PREDICTED upstream
output (LEMMA included, for la/lzh both, since their pipelines run lemmatizer before the parser) --
the exact same regime `train_arcfactored.py`'s embed reads LEMMA from at both train and inference
time. Training this embed's `LemmaVecExtractor` block against GOLD lemma here would reintroduce the
train/inference skew this project has been bitten by before (CLAUDE.md); only the UPOS TARGETS come
from gold `tr`/`te` (a tagger legitimately trains against gold, and `make_plain`'s docs and the
source gold docs share the same word order/count, so index-aligned zipping is safe).

⚠ SAVES ONLY THE EMBED'S BYTES (`embed.to_bytes()`), not a whole tagger -- `train_arcfactored.py
--pretrain` loads exactly this into a freshly-constructed, freshly-shaped `pretrain_embed` before
wrapping it in whatever contextualiser (--bilstm/--flat/MaxoutWindowEncoder default) and training the
biaffine on top, all still freshly seeded by --seed.

MUST STAY IN SYNC (the training-loop STRUCTURE, not any exported symbol) with
`train_arcfactored.py`'s own --joint loop: same per-document Adam step pattern (batch=1, one
`finish_update` per document), same `fix_random_seed`/`--decay` discipline, so a checkpoint pretrained
here composes cleanly with that script's own conventions.
"""
import argparse
import pathlib
import sys

import numpy as np
import spacy
from thinc.api import Adam, fix_random_seed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_arcfactored as tr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(tr.LANGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", required=True, help="path to write the best embed checkpoint (bytes)")
    a = ap.parse_args()
    fix_random_seed(a.seed)

    cfg = tr.LANGS[a.lang]
    nlp = spacy.load(cfg["src"])
    _, upstream = tr.encoder_and_upstream(nlp)
    print(f"  [{a.lang}] src={cfg['src']}  upstream: {upstream}", flush=True)

    gold_tr = tr.load(a.lang, "train", nlp, a.limit or None)
    gold_te = tr.load(a.lang, "test", nlp, (a.limit // 4) if a.limit else None)
    # ⚠ PREDICTED input (LEMMA included), GOLD targets -- see the module docstring's "INPUT REGIME"
    # note. Word order/count match the source gold docs 1:1, so index-aligned zip is safe.
    plain_tr = tr.make_plain(nlp, gold_tr, upstream)
    plain_te = tr.make_plain(nlp, gold_te, upstream)
    print(f"  train {len(plain_tr)} docs, test {len(plain_te)} docs", flush=True)

    y_tr = [np.array([tr.UPOS_INDEX.get(t.pos_, tr.N_UPOS - 1) for t in d]) for d in gold_tr]
    y_te = [np.array([tr.UPOS_INDEX.get(t.pos_, tr.N_UPOS - 1) for t in d]) for d in gold_te]

    embed = tr.build_embed_from_spec(cfg["pretrain_embed"])
    print(f"  embed: {cfg['pretrain_embed']['arch']} "
          f"({cfg['pretrain_embed']['kwargs'].get('attrs')})", flush=True)
    embed.initialize(X=plain_tr[:64])
    w = cfg["pretrain_embed"]["kwargs"]["width"]

    # ⚠ HAND-ROLLED HEAD, SAME PATTERN AS EVERY OTHER PARAMETER IN THIS PROJECT'S ARC-FACTORED
    # SCORERS (JointBiaffine's U/V/etc): thinc only builds/runs the embed itself; the tagging head
    # is plain numpy, trained via the SAME per-key Adam optimiser convention.
    rng = np.random.default_rng(a.seed)
    W = (rng.normal(size=(w, tr.N_UPOS)) * (1.0 / np.sqrt(w))).astype("float32")
    b = np.zeros(tr.N_UPOS, dtype="float32")
    opt = Adam(a.lr)

    best = (-1.0, -1)
    out = pathlib.Path(a.save)
    for ep in range(a.epochs):
        opt.learn_rate = a.lr * (a.decay ** ep)
        order = np.random.default_rng(ep).permutation(len(plain_tr))
        tot = 0.0
        for count, i in enumerate(order):
            doc = plain_tr[i]
            gold = y_tr[i]
            n = len(doc)
            if n == 0:
                continue
            Xs, bp = embed([doc], is_train=True)
            X = Xs[0]
            logits = X @ W + b
            Z = logits - logits.max(1, keepdims=True)
            P = np.exp(Z); P /= P.sum(1, keepdims=True)
            loss = -np.log(np.maximum(P[np.arange(n), gold], 1e-9)).sum()
            tot += loss / n
            dlogits = P.copy(); dlogits[np.arange(n), gold] -= 1.0
            gW = X.T @ dlogits; gb = dlogits.sum(0)
            dX = dlogits @ W.T
            bp([dX.astype("float32")])
            embed.finish_update(opt)
            W, _ = opt(("pretrain", "W"), W, gW.astype("float32"))
            b, _ = opt(("pretrain", "b"), b, gb.astype("float32"))
            if (count + 1) % 3000 == 0:
                print(f"    ep{ep} {count + 1}/{len(order)} loss {tot / (count + 1):.4f}",
                      flush=True)
        correct = total = 0
        for doc, gold in zip(plain_te, y_te):
            n = len(doc)
            if n == 0:
                continue
            X = embed.predict([doc])[0]
            pred = (X @ W + b).argmax(1)
            correct += int((pred == gold).sum()); total += n
        acc = correct * 100 / max(total, 1)
        print(f"  epoch {ep}: loss {tot / len(plain_tr):.4f}   UPOS_ACC {acc:.2f}"
              f"   (lr {opt.learn_rate:.2e})", flush=True)
        if acc > best[0]:
            best = (acc, ep)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(embed.to_bytes())
            print(f"    saved -> {out} (best UPOS_ACC {acc:.2f})", flush=True)


if __name__ == "__main__":
    main()
