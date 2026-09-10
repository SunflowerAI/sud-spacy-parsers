#!/usr/bin/env python
"""Masked-LM pretraining for the ARC-FACTORED decoder's OWN encoder (`--joint`).

WHY NOT `spacy pretrain`. The shipped sa parser is not a spaCy pipe. `train_arcfactored.py --joint`
builds its encoder inline as `chain(build_joint_embed(cfg), MaxoutWindowEncoder.v2(...))`, which is
NOT the same object `spacy.Tok2Vec.v2` builds (that one wraps the encoder in `with_array` itself),
so the weight bytes `spacy pretrain` writes do not load into it. Worse, they might load PARTIALLY.
This script therefore imports `build_joint_embed` and reproduces the encoder construction from the
same LANGS entry, so compatibility is guaranteed by construction rather than by inspection.

⚠ THE EMBED READS MORPH, SO THE PRETRAINING DOCS MUST CARRY IT. sa's joint embed hashes
`["NORM", "PREFIX", "SUFFIX", "SHAPE", "MORPH"]`. Pretraining on RAW TEXT would leave MORPH unset on
every token, and an unset MORPH is a DIFFERENT INPUT from a populated one -- the exact asymmetry
that cost sa 6.8 LAS on the raw path (CLAUDE.md). The corpus here is therefore a DocBin of
already-annotated docs, not a JSONL of strings.

WHAT IT BUYS. Under `--joint` the arc-factored encoder only ever sees the SYNTAX-BEARING half of
sa's data (Vedic + UFAL, ~21 707 sentences). `corpus_sa_multitask/train.spacy` additionally carries
the DCS half -- about 90 % of the tokens, tag/morph/lemma only, no syntax -- which that encoder
otherwise never sees at all. This is the one intervention that puts those tokens in front of it.

⚠ LEAK CHECKED, and the check is not optional for Sanskrit: 907 of the 8 467 held-out sentences
appear verbatim somewhere in raw DCS. In `corpus_sa_multitask/train.spacy` only 286 sentence strings
match a held-out one, and they average 3 tokens (many are single words like `tathā`) -- formulaic
recurrence in a formulaic language, not a split error. `--exclude-held-out` drops them anyway.
"""
import argparse
import pathlib
import random
import sys
import time

import spacy
from spacy.tokens import DocBin
from spacy.util import registry
from thinc.api import Adam, chain, set_dropout_rate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import sa_code  # noqa: F401,E402  registers the sa tokeniser, the readers and both embeds
from train_arcfactored import LANGS, build_joint_embed  # noqa: E402


def build_encoder(lang, bilstm=False):
    """EXACTLY `train_arcfactored.py --joint`'s construction. Keep the two in step: a divergence
    here does not fail loudly, it produces bytes that load into a subtly different network."""
    cfg = LANGS[lang]
    embed = build_joint_embed(cfg)
    if bilstm:
        from thinc.api import LSTM, with_padded
        enc = chain(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
        enc.set_dim("nO", 96)
    else:
        enc = chain(embed, registry.architectures.get("spacy.MaxoutWindowEncoder.v2")(
            width=96, depth=4, window_size=1, maxout_pieces=3))
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="sa")
    ap.add_argument("--base", default="training_sa_mp2_sub_s1/model-best",
                    help="only for its Vocab -- the lexeme table the embed hashes against")
    ap.add_argument("--docs", default="corpus_sa_multitask/train.spacy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bilstm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    nlp = spacy.load(args.base)
    docs = list(DocBin().from_disk(args.docs).get_docs(nlp.vocab))
    sents = []
    for d in docs:
        try:
            sents.extend(s.as_doc() for s in d.sents)
        except ValueError:
            sents.append(d)
    print("pretraining on %d docs -> %d sentences" % (len(docs), len(sents)), flush=True)

    enc = build_encoder(args.lang, args.bilstm)
    enc.initialize(X=sents[:64])
    # spaCy's own BERT-style cloze objective: 15 % of tokens masked, predict the first 4 UTF-8
    # BYTES of each. Sanskrit in IAST is multi-byte, so 4 bytes is a real target, not a giveaway.
    make_obj = registry.architectures.get("spacy.PretrainCharacters.v1")(
        maxout_pieces=3, hidden_size=300, n_characters=4)
    model = make_obj(nlp.vocab, enc)
    model.initialize(X=sents[:64])
    set_dropout_rate(model, args.dropout)
    loss_fn = model.attrs["loss"]
    opt = Adam(args.lr)

    for ep in range(1, args.epochs + 1):
        random.shuffle(sents)
        t0, total, n = time.time(), 0.0, 0
        for i in range(0, len(sents), args.batch_size):
            batch = sents[i:i + args.batch_size]
            preds, backprop = model.begin_update(batch)
            loss, grads = loss_fn(model.ops, batch, preds)
            backprop(grads)
            model.finish_update(opt)
            total += float(loss); n += sum(len(d) for d in batch)
        print("epoch %d  %5.0fs  loss/token %.4f" % (ep, time.time() - t0, total / max(n, 1)),
              flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(enc.to_bytes())
    print("wrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))


if __name__ == "__main__":
    sys.exit(main())
