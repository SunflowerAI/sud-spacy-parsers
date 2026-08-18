#!/usr/bin/env python3
"""Replace a recipient's LISTENER tagger with a donor's SELF-CONTAINED one, behind the morphologiser.

`graft_xpos_tagger.py` is the tool for this when the two arms differ ONLY in the tagger: it checks
that every shared component is byte-identical, so the grafted tagger is fed by the model it was
trained against. That check is right there and inapplicable here. The `lemvec` arm changes the
PARSER'S ENCODER (`sud.LemmaVecFeatsEmbed.v1`), so its tok2vec and parser necessarily differ from
the released arm's, and the byte-identity check can never pass.

WHAT MAKES THE GRAFT SOUND ANYWAY is that the donor tagger is not a listener. It carries
`sud.Tok2VecPlusFeats.v1` — its own `HashEmbedCNN` plus a FEATS block — so it reads the DOC
(the morphologiser's POS/MORPH) and its own weights, and nothing else in the recipient. A listener
tagger could not be moved this way at all, which is why that is the first thing refused.

    recipient  [morphologizer, lemmatizer, tok2vec, tagger(listener), parser]
    result     [morphologizer, lemmatizer, tok2vec, parser, tagger(standalone)]

FOUR THINGS ARE CHECKED, not assumed:

  * the donor tagger is standalone -- a listener would silently read the WRONG encoder;
  * a morphologiser runs before the tagger's new position, which is the whole point of the
    conditioned tagger and what `package_sud.sh`'s guard enforces at packaging time;
  * the result reproduces the RECIPIENT'S PARSE exactly, heads and deprels token for token. The
    parser reads the encoder and not TAG, so dropping a listener off the tok2vec should change
    nothing -- "should" is not evidence, and every LAS figure depends on it;
  * the result reproduces the DONOR'S TAGS exactly, so the graft moved what it meant to.

⚠ The verification loads the RESULT BACK FROM DISK (standing hazard 8). An in-memory check would
pass on a model whose weights never reached the file.

    graft_standalone_tagger.py training_la_lemvec_sealed training_la_xposwarm/model-best OUT \\
        --corpus corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402


def is_listener(nlp, name="tagger") -> bool:
    cfg = nlp.config["components"][name].get("model", {})
    return "Listener" in str(cfg.get("tok2vec", {}).get("@architectures", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipient")
    ap.add_argument("donor")
    ap.add_argument("out_model", type=pathlib.Path)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pipe", default="tagger")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    rec = spacy.load(args.recipient)
    don = spacy.load(args.donor)
    print(f"recipient {args.recipient}: {rec.pipe_names}")
    print(f"donor     {args.donor}: {don.pipe_names}")

    if is_listener(don, args.pipe):
        raise SystemExit(f"REFUSING: the donor's {args.pipe} is a LISTENER. Grafted into another "
                         f"arm it would read that arm's encoder, which it was never trained "
                         f"against, and the failure is silent.")
    if "morphologizer" not in rec.pipe_names:
        raise SystemExit("REFUSING: the recipient has no morphologiser, so a tagger conditioned on "
                         "UPOS+FEATS has nothing to read.")

    docs = list(DocBin().from_disk(args.corpus).get_docs(spacy.blank("la").vocab))[:args.limit]
    def build(nlp, g):
        return Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
    before = [[(t.head.i, t.dep_) for t in rec(build(rec, g))] for g in docs]
    donor_tags = [[t.tag_ for t in don(build(don, g))] for g in docs]

    if args.pipe in rec.pipe_names:
        rec.remove_pipe(args.pipe)
    # at the END, which is behind the morphologiser by construction and before any sud_* pipe the
    # packaging step adds later (those are appended after this runs).
    rec.add_pipe(args.pipe, source=don, name=args.pipe)
    print(f"result    {rec.pipe_names}")

    if args.out_model.exists():
        shutil.rmtree(args.out_model)
    rec.to_disk(args.out_model)

    out = spacy.load(args.out_model)          # RELOADED, never the in-memory one
    ok_parse = ok_tag = tot = 0
    for g, b, dt in zip(docs, before, donor_tags):
        got = out(build(out, g))
        for i, t in enumerate(got):
            tot += 1
            ok_parse += (t.head.i, t.dep_) == b[i]
            ok_tag += t.tag_ == dt[i]
    print(f"  parse preserved: {ok_parse}/{tot}    donor tags reproduced: {ok_tag}/{tot}")
    if ok_parse != tot:
        raise SystemExit("FAIL: the graft changed the parse")
    if ok_tag != tot:
        raise SystemExit("FAIL: the grafted tagger does not reproduce the donor's tags")
    pipe = out.pipe_names
    if pipe.index(args.pipe) < pipe.index("morphologizer"):
        raise SystemExit(f"FAIL: {args.pipe} still precedes morphologizer: {pipe}")
    print("  OK — parse unchanged, tags are the donor's, tagger sits behind the morphologiser")


if __name__ == "__main__":
    main()
