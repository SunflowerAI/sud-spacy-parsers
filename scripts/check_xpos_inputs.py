#!/usr/bin/env python3
"""Prove the XPOS-downstream arm actually SEES UPOS and FEATS during training.

The failure this guards against is silent: if `annotating_components` does not run the
morphologiser, the predicted docs the tagger trains on carry no POS and no MORPH, the two new
embedding channels see a constant, and training completes normally with a plausible-looking
score. Nothing raises. (CLAUDE.md records the same trap costing every `--structural` SUD arm.)

So build the corpus exactly as `spacy train` does -- same reader, same config -- run the declared
annotating components over the PREDICTED docs, and report how much of each input feature is there.

    check_xpos_inputs.py configs/config_ar_xpos.cfg --train corpus_ar_sud/train.spacy
"""
import argparse
import os
import sys

import spacy
from spacy.training.initialize import init_nlp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_code  # noqa: F401,E402   (custom readers, tokenisers and factories)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--limit", type=int, default=40, help="documents to inspect")
    a = ap.parse_args()

    overrides = {"paths.train": a.train, "paths.dev": a.dev or a.train}
    cfg = spacy.util.load_config(a.config, overrides=overrides, interpolate=True)
    nlp = init_nlp(cfg)

    annotating = cfg["training"]["annotating_components"]
    trained = [n for n in nlp.pipe_names if n not in cfg["training"]["frozen_components"]]
    print(f"pipeline            {nlp.pipe_names}")
    print(f"frozen              {list(cfg['training']['frozen_components'])}")
    print(f"annotating          {list(annotating)}")
    print(f"training            {trained}")

    tagger = nlp.get_pipe("tagger")
    tv = cfg["components"]["tagger"]["model"]["tok2vec"]
    arch = tv.get("@architectures", "")
    # two shapes: BOTTOM injection puts the channels in `embed`; TOP injection (sud.Tok2VecPlusFeats)
    # keeps a listener and hangs them off `feats_embed`.
    embed = tv.get("feats_embed", tv.get("embed", {})) or {}
    print(f"tagger tok2vec      {arch}")
    print(f"tagger embed attrs  {embed.get('attrs')}")
    print(f"tagger embed rows   {embed.get('rows')}")

    # A listener that the upstream component never registered receives a STALE buffer and nothing
    # raises -- the same class of silent failure as the missing annotating component. Nested inside
    # a `concatenate`, it is not obvious that spaCy still finds it, so this is checked, not assumed.
    if "Listener" in str(tv) and "tok2vec" in nlp.pipe_names:
        up = nlp.get_pipe("tok2vec")
        names = [getattr(l, "upstream_name", "?") for l in getattr(up, "listeners", [])]
        listening = [n for n in nlp.pipe_names
                     if n != "tok2vec" and any(nd.name == "tok2vec-listener"
                                               for nd in nlp.get_pipe(n).model.walk())]
        print(f"tok2vec listeners   {len(getattr(up, 'listeners', []))} registered {names}")
        print(f"components listening {listening}")
        if "tagger" not in listening:
            raise SystemExit("\nFAIL: the tagger holds no Tok2VecListener")
        if len(getattr(up, "listeners", [])) < len(listening):
            raise SystemExit("\nFAIL: upstream tok2vec did not register every listener -- the "
                             "tagger would read a stale buffer")
    print(f"tagger labels       {len(tagger.labels)}")
    # order matters: the tagger must come AFTER the morphologiser, or POS/MORPH are stale at runtime
    if "morphologizer" in nlp.pipe_names:
        assert nlp.pipe_names.index("morphologizer") < nlp.pipe_names.index("tagger"), \
            "tagger runs BEFORE morphologizer -- XPOS would not be downstream at inference"
        print("order               OK (morphologizer before tagger)")

    (corpus,) = spacy.util.resolve_dot_names(cfg, [cfg["training"]["train_corpus"]])
    examples = []
    for i, eg in enumerate(corpus(nlp)):
        if i >= a.limit:
            break
        examples.append(eg)

    # run exactly the components spacy train would run over the predicted docs
    docs = [eg.predicted for eg in examples]
    for name in annotating:
        docs = list(nlp.get_pipe(name).pipe(docs))

    n = sum(len(d) for d in docs)
    pos = sum(1 for d in docs for t in d if t.pos_ not in ("", None))
    morph = sum(1 for d in docs for t in d if len(t.morph) > 0)
    gold_x = sum(1 for eg in examples for t in eg.reference if t.tag_ not in ("", None))
    print(f"\n{len(docs)} docs / {n} tokens")
    print(f"  POS   set on predicted   {pos:7d}  {100*pos/max(n,1):6.2f} %")
    print(f"  MORPH set on predicted   {morph:7d}  {100*morph/max(n,1):6.2f} %")
    print(f"  XPOS  gold (the target)  {gold_x:7d}  {100*gold_x/max(n,1):6.2f} %")

    # per-FEATURE channels (sud.MultiHashEmbedFeats.v1): a channel whose feature the morphologiser
    # almost never predicts is a constant, which is exactly the dead-channel failure this script
    # exists to catch -- and it would not show up in the MORPH line above, since some OTHER feature
    # being present makes the bundle non-empty.
    feats = embed.get("feats") or []
    if feats:
        print("  per-feature channels (share of predicted tokens carrying a value):")
        for f in feats:
            got = sum(1 for d in docs for t in d if t.morph.get(f))
            flag = "   <- DEAD CHANNEL" if got == 0 else ""
            print(f"    {f:12s} {got:7d}  {100*got/max(n,1):6.2f} %{flag}")
        if all(sum(1 for d in docs for t in d if t.morph.get(f)) == 0 for f in feats):
            raise SystemExit("\nFAIL: every per-feature channel is empty")
    if pos == 0:
        raise SystemExit("\nFAIL: no POS on any predicted token -- the tagger would train on a "
                         "constant channel. Check annotating_components.")
    print("\nOK" if morph else "\nWARNING: MORPH empty everywhere -- the FEATS channel is dead")


if __name__ == "__main__":
    main()
