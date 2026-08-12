#!/usr/bin/env python3
"""Graft a `_xposwarm` tagger into a shipping arm AND move it behind the morphologiser.

`graft_pipe.py` deliberately puts a replacement pipe back WHERE IT CAME FROM -- right for a swap,
and exactly wrong here, because the point of this tagger is that it runs AFTER the morphologiser and
reads its POS/MORPH. So the pipeline goes

    [tok2vec, tagger, parser, morphologizer, lemmatizer, sud_*]      released
    [tok2vec, parser, morphologizer, lemmatizer, tagger, sud_*]      grafted

which is the order the donor was TRAINED in. The tagger lands immediately before the first `sud_*`
pipe, so it keeps the invariant that those run last (and on lzh/sa `clause_parser`, added at
packaging `before=` the first `sud_*` pipe, still lands after the tagger -- which matters, because
it stamps punctuation XPOS and must not be overwritten).

THREE THINGS ARE CHECKED, not assumed:

  * every component the two arms SHARE is byte-identical, so the grafted tagger is fed by the model
    it was trained against (the check `graft_pipe.py` makes, for the same reason);
  * the reordered pipeline reproduces the recipient's PARSE exactly -- heads and deprels, token for
    token. The parser reads the encoder, not TAG, so moving the tagger past it should change
    nothing; "should" is not evidence, and every published LAS/UAS figure depends on it;
  * the grafted tagger reproduces the DONOR's tags exactly, so the graft moved what it meant to.

`[initialize]` is stripped of the warm-start callback on the way out: it is dead weight in a wheel
(spaCy does not resolve that block on load) and it names an arm the user will not have.

    graft_xpos_tagger.py training_ar_sud/model-best training_ar_xposwarm/model-best \\
        training_ar_sud_xw --corpus corpus_ar_ext/ar_padt-sud-test.relabeled_ext.spacy
"""
import argparse
import filecmp
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import seg_code  # noqa: F401,E402  (custom tokenisers, readers, architectures)
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402

SHARED = ("tok2vec", "parser", "morphologizer", "lemmatizer")


def sample(nlp, corpus, limit):
    docs = list(DocBin().from_disk(corpus).get_docs(nlp.vocab))[:limit]
    return [([t.text for t in d], [t.whitespace_ == " " for t in d]) for d in docs]


def run(nlp, sents):
    return [nlp(Doc(nlp.vocab, words=w, spaces=s)) for w, s in sents]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipient")
    ap.add_argument("donor")
    ap.add_argument("out_model")
    ap.add_argument("--corpus", required=True, help=".spacy corpus to verify on")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()

    rec, don = pathlib.Path(a.recipient), pathlib.Path(a.donor)
    for comp in SHARED:
        x, y = rec / comp / "model", don / comp / "model"
        if x.exists() and y.exists() and not filecmp.cmp(x, y, shallow=False):
            sys.exit(f"graft_xpos_tagger: {comp} differs between the arms -- they do not share a "
                     f"base, so the tagger would be fed a different model's predictions")

    nlp = spacy.load(a.recipient)
    donor = spacy.load(a.donor)
    before = list(nlp.pipe_names)
    sents = sample(nlp, a.corpus, a.limit)
    gold_parse = [[(t.head.i, t.dep_) for t in d] for d in run(nlp, sents)]
    gold_tags = [[t.tag_ for t in d] for d in run(donor, sents)]

    # position: immediately before the first sud_* pipe, i.e. after the morphologiser/lemmatiser
    nlp.remove_pipe("tagger")
    tail = [n for n in nlp.pipe_names if n.startswith("sud_")]
    where = {"before": tail[0]} if tail else {"last": True}
    nlp.add_pipe("tagger", source=donor, **where)
    i_m, i_t = nlp.pipe_names.index("morphologizer"), nlp.pipe_names.index("tagger")
    if i_m > i_t:
        sys.exit(f"graft_xpos_tagger: tagger at {i_t} is still before morphologizer at {i_m}")

    # the warm start is an initialisation-time concern; nothing should carry it into a wheel
    nlp.config["initialize"].pop("after_init", None)
    if "components" in nlp.config["initialize"]:
        nlp.config["initialize"]["components"].pop("tagger", None)
    pathlib.Path(a.out_model).parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(a.out_model)

    # carry the tagger's own score across -- to_disk writes the RECIPIENT's meta, so without this
    # the wheel reports the score of the tagger that was replaced (the graft_pipe.py lesson)
    mp = pathlib.Path(a.out_model) / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    dperf = json.loads((don / "meta.json").read_text(encoding="utf-8")).get("performance", {})
    moved = {}
    for k in ("tag_acc", "tag_micro_p", "tag_micro_r", "tag_micro_f"):
        if k in dperf and meta.get("performance", {}).get(k) != dperf[k]:
            moved[k] = (meta["performance"].get(k), dperf[k])
            meta["performance"][k] = dperf[k]
    if moved:
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out = spacy.load(a.out_model)
    docs = run(out, sents)
    n = sum(len(d) for d in docs)
    parse_bad = sum(1 for d, g in zip(docs, gold_parse)
                    for t, (h, dep) in zip(d, g) if t.head.i != h or t.dep_ != dep)
    tag_bad = sum(1 for d, g in zip(docs, gold_tags) for t, x in zip(d, g) if t.tag_ != x)
    print(f"{a.out_model}\n  {before}\n  -> {out.pipe_names}")
    for k, (x, y) in moved.items():
        print(f"  performance.{k}: {x} -> {y}")
    print(f"  parse unchanged: {n - parse_bad}/{n} tokens" + ("" if not parse_bad else "  <-- FAIL"))
    print(f"  tags match donor: {n - tag_bad}/{n} tokens" + ("" if not tag_bad else "  <-- FAIL"))
    if parse_bad or tag_bad:
        sys.exit("graft_xpos_tagger: verification FAILED")
    print("  OK")


if __name__ == "__main__":
    main()
