#!/usr/bin/env python3
"""Where the Korean parser loses: unseen eojeol, and what a channel does about them.

WHY THIS EXISTS. `ko_sud_gsd` is tokenised by eojeol — stem plus particles fused into one
whitespace-delimited token — so every stem-and-particle combination is a fresh string. A third of
test tokens are strings the parser has never met, and the headline LAS hides which half of the data
it is losing. This splits every metric on that one fact.

It is also the FALSIFICATION TEST for `sud.KoAnalyserEmbed.v1`. That channel exists to give an
unseen eojeol a symbol the parser has a trained representation of, so its gain must land on the OOV
column. A gain spread evenly across seen and unseen tokens means something else is happening —
capacity, or the tagger — whatever the total says. `sud_lex_embed.py` records why the distinction
matters: a table keyed on the form is a function of the form, and conditioning on (form, f(form)) is
conditioning on the form, EXCEPT where the form has no trained representation to condition on.

    .venv/bin/python scripts/eval_ko_oov.py \
        corpus_ko_eojeol/ko_gsd-sud-test.relabeled_ext.spacy \
        --model rel=training_ko_eojeol_lemma/model-best \
        --model analyser=training_ko_analyser_s0/model-best

⚠ GOLD SENTENCES, GOLD TOKENS. Each gold sentence is parsed as its own Doc, which is what
`--gold-preproc` does, so these figures are comparable to the `metrics_ko_*_gp.json` set and NOT to
a raw end-to-end run. Sentence segmentation is deliberately out of scope here; it is a different
defect with a different fix (CLAUDE.md hazard 4).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import spacy
from spacy import util
from spacy.tokens import Doc, DocBin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def train_forms(conllu: pathlib.Path) -> set:
    forms = set()
    for line in conllu.open(encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        forms.add(f[1])
    return forms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--train-conllu", type=pathlib.Path,
                    default=pathlib.Path("assets_ko/SUD_Korean-GSD/"
                                         "ko_gsd-sud-train.relabeled_ext.conllu"))
    ap.add_argument("--code", type=pathlib.Path, default=pathlib.Path("scripts/seg_code.py"))
    ap.add_argument("--keys", action="store_true",
                    help="also report what the analyser's keys cover (needs the analyser installed)")
    args = ap.parse_args()

    util.import_file("cli_code", args.code)
    seen = train_forms(args.train_conllu)
    blank = spacy.blank("ko")
    gold = list(DocBin().from_disk(args.corpus).get_docs(blank.vocab))
    n_tok = sum(len(d) for d in gold)
    n_oov = sum(1 for d in gold for t in d if t.text not in seen)
    print(f"{args.corpus}: {len(gold)} docs, {sum(len(list(d.sents)) for d in gold)} sentences, "
          f"{n_tok} tokens")
    print(f"unseen eojeol (against {args.train_conllu.name}): {n_oov} = {n_oov / n_tok:.1%}\n")

    if args.keys:
        import ko_analyser
        keys = set()
        for line in args.train_conllu.open(encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            k = ko_analyser.stem(f[1])
            if k:
                keys.add(k)
        cov = Counter()
        for d in gold:
            for t in d:
                bucket = "seen" if t.text in seen else "OOV"
                cov[(bucket, "n")] += 1
                cov[(bucket, "key")] += (ko_analyser.stem(t.text) or "\0") in keys
        for b in ("seen", "OOV"):
            print(f"  first-morpheme key known to training, {b:>4} eojeol: "
                  f"{cov[(b, 'key')] / cov[(b, 'n')]:.1%}")
        print()

    hdr = f"{'arm':<22}" + "".join(f"{c:>26}" for c in ("all", "seen eojeol", "unseen eojeol"))
    print(hdr)
    print(f"{'':<22}" + "".join(f"{'UAS    LAS    TAG':>26}" for _ in range(3)))
    for spec in args.model:
        name, path = spec.split("=", 1)
        nlp = spacy.load(path)
        c: Counter = Counter()
        for g in gold:
            for sent in g.sents:
                words = [t.text for t in sent]
                spaces = [bool(t.whitespace_) for t in sent]
                pred = nlp(Doc(nlp.vocab, words=words, spaces=spaces))
                for k, gt in enumerate(sent):
                    pt = pred[k]
                    for b in ("all", "seen" if gt.text in seen else "OOV"):
                        c[(b, "tag_n")] += 1
                        c[(b, "tag")] += pt.tag_ == gt.tag_
                        # ⚠ PUNCTUATION IS EXCLUDED FROM UAS/LAS, because `spacy.parser_scorer.v1`
                        # excludes it (`ignore_labels = ("p", "punct")`). Scoring it here would put
                        # this script on a different scale from every `metrics_ko_*.json`, and the
                        # two sets would be quoted side by side in the same table — the shape of
                        # error NEGATIVE-RESULTS.md records as "never compare numbers from two
                        # different harnesses", where both numbers were right and only the
                        # comparison was invalid. Marks cost ~3 UAS here: they are 13 % of tokens
                        # and nearly free to attach.
                        if gt.dep_ in ("punct", "p"):
                            continue
                        c[(b, "n")] += 1
                        # head compared within the sentence, so a doc-level index cannot drift
                        if pt.head.i == gt.head.i - sent.start:
                            c[(b, "uas")] += 1
                            if pt.dep_ == gt.dep_:
                                c[(b, "las")] += 1
        row = f"{name:<22}"
        for b in ("all", "seen", "OOV"):
            n = c[(b, "n")] or 1
            tn = c[(b, "tag_n")] or 1
            row += (f"{100 * c[(b,'uas')] / n:8.2f}{100 * c[(b,'las')] / n:7.2f}"
                    f"{100 * c[(b,'tag')] / tn:7.2f}   ")
        print(row)


if __name__ == "__main__":
    main()
