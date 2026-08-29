#!/usr/bin/env python3
"""Go/no-go for a COMPOUND-INTERNAL-HEAD constraint on the Sanskrit parser.

The claim under test: the head of any NON-FINAL member of a compound lies inside that compound.
Before building a decode-time mask, two rates decide it, and only the second is the bar
(NEGATIVE-RESULTS, "Decode-time constraints on cross-clause arcs"):

  1. how often GOLD obeys the rule — an upper bound on what a hard ban can ever be right about;
  2. how often the arcs the ban would DESTROY are wrong — a mask pays only when the banned action
     is nearly always wrong, and 80 % consistency with gold is not a licence to constrain.

Reported per span definition and per allowed-label set, with the headroom arithmetic spelled out:
arcs currently correct that the ban destroys, against wrong ones it could fix.

    check_sa_compound_signal.py MODEL TEST.spacy [--limit N]
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from sud_constrained_parse import COMPOUND_ALLOWED, compound_spans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    parser = nlp.get_pipe("parser")
    blank = spacy.blank("sa")
    refs = list(DocBin().from_disk(a.test).get_docs(blank.vocab))
    sents = [s for d in refs for s in d.sents]
    if a.limit:
        sents = sents[: a.limit]

    def prep(s):
        """The doc the parser sees in DEPLOYMENT: predicted tag/morph/lemma, with the tokeniser's
        own `Compound` re-imposed over the morphologiser's — which is what `clause_parser` does in
        the released pipeline, and one of the three pieces the +1.30 LAS compound feature needs."""
        d = Doc(nlp.vocab, words=[t.text for t in s], spaces=[bool(t.whitespace_) for t in s])
        for pt, rt in zip(d, s):
            pt.norm_ = rt.norm_
            if rt.morph.get("Compound"):
                pt.set_morph("Compound=Yes")
        for name, pipe in nlp.pipeline:
            if name == "parser":
                break
            d = pipe(d)
        for pt, rt in zip(d, s):
            if rt.morph.get("Compound") and not pt.morph.get("Compound"):
                pt.set_morph((str(pt.morph) + "|Compound=Yes").lstrip("|"))
        return d

    stats = {}
    for variant in ("strict", "ortho"):
        stats[variant] = dict(members=0, gold_out=0, pred_out=0, pred_out_wrong=0,
                              pred_out_right=0, gold_out_rel=collections.Counter(),
                              pred_out_rel=collections.Counter(),
                              pred_out_right_rel=collections.Counter(),
                              member_right=0)
    tot = correct = 0
    for s in sents:
        gold_head = [t.head.i - s.start for t in s]
        gold_dep = [t.dep_ for t in s]
        d = prep(s)
        b = parser(d)
        tot += len(s)
        correct += sum(b[i].head.i == gold_head[i] for i in range(len(s)))
        for variant in ("strict", "ortho"):
            spans = compound_spans(b, extend_to_word=(variant == "ortho"))
            st = stats[variant]
            for i, span in enumerate(spans):
                if span is None:
                    continue          # not a non-final compound member
                lo, hi = span
                st["members"] += 1
                st["member_right"] += b[i].head.i == gold_head[i]
                if not (lo <= gold_head[i] <= hi) or gold_head[i] == i:
                    st["gold_out"] += 1
                    st["gold_out_rel"][gold_dep[i]] += 1
                ph = b[i].head.i
                if not (lo <= ph <= hi) or ph == i:
                    st["pred_out"] += 1
                    st["pred_out_rel"][b[i].dep_] += 1
                    if ph == gold_head[i]:
                        st["pred_out_right"] += 1
                        st["pred_out_right_rel"][b[i].dep_] += 1
                    else:
                        st["pred_out_wrong"] += 1

    print(f"{len(sents)} sentences / {tot} tokens; baseline UAS {correct/tot:.4f}")
    for variant in ("strict", "ortho"):
        st = stats[variant]
        m = max(st["members"], 1)
        print(f"\n== span definition: {variant}")
        print(f"  non-final compound members: {st['members']}"
              f"   (parser gets {st['member_right']/m:.4f} of their heads right)")
        print(f"  GOLD head outside the compound: {st['gold_out']} ({100*st['gold_out']/m:.2f} %)"
              f"  {st['gold_out_rel'].most_common(6)}")
        print(f"  PRED head outside the compound: {st['pred_out']} ({100*st['pred_out']/m:.2f} %)")
        po = max(st["pred_out"], 1)
        print(f"    of those, WRONG: {st['pred_out_wrong']} ({100*st['pred_out_wrong']/po:.2f} %)"
              f" — this is the bar; RIGHT: {st['pred_out_right']}")
        print(f"    labels the parser used going out: {st['pred_out_rel'].most_common(6)}")
        print(f"    labels on the ones it got RIGHT:  {st['pred_out_right_rel'].most_common(6)}")
        for label, exempt in (("ban everything", frozenset()), ("exempt `flat`", COMPOUND_ALLOWED)):
            # A ban with an exemption only ever fires on arcs whose LABEL is not exempt, so the
            # headroom is the same arithmetic over that subset.
            kill = sum(v for k, v in st["pred_out_right_rel"].items()
                       if k.split("||")[0] not in exempt)
            fix = st["pred_out"] - sum(v for k, v in st["pred_out_rel"].items()
                                       if k.split("||")[0] in exempt) - kill
            print(f"  headroom, {label}: destroys at most {kill} correct arcs "
                  f"({-100*kill/tot:+.3f} UAS), could fix at most {fix} "
                  f"({100*fix/tot:+.3f} UAS) — net ceiling {100*(fix-kill)/tot:+.3f} UAS")


if __name__ == "__main__":
    main()
