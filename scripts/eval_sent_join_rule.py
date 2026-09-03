#!/usr/bin/env python3
"""How much of the join rule's behaviour is the RULE, and how much is the tagger under it?

`sent_join`'s clause rule reads `token.pos_` (predicate test, SCONJ test, nominal test) and
`token.tag_` (the 伝達 speech-verb class). It runs LAST, so those are the arm's own predictions —
and on real text several branches were seen firing on tagging errors rather than on the
configurations they describe. This measures that directly, on the treebank test set where gold tags
exist:

  A. PREDICTED tags — the arm as it ships.
  B. GOLD tags stamped onto the SAME predicted tree, so the parse is held fixed and only the
     tagging varies.

Two readings come out. **Agreement** between A and B is the share of decisions the tagger does not
change — one minus the tagger's cost to the rule. **Accuracy** is scored only where gold actually
has an arc between the two heads the rule joined, which is the subset where the treebank has an
opinion at all: its blocks are single 句讀 units, so most joins the rule makes span two gold
sentences and gold has nothing to say about them.

Usage:
    eval_sent_join_rule.py [--corpus corpus_lzh_trad/...test....spacy]
"""
import argparse
import collections
import importlib.util
import pathlib
import sys


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the downloaded 0.2.0 wheel copy")
    ap.add_argument("--corpus", default="corpus_lzh_trad/lzh_kyoto-sud-test."
                                        "relabeled_ext.udep_ruled.punct.rulemerged.spacy")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc, DocBin
    sys.path.insert(0, "scripts")
    import sent_join  # noqa: F401

    model = a.model or ("build_sud/lzh/lzh_sud_kyoto-0.2.0/lzh_sud_kyoto/lzh_sud_kyoto-0.2.0")
    nlp = spacy.load(model)
    if "sent_join" not in nlp.pipe_names:
        nlp.add_pipe("sent_join", last=True)
    pipe = nlp.get_pipe("sent_join")
    gold_docs = list(DocBin().from_disk(a.corpus).get_docs(nlp.vocab))

    # gold arcs, as (dep index, head index) -> label, doc by doc
    def gold_arcs(g):
        return {(t.i, t.head.i): t.dep_ for t in g if t.head.i != t.i}

    runs = {}
    for label, use_gold in (("predicted tags", False), ("GOLD tags", True)):
        branches, decisions = collections.Counter(), []
        for g in gold_docs:
            d = Doc(nlp.vocab, words=[t.text for t in g],
                    spaces=[bool(t.whitespace_) for t in g])
            for name, proc in nlp.pipeline:
                if name == "sent_join":
                    break
                d = proc(d)
            if use_gold:
                # the SAME tree, only the tags swapped — so any difference is the tagger's
                for t, gt in zip(d, g):
                    t.pos_ = gt.pos_
                    t.tag_ = gt.tag_
            pipe.debug = []
            d = pipe(d)
            for r in pipe.debug:
                branches[r["branch"]] += 1
                decisions.append((r["dep_i"], r["head_i"], r["dep"], r["branch"]))
        pipe.debug = None
        runs[label] = (branches, decisions, )
        tot = sum(branches.values())
        print(f"\n{label}: {tot} decisions")
        for br, n in sorted(branches.items(), key=lambda kv: -kv[1]):
            print(f"   {br:<9}{n:6d}  {n/max(tot,1):6.1%}")

    (bp, dp), (bg, dg) = runs["predicted tags"], runs["GOLD tags"]
    # AGREEMENT: same (dependent, head, label) triple, keyed on the arc the rule chose
    mp = {(x, y): (dep, br) for x, y, dep, br in dp}
    mg = {(x, y): (dep, br) for x, y, dep, br in dg}
    keys = set(mp) | set(mg)
    same_arc = set(mp) & set(mg)
    same_lab = sum(1 for k in same_arc if mp[k][0] == mg[k][0])
    print(f"\nAGREEMENT between the two taggings, over {len(keys)} arcs either produced:")
    print(f"   same (dependent, head) pair : {len(same_arc):5d}  {len(same_arc)/len(keys):6.1%}")
    print(f"   ...and the same relation    : {same_lab:5d}  {same_lab/len(keys):6.1%}")
    print(f"   -> the tagger changes the rule's answer on "
          f"{100 - 100*same_lab/len(keys):.1f}% of its decisions")
    br_change = collections.Counter()
    for k in same_arc:
        if mp[k][1] != mg[k][1]:
            br_change[(mg[k][1], mp[k][1])] += 1
    if br_change:
        print("\n   branch taken under GOLD -> branch taken under PREDICTED (top shifts):")
        for (g_, p_), n in br_change.most_common(6):
            print(f"      {g_:<8} -> {p_:<8} {n:5d}")


if __name__ == "__main__":
    main()
