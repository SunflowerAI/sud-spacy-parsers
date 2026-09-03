#!/usr/bin/env python3
"""Which CROSS-CLAUSAL attachments does the lzh parser get wrong — and can the rules fix them?

A cross-clausal arc is one whose dependent and head sit in DIFFERENT 句讀 units of the same gold
sentence, i.e. it crosses a comma or a full stop. Those are exactly the arcs `sent_join`'s clause
rule is written to reproduce, so they are the population to look at before adjusting it.

Two disjoint sub-populations, and they need different fixes:

  ATTACHED   the parser produced an arc across the boundary. `sent_join` never fires here — it only
             joins tokens the parser left as separate ROOTS — so anything wrong is the parser's,
             and no rule change can reach it.
  SPLIT      the parser left the second unit's head as its own ROOT. `sent_join` fires, and because
             gold DOES have an arc here we can score the rule's answer directly: head first, then
             label.

⚠ The second population is the only place the treebank can grade the rule at all. Everywhere else
`sent_join` joins across a GOLD SENTENCE boundary, where gold has no arc by construction and the
convention is a stipulation rather than a prediction (docs/chinese-family.md).
"""
import argparse
import collections
import importlib.util
import pathlib
import sys

PAUSE = set("，、；：,;:")
FINAL = set("。！？!?.")


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def units(doc):
    """Index -> unit number, splitting at any pause or sentence-final mark."""
    u, out = 0, {}
    for t in doc:
        if t.text in PAUSE or t.text in FINAL:
            u += 1
            continue
        out[t.i] = u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build_sud/lzh/lzh_sud_kyoto-0.2.0/lzh_sud_kyoto/"
                                       "lzh_sud_kyoto-0.2.0")
    ap.add_argument("--corpus", default="corpus_lzh_trad/lzh_kyoto-sud-test."
                                        "relabeled_ext.udep_ruled.punct.rulemerged.spacy")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy
    from spacy.tokens import Doc, DocBin
    sys.path.insert(0, "scripts")
    import sent_join  # noqa: F401

    nlp = spacy.load(a.model)
    if "sent_join" in nlp.pipe_names:
        nlp.remove_pipe("sent_join")
    pipe = sent_join.SentJoin()
    docs = list(DocBin().from_disk(a.corpus).get_docs(nlp.vocab))
    gold = [s.as_doc() for d in docs for s in d.sents]

    n_cross = 0
    att = collections.Counter()          # parser attached across the boundary
    split_head_ok = split_lab_ok = n_split = 0
    rule_conf = collections.Counter()
    att_conf = collections.Counter()
    branch_of = collections.Counter()
    for g in gold:
        gu = units(g)
        cross = [(t.i, t.head.i, t.dep_) for t in g
                 if t.head.i != t.i and t.dep_ != "punct"
                 and gu.get(t.i) is not None and gu.get(t.head.i) is not None
                 and gu[t.i] != gu[t.head.i]]
        if not cross:
            continue
        n_cross += len(cross)
        p = Doc(nlp.vocab, words=[t.text for t in g], spaces=[bool(t.whitespace_) for t in g])
        for name, proc in nlp.pipeline:
            p = proc(p)
        before = {t.i: (t.head.i, t.dep_) for t in p}
        pipe.debug = []
        p2 = pipe(p)
        fired = {r["dep_i"]: r for r in pipe.debug}
        for i, gh, gd in cross:
            if before[i][0] == i:                      # parser left it a ROOT -> the rule fires
                n_split += 1
                ph, pd = p2[i].head.i, p2[i].dep_
                if i in fired:
                    branch_of[fired[i]["branch"]] += 1
                if ph == gh:
                    split_head_ok += 1
                    if pd == gd:
                        split_lab_ok += 1
                    else:
                        rule_conf[(gd, pd)] += 1
                else:
                    rule_conf[(gd, f"(wrong head) {pd}")] += 1
            else:
                ok_h = before[i][0] == gh
                ok_l = ok_h and before[i][1] == gd
                att["right head, right label" if ok_l else
                    ("right head, wrong label" if ok_h else "wrong head")] += 1
                if ok_h and not ok_l:
                    att_conf[(gd, before[i][1])] += 1
    print(f"cross-clausal gold arcs in the test set: {n_cross}\n")
    tot_att = sum(att.values())
    print(f"A. the parser ATTACHED across the boundary   {tot_att:5d}  ({tot_att/n_cross:.1%})")
    for k, v in att.most_common():
        print(f"     {k:<26}{v:5d}  {v/max(tot_att,1):6.1%}")
    print("     top label confusions here (gold -> parser):")
    for (gd, pd), c in att_conf.most_common(6):
        print(f"        {gd:<16} -> {pd:<16}{c:4d}")
    print(f"\nB. the parser SPLIT, so sent_join fires      {n_split:5d}  ({n_split/n_cross:.1%})")
    if n_split:
        print(f"     rule picked the RIGHT HEAD  {split_head_ok:5d}  {split_head_ok/n_split:6.1%}")
        print(f"     ...and the right LABEL      {split_lab_ok:5d}  {split_lab_ok/n_split:6.1%}")
        print("     branches used:", ' '.join(f'{k}:{v}' for k, v in branch_of.most_common()))
        print("     what the rule got wrong (gold -> rule):")
        for (gd, pd), c in rule_conf.most_common(10):
            print(f"        {gd:<16} -> {pd:<24}{c:4d}")


if __name__ == "__main__":
    main()
