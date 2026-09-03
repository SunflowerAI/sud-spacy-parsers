#!/usr/bin/env python
"""Corpus for a standalone FEATS head: TAG := the FEATS string, so spaCy's tagger can learn it.

WHY A SEPARATE HEAD. The morphologiser predicts POS and FEATS as ONE joint label (152 of them,
`POS=VERB|VerbForm=Part`), so its FEATS is a deterministic function of its UPOS -- the same softmax,
the same errors. It cannot serve as a second opinion. Trained independently, FEATS becomes a third
vote for the UPOS stacking selector, and the ceiling is high: on the 3 591 disagreements between the
morphologiser and the tagger, GOLD FEATS opines on 59.9 % and is 98.3 % right within that coverage,
picking the correct side of 55.8 % of all disagreements against the morphologiser's own 56.5 %.

⚠ That ceiling uses GOLD FEATS, and FEATS->UPOS purity is 97.3 %, so knowing it is close to knowing
the answer. A trained head will make errors correlated with UPOS errors. The tagger's XPOS has the
same property (96.9 % purity) and still yields +0.49, which is the reason to try rather than assume.
"""
import sys, collections, pathlib
sys.path.insert(0, "scripts")
import seg_code  # noqa: F401
import spacy
from spacy.tokens import DocBin, Doc

NONE = "_NOFEAT_"
C = ("corpus_lzh_resplit_ctl/lzh_kyoto-sud-%s."
     "relabeled_ext.udep_ruled.punct.rulemerged.resplit.spacy")
OUT = pathlib.Path("corpus_lzh_feats"); OUT.mkdir(exist_ok=True)
nlp = spacy.load("training_lzh_depmorph_resplit/model-best")
for split in ("train", "dev", "test"):
    docs = list(DocBin().from_disk(C % split).get_docs(nlp.vocab))
    out = DocBin(); seen = collections.Counter()
    for g in docs:
        d = Doc(nlp.vocab, words=[t.text for t in g],
                spaces=[bool(t.whitespace_) for t in g],
                heads=[t.head.i for t in g], deps=[t.dep_ for t in g])
        for i, t in enumerate(g):
            lab = str(t.morph) or NONE
            d[i].tag_ = lab; seen[lab] += 1
        out.add(d)
    p = OUT / f"lzh_feats-{split}.spacy"
    out.to_disk(p)
    print(f"  {split}: {len(docs)} docs, {sum(seen.values())} tokens, {len(seen)} FEATS labels, "
          f"{seen[NONE]*100/sum(seen.values()):.1f}% have none")
