#!/usr/bin/env python3
"""`clause_parser` pipeline component for the Classical Chinese / Sanskrit models.

Those treebanks segment text into short punctuation-free **clause units** (句讀 for Classical
Chinese, syntactic clauses for Vedic) and carry no in-text sentence boundaries — so a parser run
over running text can't find the unit boundaries and collapses. Any real edition, though, marks
those boundaries with punctuation (。，；for Classical Chinese, daṇḍa ।॥ for Sanskrit). This
component recovers them: it splits the doc at punctuation tokens, **parses each clause in
isolation** (so each goes to the parser exactly as the punctuation-free units it was trained on),
and reattaches each punctuation mark as a `punct` dependent of the root of the clause on its left.

Recovers the per-clause (gold-preproc) accuracy on punctuated input instead of the running-text
collapse. Added as the last pipe: the normal tok2vec/tagger/parser still run once over the whole
doc (harmless), then this re-parses per clause and rebuilds the doc with the corrected analysis.
"""
from spacy.language import Language
from spacy.tokens import Doc

# clause-boundary punctuation across the relevant scripts; each model overrides via its pipe
# config (Classical Chinese 句讀 vs Sanskrit daṇḍa . ? ! | || / //).
DEFAULT_PUNCT = "。．，、；：？！…।॥|/.?!"


def make_clause_parser(nlp, name, punct):
    return ClauseParser(nlp, punct)


# Guard registration: both the lzh and sa wheels bundle this module, so it is imported twice
# when both models are loaded in one process — register the factory only once.
if not Language.has_factory("clause_parser"):
    Language.factory("clause_parser", default_config={"punct": DEFAULT_PUNCT})(make_clause_parser)


class ClauseParser:
    def __init__(self, nlp, punct):
        self.nlp = nlp
        self.punct = set(punct)
        self._pipes = None

    def _subpipes(self):
        if self._pipes is None:
            self._pipes = [self.nlp.get_pipe(n) for n in ("tok2vec", "tagger", "parser")
                           if self.nlp.has_pipe(n)]
        return self._pipes

    def _is_punct(self, tok):
        return tok.text in self.punct or all(c in self.punct for c in tok.text)

    def __call__(self, doc):
        # order: a sequence of ("c", [token indices]) clause runs and ("p", index) punctuation
        order, cur = [], []
        for t in doc:
            if self._is_punct(t):
                if cur:
                    order.append(("c", cur)); cur = []
                order.append(("p", t.i))
            else:
                cur.append(t.i)
        if cur:
            order.append(("c", cur))

        n = len(doc)
        heads = list(range(n))          # default: self-head
        deps = ["dep"] * n
        tags = [doc[i].tag_ for i in range(n)]
        poss = [doc[i].pos_ for i in range(n)]

        pipes = self._subpipes()
        last_root = None
        for kind, payload in order:
            if kind == "c":
                idxs = payload
                sub = Doc(self.nlp.vocab, words=[doc[i].text for i in idxs],
                          spaces=[bool(doc[i].whitespace_) for i in idxs])
                for p in pipes:
                    sub = p(sub)
                for j, i in enumerate(idxs):
                    hj = sub[j].head.i
                    heads[i] = idxs[hj]
                    deps[i] = sub[j].dep_ or "dep"
                    tags[i] = sub[j].tag_ or tags[i]
                    poss[i] = sub[j].pos_ or poss[i]
                    if hj == j:
                        last_root = i
            else:                        # punctuation -> punct of the clause on its left
                pi = payload
                heads[pi] = last_root if last_root is not None else pi
                deps[pi] = "punct"

        out = Doc(self.nlp.vocab, words=[t.text for t in doc],
                  spaces=[bool(t.whitespace_) for t in doc],
                  heads=heads, deps=deps, tags=tags, pos=poss)
        return out
