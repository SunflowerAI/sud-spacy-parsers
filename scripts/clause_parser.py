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
import unicodedata

from spacy.language import Language
from spacy.tokens import Doc

# clause-boundary punctuation across the relevant scripts; each model overrides via its pipe
# config (Classical Chinese 句讀 vs Sanskrit daṇḍa . ? ! | || / //).
DEFAULT_PUNCT = "。．，、；：？！…।॥|/.?!"

# Canonical Kyoto punctuation tags (the treebank carries almost no punctuation, so the tagger
# never learned it and hallucinates content categories — e.g. ？ tagged 名詞,糧食 "noun, food",
# 。 tagged 動詞 "verb"). We force every punctuation token onto the 記号 ("symbol") tagset
# deterministically instead. Subclasses follow Kyoto: 句点 sentence-final, 読点 pause,
# 括弧開/括弧閉 open/close bracket.
_OPEN_BRACKETS = set("（「『【〔《〈［｛(<[{“‘")
_CLOSE_BRACKETS = set("）」』】〕》〉］｝)>]}”’")
_SENT_FINAL = set("。．！？!?।॥…")
_PAUSE = set("，、；：,;:/|")
PUNCT_TAG_DEFAULT = "s,記号,*,*"


def punct_tag(text):
    """Canonical 記号 XPOS for a punctuation token (never a content category)."""
    chars = set(text)
    if chars & _OPEN_BRACKETS:
        return "s,記号,括弧開,*"
    if chars & _CLOSE_BRACKETS:
        return "s,記号,括弧閉,*"
    if chars & _SENT_FINAL:
        return "s,記号,句点,*"
    if chars & _PAUSE:
        return "s,記号,読点,*"
    return PUNCT_TAG_DEFAULT


def is_punct_text(text):
    """True if the token is wholly punctuation (Unicode category P*) — catches brackets and
    other marks that are not clause boundaries but must still be tagged as punctuation."""
    return bool(text) and all(unicodedata.category(c).startswith("P") for c in text)


def make_clause_parser(nlp, name, punct, punct_tag):
    return ClauseParser(nlp, punct, punct_tag)


# Guard registration: both the lzh and sa wheels bundle this module, so it is imported twice
# when both models are loaded in one process — register the factory only once.
# `punct_tag`: a flat XPOS for every punctuation token; "" (default) uses the Kyoto 記号 subtype
# map above (correct for lzh, whose gold tags punctuation `s,記号,…`). Sanskrit sets it to a
# neutral "PUNCT" so the daṇḍa is not stamped with Japanese-tagset notation.
if not Language.has_factory("clause_parser"):
    Language.factory("clause_parser",
                     default_config={"punct": DEFAULT_PUNCT, "punct_tag": ""})(make_clause_parser)


class ClauseParser:
    def __init__(self, nlp, punct, punct_tag=""):
        self.nlp = nlp
        self.punct = set(punct)
        self.punct_tag = punct_tag
        self._pipes = None

    def _subpipes(self):
        if self._pipes is None:
            self._pipes = [self.nlp.get_pipe(n) for n in ("tok2vec", "tagger", "parser")
                           if self.nlp.has_pipe(n)]
        return self._pipes

    def _is_punct(self, tok):
        # any punctuation — the clause-boundary set (句讀 / daṇḍa) *or* any Unicode punctuation
        # mark (quotation brackets etc.). All of it is pulled out of the parsed clauses: it is
        # never content, and a bracket left inside a clause derails the parser (it gets tagged as
        # a noun/verb and can even become the clause ROOT).
        return (tok.text in self.punct or all(c in self.punct for c in tok.text)
                or is_punct_text(tok.text))

    @staticmethod
    def _nearest_root(i, roots):
        """Closest clause root to token i (preferring the clause on its left on a tie), so a
        punctuation mark attaches as a `punct` of a real clause rather than staying ROOT."""
        if not roots:
            return None
        return min(roots, key=lambda r: (abs(r - i), r > i))

    def __call__(self, doc):
        # order: a sequence of ("c", [token indices]) clause runs and ("p", index) punctuation.
        # Every punctuation token is a separator, so the parsed clauses contain content only.
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
        puncts = []                     # every punctuation token index, in document order
        clause_roots = []               # clause-root token indices, in document order
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
                        clause_roots.append(i)
            else:
                puncts.append(payload)

        # Attach and tag every punctuation token deterministically: it becomes a `punct` of the
        # nearest clause root and is forced onto the 記号 tagset — a punctuation mark must never
        # carry a content/semantic category (the near-punctuation-free treebank leaves the tagger
        # hallucinating e.g. ？ -> 名詞,糧食 "noun, food").
        for pi in puncts:
            tags[pi] = self.punct_tag or punct_tag(doc[pi].text)
            poss[pi] = "PUNCT"
            anchor = self._nearest_root(pi, clause_roots)
            if anchor is not None:
                heads[pi] = anchor
                deps[pi] = "punct"

        out = Doc(self.nlp.vocab, words=[t.text for t in doc],
                  spaces=[bool(t.whitespace_) for t in doc],
                  heads=heads, deps=deps, tags=tags, pos=poss)
        return out
