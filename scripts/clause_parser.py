#!/usr/bin/env python3
"""`clause_parser` pipeline component for the Classical Chinese / Sanskrit models.

Those treebanks segment text into short punctuation-free **clause units** (句讀 for Classical
Chinese, syntactic clauses for Vedic) and carry no in-text sentence boundaries — so a parser run
over running text can't find the unit boundaries and collapses. Any real edition, though, marks
those boundaries with punctuation (。，；for Classical Chinese, daṇḍa ।॥ for Sanskrit). This
component recovers them.

Each **sentence** is the span between two sentence-final marks (`sent_punct`: the daṇḍa ।॥ and
.?! for Sanskrit; for Classical Chinese the empty default makes *every* mark sentence-final, so a
sentence is one 句讀 unit). Within a sentence the content tokens are concatenated **with the
sentence-medial marks removed** (a stray comma derails the parser as much as a daṇḍa) and parsed
as a single doc — so the parser itself decides how the comma-separated units relate, rather than
the component fabricating a join. Then every punctuation mark is reinserted: a **medial** mark
(a comma) as a `punct` child of the head of the unit on its **left**; a **sentence-final** mark as
a `punct` child of the root of the sentence on its left. A comma therefore stays inside its
sentence; only a daṇḍa/full stop ends one.

Added as the last pipe: the normal tok2vec/tagger/parser still run once over the whole doc
(harmless), then this re-parses per sentence and rebuilds the doc with the corrected analysis.
"""
import unicodedata

from spacy.language import Language
from spacy.tokens import Doc

# clause-boundary punctuation across the relevant scripts; each model overrides via its pipe
# config (Classical Chinese 句讀 vs Sanskrit daṇḍa . ? ! | || / //).
DEFAULT_PUNCT = "。．，、；：？！…।॥|/.?!"

# The subset of `punct` that ends a *sentence* (as opposed to a sentence-medial pause). Every
# punctuation mark is still pulled out before parsing — a stray comma derails the parser just like
# a daṇḍa — but only a sentence-final mark ends a `doc.sents` sentence; units separated by a medial
# mark (a comma, a bracket) are parsed together (concatenated, comma removed) so they stay in one
# sentence. The empty default makes *every* mark sentence-final (correct for Classical Chinese 句讀
# units, each of which the Kyoto treebank annotates as its own sentence). Sanskrit sets it to the
# daṇḍa-class marks so a comma is medial — `.?!` and the daṇḍa ।॥ (transliterated to | / ||) end a
# sentence, but , ; : « » do not.
SENT_PUNCT_DEFAULT = ""

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


def make_clause_parser(nlp, name, punct, punct_tag, sent_punct):
    return ClauseParser(nlp, punct, punct_tag, sent_punct)


# Guard registration: both the lzh and sa wheels bundle this module, so it is imported twice
# when both models are loaded in one process — register the factory only once.
# `punct_tag`: a flat XPOS for every punctuation token; "" (default) uses the Kyoto 記号 subtype
# map above (correct for lzh, whose gold tags punctuation `s,記号,…`). Sanskrit sets it to a
# neutral "PUNCT" so the daṇḍa is not stamped with Japanese-tagset notation.
if not Language.has_factory("clause_parser"):
    Language.factory("clause_parser",
                     default_config={"punct": DEFAULT_PUNCT, "punct_tag": "",
                                     "sent_punct": SENT_PUNCT_DEFAULT})(make_clause_parser)


class ClauseParser:
    def __init__(self, nlp, punct, punct_tag="", sent_punct=SENT_PUNCT_DEFAULT):
        self.nlp = nlp
        self.punct = set(punct)
        self.punct_tag = punct_tag
        self.sent_punct = set(sent_punct)
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

    def _is_sent_boundary(self, tok):
        """Does this punctuation mark end a sentence? With an empty `sent_punct` every mark is a
        boundary (the original behaviour — each 句讀 unit is its own sentence). When `sent_punct`
        is set (Sanskrit), only those marks end a sentence; a medial mark (a comma) is still pulled
        out before parsing but keeps its neighbouring units in one sentence."""
        if not self.sent_punct:
            return True
        return tok.text in self.sent_punct or all(c in self.sent_punct for c in tok.text)

    @staticmethod
    def _unit_head(unit, heads):
        """The dependency head of a contiguous unit (the run of content tokens to the left of a
        comma): the unit token whose own head lies outside the unit — its own root, or the token
        that links the unit into the rest of the sentence tree. Leftmost such on a tie."""
        unit_set = set(unit)
        for i in unit:
            if heads[i] == i or heads[i] not in unit_set:
                return i
        return unit[0]

    def __call__(self, doc):
        # Partition the doc into sentences (spans between sentence-final marks). Each sentence keeps
        # its tokens in order as ("content", idx) or ("medial", idx) (a sentence-medial mark, e.g. a
        # comma); a sentence-final mark closes the sentence and is recorded against the sentence on
        # its left. With sent_punct empty every mark is sentence-final, so a sentence is one unit.
        sentences = []                  # each: list of ("content"|"medial", token index)
        boundary_puncts = []            # (punct index, index into `sentences` on its left, or None)
        cur = []
        for t in doc:
            if self._is_punct(t):
                if self._is_sent_boundary(t):
                    if cur:
                        sentences.append(cur); cur = []
                    left = len(sentences) - 1 if sentences else None
                    boundary_puncts.append((t.i, left))
                else:
                    cur.append(("medial", t.i))
            else:
                cur.append(("content", t.i))
        if cur:
            sentences.append(cur)

        n = len(doc)
        heads = list(range(n))          # default: self-head
        deps = ["dep"] * n
        tags = [doc[i].tag_ for i in range(n)]
        poss = [doc[i].pos_ for i in range(n)]
        pipes = self._subpipes()
        sent_roots = []                 # doc-index root of each sentence (aligned with `sentences`)

        for items in sentences:
            content = [i for kind, i in items if kind == "content"]
            if not content:
                sent_roots.append(None)
                continue
            # Parse the sentence's content as ONE doc (medial marks removed), so the parser itself
            # decides how the comma-separated units relate — no fabricated join.
            sub = Doc(self.nlp.vocab, words=[doc[i].text for i in content],
                      spaces=[bool(doc[i].whitespace_) for i in content])
            for p in pipes:
                sub = p(sub)
            root = None
            for j, i in enumerate(content):
                hj = sub[j].head.i
                heads[i] = content[hj]
                deps[i] = sub[j].dep_ or "dep"
                tags[i] = sub[j].tag_ or tags[i]
                poss[i] = sub[j].pos_ or poss[i]
                if hj == j and root is None:
                    root = i
            sent_roots.append(root)
            # Reinsert each medial mark as a `punct` child of the head of the unit on its left.
            left_unit = []
            for kind, i in items:
                if kind == "content":
                    left_unit.append(i)
                    continue
                tags[i] = self.punct_tag or punct_tag(doc[i].text)
                poss[i] = "PUNCT"
                anchor = self._unit_head(left_unit, heads) if left_unit else root
                if anchor is not None:
                    heads[i] = anchor
                    deps[i] = "punct"
                left_unit = []          # the next unit starts after this mark

        # A sentence-final mark attaches as `punct` to the root of the sentence on its left (its own
        # sentence); it is forced onto the punctuation tagset — a mark must never carry a content
        # category (the near-punctuation-free treebank leaves the tagger hallucinating e.g.
        # ？ -> 名詞,糧食 "noun, food").
        for pi, left in boundary_puncts:
            tags[pi] = self.punct_tag or punct_tag(doc[pi].text)
            poss[pi] = "PUNCT"
            anchor = sent_roots[left] if left is not None else None
            if anchor is not None:
                heads[pi] = anchor
                deps[pi] = "punct"

        out = Doc(self.nlp.vocab, words=[t.text for t in doc],
                  spaces=[bool(t.whitespace_) for t in doc],
                  heads=heads, deps=deps, tags=tags, pos=poss)
        return out
