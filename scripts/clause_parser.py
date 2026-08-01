#!/usr/bin/env python3
"""`clause_parser` pipeline component for the Classical Chinese / Sanskrit models.

Those treebanks segment text into short punctuation-free **clause units** (句讀 for Classical
Chinese, syntactic clauses for Vedic) and carry no in-text sentence boundaries — so a parser run
over running text can't find the unit boundaries and collapses. Any real edition, though, marks
those boundaries with punctuation (。，；for Classical Chinese, daṇḍa ।॥ for Sanskrit). This
component recovers them.

Each **sentence** is the span between two sentence-final marks. Which marks those are is set by
`sent_scheme`: the default uses the fixed `sent_punct` set (Classical Chinese leaves it empty, so
*every* mark is sentence-final and a sentence is one 句讀 unit); `sent_scheme="danda"` (Sanskrit)
chooses the set **per document** — ? and ! (optionally + a trailing CLOSING quotation mark, a space
before it allowed) always end a sentence, and then a period / a double daṇḍa / a single daṇḍa is the
other sentence-final mark depending on which of those the text contains (see `_sentencer`). Within a
sentence the content
tokens are concatenated **with the
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
from spacy.tokens import Doc, Token

# clause-boundary punctuation across the relevant scripts; each model overrides via its pipe
# config (Classical Chinese 句讀 vs Sanskrit daṇḍa . ? ! | || / //).
DEFAULT_PUNCT = "。．，、；：？！…।॥|/.?!‖"

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


# --- the "danda" sentence scheme (Sanskrit): the set of sentence-final marks is chosen PER DOCUMENT.
#   1. ? and ! (optionally + a trailing quotation mark) always end a sentence;
#   2. else if the text has periods (not decimal points), a period is the only other sentence-final
#      mark (daṇḍas become medial);
#   3. else if the text has any double daṇḍa, a double daṇḍa is sentence-final (a single is medial);
#   4. else a single daṇḍa is sentence-final.
# A daṇḍa may be | || / // ‖ or a daṇḍa character in any Indic script (।॥ …); the sa tokenizer already
# normalises the Indic ones to |/|| and groups runs, but `_danda_kind` handles the raw chars too.
_QEXCL = set("?？!！")                                   # question / exclamation (+ fullwidth)
# CLOSING quotation marks (straight — ambiguous but act as closers here — + curly-close + angular-close).
# Only these may trail a sentence-final mark: an OPENING quote (« ‹ “ ‘) after a final mark begins the
# NEXT (quoted) sentence, so it must not be pulled back onto the sentence just closed.
_CLOSE_QUOTES = set("\"'”’»›")


def _char_danda(c):
    """Stroke count of a single daṇḍa character: 1 (single), 2 (double), 0 (not a daṇḍa)."""
    if c in "|/":
        return 1
    if c == "‖":                                   # ‖ DOUBLE VERTICAL LINE
        return 2
    try:
        name = unicodedata.name(c)
    except ValueError:
        return 0
    if name.endswith("DANDA"):                          # any Indic-script daṇḍa (।॥ and script variants)
        return 2 if "DOUBLE DANDA" in name else 1
    return 0


def _danda_kind(text):
    """'single' / 'double' / None for a token WHOLLY composed of daṇḍa marks (| || / // ‖ ।॥ …).
    A run of single strokes counts as double (`||`, `//`, or two single daṇḍa chars)."""
    if not text:
        return None
    strokes = 0
    for c in text:
        v = _char_danda(c)
        if not v:
            return None
        strokes += v
    return "double" if strokes >= 2 else "single"


def _is_qexcl(text):
    return bool(text) and all(c in _QEXCL for c in text)


def _is_close_quote(text):
    return bool(text) and all(c in _CLOSE_QUOTES for c in text)


def _is_period(doc, i):
    """A period token (all '.') that is NOT a decimal point (a '.' flanked by digit tokens)."""
    txt = doc[i].text
    if not txt or any(c != "." for c in txt):
        return False
    prev_d = i > 0 and doc[i - 1].text[-1:].isdigit()
    next_d = i + 1 < len(doc) and doc[i + 1].text[:1].isdigit()
    return not (prev_d and next_d)


def _force_compound(morph, flag):
    """Set/clear Compound=Yes in a FEATS string, leaving every other feature alone."""
    feats = [f for f in morph.split("|") if f and not f.startswith("Compound=")]
    if flag:
        feats.append("Compound=Yes")
    return "|".join(sorted(feats))


def make_clause_parser(nlp, name, punct, punct_tag, sent_punct, sent_scheme):
    return ClauseParser(nlp, punct, punct_tag, sent_punct, sent_scheme)


# Guard registration: both the lzh and sa wheels bundle this module, so it is imported twice
# when both models are loaded in one process — register the factory only once.
# `punct_tag`: a flat XPOS for every punctuation token; "" (default) uses the Kyoto 記号 subtype
# map above (correct for lzh, whose gold tags punctuation `s,記号,…`). Sanskrit sets it to a
# neutral "PUNCT" so the daṇḍa is not stamped with Japanese-tagset notation.
if not Language.has_factory("clause_parser"):
    Language.factory("clause_parser",
                     default_config={"punct": DEFAULT_PUNCT, "punct_tag": "",
                                     "sent_punct": SENT_PUNCT_DEFAULT, "sent_scheme": ""})(make_clause_parser)


class ClauseParser:
    def __init__(self, nlp, punct, punct_tag="", sent_punct=SENT_PUNCT_DEFAULT, sent_scheme=""):
        self.nlp = nlp
        self.punct = set(punct)
        self.punct_tag = punct_tag
        self.sent_punct = set(sent_punct)
        # "" -> the fixed `sent_punct` set decides boundaries (lzh: empty set = every mark is a
        # boundary). "danda" -> the document-dependent Sanskrit scheme (see the module comment above).
        self.sent_scheme = sent_scheme
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

    def _sentencer(self, doc):
        """Return (is_sent_final, allow_trailing_quote) for THIS doc. The `danda` scheme chooses the
        sentence-final mark set per document (see the module comment): ?/! always, then a period /
        double daṇḍa / single daṇḍa depending on what the text contains. Other schemes fall back to
        the fixed `sent_punct` set (with no trailing-quote handling)."""
        if self.sent_scheme != "danda":
            return self._is_sent_boundary, False
        has_period = any(_is_period(doc, t.i) for t in doc)
        has_double = any(_danda_kind(t.text) == "double" for t in doc)
        if has_period:
            other = lambda t: _is_period(doc, t.i)                       # rule 2: periods only
        elif has_double:
            other = lambda t: _danda_kind(t.text) == "double"           # rule 3: double daṇḍa only
        else:
            other = lambda t: _danda_kind(t.text) is not None           # rule 4: single daṇḍa
        return (lambda t: _is_qexcl(t.text) or other(t)), True           # rule 1: ?/! always final

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
        sent_final, allow_trailing_quote = self._sentencer(doc)
        sentences = []                  # each: list of ("content"|"medial", token index)
        boundary_puncts = []            # (punct index, index into `sentences` on its left, or None)
        cur = []
        after_final = False             # last emitted mark was sentence-final (for trailing quotes)
        left_of_final = None            # the sentence a trailing quote should attach to
        for t in doc:
            if self._is_punct(t):
                if sent_final(t):
                    if cur:
                        sentences.append(cur); cur = []
                    left = len(sentences) - 1 if sentences else None
                    boundary_puncts.append((t.i, left))
                    after_final, left_of_final = True, left
                elif allow_trailing_quote and after_final and _is_close_quote(t.text):
                    # a CLOSING quotation mark stays with the sentence the sentence-final mark just
                    # closed (…vākyam ॥ » -> the » ends the same sentence; a space before it is fine,
                    # since whitespace is not a token). An opening quote is NOT pulled back — it starts
                    # the next quoted sentence and falls through to the medial/content path.
                    boundary_puncts.append((t.i, left_of_final))    # keep after_final for ?"» chains
                else:
                    cur.append(("medial", t.i)); after_final = False
            else:
                cur.append(("content", t.i)); after_final = False
        if cur:
            sentences.append(cur)

        n = len(doc)
        heads = list(range(n))          # default: self-head
        deps = ["dep"] * n
        tags = [doc[i].tag_ for i in range(n)]
        poss = [doc[i].pos_ for i in range(n)]
        # lemma/morph come from the whole-doc pass (the morphologizer/lemmatizer run before this
        # component); the per-clause re-parse only re-decides tag/pos/head/dep, so carry these
        # through unchanged rather than dropping them when the doc is rebuilt below.
        lemmas = [doc[i].lemma_ for i in range(n)]
        morphs = [str(doc[i].morph) for i in range(n)]
        # The Sanskrit tokeniser decides Compound=Yes deterministically from the CSL join marker
        # (precision/recall 0.9998 against the treebank, vs the morphologizer's predicted F 0.889),
        # so its verdict is authoritative in BOTH directions and has to be reimposed here: the
        # morphologizer has already overwritten token.morph with its own prediction, and the
        # re-parse below rebuilds the doc from scratch. Guarded on the extension, because the lzh
        # wheel bundles this module without `sa_tokenizer`.
        cflags = (doc._.compound_flags
                  if Doc.has_extension("compound_flags") else None)
        if cflags is not None and len(cflags) == n:
            morphs = [_force_compound(morphs[i], cflags[i]) for i in range(n)]
        else:
            cflags = None
        pipes = self._subpipes()
        sent_roots = []                 # doc-index root of each sentence (aligned with `sentences`)

        for items in sentences:
            content = [i for kind, i in items if kind == "content"]
            if not content:
                sent_roots.append(None)
                continue
            # Parse the sentence's content as ONE doc (medial marks removed), so the parser itself
            # decides how the comma-separated units relate — no fabricated join.
            # Compound=Yes is an input feature of the parser's embed (configs/config_sa.cfg reads
            # MORPH), and the whole-doc pass got it from the tokeniser — so the sub-doc must carry
            # it too, or the re-parse runs on an input the parser never saw in training. Only the
            # Compound feat is carried: that is all the tokeniser supplies, and it is all the
            # corpus reader stamps on the predicted doc at training time.
            sub = Doc(self.nlp.vocab, words=[doc[i].text for i in content],
                      spaces=[bool(doc[i].whitespace_) for i in content],
                      morphs=(None if cflags is None else
                              ["Compound=Yes" if cflags[i] else "" for i in content]))
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
                  heads=heads, deps=deps, tags=tags, pos=poss,
                  lemmas=lemmas, morphs=morphs)
        # Rebuilding the doc drops its extension data, so carry the Sanskrit tokeniser's source
        # offsets (`sa_tokenizer`: doc._.src_text / src_spans, the raw-input character span of each
        # token) across the copy — the rebuild is token-for-token, so the spans stay aligned.
        # Guarded on the extension existing at all, since the lzh wheel bundles this module without
        # `sa_tokenizer`, and on the value being set, so nothing is invented for other callers.
        for attr in ("src_text", "src_spans", "compound_flags"):
            if Doc.has_extension(attr) and getattr(doc._, attr) is not None:
                setattr(out._, attr, getattr(doc._, attr))
        # Same for the TOKEN-level extensions. `_.unsandhied` (the padapāṭha form, set by the
        # tokeniser's stage B) was silently lost here until the full front end was assembled and
        # every token came out blank — the standing rule for anything that rebuilds a Doc is that it
        # owns carrying EVERY annotation, not the ones it happens to remember.
        for attr in ("unsandhied", "translit", "ltranslit"):
            if Token.has_extension(attr):
                for old, new in zip(doc, out):
                    if getattr(old._, attr):
                        setattr(new._, attr, getattr(old._, attr))
        return out
