#!/usr/bin/env python3
"""`clause_parser` pipeline component for the Classical Chinese / Sanskrit models.

Those treebanks segment text into short punctuation-free **clause units** (句讀 for Classical
Chinese, syntactic clauses for Vedic) and carry no in-text sentence boundaries — so a parser run
over running text can't find the unit boundaries and collapses. Any real edition, though, marks
those boundaries with punctuation (。，；for Classical Chinese, daṇḍa ।॥ for Sanskrit). This
component recovers them.

Each **sentence** is the span between two sentence-final marks. Which marks those are is set by
`sent_scheme`: the default uses the fixed `sent_punct` set, which holds the genuinely
sentence-final marks only (。．.！？!?…, plus the daṇḍas), so a pause mark — a comma, a
semicolon, a bracket — leaves its 句讀 units inside ONE sentence; `sent_scheme="danda"` (Sanskrit)
chooses the set **per document** — ? and ! always end a sentence, and then a period / a double
daṇḍa / a single daṇḍa is the other sentence-final mark depending on which of those the text
contains (see `_sentencer`). Under either scheme a sentence-final mark may be followed by any run
of CLOSING quotation marks and brackets (」』）”» …), which stay with the sentence just closed;
an OPENING mark after it begins the next (quoted) sentence. Within a sentence the content
tokens are concatenated **with the
sentence-medial marks removed** (a stray comma derails the parser as much as a daṇḍa) and parsed
as a single doc — so the parser itself decides how the comma-separated units relate, rather than
the component fabricating a join. Then every punctuation mark is reinserted: a **medial** mark
(a comma) as a `punct` child of the head of the unit on its **left**; a **sentence-final** mark as
a `punct` child of the root of the sentence on its left. A comma therefore stays inside its
sentence; only a daṇḍa/full stop ends one.

A span between two sentence-final marks may still come out as MORE than one sentence, and that is
deliberate. `doc.sents` is derived from the tree, so every root the sub-parse produced opens a
sentence; all of them are kept rather than only the first. On input that carries no boundary cues —
unpunctuated 白文, the case the punctuated arm is weakest on — the parser's refusal to join two
clauses is a real signal about the text, and collapsing them under the leading root would discard
it silently. The sentence-final mark then attaches to the LAST root of the span it closes.

Added as the last pipe: the normal tok2vec/tagger/parser still run once over the whole doc
(harmless), then this re-parses per sentence and rebuilds the doc with the corrected analysis.
"""
import unicodedata

from spacy.language import Language
from spacy.tokens import Doc, Token

# clause-boundary punctuation across the relevant scripts; each model overrides via its pipe
# config (Classical Chinese 句讀 vs Sanskrit daṇḍa . ? ! | || / //).
DEFAULT_PUNCT = "。．，、；：？！…।॥|/.?!‖"

# Canonical Kyoto punctuation tags (the treebank carries almost no punctuation, so the tagger
# never learned it and hallucinates content categories — e.g. ？ tagged 名詞,糧食 "noun, food",
# 。 tagged 動詞 "verb"). We force every punctuation token onto the 記号 ("symbol") tagset
# deterministically instead. Subclasses follow Kyoto: 句点 sentence-final, 読点 pause,
# 括弧開/括弧閉 open/close bracket.
_OPEN_BRACKETS = set("（「『【〔《〈［｛(<[{“‘")
_CLOSE_BRACKETS = set("）」』】〕》〉］｝)>]}”’")
_SENT_FINAL = set("。．.！？!?।॥…")
_PAUSE = set("，、；：,;:/|")
PUNCT_TAG_DEFAULT = "s,記号,*,*"

# The subset of `punct` that ends a *sentence* (as opposed to a sentence-medial pause). Every
# punctuation mark is still pulled out before parsing — a stray comma derails the parser just like
# a daṇḍa — but only a sentence-final mark ends a `doc.sents` sentence; units separated by a medial
# mark (a comma, a bracket) are parsed together (concatenated, comma removed) so they stay in one
# sentence. The default is exactly the set that gets the 句点 tag above: a full stop, a question or
# exclamation mark, an ellipsis, a daṇḍa. A comma, a semicolon, a colon and a bracket are pauses,
# so 曰／，-separated 句讀 units stay in the sentence they belong to. Sanskrit overrides the whole
# question with `sent_scheme="danda"` (which ignores this set).
# `sent_punct=""` is kept as an escape hatch meaning *every* mark is sentence-final — the original
# Classical Chinese behaviour, one 句讀 unit per sentence, matching how the Kyoto treebank
# annotates them but NOT how a punctuated edition reads.
SENT_PUNCT_DEFAULT = "".join(sorted(_SENT_FINAL))


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
_CLOSE_QUOTES = set("\"'”’»›")
# What may TRAIL a sentence-final mark and still belong to the sentence it closed: any run of closing
# quotation marks and closing brackets (」』）】〕”» …), as in 「…也。」 or (…!). An OPENING mark
# (「 « ‹ “ ‘ （) after a final mark begins the NEXT (quoted) sentence, so it must not be pulled back.
# The straight " and ' are ambiguous and treated as closers, which is the common case after a stop.
_CLOSERS = _CLOSE_QUOTES | _CLOSE_BRACKETS


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


def _is_closer(text):
    """True for a token made wholly of CLOSING quotation marks / brackets (a run of them counts,
    and consecutive closer tokens chain, so 也。」）is one sentence)."""
    return bool(text) and all(c in _CLOSERS for c in text)


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


def make_clause_parser(nlp, name, punct, punct_tag, sent_punct, sent_scheme, keep_marks):
    return ClauseParser(nlp, punct, punct_tag, sent_punct, sent_scheme, keep_marks)


# Guard registration: both the lzh and sa wheels bundle this module, so it is imported twice
# when both models are loaded in one process — register the factory only once.
# `punct_tag`: a flat XPOS for every punctuation token; "" (default) uses the Kyoto 記号 subtype
# map above (correct for lzh, whose gold tags punctuation `s,記号,…`). Sanskrit sets it to a
# neutral "PUNCT" so the daṇḍa is not stamped with Japanese-tagset notation.
if not Language.has_factory("clause_parser"):
    Language.factory("clause_parser",
                     default_config={"punct": DEFAULT_PUNCT, "punct_tag": "", "keep_marks": False,
                                     "sent_punct": SENT_PUNCT_DEFAULT, "sent_scheme": ""})(make_clause_parser)


class ClauseParser:
    def __init__(self, nlp, punct, punct_tag="", sent_punct=SENT_PUNCT_DEFAULT, sent_scheme="",
                 keep_marks=False):
        self.nlp = nlp
        self.punct = set(punct)
        self.punct_tag = punct_tag
        self.sent_punct = set(sent_punct)
        # "" -> the fixed `sent_punct` set decides boundaries (lzh: empty set = every mark is a
        # boundary). "danda" -> the document-dependent Sanskrit scheme (see the module comment above).
        self.sent_scheme = sent_scheme
        # Strip sentence-medial marks before parsing (the default) or leave them in the sub-doc.
        # Stripping is right for a parser that has never SEEN a mark — Kyoto carries 5 punctuation
        # tokens in 374 560, so a bracket left in a clause gets tagged as a noun and can become its
        # ROOT. It is wrong for one trained on a punctuated treebank (see align_kanripo_punct.py),
        # where a comma is a boundary cue the parser was taught to use, and removing it deletes the
        # very signal that makes a multi-unit sentence parsable.
        self.keep_marks = keep_marks
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
        """Is this punctuation mark in the sentence-final set? Only those marks end a sentence; a
        medial mark (a comma, a bracket) is still pulled out before parsing but keeps its
        neighbouring units in one sentence. An empty `sent_punct` is the escape hatch meaning every
        mark is a boundary, and `_sentencer` short-circuits it before reaching here."""
        if not self.sent_punct:
            return True
        return tok.text in self.sent_punct or all(c in self.sent_punct for c in tok.text)

    def _sentencer(self, doc):
        """Return (is_sent_final, allow_trailing_closer) for THIS doc. The `danda` scheme chooses the
        sentence-final mark set per document (see the module comment): ?/! always, then a period /
        double daṇḍa / single daṇḍa depending on what the text contains. Otherwise the fixed
        `sent_punct` set decides — with the same decimal-point guard, so 3.14 is not two sentences."""
        if self.sent_scheme != "danda":
            if not self.sent_punct:                 # escape hatch: every mark ends a sentence
                return (lambda t: True), False

            def fixed(t):
                if not self._is_sent_boundary(t):
                    return False
                return _is_period(doc, t.i) if all(c == "." for c in t.text) else True

            return fixed, True
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
        # its left, as does any run of closing quotes/brackets trailing it.
        sent_final, allow_trailing_closer = self._sentencer(doc)
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
                elif allow_trailing_closer and after_final and _is_closer(t.text):
                    # a CLOSING quotation mark or bracket stays with the sentence the sentence-final
                    # mark just closed (…也。」 -> the 」 ends the same sentence; a space before it is
                    # fine, since whitespace is not a token). An OPENING mark is NOT pulled back — it
                    # starts the next quoted sentence and falls through to the medial/content path.
                    boundary_puncts.append((t.i, left_of_final))    # keep after_final: 。」）chains
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
            # `keep_marks` hands the medial marks to the parser instead of stripping them; it then
            # decides their attachment itself, and the reattachment rule below is skipped for them.
            content = [i for kind, i in items if self.keep_marks or kind == "content"]
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
                      spaces=[bool(doc[i].whitespace_) for i in content])
            # ⚠ Compound is stamped token-by-token, NOT via `morphs=[... or "" ...]`. Passing an
            # empty string sets the EMPTY morph (key 456) where an untouched token is UNSET
            # (key 0); both render as '' so no string comparison can see the difference, and the
            # encoder is handed a MORPH value it never met in training on ~94 % of tokens. This is
            # the documented 6.8-LAS bug (CLAUDE.md, sa_tokenizer.__call__) — it was reproduced
            # here, inside the re-parse, and cost 4.3 LAS on the Vedic test.
            if cflags is not None:
                for j, i in enumerate(content):
                    if cflags[i]:
                        sub[j].set_morph("Compound=Yes")
            # NORM likewise, and for exactly the same reason as the Compound feat above. For
            # Sanskrit `token.norm_` is the PADAPĀṬHA (`sa_tokenizer` stage 2), which is both an
            # embed channel and the key `sud.AnalyserFeatsEmbed.v1` looks its candidate sets up on.
            # Rebuilding without it silently reverts NORM to lower(ORTH) — the SANDHIED surface —
            # so the re-parse runs out of distribution and the analyser channel goes to its silent
            # bit. Measured on the Kathāsaritsāgara opening: it turned a correct root (`diśatu`)
            # into a wrong one (`śriyaṃ`). `Doc()` takes no `norms` argument, hence the loop.
            for j, i in enumerate(content):
                sub[j].norm_ = doc[i].norm_
            for p in pipes:
                sub = p(sub)
            # EVERY root the sub-parse produced is kept, not just the first. spaCy derives
            # `doc.sents` from the tree — a self-headed token opens a sentence — so a span the
            # parser analysed as several independent clauses comes out as several sentences, which
            # is the intended behaviour: on input carrying no boundary cues (unpunctuated 白文) the
            # parser declining to join two clauses is information, and silently gluing them under
            # the first root would throw it away.
            roots = []
            for j, i in enumerate(content):
                hj = sub[j].head.i
                heads[i] = content[hj]
                deps[i] = sub[j].dep_ or "dep"
                tags[i] = sub[j].tag_ or tags[i]
                poss[i] = sub[j].pos_ or poss[i]
                if hj == j:
                    roots.append(i)
            sent_roots.append(roots)
            root = roots[0] if roots else None
            # Reinsert each medial mark as a `punct` child of the head of the unit on its left.
            # Under `keep_marks` the parser has already attached them, so only the punctuation
            # morphology is imposed — a mark must never carry a content category either way.
            left_unit = []
            for kind, i in items:
                if kind == "content":
                    left_unit.append(i)
                    continue
                tags[i] = self.punct_tag or punct_tag(doc[i].text)
                poss[i] = "PUNCT"
                if not self.keep_marks:
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
            # the LAST root of the span on the left: if that span fragmented into several
            # sentences, the mark closes the final one, and hanging it off the first would pull a
            # sentence-final mark back into an earlier sentence.
            rs = sent_roots[left] if left is not None else None
            anchor = rs[-1] if rs else None
            if anchor is not None:
                heads[pi] = anchor
                deps[pi] = "punct"

        out = Doc(self.nlp.vocab, words=[t.text for t in doc],
                  spaces=[bool(t.whitespace_) for t in doc],
                  heads=heads, deps=deps, tags=tags, pos=poss,
                  lemmas=lemmas, morphs=morphs)
        # ...and on the way out too: a caller reading `token.norm_` off the returned doc must see
        # the padapāṭha, not lower(ORTH).
        for a, b in zip(out, doc):
            a.norm_ = b.norm_
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
