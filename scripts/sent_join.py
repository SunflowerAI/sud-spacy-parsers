#!/usr/bin/env python3
"""`sent_join`: refuse a sentence boundary where the reading convention says there is none.

WHY THIS EXISTS. The lzh arm has no `senter` and no `clause_parser` (see `package_sud.sh`):
sentence boundaries come from the PARSER, because `sud.GoldTokCorpus.v1` feeds it multi-sentence
docs and spaCy derives `doc.sents` from the tree — every self-headed token opens a sentence. It
therefore inherits the Kyoto treebank's segmentation, and Kyoto segments at 句讀 units, so reported
speech comes apart and clauses break at commas:

    子曰「學而時習之，不亦說乎。有朋自遠方來，不亦樂乎。」
      ->  子曰「學而時習之，  /  不亦說乎。  /  有朋自遠方來，  /  不亦樂乎。」

Four boundary rules, each switchable (plus `classifier_join`, which picks a relation rather than a boundary):

  `quote_spans`  a BALANCED quoted span holds no boundary, and the closing mark stays inside it.
  `pause_join`   no sentence ends at a pause mark (，、；：,;:) of any kind.
  `join_unpunctuated`  no sentence opens where NOTHING marks a boundary: when the only tokens between two
                 parser sentences are quotation marks (or none at all) the clauses are joined; any
                 other punctuation, brackets included, is a mark and leaves the boundary alone, so a
                 sentence ends only at a sentence-final mark. Unpunctuated text therefore chains up to
                 `max_sent`. `classifier_join` never fires on these joins (it was validated only on
                 comma-split pairs).
  `final_pull`   no sentence OPENS on a sentence-final mark (。．.！？!?।॥…). Generalises the
                 invariant `align_kanripo_punct.py` enforces on the GOLD corpus at build time
                 ("no unit may open with a sentence-final mark") to the PARSER's own boundaries
                 at inference: in punctuated input a real sentence ends ON its closing mark, so a
                 sentence whose last token is ordinary content — it ends on a BARE character —
                 while the very next token is a sentence-final mark means that mark was left
                 stranded opening the following sentence instead of closing this one. Pull it
                 back. Scoped to sentence-final marks only: a sentence opening on a PAUSE mark is
                 already fully handled by `pause_join` (a real merge, not just moving one token),
                 and an opening quote/bracket starting a sentence is a genuine new span.

HOW A MERGE IS DONE. `token.is_sent_start` is NOT writable on a parsed doc (spaCy raises E043), and
a sentence begins exactly where a token heads itself — so merging means giving the second
sentence's ROOT a head in the first. That is done IN PLACE (`token.head = …`), so this component
never rebuilds the `Doc` and the standing "anything that rebuilds a Doc owns carrying EVERY
annotation" trap does not arise: lemma, MORPH, NORM and every extension survive by construction.

WHAT IT ATTACHES, AND WHY IT IS NOT A GUESS. Both answers are read off the treebank
(`build_lzh_sent_joins.py` prints the derivation):

  * **Inside a quote, the first freed clause hangs off the SPEECH VERB; later ones chain.**
    ⚠ THE EVIDENCE HERE IS THIN AND MUST NOT BE OVERSTATED. It is tempting to quote "1 743 of
    Kyoto's 1 755 balanced quoted spans have exactly ONE external attachment" — true, but nearly
    vacuous, because almost all of those spans hold a SINGLE clause. The statistic that actually
    bears on this rule is spans with an internal sentence-final mark, and there are **18 of them**:
    8 attach the second clause OUTSIDE the span (`comp:obj` of the speech verb, 8/8), 7 hold the
    block root, 3 attach INSIDE (`conj:coord` 2, `comp:obj` 1). The broader cell — any clause after
    a full stop whose previous head is a VERB, n=155 — is genuinely undecided (`parataxis` 34 %,
    `comp:obj` 32 %, `conj:coord` 19 %) and `build_lzh_sent_joins.py` rejects it outright.
    So: follow the 8/18 plurality ONCE, then chain. ⚠ Kyoto does NOT use `parataxis` here; the
    91.2 % `parataxis` figure for 曰 in docs/chinese-family.md belongs to `cross_unit_rules.py`,
    which relates two 句讀 units ACROSS a block boundary — a different configuration.
  * ⚠ **AND THE ATTACHMENT IS CAPPED AT WHAT GOLD ATTESTS.** A governor in the training split has
    one `comp:obj` 96.88 % of the time and **two at most — never three**. Attaching every freed
    clause to the speech verb, which an earlier version of this component did, gives 曰 four
    `comp:obj` children on a four-clause quote: a configuration the treebank never shows. Past
    `max_same_dep` (2, the gold maximum) the rule falls back to chaining onto the previous clause
    with the harvested relation. A rule that is right on the plurality is still wrong if it emits
    trees the annotation scheme does not contain.
  * **Everywhere else the relation is conditioned on the UPOS of the previous unit's head**, from
    the harvested table: after a pause mark a VERB head takes `comp:obj` (69 % of 6 545) but a NOUN
    head takes `conj:coord` (87 % of 1 092) and a PROPN head `conj:coord` (95 % of 552). Cells that
    clear neither threshold — `pause`+AUX, `final`+VERB, where nothing reaches half — fall through
    to `default` rather than memorising a plurality.

⚠ ONLY **BALANCED** SPANS MERGE. An opening mark with no closer is ignored, so a stray 「 cannot
collapse a text into one document-long sentence; `max_span` is a second guard on the same failure.

⚠ AND `pause_join` IS A CONVENTION, NOT TREEBANK FIDELITY: **31.2 % of Kyoto's gold sentence blocks
end at a pause mark**. Refusing to break there is a deliberate reading decision imposed at
inference, and the treebank's own gold scores it down accordingly. The measured cost is in
docs/chinese-family.md.

PLACEMENT: **last**. It rewrites arcs, and the `sud_*` MISC pipes read the tree, so an earlier
position would change their input and couple this to that layer (CLAUDE.md standing hazard 5).
"""
import json
import math
import pathlib
import warnings

from spacy.language import Language
from spacy.util import ensure_path

# Quotation marks proper: corner brackets, curly quotes, guillemets. Straight " and ' are
# deliberately ABSENT — each is its own closer, so no stack can tell an opener from a closer, and a
# wrong pairing would merge across the wrong span silently.
QUOTE_PAIRS = "「」『』“”‘’«»‹›"
BRACKET_PAIRS = "（）〔〕【】《》〈〉［］｛｝()[]{}"
PAUSE = "，、；：,;:"
# Matches `clause_parser._SENT_FINAL` exactly, so the two components agree on what counts as a
# sentence-final mark even though `clause_parser` no longer runs on lzh (CLAUDE.md).
SENT_FINAL = "。．.！？!?।॥…"
DEFAULT_JOIN_DEP = "conj:coord"
# Inside a quoted span, everything after the FIRST clause. The first clause keeps whatever the
# parser gave it (`comp:obj` of 曰); the rest are juxtaposed to it.
QUOTE_DEP = "parataxis"
# Two roots the parser left separate are joined with `parataxis` — juxtaposed predications — EXCEPT
# in three configurations where the relation is coordination instead. See `_join_dep`.
CLAUSE_DEP = "parataxis"
COORD_DEP = "conj:coord"
# What counts as a predicate for exception (a). lzh has no ADJ; nominal predication is headed by the
# noun, which is exactly the case (a) is meant to catch.
PREDICATE_POS = ("VERB", "AUX")
# ⚠ BOTH CAPS ARE DERIVED FROM THE TREEBANK, and both exist because real editions break what Kyoto's
# 句讀-sized blocks never could. Kyoto's longest attested BALANCED QUOTED SPAN is 42 tokens (median
# 8, p99 24), so 60 never clips a genuine quote; p99.9 of its gold SENTENCE length is 37 (max 123,
# a single unmergeable block), so 100 never clips a genuine sentence. They bind only on runaway
# merges — the Heart Sūtra's whole discourse to Śāriputra is ONE 246-token quotation, and without
# `max_span` the sūtra comes out as 3 sentences with a 254-token one. On the released harness both
# caps are free (LAS 76.39 / SENTS_F 95.27 either way); on multi-sentence input they HELP
# (LAS 71.35 -> 71.66). 0 disables either.
MAX_SPAN = 60
MAX_SENT = 100
# `classifier_join`'s symbolic features, duplicated from (not imported from)
# scripts/analyze_direction_classifier_errors.py: that module pulls in torch/transformers/sklearn
# for LIVE SikuBERT + a random forest, and this rule is the DISTILLED alternative built specifically
# to avoid a new runtime dependency in a shipped wheel (scripts/train_lzh_sentjoin_glue.py trains it
# on the same pairs with a static, already-shipped vector table instead). See that script's module
# docstring and gold/lzh-crossunit-gluing-negative-result memory for the full history: a classifier
# that decided this same question at TRAINING-corpus-build time never beat the deployed baseline,
# because baking its calls into training data biases the parser's own learned prior; this rule makes
# the identical decision at INFERENCE time instead, on the parser's own untouched output.
GLUE_NEG = set("不弗未毋勿非無莫")
GLUE_FINAL_PART = set("也矣焉乎哉與邪")
GLUE_COND_MARKERS = set("則必雖苟縱若故")
# 0.6, not the 0.85 an earlier (methodologically flawed) validation round chose: once the marker
# gate above restricts firing to pairs carrying a DECLARED subordinator, the raw end-to-end sweep
# (0.5/0.6/0.7/0.85/0.95, 457-doc test corpus) shows 0.5 and 0.6 tied for the largest of a uniformly
# small, real (bootstrap-confirmed, not noise) parataxis-F gain, with 0.6 the more conservative of
# the two ties. See lzh-sentjoin-classifier-glue memory for the full number.
CLASSIFIER_THRESHOLD = 0.6
# The SPEECH-VERB class, read off the treebank's own tagset rather than hand-listed: Kyoto's XPOS
# field 4 `伝達` ("transmission") covers 曰 5 721, 謂 1 115, 言 1 087, 聞 790, 問 605, 命 560,
# 聽 449, 說 336, 對 315, 告 297, 教 253, 云 215 — and 41.0 % of the VERB tokens carrying it govern
# a clause-headed `comp:obj`, four times the rate of any other field-4 value. A hand-written list
# would have to be maintained; this is derived from the annotation the tagger already predicts.
SPEECH_XPOS = "伝達"
SUBJ_DEP = "subj"
COMP_DEP = "comp:obj"


def _pairs(spec):
    if len(spec) % 2:
        raise ValueError(f"sent_join: `pairs` must have an even length, got {len(spec)}: {spec!r}")
    closer_of = {spec[i]: spec[i + 1] for i in range(0, len(spec), 2)}
    return closer_of, {c: o for o, c in closer_of.items()}


def balanced_spans(texts, spec=QUOTE_PAIRS, open_quotes=False):
    """The OUTERMOST balanced (open, close) index pairs. A closer matches only the most recent
    opener OF ITS OWN KIND, so 「…『…』…」 nests and a mismatched run does not pair by accident.

    With `open_quotes`, an opener that is NEVER CLOSED also yields a span, running to the end of the
    text. ⚠ THAT IS THE CASE THE TREEBANK ACTUALLY CONTAINS: Kyoto blocks are single 句讀 units, so
    a quotative frame routinely opens in one block (子曰：「) and its content is in the next, leaving
    the span unbalanced in every block it touches. `cross_unit_rules.py` joins those with `comp:obj`
    of the speech verb — 630 of the 961 cross-unit arcs it creates — and without this they are
    invisible to the quote rule.

    ⚠ AN OPEN SPAN IS TRUNCATED AT THE NEXT BALANCED ONE. A new quotation beginning means the
    unclosed one is over; without that bound a single stray 「 would swallow the rest of the input
    into one sentence, which is the failure the balanced-only rule was written to avoid."""
    closer_of, opener_of = _pairs(spec)
    stack, spans = [], []
    for i, tx in enumerate(texts):
        if tx in closer_of:
            # ⚠ A REPEATED OPENER OF THE SAME KIND IS A CONTINUATION MARK, NOT A NESTING LEVEL.
            # CBETA punctuates a quotation that runs over several paragraphs with a fresh 「 at the
            # start of each and ONE 」 at the very end — the same convention English uses for
            # continued speech. The Heart Sūtra has 「x5 against 」x2 for exactly this reason, in
            # every edition including CBETA's own; treating each 「 as a new level leaves an
            # unclosed opener and, with `open_quotes`, swallows the rest of the text into one
            # sentence (333 of its 371 tokens). Genuine same-kind nesting is not the Chinese
            # convention — an inner quote takes 『』 — so this costs nothing real.
            if any(t == tx for t, _ in stack):
                continue
            stack.append((tx, i))
        elif tx in opener_of:
            want = opener_of[tx]
            for depth in range(len(stack) - 1, -1, -1):
                if stack[depth][0] == want:
                    if depth == 0:
                        spans.append((stack[0][1], i))
                    del stack[depth:]
                    break
    if open_quotes and stack:
        o = stack[0][1]                                  # the outermost unclosed opener
        later = [a for a, _ in spans if a > o]
        spans.append((o, (min(later) - 1) if later else len(texts) - 1))
    return sorted(spans)


@Language.factory("sent_join",
                  default_config={"pairs": QUOTE_PAIRS, "pause": PAUSE, "quote_spans": True,
                                  "pause_join": True, "joins": None,
                                  "default_dep": CLAUSE_DEP, "max_span": MAX_SPAN,
                                  "max_same_dep": 2, "quote_dep": QUOTE_DEP,
                                  "coord_dep": COORD_DEP, "predicate_pos": PREDICATE_POS,
                                  "speech_xpos": SPEECH_XPOS, "subj_dep": SUBJ_DEP,
                                  "comp_dep": COMP_DEP, "open_quotes": True,
                                  "max_sent": MAX_SENT, "final_pull": True,
                                  "sent_final": SENT_FINAL, "classifier_join": True,
                                  "classifier_threshold": CLASSIFIER_THRESHOLD,
                                  "join_unpunctuated": True, "unmarked_relations": True})
def make_sent_join(nlp, name, pairs, pause, quote_spans, pause_join, joins, default_dep, max_span,
                   max_same_dep, quote_dep, coord_dep, predicate_pos, speech_xpos, subj_dep,
                   comp_dep, open_quotes, max_sent, final_pull, sent_final, classifier_join,
                   classifier_threshold, join_unpunctuated, unmarked_relations):
    return SentJoin(pairs, pause, quote_spans, pause_join, joins, default_dep, max_span,
                    max_same_dep, quote_dep, coord_dep, predicate_pos, speech_xpos, subj_dep,
                    comp_dep, open_quotes, max_sent, final_pull, sent_final, classifier_join,
                    classifier_threshold, join_unpunctuated, unmarked_relations)


class SentJoin:
    def __init__(self, pairs=QUOTE_PAIRS, pause=PAUSE, quote_spans=True, pause_join=True,
                 joins=None, default_dep=CLAUSE_DEP, max_span=None, max_same_dep=2,
                 quote_dep=QUOTE_DEP, coord_dep=COORD_DEP, predicate_pos=PREDICATE_POS,
                 speech_xpos=SPEECH_XPOS, subj_dep=SUBJ_DEP, comp_dep=COMP_DEP,
                 open_quotes=True, max_sent=None, final_pull=True, sent_final=SENT_FINAL,
                 classifier_join=True, classifier_threshold=CLASSIFIER_THRESHOLD,
                 join_unpunctuated=True, unmarked_relations=True):
        self.pairs = pairs
        _pairs(pairs)                                 # fail at construction, not at the first call
        self.pause = set(pause)
        self.quote_spans = quote_spans
        self.pause_join = pause_join
        self.final_pull = final_pull
        self.join_unpunctuated = join_unpunctuated
        self.unmarked_relations = unmarked_relations
        self.quote_chars = set(QUOTE_PAIRS)
        self.sent_final = set(sent_final)
        self.default_dep = default_dep
        self.max_span = MAX_SPAN if max_span is None else max_span
        # The most children with ONE relation this component may give a single governor. 2 is the
        # gold maximum for `comp:obj` (96.88 % of governors have one, 3.12 % have two, none have
        # three); 0 disables the cap.
        self.max_same_dep = max_same_dep
        # The relation every clause AFTER the first inside a quoted span receives. A verb has one
        # object slot; further quoted clauses are juxtaposed utterances, not second objects.
        self.quote_dep = quote_dep
        self.coord_dep = coord_dep
        self.predicate_pos = tuple(predicate_pos)
        self.speech_xpos = speech_xpos
        self.subj_dep = subj_dep
        self.comp_dep = comp_dep
        # Also treat an UNCLOSED opening quotation mark as a span, to the next balanced quote or the
        # end of the input. This is the block-break quotative frame; see `balanced_spans`.
        self.open_quotes = open_quotes
        # Refuse a join that would make the sentence longer than this (0 = no limit). `pause_join`
        # never breaks at a comma, so a paragraph with no sentence-final mark chains end to end —
        # one test-set paragraph reached 1 003 tokens against a median of 6. That is fine as a
        # reading convention and ruinous as a TRAINING EXAMPLE for a transition parser, so the
        # corpus merger sets it; inference leaves it off by default.
        self.max_sent = MAX_SENT if max_sent is None else max_sent
        # The distilled backward/mod classifier (see the module-level GLUE_* constants and
        # scripts/train_lzh_sentjoin_glue.py). `classifier_join` is a config-level on/off switch;
        # the trained weights themselves are NOT a config value (CLAUDE.md standing hazard 4 — a
        # config path is a host path) and live only in `self._glue`, set by `load_glue_file` at
        # BUILD time and carried across save/load by `to_disk`/`from_disk` alone. With no weights
        # loaded the switch is a pure no-op: every decision falls through to the existing rule.
        self.classifier_join = classifier_join
        self.classifier_threshold = classifier_threshold
        self._glue = None
        self.debug = None               # set to [] to have `_note` record every decision
        # ⚠ `joins` IS ACCEPTED AND IGNORED. It named the harvested (mark kind, previous-head
        # UPOS) table that `_dep_for` used before the principled rule replaced it. Loading it also
        # overwrote `default_dep` with the table's own default — which silently turned every
        # `parataxis` decision into `conj:coord` in a built wheel. The parameter is kept only so an
        # old config still loads; it now warns instead of taking effect.
        if joins:
            warnings.warn(f"{name if 'name' in dir() else 'sent_join'}: `joins` is superseded by "
                          f"the clause rule and is ignored; remove it from the config.",
                          RuntimeWarning)

    # --- helpers -----------------------------------------------------------------
    def _is_mark(self, t):
        return t.is_punct or t.text in self.pause

    def _unit_head(self, doc, hi):
        """Head of the content run ending at `hi`: scan left to the previous mark, then take the
        token in the run whose own head lies outside it. Matches build_lzh_sent_joins.py exactly."""
        return self._unit_bounds(doc, hi)[1]

    def _has_subj(self, tok):
        return any(c.dep_.split("@")[0] == "subj" for c in tok.children)

    def _join_dep(self, doc, prev_head, this_head, this_start, chain_head,
                  a_lo=None, a_hi=None, b_lo=None, b_hi=None, allow_classifier=True,
                  unmarked=False):
        """The relation for joining two roots the parser left separate.

        `parataxis` — two juxtaposed predications — UNLESS one of four configurations holds, in
        which case the relation is coordination, complementation, or (d) a REVERSED `mod`:

          (a) one of the two heads is NOT A PREDICATE. Coordinating two nominals, or a nominal with
              a clause, is not parataxis. ⚠ This branch is the one the SPEC left open and the
              treebank settles: over the 2 840 joins of this shape in train, gold uses `conj:coord`
              69.4 % of the time against `parataxis` 1.2 %.
          (b) the second clause OPENS WITH A SCONJ. A subordinator makes the second clause depend on
              the first rather than stand beside it. ⚠ Unattested in the joined configuration in
              train (n=0), so this branch is a principled stipulation with no measured support —
              which is the honest status to record for it.
          (c) THE FIRST CLAUSE OF THE CHAIN has a subject and this one does not. A subjectless
              continuation is reading the same subject, which is coordination, not juxtaposition.
              Note `chain_head`, not `prev_head`: the test is against the clause that OPENED the
              chain, so a run of subjectless clauses all coordinate with it.
          (d) THE DISTILLED CLASSIFIER (`classifier_join`) calls this pair BACKWARD with confidence
              >= `classifier_threshold` (0.6) AND at least one side carries a DECLARED subordinator
              (`GLUE_COND_MARKERS` = 則/必/雖/苟/縱/若/故, the same particles
              `cross_unit_rules.py`'s `DECLARED_ANTECEDENT`/`DECLARED_CONSEQUENT` use): the FIRST
              (earlier) clause is a conditional/concessive dependent of the SECOND, as in
              若/苟/雖/縱/則/故-marked couplets, so the relation is `mod` and the direction REVERSES
              exactly as in (a2). The marker requirement is checked in `_classifier_backward_proba`
              BEFORE the learned score is even consulted — see that method's docstring for why: it
              is what makes this branch fire only on pairs an actual Classical Chinese subordinator
              licenses, not on bare statistical resemblance.

              ⚠ **THE RIGHT WAY TO VALIDATE THIS BRANCH, LEARNED THE HARD WAY (2026-09-19, four
              rounds)**: `pause_join` ALWAYS joins two roots at a comma boundary — that policy is
              fixed, its cost already measured and accepted. This branch's only job is choosing
              WHICH RELATION a join that is happening EITHER WAY gets. The first three rounds
              scored each firing against "is there a real gold arc here," which for a genuinely
              unrelated real-sentence pair is always "no" regardless of which relation is chosen —
              that check cannot validate a relation-choice rule even in principle, and it reported a
              disqualifying 2.6%-then-0% "precision" that was an artifact of asking the wrong
              question, not a real defect. The correct check (round four, replicated after adding
              the marker gate) is the AGGREGATE downstream metric on raw text, same as
              `quote_spans`/`pause_join`/`final_pull` were themselves validated with: OFF vs ON at
              `classifier_threshold`=0.6 on the 457-document raw test corpus gives a real
              (bootstrap-confirmed, not noise) but SMALL net trade — parataxis F +0.11 (90% CI
              [+0.05, +0.18]), mod F −0.04 (90% CI [−0.07, −0.02]), overall LAS unchanged (90% CI
              [0.00, 0.00]). Full four-round history in the lzh-sentjoin-classifier-glue memory.
              `_join_left`'s `cross_block`/`allow_classifier` plumbing (added in round two, when the
              cause was mis-diagnosed as a local gating bug) is harmless and stays, but the marker
              gate above — not that plumbing — is what actually makes this branch's firings
              defensible.

        ⚠ **UNMARKED JOINS USE A DIFFERENT TABLE** (`unmarked=True`, i.e. `join_unpunctuated`). The
        comp:obj/coord figures above were read off boundaries a mark sits on — and in a rule-merged
        corpus, where many of those arcs were themselves written by `cross_unit_rules`, so they say
        little about a boundary with NO mark. An editor does not put a mark inside a tight
        complement, so an unmarked boundary leans the other way, and the gold arcs bear that out
        (forward predicate->predicate arcs with no punctuation between; derived on train, checked on
        test): if the second clause has its OWN SUBJECT it is `comp:obj` of the first 78 % (train,
        n=3 974) / 83 % (test, n=329) — the current default `parataxis` is right 10 % / 6 % there;
        without a subject it is a three-way tie (parataxis 35/30, comp:obj 31/30, conj:coord
        23/27) and `parataxis` stays. Branch (c) is skipped for these joins: its `conj:coord` is the
        weakest of the three when the second clause lacks a subject (20 % train / 24 % test).
        Branches a1-a3 and b are unchanged (the non-predicate populations in gold are dominated by
        punct/flat arcs, not clause links, so there is no clean evidence to move them).

        ⚠ THIS DISAGREES WITH KYOTO'S PLURALITY AND DOES SO DELIBERATELY. In the default case gold
        uses `comp:obj` 68.9 % of the time (n=4 713), and in case (c) 58.5 % (n=1 591). Those
        reflect genuine COMPLEMENT frames — the second unit is an argument of the first unit's verb
        — which is a different question from what relates two roots the parser found no relation
        between at all. The treebank cannot answer that question because it never has two roots.
        """
        prev_pred = doc[prev_head].pos_ in self.predicate_pos
        this_pred = doc[this_head].pos_ in self.predicate_pos
        if not (prev_pred and this_pred):                            # (a), in three sub-cases
            # a1 — the first clause is headed by a SPEECH VERB, so what follows is its reported
            # content, not a coordinate of it. 曰/謂/言/聞/問 …, identified by XPOS field 4.
            if prev_pred and self._is_speech(doc[prev_head]):
                return self._note("a1", self.comp_dep, doc, this_head, other=prev_head)
            # a2 — the first unit is headed by a NOMINAL and the second by a PREDICATE: this is
            # topic-comment, and the direction REVERSES — the nominal is the SUBJECT of the verb,
            # not the verb a dependent of the nominal. 仁者，愛人 -> 仁者 subj-> 愛.
            # ⚠ Only when the first head is still a ROOT. If it has already been joined leftward,
            # re-heading it rightward would tear it out of the chain it belongs to.
            if this_pred and doc[prev_head].head.i == prev_head:
                return self._note("a2", self.subj_dep, doc, this_head, reverse=True, other=prev_head)
            return self._note("a3", self.coord_dep, doc, this_head, other=prev_head)
        if doc[this_start].pos_ == "SCONJ":
            return self._note("b", self.coord_dep, doc, this_head, other=prev_head)
        table = unmarked and self.unmarked_relations
        if not table and chain_head is not None and self._has_subj(doc[chain_head]) \
                and not self._has_subj(doc[this_head]):
            return self._note("c", self.coord_dep, doc, this_head, other=prev_head)
        if self.classifier_join and allow_classifier and self._glue is not None and a_lo is not None \
                and doc[prev_head].pos_ in ("VERB", "ADJ") and doc[this_head].pos_ in ("VERB", "ADJ") \
                and doc[prev_head].head.i == prev_head:
            p = self._classifier_backward_proba(doc, a_lo, a_hi, prev_head, b_lo, b_hi, this_head)
            if p is not None and p >= self.classifier_threshold:
                return self._note("classifier-backward", "mod", doc, this_head,
                                  reverse=True, other=prev_head)
        if table and self._has_subj(doc[this_head]):
            return self._note("u-subj", self.comp_dep, doc, this_head, other=prev_head)
        return self._note("default", self.default_dep, doc, this_head, other=prev_head)

    # --- the distilled backward/mod classifier ------------------------------------
    def load_glue_file(self, path):
        """Load `classifier_join`'s trained weights from a plain JSON file (built by
        scripts/train_lzh_sentjoin_glue.py). Call this ONCE, at BUILD time, on a freshly
        constructed pipe, before `nlp.to_disk(...)`: `to_disk` copies the resolved weights into
        this component's OWN directory, and `from_disk` reads them back only from there — never
        from `path` again. A config value naming `path` would be a host path baked into config.cfg
        (CLAUDE.md standing hazard 4); this is a plain training artefact, not something an end
        user's machine should ever be asked to resolve."""
        self._glue = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    def _unit_bounds(self, doc, hi):
        """(lo, head_i) for the content run ending at `hi` — see `_unit_head`, which this backs."""
        lo = hi
        while lo > 0 and not self._is_mark(doc[lo - 1]):
            lo -= 1
        for i in range(lo, hi + 1):
            h = doc[i].head.i
            if h == i or not (lo <= h <= hi):
                return lo, i
        return lo, lo

    def _glue_span(self, doc, lo, hi, head_i):
        """Symbolic features for one side of a candidate join, over the token range [lo, hi].
        Mirrors analyze_direction_classifier_errors.py's `side_features()`, with one approximation:
        that function excludes tokens whose CoNLL-U head field is the literal treebank-sentence
        root ("0"); here it excludes the unit's own head token instead. The two coincide for a
        genuine 句讀 unit, which is exactly what an as-yet-unmerged sentence IS at the point
        `_join_left` inspects it."""
        toks = list(doc[lo:hi + 1])
        real = [t for t in toks if not t.is_punct]
        first = real[0] if real else toks[0]
        has_neg = any(t.text in GLUE_NEG for t in toks)
        has_yi = any(t.text == "以" for t in toks) and first.text != "以"
        ends_final = toks[-1].text in GLUE_FINAL_PART if toks else False
        ends_close_quote = toks[-1].text == "」" if toks else False
        deprel_set = frozenset(t.dep_ for t in toks if t.i != head_i and not t.is_punct)
        has_internal_mod = "mod" in deprel_set
        mod_forms = [t.text for t in toks if t.i != head_i and t.dep_ == "mod"]
        has_cond_marker = any(m in GLUE_COND_MARKERS for m in mod_forms)
        has_subj = any(t.head.i == head_i and t.dep_ in ("subj", "subj@pass") for t in toks)
        return {"first_upos": first.pos_, "len": len(toks), "has_neg": has_neg, "has_yi": has_yi,
                "ends_final_particle": ends_final, "ends_close_quote": ends_close_quote,
                "deprel_set": deprel_set, "has_internal_mod": has_internal_mod,
                "has_cond_marker": has_cond_marker, "has_subj": has_subj}

    def _classifier_backward_proba(self, doc, a_lo, a_hi, a_head, b_lo, b_hi, b_head):
        """P(this pair is BACKWARD) from the raw-space-folded logistic regression `self._glue`
        holds (see train_lzh_sentjoin_glue.py step 5: the StandardScaler is folded into the weight
        vector, so this is a single dot product on the untransformed features). Returns None if no
        weights are loaded, or if this pipeline's vocab carries no vectors of the trained
        dimensionality (an older base without the SikuBERT channel) — `classifier_join` then never
        fires, rather than scoring on a feature it was not trained with.

        ⚠ **REQUIRES A DECLARED MARKER** (`GLUE_COND_MARKERS` = `cross_unit_rules.py`'s own
        `DECLARED_ANTECEDENT`/`DECLARED_CONSEQUENT` particles, 則/必/雖/苟/縱/若/故) on at least one
        side, checked BEFORE the learned score. Without this gate the classifier fires on any
        comma-adjacent pair its statistical features happen to resemble a conditional couplet,
        including plain juxtapositions of two independent real sentences that share no grammatical
        marker at all — the population that made three earlier evaluation rounds a wash (real
        effect, but +parataxis-F/−mod-F cancelling to ≈0 LAS; see the lzh-sentjoin-classifier-glue
        memory). Restricting to marker-bearing pairs ties every reversal to an actual Classical
        Chinese subordinator rather than a bare statistical resemblance — the classifier's role
        becomes deciding WHICH marked pair truly reverses (a soft confirmation on grammar the
        annotators already recognise), not inventing structure where no marker licenses it."""
        g = self._glue
        af = self._glue_span(doc, a_lo, a_hi, a_head)
        bf = self._glue_span(doc, b_lo, b_hi, b_head)
        if not (af["has_cond_marker"] or bf["has_cond_marker"]):
            return None
        bool_vals = {
            "a_has_subj": af["has_subj"], "a_has_neg": af["has_neg"], "a_has_yi": af["has_yi"],
            "a_ends_final_particle": af["ends_final_particle"],
            "b_has_subj": bf["has_subj"], "b_has_neg": bf["has_neg"], "b_has_yi": bf["has_yi"],
            "b_ends_close_quote": bf["ends_close_quote"],
            "a_has_internal_mod": af["has_internal_mod"], "b_has_internal_mod": bf["has_internal_mod"],
            "deprel_set_match": bool(af["deprel_set"]) and af["deprel_set"] == bf["deprel_set"],
            "a_has_cond_marker": af["has_cond_marker"], "b_has_cond_marker": bf["has_cond_marker"],
        }
        num_vals = {"a_len": af["len"], "b_len": bf["len"], "token_distance": abs(a_head - b_head)}
        x = [float(bool_vals[c]) for c in g["bool_cols"]] + [float(num_vals[c]) for c in g["num_cols"]]
        x += [1.0 if bf["first_upos"] == cat else 0.0 for cat in g["upos_categories"]]
        a_vec, b_vec = doc[a_head].vector, doc[b_head].vector
        if len(a_vec) != g["vec_dim"] or len(b_vec) != g["vec_dim"]:
            return None
        x += [float(bv - av) for av, bv in zip(a_vec, b_vec)]
        w = g["weight"]
        z = g["bias"] + sum(wi * xi for wi, xi in zip(w, x))
        if z > 700:
            return 1.0
        if z < -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def _is_speech(self, tok):
        """A speech verb, by the treebank's own tagset: XPOS field 4 == `伝達`. Read off `tag_`,
        which the tagger has already predicted by the time this pipe runs (it goes LAST)."""
        parts = tok.tag_.split(",")
        return len(parts) == 4 and parts[-1] == self.speech_xpos

    def _note(self, branch, dep, doc, tok, reverse=False, other=None):
        """Record which branch decided this arc, when `pipe.debug` is a list. Off by default and
        costs nothing then — but re-deriving the branch from the output afterwards means writing
        the rule twice, and the second copy is the one that is wrong.

        Returns `(dep, reverse)`; `reverse` means the FIRST clause's head becomes the dependent of
        the second's, which is the a2 topic-comment case."""
        if self.debug is not None:
            # BOTH ends, and which is the dependent: for a reversed arc the dependent is the
            # FIRST clause's head, so recording only `tok` prints the arc back to front.
            dep_i, head_i = (other, tok) if reverse else (tok, other)
            self.debug.append({"dep_i": dep_i, "head_i": head_i, "dep": dep,
                               "branch": branch, "reverse": reverse})
        return dep, reverse

    @staticmethod
    def _would_cycle(doc, root, head):
        t, guard = doc[head], 0
        while t.head.i != t.i and guard < len(doc):
            if t.i == root:
                return True
            t = t.head
            guard += 1
        return t.i == root

    def _attach(self, doc, root, head, dep):
        """Re-head `root` onto `head`, and REVERT if that does not actually merge the sentences.

        ⚠ AN ATTACHMENT CAN LEAVE THE BOUNDARY IN PLACE. spaCy derives `doc.sents` from the tree,
        but a token whose new head lies across intervening material belonging to another subtree
        makes the span DISCONTINUOUS, and the sentence does not merge — the dependent keeps a head
        outside its own sentence. Writing that to CoNLL-U produces a head index outside the block:
        53 of them in the test set before this guard, silently, with the corpus still loading and
        converting and training. Verify the postcondition instead of assuming it."""
        if head == root or self._would_cycle(doc, root, head):
            return False
        # ⚠ LENGTH IS CHECKED BEFORE MUTATING, NOT BY REVERTING. Undoing `token.head` does NOT
        # reliably re-split the sentence spaCy merged when the head was set, so an attach-then-revert
        # left ONE span holding TWO roots — four such blocks reached the CoNLL-U, each of them
        # exactly an over-length sentence. The prospective length is exact and needs no mutation:
        # the merged sentence would run from the head's sentence start to the root's sentence end.
        if self.max_sent:
            if (doc[root].sent.end - doc[head].sent.start) > self.max_sent:
                return False
        old_head, old_dep = doc[root].head.i, doc[root].dep_
        doc[root].head = doc[head]
        doc[root].dep_ = dep
        sent = doc[root].sent
        if not (sent.start <= head < sent.end):
            doc[root].head = doc[old_head]
            doc[root].dep_ = old_dep
            return False
        return True

    # --- the pipe ----------------------------------------------------------------
    def __call__(self, doc):
        if not doc.has_annotation("DEP") or len(doc) < 2:
            return doc
        texts = [t.text for t in doc]
        spans = []
        if self.quote_spans:
            spans = [(o, c) for o, c in balanced_spans(texts, self.pairs,
                                                       open_quotes=self.open_quotes)
                     if not self.max_span or c - o + 1 <= self.max_span]
        in_span = {}
        for o, c in spans:
            for i in range(o + 1, c + 1):
                in_span.setdefault(i, (o, c))
        # Everything that depends on the CURRENT tree is read off BEFORE any merge: a merge gives a
        # root an external head, which would otherwise make it look like the span's first external
        # attachment on the next iteration, and would drop it out of `doc.sents` as a start.
        roots = {s.start: s.root.i for s in doc.sents}
        anchors = {(o, c): self._quote_anchor(doc, o, c) for o, c in spans}
        chain = {}                      # attached root -> the head of the clause that opened its chain

        # Where each span is currently hanging its juxtaposed clauses, and the last one it hung.
        held = {k: [v, None] for k, v in anchors.items()}
        # A quote whose own first clause is still a ROOT hangs off its speech verb as `comp:obj`,
        # reproducing the single-attachment-point convention. Done BEFORE the loop so the anchor is
        # attached by the time later clauses are chained onto it.
        for (o, c), anchor in list(anchors.items()):
            if anchor is None or doc[anchor].head.i != anchor:
                continue
            frame = self._speech_frame(doc, o)
            if frame is not None and self._attach(doc, anchor, frame, self.comp_dep):
                if self.debug is not None:
                    self.debug.append({"dep_i": anchor, "head_i": frame, "dep": self.comp_dep,
                                       "branch": "quote-frame", "reverse": False})

        starts = sorted(roots)
        for k, s in enumerate(starts):
            if k == 0:
                continue
            root = roots[s]
            if s in in_span:
                key = in_span[s]
                anchor, last = held[key]
                if anchor is not None:
                    # FLAT on the first clause, which is what Kyoto does (9 600 flat against 110
                    # chained) — until the anchor reaches the arity gold attests, at which point
                    # the next clause hangs off the previous one instead. `parataxis` maxes out at
                    # TWO children per governor in the training split and never three, exactly as
                    # `comp:obj` does, so an unbounded flat fan-out would leave the attested space
                    # on any quote of four clauses or more.
                    if self._at_cap(doc, anchor, self.quote_dep) and last is not None:
                        anchor = last
                    if self._attach(doc, root, anchor, self.quote_dep):
                        held[key] = [anchor, root]
                        continue
                # `cross_block=True`: an open quote span can and does swallow multiple genuinely
                # separate real sentences (that is the whole reason `open_quotes` exists), so a
                # pair reached only via this fallback is never eligible for `classifier_join`.
                self._join_left(doc, s, root, chain, cross_block=True)   # no quotative frame
            elif self.pause_join and self._after_pause(doc, s):
                self._join_left(doc, s, root, chain)
            elif self.join_unpunctuated and self._no_mark_before(doc, s):
                # The previous PARSER sentence is the clause on the left (a comma-bounded unit does
                # not exist here), and `cross_block=True` keeps `classifier_join` out of it.
                self._join_left(doc, s, root, chain, cross_block=True, unmarked=True,
                                unit=(starts[k - 1], roots[starts[k - 1]]))
        # ...and the closing mark must end up inside the span it closes: one the parser made a root,
        # or attached to something after the span, would open (or be swallowed by) another sentence.
        closers = {c for _, c in balanced_spans(texts, self.pairs)}
        for o, c in spans:
            if c not in closers:            # an OPEN span ends at a content token, not a mark
                continue
            if doc[c].head.i >= c:
                left = self._content_left(doc, c - 1)
                if left is not None:
                    self._attach(doc, c, self._unit_head(doc, left), "punct")
        if self.final_pull:
            self._pull_final(doc)
        return doc

    def _pull_final(self, doc):
        """No sentence may OPEN on a sentence-final mark: if this sentence's last token is
        ordinary content (it ends on a BARE character) and the very next token — the following
        sentence's first token — is one of `self.sent_final`, that mark belongs to CLOSING this
        sentence and was left stranded opening the next one instead. Pull it back as `punct` of
        this sentence's own unit head, using `_attach` for the same cycle/length/postcondition
        checks every other merge in this module gets — never assume the reattachment actually
        moved the boundary.

        Runs LAST, after `quote_spans` and `pause_join`, over a FRESH `doc.sents` snapshot: both
        of the earlier rules can themselves create or resolve exactly this configuration, so
        checking before they run would both miss cases they introduce and misfire on ones they
        already fixed."""
        for sent in list(doc.sents):
            last = sent[-1]
            if last.is_punct:
                continue
            nxt = last.i + 1
            if nxt < len(doc) and doc[nxt].text in self.sent_final:
                self._attach(doc, nxt, self._unit_head(doc, last.i), "punct")

    def _at_cap(self, doc, gov, dep):
        """Would a further `dep` child put this governor past what the treebank ever shows?"""
        if not self.max_same_dep:
            return False
        return sum(1 for c in doc[gov].children if c.dep_ == dep) >= self.max_same_dep

    def _content_left(self, doc, i):
        """Index of the nearest non-punctuation token at or left of `i`, or None."""
        while i >= 0 and (doc[i].is_punct or doc[i].text in self.pause):
            i -= 1
        return i if i >= 0 else None

    def _quote_anchor(self, doc, o, c):
        """The span's FIRST clause: the token inside it whose head is the speech verb outside.

        That token keeps whatever the parser gave it — `comp:obj` of 曰 — and is the ANCHOR every
        later clause in the span is juxtaposed to. ⚠ Only a governor to the LEFT counts: in every
        quotative frame Kyoto annotates the speech verb PRECEDES the quote (曰 1 504 of the 1 641
        `comp:obj` attachments, 問/云/謂 the rest), so a head to the RIGHT of the span is something
        else and must not be treated as the frame."""
        for i in range(o + 1, c):
            h = doc[i].head.i
            if h == i:
                continue                    # a root contributes no external governor
            if h < o:
                return i
        # No external governor: the quote's first clause is itself still a ROOT. That is the normal
        # state when the parser split at the opening mark, and it is ALWAYS the state when this rule
        # is run over per-block gold trees, where every block is its own sentence. Anchor on that
        # root; `_speech_frame` then supplies the governor it should hang from.
        for i in range(o + 1, c):
            if doc[i].head.i == i and not doc[i].is_punct:
                return i
        return None

    def _speech_frame(self, doc, o):
        """The SPEECH VERB introducing the quote that opens at `o`, if there is one.

        Scanning left past punctuation (：「 is one frame, not two) for a token that is both a
        predicate and 伝達. This is what makes a quotative frame work when the quote's own first
        clause is unattached — the parser having split at the opening mark, or, in the corpus
        merger, every gold block being its own sentence."""
        i = o - 1
        while i >= 0 and doc[i].is_punct:
            i -= 1
        if i >= 0 and doc[i].pos_ in self.predicate_pos and self._is_speech(doc[i]):
            return i
        return None

    def _after_pause(self, doc, s):
        """Is this sentence start one a pause mark opened? Quote and bracket marks in between do
        not break the relation (…也，」X and …也，「X both count)."""
        if doc[s].text in self.pause:
            return True
        i = s - 1
        while i >= 0 and doc[i].is_punct and doc[i].text not in self.pause:
            i -= 1
        return i >= 0 and doc[i].text in self.pause

    def _no_mark_before(self, doc, s):
        """Is there NO boundary mark between the previous content and the sentence that starts at
        `s`? Only QUOTATION marks are looked through (a clause may follow a closing 」 directly).
        Every other punctuation token is a mark: a pause mark is `_after_pause`'s business, a
        sentence-final mark is a real boundary, and so are brackets — a 【…】 or 《…》 wraps an
        editorial note or a title, and joining across one chained unrelated text together (2026-09-20
        review of the first version, which treated brackets as transparent). A whitespace token and
        the start of the document are boundaries too."""
        i = s - 1
        while i >= 0 and doc[i].text in self.quote_chars:
            i -= 1
        if i < 0:
            return False
        t = doc[i]
        # ⚠ A mark is anything with no letter or digit in it, not only what spaCy calls punctuation:
        # `is_punct` is False for section symbols like ○, and the first version of this rule read
        # straight past one -- and the 。 behind it -- to join every dateline heading
        # ("…事。 ○ 九年。") to the sentence before it (found 2026-09-20 by looking at where gold put
        # the head of the joins it made).
        return not (t.is_space or t.is_punct or not any(c.isalnum() for c in t.text))

    def _join_left(self, doc, s, root, chain, cross_block=False, unit=None, unmarked=False):
        """Attach `root` to the head of the unit on its left, with the relation `_join_dep` gives.

        `chain` maps an anchor to the head of the clause that OPENED the chain hanging off it, so
        exception (c) can test the FIRST clause's subject rather than the immediately previous
        one's.

        `cross_block`, TRUE from the quote-span fallback caller, means this join may be bridging
        real separate sentences an open quote span swallowed together — never eligible for the
        `classifier_join` reversal (branch (d) of `_join_dep`), which was only trained on two
        halves of a single still-unmerged 句讀 block. Even when the caller says `False` (the plain
        pause-mark path), a SENT-FINAL mark anywhere in the punctuation actually crossed to reach
        `s` means `_after_pause` let a real sentence boundary through its quote/bracket skip, and
        that pair is refused the same way."""
        i = s - 1
        while i >= 0 and doc[i].is_punct and doc[i].text not in self.pause:
            i -= 1
        left = self._content_left(doc, i if i >= 0 else s - 1)
        if left is None:
            return
        if any(doc[k].text in self.sent_final for k in range(left + 1, s)):
            cross_block = True
        a_lo, anchor = unit if unit is not None else self._unit_bounds(doc, left)
        # the first content token of THIS clause, for the SCONJ test
        this_start = s
        while this_start < len(doc) and doc[this_start].is_punct:
            this_start += 1
        if this_start >= len(doc):
            return
        chain_head = chain.get(anchor, anchor)
        # `root`'s own (not yet merged) sentence end -- the right edge of "this clause" for the
        # classifier's span features, read before any attachment below can extend it.
        b_hi = doc[root].sent.end - 1
        dep, reverse = self._join_dep(doc, anchor, root, this_start, chain_head, a_lo, left, s, b_hi,
                                      allow_classifier=not cross_block, unmarked=unmarked)
        if reverse:
            # topic-comment: the NOMINAL first unit becomes the subject of the second's predicate,
            # so the second clause's head is the merged sentence's root and opens a fresh chain.
            if self._attach(doc, anchor, root, dep):
                chain[root] = root
        elif self._attach(doc, root, anchor, dep):
            chain[root] = chain_head        # the new clause inherits the chain's opening clause

    # --- serialisation: the join table travels inside the model ------------------
    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "cfg.json").write_text(json.dumps(
            {"default_dep": self.default_dep, "coord_dep": self.coord_dep,
             "quote_dep": self.quote_dep, "comp_dep": self.comp_dep, "subj_dep": self.subj_dep,
             "max_same_dep": self.max_same_dep, "max_sent": self.max_sent,
             "pause_join": self.pause_join, "quote_spans": self.quote_spans,
             "open_quotes": self.open_quotes, "final_pull": self.final_pull,
             "join_unpunctuated": self.join_unpunctuated,
             "unmarked_relations": self.unmarked_relations,
             "classifier_join": self.classifier_join,
             "classifier_threshold": self.classifier_threshold}, ensure_ascii=False),
            encoding="utf-8")
        # the trained weights themselves, if `load_glue_file` was called on this instance -- see
        # that method's docstring for why they are NOT a config value.
        if self._glue is not None:
            (path / "glue.json").write_text(json.dumps(self._glue, ensure_ascii=False),
                                             encoding="utf-8")

    def from_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        f = path / "cfg.json"
        if f.exists():
            for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                setattr(self, k, v)
        g = path / "glue.json"
        if g.exists():
            self._glue = json.loads(g.read_text(encoding="utf-8"))
        elif self.classifier_join:
            warnings.warn("sent_join: classifier_join is enabled but no glue.json was found in "
                          "this component's directory; the classifier-backward rule will be a "
                          "no-op until load_glue_file() is called and the model is re-saved.",
                          RuntimeWarning)
        return self
