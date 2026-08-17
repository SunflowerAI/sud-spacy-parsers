#!/usr/bin/env python3
"""End-to-end evaluation of the saṃhitā -> CSL pre-tokeniser: does it actually buy a parse?

WHY THIS NEEDS ITS OWN SCORER. spaCy's `Example` aligns a predicted doc to a reference by requiring
the two texts to agree up to whitespace. That holds for gold CSL input, but not once the segmenter
makes a mistake: a wrong coalescence label does not merely mis-split, it emits DIFFERENT CHARACTERS
(`siṃha vyāghra` vs a predicted `sim havyāghra`), because reversing CSL notation rewrites the
string. So the comparison has to be anchored on the one thing both sides share — the saṃhitā input.

Every token, gold or predicted, is therefore given the half-open range of saṃhitā characters it came
from, derived from the per-character labels, and tokens are matched by that range. This is the
standard way end-to-end parsing is scored from raw text (as for Chinese), and it is strictly
harder than the aligned score: a token whose range is wrong is simply unmatched.

Three input conditions against byte-identical gold:

  gold-preproc   gold tokens, `Compound` supplied by the reader        -- the floor
  gold CSL       the gold CSL string as raw text through the tokeniser -- the ORACLE: what a
                 perfect pre-tokeniser would deliver
  predicted CSL  saṃhitā -> `sa_presegment` -> CSL -> the tokeniser    -- the real number

Scored on the sentences in the pairs file (no elided `_` token): an elided token exists only in the
treebank, so including it measures an artefact rather than the pre-tokeniser.

    eval_sa_raw.py MODEL_DIR PRESEGMENT_DIR data_samhita/test.jsonl
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                        # noqa: E402
# The AGGREGATOR, not a hand-picked list — this is the EIGHTH script to die with E893 on a
# layer it did not know to import. seg_code.py exists precisely so the list lives in one place.
import seg_code                                     # noqa: E402,F401
import sa_tokenizer                                 # noqa: E402,F401
import clause_parser                                # noqa: E402,F401
import sud_unsandhi                                 # noqa: E402,F401
import sa_devanagari                                # noqa: E402,F401  — registers `sa_deva`
from gold_tok_corpus import CompoundCorpus, NormCorpus   # noqa: E402
from sa_presegment import Presegmenter, apply_labels  # noqa: E402

# The corpus MUST match the representation the model was trained on, or every number is nonsense
# while still looking plausible. `corpus_sa_csl_rev` is the SUPERSEDED pausa-normalised
# representation; the shipped arm trains on the DCS/MWT one, whose FORMs differ (a standalone token
# keeps its sandhied surface there). Pointing the old constants at the current model produced
# token F 0.0724 and LAS 0.0324 — a harness artefact, not a result. Overridable so a future
# representation change is a flag, not an edit.
TEST_SPACY = "corpus_sa_split/vedic_test.spacy"
TEST_CONLLU = "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-test.csl_mwt.conllu"
DIVIDERS = (" ", "-")


def token_spans(labels, samhita=None):
    """Per-character labels -> the saṃhitā character range of each token they produce.

    A label may carry more than one divider (`' ô `, where a one-vowel particle is wholly absorbed
    into its neighbour); each extra divider closes a zero-width token, which is exactly what the
    absorbed word is on the saṃhitā side.

    `samhita` must be passed whenever the input carries SPACES. Under the `iast`/`devanagari`
    spacing regimes the word boundaries are already literal spaces in the input, so the labels stop
    marking them and only mark what is left to find (compound breaks, coalescences). Counting label
    dividers alone then undercounts badly — `hasti-varcasam iti hastinam` yielded 2 spans against 4
    gold tokens, and 2491 of 2545 sentences were dropped as unreconstructable. A literal space is a
    token boundary in its own right and belongs to no token. Passing nothing keeps the old
    behaviour, which is correct for the space-free `continuous` regime.
    """
    spans, start = [], 0
    for i, lab in enumerate(labels):
        if samhita is not None and samhita[i].isspace():
            if i > start:                       # close the token the space terminates
                spans.append((start, i))
            start = i + 1                       # the space itself is in no token
            continue
        n = sum(lab.count(d) for d in DIVIDERS)
        for j in range(n):
            spans.append((start, i + 1) if j == 0 else (i + 1, i + 1))
            start = i + 1
    if start < len(labels) or not spans:
        spans.append((start, len(labels)))
    return spans


def prf(tp, np_, ng):
    p = tp / np_ if np_ else 0.0
    r = tp / ng if ng else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


class Acc:
    def __init__(self):
        self.tok = [0, 0, 0]
        self.uas = [0, 0, 0]
        self.las = [0, 0, 0]
        self.pos = [0, 0]
        self.lemma = [0, 0]

    def add(self, gold, pred):
        """gold/pred: list of (span, head_span_or_None, dep, pos, lemma)."""
        g = {t[0]: t for t in gold if t[0][1] > t[0][0]}
        p = {t[0]: t for t in pred if t[0][1] > t[0][0]}
        self.tok[0] += len(set(g) & set(p)); self.tok[1] += len(p); self.tok[2] += len(g)
        self.uas[1] += len(p); self.uas[2] += len(g)
        self.las[1] += len(p); self.las[2] += len(g)
        for span, gt in g.items():
            pt = p.get(span)
            if pt is None:
                continue
            self.pos[1] += 1; self.lemma[1] += 1
            self.pos[0] += (gt[3] == pt[3])
            self.lemma[0] += (gt[4] == pt[4])
            if gt[1] == pt[1]:
                self.uas[0] += 1
                if gt[2] == pt[2]:
                    self.las[0] += 1

    def report(self, label):
        print(f"--- {label} ---")
        for name, v in (("tokens", self.tok), ("UAS", self.uas), ("LAS", self.las)):
            print("  %-7s P %.4f  R %.4f  F %.4f" % (name, *prf(*v)))
        print(f"  pos_acc   {self.pos[0] / max(1, self.pos[1]):.4f}  "
              f"lemma_acc {self.lemma[0] / max(1, self.lemma[1]):.4f}  "
              f"(over {self.pos[1]} matched tokens)")
        return prf(*self.las)[2]


def doc_tuples(doc, spans):
    """Attach saṃhitā spans to a parsed doc -> the tuples the accumulator compares."""
    if len(spans) != len(doc):
        return None
    idx = {t.i: s for t, s in zip(doc, spans)}
    out = []
    for t in doc:
        head = None if t.head.i == t.i else idx[t.head.i]
        out.append((spans[t.i], head, t.dep_, t.pos_, t.lemma_))
    return out


def ref_tuples(ref, spans):
    if len(spans) != len(ref):
        return None
    idx = {t.i: s for t, s in zip(ref, spans)}
    out = []
    for t in ref:
        head = None if t.head.i == t.i else idx[t.head.i]
        out.append((spans[t.i], head, t.dep_, t.pos_, t.lemma_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("presegment")
    ap.add_argument("pairs")
    ap.add_argument("--test-spacy", default=TEST_SPACY)
    ap.add_argument("--test-conllu", default=TEST_CONLLU)
    a = ap.parse_args()
    globals()["TEST_SPACY"] = a.test_spacy
    globals()["TEST_CONLLU"] = a.test_conllu

    nlp = spacy.load(a.model)
    # BOTH scored conditions hand the pipeline a CSL string that this script built itself — the
    # oracle from the gold labels, the real one from the presegmenter's predicted labels. A shipped
    # v3 model, though, CSLises whatever it is given, so it would run the CSLiser a SECOND time over
    # text that is already CSL. That double pass is not a small error: it re-segmented `hāstidantaṃ`
    # to `hāsti dantam` and emitted bare `-` tokens, dragging the ORACLE (which is exact by
    # definition) down to token F 0.6639. Disabling stage 0 leaves the de-CSLizer and the trained
    # de-sandhifier, which is exactly the contract these conditions assume.
    if getattr(nlp.tokenizer, "cslise", False):
        nlp.tokenizer.cslise = False
        nlp.tokenizer.csliser = None
        print("  (CSLiser stage disabled: this script supplies CSL directly)")
    rows = {r["sent_id"]: r for r in
            (json.loads(line) for line in open(a.pairs, encoding="utf-8"))}
    # --test-conllu / --test-spacy were parsed but IGNORED here, so the floor was always scored
    # against the module defaults — i.e. the unrelabelled corpus. On a relabelled arm that reads as
    # a catastrophic LAS drop which is purely a label mismatch.
    sids = [line.split("=", 1)[1].strip() for line in open(a.test_conllu, encoding="utf-8")
            if line.startswith("# sent_id")]
    # NormCorpus, not CompoundCorpus: for an arm whose NORM is the padapāṭha, the floor must carry
    # it, or the analyser channel is silent in THIS condition only and the floor is understated.
    # The reader falls back to Compound-only behaviour when NORM equals lower(ORTH) anyway.
    gp = list(NormCorpus(a.test_spacy, gold_preproc=True)(nlp))
    assert len(gp) == len(sids), "corpus / conllu sentence counts differ"
    by_sid = dict(zip(sids, gp))
    order = [s for s in sids if s in rows]
    print(f"scoring {len(order)} sentences (no elided `_`)\n")

    seg = Presegmenter.from_disk(a.presegment)
    pred_labels = seg.predict([rows[s]["samhita"] for s in order])

    # floor: the standard gold-preproc score, for reference
    floor = nlp.evaluate([by_sid[s] for s in order])
    print("--- gold-preproc (floor, spaCy scorer) ---")
    for k in ("tag_acc", "pos_acc", "morph_acc", "lemma_acc", "dep_uas", "dep_las"):
        if isinstance(floor.get(k), float):
            print(f"  {k:10s} {floor[k]:.4f}")

    oracle, real = Acc(), Acc()
    bad_gold = bad_pred = 0
    for s in order:
        row, ref = rows[s], by_sid[s].reference
        gspans = token_spans(row["labels"], row["samhita"])
        gold = ref_tuples(ref, gspans)
        if gold is None:                       # gold labels must reproduce the gold tokenisation
            bad_gold += 1
            continue
        for acc, labels in ((oracle, row["labels"]), (real, pred_labels[order.index(s)])):
            csl = apply_labels(row["samhita"], labels)
            doc = nlp(csl)
            tup = doc_tuples(doc, token_spans(labels, row["samhita"]))
            if tup is None:
                if acc is real:
                    bad_pred += 1
                # token count disagrees with the label-derived spans: count every gold token as
                # missed rather than silently dropping the sentence
                acc.tok[2] += len(gold); acc.uas[2] += len(gold); acc.las[2] += len(gold)
                acc.tok[1] += len(doc); acc.uas[1] += len(doc); acc.las[1] += len(doc)
                continue
            acc.add(gold, tup)
    if bad_gold:
        print(f"\n  NOTE {bad_gold} sentences dropped: gold labels did not reproduce gold tokens")
    print()
    o = oracle.report("gold CSL (oracle ceiling), span-matched")
    r = real.report("PREDICTED CSL from saṃhitā (end to end), span-matched")
    print(f"\n--- summary (span-matched LAS F) ---")
    print(f"  gold-preproc  {floor['dep_las']:.4f}   (spaCy scorer, gold tokens)")
    print(f"  gold CSL      {o:.4f}   (oracle)")
    print(f"  predicted CSL {r:.4f}   ({r - o:+.4f} vs oracle)")


if __name__ == "__main__":
    main()
