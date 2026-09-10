#!/usr/bin/env python
"""Score the Provo eye-tracking corpus with the incremental left-corner parser and merge the
per-word predictors onto the reading measures.

ALIGNMENT. Provo's word units are whitespace-delimited and keep their punctuation attached
("chased,"); the parser's are spaCy's, which split it off. Each Provo word therefore covers one or
more model tokens. Surprisal is SUMMED over a word's tokens -- it is a log probability, so summing
is the only aggregation that keeps it a probability of the word -- while the memory predictors are
read at the LAST token, which is the state the reader is in when the word has been read.

SENTENCE BOUNDARIES COME FROM PROVO (`Sentence_Number`), not from a sentenciser. Letting a model
segment its own input would make the surprisal of the first word of each sentence depend on a
decision the reader never made, and Provo already tells us where the sentences are.

⚠ THE BASELINE CONTROLS ARE NOT OPTIONAL. Surprisal correlates with word length and frequency, and
a model that predicts reading times without controlling for them has predicted word length. Length,
log frequency and cloze predictability all go in the baseline, and the number that matters is the
IMPROVEMENT over that baseline, never the raw correlation.

⚠ SPILLOVER. Reading time on word t is driven partly by word t-1. The lagged predictors are emitted
alongside the current ones for exactly this reason; a fit without them systematically understates
every effect here.
"""
import argparse
import collections
import math
import sys
import warnings

import numpy as np
import pandas as pd
import spacy
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lc_features import Vocab
from lc_model import LCModel
from lc_surprisal import parse_sentence

warnings.filterwarnings("ignore")

MEASURES = {"IA_FIRST_FIXATION_DURATION": "first_fix",
            "IA_FIRST_RUN_DWELL_TIME": "gaze",
            "IA_REGRESSION_PATH_DURATION": "go_past",
            "IA_DWELL_TIME": "total"}
PRED = ["surprisal", "surprisal_lex", "depth", "depth_viterbi", "open_slots", "d_depth",
        "integ", "beam_entropy"]


PUNCT = ".,;:!?\"'()[]-\u2018\u2019\u201c\u201d\u2014"


def norm_word(w):
    """⚠ The frequency table and its lookups MUST normalise identically. Provo's word units keep
    their punctuation attached ("chased,"), the frequency corpus's do not, and a mismatch here
    silently gives every such word the unseen-word floor -- weakening the baseline for about a
    fifth of the data and so INFLATING whatever surprisal appears to add over it. That is a defect
    no model diagnostic reports: the fit still converges and every coefficient still has a t."""
    return str(w).strip(PUNCT).lower()


def load_freq(path):
    """log unigram frequency per million, from the self-training text."""
    freq = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            freq.update(norm_word(w) for w in line.split())
    freq.pop("", None)
    total = sum(freq.values())
    return {w: math.log10(1e6 * c / total) for w, c in freq.items()}, total


def score_sentences(model, vocab, tokenizer, sents, beam, expand):
    """sents: {(text_id, sent_no): [(word_number, word), ...]}. Yields aligned per-word rows."""
    with torch.no_grad():
        for (text_id, sent_no), words in sents.items():
            nums = [n for n, _ in words]
            text = " ".join(w for _, w in words)
            # Char span of each Provo word in the joined text, so alignment is exact rather than
            # a string search that could match the wrong occurrence of a repeated word.
            spans, at = [], 0
            for _, w in words:
                spans.append((at, at + len(w)))
                at += len(w) + 1
            doc = tokenizer(text)
            toks = [t for t in doc if not t.is_space]
            if not toks:
                continue
            owner = []
            for t in toks:
                k = next((i for i, (a, b) in enumerate(spans) if a <= t.idx < b), None)
                owner.append(k if k is not None else len(spans) - 1)
            rows, _ = parse_sentence(model, vocab, [t.text for t in toks] + ["<ROOT>"],
                                     beam, expand)
            if rows is None or len(rows) != len(toks):
                continue
            by_word = collections.defaultdict(list)
            for r, k in zip(rows, owner):
                by_word[k].append(r)
            for k in range(len(words)):
                rs = by_word.get(k)
                if not rs:
                    continue
                out = {"Text_ID": text_id, "Sentence_Number": sent_no, "Word_Number": nums[k],
                       "Word_In_Sentence_Number": k + 1, "n_subtokens": len(rs)}
                out["surprisal"] = sum(r["surprisal"] for r in rs)
                out["surprisal_lex"] = sum(r["surprisal_lex"] for r in rs)
                for key in PRED[2:]:
                    out[key] = rs[-1][key]
                yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eyetracking", required=True)
    ap.add_argument("--tokenizer", default="training_en_gum_ext/model-best")
    ap.add_argument("--freq-text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--expand", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(8)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    vocab = Vocab(ck["words"], ck["labels"], ck.get("chars"))
    model = LCModel(len(vocab.words), len(vocab.labels),
                    len(vocab.chars) if ck.get("chars") else 0)
    if ck.get("chars") and len(vocab.chars) > 3:
        model.set_char_table(vocab)      # buffer shape must match before load_state_dict
    model.load_state_dict(ck["model"])
    model.eval()
    tokenizer = spacy.load(args.tokenizer, exclude=["tok2vec", "tagger", "parser", "morphologizer",
                                                    "lemmatizer"]).tokenizer

    keep = (["Participant_ID", "Text_ID", "Word_Number", "Sentence_Number",
             "Word_In_Sentence_Number", "Word", "Word_Cleaned", "Word_Length",
             "OrthographicMatch", "IA_SKIP"] + list(MEASURES))
    et = pd.read_csv(args.eyetracking, usecols=keep, low_memory=False)
    et = et.rename(columns=MEASURES)
    print("eye-tracking rows %d, participants %d, texts %d"
          % (len(et), et.Participant_ID.nunique(), et.Text_ID.nunique()), file=sys.stderr)

    words = (et[["Text_ID", "Sentence_Number", "Word_Number", "Word"]]
             .drop_duplicates(["Text_ID", "Word_Number"])
             .sort_values(["Text_ID", "Word_Number"]))
    sents = collections.OrderedDict()
    for r in words.itertuples():
        if pd.isna(r.Word) or pd.isna(r.Sentence_Number):
            continue
        sents.setdefault((int(r.Text_ID), int(r.Sentence_Number)), []).append(
            (int(r.Word_Number), str(r.Word)))
    print("sentences to score: %d" % len(sents), file=sys.stderr)

    pred = pd.DataFrame(list(score_sentences(model, vocab, tokenizer, sents,
                                             args.beam, args.expand)))
    print("scored words: %d of %d" % (len(pred), len(words)), file=sys.stderr)

    freq, total = load_freq(args.freq_text)
    pred["log_freq"] = [freq.get(norm_word(w), math.log10(1e6 * 0.5 / total))
                        for w in words.set_index(["Text_ID", "Word_Number"])
                        .loc[list(zip(pred.Text_ID, pred.Word_Number)), "Word"]]

    # Lagged predictors within a sentence: spillover is the rule in reading, not the exception.
    pred = pred.sort_values(["Text_ID", "Word_Number"])
    for c in PRED + ["log_freq"]:
        pred["prev_" + c] = pred.groupby(["Text_ID", "Sentence_Number"])[c].shift(1)

    merged = et.merge(pred, on=["Text_ID", "Word_Number", "Sentence_Number",
                                "Word_In_Sentence_Number"], how="inner")
    merged.to_csv(args.out, sep="\t", index=False)
    print("wrote %s: %d rows, %d columns" % (args.out, len(merged), merged.shape[1]),
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
