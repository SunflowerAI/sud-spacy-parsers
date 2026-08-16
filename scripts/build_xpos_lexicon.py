#!/usr/bin/env python3
"""Build the per-form XPOS-FIELD lexicon `sud.LexFieldEmbed.v1` reads.

⚠ MEASURED OUTCOME (2026-08-16): THIS CHANNEL CANNOT WORK, and the reason is an identity rather
than a measurement. A per-form majority table is a deterministic function of the form, so it adds
EXACTLY 0.0000 bits about the dependency relation beyond what `NORM` already gives the parser
(gold XPOS adds 0.2222 bits, a predicted tagger 0.1475). It scored -0.19 LAS against its capacity
control. The accuracy table below is real and irrelevant: a channel can be accurate and carry
nothing. Kept because the machinery is reused and because the trap is worth being able to point at.
See NEGATIVE-RESULTS.md, "XPOS as a parser input, and kanripo vectors, for lzh".

WHY THIS EXISTS. lzh's XPOS is a four-level comma-separated ontology (`v,動詞,行為,設置`): a coarse
letter, a 品詞, and two semantic classes. As one symbol it is 118 types; decomposed it is 4 / 12 /
44 / 84 values, and the decomposition is worth far more than the tag on exactly the tokens the
parser is worst at. Measured on the released traditional-only split, majority-per-form harvested
from train, scored against test:

    form freq in train   test tokens   full 118-way tag    field1   field2   field3   field4
    unseen                       392             0.00 %    0.0 %    0.0 %    0.0 %    0.0 %
    1-2                          359            76.04 %   88.6 %   88.3 %   82.2 %   76.9 %
    3-10                         678            79.06 %   91.2 %   90.7 %   83.9 %   79.8 %
    11-50                       2096            76.96 %   85.9 %   85.2 %   80.0 %   77.5 %
    >50                        30708            87.77 %   92.3 %   90.8 %   89.3 %   89.1 %

A form seen ONCE OR TWICE has its coarse field right 88.6 % of the time against 76.0 % for the whole
tag -- +12.6 points, in the bucket where the parser's `NORM` embedding is worst estimated. That is
the argument for hashing the fields separately, and it is why field 4 (84 values, and the first to
degrade) is not in the default channel set.

There is capacity to spend on it. On **81.96 %** of lzh tokens `NORM == PREFIX == SUFFIX` -- one Han
character -- and `SHAPE` is `'x'` on 78 %, so three of the base embed's four tables already hold
three hashes of the same string and the fourth is little more than token length.

⚠ JACKKNIFING IS NOT OPTIONAL, and it is the whole reason this file writes K+1 tables. Harvested
from train and applied to train, a form seen ONCE gets a value that is that very token's own gold
field -- 100 % correct at training time against 88.6 % at inference. The parser would learn to trust
a channel that is cleaner than the one it will be given. So each token is also assigned a value
computed from the occurrences OUTSIDE its own fold, and `LexFieldEmbed` reads the fold table while
`is_train` and the full table at inference. The asymmetry then runs the SAFE way: the model is
trained on a noisier channel than it is served.

**The fold key is the concatenated FORMS of the sentence, not its text.** `Corpus` with
`gold_preproc = true` hands the layer one sentence at a time and `Span.text` drops the final token's
trailing whitespace, so a text-keyed fold would agree here (lzh has no spaces) and diverge silently
on any language that does. The key is recorded in the table as `fold_key` and `check_lex_embed.py`
verifies the builder and the layer agree on every training sentence -- per CLAUDE.md, ask the
artefact what regime it was built for rather than assume it.

Usage:

    .venv/bin/python scripts/build_xpos_lexicon.py \
        assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-train.<suffix>.conllu \
        --out models/lzh_xpos_lex.json --k 5
"""
import argparse
import collections
import json
import pathlib
import sys

from spacy.strings import hash_string

FOLD_KEY = "forms"


def read_sentences(path):
    """Yield [(form, xpos), ...] per sentence. Multiword-token and empty-node rows are skipped,
    exactly as `spacy convert --converter conllu` skips them, so the form inventory matches the
    corpus the layer will see."""
    sent = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if sent:
                yield sent
            sent = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5 or "-" in cols[0] or "." in cols[0]:
            continue
        sent.append((cols[1], cols[4]))
    if sent:
        yield sent


def fold_of(forms, k):
    """The fold a sentence belongs to. `hash_string` is spaCy's murmur hash -- stable across
    processes and vocabs, unlike Python's `hash`, which is seeded per interpreter and would put a
    sentence in a different fold on every run."""
    return hash_string("".join(forms)) % k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conllu", help="TRAINING CoNLL-U only -- harvesting from dev or test leaks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5, help="jackknife folds (0 disables jackknifing)")
    ap.add_argument("--n-fields", type=int, default=4,
                    help="how many comma-separated XPOS fields to expect")
    ap.add_argument("--whole", action="store_true",
                    help="do NOT split on commas: one field holding the whole 118-way tag. This "
                         "is the comparison arm for the decomposition -- the lexicon accuracies "
                         "say the fields beat the tag by 12.6 points on rare forms, and this "
                         "makes that claim testable by training rather than by table alone.")
    ap.add_argument("--min-count", type=int, default=1,
                    help="a form needs this many occurrences to enter the table at all")
    a = ap.parse_args()
    if a.whole:
        a.n_fields = 1

    sents = list(read_sentences(a.conllu))
    if not sents:
        sys.exit(f"no sentences read from {a.conllu}")

    # counts[field][form][value], and the same per fold, so a fold table is a plain subtraction.
    counts = [collections.defaultdict(collections.Counter) for _ in range(a.n_fields)]
    fold_counts = [[collections.defaultdict(collections.Counter) for _ in range(a.n_fields)]
                   for _ in range(max(a.k, 1))]
    ragged = 0
    ntok = 0
    for sent in sents:
        forms = [f for f, _ in sent]
        kf = fold_of(forms, a.k) if a.k else 0
        for form, xpos in sent:
            ntok += 1
            parts = [xpos] if a.whole else xpos.split(",")
            if len(parts) != a.n_fields:
                ragged += 1
                continue
            for i, v in enumerate(parts):
                counts[i][form][v] += 1
                fold_counts[kf][i][form][v] += 1

    if ragged:
        print(f"  ⚠ {ragged}/{ntok} tokens had != {a.n_fields} XPOS fields and were skipped")

    # Per-field value vocabularies. Sorted so the file is reproducible and a diff is readable.
    vocab = [sorted(set().union(*(set(c) for c in counts[i].values())) if counts[i] else set())
             for i in range(a.n_fields)]
    index = [{v: j for j, v in enumerate(vocab[i])} for i in range(a.n_fields)]

    def majority(counters, form, i):
        c = counters[i].get(form)
        if not c or sum(c.values()) < a.min_count:
            return -1
        # (count, value) so ties break on the value string rather than on dict order -- the table
        # has to be reproducible or a rebuild silently reshuffles what the parser was trained on.
        return index[i][max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]]

    forms = sorted(counts[0]) if counts[0] else []
    full = {f: [majority(counts, f, i) for i in range(a.n_fields)] for f in forms}

    folds = []
    for kf in range(a.k):
        # Everything EXCEPT fold kf: subtract this fold's counts from the totals.
        held = [collections.defaultdict(collections.Counter) for _ in range(a.n_fields)]
        for i in range(a.n_fields):
            for form, c in counts[i].items():
                rest = c - fold_counts[kf][i].get(form, collections.Counter())
                if rest:
                    held[i][form] = rest
        # Store only what DIFFERS from the full table; for most forms the majority is unchanged.
        diff = {}
        for f in forms:
            v = [majority(held, f, i) for i in range(a.n_fields)]
            if v != full[f]:
                diff[f] = v
        folds.append(diff)

    out = {
        "source": str(a.conllu),
        "n_fields": a.n_fields,
        "k": a.k,
        "fold_key": FOLD_KEY,
        "min_count": a.min_count,
        "vocab": vocab,
        "full": full,
        "folds": folds,
    }
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"  {len(sents)} sentences, {ntok} tokens, {len(forms)} form types -> {p} "
          f"({p.stat().st_size / 1024:.0f} KB)")
    for i in range(a.n_fields):
        print(f"    field {i}: {len(vocab[i])} values")
    if a.k:
        # How hard the jackknifing bites. If this is ~0 the channel is leaking and the ablation
        # would read as a win that inference cannot reproduce.
        for kf, d in enumerate(folds):
            oov = sum(1 for v in d.values() if v[0] == -1)
            print(f"    fold {kf}: {len(d)} forms differ from the full table ({oov} become OOV)")


if __name__ == "__main__":
    main()
