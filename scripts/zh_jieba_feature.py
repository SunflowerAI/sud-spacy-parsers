#!/usr/bin/env python3
"""jieba's SEGMENTATION DECISION as an input channel for the character segmenter.

Distinct from the jieba channel already tried and rejected. That one asked "is this substring in
jieba's 349 044-entry dictionary" — plain set membership, and it lost to the jackknifed corpus
lexicon (0.8859 vs 0.8902). This one asks jieba to actually SEGMENT the chunk and reads off its
verdict per character. That is the output of a DAG search over the dictionary plus an HMM over
unknown words, so it is a strictly stronger signal than a substring lookup, and it is the piece the
earlier experiment left on the table.

**Why it should help, measured before building it** (SUD_Chinese-GSD test, against the shipped
`models/zh_seg_jk`):

    jieba boundaries     P 0.9730   R 0.8793      -- very precise, and INCOMPLETE: it under-splits
    our model            P 0.9507   R 0.9556

    per character position:  both right 86.89 %   ours only 7.33 %
                             JIEBA ONLY RIGHT 4.19 %   both wrong 1.59 %

So jieba is right at **72.5 %** of the 1079 positions the shipped model gets wrong. That is the
headroom, and it is reachable only by a feature: 97 % of this segmenter's errors are CONFIDENT
errors (margin >= 0.10), which no decoder — beam, lexicon repair, transition model — can touch.
A crude hard union of the two break sets already scores 0.8938 against 0.8902 without any training.

**The code is BMES**, the standard Chinese-segmentation tagset, 4 values so the channel is exactly
the same width as a lexicon channel and the capacity control stays parameter-identical:

    0 B  first character of a jieba word     2 E  last character  (=> jieba breaks after this one)
    1 M  interior character                  3 S  single-character jieba word (also a break after)

Encoding the whole tag rather than just "break after here" costs nothing and tells the model where a
jieba word STARTS as well as where it ends — the two facts the model needs at a junction.

**jieba is EXTERNAL, so this channel needs no jackknifing.** Its dictionary was not harvested from
our training split, so train-time reliability equals test-time reliability, and the leak that made
the corpus lexicon useless without jackknifing cannot arise. A userdict harvested from our own train
split WOULD reintroduce it — see `force_split_dict`.
"""
import pathlib

B, M, E, S = 0, 1, 2, 3
N_JIEBA = 4

_STATE = {"tok": None}


def set_userdict(words=()):
    """Reset jieba to its stock dictionary, then force-split every word in `words`.

    `del_word(w)` sets w's frequency to 0, which registers a hard force-split in `finalseg`; that is
    the lever that matters here, because jieba's error against GSD is under-splitting (`这个`,
    `有人`, `为什么` are single jieba words and two GSD tokens each). `add_word` is the wrong
    direction — measured, it costs 0.8 F.

    Resetting first is what makes the per-fold jackknifing below honest: without it each fold's
    force-splits would accumulate on top of the previous fold's, so fold 4 would be segmenting with
    a dictionary derived from all five folds including its own.
    """
    import jieba
    import jieba.finalseg
    jieba.dt.FREQ.clear()
    jieba.dt.initialized = False
    jieba.finalseg.Force_Split_Words.clear()
    jieba.initialize()
    for w in words:
        jieba.del_word(w)
    _STATE["tok"] = jieba.dt
    return jieba.dt


def get_tokenizer(userdict=None):
    """jieba's global Tokenizer, optionally with a force-split userdict file applied."""
    if _STATE["tok"] is not None and not userdict:
        return _STATE["tok"]
    words = ()
    if userdict and userdict != "auto":
        words = [w for w in pathlib.Path(userdict).read_text(encoding="utf-8").split("\n") if w]
    return set_userdict(words)


def jieba_codes(text, tok=None):
    """Per character of `text`, its BMES tag under jieba's segmentation."""
    tok = tok or get_tokenizer()
    out = []
    for w in tok.cut(text, HMM=True):
        n = len(w)
        if n == 1:
            out.append(S)
        else:
            out.extend([B] + [M] * (n - 2) + [E])
    if len(out) != len(text):                 # jieba never drops characters, but never guess
        out = (out + [S] * len(text))[:len(text)]
    return out


def force_split_dict(train_rows, min_wrong=2):
    """Harvest the jieba words a treebank USUALLY splits, from `csl`/`samhita` training rows.

    Derived from gold, so a channel built on it is only honest under the same K-fold jackknifing the
    corpus lexicon needs. Measured standalone on GSD (harvested on train, scored on test), it lifts
    jieba itself from token F 0.7989 to 0.8570 and halves the positions where jieba is wrong and the
    model is right (1369 -> 958) while keeping every position where jieba rescues the model.
    """
    from collections import Counter
    tok = set_userdict()                      # harvest against STOCK jieba, never a primed one
    hit, miss = Counter(), Counter()
    for r in train_rows:
        gold, i = set(), 0
        for w in r["csl"].split(" "):
            gold.add((i, i + len(w)))
            i += len(w)
        i = 0
        for w in tok.cut(r["samhita"], HMM=True):
            (hit if (i, i + len(w)) in gold else miss)[w] += 1
            i += len(w)
    return [w for w in set(hit) | set(miss)
            if len(w) > 1 and miss[w] >= min_wrong and miss[w] > hit[w]]
