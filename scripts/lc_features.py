#!/usr/bin/env python
"""Turn CoNLL-U into the per-step training tensors for the incremental left-corner parser.

WHY THE MODEL IS GENERATIVE, NOT DISCRIMINATIVE. Reading-time work needs a prefix probability,
P(w_t | w_1..t-1), and only a model that generates the words has one. A discriminative parser scores
actions given the sentence, so its scores never normalise over what the NEXT WORD could have been,
and no amount of beam search turns them into surprisal. This is also why spaCy's beam is no use
here even setting the transition system aside (`NEGATIVE-RESULTS.md` records that beam refuted its
own premise on Latin -- but that was a beam bought for ACCURACY, which is a different purchase).

WHY NO WORD-SYNCHRONOUS BEAM MACHINERY IS NEEDED. In an RNNG the number of structural actions
between two words is unbounded, so a plain beam compares hypotheses that have consumed different
numbers of words and Stern et al. (2017) have to re-synchronise it. Here shift-type and reduce-type
actions STRICTLY ALTERNATE, so there is exactly ONE reduce action between consecutive words and
every hypothesis is at the same word at the same time, for free. Each word costs exactly two
decisions, which is what makes the surprisal in `lc_surprisal.py` a sum over a well-defined frontier.

EACH STEP'S FEATURES are the parser's own configuration, never the future: the history of what has
been emitted, plus the top two spine elements. Everything read off the stack is a token already
emitted, so the featuriser is usable unchanged at decode time -- a property worth stating because
getting it wrong is the classic way a generative parser's surprisal comes out too low.
"""
import argparse
import collections
import pickle
import sys

import numpy as np

from lc_transitions import (Config, Underivable, oracle, read_conllu, is_projective,
                            SHIFT, INSERT, LEFT_PRED, RIGHT_PRED, LEFT_COMP, RIGHT_COMP)

SHIFT_TYPES = [SHIFT, INSERT]
REDUCE_NAMES = [LEFT_PRED, RIGHT_PRED, LEFT_COMP, RIGHT_COMP]
PAD, UNK, BOS, DUMMY, NOTOK = 0, 1, 2, 3, 4       # reserved word ids
N_RESERVED = 6                                    # ... plus <root>; real words start here
NO_LABEL = "_"
MAX_LAM, MAX_DEPTH = 3, 7                          # feature bucket ceilings


MAX_WORD_CHARS = 24
C_PAD, C_EOW, C_UNK = 0, 1, 2


class Vocab:
    def __init__(self, words, labels, chars=None):
        self.words = words
        self.labels = labels
        self.chars = chars or ["<cpad>", "<eow>", "<cunk>"]
        self.w2i = {w: i for i, w in enumerate(words)}
        self.l2i = {l: i for i, l in enumerate(labels)}
        self.c2i = {c: i for i, c in enumerate(self.chars)}

    def wid(self, form):
        return self.w2i.get(form.lower(), UNK)

    def cids(self, form):
        """Character ids for the OPEN-VOCABULARY tail, terminated by <eow>.

        ⚠ Must use the same normalisation as `wid` (lower-cased), or the character model is asked
        to spell a different string from the one the word softmax failed to find."""
        cs = [self.c2i.get(c, C_UNK) for c in form.lower()[:MAX_WORD_CHARS]]
        return cs + [C_EOW]

    def __repr__(self):
        return "Vocab(%d words, %d labels, %d chars)" % (
            len(self.words), len(self.labels), len(self.chars))


def build_vocab(gold_paths, silver_paths=(), min_gold=2, min_total=5):
    """ONE vocabulary over gold and silver together, in a fixed order.

    ⚠ The two training stages MUST share this object. A warm start needs the label and word ORDER
    to match position for position, not merely the sets to be equal -- CLAUDE.md hazard 7. Building
    a vocabulary per stage would renumber every id and silently scramble the tied embedding table
    that stage two inherits, with no error and a loss curve that merely looks like a bad seed.

    A word is kept if the GOLD saw it twice, or if gold and silver together saw it `min_total`
    times. Gold-only words are cheap and are exactly the ones the dev and test sets need; silver-
    only words need a higher bar because a parser's own output repeats its own tokenisation errors."""
    gold, silver, labels = collections.Counter(), collections.Counter(), set()
    for path in gold_paths:
        for forms, heads, labs, n in read_conllu(path):
            gold.update(f.lower() for f in forms[:-1])
            labels.update(labs.values())
    for path in silver_paths:
        for forms, heads, labs, n in read_conllu(path):
            silver.update(f.lower() for f in forms[:-1])
            labels.update(labs.values())
    words = ["<pad>", "<unk>", "<bos>", "<dummy>", "<notok>", "<root>"]
    assert len(words) == N_RESERVED
    keep = {w for w, c in gold.items() if c >= min_gold}
    keep |= {w for w in set(gold) | set(silver) if gold[w] + silver[w] >= min_total}
    words += sorted(keep)
    # Characters come from EVERY word seen, kept or not: the character model exists precisely to
    # spell the words the word softmax dropped, so restricting its alphabet to kept words would
    # reintroduce an unknown symbol at exactly the point the design is meant to remove one.
    alphabet = sorted({c for w in list(gold) + list(silver) for c in w[:MAX_WORD_CHARS]})
    return Vocab(words, [NO_LABEL] + sorted(labels), ["<cpad>", "<eow>", "<cunk>"] + alphabet)


def _spine_feats(spine, vocab, wid_of):
    """(root word, node-above-dummy word, incomplete, |lam| bucket, pending label)."""
    if spine is None:
        return (NOTOK, NOTOK, 0, 0, 0)
    root = wid_of(spine.root) if spine.nodes else DUMMY
    above = wid_of(spine.above_dummy) if (spine.incomplete and spine.nodes) else NOTOK
    lam = min(len(spine.lam), MAX_LAM) if spine.incomplete else 0
    lab = vocab.l2i.get(spine.dummy_label or NO_LABEL, 0) if spine.incomplete else 0
    return (root, above, int(spine.incomplete), lam, lab)


def featurise(forms, heads, labs, n, vocab):
    """Run the oracle and record, for every step, the configuration BEFORE the action plus the
    action taken. Returns a dict of int arrays, one row per step."""
    actions, _ = oracle(heads, n, labs)
    wids = [PAD] + [vocab.wid(f) for f in forms[:-1]] + [vocab.w2i["<root>"]]
    wid_of = lambda t: wids[t] if t is not None else NOTOK

    prev, stack_feats, legal_ins, legal_comp = [], [], [], []
    is_shift, tgt_shift, tgt_word, tgt_name, tgt_label, depth = [], [], [], [], [], []
    oov_pos, oov_chars = [], []
    c = Config(n)
    last = BOS
    for name, label in actions:
        top = c.stack[-1] if c.stack else None
        second = c.stack[-2] if len(c.stack) > 1 else None
        prev.append(last)
        stack_feats.append(_spine_feats(top, vocab, wid_of) + _spine_feats(second, vocab, wid_of))
        depth.append(min(len(c.stack), MAX_DEPTH))
        legal_ins.append(int(top is not None and top.incomplete))
        legal_comp.append(int(top is not None and not top.incomplete
                              and second is not None and second.incomplete))
        if name in SHIFT_TYPES:
            is_shift.append(1)
            tgt_shift.append(SHIFT_TYPES.index(name))
            tgt_word.append(wids[c.beta])
            tgt_name.append(-100)
            tgt_label.append(-100)
            if wids[c.beta] == UNK:
                oov_pos.append(len(prev) - 1)
                oov_chars.append(vocab.cids(forms[c.beta - 1]))
            last = wids[c.beta]
        else:
            is_shift.append(0)
            tgt_shift.append(-100)
            tgt_word.append(-100)
            tgt_name.append(REDUCE_NAMES.index(name))
            tgt_label.append(vocab.l2i.get(label or NO_LABEL, 0))
            # Reduce steps feed the history a symbol drawn from a disjoint id space, so the LSTM
            # never confuses "the action LEFT-PRED" with "the word that happens to sit at id 6".
            last = len(vocab.words) + REDUCE_NAMES.index(name) * len(vocab.labels) \
                + vocab.l2i.get(label or NO_LABEL, 0)
        c.apply((name, label))
    if oov_chars:
        width = max(len(c) for c in oov_chars)
        chars = np.zeros((len(oov_chars), width), np.int32)
        for i, cs in enumerate(oov_chars):
            chars[i, :len(cs)] = cs
    else:
        chars = np.zeros((0, 1), np.int32)
    return dict(oov_pos=np.array(oov_pos, np.int32), oov_chars=chars,
                prev=np.array(prev, np.int32),
                stack=np.array(stack_feats, np.int32),
                depth=np.array(depth, np.int32),
                legal_ins=np.array(legal_ins, np.int8),
                legal_comp=np.array(legal_comp, np.int8),
                is_shift=np.array(is_shift, np.int8),
                tgt_shift=np.array(tgt_shift, np.int64),
                tgt_word=np.array(tgt_word, np.int64),
                tgt_name=np.array(tgt_name, np.int64),
                tgt_label=np.array(tgt_label, np.int64))


def prepare(paths, vocab, max_len=100, max_sents=0):
    out, skipped = [], collections.Counter()
    for path in paths:
        for forms, heads, labs, n in read_conllu(path):
            if max_sents and len(out) >= max_sents:
                return out, skipped
            if n > max_len:
                skipped["too long"] += 1
                continue
            if not is_projective(heads, n):
                skipped["non-projective"] += 1
                continue
            try:
                out.append(featurise(forms, heads, labs, n, vocab))
            except (Underivable, AssertionError):
                skipped["underivable"] += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--silver", nargs="*", default=[])
    ap.add_argument("--max-silver", type=int, default=120000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=100)
    args = ap.parse_args()

    vocab = build_vocab(args.train, args.silver, args.min_count)
    print(vocab)
    train, sk_tr = prepare(args.train, vocab, args.max_len)
    dev, sk_dv = prepare(args.dev, vocab, args.max_len)
    silver, sk_sv = prepare(args.silver, vocab, args.max_len, args.max_silver) if args.silver \
        else ([], collections.Counter())
    if silver:
        print("silver %d sentences, %d steps   skipped %s"
              % (len(silver), sum(len(d["prev"]) for d in silver), dict(sk_sv)))
    print("train %d sentences, %d steps   skipped %s"
          % (len(train), sum(len(d["prev"]) for d in train), dict(sk_tr)))
    print("dev   %d sentences, %d steps   skipped %s"
          % (len(dev), sum(len(d["prev"]) for d in dev), dict(sk_dv)))
    oov = sum((d["tgt_word"] == UNK).sum() for d in dev)
    words = sum((d["is_shift"] == 1).sum() for d in dev)
    print("dev OOV rate %.2f %%" % (100.0 * oov / words))
    with open(args.out, "wb") as fh:
        # Plain lists, never the Vocab object: pickling a class defined in a __main__
        # script makes the file unreadable from any other entry point.
        pickle.dump({"words": vocab.words, "labels": vocab.labels, "chars": vocab.chars,
                     "train": train, "dev": dev, "silver": silver}, fh, protocol=4)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    sys.exit(main())
