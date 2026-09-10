#!/usr/bin/env python
"""Parse plain English text with a local SUD arm and emit silver CoNLL-U for self-training.

⚠ THE ARM IS NAMED, NOT GUESSED (CLAUDE.md hazard 2). `--model` takes a path and there is no
default that "looks right": `training_en_gum_sud` and `training_en_sud` differ by the whole GUM
half of the training data, and a silver corpus built from the wrong one is indistinguishable from
the right one downstream -- it loads, converts and trains exactly the same.

⚠ ONLY THE PARSER'S COLUMNS ARE TRUSTWORTHY HERE. The morphologiser and lemmatiser are switched off
rather than written out as `_`-free columns, because the left-corner model reads FORM, HEAD and
DEPREL and nothing else, and a half-populated FEATS column is the kind of thing that later gets
mistaken for gold. UPOS and FEATS are written as `_` deliberately.

⚠ SENTENCE BOUNDARIES COME FROM THE PARSER on these arms, so a paragraph is handed over whole and
split by the model. That is the same regime the arm was trained under (`sud.GoldTokCorpus.v1`
yields multi-sentence docs), but it means a paragraph the parser mis-segments produces silver
sentences that are wrong as WHOLES, not just in one arc. The length filter below drops the worst of
them; it is a filter, not a fix.
"""
import argparse
import sys
import warnings

import spacy

warnings.filterwarnings("ignore")


def projective(heads):
    arcs = [(min(h, d), max(h, d)) for d, h in heads.items() if h]
    for a, b in arcs:
        for c, d in arcs:
            if a < c < b < d or c < a < d < b:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, "scripts")
    nlp = spacy.load(args.model, exclude=["morphologizer", "lemmatizer", "tagger"])
    print("pipeline:", nlp.pipe_names, file=sys.stderr)

    paras = (line.strip() for line in open(args.text, encoding="utf-8") if line.strip())
    kept = seen = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for doc in nlp.pipe(paras, batch_size=args.batch_size):
            for sent in doc.sents:
                seen += 1
                toks = [t for t in sent]
                if not (args.min_len <= len(toks) <= args.max_len):
                    continue
                base = toks[0].i
                heads, rows, ok = {}, [], True
                for k, t in enumerate(toks, 1):
                    h = 0 if t.head.i == t.i else t.head.i - base + 1
                    if not (0 <= h <= len(toks)):
                        ok = False
                        break
                    heads[k] = h
                    # spaCy names the sentence root ROOT; SUD's own column says root. The two arms
                    # otherwise share their label inventory, and this is the only rename needed --
                    # which is why `training_en_gum_ext` is the arm to use and not
                    # `training_en_gum_sud`, whose labels predate the extended-scope relabelling.
                    rows.append((k, t.text, h, "root" if h == 0 else (t.dep_ or "dep")))
                if not ok or sum(1 for h in heads.values() if h == 0) != 1:
                    continue
                if not projective(heads):
                    continue
                fh.write("# text = %s\n" % sent.text.replace("\n", " "))
                for k, form, h, dep in rows:
                    fh.write("%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_\n" % (k, form, h, dep))
                fh.write("\n")
                kept += 1
                if kept % 20000 == 0:
                    print("  %d sentences kept of %d seen" % (kept, seen), file=sys.stderr,
                          flush=True)
                if args.max_sents and kept >= args.max_sents:
                    print("kept %d of %d sentences -> %s" % (kept, seen, args.out), file=sys.stderr)
                    return 0
    print("kept %d of %d sentences -> %s" % (kept, seen, args.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
