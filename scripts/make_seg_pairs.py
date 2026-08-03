#!/usr/bin/env python3
"""Build character-tagger training pairs for word segmentation, from any SUD treebank.

Reuses the representation `make_samhita_pairs.py` invented for Sanskrit — per-character rewrite
labels over a raw string — so `sa_presegment.Presegmenter`, `train_samhita.py` and `eval_samhita.py`
all work unchanged. Only two of the 59 Sanskrit labels are needed here, because word segmentation
has no coalescence to undo:

    '='    keep this character, no boundary after it
    '= '   keep this character, word boundary after it

**Rows are per WHITESPACE CHUNK, not per sentence.** `Presegmenter.to_csl` runs the model chunk by
chunk (the space character is deliberately absent from its vocabulary), so the training data must
match that granularity or the model meets UNK at every boundary at inference. This also makes one
script serve two different jobs:

    zh   a sentence has no spaces at all, so the chunk IS the sentence and the model must find
         every boundary  (pkuseg currently scores strict token F 0.837 here)
    id   the chunk is an orthographic word and the model only has to decide whether it splits —
         e.g. the `-nya`/`-lah` enclitics that `coarsen_id.py` currently merges away

Tokens whose concatenation does not reproduce the chunk are dropped with a count, never silently:
that means the treebank's FORMs are not a segmentation of its own text (elided tokens, normalised
punctuation), and such a row cannot teach a boundary decision.

    make_seg_pairs.py TREEBANK.conllu OUT.jsonl [--min-chunk 2]
"""
import argparse
import json
import pathlib

KEEP, BREAK = "=", "= "


def read_sentences(path):
    """Yield [FORM, ...] per sentence, skipping multiword-range lines."""
    cur = []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur:
                yield cur
            cur = []
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 10 or "-" in c[0] or "." in c[0]:
            continue
        cur.append(c[1])
    if cur:
        yield cur


def chunks(tokens, space_after):
    """Group tokens into whitespace-delimited chunks using SpaceAfter=No."""
    out, cur = [], []
    for tok, sp in zip(tokens, space_after):
        cur.append(tok)
        if sp:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def read_with_spacing(path):
    """Yield (tokens, space_after) per sentence. space_after[i] is False iff SpaceAfter=No.

    MULTIWORD TOKENS are the whole point for Indonesian and must be handled explicitly. In CoNLL-U a
    range line (`16-17  penghuninya`) carries the ORTHOGRAPHIC word and its spacing, while the
    sub-token lines (`16 penghuni`, `17 nya`) carry the syntactic words and normally have NO
    `SpaceAfter=No`. Reading spacing off the sub-tokens therefore makes `penghuni` and `nya` look
    like two separate whitespace words — which is exactly the bug that made the first id segmenter
    learn punctuation splitting only and never see a single enclitic junction (1362 such chunks).
    The members of a range are bound together: every one but the last takes `SpaceAfter=No`, and the
    last inherits the range line's own spacing.
    """
    toks, sp = [], []
    pending_end, pending_space = None, True
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            if toks:
                yield toks, sp
            toks, sp = [], []
            pending_end = None
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 10 or "." in c[0]:
            continue
        if "-" in c[0]:                                   # range line: remember its extent+spacing
            a, b = c[0].split("-")
            if a.isdigit() and b.isdigit():
                pending_end = int(b)
                pending_space = "SpaceAfter=No" not in c[9]
            continue
        idx = int(c[0])
        toks.append(c[1])
        if pending_end is not None and idx <= pending_end:
            sp.append(pending_space if idx == pending_end else False)
            if idx == pending_end:
                pending_end = None
        else:
            sp.append("SpaceAfter=No" not in c[9])
    if toks:
        yield toks, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conllu")
    ap.add_argument("out")
    ap.add_argument("--min-chunk", type=int, default=1,
                    help="skip chunks shorter than this many characters (nothing to decide)")
    a = ap.parse_args()

    rows, dropped, n_split = [], 0, 0
    for si, (toks, sp) in enumerate(read_with_spacing(a.conllu)):
        for ci, group in enumerate(chunks(toks, sp)):
            surface = "".join(group)
            if len(surface) < a.min_chunk or not surface.strip():
                continue
            if any(not t for t in group):
                dropped += 1
                continue
            labels = []
            for j, t in enumerate(group):
                labels.extend([KEEP] * (len(t) - 1))
                labels.append(BREAK if j < len(group) - 1 else KEEP)
            if len(labels) != len(surface):
                dropped += 1
                continue
            rows.append({"sent_id": f"{si}_{ci}", "samhita": surface,
                         "csl": " ".join(group), "labels": labels})
            if len(group) > 1:
                n_split += 1

    with pathlib.Path(a.out).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = sum(len(r["samhita"]) for r in rows)
    breaks = sum(lb == BREAK for r in rows for lb in r["labels"])
    print(f"  {a.out}")
    print(f"    chunks {len(rows)}  characters {chars}  boundaries {breaks} "
          f"({breaks / max(chars, 1):.1%} of characters)")
    print(f"    chunks that split: {n_split} ({n_split / max(len(rows), 1):.1%})"
          + (f"   dropped {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
