#!/usr/bin/env python
"""Blank the XPOS column (field 5 -> "_") for the Perseus portion of a merged file.

The three Latin treebanks use mutually-incompatible XPOS tagsets (ITTB's Index
Thomisticus codes, PROIEL's 2-letter codes, Perseus's 9-position morphology tags).
Perseus's fine-grained tagset is far too sparse to learn from its ~1.3k training
sentences, so the tagger predicts it at ~34% and drags the combined-test TAG metric
down.  Blanking Perseus XPOS keeps the tagger coherent (ITTB/PROIEL only) while
leaving Perseus's UPOS, FORM and full dependency annotation intact for the parser.

Perseus sentences are the tail of each merged split (cat order ITTB + PROIEL +
Perseus), so we blank every sentence block from --from-sent (0-based) onward.

    blank_perseus_xpos.py <in.conllu> <out.conllu> --from-sent N
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--from-sent", type=int, required=True,
                    help="0-based index of the first Perseus sentence block")
    a = ap.parse_args()

    blocks = [b for b in open(a.inp, encoding="utf-8").read().split("\n\n") if b.strip()]
    n_blank = 0
    for bi in range(a.from_sent, len(blocks)):
        lines = blocks[bi].split("\n")
        out = []
        for ln in lines:
            if "\t" in ln:
                c = ln.split("\t")
                if c[0].isdigit() and c[4] != "_":   # token rows only
                    c[4] = "_"
                    n_blank += 1
                out.append("\t".join(c))
            else:
                out.append(ln)
        blocks[bi] = "\n".join(out)
    open(a.out, "w", encoding="utf-8").write("\n\n".join(blocks) + "\n\n")
    print(f"{a.out}: blanked XPOS on {len(blocks)-a.from_sent} sentence blocks "
          f"({n_blank} tokens), from block {a.from_sent}/{len(blocks)}")


if __name__ == "__main__":
    main()
