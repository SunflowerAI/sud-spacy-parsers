#!/usr/bin/env python
"""Combine the macron FORM transform with the ext-relabel deprel transform.

The two transforms are orthogonal: macronise_la.py changes only FORM (+ `# text`),
relabel_ext.py changes only the DEPREL cell.  Both operate on identical tokens, so
we build the ext+macron variant by taking the macronised file as the base (macron
FORMs + macronised `# text` + baseline deprels) and overwriting just the deprel
column (field 7) from the ext-relabeled file -- no need to re-run the macroniser.

    .venv/bin/python scripts/transfer_macrons.py \
        <macron_src>.conllu <ext_label_src>.conllu <out>.conllu
"""
import sys
import unicodedata


def is_token(line):
    return "\t" in line and line.split("\t", 1)[0].isdigit()


def strip_macron(s):
    n = unicodedata.normalize("NFD", s)
    n = "".join(c for c in n if c != "̄")
    return unicodedata.normalize("NFC", n)


def main(macron_path, label_path, out_path):
    macron_lines = open(macron_path, encoding="utf-8").read().splitlines()
    label_tokens = [l.split("\t") for l in open(label_path, encoding="utf-8")
                    if is_token(l.rstrip("\n"))]
    it = iter(label_tokens)
    out = []
    n_swapped = 0
    for line in macron_lines:
        if is_token(line):
            cols = line.split("\t")
            lab = next(it)
            # safety: same token (form matches after stripping macrons), same head
            assert strip_macron(cols[1]) == lab[1], f"form mismatch: {cols[1]} vs {lab[1]}"
            assert cols[6] == lab[6], f"head mismatch at {cols[0]}: {cols[6]} vs {lab[6]}"
            if cols[7] != lab[7]:
                n_swapped += 1
            cols[7] = lab[7]  # take the ext deprel
            out.append("\t".join(cols))
        else:
            out.append(line)
    # exhausted both?
    leftover = list(it)
    assert not leftover, f"{len(leftover)} label tokens left over"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"{out_path}: {len(label_tokens)} tokens, {n_swapped} deprels set from ext")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
