#!/usr/bin/env python3
"""Split the test-only SUD_Cantonese-HK treebank into train/dev/test.

UD/SUD_Cantonese-HK ships a single `test` file (1004 sentences, no train/dev), so to train a
parser at all we carve a deterministic 80/10/10 split. The split is **round-robin by sentence
index** (i%10 in {8}->dev, {9}->test, else train) rather than contiguous, because the treebank is
topically ordered (parallel_id hk/1, hk/2, ...) and a contiguous tail would skew the dev/test
genre mix. Deterministic + reproducible, no RNG.

Cantonese XPOS (CoNLL-U col 5) is empty (all `_`), so the spaCy `tagger` would have no target.
We copy UPOS (col 4) into XPOS so TAG-prediction becomes UPOS-prediction (as the other Han
treebanks already carry a populated XPOS). FORM/HEAD/DEPREL are untouched, so the relabel
pipeline sees the original annotation.
"""

SRC = "assets_yue/SUD_Cantonese-HK/yue_hk-sud-test.conllu"
OUT = "assets_yue/SUD_Cantonese-HK/yue_hk-sud-%s.conllu"


def read_blocks(path):
    block = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == "":
                if block:
                    yield block
                    block = []
            else:
                block.append(line.rstrip("\n"))
    if block:
        yield block


def copy_upos_to_xpos(line):
    """For a token row (id is an int, not a MWT range or comment), put col4 into col5."""
    if line.startswith("#"):
        return line
    cols = line.split("\t")
    if len(cols) != 10 or "-" in cols[0] or "." in cols[0]:
        return line
    cols[4] = cols[3]  # XPOS <- UPOS
    return "\t".join(cols)


def main():
    blocks = list(read_blocks(SRC))
    splits = {"train": [], "dev": [], "test": []}
    for i, block in enumerate(blocks):
        dest = "dev" if i % 10 == 8 else "test" if i % 10 == 9 else "train"
        splits[dest].append([copy_upos_to_xpos(l) for l in block])
    for name, bs in splits.items():
        with open(OUT % name, "w", encoding="utf-8") as fh:
            for block in bs:
                fh.write("\n".join(block) + "\n\n")
        print(f"{name}: {len(bs)} sentences -> {OUT % name}")


if __name__ == "__main__":
    main()
