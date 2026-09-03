#!/usr/bin/env python
"""SUPERSEDED — MEASURED NULL, DO NOT USE. Kept as the evidence for a negative result.

Normalises 爲 -> 為 (they are orthographic variants of one word: all nine works that use them use
EXACTLY ONE each, so the glyph is a SOURCE MARKER carrying no grammatical content, and Kyoto's own
LEMMA column already unifies them the other way).

WHY IT IS NOT USED. Measured against a token-for-token aligned control that differs in nothing but
this character (`corpus_lzh_resplit_ctl` vs `corpus_lzh_wnorm_resplit`, identical blocks in
identical order, verified equal on all 533 362 tokens outside ORTH/LEMMA):

    normalised   TAG 93.24   UAS 80.68   LAS 76.85
    control      TAG 93.11   UAS 80.66   LAS 76.81
    on 爲/為 tokens (n=531): XPOS 71.00 normalised vs 71.56 control  -- WORSE

    paired disagreement over 53 475 tokens: 900 normalised-only-correct, 832 control-only-correct

Near-symmetric churn, not signal. THE RE-SPLIT ALREADY FIXED THE MEASURABLE PROBLEM: once every
work is proportionally represented in all three splits the model sees enough of each glyph to learn
both, so collapsing them buys nothing. Any residual value is OFF-TREEBANK robustness, which this
test set cannot see by construction.

⚠ It also breaks the 異體字 map's no-op invariant on TRAIN (1 513 tokens), and
`bundle_lzh_variants.py --verify` now correctly REFUSES a map containing this entry.
"""
import argparse
import pathlib
import sys

SRC = "爲"
DST = "為"


def normalise(inp: pathlib.Path, out: pathlib.Path):
    n_form = n_lemma = n_text = n_tok = n_blk = 0
    other = []
    with out.open("w", encoding="utf-8") as fh:
        for ln, line in enumerate(inp.open(encoding="utf-8"), 1):
            if line.startswith("#"):
                if SRC in line:
                    if not line.startswith("# text ="):
                        other.append(f"{ln}: {SRC} in a non-`# text` comment: {line.rstrip()}")
                    n_text += line.count(SRC)
                    line = line.replace(SRC, DST)
                fh.write(line)
                continue
            if not line.strip():
                n_blk += 1
                fh.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            n_tok += 1
            for i, v in enumerate(f):
                if SRC in v and i not in (1, 2):
                    other.append(f"{ln}: {SRC} in column {i + 1}: {v}")
            if SRC in f[1]:
                n_form += f[1].count(SRC)
                f[1] = f[1].replace(SRC, DST)
            if SRC in f[2]:
                n_lemma += f[2].count(SRC)
                f[2] = f[2].replace(SRC, DST)
            fh.write("\t".join(f) + "\n")
    return dict(tokens=n_tok, blocks=n_blk, form=n_form, lemma=n_lemma, text=n_text, other=other)


def verify(inp: pathlib.Path, out: pathlib.Path):
    """Every difference must be exactly SRC -> DST, in FORM, LEMMA or a `# text` comment."""
    a = inp.read_text(encoding="utf-8").split("\n")
    b = out.read_text(encoding="utf-8").split("\n")
    if len(a) != len(b):
        sys.exit(f"REFUSING: {inp} has {len(a)} lines, {out} has {len(b)}")
    bad = []
    changed = 0
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x == y:
            continue
        changed += 1
        if x.replace(SRC, DST) != y:
            bad.append(f"{i}: not a pure {SRC}->{DST} rewrite")
            continue
        if x.startswith("#"):
            if not x.startswith("# text ="):
                bad.append(f"{i}: changed a non-`# text` comment")
            continue
        fx, fy = x.split("\t"), y.split("\t")
        for c in range(len(fx)):
            if fx[c] != fy[c] and c not in (1, 2):
                bad.append(f"{i}: column {c + 1} changed")
    if bad:
        sys.exit("REFUSING:\n  " + "\n  ".join(bad[:20]))
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for p in a.files:
        inp = pathlib.Path(p)
        out = inp.with_suffix(".wnorm.conllu")
        if out.exists():
            sys.exit(f"REFUSING to overwrite {out}")
        st = normalise(inp, out)
        n = verify(inp, out)
        print(f"{inp.name}\n  -> {out.name}")
        print(f"  {st['tokens']} tokens / {st['blocks']} blocks; "
              f"FORM {st['form']}, LEMMA {st['lemma']}, `# text` {st['text']} occurrences rewritten; "
              f"{n} lines differ")
        if st["other"]:
            print(f"  ⚠ {SRC} left in place elsewhere ({len(st['other'])}):")
            for o in st["other"][:10]:
                print("    " + o)


if __name__ == "__main__":
    main()
