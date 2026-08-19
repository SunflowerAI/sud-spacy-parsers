#!/usr/bin/env python3
"""Stage the two Tamil treebanks into `assets_ta/`, and build the combined arm's corpus.

SUD_Tamil-TTB ships train/dev/test (400/80/120 sentences, 6 329/1 263/1 989 words); SUD_Tamil-MWTT
ships TEST ONLY (534 sentences, 2 584 words). So MWTT is carved 80/10/10 the way `split_yue.py`
carves Cantonese-HK — round robin by sentence index, no RNG — and then added train->train,
dev->dev, test->test, which is how `add_perseus_la.sh` folds Perseus into Latin.

THREE THINGS HAVE TO BE RECONCILED, and each of them is silent if it is not:

1. **XPOS.** TTB carries a 9-position composite code; MWTT's column is `_` throughout. That is NOT
   a hole the tagger would skip — `spacy convert --converter conllu` does `tag = pos if tag == "_"`
   and falls XPOS back to UPOS, so the combined corpus would carry 234 composite codes beside 14
   bare UPOS strings and `tag_acc` (weighted 0.5, twice `dep_las`) would be selecting checkpoints
   off the mixture. `normalise_ta_xpos.py` renders MWTT onto TTB's tagset instead; held out on
   TTB's own test that map reproduces the gold column 90.05 %.

2. **sent_id.** MWTT numbers its sentences `1`, `2`, ... and TTB names its `train-s1`, so the two
   COLLIDE. Latin's slicing tools read the treebank off the sent_id (`docs/xpos.md`: "not off a
   sentence COUNT as the blanking script did"), so MWTT's are prefixed `mwtt-` and stay that way.

3. **The deprel inventories differ, and this is left alone deliberately.** MWTT subtypes what TTB
   writes plain — `mod@poss` (28 tokens) against TTB's bare `mod` for the same genitive, `subj@nc`
   (46) against `subj`, `udep@tmod`/`@lmod`/`@inst` against bare `udep`. That is an annotation
   DISAGREEMENT, not a tagset one, and no map can fix it without deciding which treebank is right.
   So the combined arm is built AND the TTB-only arm is kept, and both are measured: the honest
   way to price a merge is to show what it costs on the original domain, which is exactly the
   table `docs/latin.md` reports for Perseus.

    prep_ta.py            # writes assets_ta/ta_{ttb,mwtt,ttb_mwtt}-sud-{train,dev,test}.conllu
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
A = ROOT / "assets_ta"

TTB = A / "SUD_Tamil-TTB" / "ta_ttb-sud-%s.conllu"
MWTT_SRC = A / "SUD_Tamil-MWTT" / "ta_mwtt-sud-test.conllu"
SPLITS = ("train", "dev", "test")


def read_blocks(path):
    block = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if block:
                yield block
                block = []
        else:
            block.append(line)
    if block:
        yield block


def write_blocks(path, blocks):
    with open(path, "w", encoding="utf-8") as fh:
        for block in blocks:
            fh.write("\n".join(block) + "\n\n")


def prefix_sent_id(block):
    """`# sent_id = 3` -> `# sent_id = mwtt-3`. Idempotent."""
    out = []
    for line in block:
        if line.startswith("# sent_id =") and "mwtt-" not in line:
            out.append(line.replace("# sent_id =", "# sent_id = mwtt-", 1).replace(
                "= mwtt- ", "= mwtt-"))
        else:
            out.append(line)
    return out


def main() -> None:
    # 1. TTB, verbatim.
    for split in SPLITS:
        src = pathlib.Path(str(TTB) % split)
        dst = A / f"ta_ttb-sud-{split}.conllu"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"ttb  {split:5s} {sum(1 for _ in read_blocks(dst)):5d} sentences -> {dst.name}")

    # 2. MWTT, split 80/10/10 round robin (i%10 == 8 -> dev, == 9 -> test, else train).
    blocks = [prefix_sent_id(b) for b in read_blocks(MWTT_SRC)]
    parts = {s: [] for s in SPLITS}
    for i, block in enumerate(blocks):
        parts["dev" if i % 10 == 8 else "test" if i % 10 == 9 else "train"].append(block)
    for split in SPLITS:
        raw = A / f"ta_mwtt-sud-{split}.raw.conllu"
        write_blocks(raw, parts[split])
        out = A / f"ta_mwtt-sud-{split}.conllu"
        subprocess.run([sys.executable, str(HERE / "normalise_ta_xpos.py"),
                        "--learn", str(A / "ta_ttb-sud-train.conllu"),
                        "--apply", str(raw), "--out", str(out)], check=True,
                       stdout=subprocess.DEVNULL)
        raw.unlink()
        print(f"mwtt {split:5s} {len(parts[split]):5d} sentences -> {out.name} (XPOS projected)")

    # 3. The combined arm: a plain concatenation, each treebank keeping its own sent_ids.
    for split in SPLITS:
        text = "".join((A / f"ta_{tb}-sud-{split}.conllu").read_text(encoding="utf-8")
                       for tb in ("ttb", "mwtt"))
        dst = A / f"ta_ttb_mwtt-sud-{split}.conllu"
        dst.write_text(text, encoding="utf-8")
        print(f"both {split:5s} {sum(1 for _ in read_blocks(dst)):5d} sentences -> {dst.name}")


if __name__ == "__main__":
    main()
