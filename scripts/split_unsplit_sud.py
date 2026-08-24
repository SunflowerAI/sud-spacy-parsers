#!/usr/bin/env python3
"""Carve train/dev/test for the SUD corpora that ship without one.

Sixteen corpora in SUD 2.18 are distributed the Grew way -- one `.conllu` per text, no
`*-train.conllu` at all -- and they are very nearly the whole SUD-native set: Hausa (four
varieties), Naija, Haitian Creole, Ika, Beja, Zaar, Northwest Gbaya, Pesh, Bokota, Nenets, French
ParisStories and Rhapsodie. Those are the least Eurasian, least converted-from-UD corpora in the
release, so dropping them for a filename convention would take exactly the typological diversity
this experiment is for.

The split follows `split_yue.py`, which carved SUD_Cantonese-HK the same way: **deterministic
round-robin, 80/10/10, no RNG**, because these corpora are ordered by text and a contiguous tail
would hand dev and test a single genre.

⚠ **THE UNIT IS THE DOCUMENT WHERE THERE ARE ENOUGH OF THEM.** These are transcribed narratives and
conversations; consecutive sentences in one text share speakers, topic and often whole formulae, so
splitting a document across train and test leaks. Round-robin over documents keeps each text whole.
Below `--min-docs` there are too few documents to make a 10-way rotation mean anything, so the
fallback is round-robin over SENTENCES -- which is what `split_yue.py` does, and which is the right
trade when the alternative is having no dev set at all.

Output goes to a `derived/` tree, never on top of the release, so re-extracting the tarball cannot
silently half-overwrite a split and `build_tb_inventory.py` can tell the two apart.
"""
import argparse
import json
import pathlib
import sys

#: Grew build artefacts that sit beside the data.
SKIP_NAMES = {"merge.json", "metadata.json"}
SKIP_DIRS = {"_build_grew"}


def read_blocks(path):
    """Sentence blocks, verbatim lines. No parsing: this script must not alter a single column."""
    block = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                if block:
                    yield block
                    block = []
            else:
                block.append(line.rstrip("\n"))
    if block:
        yield block


def n_tokens(block):
    return sum(1 for ln in block
               if not ln.startswith("#") and "\t" in ln
               and "-" not in ln.split("\t", 1)[0] and "." not in ln.split("\t", 1)[0])


def bucket(i):
    """80/10/10 by rotation position, exactly as split_yue.py."""
    return {8: "dev", 9: "test"}.get(i % 10, "train")


def carve(files, min_docs):
    """`({split: [blocks]}, unit)`. Documents where there are enough, else sentences."""
    if len(files) >= min_docs:
        out = {"train": [], "dev": [], "test": []}
        for i, f in enumerate(sorted(files)):
            out[bucket(i)].extend(read_blocks(f))
        return out, "document"
    blocks = [b for f in sorted(files) for b in read_blocks(f)]
    out = {"train": [], "dev": [], "test": []}
    for i, b in enumerate(blocks):
        out[bucket(i)].append(b)
    return out, "sentence"


def write_blocks(path, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write("\n".join(b) + "\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sud-dir", default="assets_sud218/sud-treebanks-v2.18")
    ap.add_argument("--out", default="assets_sud218/derived")
    ap.add_argument("--min-docs", type=int, default=10,
                    help="below this many documents, rotate over sentences instead (default 10)")
    ap.add_argument("--force", action="store_true", help="rewrite splits that already exist")
    a = ap.parse_args()

    base = pathlib.Path(a.sud_dir)
    out_root = pathlib.Path(a.out)
    if not base.is_dir():
        sys.exit(f"{base} is not a directory -- run scripts/fetch_sud_release.sh first")

    report = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(("SUD_", "mSUD_"))):
        files = [p for p in d.glob("*.conllu")
                 if p.name not in SKIP_NAMES and p.parent.name not in SKIP_DIRS]
        if not files:
            continue
        # Already split by the release: leave it entirely alone.
        if any(p.name.endswith(("-train.conllu", "-dev.conllu", "-test.conllu")) for p in files):
            continue
        dest = out_root / d.name
        prefix = d.name.removeprefix("SUD_").removeprefix("mSUD_").lower().replace("-", "_")
        if not a.force and (dest / f"{prefix}-train.conllu").exists():
            print(f"have {d.name}")
            continue
        splits, unit = carve(files, a.min_docs)
        counts = {}
        for split, blocks in splits.items():
            p = dest / f"{prefix}-{split}.conllu"
            write_blocks(p, blocks)
            counts[split] = {"sents": len(blocks), "tokens": sum(n_tokens(b) for b in blocks)}
        rec = {"corpus": d.name, "unit": unit, "n_files": len(files), "counts": counts}
        report.append(rec)
        tot = sum(c["tokens"] for c in counts.values())
        print(f"{d.name:36s} {len(files):4d} files, by {unit:8s} -> "
              f"train {counts['train']['tokens']:7d}  dev {counts['dev']['tokens']:6d}  "
              f"test {counts['test']['tokens']:6d}  (total {tot})")

    if report:
        out_root.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out_root / "split_report.json", "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
    print(f"\ncarved {len(report)} corpora -> {out_root}")
    by_unit = {}
    for r in report:
        by_unit.setdefault(r["unit"], []).append(r["corpus"])
    for unit, names in by_unit.items():
        print(f"  by {unit}: {len(names)}  {' '.join(n.removeprefix('SUD_') for n in names)}")


if __name__ == "__main__":
    main()
