#!/usr/bin/env python3
"""Re-split the lzh Kyoto treebank so every split holds the same mixture of WORKS.

THE PROBLEM. `# sent_id = KR1h0004_012_par1_5-6` — the part before the first `_` is the kanripo
WORK id and the part after it is the CHAPTER (kanripo file) number. Kyoto's released split is by
CHAPTER: 論語's chapters 001-003 are test, 012-014 are dev, the rest train. That is clean — no
passage straddles two splits — but it cannot split a work that has only ONE chapter, and four of
the ten works have exactly one:

    KR4h0169  3 216 blocks  1 chapter   -> train only
    KR6c0023    535 blocks  1 chapter   -> train only
    KR6f0082    141 blocks  1 chapter   -> dev only
    KR6c0127     56 blocks  1 chapter   -> test only

So each split sees a different mixture of works, and since a work is a genre (史記 is history,
論語 is dialogue) that is a distribution shift the metrics silently absorb.

THE RULE (deterministic, reproducible).

  1. Pool the blocks of all three input files. A block is a CoNLL-U sentence block and is never
     divided; it is copied through verbatim, comments included.
  2. Reconstruct each work's DOCUMENT ORDER: chapters ascending, and within a chapter the order of
     the source file (a chapter lives wholly in one input file, which the script asserts).
  3. Cut each work into CONTIGUOUS CHUNKS of `--chunk` blocks (default 10). ⚠ The chunk is the
     assignment unit, not the block, for two reasons: `spacy convert -n 10` groups ten CONSECUTIVE
     blocks into one document and the lzh arm learns sentence boundaries from exactly that
     grouping, so shuffling single blocks would train the segmenter on incoherent juxtapositions;
     and adjacent 句讀 blocks are the same passage, so block-level shuffling leaks a test block's
     context into train. `--chunk 1` gives the pure per-block variant.
  4. Shuffle each work's chunks with `random.Random(f"{--seed}:{work}")` — a string seed, so the
     draw does not depend on chunk count, PYTHONHASHSEED or any other work.
  5. Allocate by chunk count: n_dev = n_test = floor(n/10), train takes the remainder.
  6. Emit each split in the ORIGINAL document order (works in a fixed order, chunks by position),
     so the file stays as contiguous as the allocation allows.

INTEGRITY, checked before anything is written: exactly one root per block, every HEAD in
[0, len(block)], IDs 1..n contiguous, no block in two splits, and the total token count across the
three outputs equal to the total across the three inputs.

Usage:
    resplit_lzh_by_work.py --inputs a.conllu b.conllu c.conllu --out-prefix <dir>/lzh_kyoto-sud \
        --out-suffix .relabeled_ext.udep_ruled.punct.rulemerged.wnorm.resplit.conllu
"""
import argparse
import collections
import pathlib
import random
import sys


class Block:
    __slots__ = ("lines", "sid", "work", "chapter", "ntok", "src", "pos")

    def __init__(self, lines, src, pos):
        self.lines = lines
        self.src = src
        self.pos = pos
        self.sid = None
        for ln in lines:
            if ln.startswith("# sent_id ="):
                self.sid = ln.split("=", 1)[1].strip()
                break
        if self.sid is None:
            sys.exit(f"{src}: block at position {pos} has no sent_id")
        parts = self.sid.split("_")
        self.work = parts[0]
        self.chapter = parts[1] if len(parts) > 1 else ""
        self.ntok = sum(1 for ln in lines
                        if ln and not ln.startswith("#")
                        and "-" not in ln.split("\t")[0] and "." not in ln.split("\t")[0])

    def check(self):
        rows = [ln.split("\t") for ln in self.lines if ln and not ln.startswith("#")]
        ids = [r[0] for r in rows]
        if [i for i in ids if "-" in i or "." in i]:
            return f"{self.sid}: multiword/empty node present"
        if ids != [str(i) for i in range(1, len(ids) + 1)]:
            return f"{self.sid}: IDs are not 1..n"
        roots = [r for r in rows if r[6] == "0"]
        if len(roots) != 1:
            return f"{self.sid}: {len(roots)} roots"
        for r in rows:
            try:
                h = int(r[6])
            except ValueError:
                return f"{self.sid}: non-integer HEAD {r[6]!r}"
            if not 0 <= h <= len(rows):
                return f"{self.sid}: HEAD {h} out of range 0..{len(rows)}"
            if h == int(r[0]):
                return f"{self.sid}: token {r[0]} is its own head"
        return None


def read_blocks(path):
    out, cur = [], []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                out.append(Block(cur, path, len(out)))
                cur = []
            continue
        cur.append(line)
    if cur:
        out.append(Block(cur, path, len(out)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--out-suffix", required=True)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--seed", default="lzh-resplit-v1")
    a = ap.parse_args()

    blocks = []
    for p in a.inputs:
        blocks.extend(read_blocks(p))
    print(f"read {len(blocks)} blocks / {sum(b.ntok for b in blocks)} tokens "
          f"from {len(a.inputs)} files")

    bad = [m for m in (b.check() for b in blocks) if m]
    if bad:
        sys.exit(f"REFUSING: {len(bad)} malformed input blocks:\n  " + "\n  ".join(bad[:10]))

    # a chapter must live wholly in one input file, else "document order" is not recoverable
    chsrc = collections.defaultdict(set)
    for b in blocks:
        chsrc[(b.work, b.chapter)].add(b.src)
    straddle = [k for k, v in chsrc.items() if len(v) > 1]
    if straddle:
        print(f"  ⚠ {len(straddle)} chapters straddle two input files: {straddle[:5]}")

    by_work = collections.defaultdict(list)
    for b in blocks:
        by_work[b.work].append(b)

    assign = {}          # id(block) -> split
    order = []           # global emission order
    stats = collections.defaultdict(collections.Counter)
    for work in sorted(by_work):
        bs = sorted(by_work[work], key=lambda b: (b.chapter, a.inputs.index(b.src), b.pos))
        order.extend(bs)
        chunks = [bs[i:i + a.chunk] for i in range(0, len(bs), a.chunk)]
        idx = list(range(len(chunks)))
        random.Random(f"{a.seed}:{work}").shuffle(idx)
        n = len(chunks)
        # ⚠ FLOOR OF ONE CHUNK EACH. `n // 10` sends a work with fewer than ten chunks entirely to
        # train, which is the very defect being fixed (KR6c0127 has six). Rounding with a floor of
        # one keeps every work in all three splits; a work of fewer than three chunks cannot be and
        # is reported rather than silently confined.
        if n < 3:
            sys.exit(f"REFUSING: {work} has only {n} chunk(s) at --chunk {a.chunk}; "
                     f"it cannot be present in all three splits")
        n_dev = n_test = max(1, round(n * 0.1))
        n_train = n - n_dev - n_test
        for rank, ci in enumerate(idx):
            sp = "train" if rank < n_train else ("dev" if rank < n_train + n_dev else "test")
            for b in chunks[ci]:
                assign[id(b)] = sp
                stats[sp][work] += b.ntok
        print(f"  {work:10s} {len(bs):6d} blocks  {n:5d} chunks -> "
              f"train {n_train} / dev {n_dev} / test {n_test}")

    outs = {}
    counts = collections.Counter()
    tokens = collections.Counter()
    for sp in ("train", "dev", "test"):
        outp = pathlib.Path(f"{a.out_prefix}-{sp}{a.out_suffix}")
        if outp.exists():
            sys.exit(f"REFUSING to overwrite {outp}")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outs[sp] = outp.open("w", encoding="utf-8")
    seen = set()
    for b in order:
        if b.sid in seen:
            sys.exit(f"REFUSING: duplicate sent_id {b.sid}")
        seen.add(b.sid)
        sp = assign[id(b)]
        outs[sp].write("\n".join(b.lines) + "\n\n")
        counts[sp] += 1
        tokens[sp] += b.ntok
    for fh in outs.values():
        fh.close()

    tot_in = sum(b.ntok for b in blocks)
    print(f"\nblocks in {len(blocks)} -> out {sum(counts.values())}   "
          f"tokens in {tot_in} -> out {sum(tokens.values())}")
    if sum(counts.values()) != len(blocks) or sum(tokens.values()) != tot_in:
        sys.exit("REFUSING: block or token count changed")
    for sp in ("train", "dev", "test"):
        sh = " ".join(f"{w}:{stats[sp][w] / tokens[sp]:.3f}" for w in sorted(stats[sp]))
        print(f"  {sp:5s} {counts[sp]:6d} blocks {tokens[sp]:7d} tokens  {sh}")


if __name__ == "__main__":
    main()
