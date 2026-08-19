#!/usr/bin/env python3
"""Kanripo floret vectors for lzh, keyed by character, with rows for EVERY type the parser can meet.

This is the parser-side build. It reuses the recipe from the aptness repo's
`build_lzh_ids_vectors.py` -- train floret over IDS-expanded strings so subword n-grams share
strength between characters with common components (江 ⿰氵工, 河 ⿰氵可), then write a static `.vec`
keyed by the BARE CHARACTER, each row being floret's vector for that character's IDS string. The
sharing ends up inside the row; nothing at runtime needs the IDS table. That repo measured the
alternative: plain codepoint PPMI gives effective rank 291 of 300 and mean |cos| 0.047 -- every type
near-orthogonal, no mechanism to share strength -- and scored 15.69 against 37.17.

⚠ WHY THIS EXISTS RATHER THAN A CALL TO THAT SCRIPT. It emits rows for `sorted(vocab)`, the
CORPUS vocabulary. That is correct when the corpus is all of kanripo, and silently wrong here: the
leak-free corpus has the dev/test sentences removed, which deletes 160 of 279 treebank-unseen types
from the vocabulary entirely -- so exactly the types the vectors are supposed to rescue would get NO
ROW. That is the `vectors_lzh_apt96` failure in a new place (pruned to a training vocabulary: 0 %
coverage of unseen forms, and missing every punctuation mark besides). **Prune by DIMENSION, never
by VOCABULARY.**

`--extra-types` fixes it by emitting rows for the treebank's own inventory as well. This adds no
information from the held-out text: a row is a function of the type's IDS string and the trained
subword matrix, and the text those types occurred in has been removed from training. floret would
compose the same vector at runtime; materialising it into the `.vec` is an implementation detail of
spaCy's static-vector table, not a second look at the evaluation data.

QIEYUN (`--qieyun`). Optional second channel: append each character's 廣韻 音韻地位 (母/呼/等/類/韻/聲,
e.g. 端一東平) to its IDS string, so a character rare enough that its own distribution is unusable can
also share subwords with its HOMOPHONE class. The motivation is real -- 通假字, phonetic loan
characters, are bridged by pronunciation and by nothing in the graph -- and the data is CC0
(nk2028/qieyun-data), covering 98.8 % of lzh character tokens. `assets_qieyun/` is gitignored like
every other asset dir; refetch it with

    curl -sSL -o assets_qieyun/guangyun.csv \
      https://raw.githubusercontent.com/nk2028/qieyun-data/main/%E9%9F%BB%E6%9B%B8/%E5%BB%A3%E9%9F%BB.csv

Read the caveats before using it:

  * A held-out-character probe on this treebank put Qieyun far behind the graphic channels for
    predicting lexical class: null 44.59 %, radical 57.00, IDS 55.30, **Qieyun 48.06**, all three
    57.36 -- i.e. it adds ~nothing on top of IDS.
  * 29.9 % of character tokens are polyphonic and the reading cannot be chosen at inference, so the
    channel is a BAG of that character's readings, not its reading here.
  * It lengthens the token string, so it must be a SEPARATE ARM. Mixed into the default build, a
    loss cannot be attributed to it.

SEP is load-bearing, and is taken from the original recipe: two IDS trees are individually
self-delimiting, but floret's n-grams do not parse and would manufacture units spanning two
characters. U+2016 appears in neither ids.txt nor a 音韻地位 code. The Qieyun code is joined with a
second separator so a cross-channel n-gram is likewise distinguishable from a within-channel one.
"""
import argparse
import collections
import csv
import pathlib
import sys

SEP = "‖"      # between characters of a multi-character token
QSEP = "‖‖"  # between a character's IDS string and its Qieyun code


def load_aptness(path):
    """`load_ids` and `expand` from the aptness repo, imported rather than copied so the two builds
    cannot drift apart on the decomposition depth or the tag-stripping regex."""
    p = pathlib.Path(path) / "scripts" / "build_lzh_ids_vectors.py"
    if not p.exists():
        sys.exit(f"cannot find {p}; pass --aptness pointing at the SUD-aptness checkout")
    sys.path.insert(0, str(p.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_apt_ids", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_ids, mod.expand


def load_qieyun(path):
    """character -> its 音韻地位 codes, sorted and de-duplicated.

    A polyphonic character keeps ALL its readings: which one is meant here cannot be decided at
    inference, so the honest channel is the whole set. Sorted so the string is reproducible -- an
    unsorted set would reshuffle the training data between runs for no reason."""
    out = collections.defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ch, pos = r.get("字頭"), r.get("音韻地位")
            if ch and pos and len(ch) == 1:
                out[ch].add(pos)
    return {c: "/".join(sorted(v)) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="one sentence per line, space-separated tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aptness", default="../SUD-aptness")
    ap.add_argument("--ids", default=None, help="defaults to <aptness>/assets_ids/ids.txt")
    ap.add_argument("--extra-types", action="append", default=[],
                    help="CoNLL-U whose FORM column must all get a row (repeatable)")
    ap.add_argument("--qieyun", default=None, metavar="CSV",
                    help="assets_qieyun/guangyun.csv -- builds the SEPARATE phonology arm")
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--minn", type=int, default=1)
    ap.add_argument("--maxn", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    load_ids, expand = load_aptness(a.aptness)
    ids_path = a.ids or str(pathlib.Path(a.aptness) / "assets_ids" / "ids.txt")
    ids = load_ids(ids_path)
    print(f"  {len(ids):,} characters with an IDS")
    qy = load_qieyun(a.qieyun) if a.qieyun else {}
    if qy:
        print(f"  {len(qy):,} characters with a 音韻地位 (Qieyun arm)")

    lines = pathlib.Path(a.corpus).read_text(encoding="utf-8").splitlines()
    vocab = collections.Counter(t for ln in lines for t in ln.split())
    print(f"  corpus: {len(lines):,} lines, {sum(vocab.values()):,} tokens, {len(vocab):,} types")

    extra = set()
    for p in a.extra_types:
        for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t")
            if len(c) >= 2 and "-" not in c[0] and "." not in c[0]:
                extra.add(c[1])
    missing = extra - set(vocab)
    print(f"  extra types required: {len(extra):,}; absent from the corpus: {len(missing):,}"
          f"  <- these get a row composed from subwords, and would have had NONE")

    def char_str(c):
        s = expand(c, ids, a.depth)
        if qy and c in qy:
            s = s + QSEP + qy[c]
        return s

    cache = {}

    def tok_str(t):
        got = cache.get(t)
        if got is None:
            got = cache[t] = SEP.join(char_str(c) for c in t)
        return got

    tmp = pathlib.Path(a.corpus).with_suffix(".train.txt")
    tmp.write_text("\n".join(" ".join(tok_str(t) for t in ln.split()) for ln in lines) + "\n",
                   encoding="utf-8")
    print(f"  wrote {tmp} ({tmp.stat().st_size / 1e6:.1f} MB)")

    import floret
    m = floret.train_unsupervised(str(tmp), model="cbow", dim=a.dim, minn=a.minn, maxn=a.maxn,
                                  mode="floret", hashCount=2, bucket=50000,
                                  minCount=a.min_count, epoch=a.epochs, thread=8)

    keys = sorted(set(vocab) | extra)
    rows = [(t, m.get_word_vector(tok_str(t))) for t in keys]
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(f"{len(rows)} {a.dim}\n")
        for t, v in rows:
            fh.write(t + " " + " ".join(f"{x:.5f}" for x in v) + "\n")
    print(f"  wrote {a.out}  ({len(rows):,} x {a.dim})")

    import numpy as np
    M = np.array([v for _, v in rows], dtype="float64")
    M = M[np.linalg.norm(M, axis=1) > 0]
    N = M / np.linalg.norm(M, axis=1, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    idx = np.random.default_rng(0).choice(len(N), size=min(2000, len(N)), replace=False)
    cos = N[idx] @ N[idx].T
    off = cos[~np.eye(len(idx), dtype=bool)]
    print(f"  effective rank {float((s.sum() ** 2) / (s ** 2).sum()):.1f} of {a.dim}   "
          f"mean |cos| {abs(off).mean():.3f}    (codepoint-PPMI baseline: 291.2, 0.047)")


if __name__ == "__main__":
    main()
