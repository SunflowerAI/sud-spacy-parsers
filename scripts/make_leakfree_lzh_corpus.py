#!/usr/bin/env python3
"""Strip the lzh dev/test sentences out of the kanripo corpus before vectors are trained on it.

WHY. The Kyoto treebank was built FROM the Kanseki Repository -- the sent_ids ARE kanripo ids, which
is how `align_kanripo_punct.py` restored the punctuation. So the kanripo corpus contains the
treebank's evaluation text VERBATIM: a spot check found 200 of 200 test sentences present as whole
lines. Vectors trained on it have seen every test sentence's raw string.

That is not label leakage -- no heads, no relations -- and pretrained embeddings routinely cover the
evaluation domain. It is worse than the usual case, though: it is the exact sentences, so a form the
TREEBANK has never seen still gets a vector estimated partly from its own test occurrence. And that
population is the entire point of the exercise (treebank-unseen forms have a median kanripo
frequency of 4, so a single occurrence moves their vector a lot). A gain measured that way would
overstate what a user parsing new text gets, which is the number that matters.

WHAT IS REMOVED. Only dev and test. Train text stays -- it is legitimately available, and removing
it would answer a different question. A kanripo line goes if its space-free form IS a dev/test
sentence, or CONTAINS one: the 1:1 correspondence holds for most lines, but kanripo's line breaks
are an editor's, not the treebank's, so a longer line may swallow a whole unit.

⚠ SUBSTRING SEARCH, NOT LINE EQUALITY ALONE. Equality misses exactly the cases that matter and
leaves them in the corpus silently. The index is over fixed-length prefixes so the scan stays linear
rather than testing every sentence against every line.

⚠ THE SHORT-SENTENCE FLOOR, AND THE EXEMPTION TO IT. Classical Chinese units are short and many are
formulaic (子曰, 何也); removing every line containing a 2-character test sentence would gut the
corpus for no leakage benefit, since a string made entirely of high-frequency train forms carries no
test-specific information. So sentences shorter than `--min-len` are not removal keys -- EXCEPT when
they contain a form the TRAIN split never saw, which is precisely the evidence being protected.
Measured: 127 of 279 treebank-unseen test types had their ONLY kanripo occurrence inside a test
sentence, and the plain floor left 62 more in retained short lines. The exemption costs a further
0.0x % of the corpus and removes them.

Usage:

    .venv/bin/python scripts/make_leakfree_lzh_corpus.py \\
        --corpus ../SUD-aptness/corpus_lzh_trad_tokens.txt \\
        --treebank assets_lzh/SUD_Classical_Chinese-Kyoto \\
        --prefix lzh_kyoto-sud --suffix relabeled_ext.udep_ruled.punct.rulemerged \\
        --out corpus_lzh_kanripo_leakfree.txt
"""
import argparse
import pathlib


def sentences(path):
    """Space-free sentence strings from a CoNLL-U file, matching the corpus's token stream."""
    out, cur = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur:
                out.append("".join(cur))
            cur = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 2 or "-" in c[0] or "." in c[0]:
            continue
        cur.append(c[1])
    if cur:
        out.append("".join(cur))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--treebank", required=True)
    ap.add_argument("--prefix", default="lzh_kyoto-sud")
    ap.add_argument("--suffix", default="relabeled_ext.udep_ruled.punct.rulemerged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=6,
                    help="sentences shorter than this are not removal keys (see the docstring)")
    ap.add_argument("--key-len", type=int, default=6, help="prefix length for the scan index")
    a = ap.parse_args()

    held = []
    for split in ("dev", "test"):
        p = pathlib.Path(a.treebank) / f"{a.prefix}-{split}.{a.suffix}.conllu"
        s = sentences(p)
        held += s
        print(f"  {split}: {len(s)} sentences")
    # Forms the TRAIN split attests. A held-out sentence built only from these leaks nothing about
    # the rare types the whole exercise turns on; one containing anything else leaks exactly that.
    trp = pathlib.Path(a.treebank) / f"{a.prefix}-train.{a.suffix}.conllu"
    train_forms = set()
    for line in trp.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c = line.split("\t")
        if len(c) >= 2 and "-" not in c[0] and "." not in c[0]:
            train_forms.add(c[1])
    held_toks = {}
    for split in ("dev", "test"):
        pp = pathlib.Path(a.treebank) / f"{a.prefix}-{split}.{a.suffix}.conllu"
        cur = []
        for line in pp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                if cur:
                    held_toks["".join(w for w in cur)] = list(cur)
                cur = []
                continue
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) >= 2 and "-" not in c[0] and "." not in c[0]:
                cur.append(c[1])
        if cur:
            held_toks["".join(cur)] = list(cur)

    def has_rare(sent):
        return any(w not in train_forms for w in held_toks.get(sent, ()))

    long_keys = {s for s in held if len(s) >= a.min_len}
    rare_keys = {s for s in held if len(s) < a.min_len and has_rare(s)}
    keys = sorted(long_keys | rare_keys)
    short = [s for s in held if len(s) < a.min_len and s not in rare_keys]
    print(f"  {len(keys)} distinct removal keys: {len(long_keys)} by length (>= {a.min_len}), "
          f"{len(rare_keys)} short but carrying a form train never saw")
    print(f"  {len(short)} short sentences exempt (all their forms occur in train)")

    K = min(a.key_len, min((len(k) for k in keys), default=a.key_len))
    index = {}
    for s in keys:
        index.setdefault(s[:K], []).append(s)

    kept, dropped, dropped_tokens, kept_tokens = [], 0, 0, 0
    with open(a.corpus, encoding="utf-8") as fh:
        for line in fh:
            toks = line.split()
            if not toks:
                continue
            flat = "".join(toks)
            hit = False
            for i in range(len(flat) - K + 1):
                for s in index.get(flat[i:i + K], ()):
                    if flat.startswith(s, i):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                dropped += 1
                dropped_tokens += len(toks)
            else:
                kept.append(line.rstrip("\n"))
                kept_tokens += len(toks)

    pathlib.Path(a.out).write_text("\n".join(kept) + "\n", encoding="utf-8")
    total = dropped + len(kept)
    print(f"  kanripo lines: {total:,} -> {len(kept):,} kept, {dropped:,} dropped "
          f"({dropped / max(total, 1):.2%})")
    print(f"  tokens: {kept_tokens + dropped_tokens:,} -> {kept_tokens:,} kept "
          f"({dropped_tokens / max(kept_tokens + dropped_tokens, 1):.2%} removed)")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
