#!/usr/bin/env python3
"""Build the lzh 異體字 -> treebank-orthography NORM table.

THE PROBLEM IT SOLVES. The lzh morphologiser's UPOS accuracy is 93.13 % overall but **51.79 %** on
a token containing a character the Kyoto training split never showed, and its PROPN precision there
is **39.13 %**. The reason is a corpus prior, not a modelling accident: 37.5 % of Kyoto's TYPES are
PROPN against 8.5 % of its tokens, and 49.4 % of its hapax types are — so a character the encoder
cannot identify has a learned prior of roughly one-in-two PROPN, and nothing to override it with
(the morphologiser's own `HashEmbedCNN` reads NORM/PREFIX/SUFFIX/SHAPE and 2 000 hash rows, no
lexicon and no vectors).

⚠ AND MOST OF THOSE CHARACTERS ARE NOT RARE — they are ORDINARY WORDS IN A DIFFERENT GRAPHIC FORM.
Of the 42 M characters of kanripo, 3.38 % are absent from the treebank; of that mass 80 % sits in
the 311 types that occur MORE than 500 times in kanripo. 无 (146 685 occurrences, = 無 "not have"),
隂 (97 843, = 陰), 徳 (84 213, = 德), 逺 (27 903, = 遠). Each is a high-frequency function or content
word that the released arm tags PROPN because it has never seen that glyph.

This is the type/token split that NEGATIVE-RESULTS.md's kanripo-vector entry warns about, read the
other way round. That entry found treebank-unseen TYPES have a median kanripo frequency of 4 and
concluded static vectors are empty where they are needed. Both are true: by types the unseen
population is rare, by TOKENS it is dominated by a few hundred common variants. Which statistic is
the right one depends on the task, and for tagging running text it is tokens.

THE TABLE. Two sources, symbolic first:

  1. **Unihan** (`assets_unihan/Unihan_Variants.txt`, Unicode licence — already in the tree for the
     radical channel). kSemanticVariant / kZVariant / kTraditionalVariant / kSimplifiedVariant /
     kSpecializedSemanticVariant, followed up to `--hops` links to reach a treebank character.
     Authoritative, and it covers 48.0 % of the absent mass on its own.
  2. **SikuBERT's input embedding table** (`SIKU-BERT/sikubert`, Apache-2.0) for the residue —
     nearest treebank-seen character by cosine, above `--min-cos`. It reaches 徳→德 (0.68),
     逺→遠 (0.64), 乗→乘 (0.68), 兊→兌 and 㑹, the Japanese-style forms kanripo is full of and
     Unihan does not link.

Route 2 is validated against route 1 rather than asserted: on the characters where Unihan DOES give
a treebank-seen answer, the script reports how often SikuBERT's nearest neighbour is the same
character. Read that agreement rate before trusting the residue; a low one means `--min-cos` is too
low, not that the table is finished.

⚠ NOTHING FROM SIKUBERT IS SHIPPED BUT THIS TABLE. It is a few kilobytes of JSON, and the wheel
gains no transformer, no runtime dependency and no inference cost — the whole point of distilling
the question down to "which known character is this glyph". Where the character is genuinely NEW
rather than a variant (爻, 彖 — Yijing technical terms), the nearest neighbour is a semantic one
(卦, cos 0.44 / 0.30) and `--min-cos` is what keeps it out. Inspect the emitted `--dump` list; at
0.55 it is ~500 pairs, which is a reviewable size.

Usage:
    build_lzh_variant_norm.py --out models/lzh_variant_norm.json [--min-cos 0.55] [--dump -]
"""
import argparse
import collections
import json
import pathlib
import re
import sys

TRAIN = ("assets_lzh/SUD_Classical_Chinese-Kyoto/"
         "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")
UNIHAN = "assets_unihan/Unihan_Variants.txt"
FIELDS = ("kSemanticVariant", "kZVariant", "kTraditionalVariant",
          "kSimplifiedVariant", "kSpecializedSemanticVariant")


def treebank_chars(path):
    c = collections.Counter()
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        for ch in f[1]:
            c[ch] += 1
    return c


def unihan_links(path):
    """char -> set of variant chars, over the variant fields (both directions are NOT symmetric in
    Unihan, so the caller walks the graph rather than assuming a single hop suffices)."""
    g = collections.defaultdict(set)
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        cp, field, val = line.rstrip("\n").split("\t")[:3]
        if field not in FIELDS:
            continue
        src = chr(int(cp[2:], 16))
        for m in re.findall(r"U\+[0-9A-F]+", val):
            g[src].add(chr(int(m[2:], 16)))
    return g


def unihan_target(ch, g, known, hops):
    seen, frontier = {ch}, {ch}
    for _ in range(hops):
        nxt = set()
        for x in frontier:
            for y in g.get(x, ()):
                if y in seen:
                    continue
                if y in known:
                    return y
                seen.add(y)
                nxt.add(y)
        frontier = nxt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--unihan", default=UNIHAN)
    ap.add_argument("--model", default="SIKU-BERT/sikubert")
    ap.add_argument("--hops", type=int, default=2)
    # 0.55 keeps 爻 (top neighbour 卦 at 0.44) and 彖 (0.30) OUT: a character that is genuinely new
    # rather than a variant has a SEMANTIC nearest neighbour, and semantic neighbours sit lower.
    ap.add_argument("--min-cos", type=float, default=0.55)
    ap.add_argument("--no-sikubert", action="store_true",
                    help="Unihan only — the symbolic half, no model download")
    ap.add_argument("--corpus", default="corpus_lzh_kanripo_leakfree.txt",
                    help="raw text used ONLY to weight the coverage report (not to build the table)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None, help="write the pairs as TSV ('-' for stdout)")
    a = ap.parse_args()

    known = treebank_chars(a.train)
    print(f"treebank: {len(known)} characters", flush=True)
    g = unihan_links(a.unihan)

    # The characters to build a row for: everything a text can contain that the treebank cannot.
    # Taken from the raw corpus so the table covers what real input holds, NOT from the treebank
    # (which by definition holds none of them).
    freq = collections.Counter()
    if pathlib.Path(a.corpus).exists():
        for line in pathlib.Path(a.corpus).open(encoding="utf-8"):
            freq.update(line.replace(" ", "").strip())
    absent = {c: n for c, n in freq.items() if c not in known}
    total = sum(absent.values()) or 1
    print(f"corpus characters absent from the treebank: {len(absent)} types / {total} tokens",
          flush=True)

    table, source = {}, {}
    for c in absent:
        t = unihan_target(c, g, known, a.hops)
        if t:
            table[c], source[c] = t, "unihan"
    cov = sum(absent[c] for c in table)
    print(f"  Unihan     : {len(table):5d} types  {cov:8d} tokens  {cov/total:6.1%}", flush=True)

    agree = disagree = 0
    if not a.no_sikubert:
        import numpy as np
        from transformers import AutoModel, AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.model)
        m = AutoModel.from_pretrained(a.model)
        E = m.get_input_embeddings().weight.detach().numpy()
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        unk = tk.unk_token_id
        seen_chars = [c for c in known if len(c) == 1 and tk.convert_tokens_to_ids(c) != unk]
        S = E[np.array([tk.convert_tokens_to_ids(c) for c in seen_chars])]

        def nearest(c):
            i = tk.convert_tokens_to_ids(c)
            if i == unk:
                return None, 0.0
            sims = S @ E[i]
            j = int(sims.argmax())
            return seen_chars[j], float(sims[j])

        # VALIDATION FIRST: where Unihan has the answer, does SikuBERT agree?
        for c in list(table):
            nn, cos = nearest(c)
            if cos >= a.min_cos:
                agree += (nn == table[c])
                disagree += (nn != table[c])
        tot_v = agree + disagree
        print(f"  validation : SikuBERT's nearest neighbour matches Unihan on {agree}/{tot_v}"
              f" = {agree/tot_v:.1%} of the characters Unihan resolves (cos >= {a.min_cos})",
              flush=True)

        added = 0
        for c in absent:
            if c in table:
                continue
            nn, cos = nearest(c)
            if nn and cos >= a.min_cos:
                table[c], source[c] = nn, f"sikubert:{cos:.2f}"
                added += 1
        cov2 = sum(absent[c] for c in table)
        print(f"  + SikuBERT : {added:5d} types  {cov2-cov:8d} tokens  {(cov2-cov)/total:6.1%}"
              f"   (union {cov2/total:.1%})", flush=True)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"__meta__": {"train": a.train, "model": None if a.no_sikubert else a.model,
                            "min_cos": a.min_cos, "hops": a.hops,
                            "sikubert_agrees_with_unihan": f"{agree}/{agree+disagree}"},
               "map": table, "source": source}
    pathlib.Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"wrote {a.out}: {len(table)} entries", flush=True)

    if a.dump:
        fh = sys.stdout if a.dump == "-" else open(a.dump, "w", encoding="utf-8")
        for c in sorted(table, key=lambda x: -absent.get(x, 0)):
            print(f"{c}\t{table[c]}\t{absent.get(c,0)}\t{source[c]}", file=fh)
        if fh is not sys.stdout:
            fh.close()


if __name__ == "__main__":
    main()
