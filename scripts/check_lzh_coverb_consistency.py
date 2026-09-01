#!/usr/bin/env python3
"""Is the coverb/verb distinction annotated CONSISTENTLY in Kyoto, for 與 / 以 / 自?

WHY IT MATTERS. The lzh parser's errors are 76.5 % wrong heads, and the readable examples are
dominated by one disagreement: which of two adjacent verbs heads the clause, with 與/以/自 as the
pivot (`與朋友交` — is 與 the main verb or a coverb modifying 交?). If the treebank itself annotates
the same configuration two ways, some fraction of those 4 965 errors is noise the parser cannot win
against, and no amount of context or lexical knowledge will help.

METHOD. For every occurrence of a target word, take a CONTEXT SIGNATURE — the UPOS of the two
following tokens, which is what distinguishes 與+NP+V (coverb) from 與+NP (verb) — and ask how
dominant the majority (UPOS, deprel, head direction) analysis is within each signature. A
consistently annotated word is near 100 % in every signature; a word annotated by coin-flip is near
its class prior.

⚠ THE CONTROL IS THE POINT. "76 % dominance" means nothing on its own, so the same statistic is
computed for frequent words whose category is NOT in doubt (之, 不, 也, 者). Those set the ceiling
that treebank noise and signature coarseness together allow.
"""
import argparse
import collections
import pathlib

TRAIN = ("assets_lzh/SUD_Classical_Chinese-Kyoto/"
         "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")


def blocks(path):
    cur = []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                yield cur
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f)
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--targets", default="與,以,自,於,為")
    ap.add_argument("--controls", default="之,不,也,者,而")
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()

    words = a.targets.split(",") + a.controls.split(",")
    obs = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    overall = collections.defaultdict(collections.Counter)
    for b in blocks(a.train):
        for i, f in enumerate(b):
            w = f[1]
            if w not in words:
                continue
            nxt = b[i + 1][3] if i + 1 < len(b) else "END"
            nxt2 = b[i + 2][3] if i + 2 < len(b) else "END"
            sig = (nxt, nxt2)
            h = int(f[6])
            direction = "root" if h == 0 else ("L" if h - 1 < i else "R")
            analysis = (f[3], f[7], direction)
            obs[w][sig][analysis] += 1
            overall[w][analysis] += 1

    print(f"{'word':<5}{'n':>7}{'analyses':>10}{'majority':>10}"
          f"{'weighted dominance within a context signature':>48}")
    print(f"{'':5}{'':7}{'':10}{'':10}{'(signatures with n >= ' + str(a.min_n) + ')':>48}")
    for w in words:
        tot = sum(overall[w].values())
        if not tot:
            continue
        maj = overall[w].most_common(1)[0][1] / tot
        num = den = 0
        for sig, c in obs[w].items():
            n = sum(c.values())
            if n < a.min_n:
                continue
            num += c.most_common(1)[0][1]
            den += n
        dom = num / den if den else float("nan")
        tag = "  <- target" if w in a.targets.split(",") else "  (control)"
        print(f"{w:<5}{tot:>7}{len(overall[w]):>10}{maj:>9.1%}{dom:>40.1%}{tag}")

    print("\nthe two most frequent context signatures per target, and how they are annotated:")
    for w in a.targets.split(","):
        if not overall[w]:
            continue
        sigs = sorted(obs[w].items(), key=lambda kv: -sum(kv[1].values()))[:2]
        print(f"\n  {w}")
        for sig, c in sigs:
            n = sum(c.values())
            top = c.most_common(3)
            print(f"    next two UPOS {sig}  n={n}")
            for (pos, dep, d), k in top:
                print(f"       {pos:<6} {dep:<14} head-{d}   {k:5d}  {k/n:6.1%}")


if __name__ == "__main__":
    main()
