#!/usr/bin/env python3
"""Is lzh's word-order variation CONDITIONED by UPOS or by the XPOS semantic fields?

The rigid relations are already obeyed by the parser (see check_lzh_order_rigidity.py), so the
question is whether the LOOSE ones — `mod` 96.8 % R over 94 103 arcs, `comp:obj` 88.6 % L over
82 392, `mod@lmod` 58.4 %, `comp:obl@lmod` 57.7 % — become predictable once conditioned.

Kyoto's XPOS is a four-field code, `v,動詞,行為,伝達`, whose third and fourth fields are SEMANTIC
classes (46 and 84 values: 行為 act, 人 person, 描写 description, 固定物 fixed object; 動作 motion,
役割 role, 関係 relation, 態度 attitude, 伝達 transmission). If direction is a function of the
semantics of the head or the dependent, that is a real constraint; if it is not, there is nothing
to enforce.

⚠ **DERIVED ON TRAIN, SCORED ON TEST.** Conditioning on a high-cardinality variable always raises
apparent dominance — 84 values will "explain" anything in-sample. The majority direction per cell
is harvested from train and its accuracy measured on test, against the unconditioned majority
baseline on the same test tokens. An unseen cell backs off to the relation's overall majority, so
coverage is not silently traded for accuracy.
"""
import argparse
import collections
import pathlib


def arcs(path):
    """(deprel, direction, dependent UPOS, head UPOS, dep XPOS fields, head XPOS fields)."""
    blocks, cur = [], []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f)
    if cur:
        blocks.append(cur)
    for b in blocks:
        for i, f in enumerate(b):
            h = int(f[6])
            if h == 0 or f[7] == "punct":
                continue
            hd = b[h - 1]
            df = (f[4].split(",") + ["", "", "", ""])[:4]
            hf = (hd[4].split(",") + ["", "", "", ""])[:4]
            yield f[7], ("L" if h - 1 < i else "R"), f[3], hd[3], df, hf


FEATURES = {
    "dep UPOS":        lambda d, h, df, hf: d,
    "head UPOS":       lambda d, h, df, hf: h,
    "dep XPOS f3":     lambda d, h, df, hf: df[2],
    "dep XPOS f4":     lambda d, h, df, hf: df[3],
    "head XPOS f3":    lambda d, h, df, hf: hf[2],
    "head XPOS f4":    lambda d, h, df, hf: hf[3],
    "head f3 + dep f3": lambda d, h, df, hf: (hf[2], df[2]),
    "head UPOS + dep UPOS": lambda d, h, df, hf: (h, d),
}


def main():
    ap = argparse.ArgumentParser()
    base = "assets_lzh/SUD_Classical_Chinese-Kyoto/lzh_kyoto-sud-{}.relabeled_ext.udep_ruled.punct.rulemerged.conllu"
    ap.add_argument("--train", default=base.format("train"))
    ap.add_argument("--test", default=base.format("test"))
    ap.add_argument("--rels", default="mod,comp:obj,mod@lmod,comp:obl@lmod,comp:obl,udep")
    ap.add_argument("--min-cell", type=int, default=20)
    a = ap.parse_args()

    tr = list(arcs(a.train))
    te = list(arcs(a.test))
    for rel in a.rels.split(","):
        TR = [x for x in tr if x[0] == rel]
        TE = [x for x in te if x[0] == rel]
        if len(TE) < 100:
            continue
        overall = collections.Counter(x[1] for x in TR).most_common(1)[0][0]
        base_acc = sum(1 for x in TE if x[1] == overall) / len(TE)
        print(f"\n{rel}   train {len(TR)}  test {len(TE)}   "
              f"unconditioned baseline on test {base_acc:.1%}")
        print(f"   {'feature':<24}{'cells':>7}{'covered':>9}{'accuracy':>10}{'gain':>8}")
        for name, fn in FEATURES.items():
            table = collections.defaultdict(collections.Counter)
            for _, d, dp, hp, df, hf in TR:
                table[fn(dp, hp, df, hf)][d] += 1
            rule = {k: c.most_common(1)[0][0] for k, c in table.items()
                    if sum(c.values()) >= a.min_cell}
            hit = cov = 0
            for _, d, dp, hp, df, hf in TE:
                k = fn(dp, hp, df, hf)
                pred = rule.get(k)
                if pred is not None:
                    cov += 1
                hit += (d == (pred if pred is not None else overall))
            acc = hit / len(TE)
            print(f"   {name:<24}{len(rule):>7}{cov/len(TE):>9.1%}{acc:>10.1%}"
                  f"{acc-base_acc:>+8.1%}")


if __name__ == "__main__":
    main()
