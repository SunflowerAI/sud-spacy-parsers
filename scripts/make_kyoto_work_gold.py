#!/usr/bin/env python3
"""Emit one kanripo work from a Kyoto split as a space-separated gold segmentation.

WHY THIS BEATS HAND ANNOTATION. Kyoto's Buddhist material is one work per split:

    train  KR6c0023  金剛般若波羅蜜經                Diamond Sutra          535 sents / 5,700 tok
    dev    KR6f0082  佛說阿彌陀經                    Amitabha Sutra         141 sents / 1,921 tok
    test   KR6c0127  摩訶般若波羅蜜大明呪經          Heart Sutra (Kumarajiva) 56 sents /  360 tok

KR6c0127 is a translation of the SAME text as the hand-annotated CBETA gold (which uses Xuanzang's
T08n0251), it sits in TEST so no model trained on it, and it is annotated by the treebank's own
annotators rather than by us. It is the external check on the hand gold.

⚠ KR6f0082 IS THE AMITABHA SUTRA AND IT IS KYOTO'S DEV SPLIT. Hand-annotating that text from CBETA
as a "held-out" test set -- which is exactly what was about to happen here -- would have produced a
fully contaminated evaluation. Check `sent_id` prefixes against the splits before annotating
anything.

⚠ ATTESTATION MUST BE COUNTED ON TRAIN ONLY. Because the Buddhist works are split one per split,
train+dev+test counts badly overstate what a model saw: 舍利弗 counts 41x across the treebank and
0x in train.
"""
import argparse, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="relabeled_ext.udep_ruled.punct.rulemerged")
    a = ap.parse_args()
    p = pathlib.Path("assets_lzh/SUD_Classical_Chinese-Kyoto") / \
        f"lzh_kyoto-sud-{a.split}.{a.suffix}.conllu"
    sid, cur, rows = None, [], []
    for line in p.open(encoding="utf-8"):
        if line.startswith("# sent_id"): sid = line.split("=", 1)[1].strip()
        elif not line.strip():
            if cur:
                if (sid or "").startswith(a.work): rows.append(cur)
                cur = []
        elif not line.startswith("#"):
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]: cur.append(f[1])
    if cur and (sid or "").startswith(a.work): rows.append(cur)
    if not rows: raise SystemExit(f"no sentences for {a.work} in {a.split}")
    # The treebank's first token of a work carries a stray "]" from the kanripo header; drop it.
    rows = [[w for w in r if w != "]"] for r in rows]
    rows = [r for r in rows if r]
    out = pathlib.Path(a.out)
    out.write_text("\n".join(" ".join(r) for r in rows) + "\n", encoding="utf-8")
    n = sum(len(r) for r in rows); m = sum(1 for r in rows for w in r if len(w) > 1)
    print(f"wrote {out}: {len(rows)} sentences, {n} tokens, {m} multi-char ({m/n:.2%})")

if __name__ == "__main__":
    main()
