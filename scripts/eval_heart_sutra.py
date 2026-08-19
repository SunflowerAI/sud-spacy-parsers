#!/usr/bin/env python3
"""Score a segmenter end-to-end on the hand-gold Heart Sutra, split by Kyoto attestation.

The split is the point. A multi-char unit ATTESTED in Kyoto can be recovered by memorisation; a unit
attested 0 times can only be recovered by generalisation. Reporting one number over both hides
exactly the thing under test -- the same failure the frequency-slice harness exists to prevent.
"""
import argparse, collections, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

def kyoto_counts():
    """TRAIN ONLY. Counting train+dev+test overstates what the model learned: Kyoto's Buddhist
    material is split one work per split -- KR6c0023 (Diamond Sutra) in train, KR6f0082 (Amitabha
    Sutra) in DEV, KR6c0127 (Kumarajiva Heart Sutra) in TEST -- so e.g. 舍利弗 counts 41x overall
    while the model never saw it in training at all."""
    c = collections.Counter()
    for split in ("train",):
        p = pathlib.Path("assets_lzh/SUD_Classical_Chinese-Kyoto/"
                         f"lzh_kyoto-sud-{split}.relabeled_ext.udep_ruled.punct.rulemerged.conllu")
        for line in p.open(encoding="utf-8"):
            if line.startswith("#") or not line.strip(): continue
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]: continue
            c[f[1]] += 1
    return c

def spans(ws):
    out, i = [], 0
    for w in ws:
        out.append((i, i + len(w), w)); i += len(w)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/lzh_seg_char")
    ap.add_argument("--gold", default="assets_cbeta/heart_sutra_gold.txt")
    a = ap.parse_args()
    from sa_presegment import Presegmenter
    m = Presegmenter.from_disk(pathlib.Path(a.model))
    kc = kyoto_counts()

    gt = gp = hit = 0
    seen_g = seen_h = new_g = new_h = 0
    missed, spurious = [], []
    for line in pathlib.Path(a.gold).read_text(encoding="utf-8").rstrip("\n").split("\n"):
        gold = line.split()
        raw = "".join(gold)
        pred = m.to_csl(raw).split()
        assert "".join(pred) == raw, "segmenter did not preserve the input string"
        g, q = spans(gold), spans(pred)
        gs, qs = {(s, e) for s, e, _ in g}, {(s, e) for s, e, _ in q}
        gt += len(gs); gp += len(qs); hit += len(gs & qs)
        for s, e, w in g:
            if len(w) == 1: continue
            ok = (s, e) in qs
            if kc[w]: seen_g += 1; seen_h += ok
            else:     new_g  += 1; new_h  += ok
            if not ok: missed.append((w, kc[w]))
        for s, e, w in q:
            if len(w) > 1 and (s, e) not in gs: spurious.append((w, kc[w]))

    P, R = hit / gp, hit / gt
    print(f"strict token   P {P:.4f}  R {R:.4f}  F {2*P*R/(P+R):.4f}   ({gt} gold tokens)")
    print(f"multi-char recall, ATTESTED in Kyoto : {seen_h}/{seen_g} = {seen_h/max(seen_g,1):.4f}")
    print(f"multi-char recall, UNATTESTED (0x)   : {new_h}/{new_g} = {new_h/max(new_g,1):.4f}")
    print(f"\nmissed multi-char golds ({len(missed)}):")
    for (w, k), n in collections.Counter(missed).most_common():
        print(f"   {w:8} x{n}  kyoto {k}")
    print(f"spurious multi-char predictions ({len(spurious)}):")
    for (w, k), n in collections.Counter(spurious).most_common(): print(f"   {w:8} x{n}  kyoto {k}")

if __name__ == "__main__":
    main()
