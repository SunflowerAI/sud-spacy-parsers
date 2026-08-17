#!/usr/bin/env python3
"""Score a segmenter on the untouched test split, separating HELD-OUT from RETAINED multi-char types.

The held-out column is an honest estimate of generalisation to a multi-character token the model has
never seen merged -- the condition the Heart Sutra measures on a denominator of 14, measured here on
several hundred. The retained column is the memorisation control and should barely move between
arms; if it does, the arms differ in something other than the hold-out.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

def spans(ws):
    out, i = [], 0
    for w in ws: out.append((i, i + len(w), w)); i += len(w)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data_seg_lzh_jk")
    a = ap.parse_args()
    from sa_presegment import Presegmenter
    m = Presegmenter.from_disk(pathlib.Path(a.model))
    d = pathlib.Path(a.data)
    hold = set((d / "heldout_types.txt").read_text(encoding="utf-8").split("\n"))
    rows = [json.loads(l) for l in (d / "test.jsonl").open(encoding="utf-8")]

    gt = gp = hit = 0
    h_g = h_h = r_g = r_h = 0
    for row in rows:
        gold = row["csl"].split(); raw = row["samhita"]
        pred = m.to_csl(raw).split()
        if "".join(pred) != raw:      # a segmenter must not alter the string
            raise SystemExit(f"segmenter changed the input: {row['sent_id']}")
        g, q = spans(gold), spans(pred)
        gs = {(s, e) for s, e, _ in g}; qs = {(s, e) for s, e, _ in q}
        gt += len(gs); gp += len(qs); hit += len(gs & qs)
        for s, e, w in g:
            if len(w) == 1: continue
            ok = (s, e) in qs
            if w in hold: h_g += 1; h_h += ok
            else:         r_g += 1; r_h += ok
    P, R = hit / gp, hit / gt
    print(f"{a.model}")
    print(f"  strict token          P {P:.4f}  R {R:.4f}  F {2*P*R/(P+R):.4f}   ({gt:,} gold tokens)")
    print(f"  multi-char RETAINED   {r_h:5}/{r_g:<5} = {r_h/max(r_g,1):.4f}   (memorisation control)")
    print(f"  multi-char HELD-OUT   {h_h:5}/{h_g:<5} = {h_h/max(h_g,1):.4f}   (generalisation)")

if __name__ == "__main__":
    main()
