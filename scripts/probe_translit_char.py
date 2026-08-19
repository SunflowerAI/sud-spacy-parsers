#!/usr/bin/env python3
"""Can IDS / radical / Qieyun predict that a character is used for TRANSLITERATION?

WHY THIS IS NEWLY ANSWERABLE. Feeding Qieyun into the segmenter gave it only the segmentation
objective to learn from, and it moved nothing. Wiktionary's `Chinese terms borrowed from Sanskrit`
supplies explicit labels, so a dedicated classifier can be trained on the question directly and its
output used as a feature. Supervision, not representation, was the missing piece.

POSITIVES are the characters of multi-character BORROWED terms (228 terms, 248 characters).
"Borrowed" is used rather than "derived" on purpose: `derived from Sanskrit` also contains calques
(七寶 "seven treasures", 三千大千世界), whose characters are ordinary classical vocabulary and would
poison the label.

NEGATIVES are frequent kanripo characters appearing in NO Sanskrit-derived term (the wider
`derived` union is used for exclusion, so the negatives are clean even if the positive set is not).

⚠ RANDOM FOLDS, AND A NULL. `NEGATIVE-RESULTS.md` records this exact probe scoring BELOW its null
because `cross_val_predict` used contiguous folds over codepoint-sorted characters, which groups
characters by Unicode block and therefore by radical. Folds here are shuffled with a fixed seed, and
the bias-only null is reported alongside every arm.
"""
import argparse, collections, csv, pathlib, random

def radical_table(p="assets_unihan/Unihan_IRGSources.txt"):
    t={}
    for line in pathlib.Path(p).open(encoding="utf-8"):
        if line.startswith("#") or "kRSUnicode" not in line: continue
        cp,_,v=line.rstrip("\n").split("\t")[:3]
        t[chr(int(cp[2:],16))]=v.split()[0].split(".")[0]
    return t

def qieyun_table(p="assets_qieyun/guangyun.csv"):
    t=collections.defaultdict(list)
    with pathlib.Path(p).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ch,code=row.get("字頭"),row.get("音韻地位")
            if ch and code and code not in t[ch]: t[ch].append(code)
    return t

def ids_table(p):
    t={}
    fp=pathlib.Path(p)
    if not fp.exists(): return t
    for line in fp.open(encoding="utf-8"):
        if line.startswith("#"): continue
        parts=line.rstrip("\n").split("\t")
        if len(parts)>=3: t[parts[1]]=parts[2]
    return t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ids", default="/Users/sivakalyan/Linguistics/Tools/SUD-aptness/assets_ids/ids.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, KFold

    bor=[t for t in pathlib.Path("assets_wiktionary/zh_sanskrit_borrowed.txt").read_text(encoding="utf-8").split("\n") if len(t)>=2]
    wider=set(pathlib.Path("assets_wiktionary/zh_sanskrit_terms.txt").read_text(encoding="utf-8").split("\n"))
    pos=set("".join(bor))
    excl=pos | set("".join(w for w in wider if w))
    kan=pathlib.Path("corpus_lzh_kanripo_leakfree.txt").read_text(encoding="utf-8",errors="replace")
    freq=collections.Counter(c for c in kan if '一'<=c<='鿿')
    neg=[c for c,_ in freq.most_common(4000) if c not in excl]
    rng=random.Random(a.seed); rng.shuffle(neg)
    neg=neg[:len(pos)*3]
    print(f"positives {len(pos)}  negatives {len(neg)}  (base rate {len(pos)/(len(pos)+len(neg)):.3f})")

    rad, qy, ids = radical_table(), qieyun_table(), ids_table(a.ids)
    print(f"tables: radical {len(rad):,}  qieyun {len(qy):,}  ids {len(ids):,}")
    chars=sorted(pos)+neg; y=[1]*len(pos)+[0]*len(neg)
    def feats(ch, use_rad, use_qy, use_ids):
        d={}
        if use_rad: d[f"rad={rad.get(ch,'?')}"]=1
        if use_qy:
            for c in (qy.get(ch) or ["?"]): d[f"qy={c}"]=1          # BAG of readings, not the first
        if use_ids:
            for comp in ids.get(ch,""):
                if comp!=ch: d[f"ids={comp}"]=1
        return d
    kf=KFold(n_splits=a.folds, shuffle=True, random_state=a.seed)   # shuffle=True is load-bearing
    print(f"\n{'arm':>22} {'accuracy':>9} {'pos-F':>7} {'precision':>10} {'recall':>8}")
    maj=max(sum(y), len(y)-sum(y))/len(y)
    print(f"{'NULL (majority)':>22} {maj:>9.4f} {0.0:>7.4f} {'--':>10} {'--':>8}")
    for name, ur, uq, ui in (("radical",1,0,0), ("qieyun",0,1,0), ("IDS",0,0,1),
                             ("radical+qieyun",1,1,0), ("all three",1,1,1)):
        if ui and not ids:
            print(f"{name:>22}  (ids.txt unavailable)"); continue
        v=DictVectorizer(); X=v.fit_transform([feats(c,ur,uq,ui) for c in chars])
        clf=LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        p=cross_val_predict(clf, X, y, cv=kf)
        tp=sum(1 for a_,b in zip(y,p) if a_==1 and b==1)
        fp=sum(1 for a_,b in zip(y,p) if a_==0 and b==1)
        fn=sum(1 for a_,b in zip(y,p) if a_==1 and b==0)
        acc=sum(1 for a_,b in zip(y,p) if a_==b)/len(y)
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
        F=2*P*R/(P+R) if P+R else 0
        print(f"{name:>22} {acc:>9.4f} {F:>7.4f} {P:>10.4f} {R:>8.4f}")

if __name__=="__main__":
    main()
