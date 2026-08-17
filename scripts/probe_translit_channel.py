#!/usr/bin/env python3
"""Does a RADICAL or QIEYUN backoff help decide character-pair MERGE, where identity cannot?

THE QUESTION. lzh's character segmenter recovers multi-character tokens by memorising them: on the
Heart Sutra it merged every unit attested in Kyoto (prajna, paramita, anuttara/samyak/sambodhi) and
missed every unit that is not (Sariputra, the 4-character bodhisattva). So the open question is not
"is the channel informative" but "does it generalise to a pair the treebank never merged".

THE SLICE THAT ANSWERS IT. Evaluate separately on test pairs whose bigram WAS merged somewhere in
train (memorisation suffices) and pairs whose bigram was NEVER merged in train (identity is useless
by construction -- a backoff representation is the only thing that could fire). Sariputra is the
second kind: both characters are common, the pair is not.

⚠ RUN THE NULL FIRST. The repo's earlier sub-character probe scored BELOW its null because
`cross_val_predict` used contiguous folds over codepoint-sorted characters, which groups characters
by Unicode block and therefore by radical. This uses the treebank's OWN train/test split, so that
particular leak cannot occur -- but the bias-only null is still reported for every slice.

Qieyun is a BAG of a character's readings, not its reading here: 29.9 % of lzh character tokens are
polyphonic and the reading cannot be chosen at inference.
"""
import argparse, collections, csv, pathlib, sys

def read_sents(path):
    sents, cur = [], []
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#"):
            continue
        if not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]: continue
        cur.append(f[1])
    if cur: sents.append(cur)
    return sents

def load_translit(path, threshold=2.0):
    """char -> Buddhist/classical log-odds, plus the above-threshold set.

    The inventory is INDUCED, not curated: log-odds of each character's frequency in CBETA T08
    against 42 M characters of kanripo. It reproduces the curated lists well (蜜 rank 6, 訶 8,
    薩 16, 菩 18, 耨 29, 羅 38) -- and shows why a per-character score is not enough: 帝 ranks 976
    with log-odds −1.12, because 帝 "emperor" is ordinary classical vocabulary. 揭帝 is only visible
    as a RUN.
    """
    sc = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip(): continue
        ch, v = line.split("\t"); sc[ch] = float(v)
    return sc, {c for c, v in sc.items() if v >= threshold}

def runs_of(chars, high):
    """run[i] = length of the maximal run of above-threshold characters containing position i."""
    n = len(chars); run = [0] * n; i = 0
    while i < n:
        if chars[i] in high:
            j = i
            while j < n and chars[j] in high: j += 1
            for k in range(i, j): run[k] = j - i
            i = j
        else:
            i += 1
    return run

def gaz_cover(raw, by_len, maxlen, minlen=2):
    """cover[i] = id of the gazetteer match covering char i, else -1 (longest-match, left to right)."""
    cover = [-1] * len(raw); i = 0; k = 0
    while i < len(raw):
        for L in range(min(maxlen, len(raw) - i), minlen - 1, -1):
            if raw[i:i + L] in by_len[L]:
                for j in range(i, i + L): cover[j] = k
                k += 1; i += L; break
        else:
            i += 1
    return cover

def pairs(sents, gaz=None, tl=None):
    """Adjacent character pairs inside a sentence; label 1 = the two chars share a gold token.
    With `gaz` = (by_len, maxlen), each pair also carries whether a gazetteer entry covers BOTH
    characters -- the lexicon-as-feature signal, as opposed to the lexicon-as-oracle use."""
    out = []
    for toks in sents:
        chars, same = [], []
        for t in toks:
            for j, c in enumerate(t):
                chars.append(c); same.append(j > 0)      # same[i] = char i continues the token
        cover = gaz_cover("".join(chars), *gaz) if gaz else None
        run = runs_of(chars, tl[1]) if tl else None
        for i in range(len(chars) - 1):
            g = 0
            if cover is not None and cover[i] != -1 and cover[i] == cover[i + 1]: g = 1
            t = None
            if tl is not None:
                sc = tl[0]
                t = (min(run[i], 8), min(run[i + 1], 8),
                     int(chars[i] in tl[1]), int(chars[i + 1] in tl[1]),
                     int(round(sc.get(chars[i], 0.0))), int(round(sc.get(chars[i + 1], 0.0))))
            out.append((chars[i], chars[i + 1], int(same[i + 1]), g, t))
    return out

def load_radical(path):
    rad = {}
    for line in pathlib.Path(path).open(encoding="utf-8"):
        if line.startswith("#") or "kRSUnicode" not in line: continue
        cp, _, val = line.rstrip("\n").split("\t")[:3]
        ch = chr(int(cp[2:], 16))
        rad[ch] = val.split()[0].split(".")[0]           # radical number, drop stroke count
    return rad

def load_qieyun(path):
    qy = collections.defaultdict(set)
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ch, code = row.get("字頭"), row.get("音韻地位")
            if ch and code: qy[ch].add(code)
    return qy

def featurise(c1, c2, rad, qy, use_id, use_rad, use_qy, g=0, use_gaz=False,
              t=None, use_tl=False):
    d = {}
    if use_gaz:
        d[f"gaz={g}"] = 1
    if use_tl and t is not None:
        r1, r2, h1, h2, s1, s2 = t
        d[f"run1={r1}"] = 1; d[f"run2={r2}"] = 1
        d[f"runpair={min(r1,r2)}"] = 1              # the shared run: the mantra cue
        d[f"hi={h1}{h2}"] = 1
        d[f"s1={s1}"] = 1; d[f"s2={s2}"] = 1
    if use_id:
        d[f"c1={c1}"] = 1; d[f"c2={c2}"] = 1; d[f"bi={c1}{c2}"] = 1
    if use_rad:
        r1, r2 = rad.get(c1, "?"), rad.get(c2, "?")
        d[f"r1={r1}"] = 1; d[f"r2={r2}"] = 1; d[f"rr={r1}_{r2}"] = 1
    if use_qy:
        for q in qy.get(c1, ["?"]): d[f"q1={q}"] = 1
        for q in qy.get(c2, ["?"]): d[f"q2={q}"] = 1
    return d

def prf(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0), tp + fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--radical", default="assets_unihan/Unihan_IRGSources.txt")
    ap.add_argument("--qieyun", default="assets_qieyun/guangyun.csv")
    ap.add_argument("--translit", default=None,
                    help="TSV of char<TAB>log-odds; adds the transliteration-RUN arms")
    ap.add_argument("--tl-threshold", type=float, default=2.0)
    ap.add_argument("--gazetteer", default=None,
                    help="one name per line; adds the lexicon-as-FEATURE arms")
    a = ap.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    rad, qy = load_radical(a.radical), load_qieyun(a.qieyun)
    gaz = None
    if a.gazetteer:
        gnames = {n for n in pathlib.Path(a.gazetteer).read_text(encoding="utf-8").split("\n")
                  if 2 <= len(n) <= 6}
        by_len = collections.defaultdict(set)
        for n in gnames: by_len[len(n)].add(n)
        gaz = (by_len, max(by_len))
        print(f"gazetteer: {len(gnames):,} entries from {a.gazetteer}")
    tl = None
    if a.translit:
        tl = load_translit(a.translit, a.tl_threshold)
        print(f"translit inventory: {len(tl[0]):,} scored, {len(tl[1]):,} above "
              f"log-odds {a.tl_threshold}")
    tr, te = pairs(read_sents(a.train), gaz, tl), pairs(read_sents(a.test), gaz, tl)
    merged_bigrams = {c1 + c2 for c1, c2, y, _, _ in tr if y == 1}
    print(f"train pairs {len(tr):,} ({sum(y for _, _, y, _, _ in tr):,} merge, "
          f"{sum(y for _, _, y, _, _ in tr)/len(tr):.2%})   test pairs {len(te):,}")
    print(f"distinct bigrams merged in train: {len(merged_bigrams):,}")

    seen_idx = [i for i, (c1, c2, *_r) in enumerate(te) if c1 + c2 in merged_bigrams]
    new_idx  = [i for i, (c1, c2, *_r) in enumerate(te) if c1 + c2 not in merged_bigrams]
    print(f"test slices: bigram-merged-in-train {len(seen_idx):,} | never-merged {len(new_idx):,} "
          f"(of which true merges: {sum(te[i][2] for i in new_idx):,})\n")

    ARMS = [("null (bias only)", 0,0,0,0,0), ("identity", 1,0,0,0,0),
            ("radical+qieyun", 0,1,1,0,0), ("identity+rad+qy", 1,1,1,0,0),
            ("translit runs", 0,0,0,0,1), ("identity+runs", 1,0,0,0,1),
            ("identity+rad+qy+runs", 1,1,1,0,1)]
    print(f"{'arm':>18} | {'ALL  P/R/F':>22} | {'bigram SEEN merged':>22} | {'bigram NEVER merged':>22}")
    for name, ui, ur, uq, ug, ut in ARMS:
        ytr = [y for _, _, y, _, _ in tr]; yte = [y for _, _, y, _, _ in te]
        if not (ui or ur or uq or ug or ut):
            pred = [0] * len(te)                          # majority class is SPLIT
        else:
            v = DictVectorizer()
            X = v.fit_transform([featurise(c1, c2, rad, qy, ui, ur, uq, g, ug, t, ut)
                                 for c1, c2, _, g, t in tr])
            Xt = v.transform([featurise(c1, c2, rad, qy, ui, ur, uq, g, ug, t, ut)
                              for c1, c2, _, g, t in te])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(X, ytr); pred = list(clf.predict(Xt))
        def s(idx):
            P, R, F, n = prf([yte[i] for i in idx], [pred[i] for i in idx])
            return f"{P:.3f}/{R:.3f}/{F:.3f}"
        allP, allR, allF, _ = prf(yte, pred)
        print(f"{name:>18} | {f'{allP:.3f}/{allR:.3f}/{allF:.3f}':>22} | {s(seen_idx):>22} | {s(new_idx):>22}")

if __name__ == "__main__":
    main()
