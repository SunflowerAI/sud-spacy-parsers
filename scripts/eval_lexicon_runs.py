#!/usr/bin/env python3
"""Lexicon merging on Buddhist text, gated by a transliteration RUN.

THE DIVISION OF LABOUR. A lexicon gives exact SPANS but only for entries it holds; a run of
transliteration characters gives a REGION but no boundaries (般若波羅蜜多 is one run covering TWO
tokens). So the run cannot propose merges on its own -- it gates the lexicon, which is what makes
the pair useful: precision from the run, boundaries from the lexicon.

⚠ THE RUN CUE IS POPULATION-SPECIFIC AND ITS AGGREGATE IS MISLEADING. On the Kyoto-wide
character-pair probe it is flat (identity+runs 0.200 vs identity 0.201) because only ~4 % of that
slice is transliteration at all; on Buddhist text it reaches P 0.91-0.93. Never read this cue off a
mixed corpus.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_translit_channel import load_translit, runs_of

def spans(ws):
    out, i = [], 0
    for w in ws: out.append((i, i + len(w))); i += len(w)
    return out

def merge(raw, lex, maxlen):
    out, i = [], 0
    while i < len(raw):
        for L in range(min(maxlen, len(raw) - i), 1, -1):
            if raw[i:i + L] in lex:
                out.append((i, i + L)); i += L; break
        else:
            i += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--scores", default="assets_cbeta/translit_scores_leakfree.tsv")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--min-run", type=int, default=3)
    a = ap.parse_args()
    sc, high = load_translit(a.scores, a.threshold)
    wik = {t for t in pathlib.Path("assets_wiktionary/zh_sanskrit_terms.txt")
           .read_text(encoding="utf-8").split("\n") if len(t) >= 2}
    dila = {t for t in pathlib.Path("assets_dila/person_names.txt")
            .read_text(encoding="utf-8").split("\n") if 2 <= len(t) <= 6}
    LEX = {"wiktionary": wik, "DILA": dila, "both": wik | dila}
    for spec in a.gold:
        name, _, path = spec.partition("=")
        lines = [l for l in pathlib.Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]
        gold_multi = sum(1 for l in lines for w in l.split() if len(w) > 1)
        print(f"\n== {name}: {gold_multi} gold multi-char tokens")
        print(f"   {'lexicon':>12} {'gate':>10} {'correct':>8} {'wrong':>6} {'precision':>10} {'recall':>8}")
        for lname, lex in LEX.items():
            mx = max((len(t) for t in lex), default=2)
            for gate in (False, True):
                tp = fp = 0
                for l in lines:
                    toks = l.split(); raw = "".join(toks)
                    g = {s for s in spans(toks) if s[1] - s[0] > 1}
                    r = runs_of(list(raw), high)
                    for (s, e) in merge(raw, lex, mx):
                        if gate and not all(r[k] >= a.min_run for k in range(s, e)):
                            continue
                        if (s, e) in g: tp += 1
                        else: fp += 1
                P = tp / max(tp + fp, 1); R = tp / max(gold_multi, 1)
                print(f"   {lname:>12} {('run>=%d' % a.min_run) if gate else 'none':>10} "
                      f"{tp:>8} {fp:>6} {P:>10.4f} {R:>8.4f}")

if __name__ == "__main__":
    main()
