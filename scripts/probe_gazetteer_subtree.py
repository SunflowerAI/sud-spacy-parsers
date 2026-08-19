#!/usr/bin/env python3
"""Ceiling for "merge a gazetteer match only when it is a COMPLETE SUBTREE".

WHY A CEILING AND NOT A SYSTEM. Tokenisation feeds the parser, so at runtime no tree exists when the
merge decision is made. This uses GOLD trees to ask the prior question: IF you had a perfect parse,
would the subtree constraint remove the gazetteer's false positives? If it does not, no amount of
architecture (two-pass, char-level pre-parse, joint decoding) can rescue the idea.

⚠ THE CIRCULARITY TO AVOID. Kyoto ships multi-character names as SINGLE tokens in every generation
of the chain (13,095 tokens, identical in `.conllu`, `.punct` and `.rulemerged`), so a correct match
is one node and is trivially a subtree. Scoring that would measure nothing. The test therefore only
has force on the SPURIOUS matches -- spans covering two or more gold tokens -- where the constraint
either rejects them or does not.

A span is a complete subtree iff exactly one of its tokens has its head outside the span, and that
token's descendant set is exactly the span.
"""
import argparse, collections, pathlib

def read(p):
    sents, cur = [], []
    for line in pathlib.Path(p).open(encoding="utf-8"):
        if line.startswith("#"): continue
        if not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]: continue
        ext = ""
        for kv in f[5].split("|"):
            if kv.startswith("ExtPos="): ext = kv.split("=", 1)[1]
        cur.append((f[1], int(f[6]), f[3], ext))   # form, head (1-indexed, 0=root), upos, ExtPos
    if cur: sents.append(cur)
    return sents

def is_subtree(idx, heads):
    """idx: sorted 0-based token indices. heads: 1-indexed head per token, 0 = root."""
    S = set(idx)
    roots = [i for i in idx if heads[i] == 0 or (heads[i] - 1) not in S]
    if len(roots) != 1: return False
    r = roots[0]
    kids = collections.defaultdict(list)
    for i, h in enumerate(heads):
        if h: kids[h - 1].append(i)
    seen, stack = set(), [r]
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); stack.extend(kids[n])
    return seen == S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--names", default="assets_dila/person_names.txt")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--head-pos", default="NOUN,PROPN",
                    help="restrict to subtrees whose head UPOS (or ExtPos) is one of these")
    a = ap.parse_args()

    names = {n for n in pathlib.Path(a.names).read_text(encoding="utf-8").split("\n")
             if a.min_len <= len(n) <= a.max_len}
    by_len = collections.defaultdict(set)
    for n in names: by_len[len(n)].add(n)
    maxlen = max(by_len)
    print(f"gazetteer entries: {len(names):,}")

    POS = set(a.head_pos.split(","))
    def head_ok(idx, heads, upos, ext):
        roots = [i for i in idx if heads[i] == 0 or (heads[i] - 1) not in set(idx)]
        if len(roots) != 1: return False
        r = roots[0]
        return (ext[r] or upos[r]) in POS
    tp = tp_pos = fp_sub = fp_sub_pos = fp_nosub = 0
    by_len_tp = collections.Counter(); by_len_fp = collections.Counter()
    gold_by_len = collections.Counter()
    gold_multi = 0
    misaligned = 0
    for toks in read(a.conllu):
        forms = [t for t, _, _, _ in toks]; heads = [h for _, h, _, _ in toks]
        upos = [u for _, _, u, _ in toks]; ext = [e for _, _, _, e in toks]
        raw = "".join(forms)
        start, off = [], 0
        for f in forms: start.append(off); off += len(f)
        pos = {s: i for i, s in enumerate(start)}
        end = {s + len(forms[i]): i for i, s in enumerate(start)}
        gold_multi += sum(1 for f in forms if len(f) > 1)
        for f in forms:
            if len(f) > 1: gold_by_len[len(f)] += 1
        i = 0
        while i < len(raw):
            for L in range(min(maxlen, len(raw) - i), a.min_len - 1, -1):
                if raw[i:i + L] in by_len[L]:
                    if i in pos and (i + L) in end and forms[pos[i]] == raw[i:i + L]:
                        tp += 1                                   # exactly a gold token
                        k = pos[i]
                        if (ext[k] or upos[k]) in POS: tp_pos += 1; by_len_tp[L] += 1
                    elif i in pos and (i + L) in end:
                        idx = list(range(pos[i], end[i + L] + 1))
                        if is_subtree(idx, heads):
                            fp_sub += 1
                            if head_ok(idx, heads, upos, ext):
                                fp_sub_pos += 1; by_len_fp[L] += 1
                        else: fp_nosub += 1
                    else:
                        misaligned += 1                            # crosses a token boundary
                    i += L; break
            else:
                i += 1
    proposed = tp + fp_sub + fp_nosub + misaligned
    print(f"gold multi-char tokens: {gold_multi:,}")
    print(f"gazetteer proposals   : {proposed:,}")
    print(f"  correct (= a gold token)          {tp:5}")
    print(f"  wrong, span IS a complete subtree {fp_sub:5}")
    print(f"  wrong, span is NOT a subtree      {fp_nosub:5}")
    print(f"  wrong, crosses a token boundary   {misaligned:5}")
    def pr(name, keep_tp, keep_fp):
        P = keep_tp / (keep_tp + keep_fp) if keep_tp + keep_fp else 0.0
        R = keep_tp / gold_multi
        print(f"  {name:44} P {P:.4f}  R {R:.4f}  F {2*P*R/(P+R) if P+R else 0:.4f}")
    print(f"  (of the {tp} correct, {tp_pos} are headed by {sorted(POS)};"
          f" of the {fp_sub} wrong subtrees, {fp_sub_pos} are)")
    pr("gazetteer alone", tp, fp_sub + fp_nosub + misaligned)
    pr("+ gold-tree subtree", tp, fp_sub)
    pr("+ gold-tree subtree + head POS", tp_pos, fp_sub_pos)
    print("\n  subtree + head-POS, broken down by MATCH LENGTH:")
    print(f"    {'len':>4} {'correct':>8} {'wrong':>7} {'precision':>10} {'gold at len':>12} {'recall':>8}")
    cum_tp = cum_fp = 0
    for L in sorted(set(by_len_tp) | set(by_len_fp)):
        t, f_ = by_len_tp[L], by_len_fp[L]
        print(f"    {L:>4} {t:>8} {f_:>7} {t/max(t+f_,1):>10.4f} {gold_by_len[L]:>12} "
              f"{t/max(gold_by_len[L],1):>8.4f}")
    for L in sorted(set(by_len_tp) | set(by_len_fp)):
        if L >= 3: cum_tp += by_len_tp[L]; cum_fp += by_len_fp[L]
    print(f"    length >= 3 only: correct {cum_tp}, wrong {cum_fp}, "
          f"precision {cum_tp/max(cum_tp+cum_fp,1):.4f}, recall {cum_tp/gold_multi:.4f}")

if __name__ == "__main__":
    main()
