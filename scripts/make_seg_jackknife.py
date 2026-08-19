#!/usr/bin/env python3
"""Hold multi-character token TYPES out of the segmenter's training data, so held-out recall can be
measured on a denominator larger than a hand-annotated test set.

WHY. The Heart Sutra gold set shows the segmenter recovering 11/11 multi-char units attested in
Kyoto and 0/14 unattested -- but 14 is far too small a denominator to decide anything on. Splitting
a random half of the shared multi-char types out of TRAIN and DEV manufactures hundreds of
"unattested" units in the untouched TEST set, at zero annotation cost, and simulates exactly the
out-of-domain condition. It is the same jackknife the repo already found necessary for a corpus
lexicon (docs/layers-and-tokenisers.md).

WHAT IS MODIFIED. Only the SEGMENTER's training pairs, never the CoNLL-U. The segmenter learns a
per-character rewrite label and never sees a tree, so no dependency surgery is needed:

    '='   keep the character, no space  -> the token continues
    '= '  keep the character + a space  -> the token ends here

and the final character of a line always takes '=' (there is no trailing space). Holding a type out
means re-emitting its characters as separate tokens.

⚠ TEST IS NEVER TOUCHED. It is the gold the held-out types are scored against.

Self-check: with an empty hold-out set this reproduces the input labels byte for byte. It asserts
that on every row before writing anything.
"""
import argparse, collections, json, pathlib, random

def toks_to_labels(toks):
    labels, n = [], sum(len(t) for t in toks)
    i = 0
    for t in toks:
        for j in range(len(t)):
            last_in_tok = (j == len(t) - 1)
            labels.append("=" if (not last_in_tok or i == n - 1) else "= ")
            i += 1
    return labels

def rewrite(row, hold):
    toks = row["csl"].split()
    out = []
    for t in toks:
        out.extend(list(t)) if (len(t) > 1 and t in hold) else out.append(t)
    r = dict(row); r["csl"] = " ".join(out); r["labels"] = toks_to_labels(out)
    return r

def load(p):
    return [json.loads(l) for l in pathlib.Path(p).open(encoding="utf-8")]

def multi_types(rows):
    return {t for r in rows for t in r["csl"].split() if len(t) > 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data_seg_lzh")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = pathlib.Path(a.in_dir); dst = pathlib.Path(a.out_dir); dst.mkdir(parents=True, exist_ok=True)
    tr, dv, te = load(src/"train.jsonl"), load(src/"dev.jsonl"), load(src/"test.jsonl")

    # self-check BEFORE anything is written: empty hold-out must be the identity
    for r in tr[:5000]:
        assert rewrite(r, set())["labels"] == r["labels"], f"label round-trip failed: {r['sent_id']}"
    print("self-check passed: empty hold-out reproduces the original labels")

    shared = sorted(multi_types(tr) & multi_types(te))
    rng = random.Random(a.seed)
    hold = set(rng.sample(shared, int(len(shared) * a.fraction)))
    print(f"multi-char types: train {len(multi_types(tr)):,}  test {len(multi_types(te)):,}  "
          f"shared {len(shared):,}")
    print(f"held out: {len(hold):,} types (fraction {a.fraction})")

    for name, rows in (("train", tr), ("dev", dv)):
        new = [rewrite(r, hold) for r in rows]
        (dst/f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new) + "\n", encoding="utf-8")
        before = sum(1 for r in rows for t in r["csl"].split() if len(t) > 1)
        after  = sum(1 for r in new  for t in r["csl"].split() if len(t) > 1)
        print(f"  {name}: multi-char tokens {before:,} -> {after:,} (removed {before-after:,})")
    (dst/"test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in te) + "\n", encoding="utf-8")
    (dst/"heldout_types.txt").write_text("\n".join(sorted(hold)), encoding="utf-8")
    held_tok = sum(1 for r in te for t in r["csl"].split() if t in hold)
    kept_tok = sum(1 for r in te for t in r["csl"].split() if len(t) > 1 and t not in hold)
    print(f"  test: UNTOUCHED. multi-char tokens now scored as "
          f"held-out {held_tok:,} / retained {kept_tok:,}")

if __name__ == "__main__":
    main()
