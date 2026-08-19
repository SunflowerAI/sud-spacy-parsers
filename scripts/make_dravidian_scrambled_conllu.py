#!/usr/bin/env python3
"""Render a ta/te test set in a fresh word order — same trees, same gold, only the string moves.

The counterpart of `make_la_scrambled_conllu.py`. An order-augmented arm is a TRADE, never a free
gain: it gives up some accuracy on the order the treebank happens to use, in exchange for not
collapsing on an order it does not. Reporting only the natural-order number measures the cost and
none of the benefit, which would make the arm look simply worse than its baseline.

    identity     the control. Same file, same order — it MUST score exactly what an ordinary test
                 run scores, and that is what makes every other column believable.
    order        the training policy, applied at p_sentence = 1.0
    order_free   `clause_only = 0` — scrambling under nominal heads too
    order_nohyp  no displacement, so every sentence stays projective

⚠ `--check` asserts the transform is a permutation that preserves the tree, for the reason
`check_dravidian_order.py` gives: a HEAD bug does not raise, it yields a well-formed sentence with a
different tree.

    make_dravidian_scrambled_conllu.py IN.conllu OUT.conllu --lang ta --style order
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dravidian_order import POLICIES, OrderPolicy, Tok, reorder_sentence  # noqa: E402

STYLES = {
    "identity": dict(p_sentence=0.0),
    "order": dict(p_sentence=1.0),
    "order_free": dict(p_sentence=1.0, clause_only=False),
    "order_nohyp": dict(p_sentence=1.0, p_hyperbaton=0.0),
}


def read_sentences(path):
    comments, rows = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if rows:
                yield comments, rows
            comments, rows = [], []
            continue
        if line.startswith("#"):
            comments.append(line)
        else:
            rows.append(line.split("\t"))
    if rows:
        yield comments, rows


def reorder_rows(rows, rng, policy):
    """Permute the word rows, re-indexing HEAD through the permutation.

    ⚠ MULTIWORD RANGES ARE DROPPED, deliberately. A range says which ORTHOGRAPHIC word a run of
    syntactic words came from, and after a permutation that run is no longer contiguous — so the
    range would be a false statement about the new string. These files exist to score a parser
    under `--gold-preproc`, which never consults them.
    """
    ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
    index = {int(c[0]): i for i, c in enumerate(ws)}
    toks = [Tok(form=c[1], lemma=c[2], upos=c[3], deprel=c[7],
                head=index[int(c[6])] if c[6] != "0" and int(c[6]) in index else -1,
                feats=c[5], space_after="SpaceAfter=No" not in c[9])
            for c in ws]
    r = reorder_sentence(toks, rng, policy)
    where = {old: new for new, old in enumerate(r.order)}
    out = []
    for new_i, old_i in enumerate(r.order):
        c = list(ws[old_i])
        c[0] = str(new_i + 1)
        old_head = int(c[6])
        c[6] = "0" if old_head == 0 else str(where[index[old_head]] + 1)
        misc = [m for m in c[9].split("|") if m and m != "SpaceAfter=No"]
        if not r.spaces[new_i]:
            misc.append("SpaceAfter=No")
        c[9] = "|".join(misc) if misc else "_"
        out.append(c)
    return out


def check(path, lang):
    """The tree must survive. Compared as arcs between TOKEN IDENTITIES, not between positions."""
    rng = random.Random(0)
    base = POLICIES[lang]
    policy = OrderPolicy(p_sentence=1.0, p_hyperbaton=base.p_hyperbaton)
    bad = n = 0
    for _comments, rows in read_sentences(path):
        ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
        if len(ws) < 2:
            continue
        n += 1
        new = reorder_rows(rows, rng, policy)
        if sorted(c[1] for c in ws) != sorted(c[1] for c in new):
            bad += 1
            continue
        # arcs as (head form-and-old-index, dep form-and-old-index) survive a permutation
        ROOT = ("", "", "")          # a tuple, so the sort never compares str against tuple
        old_key = {int(c[0]): (c[1], c[2], c[7]) for c in ws}
        want = sorted((old_key[int(c[6])] if c[6] != "0" else ROOT, old_key[int(c[0])])
                      for c in ws)
        new_key = {int(c[0]): (c[1], c[2], c[7]) for c in new}
        got = sorted((new_key[int(c[6])] if c[6] != "0" else ROOT, new_key[int(c[0])])
                     for c in new)
        bad += want != got
    print(f"{path}: {n} sentences, {n - bad} preserved the tree exactly"
          f"{'' if not bad else f' -- {bad} FAILURES'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--lang", required=True, choices=sorted(POLICIES))
    ap.add_argument("--style", default="order", choices=sorted(STYLES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return check(args.src, args.lang)

    base = POLICIES[args.lang]
    policy = OrderPolicy(**{"p_hyperbaton": base.p_hyperbaton, **STYLES[args.style]})
    rng = random.Random(args.seed)
    out_lines, moved, total = [], 0, 0
    for comments, rows in read_sentences(args.src):
        new = reorder_rows(rows, rng, policy)
        ws = [c for c in rows if len(c) == 10 and c[0].isdigit()]
        total += 1
        moved += [c[1] for c in ws] != [c[1] for c in new]
        # `# text` is rebuilt: the sentence really is in a different order now, and leaving the old
        # one would make the file say something false about itself.
        kept = [line for line in comments if not line.startswith("# text =")]
        text = ""
        for c in new:
            text += c[1] + ("" if "SpaceAfter=No" in c[9] else " ")
        out_lines.extend(kept + [f"# text = {text.strip()}"])
        out_lines.extend("\t".join(c) for c in new)
        out_lines.append("")
    pathlib.Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  style={args.style}  {moved}/{total} sentences re-ordered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
