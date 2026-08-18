#!/usr/bin/env python3
"""Rewrite a Latin CoNLL-U file into a re-linearised word order, for evaluation.

The counterpart of ``make_la_variant_conllu.py`` on the other axis: that one renders the test set in
a different ORTHOGRAPHY, this one in a different ORDER. The trees are identical either way — the
same arcs between the same words, only re-indexed — so every arm is scored against the same gold
and a drop is attributable to the linearisation and nothing else.

    make_la_scrambled_conllu.py IN.conllu OUT.conllu --style order
    make_la_scrambled_conllu.py IN.conllu --check      # identity round-trip

``--check`` renders with ``p_sentence = 0`` and asserts the output is the input byte for byte, which
is the test that the renumbering, the ``# text`` line and the multiword ranges all survive a pass
that is supposed to change nothing.

MULTIWORD RANGES. Only the 152 joined ``-que`` hosts carry one. A range is re-emitted when its
member tokens are still contiguous and in the same relative order (which the enclitic constraint
guarantees for exactly those) and dropped otherwise, since a range line over non-adjacent tokens is
not a fused orthographic word.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from la_order import OrderPolicy, Tok, reorder_sentence  # noqa: E402

#: Named settings, so an evaluation names a policy rather than four numbers.
STYLES: dict[str, OrderPolicy] = {
    "identity":  OrderPolicy(p_sentence=0.0),
    #: what the augmenter itself produces, at the calibrated displacement rate
    "order":     OrderPolicy(p_sentence=1.0, p_hyperbaton=0.08),
    #: projective re-linearisation only -- no discontinuity at all
    "order_proj": OrderPolicy(p_sentence=1.0, p_hyperbaton=0.0),
    #: heavy hyperbaton, well past what the corpus contains
    "order_hyper": OrderPolicy(p_sentence=1.0, p_hyperbaton=0.30),
    #: the closed-class constraints switched OFF -- an ABLATION, not a plausible edition. It is
    #: what "randomise the word order" would mean taken literally, and the gap between this and
    #: `order` is the value of every rule in la_order.py.
    "order_free": OrderPolicy(p_sentence=1.0, p_hyperbaton=0.08, respect_adp=False,
                              respect_sconj=False, respect_rel=False, respect_cc=False,
                              respect_conj=False, respect_wackernagel=False,
                              respect_enclitics=False, respect_punct=False),
}


def space_after(misc: str) -> bool:
    return "SpaceAfter=No" not in ("" if misc == "_" else misc).split("|")


def set_space_after(misc: str, has_space: bool) -> str:
    parts = [p for p in ("" if misc == "_" else misc).split("|") if p and p != "SpaceAfter=No"]
    if not has_space:
        parts.append("SpaceAfter=No")
    return "|".join(parts) if parts else "_"


def render_text(rows: list[list[str]]) -> str:
    out = []
    for f in rows:
        out.append(f[1])
        if space_after(f[9]):
            out.append(" ")
    return "".join(out).strip()


def transform_sentence(block: list[str], policy: OrderPolicy, rng: random.Random) -> list[str]:
    comments = [ln for ln in block if ln.startswith("#")]
    rows = [ln.split("\t") for ln in block if not ln.startswith("#")]
    tokens = [f for f in rows if f[0].isdigit()]
    ranges = [f for f in rows if "-" in f[0]]
    if len(tokens) < 2:
        return block

    toks = [Tok(form=f[1], lemma=f[2] if f[2] != "_" else "", upos=f[3], deprel=f[7],
                head=int(f[6]) - 1, feats="" if f[5] == "_" else f[5],
                space_after=space_after(f[9]))
            for f in tokens]
    r = reorder_sentence(toks, rng, policy)
    if r.order == list(range(len(toks))):
        return block

    where = {old: new for new, old in enumerate(r.order)}
    out_tokens = []
    for new_i, old_i in enumerate(r.order):
        f = list(tokens[old_i])
        f[0] = str(new_i + 1)
        f[1] = r.forms[new_i]
        old_head = int(f[6])
        f[6] = "0" if old_head == 0 else str(where[old_head - 1] + 1)
        f[9] = set_space_after(f[9], r.spaces[new_i])
        out_tokens.append(f)

    out_ranges = []
    for f in ranges:
        a, b = (int(x) - 1 for x in f[0].split("-"))
        members = [where[i] for i in range(a, b + 1)]
        if members == list(range(members[0], members[0] + len(members))):
            g = list(f)
            g[0] = f"{members[0] + 1}-{members[-1] + 1}"
            out_ranges.append((members[0], g))

    lines = list(comments)
    body: list[list[str]] = []
    starts = {pos: g for pos, g in out_ranges}
    for k, f in enumerate(out_tokens):
        if k in starts:
            body.append(starts[k])
        body.append(f)
    text = render_text(out_tokens)
    lines = [ln if not ln.startswith("# text =") else f"# text = {text}" for ln in lines]
    return lines + ["\t".join(f) for f in body]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path, nargs="?")
    ap.add_argument("--style", default="order", choices=sorted(STYLES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="render with p_sentence=0 and require a byte-identical round trip")
    args = ap.parse_args()

    policy = OrderPolicy(p_sentence=0.0) if args.check else STYLES[args.style]
    rng = random.Random(args.seed)

    src_lines = args.src.read_text(encoding="utf8").split("\n")
    out: list[str] = []
    block: list[str] = []
    for line in src_lines:
        if line.strip():
            block.append(line)
            continue
        if block:
            out.extend(transform_sentence(block, policy, rng))
            block = []
        out.append("")
    if block:
        out.extend(transform_sentence(block, policy, rng))
    text = "\n".join(out)

    if args.check:
        original = args.src.read_text(encoding="utf8")
        if text != original:
            for i, (a, b) in enumerate(zip(text.split("\n"), original.split("\n"))):
                if a != b:
                    raise SystemExit(f"round trip differs at line {i + 1}:\n  got {a!r}\n  want {b!r}")
            raise SystemExit("round trip differs in length")
        print(f"{args.src.name}: identity round trip is byte-identical")
        return

    if args.dest is None:
        raise SystemExit("give an output path, or pass --check")
    args.dest.write_text(text, encoding="utf8")
    print(f"wrote {args.dest} (style {args.style})")


if __name__ == "__main__":
    main()
