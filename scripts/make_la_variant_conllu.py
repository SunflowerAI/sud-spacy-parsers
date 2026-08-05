#!/usr/bin/env python3
"""Rewrite a Latin CoNLL-U file into one fixed orthographic style, for evaluation.

``la_augment.py`` samples a style per document at training time; this applies ONE, deterministically,
so a released arm can be scored on each axis separately -- macrons, breves, ``j``/``v``,
``æ``/``œ``, sentence-initial capitals -- rather than only on the two spellings the treebank
happens to contain. The trees are untouched, so every variant is scored against the same gold.

    .venv/bin/python scripts/make_la_variant_conllu.py IN.conllu OUT.conllu --style vj
    .venv/bin/python scripts/make_la_variant_conllu.py IN.conllu --check     # identity round-trip

Only FORM changes (plus the range line that spells out a multiword token, and the ``# text``
comment that has to agree with it). LEMMA stays canonical -- a variant is a different spelling of
the same word, not a different word -- so a lemmatiser scored on these files is being asked
exactly the question the augmentation is meant to teach it.

``--style plain`` on a ``*.macron.conllu`` reproduces its plain counterpart byte for byte, which is
the check that the whole transform chain is faithful (``--check`` does the identity version of the
same test on any file).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from la_orth import OrthPolicy, Style, sample_style, set_initial_case, vary_word  # noqa: E402

#: Each named style is one orthography a printed edition might actually use.
STYLES: dict[str, Style] = {
    "identity":  Style(macron_rate=1.0),                 # the macronised source, unchanged
    "plain":     Style(macron_rate=0.0),
    "macron":    Style(macron_rate=1.0),
    "mixed":     Style(macron_rate=0.5),
    "breve":     Style(macron_rate=1.0, breve_rate=1.0),
    "v":         Style(macron_rate=0.0, use_v=True),
    "vj":        Style(macron_rate=0.0, use_v=True, use_j=True),
    "lig":       Style(macron_rate=0.0, use_lig=True),
    "caps":      Style(macron_rate=0.0, capitalise=True),
    "lower":     Style(macron_rate=0.0, capitalise=False),
    "all":       Style(macron_rate=1.0, breve_rate=1.0, use_v=True, use_j=True, use_lig=True,
                       capitalise=True),
}


def space_after(misc: str) -> bool:
    return "SpaceAfter=No" not in ("" if misc == "_" else misc).split("|")


def render_text(tokens: list[list[str]]) -> str:
    """Rebuild the sentence text from the token lines.

    Multiword tokens need no special case here: ``fuse_mwt_spaceafter.py`` has already put
    ``SpaceAfter=No`` on every sub-token but the last, so walking the sub-tokens reproduces the
    fused orthographic word the range line spells out.
    """
    out = []
    for fields in tokens:
        out.append(fields[1])
        if space_after(fields[9]):
            out.append(" ")
    return "".join(out).strip()


def transform_sentence(block: list[str], style: Style, rng: random.Random,
                       policy: OrthPolicy) -> tuple[list[str], int, int]:
    """Rewrite one sentence block. Returns (lines, tokens changed, stale ``# text`` lines)."""
    rows = [(i, line.split("\t")) for i, line in enumerate(block) if not line.startswith("#")]
    tokens = {f[0]: f for _, f in rows if f[0].isdigit()}
    ranges = [f for _, f in rows if "-" in f[0]]

    old_text = render_text([f for f in tokens.values()])
    first = next(iter(tokens), None)
    changed = 0
    for tid, f in tokens.items():
        new = vary_word(f[1], style, rng, f[2] if f[2] != "_" else "")
        if tid == first and not (not style.capitalise and policy.protect_propn and f[3] == "PROPN"):
            new = set_initial_case(new, style.capitalise)
        if new != f[1]:
            changed += 1
            f[1] = new
    # A range line spells the fused surface of its sub-tokens, so it is their concatenation.
    for f in ranges:
        lo, hi = f[0].split("-")
        parts = [tokens[str(n)][1] for n in range(int(lo), int(hi) + 1) if str(n) in tokens]
        if parts:
            f[1] = "".join(parts)

    new_text = render_text([f for f in tokens.values()])
    stale = 0
    out = []
    for i, line in enumerate(block):
        if line.startswith("# text ="):
            if line[len("# text = "):] != old_text:
                stale += 1                 # the file's own text line already disagreed: leave it
            else:
                line = "# text = " + new_text
        elif not line.startswith("#"):
            line = "\t".join(dict(rows)[i])
        out.append(line)
    return out, changed, stale


def transform(text: str, style: Style, seed: int, policy: OrthPolicy,
              per_doc: bool) -> tuple[str, int, int, int]:
    rng = random.Random(seed)
    blocks = text.split("\n\n")
    out, changed, stale, toks = [], 0, 0, 0
    for block in blocks:
        if not block.strip():
            out.append(block)
            continue
        lines = block.split("\n")
        this = sample_style(rng, policy) if per_doc else style
        new_lines, n, s = transform_sentence(lines, this, rng, policy)
        toks += sum(1 for ln in new_lines if ln[:1].isdigit() and "-" not in ln.split("\t")[0])
        changed += n
        stale += s
        out.append("\n".join(new_lines))
    return "\n\n".join(out), changed, toks, stale


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("infile", type=Path)
    ap.add_argument("outfile", type=Path, nargs="?")
    ap.add_argument("--style", choices=sorted(STYLES) + ["sample"], default="identity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="apply the identity style and assert the file comes back byte-identical")
    args = ap.parse_args()

    text = args.infile.read_text(encoding="utf8")
    policy = OrthPolicy()
    if args.check:
        out, changed, toks, stale = transform(text, STYLES["identity"], args.seed, policy, False)
        ok = out == text
        print(f"identity round-trip: {'OK' if ok else 'FAILED'}  "
              f"({toks} tokens, {changed} changed, {stale} stale '# text' lines)")
        raise SystemExit(0 if ok else 1)

    style = STYLES.get(args.style, STYLES["identity"])
    out, changed, toks, stale = transform(text, style, args.seed, policy, args.style == "sample")
    if args.outfile:
        args.outfile.write_text(out, encoding="utf8")
    print(f"{args.infile.name} --{args.style}--> "
          f"{args.outfile.name if args.outfile else '(dry run)'}: "
          f"{changed}/{toks} tokens changed ({changed / max(toks, 1):.1%})"
          + (f", {stale} '# text' lines left alone (already stale)" if stale else ""))


if __name__ == "__main__":
    main()
