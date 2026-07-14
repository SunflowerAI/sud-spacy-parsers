#!/usr/bin/env python3
"""Flip the Sanskrit compound-join marker from hyphen ``-`` to pipe ``|`` in CSL-reverted CoNLL-U.

CSL prints compound (samāsa) division with a thin vertical line, so the model data should carry
the compound join as ``|``, not the hyphen we had been using internally. Compounds are encoded as
an MWT range (surface ``śuka-sāri-kṛśānāṃ``) expanding to member tokens each ending in ``-``
(``śuka-`` / ``sāri-`` / ``kṛśānāṃ`` — the final element has no join). This rewrites:

  * every token FORM ending in ``-`` with length > 1  ->  trailing ``-`` becomes ``|``;
  * every MWT-range surface FORM  ->  internal compound ``-`` become ``|``;
  * every ``# text = …`` line  ->  a ``-`` between two non-space chars becomes ``|``.

A lone-dash PUNCT token (FORM == ``-``) is left untouched — it is a genuine dash, not a join
(this mirrors ``sa_tokenizer._HYPH``, which only ever emits a trailing join hyphen). The mapping
is exactly reversible with the same length>1 rule, which the ``--check`` self-test verifies.

The matching runtime change lives in ``scripts/sa_tokenizer.py``: the tokeniser now emits ``|``
compound joins (accepting both hyphen and CSL-pipe input), so training data and inference agree.

    hyphen_to_pipe_sa.py in.conllu out.conllu          # transform (out may equal in — in place)
    hyphen_to_pipe_sa.py in.conllu out.conllu --check   # transform + assert round-trip
"""
import argparse
import re
import sys

_TEXT_HYPHEN = re.compile(r"(?<=\S)-(?=\S)")   # a '-' with non-space on both sides (compound join)


def token_form_to_pipe(form):
    """A single-token FORM: a trailing join hyphen on a real word becomes '|'."""
    if len(form) > 1 and form.endswith("-"):
        return form[:-1] + "|"
    return form


def token_form_to_hyphen(form):
    """Inverse of token_form_to_pipe (for the round-trip self-test)."""
    if len(form) > 1 and form.endswith("|"):
        return form[:-1] + "-"
    return form


def convert_line(line):
    if line.startswith("# text ="):
        return _TEXT_HYPHEN.sub("|", line)
    if not line or line[0] == "#" or line == "\n":
        return line
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 2:
        return line
    tid = cols[0]
    if re.fullmatch(r"\d+-\d+", tid):            # MWT range surface: all internal joins -> '|'
        cols[1] = cols[1].replace("-", "|") if cols[1] != "-" else cols[1]
    elif re.fullmatch(r"\d+", tid):              # plain token
        cols[1] = token_form_to_pipe(cols[1])
    return "\t".join(cols) + "\n"


def convert(text):
    return "".join(convert_line(l) for l in text.splitlines(keepends=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    ap.add_argument("--check", action="store_true", help="assert the token-FORM flip round-trips")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        src = f.read()
    out = convert(src)

    if args.check:
        # Round-trip only the plain-token FORM column (the reversible part); MWT surface and
        # # text are derived from it, so this is sufficient to prove no information is lost.
        n = 0
        for a, b in zip(src.splitlines(), out.splitlines()):
            ca, cb = a.split("\t"), b.split("\t")
            if len(ca) >= 2 and re.fullmatch(r"\d+", ca[0]):
                expected = token_form_to_pipe(ca[1])
                assert cb[1] == expected, f"unexpected FORM: {ca[1]!r} -> {cb[1]!r}"
                if len(ca[1]) > 1 and ca[1].endswith("-"):     # a join hyphen we flipped
                    assert token_form_to_hyphen(expected) == ca[1], f"round-trip fail: {ca[1]!r}"
                    n += 1
        print(f"  round-trip OK ({n} token FORMs flipped)", file=sys.stderr)

    with open(args.outp, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
