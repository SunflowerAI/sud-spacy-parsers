#!/usr/bin/env python3
"""Remove the Sanskrit compound-join marker (``|``) from compound-**element** token FORMs.

Compounds/preverbs/privatives were emitted as an MWT range (surface ``śuka|sāri|kṛśānāṃ``)
expanding to member tokens each carrying a trailing join ``|`` (``śuka|`` / ``sāri|`` /
``kṛśānāṃ`` — the final element has no join). We no longer want that visible marker on the
element FORMs: the ``Compound=Yes`` FEAT (for samāsa members) and the ``n-m`` MWT range line
(for every grouping) already record the grouping, so the marker is redundant. This rewrites
**only plain-token FORM cells**: a trailing join ``|`` on a real word (length > 1, and NOT a
run of pipes) is dropped, so ``śuka|`` -> ``śuka``.

Left untouched, deliberately:
  * the daṇḍa PUNCT tokens ``|`` / ``||`` (a run of pipes — guarded by ``form[-2] != '|'``);
  * the ``n-m`` MWT-range **surface** line (``śuka|sāri|kṛśānāṃ`` — the recoverable surface);
  * the ``# text = …`` metadata line.

The matching runtime change lives in ``scripts/sa_tokenizer.py``: the tokeniser now emits the
compound members with no join marker, so training data and inference agree.

    strip_pipe_sa.py in.conllu out.conllu           # transform (out may equal in — in place)
    strip_pipe_sa.py in.conllu out.conllu --check    # transform + report how many were stripped
"""
import argparse
import re
import sys


def token_form_strip(form):
    """A single-token FORM: drop a trailing compound-join ``|`` on a real word.

    A run of pipes (the daṇḍa ``||``) is left alone via the ``form[-2] != '|'`` guard.
    """
    if len(form) > 1 and form.endswith("|") and form[-2] != "|":
        return form[:-1]
    return form


def convert_line(line):
    if not line or line[0] == "#" or line == "\n":
        return line
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 2:
        return line
    if re.fullmatch(r"\d+", cols[0]):        # plain token only; MWT ranges (\d+-\d+) untouched
        cols[1] = token_form_strip(cols[1])
    return "\t".join(cols) + "\n"


def convert(text):
    return "".join(convert_line(l) for l in text.splitlines(keepends=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    ap.add_argument("--check", action="store_true", help="report count + assert no join | remains")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        src = f.read()
    out = convert(src)

    if args.check:
        n = 0
        for a, b in zip(src.splitlines(), out.splitlines()):
            ca, cb = a.split("\t"), b.split("\t")
            if len(ca) >= 2 and re.fullmatch(r"\d+", ca[0]):
                if len(ca[1]) > 1 and ca[1].endswith("|") and ca[1][-2] != "|":
                    assert cb[1] == ca[1][:-1], f"unexpected strip: {ca[1]!r} -> {cb[1]!r}"
                    n += 1
                else:
                    assert cb[1] == ca[1], f"FORM changed unexpectedly: {ca[1]!r} -> {cb[1]!r}"
        # no plain-token FORM should still carry a trailing single-pipe join
        for b in out.splitlines():
            cb = b.split("\t")
            if len(cb) >= 2 and re.fullmatch(r"\d+", cb[0]):
                assert not (len(cb[1]) > 1 and cb[1].endswith("|") and cb[1][-2] != "|"), \
                    f"residual join: {cb[1]!r}"
        print(f"  stripped {n} compound-element join markers", file=sys.stderr)

    with open(args.outp, "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
