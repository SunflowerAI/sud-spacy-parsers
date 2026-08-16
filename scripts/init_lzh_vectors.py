#!/usr/bin/env python3
"""Turn a `.vec` file into a spaCy vectors package whose language is `lzh`.

WHY NOT `spacy init vectors`. That command takes a LANG argument but no `--code`, so it cannot
register this repo's custom `lzh` language (`scripts/lzh_tokenizer.py`) and the best it can do is
write an `xx` package. Loading one of those from `[initialize] vectors` then dies with **E150**, "the
language of the `nlp` object and the `vocab` should be the same, but found 'xx' and 'lzh'" -- the
check fires at `Language.__init__`, before anything trains, so it is loud rather than silent. This
does the same job with the language registered.

Usage:

    .venv/bin/python scripts/init_lzh_vectors.py vectors_lzh_ids_leakfree.vec vectors_lzh_leakfree
"""
import argparse
import pathlib
import sys

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import seg_code  # noqa: E402,F401  (registers the custom `lzh` language)
import spacy  # noqa: E402
from spacy.vectors import Vectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vec")
    ap.add_argument("out")
    ap.add_argument("--lang", default="lzh")
    a = ap.parse_args()

    keys, rows = [], []
    with open(a.vec, encoding="utf-8") as fh:
        n, dim = fh.readline().split()
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != int(dim) + 1:
                continue
            keys.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    data = numpy.asarray(rows, dtype="float32")
    print(f"  read {a.vec}: {len(keys)} keys x {data.shape[1]}")

    nlp = spacy.blank(a.lang)
    nlp.vocab.vectors = Vectors(
        strings=nlp.vocab.strings,
        data=data,
        keys=[nlp.vocab.strings.add(k) for k in keys],
    )
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(a.out)
    print(f"  wrote {a.out}  (lang={nlp.lang}, vectors={nlp.vocab.vectors.shape})")


if __name__ == "__main__":
    main()
