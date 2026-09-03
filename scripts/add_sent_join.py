#!/usr/bin/env python
"""Append the (non-trainable) `sent_join` pipe to a trained model and save it.

`sent_join` refuses a sentence boundary where the reading convention says there is none: inside a
balanced or OPEN quoted span (closing mark included), and after a pause mark of any kind. It merges
by re-heading the roots the parser opened there — `token.is_sent_start` is not writable on a parsed
doc — and chooses the relation by the clause rule in scripts/sent_join.py: `parataxis` between two
predications, `conj:coord` where one head is not a predicate / the clause opens with a SCONJ / the
chain head has a subject this clause lacks, `comp:obj` to the speech verb inside a quotative frame.

⚠ `--joins` IS SUPERSEDED AND IGNORED. It named the harvested (mark kind, previous-head UPOS)
table. Passing it also overwrote `default_dep` with that table's own default, which silently turned
every `parataxis` decision into `conj:coord` in a built wheel — caught only by reading the shipped
pipe's config back.

Usage:
    add_sent_join.py IN_MODEL OUT_MODEL [--no-quote-spans] [--no-pause-join]
                     [--pairs CHARS] [--max-span N] [--max-sent N]

It goes LAST. The `sud_*` MISC pipes read the tree, so an earlier position would change their input
and couple this pipe to that layer (CLAUDE.md standing hazard 5); last, it changes the emitted
segmentation and nothing those pipes saw.

⚠ THE KYOTO GOLD USES THE OTHER CONVENTION ON BOTH COUNTS — 5 944 of its 59 215 punctuation-restored
training blocks have unbalanced 「」, and 31.2 % of its blocks END at a pause mark — so it scores
this down on any harness spanning more than one gold sentence, and is neutral under `--gold-preproc`.
The measured table is in docs/chinese-family.md.
"""
import argparse
import importlib.util
import pathlib

import spacy


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--joins", default=None,
                    help="SUPERSEDED and ignored; the clause rule replaced the harvested table")
    ap.add_argument("--no-quote-spans", action="store_true")
    ap.add_argument("--no-pause-join", action="store_true")
    # Quotation marks only by default; append sent_join.BRACKET_PAIRS to treat a parenthetical the
    # same way. Read two characters at a time, open then close.
    ap.add_argument("--pairs", default=None)
    # Refuse to merge a span longer than N tokens (0 = no limit). A second guard on top of
    # "balanced spans only", against a text whose marks pair up across half a chapter.
    ap.add_argument("--max-span", type=int, default=None)
    ap.add_argument("--max-sent", type=int, default=None)
    a = ap.parse_args()

    load_code("scripts/seg_code.py")           # registers sent_join and every custom tokenizer

    nlp = spacy.load(a.in_model)
    for stale in ("sent_join", "quote_sents"):   # replace, so new code + config take effect
        if stale in nlp.pipe_names:
            nlp.remove_pipe(stale)
    config = {"quote_spans": not a.no_quote_spans, "pause_join": not a.no_pause_join}
    if a.joins:
        print(f"  ⚠ ignoring --joins {a.joins}: superseded by the clause rule")
    if a.pairs is not None:
        config["pairs"] = a.pairs
    if a.max_span is not None:
        config["max_span"] = a.max_span
    if a.max_sent is not None:
        config["max_sent"] = a.max_sent
    nlp.add_pipe("sent_join", last=True, config=config)
    pipe = nlp.get_pipe("sent_join")
    print(f"{a.out_model}: {nlp.pipe_names}\n  quote_spans={pipe.quote_spans} "
          f"pause_join={pipe.pause_join} default={pipe.default_dep!r}")
    nlp.to_disk(a.out_model)


if __name__ == "__main__":
    main()
