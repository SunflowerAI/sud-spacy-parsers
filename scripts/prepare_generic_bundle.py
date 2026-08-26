#!/usr/bin/env python3
"""Put the generic bundle into a shippable state, idempotently, and REFUSE if it is not.

`package_generic_v2.sh` calls this before `spacy package`, so the two things that make the wheel
honour its own contract cannot be forgotten the next time the bundle is reassembled. A comment
telling the next person is not the fix (CLAUDE.md hazard 2); a default plus a refusal is.

TWO CHANGES, both about the same thing: **UPOS is an input to this arm, never an output.**

1. `morphologizer.overwrite = false`. spaCy's morphologiser predicts a joint `POS=X|Feat=Val` label
   and writes BOTH halves whenever `overwrite` is on, so the shipped 0.1.0 replaced the user's UPOS
   with its own guess before the parser read it -- and clobbered any FEATS the user supplied along
   with it. See `fix_generic_pos_write.py` for the mechanism and the measurements.

2. A `sud_require_upos` guard, first in the pipeline. With (1) in place the POS write is unreachable
   for any token that HAS a UPOS; this makes sure every token does. Without it, a caller who forgot
   the tag column would get a silent parse over `POS=` -- the arm's one indispensable input, missing,
   with nothing raised.

⚠ VERIFIED ON THE RELOADED MODEL, never on the object this script just edited: `overwrite` lives in
the component's serialised `cfg` as well as in `config.cfg`, and `from_disk` restores the former.

⚠ THE PARSE MUST NOT MOVE. Neither change touches a weight, so `--verify` re-parses a fixed probe
and compares against the digest taken before the edits. A differing digest means something other
than annotation-writing changed, and that is a refusal rather than a warning.
"""
import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import sud_generic_embed_v2  # noqa: E402,F401  (registers the layer AND sud_require_upos)
from fix_generic_pos_write import patch  # noqa: E402

GUARD = "sud_require_upos"

#: A probe whose UPOS is fully supplied, so it parses identically before and after both changes.
PROBE_WORDS = ["the", "cat", "sat", "on", "the", "mat"]
PROBE_POS = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]


def probe(nlp, lang="en"):
    doc = Doc(nlp.vocab, words=PROBE_WORDS)
    for t, p in zip(doc, PROBE_POS):
        t.pos_ = p
    doc._.tb_lang = lang
    out = nlp(doc)
    return out, hashlib.sha256(
        "|".join(f"{t.text}/{t.head.i}/{t.dep_}" for t in out).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if work is needed")
    a = ap.parse_args()
    d = pathlib.Path(a.model)

    before_digest = probe(spacy.load(d))[1]
    print(f"parse digest before: {before_digest}")

    needs_overwrite = patch(d, "morphologizer", check=True)
    nlp = spacy.load(d)
    needs_guard = GUARD not in nlp.pipe_names
    if needs_guard:
        print(f"  {d}: pipeline {nlp.pipe_names} has no {GUARD}  -> insert first")
    else:
        print(f"  {d}: {GUARD} already present")

    if a.check:
        if needs_overwrite or needs_guard:
            sys.exit(f"{d} is NOT shippable: "
                     f"{'overwrite ' if needs_overwrite else ''}"
                     f"{'guard' if needs_guard else ''}")
        print(f"{d} is shippable")
        return

    if needs_overwrite:
        patch(d, "morphologizer", check=False)
    if needs_guard:
        nlp = spacy.load(d)                 # reload, so the overwrite edit is the one we build on
        nlp.add_pipe(GUARD, first=True)
        nlp.to_disk(d)

    # --- everything below runs against the model AS RELOADED FROM DISK ---
    nlp = spacy.load(d)
    if nlp.pipe_names[0] != GUARD:
        sys.exit(f"FAILED: {GUARD} is not first in {nlp.pipe_names}")
    out, after_digest = probe(nlp)
    if after_digest != before_digest:
        sys.exit(f"FAILED: the parse moved, {before_digest} -> {after_digest}. Neither change "
                 f"touches a weight, so something else did.")
    if [t.pos_ for t in out] != PROBE_POS:
        sys.exit(f"FAILED: UPOS still rewritten: {PROBE_POS} -> {[t.pos_ for t in out]}")

    # the guard must actually fire
    bare = Doc(nlp.vocab, words=PROBE_WORDS)
    bare._.tb_lang = "en"
    try:
        nlp(bare)
    except ValueError as e:
        if GUARD not in str(e):
            raise
        print(f"  guard fires on untagged input: {str(e).split('.')[0]}.")
    else:
        sys.exit(f"FAILED: {GUARD} did not refuse a doc with no UPOS")

    print(f"pipeline {nlp.pipe_names}")
    print(f"parse digest after:  {after_digest}  (unchanged)")
    print(f"{d} is shippable")


if __name__ == "__main__":
    main()
