#!/usr/bin/env python
"""Load the trained character segmenter into an Indonesian arm and save it self-contained.

`sud.CharSegTokenizer.v1` builds with NO segmenter -- the config names the factory, not the
weights -- so an arm straight out of `spacy train` tokenises on nothing until this runs. That was
done inline in run_round3.sh when the split arm was first built (`build_id_charseg`); it is a
script now because the SUD layer adds a pipe ON TOP of that arm, so the bundling has to happen
again, downstream of training, rather than once by hand.

    bundle_id_charseg.py training_id_sud/model-best build_id_sud [--segmenter models/id_seg_char2]

The wrapper serialises the segmenter beside the weights, so the wheel is self-contained. Verify
the reload rather than the in-memory object -- assigning a tokeniser does not by itself update the
config, which is how a zh wheel once shipped a tokeniser that silently split nothing (CLAUDE.md).
"""
import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import char_seg_tokenizer  # noqa: E402,F401  (registers sud.CharSegTokenizer.v1)
import seg_code  # noqa: E402,F401  (sud_tagger + the rest of the custom factories)
import spacy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--segmenter", default="models/id_seg_char2")
    args = ap.parse_args()

    nlp = spacy.load(args.in_model)
    nlp.tokenizer.load_segmenter(args.segmenter)
    nlp.to_disk(args.out_model)

    # Reload and check the segmenter came back, not just that it was set. A CharSegTokenizer with
    # no model loads, runs, splits nothing and says nothing -- exactly the silent degradation
    # bundle_zh_charseg.py refuses to ship.
    reloaded = spacy.load(args.out_model)
    probe = "penghuninya"
    if len(reloaded(probe)) < 2:
        sys.exit(f"{args.out_model}: segmenter did not survive the round trip "
                 f"({probe!r} came back as one token)")
    print(f"{args.out_model}: pipeline {reloaded.pipe_names}; "
          f"{probe} -> {[t.text for t in reloaded(probe)]}")


if __name__ == "__main__":
    main()
