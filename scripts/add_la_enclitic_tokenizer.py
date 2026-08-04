#!/usr/bin/env python3
"""Swap the `-que`-splitting tokeniser into a trained Latin arm. No retrain needed.

Every la config trains through `sud.GoldTokCorpus.v1` under `gold_preproc`, so the predicted
doc is built from GOLD WORDS and the tokeniser never runs during training -- the parser is
segmenter-agnostic, exactly as for the zh character segmenter (`bundle_zh_charseg.py`) and
the ko eojeol switch. So the tokeniser can be replaced in a released arm post hoc, and the
weights come out byte-identical; `--verify` checks that rather than trusting it.

    .venv/bin/python scripts/add_la_enclitic_tokenizer.py training_la_sud/model-best out_dir \
        --code sud_tagger.py,sud_misc.py
"""

from __future__ import annotations

import argparse
import filecmp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spacy                                            # noqa: E402
import la_tokenizer                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true",
                    help="assert every component's weights survive the swap byte-identical")
    ap.add_argument("--code", default="",
                    help="comma-separated scripts/ modules to import first, for arms carrying "
                         "custom pipes (the la SUD arm has sud_tagger)")
    args = ap.parse_args()

    for module in filter(None, (m.strip() for m in args.code.split(","))):
        __import__(Path(module).stem)

    probe = "Arma virumque cano neque relinque"
    nlp = spacy.load(args.src)
    before = [t.text for t in nlp.tokenizer(probe)]
    nlp.tokenizer = la_tokenizer.make_la_enclitic_tokenizer()(nlp)
    # Assigning `nlp.tokenizer` does NOT update the config -- `to_disk` writes the config as it
    # stands, so without this line the reloaded model rebuilds a stock `spacy.Tokenizer.v1` and
    # `from_disk` quietly refills it with the base rules. The model then loads, runs, and splits
    # nothing, saying nothing. Same shape as the zh `jieba_source` marker (`bundle_zh_charseg.py`).
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.LatinEncliticTokenizer.v1"}
    after = [t.text for t in nlp.tokenizer(probe)]
    nlp.to_disk(args.dst)
    print(f"pipeline: {nlp.pipe_names}")
    print(f"  before: {before}")
    print(f"  after : {after}")

    # Round-trip: prove the SAVED model splits, rather than the in-memory one we just built.
    reloaded = spacy.load(args.dst)
    round_trip = [t.text for t in reloaded.tokenizer(probe)]
    if round_trip != after:
        print(f"!! REFUSING: reloaded model tokenises {round_trip}, not {after}")
        return 1
    print(f"  reload: {round_trip}  ({type(reloaded.tokenizer).__name__})")

    if args.verify:
        src, dst = Path(args.src), Path(args.dst)
        bad = []
        for model in sorted(src.rglob("model")):
            twin = dst / model.relative_to(src)
            if not twin.exists() or not filecmp.cmp(model, twin, shallow=False):
                bad.append(str(model.relative_to(src)))
            else:
                print(f"  weights identical: {model.relative_to(src)}")
        if bad:
            print(f"!! WEIGHTS CHANGED: {bad}")
            return 1
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
