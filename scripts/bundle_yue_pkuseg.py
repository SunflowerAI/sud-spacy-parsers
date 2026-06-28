#!/usr/bin/env python
"""Swap the pkuseg segmenter into the released Cantonese model (post-hoc, like zh).

gold_preproc training is segmenter-agnostic, so the raw-text tokeniser is swapped into the
trained ext model after the fact: load training_yue_ext/model-best (char tokeniser), replace its
tokenizer with `yue.PkusegTokenizer.v1` initialised from the from-scratch yue pkuseg model
(models/yue_hk_pkuseg_scratch — fine-tuning from Mandarin ties it, so we ship the self-contained
one), point the config at the pkuseg tokeniser, and save to training_yue_ext_pkuseg/. The pkuseg
model files (weights.npz + features.msgpack) serialise into the pipeline's tokenizer/ dir, so the
packaged wheel is self-contained.

Then package + release:
  mkdir -p build_yue
  .venv/bin/python -m spacy package training_yue_ext_pkuseg build_yue \
    --name sud_hk --version 0.1.0 --code scripts/yue_tokenizer.py --build wheel --force
  gh release upload v0.1.0 build_yue/yue_sud_hk-0.1.0/dist/yue_sud_hk-0.1.0-py3-none-any.whl --clobber
(remember to add spacy-pkuseg to training_yue_ext_pkuseg/meta.json "requirements" first.)
"""
import importlib.util, json, spacy

SRC = "training_yue_ext/model-best"
PKUSEG = "models/yue_hk_pkuseg_scratch"
OUT = "training_yue_ext_pkuseg"


def main():
    spec = importlib.util.spec_from_file_location("yue_tokenizer", "scripts/yue_tokenizer.py")
    yt = importlib.util.module_from_spec(spec); spec.loader.exec_module(yt)

    nlp = spacy.load(SRC)
    seg_tok = yt.PkusegTokenizer(nlp.vocab)
    seg_tok.initialize(pkuseg_model=PKUSEG)
    nlp.tokenizer = seg_tok
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "yue.PkusegTokenizer.v1"}
    nlp.to_disk(OUT)

    # declare the runtime dep so the wheel pulls spacy-pkuseg
    mp = f"{OUT}/meta.json"
    m = json.load(open(mp))
    m["requirements"] = sorted(set(m.get("requirements") or []) | {"spacy-pkuseg>=0.0.27"})
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {OUT} (pkuseg tokeniser bundled; requirements={m['requirements']})")


if __name__ == "__main__":
    main()
