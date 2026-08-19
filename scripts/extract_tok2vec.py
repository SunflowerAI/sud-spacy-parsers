#!/usr/bin/env python3
"""Dump a trained pipeline's tok2vec to a blob for `[initialize] init_tok2vec`.

WHY A BLOB AND NOT `source=`. spaCy's cross-language `source=` is blocked by E150 ("the language of
the nlp object and the vocab should be the same"), which is exactly the case here: the donor is
`lzh` (a custom language registered by scripts/seg_code.py) and the recipient is `zh`. The blob
route sidesteps the vocab-language check because bytes carry no language. It is the same route
yue's Mandarin init takes (`zh_both_tok2vec.bin`).

The recipient config MUST carry a filled `[pretraining]` block (component/layer) -- spaCy's
`init_tok2vec` resolves the target through `get_tok2vec_ref(nlp, pretrain_config)`, so an EMPTY
`[pretraining]` (what `config_zh_seg.cfg` ships) raises rather than loading.

    .venv/bin/python scripts/extract_tok2vec.py training_lzh_trad/model-best lzh_trad_tok2vec.bin
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: F401  (registers the custom `lzh` language + sud.GoldTokCorpus.v1)
import spacy

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--component", default="tok2vec")
a = ap.parse_args()

nlp = spacy.load(a.model)
model = nlp.get_pipe(a.component).model
data = model.to_bytes()
pathlib.Path(a.out).write_bytes(data)
shapes = [(n.name, pn, n.get_param(pn).shape)
          for n in model.walk() for pn in n.param_names
          if n.has_param(pn) and n.get_param(pn) is not None]
print(f"wrote {a.out}  ({len(data)/1e6:.2f} MB) from {a.model} (lang={nlp.lang}, pipe={a.component})")
print(f"  {len(shapes)} parameter tensors; embed rows "
      f"{[s[2] for s in shapes if s[0]=='hashembed']}")
