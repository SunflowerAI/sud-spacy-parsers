#!/usr/bin/env python
"""Convert a trained `sud.TinyBertPlusHash.v1` (torch) model directory into one whose tok2vec is the
torch-free `sud.TinyBertPlusHashNumpy.v1`, with every weight and the vocab SEALED in `tok2vec/model`.

    seal_tinybert_tok2vec.py SRC_MODEL_DIR OUT_MODEL_DIR

Needs torch + transformers (TRAINING environment only); the OUTPUT needs neither. Everything except
`config.cfg`'s tok2vec block and the `tok2vec/model` file is copied byte for byte (parser/model, tagger/model,
morphologizer, ... are therefore identical -- the script verifies it and prints the hashes).
The converted tok2vec is checked against the torch one on a fixed set of strings before anything is written.
"""
import argparse, hashlib, pathlib, shutil, sys

import numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: F401,E402
import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402
from thinc.api import Config  # noqa: E402

import sud_tinybert_infer as inf  # noqa: E402


def torch_module(tok2vec_model):
    bert_model = tok2vec_model.layers[0]            # the TinyBertTok2Vec model inside concatenate(...)
    return bert_model, bert_model.layers[0].shims[0]._model


def weights_from_torch(mod):
    sd = {k: v.detach().cpu().numpy() for k, v in mod.bert.state_dict().items()}
    cfg = mod.bert.config
    assert cfg.hidden_act == "gelu" and getattr(cfg, "position_embedding_type", "absolute") == "absolute" and mod.mix
    assert isinstance(mod.proj, __import__("torch").nn.Identity), "projection layer not supported"
    e = "embeddings."
    w = {"word": sd[e + "word_embeddings.weight"], "pos": sd[e + "position_embeddings.weight"],
         "type0": sd[e + "token_type_embeddings.weight"][0], "emb_g": sd[e + "LayerNorm.weight"],
         "emb_b": sd[e + "LayerNorm.bias"], "mix_logits": mod.mix_logits.detach().cpu().numpy(),
         "gamma": mod.gamma.detach().cpu().numpy()}
    for i in range(cfg.num_hidden_layers):
        L, p = f"encoder.layer.{i}.", f"l{i}."
        for short, name in (("q", "attention.self.query"), ("k", "attention.self.key"), ("v", "attention.self.value"),
                            ("o", "attention.output.dense"), ("i", "intermediate.dense"), ("f", "output.dense")):
            w[p + short + "_w"] = sd[L + name + ".weight"].T          # torch Linear is (out, in)
            w[p + short + "_b"] = sd[L + name + ".bias"]
        w[p + "ao_g"], w[p + "ao_b"] = sd[L + "attention.output.LayerNorm.weight"], sd[L + "attention.output.LayerNorm.bias"]
        w[p + "fo_g"], w[p + "fo_b"] = sd[L + "output.LayerNorm.weight"], sd[L + "output.LayerNorm.bias"]
    meta = {"hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "layers": cfg.num_hidden_layers,
            "eps": float(cfg.layer_norm_eps), "source": "huawei-noah/TinyBERT_General_4L_312D (Apache-2.0)"}
    return meta, w


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--vocab", default="models/tinybert_general_4l_312d/vocab.txt")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    nlp = spacy.load(src)
    t2v = nlp.get_pipe("tok2vec")
    bert_model, mod = torch_module(t2v.model)
    meta, w = weights_from_torch(mod)
    vocab = pathlib.Path(a.vocab).read_text(encoding="utf-8").split("\n")
    if vocab and vocab[-1] == "":
        vocab.pop()
    assert len(vocab) == w["word"].shape[0], (len(vocab), w["word"].shape)

    cfg = Config().from_disk(src / "config.cfg", interpolate=False)
    blk = cfg["components"]["tok2vec"]["model"]
    assert blk["@architectures"] == "sud.TinyBertPlusHash.v1"
    new = {"@architectures": "sud.TinyBertPlusHashNumpy.v1", "hidden": meta["hidden"], "max_len": blk["max_len"],
           "embed": blk["embed"], "encode": blk["encode"]}
    cfg["components"]["tok2vec"]["model"] = new
    from spacy.util import registry as sreg
    np_model = sreg.resolve({"m": new}, validate=True)["m"]
    shim = np_model.layers[0].shims[0].set_weights(meta, vocab, w)
    shim.max_len = int(blk["max_len"])
    # CNN branch: identical structure, so the torch model's second layer serialises straight into the twin
    np_model.layers[1].from_bytes(t2v.model.layers[1].to_bytes())
    # dims of the concatenation itself: an un-initialised twin has no `nO`, and spaCy's Tok2Vec.predict reads it for an
    # EMPTY batch (nlp("") then raised "Cannot get dimension 'nO'"). Copy every dim the trained model had set.
    for dim in ("nO", "nI"):
        if t2v.model.has_dim(dim) and t2v.model.get_dim(dim) is not None:
            np_model.set_dim(dim, t2v.model.get_dim(dim))
    assert np_model.get_dim("nO") == t2v.model.get_dim("nO") == 408

    # --- check the twin against the torch tok2vec before writing anything
    texts = ["The quick brown fox jumps over the lazy dog .", "  ", "Don't panic : naïve café ﬁne 北京 !", "x " * 400]
    docs = [Doc(nlp.vocab, words=t.split() if t.strip() else [" "]) for t in texts] + [Doc(nlp.vocab, words=[])]
    a_, b_ = t2v.model.predict(docs), np_model.predict(docs)
    worst = max((float(abs(x - y).max()) if len(x) else 0.0) for x, y in zip(a_, b_))
    print("pre-write max |torch - numpy| over probe docs:", worst)
    assert worst < 1e-4, worst

    shutil.copytree(src, out, dirs_exist_ok=False)
    cfg.to_disk(out / "config.cfg")
    (out / "tok2vec" / "model").write_bytes(np_model.to_bytes())
    for comp in ("parser", "tagger", "morphologizer", "lemmatizer"):
        if (src / comp / "model").exists():
            same = sha(src / comp / "model") == sha(out / comp / "model")
            print(f"{comp}/model byte-identical: {same}")
            assert same
    print("wrote", out, "tok2vec/model bytes:", (out / "tok2vec" / "model").stat().st_size)


if __name__ == "__main__":
    main()
