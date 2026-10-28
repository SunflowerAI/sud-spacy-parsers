#!/usr/bin/env python
"""A small pretrained transformer as spaCy's `tok2vec`, with a learned mix of ALL its layers.

`sud.TinyBertTok2Vec.v1` is a thinc `Model[List[Doc], List[Floats2d]]`, so it drops into
`[components.tok2vec.model]` and every downstream component keeps its ordinary
`spacy.Tok2VecListener.v1`. It deliberately does NOT use `spacy-transformers`: that package needs
torch AT INFERENCE, which the serverless budget rules out (docs/packaging-and-release.md), and its
listener pools only the last layer. Here the encoder is a plain `transformers.BertModel` inside
thinc's `PyTorchWrapper_v2`, and the forward pass is small enough (4 layers x 312-d for TinyBERT) to
be ported to numpy later, so training is the only place torch is needed.

Pipeline of one call:
  words -> WordPiece (the HF fast tokenizer, one sequence per chunk of <= `max_len` pieces,
  [CLS]/[SEP] added) -> BERT with output_hidden_states -> softmax-weighted sum of the embedding
  output and every layer (ELMo/UDify-style scalar mix; `mix = false` takes the last layer only, the
  control; `freeze = true` keeps every BERT weight fixed and trains only the mix, a scale, the
  optional projection and whatever sits downstream) -> the FIRST piece of each word -> optional linear projection to `width`.

Every word gets at least one piece (whitespace-only tokens fall back to [UNK]), so the gather index
is total and no word is silently dropped. Chunks split only between words.

HAZARD 4 (docs / CLAUDE.md): `model_path` is a HOST path, serialised verbatim into the config. That
is fine for training here and WRONG for a packaged wheel: before shipping, the weights and
vocab must be sealed into the component's own `to_disk` bytes and the path replaced.
"""
from typing import List

import torch
from spacy.tokens import Doc
from spacy import registry
from spacy.ml.models.tok2vec import build_Tok2Vec_model
from thinc.api import Model, PyTorchWrapper_v2, concatenate
from thinc.types import ArgsKwargs, Floats2d
from transformers import AutoTokenizer, BertConfig, BertModel

_TOKENIZERS = {}


def _tokenizer(path):
    # Kept out of `model.attrs`: thinc msgpack-serialises attrs and a tokenizer is not msgpack.
    if path not in _TOKENIZERS:
        _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path, use_fast=True)
    return _TOKENIZERS[path]


class _BertMix(torch.nn.Module):
    def __init__(self, path: str, width: int, mix: bool, dropout: float, freeze: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(path, add_pooling_layer=False)
        cfg: BertConfig = self.bert.config
        hidden = cfg.hidden_size
        self.mix, self.freeze = mix, freeze
        if freeze:
            self.bert.requires_grad_(False)
        if mix:
            # one scalar per hidden state (embedding output + each layer); zeros => uniform softmax
            self.mix_logits = torch.nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        if mix or freeze:
            # a trainable scale, so a frozen last-layer arm still has a parameter to backprop into
            self.gamma = torch.nn.Parameter(torch.ones(1))
        self.drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Identity() if width in (0, hidden) else torch.nn.Linear(hidden, width)

    def forward(self, ids, mask, gather):
        if self.freeze:
            self.bert.eval()  # thinc's shim calls .train() on the whole module; a frozen encoder has no dropout
            with torch.no_grad():
                hs = self.bert(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        else:
            hs = self.bert(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        if self.mix:
            w = torch.softmax(self.mix_logits, dim=0)
            h = sum(w[i] * hs[i] for i in range(len(hs)))
        else:
            h = hs[-1]
        if self.mix or self.freeze:
            h = self.gamma * h
        flat = h.reshape(-1, h.shape[-1])
        return self.proj(self.drop(flat.index_select(0, gather)))


def _encode(docs: List[Doc], path: str, max_len: int):
    """Return (ids, mask, gather, lengths): padded piece ids, attention mask, the flat position of
    each word's first piece in the padded batch, and the number of words per doc."""
    tok = _tokenizer(path)
    cls_id, sep_id, unk_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.unk_token_id, tok.pad_token_id
    chunks = []  # (ids list, first-piece offset per word in that chunk)
    for doc in docs:
        words = [t.text for t in doc]
        if not words:
            continue
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        pieces = [[] for _ in words]
        for pos, wid in enumerate(enc.word_ids()):
            if wid is not None:
                pieces[wid].append(enc["input_ids"][pos])
        ids, firsts = [cls_id], []
        for p in pieces:
            p = p or [unk_id]
            if len(ids) + len(p) > max_len - 1 and firsts:
                chunks.append((ids + [sep_id], firsts))
                ids, firsts = [cls_id], []
            firsts.append(len(ids))
            ids.extend(p[: max_len - 2])
        chunks.append((ids + [sep_id], firsts))
    width = max((len(c[0]) for c in chunks), default=2)
    ids = torch.full((len(chunks), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chunks), width), dtype=torch.long)
    gather = []
    for r, (cids, firsts) in enumerate(chunks):
        ids[r, : len(cids)] = torch.tensor(cids)
        mask[r, : len(cids)] = 1
        gather.extend(r * width + f for f in firsts)
    return ids.numpy(), mask.numpy(), torch.tensor(gather, dtype=torch.long).numpy(), [len(d) for d in docs]


def _convert_inputs(model, X, is_train):
    device = model.shims[0].device
    args = tuple(torch.as_tensor(a, device=device) for a in X)
    return ArgsKwargs(args=args, kwargs={}), lambda dX: []


def _forward(model: Model, docs: List[Doc], is_train: bool):
    ids, mask, gather, lengths = _encode(docs, model.attrs["model_path"], model.attrs["max_len"])
    inner = model.layers[0]
    Y, bp_inner = inner((ids, mask, gather), is_train)

    def backprop(dYs):
        bp_inner(model.ops.flatten(dYs))
        return []

    return model.ops.unflatten(Y, lengths), backprop


@registry.architectures.register("sud.TinyBertTok2Vec.v1")
def TinyBertTok2Vec(
    model_path: str, width: int = 0, mix: bool = True, dropout: float = 0.1, max_len: int = 256,
    freeze: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    """`width = 0` keeps the encoder's own hidden size (312 for TinyBERT-4L)."""
    hidden = BertConfig.from_pretrained(model_path).hidden_size
    out = width or hidden
    module = _BertMix(model_path, out, mix, dropout, freeze)
    wrapper = PyTorchWrapper_v2(module, convert_inputs=_convert_inputs)
    return Model(
        "sud.tinybert_tok2vec",
        _forward,
        layers=[wrapper],
        dims={"nO": out},
        attrs={"model_path": model_path, "max_len": max_len},
    )


@registry.architectures.register("sud.TinyBertPlusHash.v1")
def TinyBertPlusHash(
    embed: Model, encode: Model, model_path: str, width: int = 0, mix: bool = True,
    dropout: float = 0.1, max_len: int = 256, freeze: bool = False,
) -> Model[List[Doc], List[Floats2d]]:
    """TinyBERT beside spaCy's own hash-embed CNN, concatenated per token.

    The CNN reads NORM/PREFIX/SUFFIX/SHAPE, so it still sees the capitalisation an UNCASED encoder
    discards. The listener width downstream is the SUM of the two (312 + the CNN's width).
    """
    bert = TinyBertTok2Vec(model_path, width=width, mix=mix, dropout=dropout, max_len=max_len, freeze=freeze)
    return concatenate(bert, build_Tok2Vec_model(embed, encode))
