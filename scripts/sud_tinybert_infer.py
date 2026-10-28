#!/usr/bin/env python
"""Inference-only TinyBERT tok2vec: numpy forward pass + pure-python WordPiece, NO torch, NO transformers.

`sud.TinyBertNumpy.v1` / `sud.TinyBertPlusHashNumpy.v1` reproduce, number for number,
`sud.TinyBertTok2Vec.v1` / `sud.TinyBertPlusHash.v1` from `sud_tinybert_embed.py` (the TRAINING module,
which needs torch + transformers and is left untouched):

  words -> WordPiece (same normalisation as the HF `BertTokenizerFast` used in training: clean-text,
  CJK spacing, accent strip + lowercase, whitespace/punctuation pre-split, greedy longest-match)
  -> chunks of <= `max_len` pieces split only BETWEEN words, [CLS]/[SEP] added, whitespace-only tokens
  fall back to [UNK] -> 4-layer BERT in numpy (post-LN, exact-erf GELU) keeping all 5 hidden states
  -> gamma * softmax(mix_logits)-weighted sum -> the FIRST piece of each word.

SEALING (CLAUDE.md standing hazard 4). The config names NO file. The BERT weights, the 30 522-entry
vocab, the BERT hyper-parameters, the layer-mix logits and gamma all live in the component's own
`to_disk` bytes: the numpy encoder is a thinc `Shim` whose `to_bytes` carries all of it, so the
`tok2vec/model` file of the packaged wheel is self-contained and loading never opens a path.
An unfilled shim (weights not yet loaded) REFUSES to run rather than emit zeros.

Weights are converted from a trained torch arm by `scripts/seal_tinybert_tok2vec.py`.

Licence of the encoder weights and vocab: TinyBERT_General_4L_312D, Apache-2.0 (see the wheel NOTICE).
"""
import math
import re
import unicodedata
from typing import Dict, List

import numpy
import srsly
from spacy import registry
from spacy.ml.models.tok2vec import build_Tok2Vec_model
from spacy.tokens import Doc
from thinc.api import Model, concatenate
from thinc.shims import Shim
from thinc.types import Floats2d

# --------------------------------------------------------------------------------------- WordPiece


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F
    )


def _is_punct(c: str) -> bool:
    cp = ord(c)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(c).startswith("P")


def _is_ws(c: str) -> bool:
    # controls (other than \t \n \r) are already dropped, so str.isspace matches the Rust White_Space set
    return c.isspace()


def _normalise(word: str) -> str:
    """The HF BertNormalizer (clean_text, handle_chinese_chars, strip_accents, lowercase), in order."""
    out = []
    for c in word:
        if c == "\0" or c == "�":
            continue
        if c not in "\t\n\r" and unicodedata.category(c) in ("Cc", "Cf", "Co"):
            continue
        if _is_ws(c):
            out.append(" ")
        elif _is_cjk(ord(c)):
            out.append(" " + c + " ")
        else:
            out.append(c)
    s = "".join(out)
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return s.lower()


def _pre_split(s: str) -> List[str]:
    """Whitespace split, then each punctuation character is its own pre-token."""
    toks = []
    for chunk in s.split():
        cur = []
        for c in chunk:
            if _is_punct(c):
                if cur:
                    toks.append("".join(cur))
                    cur = []
                toks.append(c)
            else:
                cur.append(c)
        if cur:
            toks.append("".join(cur))
    return toks


class WordPiece:
    def __init__(self, vocab: List[str], unk: str = "[UNK]", max_chars: int = 100):
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.unk, self.unk_id = unk, self.vocab[unk]
        self.cls_id, self.sep_id, self.pad_id = self.vocab["[CLS]"], self.vocab["[SEP]"], self.vocab["[PAD]"]
        self.max_chars = max_chars
        self._specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in sorted(self._specials)) + ")")
        self._cache: Dict[str, List[int]] = {}

    def _piece(self, tok: str) -> List[int]:
        if len(tok) > self.max_chars:
            return [self.unk_id]
        out, start = [], 0
        while start < len(tok):
            end, cur = len(tok), None
            while start < end:
                sub = tok[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                return [self.unk_id]
            out.append(cur)
            start = end
        return out

    def word(self, word: str) -> List[int]:
        """Word-piece ids of ONE already-isolated word (a spaCy token); [] if it normalises to nothing."""
        hit = self._cache.get(word)
        if hit is None:
            hit = []
            # HF splits ADDED/special tokens out of the raw text first, so a literal "[SEP]" is id 102
            for part in self._special_re.split(word):
                if part in self._specials:
                    hit.append(self.vocab[part])
                    continue
                for tok in _pre_split(_normalise(part)):
                    hit.extend(self._piece(tok))
            if len(self._cache) < 200_000:
                self._cache[word] = hit
        return hit


# ------------------------------------------------------------------------------------------ BERT


def _erf(x: numpy.ndarray) -> numpy.ndarray:
    # Abramowitz & Stegun 7.1.26 is only 1.5e-7; use the Numerical-Recipes erfc Chebyshev fit (<1.2e-7
    # RELATIVE) via float64 so GELU matches torch's exact-erf GELU far below the 1e-4 tolerance.
    z = numpy.abs(x.astype(numpy.float64))
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    erfc = t * numpy.exp(poly)
    r = 1.0 - erfc
    return numpy.where(x >= 0, r, -r).astype(x.dtype)


def _gelu(x: numpy.ndarray) -> numpy.ndarray:
    return 0.5 * x * (1.0 + _erf(x * numpy.float32(1.0 / math.sqrt(2.0))))


def _layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / numpy.sqrt(var + eps) * g + b


class NumpyBert(Shim):
    """Holds every weight; `to_bytes`/`from_bytes` seal them (and the vocab) into the model file."""

    def __init__(self, max_len: int = 256):
        super().__init__(None)
        self.max_len = max_len
        self.meta: Dict = {}
        self.w: Dict[str, numpy.ndarray] = {}
        self.vocab_list: List[str] = []
        self._tok = None

    # ---- (de)serialisation: everything is in these bytes, nothing is read from a path
    def set_weights(self, meta, vocab_list, weights):
        self.meta, self.vocab_list = dict(meta), list(vocab_list)
        self.w = {k: numpy.ascontiguousarray(v, dtype=numpy.float32) for k, v in weights.items()}
        self._tok = None
        return self

    def to_bytes(self):
        return srsly.msgpack_dumps({"meta": self.meta, "vocab": self.vocab_list, "w": self.w, "max_len": self.max_len})

    def from_bytes(self, data):
        msg = srsly.msgpack_loads(data)
        self.max_len = int(msg["max_len"])
        return self.set_weights(msg["meta"], msg["vocab"], msg["w"])

    def to_device(self, device_type, device_id):
        pass

    def finish_update(self, optimizer):
        pass

    def __call__(self, inputs, is_train):
        raise NotImplementedError("sud.TinyBertNumpy is inference-only")

    @property
    def tok(self) -> WordPiece:
        if not self.w:
            raise ValueError("sud.TinyBertNumpy has no weights: the model was built from config but never loaded "
                             "from its serialised bytes (refusing to emit zeros)")
        if self._tok is None:
            self._tok = WordPiece(self.vocab_list)
        return self._tok

    # ---- forward
    def hidden_states(self, ids: List[int]) -> List[numpy.ndarray]:
        w, m = self.w, self.meta
        H, nh, eps = m["hidden"], m["heads"], m["eps"]
        dh = H // nh
        n = len(ids)
        x = w["word"][ids] + w["pos"][:n] + w["type0"]
        x = _layer_norm(x, w["emb_g"], w["emb_b"], eps)
        hs = [x]
        scale = numpy.float32(1.0 / math.sqrt(dh))
        for i in range(m["layers"]):
            p = f"l{i}."
            q = (x @ w[p + "q_w"] + w[p + "q_b"]).reshape(n, nh, dh).transpose(1, 0, 2)
            k = (x @ w[p + "k_w"] + w[p + "k_b"]).reshape(n, nh, dh).transpose(1, 2, 0)
            v = (x @ w[p + "v_w"] + w[p + "v_b"]).reshape(n, nh, dh).transpose(1, 0, 2)
            s = (q @ k) * scale
            s -= s.max(-1, keepdims=True)
            e = numpy.exp(s)
            a = e / e.sum(-1, keepdims=True)
            ctx = (a @ v).transpose(1, 0, 2).reshape(n, H)
            x = _layer_norm(x + ctx @ w[p + "o_w"] + w[p + "o_b"], w[p + "ao_g"], w[p + "ao_b"], eps)
            h = _gelu(x @ w[p + "i_w"] + w[p + "i_b"])
            x = _layer_norm(x + h @ w[p + "f_w"] + w[p + "f_b"], w[p + "fo_g"], w[p + "fo_b"], eps)
            hs.append(x)
        return hs

    def mixed(self, ids: List[int]) -> numpy.ndarray:
        hs = self.hidden_states(ids)
        lg = self.w["mix_logits"].astype(numpy.float64)
        wts = numpy.exp(lg - lg.max())
        wts = (wts / wts.sum()).astype(numpy.float32)
        out = wts[0] * hs[0]
        for i in range(1, len(hs)):
            out = out + wts[i] * hs[i]
        return out * self.w["gamma"][0]


def encode_chunks(tok: WordPiece, words: List[str], max_len: int):
    """Mirror of `_encode` in sud_tinybert_embed.py for ONE doc: [(ids, first-piece offsets)]."""
    chunks = []
    ids, firsts = [tok.cls_id], []
    for word in words:
        p = tok.word(word) or [tok.unk_id]
        if len(ids) + len(p) > max_len - 1 and firsts:
            chunks.append((ids + [tok.sep_id], firsts))
            ids, firsts = [tok.cls_id], []
        firsts.append(len(ids))
        ids.extend(p[: max_len - 2])
    chunks.append((ids + [tok.sep_id], firsts))
    return chunks


def _forward(model: Model, docs: List[Doc], is_train: bool):
    if is_train:
        raise ValueError("sud.TinyBertNumpy is inference-only; train with sud.TinyBertTok2Vec.v1 (torch)")
    shim: NumpyBert = model.shims[0]
    out = []
    for doc in docs:
        words = [t.text for t in doc]
        if not words:
            out.append(numpy.zeros((0, model.get_dim("nO")), dtype="float32"))
            continue
        rows = []
        for ids, firsts in encode_chunks(shim.tok, words, shim.max_len):
            rows.append(shim.mixed(ids)[firsts])
        out.append(numpy.concatenate(rows, axis=0).astype("float32", copy=False))
    return out, lambda dY: []


@registry.architectures.register("sud.TinyBertNumpy.v1")
def TinyBertNumpy(hidden: int = 312, max_len: int = 256) -> Model[List[Doc], List[Floats2d]]:
    """Weights are NOT in the config: they arrive through `Model.from_bytes` (the component's own bytes)."""
    return Model("sud.tinybert_numpy", _forward, dims={"nO": hidden}, shims=[NumpyBert(max_len)])


@registry.architectures.register("sud.TinyBertPlusHashNumpy.v1")
def TinyBertPlusHashNumpy(embed: Model, encode: Model, hidden: int = 312, max_len: int = 256):
    """The numpy twin of `sud.TinyBertPlusHash.v1`: encoder beside spaCy's hash-embed CNN, concatenated."""
    return concatenate(TinyBertNumpy(hidden, max_len), build_Tok2Vec_model(embed, encode))
