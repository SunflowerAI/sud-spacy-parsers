#!/usr/bin/env python3
"""Character tagger that turns continuous saṃhitā into CSL notation.

The model is a per-character tagger, NOT a seq2seq rewriter, for two reasons that both bind hard
in this project:

  * `doc._.src_spans` tiles the raw input exactly (209 455/209 455 corpus tokens). A seq2seq model
    rewrites the string and destroys the character correspondence; a per-character label keeps it
    by construction, since every input character's expansion is known.
  * The released wheels are ~12 MB. This is ~0.2-0.6 M parameters in Thinc, which spaCy already
    depends on, so nothing new ships. `chronbmm/sanskrit5-multitask` (ByT5-Sanskrit), the current
    published state of the art, is a ~555 MB ByT5 checkpoint and needs transformers + torch.

Architecture mirrors `spacy.MaxoutWindowEncoder.v2` one level down, over characters instead of
tokens: Embed -> depth x residual(expand_window(1) + Maxout) -> Softmax over the label set. The
character inventory is ~40 after `sa_tokenizer.normalise`, so a plain `Embed` is exact and no
hashing is needed. `with_array(..., pad=)` keeps the convolution from reading across sentence
boundaries in a batch.

Depth sets the receptive field (depth x window_size on each side). The n-gram baseline in
`scripts/baseline_samhita.py` plateaus at +/-3 characters, so depth 6 (+/-6) is already generous.

Labels are produced by `scripts/make_samhita_pairs.py`; see its docstring for the scheme.

Decoding is plain greedy argmax. A lexicon-reranked beam decoder was built and measured against it,
and dropped — see the negative result in CLAUDE.md.
"""
import json
import pathlib

import srsly

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                                          # Devanagari -> IAST, accents, NFC
    from sa_tokenizer import normalise        # noqa: E402
except ImportError:                           # pragma: no cover
    # This module is also the character segmenter for zh/id (`sud.CharSegTokenizer.v1`), whose
    # wheels have no reason to carry the Sanskrit tokeniser. `normalise` is only meaningful for
    # Sanskrit input — Devanagari transliteration and accent stripping — so outside that setting
    # the identity is exactly right, and importing sa_tokenizer would drag Sanskrit into a Chinese
    # wheel. `to_csl` is the only caller (L116); the trained models for zh/id never see Devanagari.
    def normalise(text):
        return text
from thinc.api import (Embed, Maxout, Softmax, chain, clone, expand_window, residual,
                       with_array)

PAD_CHAR = "\x00"
UNK_CHAR = "\x01"
KEEP = "="


def build_model(n_chars, n_labels, width=64, depth=6, window_size=1, maxout_pieces=3):
    """List[Ints2d] (character ids, one column) -> List[Floats2d] (label scores)."""
    cnn = chain(
        expand_window(window_size=window_size),
        Maxout(nO=width, nI=width * (window_size * 2 + 1), nP=maxout_pieces,
               dropout=0.0, normalize=True),
    )
    encoder = clone(residual(cnn), depth)
    encoder.set_dim("nO", width)
    return chain(
        with_array(Embed(nO=width, nV=n_chars, column=0, dropout=0.0)),
        with_array(encoder, pad=window_size * depth),
        with_array(Softmax(nO=n_labels, nI=width)),
    )


class Presegmenter:
    """A trained tagger plus its character and label inventories."""

    def __init__(self, chars, labels, model=None, width=64, depth=6):
        self.chars = list(chars)
        self.labels = list(labels)
        self.width, self.depth = width, depth
        self.char_id = {c: i for i, c in enumerate(self.chars)}
        self.model = model if model is not None else build_model(
            len(self.chars), len(self.labels), width, depth)

    @property
    def reads_spaces(self):
        """True if this model was trained on SPACED text and wants whole strings, not chunks.

        The inventory answers it exactly: `build_vocabs` collects the characters actually seen in
        training, so a space is in `chars` if and only if training rows contained one. Callers must
        ask this rather than assume — the two CSLisers differ on it and the wrong choice is silent
        (see `to_csl`).
        """
        return " " in self.char_id

    # ---- encoding ---------------------------------------------------------------------------
    def encode_chars(self, text):
        unk = self.char_id[UNK_CHAR]
        return self.model.ops.asarray2i(
            [[self.char_id.get(c, unk)] for c in text], dtype="i")

    def encode_labels(self, labels):
        idx = {lb: i for i, lb in enumerate(self.labels)}
        return [idx[lb] for lb in labels]

    # ---- inference --------------------------------------------------------------------------
    def predict(self, texts):
        """List of saṃhitā strings -> list of label lists."""
        nonempty = [t for t in texts if t]
        if not nonempty:
            return [[] for _ in texts]
        scores = self.model.predict([self.encode_chars(t) for t in nonempty])
        it = iter(scores)
        out = []
        for t in texts:
            if not t:
                out.append([])
                continue
            s = next(it)
            out.append([self.labels[int(i)] for i in s.argmax(axis=1)])
        return out

    def to_csl(self, text):
        """saṃhitā -> CSL, ready for `sa_tokenizer`. Accepts Devanagari or IAST.

        Chunking is CONDITIONAL on the loaded model, and getting it wrong is silent and expensive
        in both directions.

        A model trained on continuous saṃhitā (`make_samhita_pairs.py`, the hardest case) has never
        seen a space, so the character is not in its vocabulary and feeding it a spaced string hands
        the encoder UNK at every word boundary. For those the model MUST be run per whitespace
        chunk, which is also the right semantics: a space in the input is already a word boundary,
        so the model only has to find the boundaries INSIDE a chunk.

        A model trained on SPACED text — `sa_presegment_ortho`, which is the one the released wheel
        carries, 381 775 of its 386 260 training rows containing a space — wants the whole string.
        Chunking it throws away every cue that crosses a space, which is exactly the context that
        locates the remaining breaks. Measured on the Vedic IAST test, chunking the ortho model
        costs **split-location F 0.8731 -> 0.8248 and sentence PM 0.7882 -> 0.7269**; that is what
        the released model was doing, because this method kept chunking after the CSLiser was
        replaced. `reads_spaces` now decides it from the model's own vocabulary.

        This matters more for Devanagari than for IAST, because the two scripts do not agree on
        where spaces go: Devanagari cannot write a bare consonant before a vowel-initial word, so
        `vahniḥ + idraḥ` is printed solid (वह्निरिद्रः) where IAST prints `vahnir idraḥ`, and the
        avagraha is likewise solid (नमोऽस्तु vs `namo 'stu`). Devanagari therefore hands the model
        LONGER chunks with more boundaries to find — which is exactly the space-free case it was
        trained on.
        """
        norm = normalise(text)
        if self.reads_spaces:
            return apply_labels(norm, self.predict([norm])[0]) if norm else ""
        return " ".join(self.to_csl_chunk(c) for c in norm.split(" "))

    def to_csl_chunk(self, chunk):
        """CSL for a single whitespace-free chunk."""
        return apply_labels(chunk, self.predict([chunk])[0]) if chunk else ""

    # ---- serialisation ----------------------------------------------------------------------
    def to_disk(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "vocab.json").write_text(json.dumps(
            {"chars": self.chars, "labels": self.labels,
             "width": self.width, "depth": self.depth}, ensure_ascii=False), encoding="utf-8")
        srsly.write_msgpack(path / "model.bin", self.model.to_bytes())

    @classmethod
    def from_disk(cls, path):
        path = pathlib.Path(path)
        meta = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        obj = cls(meta["chars"], meta["labels"], width=meta["width"], depth=meta["depth"])
        obj.model.initialize()
        obj.model.from_bytes(srsly.read_msgpack(path / "model.bin"))
        return obj


def apply_labels(samhita, labels):
    """Apply per-character labels to a saṃhitā string, producing CSL.

    A label starting with `=` keeps the character and appends the rest; anything else replaces it
    (an empty label deletes it). Mirrors `make_samhita_pairs.expand` — kept here so the runtime has
    no dependency on the data-generation script.
    """
    out = []
    for ch, lab in zip(samhita, labels):
        out.append(ch + lab[1:] if lab.startswith(KEEP) else lab)
    return "".join(out)


def build_vocabs(rows):
    """Character and label inventories from training rows, with PAD/UNK reserved at 0/1."""
    chars, labels = set(), set()
    for r in rows:
        chars.update(r["samhita"])
        labels.update(r["labels"])
    return ([PAD_CHAR, UNK_CHAR] + sorted(chars),
            [KEEP] + sorted(labels - {KEEP}))
