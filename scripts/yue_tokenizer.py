#!/usr/bin/env python3
"""Tokeniser registration for Cantonese (yue).

spaCy ships no `yue` language module, so the Cantonese pipeline runs on a custom `yue`
language (registered here, loaded via `spacy ... --code scripts/yue_tokenizer.py`) so the
model packages as `yue_sud_hk`.

Unlike Classical Chinese (Kyoto = one Han char per token), SUD_Cantonese-HK is **word**-
tokenised (你, 喺度, 乜嘢 …) with no whitespace, so word boundaries are statistical, not a
deterministic function of the text — exactly the zh situation. The research pipeline therefore
runs on **gold tokens** (`gold_preproc = true` in the config + `--gold-preproc` at eval); the
tokeniser below only matters for raw-text inference.

For raw text the released model ships a **pkuseg word segmenter** trained on this treebank's gold
tokens (`yue.PkusegTokenizer.v1`; word-F1 0.95 vs the char fallback's 0.63). It bundles the model
into the pipeline (tokenizer/ dir) so `spacy.load` needs no external model. If pkuseg is
unavailable or no model is loaded, it degrades to the deterministic **character** tokeniser below
(one non-space char = one token, like lzh — reversible, never crashes, but at the wrong
*granularity* for multi-character Cantonese words). Training/eval use gold tokens (gold_preproc),
so the tokeniser choice does not affect the disambiguation metrics.
"""
from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import ensure_path, get_words_and_spaces, registry


class CantoneseDefaults(Language.Defaults):
    """Minimal defaults; the character tokeniser is supplied via the config's
    [nlp.tokenizer] @tokenizers = "yue.CharTokenizer.v1"."""


@registry.languages("yue")
class Cantonese(Language):
    """A real `yue` language so the model packages as `yue_sud_hk` (spaCy ships no native
    Cantonese module). Registered when this file is loaded via `spacy ... --code`."""
    lang = "yue"
    Defaults = CantoneseDefaults


@registry.tokenizers("yue.CharTokenizer.v1")
def make_char_tokenizer():
    def create(nlp):
        return CharTokenizer(nlp.vocab)
    return create


class CharTokenizer:
    """One token per non-whitespace character; whitespace sets the preceding token's
    trailing space (so the original text round-trips). Deterministic raw-text fallback —
    the parser itself is trained/evaluated on gold tokens via gold_preproc."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words, spaces = [], []
        for ch in text:
            if ch.isspace():
                if spaces:
                    spaces[-1] = True
                continue
            words.append(ch)
            spaces.append(False)
        if not words:
            return Doc(self.vocab, words=[], spaces=[])
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self


@registry.tokenizers("yue.PkusegTokenizer.v1")
def make_pkuseg_tokenizer():
    def create(nlp):
        return PkusegTokenizer(nlp.vocab)
    return create


class PkusegTokenizer:
    """Cantonese word segmenter backed by a pkuseg CRF model trained on the SUD_Cantonese-HK gold
    tokens. The model is loaded via `initialize(pkuseg_model=...)` at build time and serialised
    into the pipeline's tokenizer/ dir (model.save + feature_extractor.save), so a packaged model
    is self-contained. Falls back to the character tokeniser when no model is loaded (or
    spacy-pkuseg is missing). No user dict — it regressed word-F1 on this treebank."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.seg = None
        self._char = CharTokenizer(vocab)   # fallback before a model is loaded

    def initialize(self, get_examples=None, *, nlp=None, pkuseg_model=None):
        if pkuseg_model:
            import spacy_pkuseg
            self.seg = spacy_pkuseg.pkuseg(pkuseg_model)

    def __call__(self, text):
        if self.seg is None:
            return self._char(text)
        words = [w for w in self.seg.cut(text) if w]
        words, spaces = get_words_and_spaces(words, text)   # recover SpaceAfter from raw text
        return Doc(self.vocab, words=words, spaces=spaces)

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes, **kwargs):
        return self

    def to_disk(self, path, **kwargs):
        path = ensure_path(path)
        if self.seg is not None:
            if not path.exists():
                path.mkdir(parents=True)
            self.seg.model.save(str(path))
            self.seg.feature_extractor.save(str(path))

    def from_disk(self, path, **kwargs):
        path = ensure_path(path)
        if path.exists() and any(path.iterdir()):
            import spacy_pkuseg
            self.seg = spacy_pkuseg.pkuseg(str(path))
        return self
