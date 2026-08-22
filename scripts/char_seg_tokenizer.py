#!/usr/bin/env python3
"""A spaCy tokenizer driven by the character segmenter trained in `sa_presegment.py`.

Registered as **`sud.CharSegTokenizer.v1`**, language-neutral. It exists because the shipped
statistical segmenters disagree with the treebanks the parsers were trained on, and a treebank-
trained tagger is only worth what its tokenisation is worth:

    zh   pkuseg 0.8385 -> 0.8725 plain -> 0.8898 with a jackknifed corpus lexicon -> 0.9202 with
         jieba's own segmentation decision as a second channel (`zh_jieba_feature.py`)
    id   the enclitic split `coarsen_id.py` merges away: 0.9985, 99.91 % of words segmented exactly

All strict whole-token F, scored PER TEXT. Which rows share a `predict` call USED to matter:
`with_array` concatenates the batch, so before `build_lex_model` passed `pad`, the first character
of each row saw its neighbour instead of zero padding (|Δ| 0.81 at position 0, ~0 by position 2 —
worth 0.27 F on zh, where the sentence-initial split is genuinely uncertain, and 0.00 on id/yue/sa).
Both builders now pad by `window_size * depth`, the full receptive field, so no character can read
across a row boundary and rows are INDEPENDENT. That is what lets `__call__` cap its batch at
`SEG_BATCH` below without touching a single output token — verified, not assumed: 320 k characters
of Chinese Union Version give the same 158,712 tokens at batch 64, 512 and unbatched, and the id,
lzh and ta segmenters agree with themselves the same way.

Boundaries come from the same two-label scheme `make_seg_pairs.py` writes (`=` keep, `= ` break),
and inference runs per WHITESPACE CHUNK — the model never saw a space, so feeding it one hands the
encoder UNK at every word boundary. Whitespace in the input is therefore always a boundary, which is
also the right semantics: the model only has to find the boundaries a writer did not mark.

Serialisation is real, not a no-op: `to_disk` writes the segmenter beside the tokenizer entry so a
packaged wheel keeps it. `from_disk` restores it, and a model whose directory has no segmenter falls
back to whitespace rather than raising, so a partially-built pipeline still loads.
"""
import pathlib

from spacy.tokens import Doc
from spacy.util import registry

KEEP, BREAK = "=", "= "

# How many whitespace chunks go into ONE `predict` call. Not a speed knob -- a MEMORY cap.
# `__call__` used to hand the whole input to a single `predict`, and `with_array` flattens that
# batch into one array, so peak memory was linear in the length of the CALLING STRING with a
# constant of 10-14 kB per character: 320 k characters of the Chinese Union Version peaked between
# 3,457 and 4,626 MB across runs, so a whole Bible in one call wants order 10 GB and needs more
# address space than a browser tab has. Under Pyodide that is fatal before it is slow.
#
# Batching is output-IDENTICAL rather than an approximation, and that is a property of the
# architecture, not a hope: both `build_lex_model` and `sa_presegment.build_model` pass
# `pad=window_size * depth`, which is exactly the receptive field, so `with_array` separates rows
# by enough zero padding that no character can see across a row boundary. Rows are therefore
# independent and it cannot matter which call they arrive in. Verified on 320 k characters of CUV:
# batch 512 and batch 64 both reproduce the unbatched 158,712 tokens byte for byte, at 1,050 MB
# and 371 MB peak -- and both are FASTER than the single call (9.6 s / 8.9 s against 11.4 s).
#
# ⚠ Batching the chunk list is not the same as slicing the TEXT. Slicing text cuts chunks in half
# and does change the tokenisation (158,712 -> 158,726 at an 8 k-character step). Only the chunk
# list may be batched.
SEG_BATCH = 256


@registry.tokenizers("sud.CharSegTokenizer.v1")
def create_char_seg_tokenizer():
    def make(nlp):
        return CharSegTokenizer(nlp.vocab)
    return make


class CharSegTokenizer:
    def __init__(self, vocab, model_path=None):
        self.vocab = vocab
        self.seg = None
        self.lexicon = None
        if model_path:
            self.load_segmenter(model_path)

    def load_segmenter(self, path, lexicon=None):
        """Load either a plain character segmenter or a lexicon-feature one.

        A `LexPresegmenter` needs its lexicon at inference, so the word list is bundled beside the
        weights and reloaded here. It must be the FULL training lexicon: the jackknifing that made
        this model work applies only during TRAINING (each fold saw a lexicon built from the other
        folds, so train-time coverage matched the ~87.6 % it meets at test). At inference the model
        expects the complete list, which is what it was evaluated with.
        """
        import json
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        path = pathlib.Path(path)
        meta = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        lex_file = pathlib.Path(lexicon) if lexicon else path / "lexicon.txt"
        if "n_sources" in meta and lex_file.exists():
            import sa_presegment_lex as spl
            if meta.get("jieba_source") is not None:
                # One channel is jieba's own segmentation of the chunk, not a word list. It must be
                # switched on BEFORE the model is built, since `multi_codes` reads the module state.
                ud = path / "jieba_force_split.txt"
                # `jieba_t2s` and `jieba_dict` travel with the weights for the same reason
                # `jieba_source` does: a traditional segmenter trained on jieba's view of the
                # SIMPLIFIED rendering, or on a traditional dictionary, must be asked the same
                # question here, and nothing would raise if it were not.
                dict_path = None
                if meta.get("jieba_dict"):
                    import zh_jieba_feature as jf
                    dict_path = path / jf.TRAD_DICT_FILE
                    if not dict_path.is_file():
                        # Refuse rather than fall back on jieba's simplified dictionary. Falling
                        # back is precisely how a wheel ships one input quietly deleted: the model
                        # would load, segment, and be wrong only where the vocabulary differs.
                        raise FileNotFoundError(
                            f"{path} was trained against the {meta['jieba_dict']} jieba dictionary "
                            f"but {dict_path.name} is not beside its weights — refusing to segment "
                            f"with a dictionary the model was not trained on")
                spl.enable_jieba(meta["jieba_source"], str(ud) if ud.exists() else None,
                                 t2s=meta.get("jieba_t2s", False), dict_path=dict_path)
            entries = {w for w in lex_file.read_text(encoding="utf-8").split("\n") if w}
            self.lexicon = entries
            self.seg = spl.LexPresegmenter.from_disk(path, [entries] * meta["n_sources"])
        else:
            from sa_presegment import Presegmenter
            self.seg = Presegmenter.from_disk(path)

    # ---- inference ---------------------------------------------------------------------------
    def _split_chunk(self, chunk, labels):
        out, cur = [], ""
        for ch, lb in zip(chunk, labels):
            cur += ch
            if lb == BREAK:
                out.append(cur)
                cur = ""
        if cur:
            out.append(cur)
        return out

    def __call__(self, text):
        if not text.strip():
            return Doc(self.vocab, words=[], spaces=[])
        # chunk on whitespace, remembering whether each chunk was followed by a space
        chunks, spaces, i = [], [], 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            chunks.append(text[i:j])
            spaces.append(j < len(text))
            i = j

        if self.seg is None:                      # no model: whitespace only, never raise
            return Doc(self.vocab, words=chunks, spaces=spaces)

        preds = []
        for i in range(0, len(chunks), SEG_BATCH):
            preds.extend(self.seg.predict(chunks[i:i + SEG_BATCH]))
        words, sp = [], []
        for chunk, labels, trailing in zip(chunks, preds, spaces):
            parts = self._split_chunk(chunk, labels)
            for k, part in enumerate(parts):
                words.append(part)
                sp.append(trailing if k == len(parts) - 1 else False)
        return Doc(self.vocab, words=words, spaces=sp)

    def pipe(self, texts, **kwargs):
        for t in texts:
            yield self(t)

    # ---- serialisation -----------------------------------------------------------------------
    def to_disk(self, path, **kwargs):
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self.seg is not None:
            self.seg.to_disk(p / "segmenter")
            if self.lexicon:                       # bundle the word list beside the weights
                (p / "segmenter" / "lexicon.txt").write_text(
                    "\n".join(sorted(self.lexicon)), encoding="utf-8")

    def from_disk(self, path, **kwargs):
        p = pathlib.Path(path)
        if (p / "segmenter").exists():
            self.load_segmenter(p / "segmenter")
        return self

    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, data, **kwargs):
        return self
