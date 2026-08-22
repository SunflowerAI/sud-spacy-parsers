#!/usr/bin/env python3
"""`sud.TamilSandhiTokenizer.v1` — the trained character segmenter, run in DECOMPOSED space.

WHAT IT IS FOR. SUD_Tamil-TTB's syntactic words are not a whitespace segmentation of Tamil text:
835 orthographic words in the treebank split into 1 781, and 94.2 % of those splits rewrite at the
seam. spaCy's rule tokeniser cannot produce them at all, which caps strict token F at about 0.92
however good the parser is — and no parsing metric in this project would ever show it, because
`--gold-preproc` bypasses the tokeniser and `sud.GoldTokCorpus.v1` makes the parser
segmenter-agnostic (the same blind spot `docs/lzh-tokenisation.md` records for 孔子).

HOW IT DIFFERS FROM `sud.CharSegTokenizer.v1`, which it subclasses. That tokeniser can only CUT a
string, and cutting is exactly what Tamil does not do:

    நிலையத்துக்குக்கான  ->  நிலையத்துக்குக்க்  +  ஆன        the seam falls INSIDE the character கா

`scripts/ta_sandhi.py` makes the cut a cut: Tamil is an abugida, so decomposing every akṣara into
consonant + virāma + independent vowel puts a boundary where the treebank wants one. Then the split
is ordinary segmentation and each piece is recomposed on the way out. Measured over both treebanks,
the round trip is exact on 13 043 of 13 043 tokens and the gold parts are a clean segmentation of
the decomposed surface on 95.90 % of ranges (5.8 % undecomposed).

⚠ **THE INPUT REGIME TRAVELS WITH THE WEIGHTS** (`reads_decomposed` in the bundled
`ta_tokenizer.json`) and is READ BACK, not assumed. Standing hazard 10 in CLAUDE.md is this exact
mistake twice over: a CSLiser trained on spaced text was fed space-split chunks for a whole
generation at −4.83 F. A segmenter trained on decomposed Tamil and handed composed Tamil would not
raise — it would meet a character inventory it half recognises and quietly under-split — so the
marker is checked and a mismatch refuses.

⚠ **THIS TOKENIZER REWRITES ITS INPUT**, as Sanskrit's does. `doc.text` is the recomposed parts and
does not always equal the string handed in — where the treebank undoes gemination
(`கஷ்டப்படுகிறான்` -> `கஷ்ட` + `படுகிறான்`) the doubled consonant is gone. So a Doc from this
tokenizer must never be re-tokenised, and anything rebuilding a Doc must build it from the WORDS.

With no segmenter on disk it degrades to whitespace rather than raising, so a partially-built
pipeline still loads — the same contract `CharSegTokenizer` has.
"""
from __future__ import annotations

import json
import pathlib
import sys

from spacy.tokens import Doc
from spacy.util import registry

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from char_seg_tokenizer import BREAK, SEG_BATCH, CharSegTokenizer  # noqa: E402
from ta_sandhi import decompose, recompose  # noqa: E402

MARKER = "ta_tokenizer.json"


@registry.tokenizers("sud.TamilSandhiTokenizer.v1")
def create_ta_sandhi_tokenizer():
    def make(nlp):
        return TamilSandhiTokenizer(nlp.vocab)
    return make


class TamilSandhiTokenizer(CharSegTokenizer):
    """`CharSegTokenizer`, but the segmenter sees decomposed Tamil and the caller sees composed."""

    def _decomposed_parts(self, chunk: str, labels) -> list[str]:
        parts, cur = [], ""
        for ch, label in zip(chunk, labels):
            cur += ch
            if label == BREAK:
                parts.append(cur)
                cur = ""
        if cur:
            parts.append(cur)
        return parts

    def __call__(self, text):
        if not text.strip():
            return Doc(self.vocab, words=[], spaces=[])

        chunks, trailing, i = [], [], 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            chunks.append(text[i:j])
            trailing.append(j < len(text))
            i = j

        if self.seg is None:                       # no model: whitespace only, never raise
            return Doc(self.vocab, words=chunks, spaces=trailing)

        # Batched at `SEG_BATCH`, exactly as `CharSegTokenizer.__call__` does it, and for the
        # memory reason recorded there: an unbatched `predict` is linear in the length of the
        # CALLING STRING at 10-14 kB per character, which is fatal under Pyodide on a book-sized
        # input. Rows cannot see each other -- `build_lex_model` pads by the full receptive field
        # -- so the batch size does not touch the output.
        decomposed = [decompose(c) for c in chunks]
        preds = []
        for i in range(0, len(decomposed), SEG_BATCH):
            preds.extend(self.seg.predict(decomposed[i:i + SEG_BATCH]))

        words, spaces = [], []
        for chunk, labels, space in zip(decomposed, preds, trailing):
            parts = [recompose(p) for p in self._decomposed_parts(chunk, labels)]
            parts = [p for p in parts if p]        # a boundary at the very end yields no piece
            if not parts:
                parts = [recompose(chunk)]
            for k, part in enumerate(parts):
                words.append(part)
                spaces.append(space if k == len(parts) - 1 else False)
        return Doc(self.vocab, words=words, spaces=spaces)

    # ---- serialisation ---------------------------------------------------------------------
    def to_disk(self, path, **kwargs):
        super().to_disk(path, **kwargs)
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / MARKER).write_text(json.dumps({"reads_decomposed": True}), encoding="utf-8")

    def from_disk(self, path, **kwargs):
        p = pathlib.Path(path)
        super().from_disk(path, **kwargs)
        if self.seg is not None:
            marker = p / MARKER
            regime = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
            if not regime.get("reads_decomposed"):
                raise ValueError(
                    f"sud.TamilSandhiTokenizer.v1: the segmenter at {p} does not declare "
                    f"`reads_decomposed`. This tokenizer feeds its model akṣara-DECOMPOSED Tamil "
                    f"(scripts/ta_sandhi.py); a model trained on composed text would not raise "
                    f"here, it would quietly under-split. Retrain with scripts/train_ta_charseg.sh "
                    f"or use sud.CharSegTokenizer.v1 if the model really is a composed-text one.")
        return self
