#!/usr/bin/env python3
"""`sud.TeluguSplitTokenizer.v1` — a LOOKUP splitter for the multiword tokens added to MTG.

WHY IT HAS TO EXIST. `scripts/split_te_mwt.py` gives SUD_Telugu-MTG the multiword tokens it ships
without, and the parser is then trained on the split words. But the arm trains through
`sud.GoldTokCorpus.v1`, which hands it GOLD tokens — so nothing in training ever exercised a
tokeniser, and a wheel shipping spaCy's rule tokeniser would meet `ఇళ్ళున్నాయి` as ONE token at
inference having only ever seen it as `ఇళ్ళు` + `ఉన్నాయి`. That is CLAUDE.md's standing hazard 10,
input regime, in its purest form: the model is asked a different question at inference than at
training, and nothing raises.

⚠ **A LOOKUP, NOT A MODEL, AND THAT IS THE HONEST SHAPE HERE.** Tamil gets a trained character
segmenter because TTB splits 835 orthographic words and there is something to learn.
`split_te_mwt.py` commits **20 splits over 8 distinct types**, which is far too few to train
anything — and a segmenter fitted to eight types would be a lookup table wearing a model's clothes,
with the added property of firing confidently on words nobody checked. So this splits exactly the
types the re-annotation committed and nothing else, and says so.

⚠ **THIS DOES NOT GENERALISE, BY CONSTRUCTION.** Telugu's enunciative *-u* elision is productive;
`split_te_mwt.py` found 19 further orthographically-licensed candidates it could not commit because
the treebank does not determine their relation. Those stay fused here too. Extending coverage means
extending the ANNOTATION first — the table is derived from the split treebank, so it grows when
that does, and it can never claim a split the gold does not contain.

The table travels in the model directory (`te_split.json`), so the wheel is self-contained; with no
table the tokeniser degrades to whitespace-and-punctuation rather than raising, as
`sud.CharSegTokenizer.v1` does.
"""
from __future__ import annotations

import json
import pathlib

from spacy.tokens import Doc
from spacy.util import registry

TABLE = "te_split.json"


@registry.tokenizers("sud.TeluguSplitTokenizer.v1")
def create_te_split_tokenizer():
    def make(nlp):
        return TeluguSplitTokenizer(nlp.vocab)
    return make


class TeluguSplitTokenizer:
    """spaCy's rule tokeniser, plus a lookup that splits the re-annotated multiword tokens."""

    def __init__(self, vocab, table=None):
        self.vocab = vocab
        self.table: dict[str, list[str]] = dict(table or {})
        # Built against OUR vocab rather than borrowed from a fresh `Telugu()` and re-pointed:
        # `Tokenizer.vocab` is read-only, and a tokeniser holding a different vocab would mint
        # its own lexemes.
        from spacy.lang.te import Telugu
        from spacy.tokenizer import Tokenizer
        from spacy.util import (compile_infix_regex, compile_prefix_regex,
                                compile_suffix_regex)
        d = Telugu.Defaults
        self._base = Tokenizer(
            vocab,
            rules=d.tokenizer_exceptions,
            prefix_search=compile_prefix_regex(d.prefixes).search if d.prefixes else None,
            suffix_search=compile_suffix_regex(d.suffixes).search if d.suffixes else None,
            infix_finditer=compile_infix_regex(d.infixes).finditer if d.infixes else None,
            token_match=d.token_match,
            url_match=d.url_match,
        )

    def __call__(self, text):
        words, spaces = [], []
        for tok in self._base(text):
            parts = self.table.get(tok.text)
            if parts:
                for k, part in enumerate(parts):
                    words.append(part)
                    spaces.append(bool(tok.whitespace_) if k == len(parts) - 1 else False)
            else:
                words.append(tok.text)
                spaces.append(bool(tok.whitespace_))
        return Doc(self.vocab, words=words, spaces=spaces)

    def pipe(self, texts, **kwargs):
        for t in texts:
            yield self(t)

    def to_disk(self, path, **kwargs):
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / TABLE).write_text(json.dumps(self.table, ensure_ascii=False, indent=1),
                               encoding="utf-8")

    def from_disk(self, path, **kwargs):
        f = pathlib.Path(path) / TABLE
        if f.exists():
            self.table = json.loads(f.read_text(encoding="utf-8"))
        return self

    def to_bytes(self, **kwargs):
        return json.dumps(self.table, ensure_ascii=False).encode("utf-8")

    def from_bytes(self, data, **kwargs):
        self.table = json.loads(data.decode("utf-8"))
        return self


def table_from_conllu(paths) -> dict[str, list[str]]:
    """Harvest `orthographic word -> [syntactic words]` from the MWT ranges of a split treebank.

    Read off the ANNOTATION, never written by hand: the tokeniser can then only ever split what the
    treebank actually contains, and a re-run of `split_te_mwt.py` that commits more (or fewer)
    splits produces a table that matches it without anyone remembering to update this file.
    """
    table: dict[str, list[str]] = {}
    for path in paths:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            cols = lines[i].split("\t")
            if len(cols) == 10 and "-" in cols[0] and cols[0][0].isdigit():
                end = int(cols[0].split("-")[1])
                parts, j = [], i + 1
                while j < len(lines):
                    d = lines[j].split("\t")
                    if len(d) == 10 and d[0].isdigit() and int(d[0]) <= end:
                        parts.append(d[1])
                        j += 1
                    else:
                        break
                if parts:
                    table[cols[1]] = parts
                i = j
                continue
            i += 1
    return table
