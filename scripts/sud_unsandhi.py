#!/usr/bin/env python3
"""`sud_unsandhi` — a learned sandhi-reversal component, writing `Token._.unsandhied`.

WHAT IT IS FOR. Under the DCS representation (`scripts/restructure_sa_csl.py`) a token that is its
own orthographic word keeps its SANDHIED surface, so the parse is over sandhied forms and the
unsandhied (padapāṭha) form is no longer recoverable from the FORM column. The treebank records it
as `Unsandhied=` on 100 % of Vedic tokens, and this component learns to predict it.

WHY LEARNED RATHER THAN RULE-BASED. `sa_tokenizer.desandhi_csl` reverses sandhi by rule and gets
~96 % of MWT-internal tokens right, but the residue is provably not rule-reachable from the
surface: the treebank wants `saṃ`->`sam`, `udag`->`udak`, `nir`->`niḥ` (the final reductions APPLY)
but `prāc`->`prāc`, `catur`->`catur`, `ahar`->`ahar`, `tad`->`tad` (they do NOT). Those pull in
opposite directions on identical surface shapes, so the choice is lexical. A transducer trained on
the gold learns it; a rule cannot.

WHY AN EDIT-TREE MODEL. sandhied -> unsandhied is a string transduction over a closed set of
alternations, which is exactly what spaCy's `EditTreeLemmatizer` is: it learns FORM->TARGET
character edits and is script-agnostic. So this subclasses it and changes only the OUTPUT
attribute. Training uses the stock `trainable_lemmatizer` on a corpus whose LEMMA column holds the
`Unsandhied` value (`scripts/make_unsandhi_corpus.py`) — because `Unsandhied` lives in MISC and does
not survive `spacy convert` — and the trained weights are then loaded into this class, which writes
`Token._.unsandhied` and leaves `token.lemma_` to the real lemmatiser.

The two components must therefore both be present and must NOT collide: `lemmatizer` writes
`lemma_`, `sud_unsandhi` writes `_.unsandhied`.
"""
from typing import Iterable

from spacy.language import Language
from spacy.pipeline.edit_tree_lemmatizer import EditTreeLemmatizer
from spacy.tokens import Doc, Token

if not Token.has_extension("unsandhied"):
    # str, defaulting to "" — a caller can always ask; an empty value means "not predicted".
    Token.set_extension("unsandhied", default="")


@Language.factory(
    "sud_unsandhi",
    default_config={
        "model": {
            "@architectures": "spacy.Tagger.v2",
            "nO": None,
            "normalize": False,
            "tok2vec": {
                "@architectures": "spacy.HashEmbedCNN.v2",
                "pretrained_vectors": None,
                "width": 64,
                "depth": 3,
                "embed_size": 2000,
                "window_size": 1,
                "maxout_pieces": 3,
                "subword_features": True,
            },
        },
        "backoff": "orth",
        "min_tree_freq": 3,
        "overwrite": False,
        "top_k": 1,
        "scorer": None,
    },
    default_score_weights={},
)
def make_sud_unsandhi(nlp, name, model, backoff, min_tree_freq, overwrite, top_k, scorer):
    return SudUnsandhi(nlp.vocab, model, name=name, backoff=backoff,
                       min_tree_freq=min_tree_freq, overwrite=overwrite, top_k=top_k,
                       scorer=scorer)


class SudUnsandhi(EditTreeLemmatizer):
    """EditTreeLemmatizer that writes `Token._.unsandhied` instead of `token.lemma_`."""

    def set_annotations(self, docs: Iterable[Doc], batch_tree_ids):
        for i, doc in enumerate(docs):
            doc_tree_ids = batch_tree_ids[i]
            if hasattr(doc_tree_ids, "get"):
                doc_tree_ids = doc_tree_ids.get()
            for j, tree_id in enumerate(doc_tree_ids):
                if tree_id == -1:
                    # no applicable tree: fall back to the surface form, as the lemmatiser does
                    doc[j]._.unsandhied = doc[j].text
                else:
                    doc[j]._.unsandhied = self.trees.apply(tree_id, doc[j].text)
