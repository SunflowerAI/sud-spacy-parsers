#!/usr/bin/env python3
"""Train-time vocalisation augmentation for Arabic and Persian: one copy, resampled every epoch.

The direct counterpart of `la_augment.py`, and it exists because the released arms fall off a
cliff on text they were never shown. Measured on the SAME trees with only the FORM column rewritten
(`make_ar_variant_conllu.py` / `eval_ar_variants.py`), the released Arabic arm reads:

    bare (as trained)   LAS 72.92        p50 (half the tokens pointed)   LAS 44.81
    shadda only         LAS 63.72        fully vocalised                 LAS 18.50

a spread of **54.42 LAS** -- which is, to the decimal, the spread Latin had before its own
augmentation (54.4). Fully pointed Arabic is not exotic: it is scripture, children's books, poetry,
dictionaries and language teaching, and partial pointing is ordinary in edited prose.

THE CORPUS IS STORED FULLY VOCALISED AND EVERY LIGHTER SPELLING IS DERIVED, exactly as Latin stores
the macronised copy and strips. `make_ar_vocalised_corpus.py` writes FORM = Vform; the augmenter
only ever REMOVES marks. So the marked copy is a strict superset, nothing is stored twice, and the
bare spelling cannot drift from the pointed one.

    [corpora.train]
    @readers = "sud.GoldTokCorpus.v1"
    shuffle = true
    [corpora.train.augmenter]
    @augmenters = "sud.ar_vocal_variants.v1"

⚠ **`max_epochs` must be `-1`**, and it brings two companions. At `0` spaCy's
`create_train_batches` does `examples = list(corpus(nlp))` ONCE and reshuffles that same list every
epoch, so a corpus-level augmenter samples ONE style per document for the whole run -- the run
looks entirely normal and trains on a single fixed perturbation. `-1` streams instead, which also
turns off the loop's own shuffling, hence `shuffle = true` on the reader. And `init_nlp`
initialises from `islice(train_corpus(nlp), 100)`, which truncates label inventories; see
`init_aug_labels.py`. For these two arms the LEMMATISER is again the component that needs it --
edit trees are properties of the FORM, so `كتاب` and `كِتاب` are different trees, and a missing one
does NOT raise but is silently taught label 0.

Nothing here ships in a wheel: this is a training-time reader hook, and what comes out is an
ordinary spaCy pipeline.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Iterator

from spacy.language import Language
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar_orth   # noqa: E402
import fa_orth   # noqa: E402

# `spacy train --code` takes ONE file (unlike `spacy package`), so this module has to register
# everything the config names: the reader, the per-language tokenisers and the tagger's feats
# embed. That is what `seg_code.py` exists for on the other arms. Module scope is fine here --
# nothing in this file ever travels inside a wheel, so the rule against training-only imports at
# module scope (which broke the zh wheel twice) does not apply.
import gold_tok_corpus      # noqa: E402,F401  sud.GoldTokCorpus.v1
import sud_feats_embed      # noqa: E402,F401  sud.Tok2VecPlusFeats.v1
try:
    import ar_tokenizer     # noqa: F401       ar.CamelAtbTokenizer.v1
except Exception:           # camel-tools absent: fine unless an ar config asks for it
    pass


def _rebuild(nlp: Language, example: Example, words: list[str]) -> Example:
    """Swap the word forms, keeping every other annotation on its token.

    Round-trips through `example.to_dict()` rather than copying fields by hand -- this project has
    been bitten twice by a Doc rebuild that dropped an annotation nobody thought about (lemma and
    morph, then Token extensions). LEMMA is deliberately left canonical: it is the lemmatiser's
    target, and the whole point is that a form written four ways still reaches one lemma.
    """
    ref = example.reference
    if words == [t.text for t in ref]:
        return example
    data = example.to_dict()
    data["token_annotation"]["ORTH"] = words
    predicted = Doc(nlp.vocab, words=words, spaces=[bool(t.whitespace_) for t in ref])
    return Example.from_dict(predicted, data)


@registry.augmenters("sud.ar_vocal_variants.v1")
def create_ar_vocal_augmenter(
    p_bare: float = 0.40,
    p_full: float = 0.15,
    min_rate: float = 0.05,
    max_rate: float = 0.95,
    p_hamza_fold: float = 0.5,
    p_digit_fold: float = 0.5,
    keep_original: bool = False,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """See `ar_orth.OrthPolicy` for what each rate means.

    `p_bare` is the one to think about: undiacritised text is the overwhelming majority of written
    Arabic and is what every published figure for this arm is measured on, so the augmentation must
    widen the model's range without moving its centre of mass off the spelling it is judged on.
    """
    policy = ar_orth.OrthPolicy(p_bare=p_bare, p_full=p_full, min_rate=min_rate,
                                max_rate=max_rate, p_hamza_fold=p_hamza_fold,
                                p_digit_fold=p_digit_fold)
    rng = random.Random(seed)

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if keep_original:
            yield example
        style = ar_orth.sample_style(rng, policy)
        yield _rebuild(nlp, example,
                       [ar_orth.vary_word(t.text, style, rng) for t in example.reference])

    return augmenter


@registry.augmenters("sud.fa_vocal_variants.v1")
def create_fa_vocal_augmenter(
    lut: str = "scripts/fa_vocalise_lut.json.gz",
    ezafe_rules: str = "scripts/fa_ezafe_rules.json",
    p_bare: float = 0.45,
    p_full: float = 0.10,
    p_ezafe: float = 0.35,
    p_arabic: float = 0.20,
    p_zwnj: float = 0.20,
    keep_original: bool = False,
    seed: int = 0,
) -> Callable[[Language, Example], Iterator[Example]]:
    """Persian runs the OPPOSITE way to Arabic: the corpus stays as the treebank writes it and
    marks are ADDED, because no vocalised Persian gold exists to strip from. The table is the same
    reconstruction `fa_vocalise` ships against and the ezāfe rules the same syntactically-derived
    ones -- so the augmentation and the component agree by construction rather than by coincidence.

    ⚠ The ezāfe here is read off the GOLD parse in `example.reference`, which is exactly right for
    training-time augmentation and is NOT leakage: the target is the tree, the ezāfe is being put
    into the INPUT, and at inference the component derives it from the predicted parse instead.
    """
    import gzip
    import json

    forms = {}
    p = Path(lut)
    if p.exists():
        forms = dict(json.loads(gzip.open(p, "rb").read().decode("utf-8")).get("F", []))
    rules = {}
    pr = Path(ezafe_rules)
    if pr.exists():
        rules = json.loads(pr.read_text(encoding="utf-8"))
    if not forms:
        raise ValueError(
            f"sud.fa_vocal_variants.v1 found no table at {lut}. Build one with "
            "`python scripts/build_fa_vocalise_lut.py --kaamel KaamelDict.csv`. Unlike the "
            "runtime component, which degrades on purpose, an augmenter with no data would "
            "train a plain arm while claiming to train an augmented one -- so this raises.")
    policy = fa_orth.OrthPolicy(p_bare=p_bare, p_full=p_full, p_ezafe=p_ezafe,
                                p_arabic=p_arabic, p_zwnj=p_zwnj)
    rng = random.Random(seed)

    def takes_ezafe(ref, i):
        """The next token is this token's own dependent, in a configuration the derived table
        keeps. Gold heads, since this is the reference doc."""
        if i + 1 >= len(ref):
            return False
        nxt = ref[i + 1]
        if nxt.head.i != i:
            return False
        return "|".join((ref[i].pos_, nxt.dep_, nxt.pos_)) in rules

    def augmenter(nlp: Language, example: Example) -> Iterator[Example]:
        if keep_original:
            yield example
        style = fa_orth.sample_style(rng, policy)
        ref = example.reference
        words = [fa_orth.vary_word(t.text, forms.get(fa_orth.strip_diac(t.text)),
                                   takes_ezafe(ref, i), style, rng)
                 for i, t in enumerate(ref)]
        yield _rebuild(nlp, example, words)

    return augmenter
