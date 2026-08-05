"""Registers the `sud_shared_rule` factory: `Shared=` from the parse, no model.

The alternative to the trained `sud_tagger` for this feature, and the arm to beat. What it does is
read the coordination out of the predicted tree (`sud_shared_data.candidates`) and look the
decision up in a table harvested from train (`build_sud_shared_frames.py`).

It exists because `Shared` is not a lexical fact at all. It says whether a dependent of a conjunct
is shared with the other conjuncts, so everything that bears on it -- is my head a conjunct, where
do the conjuncts sit, am I inside or outside their span -- is in the tree, and a table over
(deprel, head UPOS, position) can read it directly. That is also exactly what the morphologiser
cannot do: it has been predicting `Shared` all along, as part of the FEATS bundles the treebanks
put it in, from a small local encoder over word forms with no view of the parse. On English test
that gets P 0.68 / R 0.15, with `Shared=Yes` correct 4 times out of 247.

Which arm ships is measured per language with `scripts/eval_sud_shared.py`, on the same footing --
both end to end over gold tokens, since this component reads PREDICTED heads and relations and
therefore degrades with parse quality, while the trained pipe reads its own encoder.

LIKE THE TRAINED PIPE, THIS TAKES THE FEATURE OVER: it deletes `Shared` from `token.morph` before
writing the slot, so an arm carrying it has one answer rather than two contradictory ones. Set
`clear_morph = false` if you want to see both (for a diagnostic; do not ship it that way).

Load with `spacy ... --code scripts/sud_misc.py,scripts/sud_shared_data.py,scripts/sud_shared_rule.py`.
"""
from spacy.language import Language


def _sibling(name):
    """Import a sibling module across all three ways this file gets loaded.

    Wheel (relative import), `seg_code.py` (scripts/ on sys.path), and `spacy package` (each
    --code file loaded standalone, where only the file-path fallback works).
    """
    import importlib
    import importlib.util
    import pathlib
    import sys as _sys

    if __package__:
        try:
            return importlib.import_module("." + name, __package__)
        except ImportError:
            pass
    if name in _sys.modules:
        return _sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    _sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sud_misc = _sibling("sud_misc")
sud_shared_data = _sibling("sud_shared_data")
sud_shared_frames = _sibling("sud_shared_frames")
set_misc = sud_misc.set_misc
backoff_keys = sud_shared_data.backoff_keys
doc_candidates = sud_shared_data.doc_candidates

NEG = "O"


class SudSharedRule:
    """Stamp `Shared=` on the dependents of conjuncts, from the harvested decision table."""

    def __init__(self, nlp, name, lang, clear_morph):
        self.lang = lang
        self.clear_morph = clear_morph
        self.table = dict(sud_shared_frames.TABLE.get(lang, {}))

    def _lookup(self, keys):
        for key in keys:
            value = self.table.get(key)
            if value is not None:
                return value
        return NEG

    def __call__(self, doc):
        if self.clear_morph:
            # Unset MORPH rather than stamping an empty one where nothing is left: `set_morph({})`
            # yields morph key 456 and an untouched token key 0, both rendering as `''`, so the
            # difference is invisible to any string check but not to an encoder (CLAUDE.md).
            for token in doc:
                if token.morph.get("Shared"):
                    rest = {k: v for k, v in token.morph.to_dict().items() if k != "Shared"}
                    token.set_morph(rest or None)
        if not self.table:
            return doc
        for i, position in doc_candidates(doc):
            token = doc[i]
            value = self._lookup(backoff_keys(token.dep_, token.head.pos_, position))
            if value != NEG:
                set_misc(token, "Shared", value)
        return doc


def make_sud_shared_rule(nlp, name, lang, clear_morph):
    return SudSharedRule(nlp, name, lang, clear_morph)


if not Language.has_factory("sud_shared_rule"):
    Language.factory("sud_shared_rule",
                     default_config={"lang": "en", "clear_morph": True})(make_sud_shared_rule)
