"""Registers the `sud_subject_rule` factory: `Subject=` from lexical raising frames, no model.

The alternative to the trained `sud_tagger` for this feature, and for some languages the better
one. Two measurements motivate it:

  * The VALUE is not in doubt. Given (deprel, head UPOS), `SubjRaising` vs `ObjRaising` is
    determined at 100% in en/fa/la/sa/yue and 91% in zh -- there are only 3-10 distinct contexts
    per language. Nothing needs to learn it.
  * The PRESENCE is what is hard, and how hard varies enormously by language. A frame table keyed
    on (head lemma, deprel, head UPOS), harvested from train and scored on test, gets F 95.3 in
    Classical Chinese but only 36-74 elsewhere. Classical Chinese raising is carried by a handful
    of verbs (可, 能, 欲); English and Persian raising is spread across a long lexical tail that a
    table cannot cover.

So neither approach dominates, and the choice is empirical per language -- compare with
`scripts/eval_sud_subject.py` and ship whichever wins. Note the two are not on the same footing at
inference: this component reads the PARSER's deprel and the head's predicted UPOS, so its accuracy
degrades with parse quality, whereas `sud_tagger` reads only surface forms through its own encoder.
`eval_sud_subject.py` therefore compares them end-to-end, not on gold trees.

The frame table is built by `scripts/build_sud_subject_frames.py` and embedded there as a literal,
in the manner of `id_lemma_case_fix`'s lookup table.

Load with `spacy ... --code scripts/sud_misc.py,scripts/sud_subject_rule.py`.
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
sud_subject_frames = _sibling("sud_subject_frames")
set_misc = sud_misc.set_misc


class SudSubjectRule:
    """Stamp `Subject=` where the head lemma is a known raising frame."""

    def __init__(self, nlp, name, lang):
        self.lang = lang
        self.frames = set(map(tuple, sud_subject_frames.FRAMES.get(lang, [])))
        # (deprel, head UPOS) -> value; the value side is deterministic (see the docstring)
        self.values = {tuple(k.split("\t")): v
                       for k, v in sud_subject_frames.VALUES.get(lang, {}).items()}

    def __call__(self, doc):
        for tok in doc:
            head = tok.head
            if head.i == tok.i:
                continue
            dep = tok.dep_
            key = (head.lemma_, dep, head.pos_)
            if key not in self.frames:
                continue
            # Frames are keyed on the full deprel; the value table may only have the base
            # relation, so fall back to it with the `@`-suffixed deep feature stripped.
            value = self.values.get((dep, head.pos_)) \
                or self.values.get((dep.split("@", 1)[0], head.pos_))
            if value:
                set_misc(tok, "Subject", value)
        return doc


def make_sud_subject_rule(nlp, name, lang):
    return SudSubjectRule(nlp, name, lang)


if not Language.has_factory("sud_subject_rule"):
    Language.factory("sud_subject_rule", default_config={"lang": "en"})(make_sud_subject_rule)
