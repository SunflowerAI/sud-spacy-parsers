"""Registers the `sud_idiom` factory: SUD's `Idiom=Yes` / `InIdiom=Yes` MISC layer, by rule.

SUD annotates multiword idioms and titles with features rather than a `fixed` relation, so that
their internal syntactic structure is preserved: the head carries `Idiom=Yes` plus an `ExtPos`
recording the unit's function in the wider sentence, and every remaining member carries
`InIdiom=Yes`. Where a member has no analysable internal structure it attaches by `unk`.

That description is also a recipe, and it turns out to be an exact one. Measured over the training
split of all seven treebanks that annotate idioms (en/lzh/ja/fa/ar/la/sa):

    Idiom=Yes    <=>  token has ExtPos AND has an `unk` dependent      P = R = 100%
    InIdiom=Yes  <=>  token attaches by `unk`, and walking up through
                      consecutive `unk` links reaches a head with ExtPos    P = R = 100%
                      (la 99.9% -- one token)

Both sides read only what the released pipeline already predicts: `ExtPos` is a FEATS feature the
morphologiser emits, and `unk` is an ordinary parser label. So this component needs no training,
no corpus rebuild and no retrain -- it is appended at packaging time, like `id_lemma_case_fix`.

The looser rules are NOT good enough, which is why the two conjuncts are both needed:
  * `unk` alone over-predicts `InIdiom` badly -- precision fa 6.5%, ar 53%, en 75%. Arabic's other
    `unk` tokens are newswire dateline artifacts; Persian's are something else again.
  * `ExtPos` alone over-predicts `Idiom` in English (702 ExtPos vs 477 Idiom) -- EWT uses ExtPos
    for fixed-expression heads that are not idioms in SUD's sense. Requiring an `unk` dependent
    separates them exactly.

NB the 100% figures are against GOLD trees. End to end the component inherits the morphologiser's
ExtPos errors and the parser's `unk` errors, so the released accuracy is lower; the gap between the
two numbers is the honest measure of what this adds.

Load with `spacy ... --code scripts/sud_misc.py,scripts/sud_idiom.py`.
"""
from spacy.language import Language


def _sibling(name):
    """Import a sibling module across all three ways this file gets loaded.

    A packaged wheel imports it as `pkg.sud_idiom`, so a relative import works; `seg_code.py`
    puts scripts/ on sys.path, so a plain import works; but `spacy package` loads each --code
    file standalone via spec_from_file_location, where NEITHER works -- hence the file-path
    fallback. Without it, packaging dies with `ModuleNotFoundError: No module named 'sud_misc'`.
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
set_misc = sud_misc.set_misc
deprel_base = sud_misc.deprel_base

UNK = "unk"


def _idiom_root(token):
    """Walk up through consecutive `unk` links; return the first head bearing ExtPos, else None.

    Idiom chains can be three or more tokens deep (a member whose head is itself `unk`), so a
    single hop to `token.head` is not enough -- in Japanese alone ~1600 `unk` tokens hang off
    another `unk`. Guarded against cycles and against the root, whose head in spaCy is itself.
    """
    seen = set()
    cur = token
    while cur.i not in seen:
        seen.add(cur.i)
        head = cur.head
        if head.i == cur.i:  # root: nothing above it
            return None
        if head.morph.get("ExtPos"):
            return head
        if deprel_base(head) != UNK:  # chain ended without reaching an ExtPos head
            return None
        cur = head
    return None


class SudIdiom:
    """Stamp `Idiom=Yes` / `InIdiom=Yes` into the MISC slot from the predicted parse.

    Only these two keys are touched; anything else already in `token._.sud_misc` is left alone,
    so this composes with the trained `sud_tagger` pipes regardless of pipeline order.
    """

    def __init__(self, nlp, name):
        pass

    def __call__(self, doc):
        # An idiom head is an ExtPos token with at least one `unk` child. Collect the heads of all
        # `unk` tokens first so the membership test is O(1) rather than a children scan per token.
        unk_heads = {t.head.i for t in doc if deprel_base(t) == UNK and t.head.i != t.i}
        for tok in doc:
            set_misc(tok, "Idiom",
                     "Yes" if (tok.morph.get("ExtPos") and tok.i in unk_heads) else None)
            set_misc(tok, "InIdiom",
                     "Yes" if (deprel_base(tok) == UNK and _idiom_root(tok) is not None) else None)
        return doc


def make_sud_idiom(nlp, name):
    return SudIdiom(nlp, name)


# Guarded, like clause_parser's: loading two models in one process -- or a wheel that imports this
# module alongside a `--code` load of the same file -- registers the factory twice.
if not Language.has_factory("sud_idiom"):
    Language.factory("sud_idiom")(make_sud_idiom)
