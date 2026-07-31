"""Registers the `sud_reported_rule` factory: `Reported=Yes` from the direct-speech diagnostics.

WHY A RULE AND NOT THE TRAINED PIPE. `sud_tagger` was trained on the bootstrapped gold for all five
languages and scored F 0.12-0.40, recall-limited throughout (dev): sa 0.402, ar 0.363, en 0.357,
la 0.182, fa 0.121. The cause is architectural, not a tuning problem. The pipe carries its own small
`HashEmbedCNN` over NORM/PREFIX/SUFFIX/SHAPE at depth 3 -- a few tokens of context -- but every piece
of evidence for reported speech is non-local:

  * the governing verb must be a speech verb, and it can sit far from the clause head;
  * quotation marks sit at the clause EDGES, routinely outside the window;
  * Latin's diagnostic is the complement's own VerbForm/Mood plus the ABSENCE of a subordinator.

Contrast `Subject`, where the raising complement is adjacent to its control verb and the same
architecture reaches F 0.72-0.92. So this component reads the parse instead, applying at inference
exactly the tests `sud_reported_gold.py` used to build the gold.

READ THE SCORES WITH CARE. There is no independent gold for `Reported` -- the treebanks annotate it
nowhere, so the "gold" is itself these rules plus an LLM pass over the residue. This component
therefore scores well against that gold substantially BY CONSTRUCTION, and its agreement figure is
not evidence that the annotation is correct, only that it is reproducible. What it does honestly
recover is the rule-committed portion: la 285/314, ar 997/1047, sa 1321/1814, en 204/456, fa 110/836.
The LLM-decided remainder is out of reach of any rule, by definition.

Load with `spacy ... --code scripts/sud_misc.py,scripts/sud_reported_data.py,scripts/sud_reported_rule.py`.
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
data = _sibling("sud_reported_data")
set_misc = sud_misc.set_misc
deprel_base = sud_misc.deprel_base


class SudReportedRule:
    """Stamp `Reported=Yes` on the head of a directly-reported clause."""

    def __init__(self, nlp, name, lang):
        self.lang = lang
        self.speech = data.SPEECH_VERBS.get(lang, set())
        self.marks = data.COMPLEMENTISERS.get(lang, set())

    def _is_complementiser(self, tok):
        return tok.lemma_ in self.marks and tok.pos_ in ("SCONJ", "ADP", "PART")

    def _la_finite_direct(self, comp, subtree):
        """Latin: a finite complement with no subordinator is direct speech (see CLAUDE.md).

        Indirect statement is accusative-and-infinitive; every finite indirect clause carries an
        overt subordinator, which under the functional-head analysis IS the complement token. The
        one exception, the indirect question, requires the subjunctive -- so an indicative clause
        containing `qui` has a relative pronoun, and a subjunctive with no interrogative is a
        jussive inside a quote (`fiat lux`).
        """
        if "Fin" not in comp.morph.get("VerbForm"):
            return False
        if "Sub" not in comp.morph.get("Mood"):
            return True
        return not any(t.lemma_ in data.LA_INTERROGATIVE and t.pos_ in ("PRON", "DET", "ADV")
                       for t in subtree)

    def __call__(self, doc):
        for tok in doc:
            set_misc(tok, "Reported", None)
        for tok in doc:
            base = deprel_base(tok)
            if not (base.startswith("comp:") or base == "parataxis"):
                continue
            head = tok.head
            if head.i == tok.i or head.pos_ not in ("VERB", "AUX"):
                continue
            if head.lemma_ not in self.speech:
                continue

            subtree = list(tok.subtree)
            # Reported speech is a CLAUSE: a speech verb's ordinary nominal and prepositional
            # objects (`dicit hoc`, `loquor de X`) are not reported speech. Only a direct-evidence
            # hit can override this, since a verbatim quote may itself be verbless (`he said "yes"`).
            clausal = tok.pos_ in ("VERB", "AUX", "SCONJ") or bool(tok.morph.get("VerbForm"))

            direct = any(any(c in data.QUOTES for c in t.text) for t in subtree)
            # Only verbatim speech can host the speaker's own discourse markers: an indirect clause
            # is recast from the narrator's viewpoint and cannot carry them.
            direct = direct or any(deprel_base(t) == "discourse" for t in subtree)
            if self.lang == "sa":
                direct = direct or any(t.lemma_ in data.SA_QUOTATIVE for t in subtree)

            indirect = self._is_complementiser(tok)
            if self.lang == "la":
                indirect = indirect or "Inf" in tok.morph.get("VerbForm")
                if not indirect and clausal and self._la_finite_direct(tok, subtree):
                    direct = True

            if direct and not indirect and (clausal or direct):
                set_misc(tok, "Reported", "Yes")
        return doc


def make_sud_reported_rule(nlp, name, lang):
    return SudReportedRule(nlp, name, lang)


if not Language.has_factory("sud_reported_rule"):
    Language.factory("sud_reported_rule", default_config={"lang": "en"})(make_sud_reported_rule)
