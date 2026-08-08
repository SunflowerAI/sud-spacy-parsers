"""Registers the `sud_tagger` factory: a per-token classifier for one SUD MISC feature.

Why a custom pipe at all. spaCy ships no generic token classifier:

  * `Tagger` hardcodes its output (`doc.c[j].tag`) and its gold source (`eg.get_aligned("TAG")`),
    and always reports `tag_acc` -- a second instance is a functional clone, not an independent
    layer.
  * `Morphologizer` writes the WHOLE morph string, so a second one placed downstream would wipe
    out the first one's FEATS.
  * `spancat` has a configurable output slot but is a span model with a suggester, which is the
    wrong shape for a dense per-token flag.
  * `Token._.` extensions cannot be reached by `Example.get_aligned` at all (`E983`: only the
    registered attr IDs work), so the gold has to be read another way regardless.

What survives is: subclass `Tagger`, keep its model unchanged (`spacy.Tagger.v2` is just
`chain(tok2vec, with_array(Softmax_v2))`), and override the four places that are hardcoded --
the output slot, the gold source in `get_loss` and `initialize`, and the scorer.

One class serves every SUD MISC feature; `feat` selects which:

    [components.sud_subject]
    factory = "sud_tagger"
    feat = "Subject"
    labels = ["O", "SubjRaising", "ObjRaising"]

Separate instances rather than one joint softmax, because a token can in principle be both a
raising complement and a reported-speech complement, which a single label set cannot express.

TWO THINGS THAT ARE EASY TO GET WRONG:

1. `O` is a real class, not missing annotation. `Tagger.get_loss` maps the empty label `""` to
   None, which the loss treats as *missing* and produces no gradient from. For this task "no
   feature here" is the majority class and must be learned, so the negative is an explicit `O`
   and `""` is never used as a label.

2. Gold arrives through FEATS, not MISC. `spacy convert` discards MISC, so `hoist_sud_gold.py`
   copies the gold into the FEATS column under a `Sud` prefix (`SudSubject=SubjRaising`). This
   pipe reads only `Sud`-prefixed keys -- it therefore cannot train on a genuine morphological
   feature by accident -- and at inference writes to the slot (`Token._.sud_misc`), never to
   `token.morph`.

`clear_morph` is the one exception to that last point, and it exists for `Shared`. Unlike the
other SUD keys, `Shared` is a FEATS feature in the treebanks, so the morphologiser has been
learning it all along as part of its FEATS bundles -- badly (en test P 0.68 / R 0.15; `Shared=Yes`
correct 4 times in 247), because a small local encoder over word forms cannot see the coordination
the feature is about. With `clear_morph = true` this pipe DELETES its own feature from
`token.morph` before writing the slot, so the arm has exactly one answer for it instead of two
contradictory ones. Set it only where this pipe genuinely takes a FEATS feature over.

Load with `spacy ... --code scripts/seg_code.py` (which imports this module).
"""
from itertools import islice

from spacy import util
from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline.tagger import Tagger
from spacy.scorer import PRFScore
from spacy.tokens import Doc
from spacy.training import validate_examples, validate_get_examples
from spacy.util import registry
from thinc.api import Config


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
HOIST_PREFIX = sud_misc.HOIST_PREFIX
set_misc = sud_misc.set_misc
get_misc = sud_misc.get_misc

# The explicit negative class. See note 1 in the module docstring: it must not be "".
NEG = "O"


# --------------------------------------------------------------------------------------------
# Candidate masks. A mask says WHERE in the doc the feature's question is even asked; outside it
# the answer is `O` by construction, and the pipe neither predicts nor takes gradient.
#
# This is the same move `sud_reported_gold` makes with its `clausal` flag -- a speech verb also
# takes ordinary nominal objects, and letting those reach the model wastes it on cases that have
# no answer. For `Shared` the effect is larger, because the mask is not a filter on a minority: it
# cuts English train from 204 578 tokens to 15 499, of which 63 % carry the feature. Without it
# the pipe is choosing `O` 95 times in 100 and learns to say `O`.
#
# A mask is named in the config rather than passed as a function so it serialises with the model.
# --------------------------------------------------------------------------------------------
def _coordination_mask(doc):
    """Dependents of a conjunct that lie outside the coordination -- see sud_shared_data."""
    data = _sibling("sud_shared_data")
    return {i for i, _position in data.doc_candidates(doc)}


MASKS = {"coordination": _coordination_mask}

# Same encoder as every other added layer in this project (morphologiser, lemmatiser): a small
# DEDICATED HashEmbedCNN rather than a listener on the frozen shared tok2vec, so the frozen
# components stay byte-identical and the layer is self-contained.
default_model_config = """
[model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 64
depth = 3
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
"""
DEFAULT_SUD_MODEL = Config().from_str(default_model_config)["model"]


def make_sud_scorer(feat):
    """P/R/F over tokens that CARRY the feature -- the negative class is excluded.

    Plain accuracy is useless here: `Subject` sits on ~1% of tokens, so predicting `O` everywhere
    would score ~99%. This mirrors the project's existing practice of reading per-label `comp:obl`
    F rather than headline LAS.
    """
    key = feat.lower()

    def sud_score(examples, **kwargs):
        prf = PRFScore()
        for eg in examples:
            align = eg.alignment.x2y
            gold = set()
            for gold_i, token in enumerate(eg.reference):
                value = get_misc(token, feat) or _hoisted(token, feat)
                if value:
                    gold.add((gold_i, value))
            pred = set()
            for token in eg.predicted:
                if token.orth_.isspace() or align.lengths[token.i] != 1:
                    continue
                value = get_misc(token, feat)
                if value:
                    pred.add((align[token.i][0], value))
            prf.score_set(pred, gold)
        return {
            f"sud_{key}_p": prf.precision,
            f"sud_{key}_r": prf.recall,
            f"sud_{key}_f": prf.fscore,
        }

    return sud_score


def _hoisted(token, feat):
    """Gold value for `feat`, read from the hoisted `Sud`-prefixed FEATS key (see the docstring)."""
    values = token.morph.get(HOIST_PREFIX + feat)
    return values[0] if values else ""


def make_sud_tagger(nlp, name, model, feat, overwrite, clear_morph, mask):
    return SudTagger(nlp.vocab, model, name, feat=feat, overwrite=overwrite,
                     clear_morph=clear_morph, mask=mask)


# Guarded, like clause_parser's: loading two models in one process -- or a wheel that imports this
# module alongside a `--code` load of the same file -- registers the factory twice.
if not Language.has_factory("sud_tagger"):
    Language.factory(
        "sud_tagger",
        default_config={
            "feat": "Subject",
            "model": DEFAULT_SUD_MODEL,
            "overwrite": True,
            "clear_morph": False,
            "mask": "",
        },
        default_score_weights={},
    )(make_sud_tagger)


class SudTagger(Tagger):
    """Predict one SUD MISC feature per token and write it to `Token._.sud_misc`."""

    def __init__(self, vocab, model, name="sud_tagger", *, feat="Subject", overwrite=True,
                 clear_morph=False, mask=""):
        super().__init__(vocab, model, name, overwrite=overwrite,
                         scorer=make_sud_scorer(feat))
        # `feat` lives in cfg so it is serialised with the component and survives save/load --
        # the annotation would silently go to the wrong key otherwise. Same for `clear_morph` and
        # `mask`: a reloaded model that forgot either would quietly change what it emits, and
        # nothing would raise.
        self.cfg["feat"] = feat
        self.cfg["clear_morph"] = clear_morph
        self.cfg["mask"] = mask
        if mask and mask not in MASKS:
            raise ValueError(f"unknown mask {mask!r}; known: {sorted(MASKS)}")

    @property
    def feat(self):
        return self.cfg["feat"]

    def _mask(self, doc):
        """Indices where this pipe may answer, or None when it may answer anywhere."""
        name = self.cfg.get("mask") or ""
        return MASKS[name](doc) if name else None

    def _clear_morph(self, token):
        """Delete this pipe's feature from `token.morph` (see `clear_morph` in the docstring).

        Only touches a token that actually carries the key, and unsets MORPH entirely rather than
        stamping an empty one when nothing is left: `set_morph({})` yields morph key 456 and an
        untouched token key 0, both of which render as `''`, so the difference is invisible to
        every string-level check but not to an encoder that reads MORPH. That distinction has
        already cost this project 6.8 LAS once (see CLAUDE.md).
        """
        if not token.morph.get(self.feat):
            return
        rest = {k: v for k, v in token.morph.to_dict().items() if k != self.feat}
        token.set_morph(rest or None)

    def _gold_labels(self, eg):
        """Per predicted token: the gold label, or None where alignment is not 1:1.

        None means *missing* to the loss (no gradient), which is the right treatment for a token
        whose gold counterpart is ambiguous. In practice every arm here trains under either
        `gold_preproc = true` or `sud.GoldTokCorpus.v1`, both of which make the predicted doc
        token-for-token identical to the reference, so the fallback almost never fires.
        """
        align = eg.alignment.x2y
        reference = eg.reference
        mask = self._mask(eg.predicted)
        out = []
        for token in eg.predicted:
            if align.lengths[token.i] != 1 or (mask is not None and token.i not in mask):
                # Outside the mask is `None`, i.e. MISSING, not `O`: the answer there is fixed by
                # construction, so training on it would spend the model's capacity learning to
                # reproduce a rule it is already being given. The cost is a recall ceiling -- the
                # mask misses 7.1 % of English gold `Shared` -- which is why it is worth measuring
                # against the unmasked arm rather than assuming.
                out.append(None)
                continue
            gold_token = reference[align[token.i][0]]
            out.append(_hoisted(gold_token, self.feat) or NEG)
        return out

    def set_annotations(self, docs, batch_tag_ids):
        if isinstance(docs, Doc):
            docs = [docs]
        labels = self.labels
        clear = self.cfg.get("clear_morph", False)
        for i, doc in enumerate(docs):
            doc_tag_ids = batch_tag_ids[i]
            if hasattr(doc_tag_ids, "get"):
                doc_tag_ids = doc_tag_ids.get()
            mask = self._mask(doc)
            for j, tag_id in enumerate(doc_tag_ids):
                label = labels[tag_id]
                if mask is not None and j not in mask:
                    label = NEG        # outside the mask the answer is fixed, not predicted
                if clear:
                    self._clear_morph(doc[j])
                set_misc(doc[j], self.feat, None if label == NEG else label)

    def get_loss(self, examples, scores):
        validate_examples(examples, "SudTagger.get_loss")
        from thinc.api import SequenceCategoricalCrossentropy

        loss_func = SequenceCategoricalCrossentropy(
            names=self.labels, normalize=False, neg_prefix=self.cfg["neg_prefix"],
            label_smoothing=self.cfg["label_smoothing"],
        )
        truths = [self._gold_labels(eg) for eg in examples]
        d_scores, loss = loss_func(scores, truths)
        if self.model.ops.xp.isnan(loss):
            raise ValueError(Errors.E910.format(name=self.name))
        return float(loss), d_scores

    def initialize(self, get_examples, *, nlp=None, labels=None):
        validate_get_examples(get_examples, "SudTagger.initialize")
        util.check_lexeme_norms(self.vocab, "sud_tagger")
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            # NEG first so the negative class always exists even in a shard with no positives.
            found = {NEG}
            for example in get_examples():
                for token in example.y:
                    value = _hoisted(token, self.feat)
                    if value:
                        found.add(value)
            self.add_label(NEG)
            for label in sorted(found - {NEG}):
                self.add_label(label)

        doc_sample = []
        label_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
            gold = self._gold_labels(example)
            label_sample.append(self.model.ops.asarray(
                [[1.0 if label == g else 0.0 for label in self.labels] for g in gold],
                dtype="float32",
            ))
        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=label_sample)


# --------------------------------------------------------------------------------------------
# Tree-aware encoder: [ own | head | mean of immediate DEPENDENTS ].
#
# The convolutional encoders above mix over LINEAR neighbours, but the evidence for reported speech
# is tree evidence: is my HEAD a speech verb, and what does my clause CONTAIN. Widening the window
# only approximates that by proximity -- which is why `--structural` helped (ar test F 37.4 -> 46.7)
# without closing the gap to the rule (73.5).
#
# NEGATIVE RESULT, kept because it explains this design: pooling the whole SUBTREE instead of the
# immediate dependents does not work. Dev F peaked at 9.5 by step 200 and decayed to ~4 (against 42
# by step 400 for the plain structural encoder) while the loss fell throughout -- the encoder was
# being wrecked, not converging. The reason is gradient accumulation: a token belongs to the subtree
# of every one of its ancestors, so its vector collects O(depth) pooled gradients on top of its own,
# and tokens near the root swamp everything else. Stop-gradient on the pooled branches did not
# rescue it either (dev F 0.08), so the problem is not only scale.
#
# Immediate dependents avoid that by construction: every token is a child of exactly ONE parent, so
# it receives exactly one pooled gradient. It is also the more faithful feature -- the rule asks
# what hangs directly off the clause head (a quotation mark, a `discourse` marker), not what sits
# anywhere beneath it.
#
# Requires the parse, so the config must list the frozen parser in `annotating_components`
# (make_sud_config.py --structural does).
# --------------------------------------------------------------------------------------------
from thinc.api import Model, Softmax_v2, chain, with_array  # noqa: E402


# Which dependents to pool. See build_head_deps_tagger for the rationale of each.
CLOSED_CLASS = frozenset(("PUNCT", "PART", "ADP", "SCONJ", "CCONJ", "DET",
                          "AUX", "PRON", "INTJ", "ADV"))


def _parsed(doc):
    """Whether the tree slices can be read at all.

    `Token.children` walks the parser's own left/right kid pointers, and on a doc that carries NO
    parse those are uninitialised -- reading them SEGFAULTS, with no Python-level error to catch.
    That is not a hypothetical: `initialize` samples `example.x`, which the corpus readers build
    from gold words alone, and `annotating_components` only runs during the training loop, so the
    very first thing this layer ever sees is an unparsed doc. Guard, do not assume.
    """
    return doc.has_annotation("DEP")


def _pool_indices(doc, mode, xp):
    """Per token, the indices whose vectors get averaged into the third slice."""
    if mode == "none" or not _parsed(doc):   # "none" is the diagnostic: no tree information
        return [xp.zeros((0,), dtype="i") for _ in doc]
    out = []
    for t in doc:
        if mode == "deps":
            idx = [c.i for c in t.children]
        elif mode == "closed":
            # Only closed-class dependents. Quotation marks (PUNCT) and discourse markers
            # (INTJ/PART/ADV -- en `well`/`no`, la `autem`/`enim`, sa `vai`/`eva`) survive;
            # the open-class dependents that make up the clause's content drop out.
            idx = [c.i for c in t.children if c.pos_ in CLOSED_CLASS]
        elif mode == "closed2":
            # Closed-class at TWO levels: reaches past a complementiser head (SUD makes the
            # subordinator the complement token, so the clause's quotes and discourse markers
            # hang off ITS child) while keeping the open-class content out of the average.
            idx = [c.i for c in t.children if c.pos_ in CLOSED_CLASS]
            idx += [g.i for c in t.children for g in c.children if g.pos_ in CLOSED_CLASS]
        elif mode == "deps2":
            # Two levels. Needed because SUD is functional-head: where there IS an overt
            # complementiser it is the complement token itself, so the clause's verb -- and the
            # quotes and discourse markers hanging off it -- sit a level BELOW the token.
            idx = [c.i for c in t.children]
            idx += [g.i for c in t.children for g in c.children]
        else:
            raise ValueError(f"unknown pool mode {mode!r}")
        out.append(xp.asarray(idx, dtype="i"))
    return out


def _head_deps_forward(model, docs, is_train):
    tok2vec = model.get_ref("tok2vec")
    Xs, bp_tok2vec = tok2vec(docs, is_train)
    ops = model.ops
    xp = ops.xp
    outs, metas = [], []
    for doc, X in zip(docs, Xs):
        n, w = X.shape
        if n == 0:
            outs.append(ops.alloc2f(0, w * 3))
            metas.append((None, None, 0, w))
            continue
        mode = model.attrs["pool"]
        heads = (xp.arange(n, dtype="i") if mode == "none" or not _parsed(doc)
                 else xp.asarray([t.head.i for t in doc], dtype="i"))
        kids = _pool_indices(doc, mode, xp)
        D = ops.alloc2f(n, w)          # leaves keep a zero vector: "nothing hangs off me"
        for i, idx in enumerate(kids):
            if len(idx):
                D[i] = X[idx].mean(axis=0)
        outs.append(xp.hstack([X, X[heads], D]))
        metas.append((heads, kids, n, w))

    detach = model.attrs["detach"]

    def backprop(dOuts):
        dXs = []
        for dY, (heads, kids, n, w) in zip(dOuts, metas):
            dX = ops.alloc2f(n, w)
            if n == 0:
                dXs.append(dX)
                continue
            dX += dY[:, :w]
            if not detach:
                # d/dX of X[heads]: scatter back onto each head's own row.
                # `ops.scatter_add`, not `xp.add.at`: cupy < 13 -- what spaCy 3.8 pins for
                # cuda12x -- has no `ufunc.at`, so the raw form trains fine on CPU and dies
                # in BACKPROP on GPU, i.e. minutes into the longest run of the chain.
                # NB call it BARE. NumpyOps (Cython) and CupyOps (cupyx.scatter_add) both
                # mutate `table` and return None; only the base Ops.scatter_add returns
                # anything, and that one is just `xp.add.at` again. `dX = ops.scatter_add(...)`
                # would therefore set dX to None on every backend that actually works.
                ops.scatter_add(dX, heads, dY[:, w:2 * w])
                # d/dX of the dependent mean: split evenly over the children it averaged
                for i, idx in enumerate(kids):
                    if len(idx):
                        ops.scatter_add(dX, idx, dY[i, 2 * w:] / len(idx))
            dXs.append(dX)
        return bp_tok2vec(dXs)

    return outs, backprop


def _head_deps_init(model, X=None, Y=None):
    tok2vec = model.get_ref("tok2vec")
    tok2vec.initialize(X=X)
    if tok2vec.has_dim("nO"):
        model.set_dim("nO", tok2vec.get_dim("nO") * 3)
    return model


def HeadDeps(tok2vec, pool="deps", detach=False):
    """Concatenate each token's vector with its head's and a pooled dependent mean (3x width)."""
    return Model(
        "sud_head_deps",
        _head_deps_forward,
        init=_head_deps_init,
        layers=[tok2vec],
        refs={"tok2vec": tok2vec},
        attrs={"pool": pool, "detach": detach},
        dims={"nO": (tok2vec.get_dim("nO") * 3) if tok2vec.has_dim("nO") else None},
    )


@registry.architectures("sud.HeadDepsTagger.v1")
def build_head_deps_tagger(tok2vec, nO=None, normalize=False, pool="deps", detach=False):
    """A Tagger head over [own | head | pooled dependents], for tree-structured features.

    `pool` selects what the third slice averages:
      none    nothing -- head is the token itself and the pool is empty, so the layer carries NO
              tree information and must reproduce the plain structural encoder. A DIAGNOSTIC: if
              this does not match, the fault is in this wrapper, not in the choice of pooling.
      deps    all immediate dependents
      closed  closed-class immediate dependents only (quotes, discourse markers)
      deps2   dependents and their dependents (reaches past a complementiser head)
      closed2 closed-class dependents at two levels -- both restrictions at once
    """
    t2v = HeadDeps(tok2vec, pool=pool, detach=detach)
    width = tok2vec.get_dim("nO") * 3 if tok2vec.has_dim("nO") else None
    output_layer = Softmax_v2(nO, width, normalize_outputs=normalize)
    model = chain(t2v, with_array(output_layer))
    model.set_ref("tok2vec", t2v)
    model.set_ref("output_layer", output_layer)
    model.attrs["multi_label"] = False
    return model
