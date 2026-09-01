#!/usr/bin/env python3
"""`multifield_tagger` — one softmax per comma-separated XPOS FIELD, not one over whole tags.

WHY. Kyoto's XPOS is a four-field code — `v,動詞,行為,伝達` — with field cardinalities 4 / 12 / 46
/ 84. The shipped tagger treats the whole thing as **121 atomic labels**, so `v,動詞,行為,伝達` and
`v,動詞,行為,交流` are unrelated symbols sharing no parameters, and a tag absent from training is
unreachable however obvious its parts. Per-field heads share the encoder and predict each field
independently, so evidence for `動詞` accrues across all 30-odd tags containing it.

⚠ **AND THE GRID IS ALMOST ENTIRELY EMPTY: 121 of 4x12x46x84 = 185 472 combinations are attested,
0.07 %.** Independent heads can therefore emit codes the tagset does not contain. `project` is on
by default and fixes the joint prediction onto the attested inventory (nearest attested tag by
summed field log-probability); `--no-project` leaves it free so the off-grid rate can be MEASURED
rather than assumed. Read that rate before trusting any per-field accuracy: a head can be right
about its own field and still contribute to a tag that does not exist.

⚠ **`upos_mask` IS A RUNTIME MODE, AND IT DEFAULTS TO OFF.** It is worth **+3.11 TAG on GOLD UPOS
and -0.33 on PREDICTED UPOS** (92.72 -> 95.83 / 92.39), because masking makes XPOS inherit whatever
the UPOS is — including the morphologiser's own 6.87 % error rate. So it is a large gain exactly in
the editing workflow it was built for, where a human has corrected the UPOS, and a small loss on
fully automatic output where nobody has. Ship it OFF; turn it on per call:

    nlp.get_pipe("multifield_tagger").cfg["upos_mask"] = True

⚠ **UPOS IS A CONSTRAINT, NOT JUST A FEATURE, AND THAT IS WHAT `upos_mask` BUYS.** The released
tagger already reads UPOS through `sud.MultiHashEmbedFeats.v1`, and it is measurably live —
forcing a token's UPOS moves the XPOS logits by up to 3.66. But it never flips the argmax, because
the tagger was trained where context predicts UPOS and XPOS jointly, so a hand-edited UPOS is
counterfactual and the lexical evidence overwhelms it. For an editing workflow — a linguist fixes a
UPOS and expects XPOS to follow — the feature is not enough and a MASK is exact: each UPOS admits a
mean of 11.1 of the 121 tags, and SCONJ, CCONJ and INTJ admit exactly one. The mask is built from
the training treebank by `build_lzh_xpos_tables.py` and travels inside the component.

⚠ Tokens whose UPOS was never seen with any tag are left UNMASKED rather than forced to an empty
candidate set — a mask that can select nothing is a component that silently emits garbage.
"""
import json
import pathlib
import warnings
from itertools import islice

import numpy as np
from spacy import util
from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.training import Example, validate_examples
from spacy.util import registry
from thinc.api import (Config, Model, SequenceCategoricalCrossentropy, Softmax_v2, chain,
                       concatenate, with_array)


@registry.architectures("sud.MultiFieldTagger.v2")
def CombinedTaggerModel(tok2vec: Model, field_sizes, n_joint: int) -> Model:
    """The four per-field softmaxes AND a joint softmax over the attested codes, side by side.

    WHY BOTH. Splitting the tagger into independent field heads costs **0.59 TAG** (92.72 -> 92.13)
    and the loss is structural, not under-training: the fields are strongly dependent (`n` never
    co-occurs with 動詞) and one 121-way softmax gets that for free. But the field heads are what
    expose per-field confidences, and the UPOS MASK — which is what actually makes a hand-edited
    UPOS propagate — needs a ranking over whole codes to restrict.

    So keep the joint head as the primary ranking and let the field heads refine it. `field_weight`
    at decode time sets the blend; at 0.0 this is exactly the shipped tagger's architecture plus a
    mask, which is the configuration that gives the editing behaviour at NO accuracy cost.

    Output columns: [field_1 … field_4 | joint]. The joint block is last so the field offsets are
    unchanged from v1 and `get_loss` can slice both without special-casing."""
    heads = [Softmax_v2(nO=int(n)) for n in field_sizes] + [Softmax_v2(nO=int(n_joint))]
    return chain(tok2vec, with_array(concatenate(*heads)))


@registry.architectures("sud.MultiFieldTagger.v1")
def MultiFieldTaggerModel(tok2vec: Model, field_sizes) -> Model:
    """One INDEPENDENT softmax per XPOS field, concatenated.

    ⚠ `spacy.Tagger.v2` CANNOT BE USED HERE, and the failure is silent-ish rather than an error:
    it puts a SINGLE `Softmax_v2` over all 146 outputs, so the four fields share one probability
    mass that sums to 1. Per-field cross-entropy on slices of that is self-contradictory — raising
    field 1's target necessarily lowers field 2's — and the run DIVERGES: loss 363 -> 112 692 and
    TAG_ACC 62.33 -> 34.85 over 800 steps, with no exception raised anywhere.

    `field_sizes` is passed in the config rather than discovered at initialize because `concatenate`
    needs its members sized at construction. It is checked against the data in
    `MultiFieldTagger.initialize`, so a stale config fails loudly instead of silently mis-slicing."""
    heads = [Softmax_v2(nO=int(n)) for n in field_sizes]
    return chain(tok2vec, with_array(concatenate(*heads)))

DEFAULT_MODEL = Config().from_str("""
[model]
@architectures = "spacy.Tagger.v2"
nO = null
normalize = false

[model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 96
upstream = "tok2vec"
""")["model"]


@Language.factory("multifield_tagger",
                  default_config={"model": DEFAULT_MODEL, "sep": ",", "n_fields": 4,
                                  "tables": None, "project": True, "upos_mask": False,
                                  "overwrite": False, "joint": True, "field_weight": 0.0},
                  default_score_weights={"tag_acc": 1.0, "tag_field_acc": None})
def make_multifield_tagger(nlp, name, model, sep, n_fields, tables, project, upos_mask, overwrite,
                           joint, field_weight):
    return MultiFieldTagger(nlp.vocab, model, name, sep=sep, n_fields=n_fields, tables=tables,
                            project=project, upos_mask=upos_mask, overwrite=overwrite,
                            joint=joint, field_weight=field_weight)


class MultiFieldTagger(TrainablePipe):
    def __init__(self, vocab, model, name="multifield_tagger", *, sep=",", n_fields=4,
                 tables=None, project=True, upos_mask=False, overwrite=False,
                 joint=True, field_weight=0.0):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"sep": sep, "n_fields": n_fields, "project": project, "field_sizes": None, "joint": joint,
                    "field_weight": field_weight,
                    "upos_mask": upos_mask, "overwrite": overwrite,
                    "fields": [[] for _ in range(n_fields)]}
        self.attested = []          # every XPOS tag the training data contains
        self.allowed = {}           # UPOS -> the tags it was seen with
        # ⚠ NEVER LOAD A CONFIG PATH THAT MAY NOT EXIST AT LOAD TIME. `tables` is a BUILD-time
        # convenience: it seeds the tables when the component is first created for training. In a
        # PACKAGED model the authoritative copy is `tables.json` inside the component directory,
        # which `from_disk` reads and REFUSES to run without — so deferring here cannot leave the
        # tables silently empty. Loading eagerly shipped a wheel whose config carried
        # `models/lzh_xpos_tables.json`, a path relative to the CWD: it resolved on the build
        # machine and raised FileNotFoundError everywhere else, Pyodide's filesystem root included.
        if tables and pathlib.Path(tables).is_file():
            self.load_tables(tables)
        elif tables:
            warnings.warn(f"{name}: `tables` path {tables!r} does not exist; deferring to the "
                          f"copy bundled with the model. If this is a TRAINING run, the path is "
                          f"wrong and initialize will fail.", RuntimeWarning)

    # ---- labels ----------------------------------------------------------------------------
    @property
    def sep(self):
        return self.cfg["sep"]

    @property
    def fields(self):
        """The per-field label lists, in field order."""
        return [list(f) for f in self.cfg["fields"]]

    @property
    def labels(self):
        """Flat, for spaCy's label bookkeeping: `field index + separator + value`."""
        return tuple(f"{i}{self.sep}{v}" for i, f in enumerate(self.cfg["fields"]) for v in f)

    @property
    def offsets(self):
        """(per-field start offsets, total width). The JOINT block, when present, occupies the
        columns after the fields — so field offsets are identical with and without it."""
        out, n = [], 0
        for f in self.cfg["fields"]:
            out.append(n)
            n += len(f)
        if self.cfg.get("joint"):
            n += len(self.attested)
        return out, n

    @property
    def joint_offset(self):
        return sum(len(f) for f in self.cfg["fields"]) if self.cfg.get("joint") else None

    def add_field_value(self, i, value):
        if value not in self.cfg["fields"][i]:
            self.cfg["fields"][i].append(value)
            return 1
        return 0

    def load_tables(self, source):
        d = json.loads(pathlib.Path(source).read_text(encoding="utf-8")) \
            if isinstance(source, (str, pathlib.Path)) else dict(source)
        self.attested = list(d.get("attested", []))
        self.allowed = {k: list(v) for k, v in d.get("allowed", {}).items()}
        return self

    # ---- inference -------------------------------------------------------------------------
    def predict(self, docs):
        if not any(len(d) for d in docs):
            _, width = self.offsets
            return [self.model.ops.alloc2f(0, width) for _ in docs]
        return self.model.predict(docs)

    def _decode(self, scores, doc):
        """Per-field argmax, then the two corrections, in this order: mask by UPOS (a hard
        constraint the user may have just edited), then project onto the attested inventory."""
        offs, _ = self.offsets
        fields = self.cfg["fields"]
        # log-probabilities per field, so field scores are commensurable when projecting
        # each block is ALREADY a normalised distribution (its own Softmax_v2 head), so take the
        # log directly — re-softmaxing here would flatten the very confidences the projection ranks by
        logp = [np.log(scores[:, offs[i]:offs[i] + len(f)] + 1e-12) for i, f in enumerate(fields)]
        jlogp = None
        if self.cfg.get("joint") and self.attested:
            jo = self.joint_offset
            jlogp = np.log(scores[:, jo:jo + len(self.attested)] + 1e-12)
            jindex = {t: i for i, t in enumerate(self.attested)}
        fw = float(self.cfg.get("field_weight", 0.0))
        out = []
        for k, tok in enumerate(doc):
            free = self.sep.join(fields[i][int(logp[i][k].argmax())] for i in range(len(fields)))
            cand = self.attested if (self.cfg["project"] and self.attested) else None
            if self.cfg["upos_mask"] and self.allowed:
                allowed = self.allowed.get(tok.pos_)
                # ⚠ an unknown UPOS leaves the token UNMASKED; a mask that admits nothing would
                # make the component silently emit whatever `argmax` of an empty slice returns.
                if allowed:
                    cand = [t for t in (cand or self.attested or [free]) if t in allowed] or None
            if cand is None:
                out.append(free)
                continue
            best, best_s = free, -1e18
            for tag in cand:
                parts = tag.split(self.sep)
                if len(parts) != len(fields):
                    continue
                # The JOINT head is the primary ranking; the field heads refine it by `field_weight`
                # (0.0 = joint only, which is the shipped tagger's ranking plus the UPOS mask).
                s = 0.0
                if jlogp is not None and tag in jindex:
                    s += float(jlogp[k][jindex[tag]])
                if fw or jlogp is None:
                    w = fw if jlogp is not None else 1.0
                    for i, p in enumerate(parts):
                        j = fields[i].index(p) if p in fields[i] else None
                        if j is None:
                            s = -1e18
                            break
                        s += w * float(logp[i][k][j])
                if s > best_s:
                    best, best_s = tag, s
            out.append(best)
        return out

    def set_annotations(self, docs, batch_scores):
        for doc, scores in zip(docs, batch_scores):
            tags = self._decode(np.asarray(scores), doc)
            for tok, tag in zip(doc, tags):
                if self.cfg["overwrite"] or not tok.tag_:
                    tok.tag_ = tag

    # ---- training --------------------------------------------------------------------------
    def get_loss(self, examples, scores):
        validate_examples(examples, "MultiFieldTagger.get_loss")
        offs, width = self.offsets
        # ⚠ ONE LOSS OBJECT PER FIELD, each carrying ITS OWN label names. A single shared
        # `SequenceCategoricalCrossentropy` cannot map a string truth to a column without them —
        # thinc raises "Cannot calculate loss from list of strings without names" — and a loss
        # built with the WRONG field's names would map silently to the wrong column instead.
        total = 0.0
        per_field_truth = [[] for _ in self.cfg["fields"]]
        for eg in examples:
            gold = [g if g else "" for g in eg.get_aligned("TAG", as_string=True)]
            for i in range(len(self.cfg["fields"])):
                per_field_truth[i].append(
                    [(g.split(self.sep)[i] if g and len(g.split(self.sep)) > i else "")
                     for g in gold])
        # one softmax per field, over its own column block; gradients are written back in place
        d_scores = [self.model.ops.alloc2f(s.shape[0], width) for s in scores]
        if self.cfg.get("joint") and self.attested:
            jo, nj = self.joint_offset, len(self.attested)
            jt = [[g if g in set(self.attested) else "" for g in
                   [(x if x else "") for x in eg.get_aligned("TAG", as_string=True)]]
                  for eg in examples]
            jl = SequenceCategoricalCrossentropy(names=list(self.attested), normalize=False,
                                                 missing_value="")
            jblock = [np.asarray(s[:, jo:jo + nj]) for s in scores]
            dj, lj = jl(jblock, jt)
            total += float(lj)
            for k in range(len(scores)):
                d_scores[k][:, jo:jo + nj] = dj[k]
        for i, f in enumerate(self.cfg["fields"]):
            loss_func = SequenceCategoricalCrossentropy(names=list(f), normalize=False,
                                                        missing_value="")
            block = [np.asarray(s[:, offs[i]:offs[i] + len(f)]) for s in scores]
            d, l = loss_func(block, per_field_truth[i])
            total += float(l)
            for k in range(len(scores)):
                d_scores[k][:, offs[i]:offs[i] + len(f)] = d[k]
        return float(total), d_scores

    def initialize(self, get_examples, *, nlp=None, labels=None):
        util.check_lexeme_norms(self.vocab, self.name)
        sub = list(islice(get_examples(), 100))
        for eg in get_examples():
            for tok in eg.reference:
                if not tok.tag_:
                    continue
                parts = tok.tag_.split(self.sep)
                if len(parts) != len(self.cfg["fields"]):
                    raise ValueError(
                        f"{self.name}: expected {len(self.cfg['fields'])} {self.sep!r}-separated "
                        f"fields, got {len(parts)} in {tok.tag_!r}. A tagset whose codes do not all "
                        f"have the same arity cannot be split this way — en's literal ',' tag is "
                        f"exactly this case, which is why this component is not language-neutral.")
                for i, p in enumerate(parts):
                    self.add_field_value(i, p)
        for f in self.cfg["fields"]:
            f.sort()
        declared = self.cfg.get("field_sizes")
        found = [len(f) for f in self.cfg["fields"]]
        if declared and list(declared) != found:
            raise ValueError(
                f"{self.name}: the config declares field_sizes {list(declared)} but the training "
                f"data has {found}. `concatenate` sized its heads from the config, so training on "
                f"this would slice the output at the wrong offsets — rebuild the config with "
                f"scripts/build_lzh_xpos_tables.py's sizes.")
        _, width = self.offsets
        if not sub:
            raise ValueError(f"{self.name}: initialize got no examples to size the heads from")
        self.model.initialize(X=[eg.reference for eg in sub],
                              Y=[self.model.ops.alloc2f(len(eg.reference), width) for eg in sub])
        return self

    # ---- scoring ---------------------------------------------------------------------------
    def score(self, examples, **kwargs):
        """Whole-tag accuracy, PLUS accuracy per field and the off-grid rate.

        ⚠ The per-field numbers are what this component exists to expose and the whole-tag number
        is what it is judged on: four heads can each be right more often than the joint tagger and
        still assemble fewer correct CODES, because the fields are strongly dependent (`n` never
        co-occurs with 動詞). Reporting only the field accuracies would hide exactly that."""
        from spacy.scorer import Scorer
        scores = Scorer.score_token_attr(examples, "tag", **kwargs)
        nf = len(self.cfg["fields"])
        right = [0] * nf
        total = offgrid = 0
        attested = set(self.attested)
        for eg in examples:
            for pred, gold in zip(eg.predicted, eg.reference):
                if not gold.tag_:
                    continue
                total += 1
                if attested and pred.tag_ not in attested:
                    offgrid += 1
                p, g = pred.tag_.split(self.sep), gold.tag_.split(self.sep)
                for i in range(nf):
                    if i < len(p) and i < len(g) and p[i] == g[i]:
                        right[i] += 1
        if total:
            scores["tag_field_acc"] = {f"field{i+1}": right[i] / total for i in range(nf)}
            scores["tag_offgrid"] = offgrid / total
        return scores

    # ---- serialisation ---------------------------------------------------------------------
    def to_disk(self, path, exclude=tuple()):
        path = util.ensure_path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "cfg").write_text(json.dumps(self.cfg, ensure_ascii=False), encoding="utf-8")
        (path / "tables.json").write_text(
            json.dumps({"attested": self.attested, "allowed": self.allowed}, ensure_ascii=False),
            encoding="utf-8")
        (path / "model").write_bytes(self.model.to_bytes())

    def from_disk(self, path, exclude=tuple()):
        path = util.ensure_path(path)
        self.cfg.update(json.loads((path / "cfg").read_text(encoding="utf-8")))
        f = path / "tables.json"
        if not f.exists():
            raise OSError(
                f"{self.name}: {f} is missing. The attested-tag inventory and the UPOS mask ARE "
                f"this component's decoding contract — without them it emits free per-field "
                f"argmaxes, which land off the 121-tag grid. Rebuild with "
                f"scripts/build_lzh_xpos_tables.py.")
        self.load_tables(f)
        # ⚠ NO `initialize` HERE. thinc's `from_bytes` restores every dim along with the weights
        # (verified: a Linear round-trips nO/nI), and calling `initialize` first would need a
        # sample Doc — which `StaticVectors` requires in order to infer `nM`, and which `from_disk`
        # does not have. Initializing without one raises E905 and the model never loads.
        self.model.from_bytes((path / "model").read_bytes())
        return self
