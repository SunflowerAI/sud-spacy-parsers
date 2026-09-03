#!/usr/bin/env python3
"""`sud.FrozenPipeTok2Vec.v1` — hand one component's TRAINED encoder to another, frozen.

THE QUESTION IT EXISTS TO ASK. The morphologiser's own `HashEmbedCNN` — 64 wide, 499 456 params,
reading NORM/PREFIX/SUFFIX/SHAPE *and* the PCA'd SikuBERT table — has been fitted under UPOS
supervision, and it is measurably better at category than anything the parser's encoder can learn
from the treebank alone: 73.98 % UPOS on treebank-unseen forms against a shuffled control's 66.92,
+13.33 on tokens holding a character the treebank never showed. Giving the PARSER that
representation is not the same experiment as giving it the raw vectors, and the difference is the
point:

  * `sud.StaticVecChannel.v1` hands the parser a raw 96-d row and asks it to learn, from PARSE
    supervision alone and on the 1.15 % of tokens where it matters, what category that row implies.
    Measured over three seeds: nothing, on any frequency slice (NEGATIVE-RESULTS.md).
  * This layer hands over the extraction ALREADY DONE — a 64-d representation shaped by dedicated
    UPOS supervision over all 460 k training tokens — as a dense channel rather than a 15-way label.

It also passes both of NEGATIVE-RESULTS.md's pre-flight checks, which the predicted-XPOS channel
failed. The donor is a SEPARATE deep encoder, not a `Tok2VecListener` on the parser's own tok2vec,
so its output is not linearly decodable from the parser's input the way a co-encoder head's
prediction is; and it is not a deterministic function of anything the parser already reads, because
the SikuBERT rows enter it and enter nothing else.

⚠ **THE HOST PIPELINE MUST LOAD THE DONOR'S OWN VECTOR TABLE.** `spacy.StaticVectors.v2` reads
`doc.vocab.vectors` at FORWARD time — from the doc it is handed, not from the pipeline it was
trained in. A donor trained against `vectors_lzh_siku96` dropped into a host holding the shuffled
table would load, run, and be wrong in exactly the way CLAUDE.md standing hazard 11 describes. The
config generator pairs them; `--check` on this module's `__main__` re-asserts it.

⚠ **FROZEN MEANS FROZEN, AND IT IS ASSERTED, NOT ASSUMED.** The inner model is called with
`is_train=False` and its backprop callback is DISCARDED, so no gradient ever reaches its parameters
and `finish_update` finds nothing to apply. `check_frozen_pipe_tok2vec.py` compares the donor's
weights before and after a training run and refuses to report a result if a single one moved.
"""
import pathlib
import sys

from spacy.util import registry
from thinc.api import Model

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def load_encoder(path, component="morphologizer"):
    """The trained `tok2vec` sub-model of `component` in the pipeline saved at `path`."""
    import seg_code  # noqa: F401  (registers every custom factory the donor arm may name)
    import spacy
    nlp = spacy.load(path)
    if component not in nlp.pipe_names:
        raise ValueError(f"{path} has no component {component!r}; it has {nlp.pipe_names}")
    model = nlp.get_pipe(component).model
    if "tok2vec" not in model.ref_names:
        raise ValueError(f"{path}:{component} exposes no 'tok2vec' ref (has {list(model.ref_names)})")
    return model.get_ref("tok2vec")


@registry.architectures("sud.FrozenPipeTok2Vec.v1")
def FrozenPipeTok2Vec(path: str, component: str = "morphologizer") -> Model:
    inner = load_encoder(path, component)

    def forward(model, X, is_train):
        sub = model.layers[0]
        # is_train=False and the callback DISCARDED: no graph, so no gradient can reach the donor.
        Y, _ = sub(X, is_train=False)
        # Docs are not differentiable, so there is nothing to hand back up the concatenation.
        return Y, lambda dY: []

    def init(model, X=None, Y=None):
        return model                       # the donor arrives trained; never re-initialise it

    return Model("frozen_pipe_tok2vec", forward, init=init, layers=[inner],
                 dims={"nO": inner.get_dim("nO")})
