#!/usr/bin/env python3
"""`sud.WarmStartTagger.v1` — start a conditioned tagger AS the released one, then let it improve.

WHY. `make_xpos_config.py --top` concatenates a morphology side channel under the tagger's softmax,
so the classifier's input grows from the encoder's width W to W + S. Trained from scratch the head
has to relearn everything the released tagger already knew, and on the arms where the released head
was the better one that deficit eats the gain the side channel buys (measured: ko -0.31, zh -0.47 on
the retrained-head control alone).

WHAT. After initialisation, copy the RELEASED tagger into the fresh one:

  * the output layer's `W` goes into the FIRST W columns and the new S columns are set to ZERO, so
    the side channel contributes exactly nothing at step 0;
  * `b` is copied outright;
  * the inner tok2vec's parameters are copied when it has any -- which is what extends this to la
    and en_gum, whose shipping taggers carry a dedicated `HashEmbedCNN` rather than a listener. On a
    listener arm there is nothing to copy and the frozen shared encoder already supplies it.

Together those mean the model at step 0 IS the released tagger, to the bit. Training can only move
away from it, and the side channel has to earn every column it uses.

⚠ **LABEL ORDER IS THE TRAP.** `W` is indexed by label id, so copying it into a tagger whose labels
sit in a different order silently scrambles every class -- the same hazard `rename_deprel_label.py`
guards for the parser's action table. Rather than permute, `make_xpos_config.py --warm-start` writes
the released tagger's label list to disk and makes the new tagger initialise from it, so the orders
match by construction; this callback then REFUSES to copy unless they are identical, position for
position.

`scripts/check_warm_start.py` verifies the whole thing end to end: it initialises the config and
checks the untrained model reproduces the released tagger's predictions exactly.

Config usage (written by `make_xpos_config.py --warm-start <arm>`):

    [initialize.after_init]
    @callbacks = "sud.WarmStartTagger.v1"
    source = "training_ar_lemma/model-best"
"""
import os
import sys

from spacy.util import registry

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _param_sig(model):
    return [(nd.name, k, tuple(nd.get_param(k).shape))
            for nd in model.walk() for k in sorted(nd.param_names) if nd.has_param(k)]


def _copy_params(src, dst):
    for s_node, d_node in zip(src.walk(), dst.walk()):
        for k in s_node.param_names:
            if s_node.has_param(k):
                d_node.set_param(k, s_node.get_param(k).copy())


@registry.callbacks("sud.WarmStartTagger.v1")
def WarmStartTagger(source: str, pipe: str = "tagger"):
    def warm_start(nlp):
        import spacy
        src_nlp = spacy.load(source)
        s_pipe, d_pipe = src_nlp.get_pipe(pipe), nlp.get_pipe(pipe)

        if list(s_pipe.labels) != list(d_pipe.labels):
            raise ValueError(
                f"sud.WarmStartTagger: label ORDER differs between {source} and the new arm "
                f"({len(s_pipe.labels)} vs {len(d_pipe.labels)} labels). The output layer is "
                f"indexed by label id, so copying it would scramble every class. Point "
                f"initialize.components.{pipe}.labels at the released arm's label list.")

        s_out, d_out = s_pipe.model.get_ref("output_layer"), d_pipe.model.get_ref("output_layer")
        Ws, Wd = s_out.get_param("W"), d_out.get_param("W")
        if Ws.shape[0] != Wd.shape[0] or Ws.shape[1] > Wd.shape[1]:
            raise ValueError(f"sud.WarmStartTagger: incompatible output layers {Ws.shape} -> {Wd.shape}")
        new_W = d_out.ops.alloc2f(*Wd.shape)          # zeros: the side channel starts inert
        new_W[:, :Ws.shape[1]] = Ws
        d_out.set_param("W", new_W)
        d_out.set_param("b", s_out.get_param("b").copy())

        # the inner encoder, when the released tagger has one of its own (la, en_gum)
        s_tv = s_pipe.model.get_ref("tok2vec")
        d_tv = d_pipe.model.get_ref("tok2vec")
        sig = _param_sig(s_tv)
        copied = "listener (no parameters)"
        if sig:
            # --top wraps the inner encoder in a concatenate, so the target may be a sub-layer.
            cands = [d_tv] + list(getattr(d_tv, "layers", []))
            match = next((c for c in cands if _param_sig(c) == sig), None)
            if match is None:
                raise ValueError(
                    f"sud.WarmStartTagger: no sub-model of the new tagger's tok2vec matches the "
                    f"released encoder's parameter signature ({len(sig)} tensors) -- refusing to "
                    f"copy weights into a different architecture")
            _copy_params(s_tv, match)
            copied = f"inner encoder ({len(sig)} tensors)"
        print(f"sud.WarmStartTagger: {pipe} warm-started from {source} -- "
              f"head {Ws.shape} -> {Wd.shape} (new columns zeroed), {copied}")
        return nlp
    return warm_start
