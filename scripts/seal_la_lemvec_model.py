#!/usr/bin/env python3
"""Make a lemma-vector arm loadable away from the machine that built its table, and PROVE it.

THE FAILURE THIS PREVENTS, and it is the one that kept this layer out of a wheel. The config names
the table by PATH (`scripts/la_lemmavec_96.npz`). spaCy stores that config inside the model and
rebuilds the architecture from it BEFORE restoring any weights, so on a machine without that
relative path the model raises at construction — or, with the pre-sealing code, constructed against
a table that simply was not there. This rewrites the stored config to `vectors = null` +
`vector_dim = N`, since the 4 553-lemma table itself travels in the model's own bytes: the payload
lives in the extractor's `attrs`, which thinc serialises with the weights.

⚠ VERIFY THE RELOADED MODEL, NEVER THE IN-MEMORY ONE (standing hazard 8). The check here reloads
from disk with the build-time .npz HIDDEN, confirms the layer still holds every vector, and compares
parses token for token against the original. A model that silently lost its table would load
cleanly and parse slightly worse — indistinguishable from a capacity control, which is exactly how
the zh one-token-per-string wheel shipped.

    seal_la_lemvec_model.py training_la_lemvec/model-best training_la_lemvec_sealed
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code  # noqa: E402,F401
import spacy  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402

SEALABLE = {"sud.LemmaVecEmbed.v1", "sud.LemmaVecFeatsEmbed.v1", "sud.LemmaVecFeatsAgreeEmbed.v1"}


def embed_section(cfg):
    """The embed block of whichever component carries a sealable architecture."""
    for comp in cfg.get("components", {}).values():
        model = comp.get("model", {})
        for key in ("embed", "tok2vec"):
            sub = model.get(key, {})
            if not isinstance(sub, dict):
                continue
            if sub.get("@architectures") in SEALABLE:
                return sub
            inner = sub.get("embed", {})
            if isinstance(inner, dict) and inner.get("@architectures") in SEALABLE:
                return inner
    return None


def extractor_nodes(nlp):
    out = []
    for _, pipe in nlp.pipeline:
        model = getattr(pipe, "model", None)
        if model is None:
            continue
        for node in model.walk():
            if node.name == "extract_lemma_vectors":
                out.append(node)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dest", type=pathlib.Path)
    ap.add_argument("--corpus", default="corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy")
    ap.add_argument("--docs", type=int, default=25)
    args = ap.parse_args()

    # 1. load WITH the table file present: the architecture reads it into the payload
    nlp = spacy.load(args.src)
    nodes = extractor_nodes(nlp)
    if not nodes:
        raise SystemExit(f"{args.src}: no lemma-vector block found — nothing to seal")
    payloads = [n.attrs.get("lv_payload") or {} for n in nodes]
    live = [p for p in payloads if p]
    if not live:
        raise SystemExit(f"{args.src}: the block is a capacity control (constant) — nothing to seal")
    dim = live[0]["shape"][1]
    print(f"{args.src}: {len(nodes)} lemma-vector block(s), "
          f"{len(live[0]['keys'])} lemmas x {dim} dims in the payload")

    # 2. rewrite the config BEFORE writing, so the saved config.cfg never names the path
    sec = embed_section(nlp.config)
    if sec is None:
        raise SystemExit(f"{args.src}: no sealable architecture in the stored config")
    was = sec.get("vectors")
    sec["vectors"] = None
    sec["vector_dim"] = int(dim)
    if args.dest.exists():
        shutil.rmtree(args.dest)
    nlp.to_disk(args.dest)
    print(f"  sealed -> {args.dest}   (vectors {was!r} -> null, vector_dim = {dim})")

    # 3. the only check that counts: reload from disk with the .npz HIDDEN
    src_table = pathlib.Path(was) if was else None
    hidden = None
    if src_table and src_table.exists():
        hidden = src_table.with_suffix(src_table.suffix + ".hidden")
        src_table.rename(hidden)
    try:
        reloaded = spacy.load(args.dest)
        rnodes = extractor_nodes(reloaded)
        rlive = [n.attrs.get("lv_payload") or {} for n in rnodes]
        if not rlive or not rlive[0]:
            raise SystemExit("FAIL: the reloaded model has an EMPTY payload — the table did not "
                             "travel in the bytes")
        if len(rlive[0]["keys"]) != len(live[0]["keys"]):
            raise SystemExit(f"FAIL: reloaded table has {len(rlive[0]['keys'])} lemmas, "
                             f"original {len(live[0]['keys'])}")
        print(f"  reloaded with the table file hidden: {len(rlive[0]['keys'])} lemmas present")

        docs = list(DocBin().from_disk(args.corpus).get_docs(spacy.blank("la").vocab))[:args.docs]
        same = tot = 0
        for g in docs:
            words = [t.text for t in g]
            spaces = [bool(t.whitespace_) for t in g]
            a = nlp(Doc(nlp.vocab, words=words, spaces=spaces))
            b = reloaded(Doc(reloaded.vocab, words=words, spaces=spaces))
            for x, y in zip(a, b):
                tot += 1
                same += (x.head.i == y.head.i and x.dep_ == y.dep_
                         and x.tag_ == y.tag_ and x.lemma_ == y.lemma_
                         and str(x.morph) == str(y.morph))
        print(f"  parses identical on {same}/{tot} tokens")
        if same != tot:
            raise SystemExit("FAIL: the sealed model does not reproduce the original")
    finally:
        if hidden and hidden.exists():
            hidden.rename(src_table)
    print("  OK — sealed, reloaded without the build-time table, and byte-for-byte equivalent")


if __name__ == "__main__":
    main()
