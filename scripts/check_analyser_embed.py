#!/usr/bin/env python3
"""Verify `sud.AnalyserFeatsEmbed.v1` before an arm is trained through it.

Four things, each of which has a precedent for failing SILENTLY in this repo:
  1. `feats = []` is exactly `spacy.MultiHashEmbed.v2`, so switching an arm over stays
     single-variable (as `check_feats_embed.py` / `check_affix_embed.py` guarantee for their layers).
  2. The multi-hot block actually VARIES across tokens — a layer that reads the table but finds
     nothing scores like its own capacity control instead of raising.
  3. The table survives `to_bytes`/`from_bytes`. thinc drops an unserialisable attr without saying
     so, which is standing hazard 8 in its exact shape.
  4. `constant = true` really is information-free.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy, spacy                                        # noqa: E402
from spacy.tokens import DocBin                            # noqa: E402
from spacy.util import registry                            # noqa: E402
import sud_analyser_embed                                   # noqa: E402,F401

TABLE = "scripts/sa_analyser_lut.json.gz"
FEATS = ["Case", "Number", "Gender", "Person"]
ATTRS, ROWS = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "MORPH"], [5000, 1000, 2500, 2500, 64]

nlp = spacy.blank("sa")
docs = list(DocBin().from_disk("corpus_sa_mwt_norm/sa_vedic-sud-test.csl_mwt.spacy")
            .get_docs(nlp.vocab))[:20]

def build(arch, **kw):
    numpy.random.seed(0)
    m = registry.architectures.get(arch)(width=96, attrs=ATTRS, rows=ROWS,
                                         include_static_vectors=False, **kw)
    m.initialize(X=docs[:4])
    return m

# 1 — equivalence with stock when no feature is configured
a = build("sud.AnalyserFeatsEmbed.v1", table=TABLE, feats=[])
b = build("spacy.MultiHashEmbed.v2")
assert a.to_bytes() == b.to_bytes(), "feats=[] is NOT byte-equivalent to spacy.MultiHashEmbed.v2"
print("1 ok  feats=[] is byte-identical to spacy.MultiHashEmbed.v2")

# 2 — the block varies, and the silent bit is not the whole story
m = build("sud.AnalyserFeatsEmbed.v1", table=TABLE, feats=FEATS)
ext = [n for n in m.walk() if n.name == "extract_analyser_sets"][0]
X, _ = ext(docs, False)
V = numpy.vstack(X)
sil = [sum(len(ext.attrs["an_payload"]["values"][f]) for f in FEATS[:i]) + i + len(ext.attrs["an_payload"]["values"][FEATS[i]])
       for i in range(len(FEATS))]
covered = 1.0 - V[:, sil].mean(axis=0)
print(f"2 ok  block {V.shape[1]} dims; distinct rows {len(set(map(tuple, V.tolist())))}; "
      f"analyser answered " + ", ".join(f"{f} {c:.1%}" for f, c in zip(FEATS, covered)))
assert V.sum() > 0 and covered.max() > 0.5

# 3 — the table survives a round trip
m2 = build("sud.AnalyserFeatsEmbed.v1", table=TABLE, feats=FEATS)
m2.from_bytes(m.to_bytes())
ext2 = [n for n in m2.walk() if n.name == "extract_analyser_sets"][0]
assert ext2.attrs["an_payload"].get("table"), "the table did NOT survive serialisation"
Y, _ = ext2(docs, False)
assert all((p == q).all() for p, q in zip(X, Y))
print(f"3 ok  table survived to_bytes/from_bytes ({len(ext2.attrs['an_payload']['table'])} forms)")

# 4 — the control carries no information
c = build("sud.AnalyserFeatsEmbed.v1", table=TABLE, feats=FEATS, constant=True)
extc = [n for n in c.walk() if n.name == "extract_analyser_sets"][0]
Z, _ = extc(docs, False)
W = numpy.vstack(Z)
assert len(set(map(tuple, W.tolist()))) == 1, "constant=true is not information-free"
print("4 ok  constant=true emits one row for every token (capacity control)")
print("\nall checks passed")
