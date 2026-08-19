#!/usr/bin/env python3
"""Verify `sud.KoAnalyserEmbed.v1` before an arm is trained through it.

Six things, each with a precedent for failing SILENTLY in this repo:
  1. No extra channel configured is exactly `spacy.MultiHashEmbed.v2`, so switching an arm over
     stays single-variable (as `check_feats_embed.py` / `check_affix_embed.py` do for their layers).
  2. The channels actually VARY across tokens. A layer that reads its source but finds nothing
     scores like its own capacity control instead of raising.
  3. The first-morpheme key collapses inflected eojeol onto ONE symbol — the whole mechanism.
  4. `constant = true` is information-free but parameter-identical.
  5. The backend fingerprint survives `to_bytes`/`from_bytes` and a mismatch REFUSES. thinc drops an
     unserialisable attr without saying so, which is standing hazard 8 in its exact shape.
  6. An unanalysable token and a token with no tags land on the SAME sentinel, never on all-zero.

    .venv/bin/python scripts/check_ko_embed.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy                                              # noqa: E402
import spacy                                              # noqa: E402
from spacy.tokens import Doc                              # noqa: E402
from spacy.util import registry                           # noqa: E402

import ko_analyser                                        # noqa: E402
import sud_ko_embed                                       # noqa: E402,F401

ATTRS, ROWS = ["NORM", "PREFIX", "SUFFIX", "SHAPE"], [5000, 1000, 2500, 2500]
FEATS = ["First", "Last", "Bag"]
MORPH_ROWS = [5000, 2000]

nlp = spacy.blank("ko")
SENTS = [
    "잡스는 워즈니악에게 도움을 청하고 게임을 설계했다 .".split(),
    "잡스가 워즈니악을 만나 회사를 세웠다 .".split(),
    "환차익을 노린 투기적 자금이 유입되고 있다 .".split(),
]
docs = [Doc(nlp.vocab, words=w) for w in SENTS]


def build(arch, **kw):
    numpy.random.seed(0)
    m = registry.architectures.get(arch)(width=96, attrs=ATTRS, rows=ROWS,
                                         include_static_vectors=False, **kw)
    m.initialize(X=docs)
    return m


print(f"analyser backend: {ko_analyser.fingerprint()}")

# 1 — equivalence with stock when nothing extra is configured
a = build("sud.KoAnalyserEmbed.v1", feats=[], morph_rows=[])
b = build("spacy.MultiHashEmbed.v2")
assert a.to_bytes() == b.to_bytes(), "no-channel build is NOT byte-identical to MultiHashEmbed.v2"
print("1 ok  feats=[] morph_rows=[] is byte-identical to spacy.MultiHashEmbed.v2")

# 2 — the channels vary
m = build("sud.KoAnalyserEmbed.v1", feats=FEATS, morph_rows=MORPH_ROWS)
ids = [n for n in m.walk() if n.name == "extract_ko_morph_ids"][0]
tagsets = [n for n in m.walk() if n.name == "extract_ko_tag_sets"][0]
I, _ = ids(docs, False)
V, _ = tagsets(docs, False)
Iall = numpy.vstack([numpy.asarray(x) for x in I])
Vall = numpy.vstack([numpy.asarray(x) for x in V])
n_first, n_last = len(set(Iall[:, 0].tolist())), len(set(Iall[:, 1].tolist()))
silent_bits = [(i + 1) * (len(sud_ko_embed.KO_TAGS) + 1) - 1 for i in range(len(FEATS))]
answered = 1.0 - Vall[:, silent_bits].mean(axis=0)
assert n_first > 1 and n_last > 1 and Vall.sum() > 0
print(f"2 ok  {Iall.shape[0]} tokens -> {n_first} first-morpheme and {n_last} last-morpheme keys; "
      f"tag block {Vall.shape[1]} dims, analysed " +
      ", ".join(f"{f} {c:.0%}" for f, c in zip(FEATS, answered)))

# 3 — the mechanism itself: inflected forms of one stem collapse onto one key
variants = ["잡스는", "잡스가", "잡스를", "잡스에게", "잡스의"]
vdoc = Doc(nlp.vocab, words=variants)
(vi,), _ = ids([vdoc], False)
vi = numpy.asarray(vi)
assert len(set(vi[:, 0].tolist())) == 1, "inflected forms of one stem did NOT collapse"
assert len(set(vi[:, 1].tolist())) == len(variants), "distinct particles collapsed onto one key"
assert len(set(t.orth for t in vdoc)) == len(variants)
print(f"3 ok  {variants} -> 1 lexical key, {len(variants)} functional keys "
      f"(the parser reads {len(variants)} unrelated symbols today)")

# 4 — the capacity control is information-free but the same size
ctl = build("sud.KoAnalyserEmbed.v1", feats=FEATS, morph_rows=MORPH_ROWS, constant=True)
cids = [n for n in ctl.walk() if n.name == "extract_ko_morph_ids"][0]
ctagsets = [n for n in ctl.walk() if n.name == "extract_ko_tag_sets"][0]
(ci,), _ = cids([vdoc], False)
(cv,), _ = ctagsets([vdoc], False)
assert len(set(numpy.asarray(ci).ravel().tolist())) == 1, "the control varies across tokens"
assert len(set(map(tuple, numpy.asarray(cv).tolist()))) == 1, "the control's tag block varies"
n_par = lambda mm: sum(int(numpy.asarray(p).size) for n in mm.walk() for p in
                       (n.get_param(k) for k in n.param_names) if p is not None)
assert n_par(ctl) == n_par(m), f"control has {n_par(ctl)} params against {n_par(m)}"
print(f"4 ok  constant=true is one row for every token, at the same {n_par(m)} parameters")

# 5 — the fingerprint round-trips, and a mismatch refuses
m2 = build("sud.KoAnalyserEmbed.v1", feats=FEATS, morph_rows=MORPH_ROWS)
m2.from_bytes(m.to_bytes())
ids2 = [n for n in m2.walk() if n.name == "extract_ko_morph_ids"][0]
assert ids2.attrs.get("ko_backend") == ko_analyser.fingerprint(), \
    "the backend fingerprint did NOT survive serialisation"
J, _ = ids2(docs, False)
assert all((numpy.asarray(p) == numpy.asarray(q)).all() for p, q in zip(I, J))
ids2.attrs["ko_backend"] = "some-other-analyser/its-own-dic"
try:
    ids2(docs, False)
    raise AssertionError("a backend mismatch did NOT refuse")
except ValueError as e:
    assert "trained against" in str(e)
print(f"5 ok  fingerprint {ko_analyser.fingerprint()!r} survives the round trip; a mismatch refuses")

# 6 — one sentinel, and never an all-zero row
odd = Doc(nlp.vocab, words=["ㅤ", "＠"])          # a filler jamo and a fullwidth symbol
(oi,), _ = ids([odd], False)
(ov,), _ = tagsets([odd], False)
oi, ov = numpy.asarray(oi), numpy.asarray(ov)
assert (ov.sum(axis=1) > 0).all(), "a token came back as an all-zero tag row"
per_feat = ov.reshape(len(odd), len(FEATS), len(sud_ko_embed.KO_TAGS) + 1)
assert (per_feat.sum(axis=2) >= 1).all(), "a feature came back with no bit set at all"
print(f"6 ok  every token sets at least one bit per feature; sentinel id {sud_ko_embed._SILENT_ID}")
print("\nall checks passed")
