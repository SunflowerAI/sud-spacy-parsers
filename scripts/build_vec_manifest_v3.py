#!/usr/bin/env python3
"""Name the vector sources for the v3 lexical channel, from the v2 split rather than by hand.

WHY A GENERATED MANIFEST AND NOT A DICT IN THE SCRIPT. `align_vectors.py` carries `SOURCES` and
`TREEBANK` as hand-written literals for thirteen languages. Thirty-eight is past the point where a
literal stays correct, and this repo has paid four times over for a default that named the wrong
corpus -- `train_sud.sh` was still pointing Sanskrit at a superseded generation long after the
rebuild driver was marked superseded, and a superseded corpus loads, converts and trains exactly
like a current one. The corpus each language uses is DERIVED here from the same v2 manifest the
parser was trained against, so the two cannot drift apart.

THE KEY IS THE LEMMA, AND THAT IS THE WHOLE POINT OF THE ARM. The channel is trained on the aligned
vector of a token's LEMMA and filled at deployment from the vector of an English GLOSS, which is a
lexeme-level translation. Keying by surface form would put an inflected form's vector against a
citation-form gloss. Measured, on Arabic PADT against the v1 form-keyed asset: querying lemmas into
a form-keyed table gives 25.3 % coverage and a mean cosine of -0.075 against its own gloss, only
57 % beating a shuffled control -- the hits are largely collisions between vocalised citation forms
and unvocalised surface forms. A lemma-keyed table has to be BUILT lemma-keyed; it cannot be
queried out of a form-keyed one.

ENGLISH IS KEYED BY FORM, DELIBERATELY, and it is the one language that is. English is not a
training language here -- it is the DEPLOYMENT LOOKUP, the table an English gloss is resolved
against. Glosses are not reliably citation forms (`Gloss=made`, `Gloss=things` in Yoruba-YTB), so a
lemma-keyed English table would miss them. This leaves a mismatch worth recording rather than
hiding: the parser learns on a lemma vector and is asked at inference for an inflected gloss vector.
fastText places `make` and `made` close together, so the cost is small, but it is not zero and it is
not measured yet.

⚠ TEST LANGUAGES GET NO TREEBANK PATHS. Their tables exist only for the diagnostic upper bound --
"how much better would a real aligned table have been than an English gloss?" -- and a deployer of
this arm has no treebank by construction. Building their vocabulary from a frequency head alone
keeps the diagnostic honest and keeps every test `.conllu` unread. Enforced below, not advised.
"""
from __future__ import annotations
import argparse, json, pathlib, sys

# fastText publishes 44 languages already rotated into the English space (`vectors-aligned`).
# Membership is a property of fastText, not of us: every one of these was confirmed to return 200.
ALIGNED44 = set(
    "af ar bg bn bs ca cs da de el en es et fa fi fr he hi hr hu id it ko lt lv mk ms nl no pl pt "
    "ro ru sk sl sq sv ta th tl tr uk vi zh".split())

SRC = pathlib.Path("assets_vec/src")

#: Languages whose LEMMA column and whose source space do not share a spelling convention, with the
#: fold that makes them meet. Each entry is a measured token-coverage gain, not a tidy-up -- the
#: rules themselves live in `aligned_vectors.py` so that the builder and the layer cannot diverge.
#:
#:     ar   41.2 % -> 96.2 %   vocalised PADT citation forms against unvocalised fastText
#:     ko   36.4 % -> 83.8 %   `+`-segmented morpheme lemmas against orthographic words
#:     et   82.6 % -> 88.4 %   compound-boundary marks (`maa_ilm`) that nobody writes
#:     fi   84.7 % -> 88.6 %   the same, spelled `#` (`yli#opisto`)
#:
#: ⚠ THE ABSENT FOLD IS THE DANGEROUS CASE, not a wrong one. A 41 % channel trains, converges and
#: reports a normal loss curve; it is simply worse, and no metric in the sweep names the cause.
KEY_NORM_FOR = {"ar": "ar", "ko": "ko", "et": "et", "fi": "fi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--out", default="assets_vec/sources_v3.json")
    ap.add_argument("--min-lemma-fill", type=float, default=0.50,
                    help="a training language below this cannot key a table by lemma")
    ap.add_argument("--allow-missing-vec", action="store_true",
                    help="write the manifest before the fetch has finished (build will still refuse)")
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))["languages"]
    inv = {r["name"]: r for r in json.load(open(a.inventory, encoding="utf-8"))["corpora"]}

    langs, skipped = {}, []
    for lc, v in sorted(man.items()):
        if lc not in ALIGNED44:
            skipped.append((lc, v["pool"], "not in fastText aligned-44"))
            continue
        rec = inv.get(v["corpus"], {})
        lf = rec.get("lemma_fill")
        vec = SRC / f"align.{lc}.vec"
        if not vec.exists() and not a.allow_missing_vec:
            print(f"MISSING source space: {vec}", file=sys.stderr); return 2

        if lc == "en":
            key, route = "form", "hub"            # the deployment lookup; see the docstring
        else:
            key, route = "lemma", "pre"
        if v["pool"] == "train" and lc != "en" and (lf is None or lf < a.min_lemma_fill):
            skipped.append((lc, v["pool"], f"lemma_fill {lf} < {a.min_lemma_fill}"))
            continue

        # Test languages are built from a frequency head ONLY -- no test .conllu is ever opened.
        tb = []
        if v["pool"] == "train":
            for split in ("train", "dev"):
                tb += rec.get("paths", {}).get(split, [])

        langs[lc] = dict(vec=str(vec), route=route, key=key, pool=v["pool"],
                         src=f"fastText aligned (wiki.{lc}.align)",
                         corpus=v["corpus"], lemma_fill=lf, treebank=tb,
                         key_norm=KEY_NORM_FOR.get(lc))

    # The invariant the diagnostic rests on, asserted rather than trusted.
    for lc, s in langs.items():
        if s["pool"] == "test" and s["treebank"]:
            print(f"REFUSING: test language {lc} was given treebank paths", file=sys.stderr); return 2

    tr = sorted(k for k, s in langs.items() if s["pool"] == "train")
    te = sorted(k for k, s in langs.items() if s["pool"] == "test")
    out = dict(meta=dict(
        aligned44=sorted(ALIGNED44), min_lemma_fill=a.min_lemma_fill,
        train=tr, test=te,
        basis_langs=tr,          # read by --stage basis; test languages are projected, never fitted
        note=("key=lemma everywhere except en, which is the deployment lookup and is keyed by form. "
              "Test tables carry a frequency head only and no treebank vocabulary."),
    ), languages=langs)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"wrote {a.out}: {len(tr)} train + {len(te)} test")
    print(f"  train {tr}")
    print(f"  test  {te}")
    print(f"  skipped {len(skipped)}:")
    for lc, pool, why in skipped:
        print(f"    {lc:4s} {pool:5s} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
