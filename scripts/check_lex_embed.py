#!/usr/bin/env python3
"""Verify `sud.LexFieldEmbed.v1` before any arm is trained on it.

Five checks, each standing for a way this layer could look healthy and be wrong:

  1. **The table survives serialisation.** thinc's `Model.to_dict` skips an attr whose value raises
     TypeError, WITHOUT SAYING SO. A dropped table does not crash: it makes every token `<OOV>` and
     the arm scores like its own capacity control. Verified on the RELOADED model, never the
     in-memory one (standing hazard 8).
  2. **An OOV form and a form with no majority are the same input** -- one sentinel, no second one
     to get wrong (the unset-vs-empty MORPH failure, which cost sa 6.8 LAS).
  3. **The control carries nothing.** `constant = true` must give byte-identical columns for every
     token, or the "capacity control" is not one.
  4. **The builder and the layer agree on the fold key.** They compute it in two different places
     from two different objects (a CoNLL-U file, a `Doc`). If they disagree, jackknifing still
     "works" -- it just injects an unrelated noise pattern, and the ablation reads as a result.
  5. **Jackknifing bites, and by how much.** NEGATIVE-RESULTS.md records a whole sweep invalidated
     because a parameter was never wired through, so the rate is printed rather than assumed: the
     train-time OOV rate should sit near the rate unseen text will really see, not near zero.

Usage:

    .venv/bin/python scripts/check_lex_embed.py --table models/lzh_xpos_lex.json \\
        --train assets_lzh/.../lzh_kyoto-sud-train.<suffix>.conllu \\
        --test  assets_lzh/.../lzh_kyoto-sud-test.<suffix>.conllu
"""
import argparse
import json
import pathlib
import sys

import numpy
import spacy
from spacy.strings import hash_string
from spacy.tokens import Doc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_xpos_lexicon import fold_of, read_sentences  # noqa: E402
from sud_lex_embed import OOV, LexFieldEmbed  # noqa: E402

FAIL = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(label)


def build(table, fields, rows, constant=False):
    m = LexFieldEmbed(width=16, fields=fields, rows=rows, table=table, constant=constant)
    m.initialize()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--fields", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    tbl = json.loads(pathlib.Path(a.table).read_text(encoding="utf-8"))
    nlp = spacy.blank("xx")
    rows = [8, 16, 64][:len(a.fields)]

    known = next(f for f, v in tbl["full"].items() if v[0] != OOV)
    doc = Doc(nlp.vocab, words=[known, "ABSENT"], spaces=[False, False])  # one of each
    print(f"table: {a.table}  ({len(tbl['full'])} forms, k={tbl['k']}, "
          f"fold_key={tbl['fold_key']!r})")

    # 1. round trip
    m = build(a.table, a.fields, rows)
    before = m.predict([doc])[0]
    m2 = build(a.table, a.fields, rows)
    m2.from_bytes(m.to_bytes())
    after = m2.predict([doc])[0]
    check(numpy.allclose(before, after), "table survives to_bytes/from_bytes",
          f"max |Δ| = {float(numpy.abs(before - after).max()):.2e}")
    # ...and that the surviving table is the real one, not an empty dict that happens to round-trip.
    reloaded = [n for n in m2.walk() if n.name == "extract_lex_fields"][0]
    check(len(reloaded.attrs["lex_table"].get("full", {})) == len(tbl["full"]),
          "the RELOADED model still holds every form",
          f"{len(reloaded.attrs['lex_table'].get('full', {}))} of {len(tbl['full'])}")

    # 2. one sentinel
    ex = [n for n in m.walk() if n.name == "extract_lex_fields"][0]
    cols, _ = ex([Doc(nlp.vocab, words=["ABSENT"], spaces=[False])], is_train=False)
    want = numpy.asarray([hash_string(f"xpos{f}=<OOV>") for f in a.fields], dtype="uint64")
    check(bool((numpy.asarray(cols[0])[0] == want).all()),
          "an absent form hashes to the <OOV> sentinel in every column")

    # 3. the control carries nothing
    c = build(None, a.fields, rows, constant=True)
    cx = [n for n in c.walk() if n.name == "extract_lex_fields"][0]
    words = list(tbl["full"])[:50]
    ccols, _ = cx([Doc(nlp.vocab, words=words, spaces=[False] * len(words))], is_train=False)
    arr = numpy.asarray(ccols[0])
    check(bool((arr == arr[0]).all()), "constant=true gives identical columns for every token",
          f"{arr.shape[0]} tokens, {len(set(map(tuple, arr)))} distinct rows")
    def nparams(model):
        return sum(int(nd.get_param(n).size) for nd in model.walk()
                   for n in nd.param_names if nd.has_param(n))

    check(nparams(c) == nparams(m), "control and arm have the same parameter count",
          f"{nparams(c)} vs {nparams(m)}")

    # 4. builder and layer agree on the fold key
    k = tbl["k"]
    agree = total = 0
    for sent in read_sentences(a.train):
        forms = [f for f, _ in sent]
        d = Doc(nlp.vocab, words=forms, spaces=[False] * len(forms))
        total += 1
        agree += fold_of(forms, k) == hash_string("".join(t.text for t in d)) % k
    check(agree == total, "builder and layer assign the same fold to every training sentence",
          f"{agree}/{total}")

    # 5. how hard it bites
    def oov_rate(path, is_train):
        seen = miss = 0
        for sent in read_sentences(path):
            forms = [f for f, _ in sent]
            diff = tbl["folds"][fold_of(forms, k) % k] if (is_train and k) else None
            for f in forms:
                seen += 1
                codes = diff.get(f) if diff is not None else None
                if codes is None:
                    codes = tbl["full"].get(f)
                if codes is None or codes[0] == OOV:
                    miss += 1
        return miss / max(seen, 1)

    tr_jk = oov_rate(a.train, True)
    tr_full = oov_rate(a.train, False)
    print(f"  ----  train OOV rate: jackknifed {tr_jk:.2%}, full table {tr_full:.2%}")
    if a.test:
        te = oov_rate(a.test, False)
        print(f"  ----  test  OOV rate (inference regime): {te:.2%}")
        check(tr_jk > tr_full, "jackknifing raises the train-time OOV rate above zero",
              f"{tr_full:.2%} -> {tr_jk:.2%} against test's {te:.2%}")

    print()
    if FAIL:
        sys.exit("FAILED: " + "; ".join(FAIL))
    print("all checks passed")


if __name__ == "__main__":
    main()
