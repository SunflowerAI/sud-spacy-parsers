#!/usr/bin/env python3
"""What the v2 arm actually receives, asserted rather than assumed. Run before any training.

Every check here corresponds to a failure this repo has already paid for, or to one this build
already made. In particular:

  * check 4 caught seven test languages carrying `-train.conllu` files left behind by an earlier
    split, which silently un-held-them-out. The corpus reader discovers languages by which files
    exist, so nothing downstream could have told the difference.
  * check 8 is THE GATE. v1's +12.74 zero-shot typology result was never reportable because the
    held-out language's profile came from its own gold treebank. If any test bit here says
    `treebank`, the whole experiment is measuring an oracle again.
  * check 16 exists because 8 bits collide: an index-derangement can hand a language a
    bit-IDENTICAL profile and quietly stop being a control.
  * check 21 is the unset-vs-empty MORPH distinction, which cost Sanskrit 6.8 LAS. v1 argued it was
    safe by construction; v2 mixes ~100 treebanks with FEATS fill from 0 % to 100 %, so it is
    asserted.
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from spacy.tokens import Doc, DocBin  # noqa: E402
from spacy.util import registry  # noqa: E402
from thinc.api import Config  # noqa: E402

import generic_code_v2  # noqa: E402,F401  (registers the layer and the reader)
import spacy  # noqa: E402
from sud_generic_embed_v2 import derange_bits, load_typology  # noqa: E402

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]
LEAKY = ["Shared", "NameType", "Typo", "Style", "Foreign", "Abbr", "Hyph"]

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)
    return ok


def build_embed(cfg_path):
    cfg = Config().from_disk(cfg_path, interpolate=False).interpolate()
    spec = cfg["components"]["tok2vec"]["model"]["embed"]
    return registry.resolve({"m": spec})["m"]


def n_params(model):
    tot = 0
    for node in model.walk():
        for name in node.param_names:
            if node.has_param(name):
                tot += int(np.prod(node.get_param(name).shape))
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conllu", default="assets_generic_v2")
    ap.add_argument("--corpus", default="corpus_generic_v2")
    ap.add_argument("--manifest", default="assets_generic_v2/manifest.json")
    ap.add_argument("--typology", default="assets_typ/typology_v2.json")
    ap.add_argument("--codes", default="assets_typ/codes_and_flags.yaml")
    ap.add_argument("--baseline", default="metrics/generic_v2/baseline.json")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--tolerance", type=float, default=0.15)
    a = ap.parse_args()

    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))
    L = man["languages"]
    typ = json.loads(pathlib.Path(a.typology).read_text(encoding="utf-8"))["languages"]
    train = sorted(k for k, v in L.items() if v["pool"] == "train")
    test = sorted(k for k, v in L.items() if v["pool"] == "test")

    print("DATA")
    # 1-2. Leak removal, each paired with a non-vacuity check against a source file.
    for key in LEAKY:
        hits = [p for p in glob.glob(f"{a.conllu}/*.conllu")
                if any(f"{key}=" in ln for ln in open(p, encoding="utf-8"))]
        check(f"1. `{key}=` absent from the prepared corpus", not hits, f"{len(hits)} files" if hits else "")
    srcs = {"Shared": "assets/en_ewt-sud-train.relabeled_ext.conllu",
            "NameType": "assets_lzh/SUD_Classical_Chinese-Kyoto/"
                        "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu"}
    for key, src in srcs.items():
        n = sum(1 for ln in open(src, encoding="utf-8") if f"{key}=" in ln) if os.path.exists(src) else 0
        check(f"2. ...and `{key}=` IS in a source file (check 1 is not vacuous)", n > 0, f"{n} lines")
    # Parse the FEATS column rather than grepping the line: a substring test matched the WORDFORMS
    # `Sudan`, `Sudeste` and `Sudetenland` and reported a leak that was not there.
    def sud_keys(path):
        for ln in open(path, encoding="utf-8"):
            if ln[0] == "#" or not ln.strip():
                continue
            c = ln.split("\t")
            if len(c) < 6 or "-" in c[0] or "." in c[0] or c[5] == "_":
                continue
            for kv in c[5].split("|"):
                if kv.split("=", 1)[0].startswith("Sud"):
                    return True
        return False
    hits = [p for p in glob.glob(f"{a.conllu}/*.conllu") if sud_keys(p)]
    check("2b. no `Sud`-prefixed FEATS key in the prepared corpus", not hits, " ".join(hits[:4]))

    # 3. Genus disjointness, asserted against the ORIGINAL metadata rather than the manifest's copy.
    codes = yaml.safe_load(pathlib.Path(a.codes).read_text(encoding="utf-8"))
    name2genus = {n: (m.get("genus") or m.get("family") or n)
                  for n, m in codes.items() if isinstance(m, dict)}
    tr_genera = {L[k]["genus"] for k in train}
    te_genera = {L[k]["genus"] for k in test}
    check("3. train and test language sets are disjoint", not (set(train) & set(test)))
    check("3b. no genus appears on both sides", not (tr_genera & te_genera),
          f"overlap: {sorted(tr_genera & te_genera)}" if tr_genera & te_genera else "")
    check("3c. every language has a non-empty genus label", all(L[k]["genus"] for k in train + test))
    # The manifest copies genus from the inventory; re-derive from codes_and_flags so a stale copy
    # cannot make a genus-overlapping split look disjoint.
    redrift = [k for k in train + test
               if L[k].get("lang_name") and name2genus.get(L[k]["lang_name"], L[k]["genus"]) != L[k]["genus"]]
    check("3d. manifest genus agrees with codes_and_flags.yaml", not redrift, " ".join(redrift[:8]))

    # 4. Stale corpus files. THIS ONE HAS ALREADY FIRED ONCE.
    stale = [f"{k}-{s}" for k in test for s in ("train", "dev")
             if os.path.exists(f"{a.conllu}/{k}-{s}.conllu")
             or os.path.exists(f"{a.corpus}/{k}-{s}.spacy")]
    check("4. no test language has a train/dev corpus file", not stale, " ".join(stale))
    missing = [k for k in train if not os.path.exists(f"{a.corpus}/{k}-train.spacy")]
    check("4b. every training language has a converted train corpus", not missing, " ".join(missing[:8]))

    # 5. Dev is train-side only, or model selection peeks at the test typology.
    devs = {os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{a.corpus}/*-dev.spacy")}
    check("5. dev contains no test-side language", not (devs & set(test)), " ".join(sorted(devs & set(test))))

    # 6. Label inventory.
    def labels(pat):
        out = collections.Counter()
        for p in glob.glob(pat):
            for ln in open(p, encoding="utf-8"):
                if ln[0] == "#" or not ln.strip():
                    continue
                c = ln.split("\t")
                if len(c) >= 8 and "-" not in c[0] and "." not in c[0]:
                    out[c[7]] += 1
        return out
    tr_lab, te_lab = labels(f"{a.conllu}/*-train.conllu"), labels(f"{a.conllu}/*-test.conllu")
    unreachable = sorted(set(te_lab) - set(tr_lab))
    check("6. every test label is attested in training", not unreachable, " ".join(unreachable))
    rare = sorted(k for k, v in tr_lab.items() if v < 20)
    print(f"        {len(tr_lab)} labels; under 20 tokens in train: {' '.join(rare) or 'none'}")

    # 7-8. Profiles. Check 8 is the gate.
    bad = [k for k in train + test
           if k not in typ or len(typ[k]["bits"]) != 8
           or any(b not in (0, 1) for b in typ[k]["bits"])]
    check("7. every language has a profile of exactly 8 bits in {0,1}", not bad, " ".join(bad[:8]))
    oracle = [f"{k}:{f}" for k in test for f, s in typ.get(k, {}).get("sources", {}).items()
              if s == "treebank"]
    check("8. NO TEST LANGUAGE CARRIES A TREEBANK-DERIVED BIT  <-- the gate",
          not oracle, " ".join(oracle[:10]))
    unsourced = [f"{k}:{f}" for k in test for f, s in typ.get(k, {}).get("sources", {}).items()
                 if s == "none"]
    check("8b. every test bit has a named source", not unsourced, " ".join(unsourced[:10]))

    # 9. Per-cell budget parity.
    cells = collections.defaultdict(int)
    for k in train:
        cells[L[k]["cell"]] += L[k].get("train_tokens", 0)
    target = man["meta"]["budget_per_cell"]
    off = {c: v for c, v in cells.items() if abs(v - target) / target > a.tolerance}
    named = set(man["meta"].get("cell_shortfall", {}))
    check("9. every cell is within tolerance of the budget, or named as a shortfall",
          not (set(off) - named), " ".join(sorted(set(off) - named)))

    # 10. Collision is DESIRED here: it is what stops the profile being a language identifier.
    codes_seen = ["".join(str(b) for b in typ[k]["bits"]) for k in train]
    uniq = len(set(codes_seen))
    rate = uniq / max(len(codes_seen), 1)
    check("10. train profiles collide (distinctness <= 0.6, i.e. not a language id)", rate <= 0.6,
          f"{uniq}/{len(codes_seen)} = {rate:.2f}")

    # 11. Reachability: a test profile the training side never showed is extrapolation, not
    #     conditioning. Reported as a caveat rather than a failure.
    tr_codes = set(codes_seen)
    def near(code):
        return any(sum(x != y for x, y in zip(code, t)) <= 1 for t in tr_codes)
    unseen = [k for k in test if "".join(str(b) for b in typ[k]["bits"]) not in tr_codes]
    far = [k for k in unseen if not near("".join(str(b) for b in typ[k]["bits"]))]
    print(f"  INFO  11. test profiles unseen in train: {len(unseen)}/{len(test)} "
          f"({' '.join(unseen)}); more than 1 bit away: {' '.join(far) or 'none'}")

    print("\nMODEL")
    models = {}
    cfgs = {arm: f"{a.configs}/config_{arm}.cfg"
            for arm in ("g2_base", "g2_typ", "g2_typ_ctl", "g2_typ_der", "g2_langid")}
    missing_cfg = [c for c in cfgs.values() if not os.path.exists(c)]
    if missing_cfg:
        check("12. configs exist", False, " ".join(missing_cfg))
    else:
        models = {arm: build_embed(p) for arm, p in cfgs.items()}
        for m in models.values():
            m.initialize()
        p_typ, p_ctl, p_der = (n_params(models[k]) for k in ("g2_typ", "g2_typ_ctl", "g2_typ_der"))
        check("12. g2_typ / g2_typ_ctl / g2_typ_der are capacity-matched to the parameter",
              p_typ == p_ctl == p_der, f"{p_typ} / {p_ctl} / {p_der}")
        base_cfg = Config().from_disk(cfgs["g2_typ"], interpolate=False).interpolate()
        width = base_cfg["components"]["tok2vec"]["model"]["encode"]["width"]
        expect = width * 3 * width
        got = p_typ - n_params(models["g2_base"])
        # The typology block adds its Linear as well as the Maxout column, so the difference is the
        # Maxout block plus width*dim + dim.
        lin = width * 8 + width
        check("13. g2_typ exceeds g2_base by exactly the typology block",
              got == expect + lin, f"{got} vs {expect + lin}")

        node = next(n for n in models["g2_typ"].walk() if n.name == "extract_features_feats")
        check("14. the embed reads NO string channel (attrs == ['POS'])",
              node.attrs["columns"] == ["POS"], str(node.attrs["columns"]))

        # 15. The channel is live.
        nlp = spacy.blank("xx")
        d = Doc(nlp.vocab, words=["a", "b", "c"])
        for t in d:
            t.pos_ = "NOUN"
            t.set_morph("Case=Nom")
        d._.tb_lang = train[0]
        y_typ = models["g2_typ"].predict([d])[0]
        y_ctl = models["g2_typ_ctl"].predict([d])[0]
        check("15. the typology channel is live (output differs from its constant control)",
              float(np.abs(np.asarray(y_typ) - np.asarray(y_ctl)).max()) > 1e-6)

        y_der = models["g2_typ_der"].predict([d])[0]
        check("15b. the deranged arm differs from the real one on the same doc",
              float(np.abs(np.asarray(y_typ) - np.asarray(y_der)).max()) > 1e-6)

        # 16. Bit-distinctness of the derangement.
        langs, vecs, dim, _ = load_typology(a.typology)
        der, shift = derange_bits(langs, vecs)
        same = [lg for lg in langs if tuple(der[lg]) == tuple(vecs[lg])]
        check("16. the shuffle is a BIT-DISTINCT derangement, not merely index-distinct",
              not same, f"shift={shift}, kept own profile: {len(same)}")

        # 17-19. The refusals.
        d2 = Doc(nlp.vocab, words=["x"])
        d2[0].pos_ = "NOUN"
        try:
            models["g2_typ"].predict([d2])
            check("17. an unset Doc._.tb_lang raises", False)
        except ValueError:
            check("17. an unset Doc._.tb_lang raises", True)
        d3 = Doc(nlp.vocab, words=["x"])
        d3[0].pos_ = "NOUN"
        d3._.tb_lang = "zzz-not-a-language"
        try:
            models["g2_typ"].predict([d3])
            check("18. a tb_lang with no profile raises", False)
        except ValueError:
            check("18. a tb_lang with no profile raises", True)
        d4 = Doc(nlp.vocab, words=["x"])
        d4[0].pos_ = "NOUN"
        d4._.tb_lang = test[0]
        try:
            models["g2_langid"].predict([d4])
            check("19. lang_id refuses a language outside its inventory", False)
        except ValueError:
            check("19. lang_id refuses a language outside its inventory", True)

    # 20. What the reader puts on the predicted doc.
    from generic_corpus import _example  # noqa: E402
    nlp2 = spacy.blank("xx")
    ref = list(DocBin().from_disk(f"{a.corpus}/{train[0]}-train.spacy").get_docs(nlp2.vocab))[0]
    eg = _example(nlp2, train[0], ref)
    pred = eg.predicted
    check("20. predicted doc carries UPOS", all(t.pos != 0 for t in pred))
    check("20b. predicted doc carries FEATS where the reference has them",
          all(str(p.morph) == str(r.morph) for p, r in zip(pred, ref)))
    check("20c. predicted doc carries NO head/deprel",
          not pred.has_annotation("DEP") and not pred.has_annotation("HEAD"))
    check("20d. tb_lang is stamped on both docs",
          pred._.tb_lang == train[0] and ref._.tb_lang == train[0])

    # 21. Unset vs empty MORPH must land on the same row.
    ue = Doc(nlp2.vocab, words=["alpha", "beta"])
    ue[1].set_morph("")
    for t in ue:
        t.pos_ = "NOUN"
    ue._.tb_lang = train[0]
    if not missing_cfg:
        node = next(n for n in models["g2_typ"].walk() if n.name == "extract_features_feats")
        feats_arr = node.predict([ue])[0]
        check("21. differing morph KEYS (so the trap is real)", ue[0].morph.key != ue[1].morph.key)
        check("21b. unset and empty MORPH land on identical feature columns",
              all(feats_arr[0, c] == feats_arr[1, c] for c in range(1, feats_arr.shape[1])))

    # 22-23.
    bad_freq = []
    for p in sorted(glob.glob(f"{a.configs}/config_g2_*.cfg")):
        c = Config().from_disk(p, interpolate=False)
        if int(c["components"]["parser"]["min_action_freq"]) != 1:
            bad_freq.append(os.path.basename(p))
    check("22. min_action_freq == 1 in every v2 config", not bad_freq, " ".join(bad_freq))
    check("23. baselines have been measured (baseline.json exists)", os.path.exists(a.baseline),
          "run scripts/baseline_generic.py first" if not os.path.exists(a.baseline) else "")

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
