#!/usr/bin/env python3
"""Inventory every SUD 2.18 corpus plus this repo's thirteen local treebanks, for the v2 sampler.

Writes `inventory.json` (one record per corpus that survives) and `excluded.json` (one record per
corpus that does not, WITH ITS REASON). Both are inputs to `prep_generic_v2.py`, which does the
typological balancing; nothing here decides what goes into the corpus, only what is eligible and
what each candidate looks like.

Three things here are load-bearing and easy to get wrong:

⚠ **EXCLUSIONS ARE COUNTED AND NAMED, NEVER SILENT.** A predicate that accidentally matches half the
release would otherwise show up much later as a hole in the typological sample -- a cell with one
language in it reads exactly like a cell that only ever had one language in it. Every drop is
written to `excluded.json` with the rule that dropped it and the value that tripped it, and the
tally is printed.

⚠ **"NO TRAIN SPLIT" DISQUALIFIES A CORPUS FROM THE TRAIN POOL ONLY.** A test-only treebank (the PUD
family, and a good many small ones) is an excellent zero-shot TEST set and v1's blanket rule would
have thrown away exactly the low-resource languages this arm exists for. Eligibility is therefore
two booleans, not one.

⚠ **ONE CORPUS PER LANGUAGE IS DECIDED HERE, BEFORE ANY CELL IS COMPUTED.** If the sampler chose,
it could pick whichever of a language's treebanks lands in the cell it needs filled, which is
choosing the data to fit the hypothesis. The order is fixed and stated: local, then SUD-native, then
largest train, then alphabetical.

The local thirteen keep their `udep` relabelling (`.relabeled_ext`), which is a deliberate decision
recorded in `docs/generic-parser-v2.md`: it puts ~a tenth of their tokens on a different label
policy from the stock test side. `udep_rate` is recorded per corpus precisely so the size of that
split is visible rather than inferred.
"""
import argparse
import collections
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import yaml  # noqa: E402

from prep_generic import SA, SRC, strip_subtype  # noqa: E402

#: UPOS below this and the corpus is unusable INPUT, not a hard language: UPOS is the arm's primary
#: channel and a treebank without it would train the model to read an empty column.
MIN_UPOS_FILL = 0.95

#: `Sign Language` in codes_and_flags.yaml. A glossed stream is not a comparable token sequence.
SIGN_FAMILY = "Sign Language"

#: Likewise a pseudo-family: these corpora are two languages interleaved.
CODESWITCH_FAMILY = "Code switching"

#: SUD 2.18 ships 11 natively-annotated SUD corpora. The count is asserted rather than trusted,
#: because `source` is a tie-break in `pick_one_per_language` and a drift would change the sample.
EXPECT_NATIVE = 11

#: SUD-native corpora in 2.18, by directory name. Taken from the release's own data page. Anything
#: not listed is a UD conversion.
NATIVE = {
    "SUD_French-GSD", "SUD_French-ParisStories", "SUD_French-Rhapsodie", "SUD_French-Sequoia",
    "SUD_Haitian_Creole-Autogramm", "SUD_Hausa-EasternAutogramm", "SUD_Hausa-NorthernAutogramm",
    "SUD_Hausa-SouthernAutogramm", "SUD_Hausa-WesternAutogramm", "SUD_Ika-ChibErgIS",
    "SUD_Naija-NSC",
}


# --------------------------------------------------------------------------------------------
# CoNLL-U scanning


def scan_conllu(path):
    """One streaming pass. Counts words, and separately the MWT/empty rows a word count must skip.

    Deliberately NOT `prep_generic.read_conllu`: that drops MWT ranges and empty nodes before the
    caller can see them, and whether a treebank HAS them is itself a fact worth recording (a
    multiword-token treebank and one that has silently lost its MWTs look identical downstream).
    """
    st = {
        "tokens": 0, "sents": 0, "mwt": 0, "empty": 0,
        "upos": 0, "feats": 0, "lemma": 0,
        "feats_keys": collections.Counter(),
        "deprels": collections.Counter(),
        "upos_counts": collections.Counter(),
    }
    open_sent = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                if open_sent:
                    st["sents"] += 1
                    open_sent = False
                continue
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            if "-" in f[0]:
                st["mwt"] += 1
                continue
            if "." in f[0]:
                st["empty"] += 1
                continue
            open_sent = True
            st["tokens"] += 1
            if f[3] != "_":
                st["upos"] += 1
                st["upos_counts"][f[3]] += 1
            # `_` is a real string in a CoNLL-U column, not a missing value, and spaCy keeps it as
            # one (CLAUDE.md). Counting it as a filled lemma is how te's all-`_` column went
            # unnoticed for a generation.
            if f[2] != "_":
                st["lemma"] += 1
            if f[5] != "_":
                st["feats"] += 1
                for kv in f[5].split("|"):
                    k = kv.split("=", 1)[0]
                    if k:
                        st["feats_keys"][k] += 1
            st["deprels"][strip_subtype(f[7])] += 1
    if open_sent:
        st["sents"] += 1
    return st


def merge(a, b):
    """Sum two scan results. Counters and ints both add with `+`, so there is no special case."""
    return {k: v + b[k] for k, v in a.items()}


# --------------------------------------------------------------------------------------------
# Metadata


def load_codes(path):
    """UD language name -> {family, genus, iso3, lcode}. Multi-word names carry spaces here and
    underscores in directory names, so the caller must translate one into the other."""
    blob = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, m in blob.items():
        if not isinstance(m, dict):
            continue
        out[name] = {
            "family": m.get("family") or "",
            # Isolates and small families carry no `genus`; fall back to the family, then to the
            # language itself, so a genus-disjoint split never silently treats two unrelated
            # languages as relatives by both having an empty genus.
            "genus": m.get("genus") or m.get("family") or name,
            "iso3": m.get("iso3") or "",
            "lcode": m.get("lcode") or "",
        }
    return out


LICENCE_RE = re.compile(r"^\s*License:\s*(.+?)\s*$", re.MULTILINE)

#: `xx-sud-train.conllu`, and also `fr_gsd-sud-train_A.conllu`. SUD_French-GSD chunks its train
#: split across five files; matching only the bare form scored it at zero train tokens while its
#: -dev and -test files kept it looking like a healthy test-only corpus.
SPLIT_RE = re.compile(r"-(train|dev|test)(?:_[A-Za-z0-9]+)?\.conllu$")


def licence_of(d: pathlib.Path) -> str:
    """The README `License:` field first -- it is the machine-readable one -- then LICENSE.txt's
    first substantive line. Both are recorded upstream by UD/SUD and they do sometimes disagree
    (SUD_Telugu-MTG contradicts itself; `stamp_model_meta.py` records that one)."""
    for readme in ("README.md", "README.txt"):
        p = d / readme
        if p.exists():
            m = LICENCE_RE.search(p.read_text(encoding="utf-8", errors="replace"))
            if m:
                return m.group(1)
    p = d / "LICENSE.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                return line.strip()[:120]
    return ""


def is_nc(licence: str) -> bool:
    """Tokenise on non-alphanumerics rather than pattern-matching the string. `CC BY-NC 4.0` and
    `CC BY-NC-SA 3.0` punctuate `nc` differently, and a substring test for `-NC-` silently misses
    the first -- which would understate the corpus's licence union."""
    lo = licence.lower()
    parts = re.split(r"[^a-z0-9]+", lo)
    return "nc" in parts or "noncommercial" in lo.replace("-", "").replace(" ", "")


# --------------------------------------------------------------------------------------------
# Candidate construction


def splits_of(d: pathlib.Path, derived: pathlib.Path):
    """`{split: [paths]}`, from the release directory or, failing that, the carved `derived/` tree.

    Sixteen 2.18 corpora ship one `.conllu` per text with no split at all -- and they are very
    nearly the whole SUD-native set (Hausa, Naija, Haitian Creole, Ika, Beja, Zaar, Gbaya, Pesh,
    Bokota, Nenets). `split_unsplit_sud.py` carves those into `derived/`; without this fallback the
    least Eurasian corpora in the release drop out on a filename convention.
    """
    def collect(where):
        got = {}
        for p in sorted(where.glob("*.conllu")):
            m = SPLIT_RE.search(p.name)
            if m:
                got.setdefault(m.group(1), []).append(p)
        return got

    out = collect(d)
    if out:
        return out
    dd = derived / d.name
    return collect(dd) if dd.is_dir() else {}


def profile(paths_by_split):
    """`({split: stats}, aggregate)`. Refuses an empty input rather than returning `None`, which a
    caller would happily subscript into an AttributeError three frames away."""
    if not paths_by_split:
        raise ValueError("profile() needs at least one split")
    per_split, agg = {}, None
    for split, paths in paths_by_split.items():
        st = None
        for p in paths:
            one = scan_conllu(p)
            st = one if st is None else merge(st, one)
        assert st is not None, f"{split}: splits_of() produced an empty path list"
        per_split[split] = st
        agg = st if agg is None else merge(agg, st)
    assert agg is not None
    return per_split, agg


def record(name, lang_name, kind, paths_by_split, licence, codes, root):
    per_split, agg = profile(paths_by_split)
    # The fill rates are read off TRAIN where there is one, because that is what the model learns
    # from; a test-only corpus is described by its test split. Reading them off the aggregate would
    # let a large train split mask a test split annotated to a different depth.
    ref = per_split.get("train") or per_split.get("test") or per_split.get("dev") or agg
    assert ref is not None
    tok = max(ref["tokens"], 1)
    meta = codes.get(lang_name, {})
    return {
        "name": name,
        "lang_name": lang_name,
        "lcode": meta.get("lcode", ""),
        "iso3": meta.get("iso3", ""),
        "family": meta.get("family", ""),
        "genus": meta.get("genus", ""),
        "source": kind,
        "licence": licence,
        "nc": is_nc(licence),
        "paths": {s: [os.path.relpath(p, root) for p in ps] for s, ps in paths_by_split.items()},
        "tokens": {s: per_split[s]["tokens"] for s in per_split},
        "sents": {s: per_split[s]["sents"] for s in per_split},
        "upos_fill": round(ref["upos"] / tok, 4),
        "feats_fill": round(ref["feats"] / tok, 4),
        "lemma_fill": round(ref["lemma"] / tok, 4),
        "has_mwt": bool(agg["mwt"]),
        "has_empty_nodes": bool(agg["empty"]),
        "feats_keys": dict(ref["feats_keys"].most_common()),
        "upos_counts": dict(ref["upos_counts"].most_common()),
        "deprels": dict(ref["deprels"].most_common()),
        "udep_rate": round(ref["deprels"].get("udep", 0) / tok, 4),
        "comp_obl_rate": round(ref["deprels"].get("comp:obl", 0) / tok, 4),
        "unk_rate": round(ref["deprels"].get("unk", 0) / tok, 4),
    }


def local_candidates(codes, root):
    """This repo's thirteen, from `prep_generic.SRC`/`SA` -- i.e. WITH the `udep` relabelling."""
    #: lcode -> the UD language name, so the local arms join to the same metadata as the release.
    #: From `stamp_model_meta.py`, which is the repo's authority on this. Four of the thirteen are
    #: NonCommercial and they are what the corpus's licence union turns on.
    LICENCE = {"la": "CC BY-NC-SA 4.0", "ar": "CC BY-NC-SA 4.0",
               "ta": "CC BY-NC-SA 3.0", "te": "CC BY-NC-SA 3.0"}
    DEFAULT_LICENCE = "CC BY-SA 4.0"
    NAMES = {
        "en": "English", "zh": "Chinese", "yue": "Cantonese", "lzh": "Classical Chinese",
        "fa": "Persian", "ar": "Arabic", "la": "Latin", "id": "Indonesian", "ko": "Korean",
        "ja": "Japanese", "ta": "Tamil", "te": "Telugu", "sa": "Sanskrit",
    }
    out = []
    for lang, tmpl in SRC.items():
        paths = {}
        for split in ("train", "dev", "test"):
            p = pathlib.Path(SA[split] if lang == "sa" else tmpl % split)
            if p.exists():
                paths[split] = [p]
        if not paths:
            print(f"  !! local {lang}: no files found, skipping", file=sys.stderr)
            continue
        name = f"LOCAL_{NAMES[lang].replace(' ', '_')}"
        out.append(record(name, NAMES[lang], "local", paths,
                          LICENCE.get(lang, DEFAULT_LICENCE), codes, root))
    return out


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sud-dir", default="assets_sud218",
                    help="the extracted SUD release (the directory holding SUD_* subdirectories, "
                         "or its parent -- it is found either way)")
    ap.add_argument("--codes", default="assets_typ/codes_and_flags.yaml")
    ap.add_argument("--derived", default="assets_sud218/derived",
                    help="carved splits for the corpora the release ships unsplit "
                         "(scripts/split_unsplit_sud.py)")
    ap.add_argument("--out", default="assets_sud218/inventory.json")
    ap.add_argument("--excluded", default="assets_sud218/excluded.json")
    ap.add_argument("--min-train-tokens", type=int, default=2000,
                    help="a treebank below this contributes noise; the per-cell "
                         "budget already handles small ones, and v1 trained on te "
                         "at 5 097 tokens")
    ap.add_argument("--min-test-tokens", type=int, default=1000,
                    help="per-language LAS below this is noise; thin rows are "
                         "reported with their token count rather than hidden")
    ap.add_argument("--min-upos-fill", type=float, default=MIN_UPOS_FILL)
    ap.add_argument("--no-local", action="store_true",
                    help="skip this repo's thirteen (they are included by default)")
    a = ap.parse_args()

    root = pathlib.Path(".").resolve()
    codes = load_codes(a.codes)
    print(f"codes_and_flags: {len(codes)} languages")

    base = pathlib.Path(a.sud_dir)
    derived = pathlib.Path(a.derived)
    # `derived/` lives under the release directory, so exclude it from the corpus walk or every
    # carved corpus is discovered twice.
    dirs = sorted(d for d in base.rglob("*SUD_*")
                  if d.is_dir() and list(d.glob("*.conllu")) and derived not in d.parents)
    if not dirs:
        sys.exit(f"no SUD_* corpus directories with .conllu under {base} -- run "
                 f"scripts/fetch_sud_release.sh first")
    print(f"release corpora with CoNLL-U: {len(dirs)}")

    cands, excluded = [], []

    def drop(name, rule, value, extra=None):
        excluded.append({"name": name, "rule": rule, "value": value, **(extra or {})})

    n_native = 0
    for d in dirs:
        name = d.name
        # Rule 1: segmentation regime. mSUD is morpheme-level and pSUD prosodic; neither is a
        # comparable token stream, and mixing one in would teach the parser a different notion of
        # what a word is.
        if name.startswith("mSUD_") or name.startswith("pSUD_"):
            drop(name, "segmentation-regime", name.split("_")[0])
            continue
        lang_name = name.split("-", 1)[0].removeprefix("SUD_").replace("_", " ")
        meta = codes.get(lang_name)
        if meta is None:
            drop(name, "no-metadata", lang_name)
            continue
        # Rule 2: sign languages are glossed, not tokenised text.
        if meta["family"] == SIGN_FAMILY:
            drop(name, "sign-language", meta["family"])
            continue
        # Rule 2b: a code-switched corpus is two languages at once. It carries a pseudo ISO code
        # (qfn, qaf, qtd), no database has a typological profile for it, and "the word order of
        # Turkish-German" is not a question this encoding can answer.
        if meta["family"] == CODESWITCH_FAMILY:
            drop(name, "code-switching", meta["family"])
            continue
        paths = splits_of(d, derived)
        if not paths:
            drop(name, "no-conllu-splits", "")
            continue
        kind = "sud-native" if name in NATIVE else "ud-converted"
        n_native += kind == "sud-native"
        cands.append(record(name, lang_name, kind, paths, licence_of(d), codes, root))

    if n_native != EXPECT_NATIVE:
        print(f"!! expected {EXPECT_NATIVE} SUD-native corpora, found {n_native}. The NATIVE set in "
              f"this file is stale; `source` is a tie-break in pick_one_per_language, so fix it "
              f"before sampling.", file=sys.stderr)

    if not a.no_local:
        loc = local_candidates(codes, root)
        print(f"local treebanks: {len(loc)}")
        cands += loc

    # Rule 3: UPOS is the primary input channel.
    keep = []
    for c in cands:
        if c["upos_fill"] < a.min_upos_fill:
            drop(c["name"], "upos-fill", c["upos_fill"])
            continue
        keep.append(c)
    cands = keep

    # Rules 4 and 5: eligibility is TWO booleans. A corpus with no train split is still a fine
    # zero-shot test set, and refusing it would discard exactly the low-resource languages this
    # experiment exists to serve.
    for c in cands:
        tr = c["tokens"].get("train", 0)
        te = c["tokens"].get("test", 0)
        c["train_eligible"] = tr >= a.min_train_tokens
        c["test_eligible"] = te >= a.min_test_tokens
        c["eligible_reason"] = {
            "train": "ok" if c["train_eligible"] else (
                "no-train-split" if tr == 0 else f"train-tokens<{a.min_train_tokens}"),
            "test": "ok" if c["test_eligible"] else (
                "no-test-split" if te == 0 else f"test-tokens<{a.min_test_tokens}"),
        }
    for c in [c for c in cands if not c["train_eligible"] and not c["test_eligible"]]:
        drop(c["name"], "too-small-for-either-pool",
             {"train": c["tokens"].get("train", 0), "test": c["tokens"].get("test", 0)})
    cands = [c for c in cands if c["train_eligible"] or c["test_eligible"]]

    # Rule 6: one corpus per language, decided HERE so the sampler cannot choose the treebank that
    # lands in the cell it wants filled.
    by_lang = collections.defaultdict(list)
    for c in cands:
        by_lang[c["lcode"] or c["lang_name"]].append(c)
    rank = {"local": 0, "sud-native": 1, "ud-converted": 2}
    chosen = []
    for lang, group in sorted(by_lang.items()):
        group.sort(key=lambda c: (rank[c["source"]], -c["tokens"].get("train", 0), c["name"]))
        chosen.append(group[0])
        for c in group[1:]:
            drop(c["name"], "not-the-chosen-corpus-for-language",
                 {"lang": lang, "chosen": group[0]["name"], "chosen_source": group[0]["source"]})

    chosen.sort(key=lambda c: c["name"])
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"sud_dir": str(base), "min_train_tokens": a.min_train_tokens,
                        "min_test_tokens": a.min_test_tokens,
                        "min_upos_fill": a.min_upos_fill,
                        "native_found": n_native, "native_expected": EXPECT_NATIVE},
               "corpora": chosen},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(excluded, open(a.excluded, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    tally = collections.Counter(e["rule"] for e in excluded)
    print(f"\nkept {len(chosen)} corpora ({len(by_lang)} languages) -> {a.out}")
    print(f"excluded {len(excluded)} -> {a.excluded}")
    for rule, n in tally.most_common():
        print(f"  {rule:38s} {n:4d}")
    tr = [c for c in chosen if c["train_eligible"]]
    te = [c for c in chosen if c["test_eligible"]]
    print(f"\ntrain-eligible {len(tr)}  ({sum(c['tokens'].get('train', 0) for c in tr):,} tokens)")
    print(f"test-eligible  {len(te)}  ({sum(c['tokens'].get('test', 0) for c in te):,} tokens)")
    print(f"sud-native {sum(c['source'] == 'sud-native' for c in chosen)}  "
          f"local {sum(c['source'] == 'local' for c in chosen)}  "
          f"NC-licensed {sum(c['nc'] for c in chosen)}")


if __name__ == "__main__":
    main()
