#!/usr/bin/env python3
"""Build the typologically balanced train corpus and the disjoint zero-shot test corpus.

Replaces v1's hand-written seven `GROUPS` with cells derived from the four typological features, and
replaces "hold one language out of thirteen" with a SEPARATE sample of languages the training set
never sees. Everything else -- the CoNLL-U normalisation, block sampling and the equal-share budget
with intra-group redistribution -- is `prep_generic.py`'s, imported rather than copied.

A **cell** is the first three fields only: OV-axis x SV-axis x marking-axis, at their natural
cardinalities ({OV, VO, both} x {SV, VS, both} x {head, dep, double, neither/unknown}) = 36 nominal
cells. Gender is a secondary stratifier WITHIN a cell rather than a fourth axis: the full
cross-product is 256 cells against ~50 training languages, and gender is the field the databases
cover worst, so making it an axis would mostly manufacture empty cells.

⚠ **A LANGUAGE IS CELLED BY THE PROFILE IT WILL ACTUALLY BE CONDITIONED ON.** Training languages read
their profile off their own treebank; test languages read theirs off Grambank/WALS. Those two
sources agree on only 52-71 % of fields (`compare_typology.py`), so celling everything by one of them
would misplace roughly a third of the other pool. Test eligibility therefore requires a COMPLETE
external profile, and that profile is what cells the language.

⚠ **THE SPLIT IS GENUS-DISJOINT, AND GENUS IS A GLOBAL CONSTRAINT.** A genus goes entirely to one
side, so the assignment is made over genera and languages follow. Pinning this repo's thirteen to
train locks ten genera (Germanic, Iranian, Italic, Indic, Chinese, Japanese, Korean, Semitic,
Malayo-Sumbawan, Dravidian) out of the test pool -- which is the point, but it is printed rather
than left for someone to discover in the results.

⚠ **A CELL THAT EMPTIES IS REPORTED, NOT DROPPED.** "Even sampling" is a target the inventory may not
permit; the manifest records what was achievable, per cell, on both sides.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import prep_generic as pg  # noqa: E402
from prep_generic import allocate, normalise, read_conllu, sample_blocks, write_conllu  # noqa: E402

FIELDS = ["OV", "VO", "SV", "VS", "HM", "DM", "SEX", "NOSEX"]

#: Beyond v1's `Shared`. `NameType` (Giv/Sur/Geo/Nat) is the important addition and it is LEAKAGE
#: rather than clutter: it tells the parser that a token is a personal name, which is information
#: about the WORDFORM in an arm whose entire claim is that it reads no wordform. It is on 39 000
#: tokens of lzh alone. The rest are annotation practice, not morphology.
EXTRA_LEAKY = {"NameType", "Typo", "Style", "Foreign", "Abbr", "Hyph"}

#: Genera that are historically continuous and must move to the same side of the split, even though
#: the UD taxonomy names them separately.
#:
#: ⚠ Without this, Latin (genus `Italic`, pinned to TRAIN) sits happily alongside eight Romance test
#: languages -- Spanish, French, Portuguese, Catalan, Italian, Galician, Occitan, Sicilian -- and
#: genus-disjointness passes on a technicality while the test set is a third full of Latin's own
#: daughters. "Zero-shot" then measures transfer from an ancestor, which is precisely the claim it
#: is supposed to exclude.
GENUS_KIN = {
    "Italic": {"Romance"},
    "Romance": {"Italic"},
}


#: The SUD relations whose `:` component is CORE, not a subtype. Everything else collapses to the
#: part before the colon.
#:
#: ⚠ v1 stripped only `@` subtypes and got 27 labels out of thirteen treebanks, on the reasoning
#: that `:` is core SUD. Across ~100 treebanks that is no longer true: the release carries
#: `mod:periph$cond`, `comp:obj$utter`, `compound:svc$purp`, `dislocated:obj`, `parataxis:parenth`
#: and a long tail of others, ~100 labels in all, most of them attested in a single treebank and
#: some on a single token. With `min_action_freq = 1` every one becomes a parser action, and a
#: label a test language uses but training never saw is unreachable by construction.
CORE_SUB = {"comp:obj", "comp:obl", "comp:aux", "comp:pred", "comp:cleft",
            "conj:coord", "conj:appos", "conj:dicto"}

#: UD relations a few converted treebanks keep where SUD has its own name. `appos` is the only one
#: that mattered -- it was the single label appearing in test and never in training, i.e.
#: unreachable by construction -- and the repo already treats appos as `conj:appos`
#: (docs, `sud-relation-conformance`). `aux` and bare `conj` are left alone: those treebanks are
#: genuinely underspecified rather than differently named, and merging them would be a guess.
RENAME = {"appos": "conj:appos"}

#: The seventeen. A treebank occasionally coins its own -- SUD_Hausa writes `IDEO` for ideophones --
#: and spaCy refuses the whole file with E1021 rather than the token, so one tag in 1.88 M costs a
#: whole split. UPOS is this arm's primary channel, so an out-of-inventory tag is genuinely unknown
#: to it: map to `X`, which is what UD's own "other" category is for, and report the count.
UD_UPOS = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
           "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}
NON_UD_UPOS = collections.Counter()


def coarsen_deprel(dep: str) -> str:
    """`mod@relcl` -> `mod`, `comp:obj$utter` -> `comp:obj`, `mod:periph$cond` -> `mod`.

    Order matters: `@`, `$` and `/` markers come off first, then a `:` component is kept only if the
    resulting relation is one SUD actually defines.
    """
    d = dep.split("@", 1)[0].split("$", 1)[0].split("/", 1)[0]
    if d in CORE_SUB:
        return d
    d = d.split(":", 1)[0]
    return RENAME.get(d, d)


def normalise_v2(sent):
    """`prep_generic.normalise`, plus the UPOS inventory guard."""
    out = normalise(sent)
    for f in out.rows:
        if f[3] not in UD_UPOS:
            NON_UD_UPOS[f[3]] += 1
            f[3] = "X"
    return out


def cell_of(bits):
    """`(ov, sv, mark)` as short strings. Gender is deliberately not an axis."""
    ov = {(1, 0): "OV", (0, 1): "VO", (1, 1): "OV+VO", (0, 0): "?"}[(bits[0], bits[1])]
    sv = {(1, 0): "SV", (0, 1): "VS", (1, 1): "SV+VS", (0, 0): "?"}[(bits[2], bits[3])]
    mk = {(1, 0): "head", (0, 1): "dep", (1, 1): "double", (0, 0): "?"}[(bits[4], bits[5])]
    return f"{ov}|{sv}|{mk}"


def split_languages(langs, cells, genus, pinned, test_ok, want_test, verbose=True):
    """Assign whole GENERA to train or test so that no test language has a relative in training.

    Greedy, hardest cell first: a cell with two genera has exactly one way to be represented on both
    sides, and spending one of its genera on some easier cell would empty it.
    """
    by_genus = collections.defaultdict(list)
    for lg in langs:
        by_genus[genus[lg]].append(lg)

    locked = {genus[lg] for lg in pinned}
    kin = {k for g in locked for k in GENUS_KIN.get(g, ())} - locked
    locked |= kin
    if verbose:
        print(f"pinned to train: {len(pinned)} languages, locking {len(locked)} genera")
        print("  " + " ".join(sorted(locked)))
        if kin:
            print(f"  (of which {' '.join(sorted(kin))} locked as historically continuous with a "
                  f"pinned genus, not because a pinned language is in it)")

    # A genus is eligible for the test side if ANY of its languages can be tested. The ones that
    # cannot are then DROPPED rather than left in training -- see the return value below.
    free = {g: ls for g, ls in by_genus.items()
            if g not in locked and any(lg in test_ok for lg in ls)}

    cell_genera = collections.defaultdict(set)
    for lg in langs:
        if genus[lg] in free and lg in test_ok:
            cell_genera[cells[lg]].add(genus[lg])

    test_genera = set()
    n_test = 0
    # Round-based, hardest cell first. One pass takes at most one genus per cell, so a second pass
    # is what gets the test set past a dozen languages; each pass still leaves a genus on the train
    # side of every cell it touches, so no cell is emptied to feed the target.
    while n_test < want_test:
        took = False
        for cell in sorted(cell_genera, key=lambda c: (len(cell_genera[c]), c)):
            if n_test >= want_test:
                break
            remaining = cell_genera[cell] - test_genera
            if len(remaining) < 2:
                continue                      # taking one would leave this cell with no train side
            # Prefer the genus that costs training the least, so balance is bought cheaply.
            pick = min(sorted(remaining), key=lambda g: (len(free[g]), g))
            test_genera.add(pick)
            n_test += sum(1 for lg in free[pick] if lg in test_ok)
            took = True
        if not took:
            break

    test = {lg for g in test_genera for lg in by_genus[g] if lg in test_ok}
    # ⚠ Every OTHER language of a test genus has to leave the training set, not merely fail to join
    # the test set. Keeping them is how Western Armenian ended up in training while Eastern and
    # Classical Armenian were being scored as zero-shot -- the genus check passed on the test
    # languages and said nothing about their relatives on the other side.
    dropped = {lg for g in test_genera for lg in by_genus[g] if lg not in test_ok}
    train = {lg for lg in langs if lg not in test and lg not in dropped}
    if verbose and dropped:
        print(f"dropped from BOTH pools ({len(dropped)}): relatives of a test language that are "
              f"not themselves testable -- " + " ".join(sorted(dropped)))
    return train, test, test_genera, locked, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="assets_sud218/inventory.json")
    ap.add_argument("--treebank-typ", default="assets_typ/typology_treebank.json")
    ap.add_argument("--external-typ", default="assets_typ/typology_external.json")
    ap.add_argument("--out", default="assets_generic_v2")
    ap.add_argument("--typology-out", default="assets_typ/typology_v2.json")
    ap.add_argument("--budget", type=int, default=100000,
                    help="TRAIN tokens per typological cell (default 100000)")
    ap.add_argument("--max-lang-tokens", type=int, default=40000,
                    help="ceiling on any ONE language's contribution (default 40000)")
    ap.add_argument("--min-cell-tokens", type=int, default=20000,
                    help="a cell this thin is exempted from the family ceiling rather than "
                         "shrunk further (default 20000). 0 disables the protection.")
    ap.add_argument("--family-ceiling", type=float, default=0.40,
                    help="no FAMILY may exceed this share of train tokens (default 0.40, which is "
                         "the test set's own Indo-European share). 1.0 disables it.")
    ap.add_argument("--dev-budget", type=int, default=1000,
                    help="DEV tokens per training language. Dev is re-scored every eval_frequency "
                         "steps, so with ~50 languages this is what sets wall-clock, not train size")
    ap.add_argument("--test-cap", type=int, default=20000, help="TEST tokens per test language")
    ap.add_argument("--want-test", type=int, default=20, help="target number of test languages")
    ap.add_argument("--block", type=int, default=10,
                    help="sentences per sampling block; must match `spacy convert -n`")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pin-local", action="store_true",
                    help="do not force this repo's thirteen onto the train side")
    a = ap.parse_args()

    # The extended leak list has to be installed before `normalise` is called, because
    # `clean_feats` reads the module-level set.
    pg.LEAKY_FEATS = set(pg.LEAKY_FEATS) | EXTRA_LEAKY
    # `normalise` reads both of these off the module, so they must be installed before it runs.
    pg.strip_subtype = coarsen_deprel
    pg.normalise = normalise_v2
    print(f"leaky FEATS removed: {' '.join(sorted(pg.LEAKY_FEATS))}")

    inv = json.loads(pathlib.Path(a.inventory).read_text(encoding="utf-8"))["corpora"]
    tb = json.loads(pathlib.Path(a.treebank_typ).read_text(encoding="utf-8"))["languages"]
    ex = json.loads(pathlib.Path(a.external_typ).read_text(encoding="utf-8"))["languages"]

    rec = {c["lcode"] or c["lang_name"]: c for c in inv}
    genus = {k: c["genus"] for k, c in rec.items()}
    pinned = set() if a.no_pin_local else {k for k, c in rec.items() if c["source"] == "local"}

    # Test eligibility: enough test tokens AND a COMPLETE external profile. The second half is what
    # makes the zero-shot claim clean -- a test language with a gap would have to be filled from its
    # own treebank, i.e. from the data being tested on.
    def full_external(k):
        e = ex.get(k)
        return bool(e) and all(v != "none" for v in e["sources"].values())

    test_ok = {k for k, c in rec.items() if c["test_eligible"] and full_external(k)}
    train_ok = {k for k, c in rec.items() if c["train_eligible"]}
    langs = sorted(train_ok | test_ok)
    print(f"{len(langs)} candidate languages: {len(train_ok)} train-eligible, "
          f"{len(test_ok)} test-eligible with a complete external profile")

    # Cell by the profile the language would be conditioned on in the pool it can join.
    cells, basis = {}, {}
    for k in langs:
        if k in test_ok:
            cells[k], basis[k] = cell_of(ex[k]["bits"]), "external"
        elif k in tb:
            cells[k], basis[k] = cell_of(tb[k]["bits"]), "treebank"
        else:
            cells[k], basis[k] = "?|?|?", "none"

    train, test, test_genera, locked, dropped = split_languages(
        langs, cells, genus, pinned, test_ok, a.want_test)
    train = {k for k in train if k in train_ok}
    print(f"\nsplit: {len(train)} train, {len(test)} test "
          f"({len(test_genera)} test genera: {' '.join(sorted(test_genera))})")

    # Re-cell each language by the profile it will ACTUALLY carry, and build the final table.
    typ, final_cell = {}, {}
    for k in sorted(train | test):
        if k in test:
            src = ex[k]
            typ[k] = {"bits": src["bits"], "sources": src["sources"], "pool": "test",
                      "class_system": src.get("class_system", "unknown"),
                      "glottocode": src.get("glottocode"), "join": src.get("join")}
        else:
            src = tb.get(k)
            if src is None:
                print(f"  !! {k}: no treebank profile, dropping from train", file=sys.stderr)
                continue
            typ[k] = {"bits": src["bits"], "sources": src["sources"], "pool": "train",
                      "class_system": src.get("class_system", "unknown")}
        final_cell[k] = cell_of(typ[k]["bits"])
    train = {k for k in train if k in typ}

    # --- budget, per cell -------------------------------------------------------------------
    cell_langs = collections.defaultdict(list)
    for k in train:
        cell_langs[final_cell[k]].append(k)
    # ⚠ The per-cell budget neutralises CORPUS SIZE -- German contributes the same as Chinese --
    # but not CELL SPARSITY. Seven languages alone in their cell held 26.6 % of all training tokens
    # at 100 k each, and a configuration attested by one treebank then teaches the model that
    # treebank's idiosyncrasies as the cell's signature. The ceiling caps that without touching
    # cells of three or more languages, where the equal share is already below it.
    family = {k: rec[k]["family"] for k in train}
    caps = {k: min(rec[k]["tokens"].get("train", 0), a.max_lang_tokens) for k in train}

    def allocate_all(caps):
        got = {}
        for cell, ls in cell_langs.items():
            got.update(allocate(a.budget, {k: caps[k] for k in ls}))
        return got

    # ⚠ THE PER-CELL BUDGET BALANCES TYPOLOGY, NOT GENEALOGY, AND THE TWO COME APART HERE.
    # Indo-European is 55 of the 97 train-eligible languages in SUD 2.18, and it populates several
    # of the sparser cells, so equal-per-cell weighting pushed it to 68.3 % of train tokens against
    # 39.9 % of the TEST tokens. A model whose prior is two-thirds one family is not the
    # language-agnostic parser this arm claims to be. The ceiling scales an over-represented
    # family's per-language caps down and re-allocates; `allocate` then hands the freed budget to
    # the other languages in the same cell, which are by construction from other families.
    base_caps = dict(caps)
    alloc = allocate_all(caps)
    protected: set = set()
    if a.family_ceiling < 1.0:
        for _ in range(400):
            # ⚠ PROTECT THIN CELLS FIRST. Six cells are a single Indo-European language -- Manx,
            # Faroese, Latin, Swedish, Norwegian, Dutch -- so an unguarded family ceiling cuts
            # exactly the configurations that have only one representative, and three of those
            # cells have a TEST language scored against them. Protection is sticky and only ever
            # restores a cap, so the loop cannot oscillate. It costs almost nothing: it moves the
            # IE share by half a point and recovers ~200 k tokens, because the freed budget lands
            # in cells that were starving.
            thin = {k for cell, ls in cell_langs.items()
                    if sum(alloc[k] for k in ls) < a.min_cell_tokens for k in ls}
            fresh = thin - protected
            if fresh:
                protected |= fresh
                for k in fresh:
                    # PARTIAL protection: enough to hold the cell at the floor, not the full cap.
                    # Exempting a protected language outright is what produced the pathology this
                    # replaced -- twelve protected IE languages claimed 480 k against a family
                    # budget of 417 k, so the ceiling took the entire shortfall out of the other
                    # thirty-five and crushed English to 263 tokens and Hindi to 203.
                    share = a.min_cell_tokens // max(len(cell_langs[final_cell[k]]), 1)
                    caps[k] = min(base_caps[k], max(share, caps[k]))
                alloc = allocate_all(caps)
                continue
            tot = sum(alloc.values()) or 1
            ft = collections.Counter()
            for k, v in alloc.items():
                ft[family[k]] += v
            over = {f: t for f, t in ft.items() if t / tot > a.family_ceiling + 0.002}
            if not over:
                break
            moved = False
            for f, t in over.items():
                targets = [k for k in train if family[k] == f and k not in protected]
                if not targets:
                    continue          # the whole family is protecting thin cells; nothing to give
                moved = True
                # WATER-FILLING, not multiplicative scaling. One common ceiling for the family's
                # unprotected members, found by bisection, so everyone is levelled to the same cap
                # and the small ones keep what they have. Scaling every cap by the same FACTOR
                # instead drives the tail towards the floor while the head stays large, which is
                # how a 47-language family came to be represented by twelve of its members.
                held = sum(alloc[k] for k in train
                           if family[k] == f and k in protected)
                need = max(0.0, a.family_ceiling * tot - held)
                lo, hi = 0.0, float(max(base_caps[k] for k in targets))
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if sum(min(base_caps[k], mid) for k in targets) > need:
                        hi = mid
                    else:
                        lo = mid
                for k in targets:
                    caps[k] = max(200, int(min(base_caps[k], lo)))
            if not moved:
                break
            alloc = allocate_all(caps)

    shortfall = {}
    for cell, ls in cell_langs.items():
        have = sum(alloc[k] for k in ls)
        if have < a.budget:
            shortfall[cell] = {"wanted": a.budget, "got": have, "langs": sorted(ls)}

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(out.glob("*.conllu"))
    if stale:
        # A run with a different split leaves files behind, and a leftover `<lang>-train.conllu`
        # for a language that is now on the TEST side silently un-holds-it-out -- the corpus reader
        # discovers languages by which files exist. Clear rather than overwrite.
        print(f"clearing {len(stale)} CoNLL-U files from a previous run in {out}")
        for f in stale:
            f.unlink()
    manifest = {"meta": {"budget_per_cell": a.budget,
                         "max_lang_tokens": a.max_lang_tokens,
                         "family_ceiling": a.family_ceiling,
                         "min_cell_tokens": a.min_cell_tokens,
                         "protected_from_ceiling": sorted(protected),
                         "dev_budget": a.dev_budget,
                         "test_cap": a.test_cap, "block": a.block, "seed": a.seed,
                         "leaky_feats_removed": sorted(pg.LEAKY_FEATS),
                         "locked_genera": sorted(locked),
                         "test_genera": sorted(test_genera)},
                "languages": {}}

    def emit(k, split, paths, budget):
        sents = [s for p in paths for s in read_conllu(p)]
        sents = [normalise_v2(s) for s in sents]
        got, sub = sample_blocks(sents, budget, a.block, pg.lang_seed(a.seed, k))
        write_conllu(out / f"{k}-{split}.conllu", got)
        return sum(len(s) for s in got), len(got), sub

    print("\nwriting corpora")
    for k in sorted(train | test):
        c = rec[k]
        info = {"pool": typ[k]["pool"], "cell": final_cell[k], "basis": basis.get(k),
                "genus": genus[k], "family": c["family"], "source": c["source"],
                "licence": c["licence"], "nc": c["nc"], "corpus": c["name"],
                "feats_fill": c["feats_fill"], "udep_rate": c["udep_rate"],
                "bits": typ[k]["bits"]}
        if k in train:
            for split, budget in (("train", alloc[k]), ("dev", a.dev_budget)):
                paths = c["paths"].get(split)
                if not paths and split == "dev":
                    # No dev of its own: carve one from the tail of train. The alternative is a
                    # training language that never contributes to model selection.
                    paths = c["paths"].get("train")
                if not paths:
                    continue
                tok, ns, sub = emit(k, split, paths, budget)
                info[f"{split}_tokens"], info[f"{split}_sents"] = tok, ns
                info[f"{split}_subsampled"] = sub
        else:
            paths = c["paths"].get("test")
            tok, ns, sub = emit(k, "test", paths, a.test_cap)
            info["test_tokens"], info["test_sents"], info["test_subsampled"] = tok, ns, sub
        manifest["languages"][k] = info

    tr_tok = sum(v.get("train_tokens", 0) for v in manifest["languages"].values())
    te_tok = sum(v.get("test_tokens", 0) for v in manifest["languages"].values())
    relabelled = sum(v.get("train_tokens", 0) for v in manifest["languages"].values()
                     if v["source"] == "local")
    manifest["meta"].update({
        "train_tokens": tr_tok, "test_tokens": te_tok,
        "n_train_langs": len(train), "n_test_langs": len(test),
        # The relabelling puts these tokens on a different `udep` policy from every test treebank.
        # Recorded as a SHARE OF TOKENS, not of treebanks, because that is the size of the skew.
        "relabelled_train_tokens": relabelled,
        "relabelled_share": round(relabelled / max(tr_tok, 1), 4),
        "cell_shortfall": shortfall,
    })
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    json.dump({"meta": {"fields": FIELDS,
                        "note": "train bits are treebank-derived, test bits are Grambank/WALS. "
                                "No test bit may have source 'treebank'."},
               "languages": typ},
              open(a.typology_out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # --- the cell-occupancy table, which is a result rather than bookkeeping ------------------
    print(f"\ntrain {tr_tok:,} tokens over {len(train)} languages; "
          f"test {te_tok:,} over {len(test)}")
    print(f"relabelled (local) share of train tokens: {manifest['meta']['relabelled_share']:.1%}")
    # Family shares on BOTH sides, side by side: the training prior is what the ceiling exists to
    # control, and it is only meaningful against the distribution it will be tested on.
    ftr, fte = collections.Counter(), collections.Counter()
    ltr, lte = collections.Counter(), collections.Counter()
    for k, v in manifest["languages"].items():
        if v["pool"] == "train":
            ftr[v["family"]] += v.get("train_tokens", 0)
            ltr[v["family"]] += 1
        else:
            fte[v["family"]] += v.get("test_tokens", 0)
            lte[v["family"]] += 1
    ttr, tte = max(sum(ftr.values()), 1), max(sum(fte.values()), 1)
    print(f"\n{'family':20s} {'train tok':>9s} {'train lg':>9s}   {'test tok':>9s} {'test lg':>8s}")
    for f in sorted(set(ftr) | set(fte), key=lambda f: -ftr[f]):
        print(f"  {f:18s} {100 * ftr[f] / ttr:8.1f}% {ltr[f]:9d}   "
              f"{100 * fte[f] / tte:8.1f}% {lte[f]:8d}")
    print("\ncell                        train langs / tokens        test langs")
    allcells = sorted(set(final_cell.values()))
    for cell in allcells:
        trl = sorted(k for k in train if final_cell[k] == cell)
        tel = sorted(k for k in test if final_cell[k] == cell)
        tok = sum(manifest["languages"][k].get("train_tokens", 0) for k in trl)
        if trl and not tel:
            flag = "  <- no test side"
        elif tel and not trl:
            flag = "  <- NO TRAIN SIDE"
        else:
            flag = ""
        print(f"  {cell:26s} {len(trl):3d} / {tok:8,d}   {len(tel):3d}  "
              f"{' '.join(tel)[:34]}{flag}")
    if protected:
        print(f"\n{len(protected)} languages exempted from the family ceiling to keep their cell "
              f"above {a.min_cell_tokens:,} tokens:\n  " + " ".join(sorted(protected)))
    if shortfall:
        print(f"\n{len(shortfall)} cells could not reach the budget:")
        for cell, s in sorted(shortfall.items()):
            print(f"  {cell:26s} {s['got']:,} of {s['wanted']:,}  ({' '.join(s['langs'])})")
    if NON_UD_UPOS:
        print(f"\nnon-UD UPOS mapped to X: {sum(NON_UD_UPOS.values())} tokens  "
              + " ".join(f"{k}={v}" for k, v in NON_UD_UPOS.most_common()))
    print(f"\nwrote {out}/manifest.json and {a.typology_out}")


if __name__ == "__main__":
    main()
