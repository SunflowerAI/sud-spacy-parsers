#!/usr/bin/env python3
"""Add the `ar_vocalise` / `fa_vocalise` pipe to a built Arabic or Persian arm.

Surgery rather than a retrain, for the same reason `add_la_macronise.py` is: nothing in the
pipeline READS the vocalisation, so no weight moves and every published Arabic figure stands.
`--verify` re-checks that out of the RELOADED model rather than the in-memory one -- the lesson
`add_la_enclitic_tokenizer.py` paid for, where assigning a component left the config untouched and
the reloaded model quietly rebuilt the stock one.

The pipe goes LAST. It reads `token.pos_` and `token.morph`, so it must follow the morphologiser;
last is where it can neither disturb the `sud_*` pipes nor be disturbed by them, since it only
writes `token._.vocalised` and nothing downstream reads that.

⚠ THE RELEASE SHIPS IT BARE (`--no-lut`), exactly as la ships `la_macronise`. The table is
extracted from SUD_Arabic-PADT (CC BY-NC-SA 3.0) and the ar wheel does not declare NC, so the data
cannot travel with it -- but the COMPONENT can, and it starts vocalising the moment the user
builds a table with `build_ar_vocalise_lut.py`. Until then it passes every token through unchanged
and warns once, naming both routes.

⚠ ar and fa are the SAME surgery but NOT the same proposition. ar's table is gold, harvested
from PADT's `Vform`, and the pipeline's morphology disambiguates it; fa's is reconstructed from a
one-pronunciation-per-word lexicon and cannot be scored, because no vocalised Persian gold exists
here. See each component's own docstring before quoting anything about either.

    python scripts/add_vocalise.py --lang ar in_model out_model --no-lut --verify --code ...
"""
import argparse
import gzip
import importlib.util
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_code(path, required=True):
    """Execute one component module so its @Language.factory registers.

    Every factory the INPUT model's config names must be registered before spacy.load, or it fails
    with E002 -- so this runs over the model's other components too, not just ours. In the release
    pipeline this script is handed a model already carrying sud_misc/sud_idiom/sud_tagger, and
    loading only ar_vocalise.py would make attaching the vocaliser the step that cannot open it."""
    path = str(path)
    if "/" not in path:
        path = str(_HERE / path)
    name = path.split("/")[-1][:-3]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except FileNotFoundError:
        if required:
            raise
    except Exception:
        if required:
            raise


def _size(comp):
    """Entry count, whichever component this is: ar carries three rungs, fa one."""
    if hasattr(comp, "forms"):
        return len(comp.forms)
    return len(comp.l1) + len(comp.l2) + len(comp.l3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("ar", "fa"), default="ar")
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--lut", default=None,
                    help="default: scripts/<lang>_vocalise_lut.json.gz")
    ap.add_argument("--no-lut", action="store_true",
                    help="attach the pipe with no table (how the released wheel is built)")
    ap.add_argument("--camel", default="true",
                    help="true/false: enable the calima-msa analyser fall-through (never bundled)")
    ap.add_argument("--code", default="",
                    help="comma-separated sibling modules whose factories the INPUT model needs "
                         "(e.g. sud_misc.py,sud_idiom.py); missing ones are skipped, not fatal")
    ap.add_argument("--verify", action="store_true",
                    help="reload from disk and confirm no weight file moved")
    a = ap.parse_args()
    pipe = f"{a.lang}_vocalise"
    lut = a.lut or str(_HERE / f"{a.lang}_vocalise_lut.json.gz")

    load_code(_HERE / f"{pipe}.py")
    load_code(_HERE / "ar_tokenizer.py", required=False)
    for extra in filter(None, (c.strip() for c in a.code.split(","))):
        load_code(extra, required=False)

    import spacy
    nlp = spacy.load(a.in_model)
    if pipe in nlp.pipe_names:               # replace, so a rebuild picks up new code/table
        nlp.remove_pipe(pipe)
    # Add with lut=None so the SAVED config carries no build-time path, then populate the table
    # directly. Writing nlp.config[...] afterwards does NOT work -- spaCy regenerates the component
    # block from the factory's own config -- and a shipped config naming
    # "scripts/ar_vocalise_lut.json.gz" would send the installed wheel looking for a file that is
    # not there. The table is serialised into the model directory by the component's to_disk.
    cfg = {"lut": None, "ezafe": None} if a.lang == "fa" else {"lut": None}
    if a.lang == "ar":
        cfg["camel"] = a.camel.lower() != "false"
    nlp.add_pipe(pipe, config=cfg, last=True)
    comp = nlp.get_pipe(pipe)
    if not a.no_lut:
        p = Path(lut)
        if not p.exists():
            raise SystemExit(
                f"no table at {p} -- run build_{a.lang}_vocalise_lut.py, or pass --no-lut")
        comp._load_blob(json.loads(gzip.open(p, "rb").read().decode("utf-8")))
    if a.lang == "fa":
        # OUTSIDE the --no-lut branch on purpose. The two fa data files have different provenance
        # and therefore different fates: the LEXICON is KaamelDict (GPL) plus Tihu and can never
        # travel in a CC BY-SA wheel, but the EZAFE RULES are derived from SUD_Persian-PerDT, which
        # is CC BY-SA 4.0 -- the wheel's own licence and its own training data. So the released
        # wheel ships bare of the lexicon and still inserts ezafe out of the box.
        # Loaded onto the component rather than named in the config, for the same reason the table
        # is: a shipped config naming a build-time path sends the installed wheel looking for a
        # file that is not there. to_disk writes it into the model directory.
        ez = _HERE / "fa_ezafe_rules.json"
        if ez.exists():
            comp.ezafe = json.loads(ez.read_text(encoding="utf-8"))
            print(f"  ezafe cells: {len(comp.ezafe)}")

    out = Path(a.out_model)
    if out.exists():
        shutil.rmtree(out)
    nlp.to_disk(out)
    n = _size(comp)
    print(f"wrote {out}\n  pipeline: {nlp.pipe_names}\n  table entries: {n}")

    if a.verify:
        back = spacy.load(out)
        assert back.pipe_names[-1] == pipe, back.pipe_names
        n_back = _size(back.get_pipe(pipe))
        if n_back != n:
            raise SystemExit(f"table did not survive the round trip: {n} -> {n_back}")
        moved = []
        for f in sorted(Path(a.in_model).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(a.in_model)
            g = out / rel
            if not g.exists():
                moved.append(f"{rel} MISSING")
            elif g.read_bytes() != f.read_bytes():
                moved.append(str(rel))
        weights = [m for m in moved if m.endswith("model")]
        print(f"  reloaded table entries: {n_back}")
        print(f"  files differing: {len(moved)}  (weight files: {len(weights)})")
        for m in moved:
            print("    ", m)
        if weights:
            raise SystemExit("a weight file moved -- this was supposed to be pure surgery")


if __name__ == "__main__":
    main()
