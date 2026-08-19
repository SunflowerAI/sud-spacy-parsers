#!/usr/bin/env python3
"""Audit the PUBLISHED ko wheel against the arm it is supposed to be.

A directory is not a release, and neither is a local build: `build_sud/` has held two wheels of the
same name at different generations, so the only evidence that the right bytes went out is a hash
taken from the DOWNLOADED asset. This checks three things, in the order they can go wrong:

  1. the published wheel's weights match the training directory that was measured
  2. they DIFFER from the previous release's, which is what proves a new arm actually shipped
     (a --clobber that silently uploaded the old file would pass check 1 and fail this)
  3. its declared metadata is what it should be

    gh release download v0.3.0 -p 'ko_sud_gsd-0.3.0-*.whl' -D /tmp/dl
    gh release download v0.2.0 -p 'ko_sud_gsd-0.2.0-*.whl' -D /tmp/dl
    .venv/bin/python scripts/audit_ko_release.py /tmp/dl/ko_sud_gsd-0.3.0-py3-none-any.whl \
        --arm training_ko_an_senter/model-best --previous /tmp/dl/ko_sud_gsd-0.2.0-py3-none-any.whl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile

#: every component whose weights must be identical to the measured arm
COMPONENTS = ("tok2vec", "senter", "parser", "morphologizer", "lemmatizer", "tagger")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wheel_weights(path: pathlib.Path) -> dict:
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and parts[-1] == "model" and parts[-2] in COMPONENTS:
                out[parts[-2]] = sha(z.read(name))
    return out


def wheel_meta(path: pathlib.Path) -> dict:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist()
                if n.endswith("/meta.json") and n.count("/") == 2][0]
        return json.loads(z.read(name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=pathlib.Path, help="the DOWNLOADED asset, not the local build")
    ap.add_argument("--arm", type=pathlib.Path, required=True)
    ap.add_argument("--previous", type=pathlib.Path, help="the previous release's asset")
    args = ap.parse_args()

    published = wheel_weights(args.wheel)
    local = {c: sha((args.arm / c / "model").read_bytes())
             for c in COMPONENTS if (args.arm / c / "model").exists()}

    print(f"published: {args.wheel.name}")
    print(f"measured : {args.arm}\n")
    ok = True
    for c in COMPONENTS:
        p, l = published.get(c), local.get(c)
        if p is None and l is None:
            continue
        match = p == l
        ok &= match
        print(f"  {c:<14} {(p or '-')[:16]}  {'==' if match else '!='}  {(l or '-')[:16]}"
              f"{'' if match else '   MISMATCH'}")
    if set(published) != set(local):
        print(f"  ⚠ component sets differ: published {sorted(published)} vs local {sorted(local)}")
        ok = False

    if args.previous:
        prev = wheel_weights(args.previous)
        moved = [c for c in COMPONENTS if c in prev and prev.get(c) != published.get(c)]
        same = [c for c in COMPONENTS if c in prev and prev.get(c) == published.get(c)]
        print(f"\nagainst {args.previous.name}: {len(moved)} component(s) changed {moved}")
        if same:
            print(f"  ⚠ UNCHANGED from the previous release: {same}")
        if "parser" not in moved:
            print("  ⚠ THE PARSER DID NOT CHANGE — the new arm did not ship")
            ok = False

    m = wheel_meta(args.wheel)
    print(f"\nmeta: {m['lang']}_{m['name']} {m['version']} | {m['license']} | {m['requirements']}")
    print(f"      pipeline {m['pipeline']}")
    print("\nAUDIT PASSED" if ok else "\nAUDIT FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
