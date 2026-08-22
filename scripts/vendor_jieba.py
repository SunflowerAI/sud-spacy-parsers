"""Vendor the pruned jieba INTO a built model directory, and drop the pip requirement.

`spacy package` copies the model directory wholesale and `setup.py`'s `list_files` walks it
recursively, so anything dropped inside the model dir ships as package_data with no change to the
generated setup.py. That is why the tree goes here rather than beside the `--code` modules, which
are listed one file at a time and cannot carry a subpackage.

WHY VENDOR AT ALL. The wheel declared `jieba>=0.42.1`, so every install pulled 42 MB of which
~6 MB is reachable: `posseg` (POS tagging), `lac_small` (a neural model) and `analyse` (keyword
extraction) are never imported by `zh_jieba_feature`. For a serverless target with a 250 MB
unzipped budget that is a quarter of the allowance spent on dead files. Vendoring the reachable
subset makes the wheel self-contained: +6.4 MB on the artefact, −36 MB on any install of it.

Licence: jieba is MIT, so redistribution inside this wheel is permitted; its LICENSE travels with
the tree. The zh wheel itself stays CC BY-SA.

The allowlist is imported from `slim_jieba` rather than restated, so the vendored tree and the
deployment-pruning path can never disagree about what is reachable.

WHICH allowlist is decided by the MODEL, not by a flag. A segmenter that records `jieba_dict` in
its `vocab.json` ships the traditional dictionary it was trained against beside its own weights and
calls `jieba.set_dictionary` with it, so jieba's simplified `dict.txt` is never opened and is
dropped from the vendored tree (`KEEP_MODEL_DICT`). That is what keeps the traditional dictionary
free: 5.06 MB in, 5.07 MB out. On the older `jieba_t2s` regime the model has no dictionary of its
own, jieba's is the one it reads, and it stays.

    python scripts/vendor_jieba.py <built-model-dir>
"""
import argparse
import json
import pathlib
import shutil
import sys


def _sibling(name):
    import importlib
    import importlib.util
    if __package__:
        try:
            return importlib.import_module("." + name, __package__)
        except ImportError:
            pass
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SLIM = _sibling("slim_jieba")
KEEP = _SLIM.KEEP


def vendor(model_dir: pathlib.Path) -> None:
    import jieba
    src = pathlib.Path(jieba.__file__).parent
    dst = model_dir / "vendor" / "jieba"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Ask the artefact which dictionary it reads rather than assuming, the same way the segmenter
    # itself reads `jieba_dict` back out of vocab.json instead of trusting a remembered flag.
    own = next(iter(sorted(model_dir.rglob(_sibling("zh_jieba_feature").TRAD_DICT_FILE))), None)
    keep = _SLIM.KEEP_MODEL_DICT if own else KEEP
    if own:
        print(f"  model carries its own jieba dictionary ({own.relative_to(model_dir)}, "
              f"{own.stat().st_size / 1e6:.1f} MB) — vendoring jieba WITHOUT dict.txt")

    missing = [rel for rel in keep if not (src / rel).is_file()]
    if missing:
        sys.exit(f"installed jieba at {src} is missing {missing} — refusing to vendor a partial copy")
    for rel in sorted(keep):
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, out)
    # jieba ships NO licence file -- its dist-info records only `License: MIT` in METADATA -- so
    # vendoring means writing the notice ourselves. Redistribution without it would not satisfy MIT.
    # Facts below come from the installed distribution's own metadata, not from memory.
    ver = getattr(jieba, "__version__", "unknown")
    (dst / "NOTICE").write_text(
        f"""This directory is a pruned redistribution of jieba {ver}.

    Upstream:  https://github.com/fxsjy/jieba
    Author:    Sun, Junyi <ccnusjy@gmail.com>
    Licence:   MIT (as declared in the distribution's own METADATA)

Only the files the zh segmenter loads are included; see scripts/slim_jieba.py for the
reachability analysis. No file has been modified -- they are byte-for-byte copies.

The MIT Licence permits redistribution provided the copyright notice and this permission
notice accompany the software. The canonical licence text is published in the upstream
repository above.

MIT License

Copyright (c) 2013 Sun Junyi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""", encoding="utf-8")

    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"  vendored jieba {getattr(jieba, '__version__', '?')} -> {dst}  ({size:.1f} MB, "
          f"{len(list(dst.rglob('*.py')))} modules)")

    # Drop the pip requirement: the wheel now carries the code, and leaving the declaration would
    # pull the full 42 MB back in on every install — exactly what this exists to avoid.
    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    before = list(meta.get("requirements") or [])
    meta["requirements"] = [r for r in before if not r.split(">=")[0].split("==")[0].strip() == "jieba"]
    if before != meta["requirements"]:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        print(f"  requirements: {before} -> {meta['requirements']}")
    else:
        print(f"  requirements unchanged (no jieba declared): {before}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=pathlib.Path)
    args = ap.parse_args()
    if not (args.model_dir / "meta.json").is_file():
        sys.exit(f"not a built model directory (no meta.json): {args.model_dir}")
    vendor(args.model_dir)


if __name__ == "__main__":
    main()
