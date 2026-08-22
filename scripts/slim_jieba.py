"""Prune an installed jieba down to what the zh segmenter actually loads (42 MB -> 7 MB).

WHERE THIS BELONGS. Not in the wheel. `zh_sud_gsd` declares `jieba>=0.42.1` as a runtime
requirement and pip installs the whole distribution; a model wheel cannot prune another package's
files. This runs against a DEPLOYMENT TREE -- the site-packages of a Lambda bundle, a container
layer, a venv about to be zipped -- after the dependencies are installed and before the artefact is
sealed.

WHAT IS SAFE TO DROP, and how that was established: import the feature path and read
`sys.modules`. Only `jieba`, `jieba._compat` and `jieba.finalseg` (+ its three probability tables)
are ever loaded by `zh_jieba_feature.jieba_codes`, which is the only way this project uses jieba.

    posseg/       15 MB   part-of-speech tagging          never imported
    lac_small/    13 MB   the LAC neural model            never imported
    analyse/       6 MB   TF-IDF / TextRank keywords      never imported
    */*.p          1 MB   pickled probability tables      CPython imports the .py twins instead

Verified byte-identical: the shipped `zh_sud_gsd` wheel over 500 test sentences gives the same
11 949 tokens and the same full-parse digest on the pruned tree as on the full one.

⚠ `dict.txt` (4.8 MB of the surviving 7 MB) IS the feature -- `CLAUDE.md` records jieba's
traditional-vs-`t2s` gap as "entirely vocabulary", and the channel is worth +4.42 token F. The
corpus lexicon already covers corpus vocabulary; jieba's job is the words outside it. So prune it
in exactly ONE case, `--drop-dict`: a model whose segmenter records `jieba_dict` carries its own
traditional dictionary beside its weights and calls `set_dictionary` with it, so jieba's own file
is never opened. That is what `vendor_jieba.py` does, via `KEEP_MODEL_DICT` below -- and it is why
the traditional dictionary costs the wheel nothing rather than 5 MB. Dropping it against a model
still on the `jieba_t2s` regime breaks that model outright, which is the point of tying the
decision to the artefact rather than to a flag someone remembers.

⚠ jieba writes a ~9 MB cache to `tempfile.gettempdir()` on first use. On Lambda that is /tmp
(writable, 512 MB) so it works, but the first container in a cold pool pays to build it.

    python scripts/slim_jieba.py <site-packages>              # prune in place
    python scripts/slim_jieba.py <site-packages> --check      # report only, change nothing
    python scripts/slim_jieba.py <site-packages> --drop-dict  # also drop dict.txt (see above)
"""
import argparse
import pathlib
import shutil
import sys

# Everything the zh segmenter loads. Anything in the jieba package not matching one of these is
# removed. Kept as an ALLOWLIST rather than a list of things to delete, so a future jieba that
# adds another giant subpackage is pruned by default instead of silently shipping.
KEEP = {
    "__init__.py",
    "_compat.py",
    "dict.txt",
    "finalseg/__init__.py",
    "finalseg/prob_emit.py",
    "finalseg/prob_start.py",
    "finalseg/prob_trans.py",
}

# The same allowlist for a model that brings its OWN jieba dictionary (`jieba_dict` in the
# segmenter's vocab.json). Derived rather than restated, so a future addition to KEEP cannot be
# forgotten here.
KEEP_MODEL_DICT = KEEP - {"dict.txt"}


def _mb(p: pathlib.Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def slim(site_packages: pathlib.Path, check: bool = False, drop_dict: bool = False) -> int:
    keep = KEEP_MODEL_DICT if drop_dict else KEEP
    root = site_packages / "jieba"
    if not root.is_dir():
        print(f"no jieba in {site_packages} — nothing to do")
        return 0
    before = _mb(root)
    removed = []
    for item in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = item.relative_to(root).as_posix()
        if rel in keep or item.name == "__pycache__":
            continue
        # keep a directory only while it still holds something on the allowlist
        if item.is_dir():
            if any(k.startswith(rel + "/") for k in keep):
                continue
            if not check:
                shutil.rmtree(item, ignore_errors=True)
            removed.append((rel + "/", 0.0))
        else:
            size = _mb(item)
            if not check:
                item.unlink(missing_ok=True)
            removed.append((rel, size))
    for rel, size in sorted(removed, key=lambda r: -r[1])[:6]:
        print(f"  {'would remove' if check else 'removed'}  {size:6.1f} MB  {rel}")
    after = before if check else _mb(root)
    saved = before - after if not check else sum(s for _, s in removed)
    print(f"  jieba {before:.1f} MB -> {before - saved:.1f} MB   (saved {saved:.1f} MB"
          f"{', dry run' if check else ''})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site_packages", type=pathlib.Path,
                    help="the deployment tree's site-packages (NOT your dev venv)")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--drop-dict", action="store_true",
                    help="also remove jieba's own dict.txt (4.8 MB). ONLY for a tree that serves a "
                         "model whose segmenter records `jieba_dict`: that model ships the "
                         "dictionary it was trained on and never opens jieba's. A model on the "
                         "`jieba_t2s` regime cannot initialise without it.")
    args = ap.parse_args()
    if not args.site_packages.is_dir():
        sys.exit(f"not a directory: {args.site_packages}")
    sys.exit(slim(args.site_packages, args.check, args.drop_dict))


if __name__ == "__main__":
    main()
