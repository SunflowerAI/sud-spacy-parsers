#!/usr/bin/env python3
"""Make a trained arm's tagger actually WRITE its predictions. Verified on the RELOADED model.

THE DEFECT. spaCy's tagger sets a tag only where ``token.tag == 0`` unless ``overwrite`` is on
(``spacy/pipeline/tagger.pyx``, ``set_annotations``). Every config in this repo carries the stock
``overwrite = false`` -- harmless for eleven arms, because nothing sets a tag before the tagger
runs. ja is the exception: ``spacy.ja.JapaneseTokenizer`` assigns ``token.tag_ = dtoken.tag`` at
TOKENISATION (lang/ja/__init__.py), so every token already has a tag and the trained tagger is a
NO-OP at inference. Measured on the ja test set, over the gold tokens the tokeniser reproduces
exactly: users get SudachiPy's raw UniDic tag at 0.7673, where the trained tagger would give
0.9457. Confirmed in the DOWNLOADED ja_sud_gsd-0.2.0 wheel, not merely in a training directory.

``gold_preproc`` hides it completely (CLAUDE.md standing hazard 4): the predicted doc is built from
gold words and carries no tag, so ``tag == 0`` and the tagger DOES write. Every ``tag_acc`` ever
published for ja therefore describes a component whose output no user receives.

WHY IT NEEDS TWO EDITS. ``overwrite`` lives in the config AND in the component's own serialised
``cfg`` file, and ``from_disk`` restores the latter -- so patching ``config.cfg`` alone changes
nothing on load. This is the in-memory-versus-reloaded trap (standing hazard 8), so the fix is
checked by RELOADING the arm from disk and re-running it, never by inspecting the object it just
edited.

``overwrite = true`` is correct for every arm, not just ja: where nothing pre-sets a tag, ``tag``
is already 0 and the flag changes nothing. That is what lets ``package_sud.sh`` refuse
unconditionally instead of carrying a list of affected languages.

Usage:
    fix_tagger_overwrite.py MODEL_DIR [MODEL_DIR ...] [--pipe tagger] [--check]
        --check   report only, change nothing (exit 1 if any arm would need fixing)
"""
import json
import pathlib
import sys

from thinc.api import Config

PROBES = {  # only needed to DEMONSTRATE the change; the flag itself is language-agnostic
    "ja": "彼は東京へ行かない。",
    "ko": "나는 어제 서울에 갔다.",
}


def patch(model_dir: pathlib.Path, pipe: str, check: bool) -> bool:
    """Return True if the arm needed fixing."""
    cfg_path = model_dir / "config.cfg"
    pipe_cfg = model_dir / pipe / "cfg"
    if not cfg_path.exists():
        print(f"  {model_dir}: no config.cfg -- skip")
        return False
    cfg = Config().from_disk(cfg_path, interpolate=False)   # interpolate=False: CLAUDE.md (E913)
    comp = cfg.get("components", {}).get(pipe)
    if comp is None:
        print(f"  {model_dir}: no '{pipe}' component -- skip")
        return False
    on_disk = json.loads(pipe_cfg.read_text()) if pipe_cfg.exists() else {}
    # the serialised cfg is what from_disk restores, so IT is the authority, not config.cfg
    needs = on_disk.get("overwrite") is not True
    if not needs:
        print(f"  {model_dir}: {pipe}.overwrite already true")
        return False
    if check:
        print(f"  {model_dir}: {pipe}.overwrite is {on_disk.get('overwrite')!r} -- NEEDS FIX")
        return True
    comp["overwrite"] = True
    cfg.to_disk(cfg_path)
    on_disk["overwrite"] = True
    pipe_cfg.write_text(json.dumps(on_disk))
    print(f"  {model_dir}: {pipe}.overwrite -> true (config.cfg + {pipe}/cfg)")
    return True


def verify(model_dir: pathlib.Path, pipe: str):
    """Reload from disk and prove the flag survived -- and, where a probe exists, that it bites."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import seg_code  # noqa: F401  (registers custom architectures/tokenisers)
    import spacy
    nlp = spacy.load(model_dir)
    got = nlp.get_pipe(pipe).cfg.get("overwrite")
    if got is not True:
        raise SystemExit(f"FATAL: {model_dir} reloaded with {pipe}.overwrite={got!r}")
    lang = nlp.lang
    probe = PROBES.get(lang)
    if probe is None:
        print(f"  verified (reloaded): {pipe}.overwrite=True  [{lang}: tokeniser sets no tag]")
        return
    raw, out = nlp.make_doc(probe), nlp(probe)
    changed = sum(1 for a, b in zip(raw, out) if a.tag_ != b.tag_)
    print(f"  verified (reloaded): {pipe}.overwrite=True, tagger now rewrites "
          f"{changed}/{len(out)} tokens on the {lang} probe")
    if changed == 0:
        print(f"  ⚠ {model_dir}: still rewrites nothing -- the tokeniser's tags may already agree")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    pipe = "tagger"
    if "--pipe" in sys.argv:
        pipe = sys.argv[sys.argv.index("--pipe") + 1]
    if not args:
        sys.exit(__doc__.strip().splitlines()[-1])
    fixed = [patch(pathlib.Path(a), pipe, check) for a in args]
    if check:
        sys.exit(1 if any(fixed) else 0)
    for a in args:
        if (pathlib.Path(a) / "config.cfg").exists():
            verify(pathlib.Path(a), pipe)


if __name__ == "__main__":
    main()
