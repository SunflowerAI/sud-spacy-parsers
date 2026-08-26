#!/usr/bin/env python3
"""Stop the generic arm's morphologiser from WRITING the UPOS it was given. Verified on RELOAD.

THE DEFECT. spaCy's morphologiser predicts a single joint label -- `POS=NOUN|Number=Sing` -- and
`set_annotations` writes both halves of it (`spacy/pipeline/morphologizer.pyx`)::

    if doc.c[j].morph == 0 or overwrite or extend:  ... doc.c[j].morph = labels_morph[morph]
    if doc.c[j].pos == 0 or overwrite:              doc.c[j].pos = labels_pos[morph]

`xx_sud_generic` shipped with `overwrite = true`, so both writes fired unconditionally. The whole
contract of this wheel is that **UPOS is the user's input** -- the one column that does not transfer
across languages and therefore the one the wheel refuses to guess -- and the morphologiser was
overwriting it with a guess before the parser ever saw it. The same flag also clobbered any FEATS
the user supplied: measured on the held-in dev sets, 11 % of gold-FEATS tokens in English and 40 %
in Latin came out of the pipeline with different morphology than went in.

THE FIX is `overwrite = false`, which makes both writes conditional on the slot being EMPTY. The
morphologiser then does exactly what the wheel advertises: it fills FEATS in where there are none,
and never touches UPOS, because a UPOS is always there.

⚠ **IT NEEDS TWO EDITS.** `overwrite` lives in `config.cfg` AND in the component's own serialised
`cfg`, and `from_disk` restores the latter -- so patching `config.cfg` alone changes nothing on
load. Same shape as `fix_tagger_overwrite.py`, and the same reason: this is the
in-memory-versus-reloaded trap (CLAUDE.md hazard 8), so `--verify` RELOADS the arm from disk and
runs a doc through it rather than inspecting the object it just edited.

⚠ **`labels_pos` IS LEFT INTACT.** Zeroing it would make the POS write structurally impossible, but
it would also destroy information the artefact carries and make the change irreversible. With
`overwrite = false` plus the `sud_require_upos` guard in front, every token reaching the
morphologiser already has a UPOS, so the POS branch is unreachable -- which is the same guarantee,
reversibly.

Usage:
    fix_generic_pos_write.py MODEL_DIR [MODEL_DIR ...] [--pipe morphologizer] [--check] [--verify]
        --check    report only, change nothing (exit 1 if any arm would need fixing)
        --verify   reload each arm afterwards and assert it preserves a supplied UPOS and FEATS
"""
import argparse
import json
import pathlib
import sys

from thinc.api import Config

PIPE = "morphologizer"


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
    on_disk = json.loads(pipe_cfg.read_text(encoding="utf-8")) if pipe_cfg.exists() else {}
    # the serialised cfg is what from_disk restores, so IT is the authority, not config.cfg
    needs = on_disk.get("overwrite") is not False or comp.get("overwrite") is not False
    if not needs:
        print(f"  {model_dir}: {pipe}.overwrite already false")
        return False
    print(f"  {model_dir}: {pipe}.overwrite = "
          f"{on_disk.get('overwrite')} (cfg) / {comp.get('overwrite')} (config.cfg) -> false")
    if check:
        return True
    comp["overwrite"] = False
    cfg.to_disk(cfg_path)
    if pipe_cfg.exists():
        on_disk["overwrite"] = False
        pipe_cfg.write_text(json.dumps(on_disk), encoding="utf-8")
    return True


def verify(model_dir: pathlib.Path) -> None:
    """Reload from disk and assert the arm preserves UPOS and FEATS token for token."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import spacy
    from spacy.tokens import Doc
    import sud_generic_embed_v2  # noqa: F401  (registers the layer and the guard)

    nlp = spacy.load(model_dir)
    words = ["the", "cat", "sat", "on", "the", "mat"]
    pos = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"]
    feats = ["Definite=Def|PronType=Art", "Number=Sing", "", "", "", ""]
    doc = Doc(nlp.vocab, words=words)
    for t, p, f in zip(doc, pos, feats):
        t.pos_ = p
        if f:
            t.set_morph(f)
    if Doc.has_extension("tb_lang"):
        doc._.tb_lang = "en"
    out = nlp(doc)
    got = [t.pos_ for t in out]
    if got != pos:
        sys.exit(f"  {model_dir}: FAILED -- UPOS was rewritten: {pos} -> {got}")
    kept = [str(t.morph) for t, f in zip(out, feats) if f]
    want = [f for f in feats if f]
    if kept != want:
        sys.exit(f"  {model_dir}: FAILED -- supplied FEATS was rewritten: {want} -> {kept}")
    filled = sum(1 for t, f in zip(out, feats) if not f and str(t.morph))
    print(f"  {model_dir}: UPOS preserved, {len(want)} supplied FEATS preserved, "
          f"{filled} empty FEATS filled -- "
          + "  ".join(f"{t.text}/{t.pos_}/{t.dep_}" for t in out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--pipe", default=PIPE)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    fixed = 0
    for m in a.models:
        if patch(pathlib.Path(m), a.pipe, a.check):
            fixed += 1
    if a.check:
        print(f"{fixed} of {len(a.models)} arms would be changed")
        sys.exit(1 if fixed else 0)
    print(f"{fixed} of {len(a.models)} arms changed")
    if a.verify:
        for m in a.models:
            verify(pathlib.Path(m))


if __name__ == "__main__":
    main()
