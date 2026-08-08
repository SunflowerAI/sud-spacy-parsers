#!/usr/bin/env python
"""Rename one dependency label inside a TRAINED parser, without retraining.

Why this exists. `reparandum` -> `conj:dicto` is a PURE LABEL RENAME: SUD analyses disfluency
repair as a subtype of `conj`, the head and the attachment are identical, and only the string
differs (see normalise_reparandum.py). Retraining a parser on renamed data therefore produces the
same model up to RNG -- the label is just the name of an action. Doing the rename directly on the
trained model is the EXACT analogue, and it is strictly better for a released artefact: every
weight stays byte-identical, so every published metric (LAS, TAG, per-label F) remains valid, and
a re-release changes the one thing that was wrong instead of everything at once.

THE THING THAT MAKES THIS DANGEROUS, and the only reason this is a script rather than a sed.
spaCy assigns action indices in `TransitionSystem.initialize_actions` by sorting each move type's
labels on `(frequency, label_string)` DESCENDING. The label STRING is a tiebreak. So renaming a
label can move it past another label of equal frequency, which renumbers the actions -- and the
model's output rows are indexed by action. The weights would then be silently misaligned: the
model loads, runs, and emits confident nonsense.

    en's `reparandum` has frequency 31 in LEFT-ARC, and so does `comp:aux@pass`.
    "reparandum" > "comp:aux@pass" and "conj:dicto" > "comp:aux@pass", so the order survives --
    but that is a coincidence of two strings, not a property of the rename.

This script therefore replicates spaCy's ordering exactly, computes the full (action, label)
sequence before and after, and REFUSES unless the two are identical position for position with
only the renamed entry differing. `--verify-parses` then proves it end to end.

    rename_deprel_label.py MODEL_DIR --from reparandum --to conj:dicto [--verify-parses FILE.conllu]
"""
import argparse
import json
import pathlib
import shutil

import srsly


def action_sequence(labels_by_action):
    """The (action, label) order spaCy will assign indices in. Mirrors initialize_actions."""
    seq = []
    for action, label_freqs in sorted(labels_by_action.items()):
        sorted_labels = [(f, L) for L, f in label_freqs.items()]
        sorted_labels.sort()
        sorted_labels.reverse()
        # negative frequencies are appended after everything else, in their own sorted order
        normal = [(f, L) for f, L in sorted_labels if f >= 0]
        added = sorted([(f, L) for f, L in sorted_labels if f < 0], reverse=True)
        for _freq, label in normal:
            seq.append((int(action), label))
        for _freq, label in added:
            seq.append((int(action), label))
    return seq


def rename_in_moves(labels_by_action, old, new):
    """Rename in place, PRESERVING dict insertion order (the frequency travels with the label)."""
    out, n = {}, 0
    for action, label_freqs in labels_by_action.items():
        if new in label_freqs and old in label_freqs:
            raise SystemExit(f"refusing: move {action} already has a `{new}`, so the rename would "
                             f"collide with it and lose one of the two actions.")
        renamed = {}
        for label, freq in label_freqs.items():
            if label == old:
                renamed[new] = freq
                n += 1
            else:
                renamed[label] = freq
        out[action] = renamed
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--component", default="parser")
    ap.add_argument("--verify-parses", metavar="CONLLU",
                    help="parse this file with the model before and after and require every head "
                         "and every deprel to match, modulo the rename. The real proof.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = pathlib.Path(args.model_dir)
    moves_p = model / args.component / "moves"
    if not moves_p.exists():
        raise SystemExit(f"no {moves_p}")

    blob = srsly.msgpack_loads(moves_p.read_bytes())
    labels_by_action = srsly.json_loads(blob["moves"])

    present = {a: f[args.old] for a, f in labels_by_action.items() if args.old in f}
    if not present:
        raise SystemExit(f"`{args.old}` is not a label of this {args.component} -- nothing to do.")
    print(f"  `{args.old}` appears in move(s) {present} (move type -> training frequency)")

    before = action_sequence(labels_by_action)
    renamed, n = rename_in_moves(labels_by_action, args.old, args.new)
    after = action_sequence(renamed)

    # THE CHECK. Same length, same order, and every position either identical or exactly the
    # substitution asked for. Anything else renumbers the actions and misaligns the weights.
    if len(before) != len(after):
        raise SystemExit(f"refusing: action count {len(before)} -> {len(after)}")
    bad = [(i, b, a) for i, (b, a) in enumerate(zip(before, after))
           if b != a and not (b[0] == a[0] and b[1] == args.old and a[1] == args.new)]
    if bad:
        print("  ACTION ORDER CHANGED at:")
        for i, b, a in bad[:10]:
            print(f"    index {i}: {b} -> {a}")
        raise SystemExit("refusing: the rename renumbers the actions, so the trained weights would "
                         "be silently misaligned. Retrain instead.")
    moved = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    print(f"  action order preserved: {len(before)} actions, only index/indices {moved} relabelled")

    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    before_parses = None
    if args.verify_parses:
        before_parses = _parse_all(model, args.verify_parses)

    backup = model.with_name(model.name + f".pre_{args.old}_rename")
    if not backup.exists():
        shutil.copytree(model, backup)
        print(f"  backup: {backup}")

    blob["moves"] = srsly.json_dumps(renamed)
    moves_p.write_bytes(srsly.msgpack_dumps(blob))
    print(f"  wrote {moves_p} ({n} label(s) renamed)")

    # meta.json carries a labels list per component; a stale one is cosmetic but misleading.
    for meta_p in (model / "meta.json",):
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        labels = (meta.get("labels") or {}).get(args.component)
        if labels and args.old in labels:
            meta["labels"][args.component] = [args.new if x == args.old else x for x in labels]
            meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  updated {meta_p} labels.{args.component}")

    if args.verify_parses:
        after_parses = _parse_all(model, args.verify_parses)
        _compare(before_parses, after_parses, args.old, args.new)


def _parse_all(model_dir, conllu):
    """Parse every sentence of a CoNLL-U file over GOLD tokens; return (heads, deprels) per sent."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import seg_code  # noqa: F401  (registers the custom factories these arms need)
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(model_dir)
    out = []
    words = []
    for line in pathlib.Path(conllu).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if words:
                doc = nlp(Doc(nlp.vocab, words=words))
                out.append(([t.head.i for t in doc], [t.dep_ for t in doc]))
                words = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        words.append(c[1])
    if words:
        doc = nlp(Doc(nlp.vocab, words=words))
        out.append(([t.head.i for t in doc], [t.dep_ for t in doc]))
    return out


def _compare(before, after, old, new):
    assert len(before) == len(after), (len(before), len(after))
    head_diff = dep_other = renamed = 0
    for (hb, db), (ha, da) in zip(before, after):
        head_diff += sum(x != y for x, y in zip(hb, ha))
        for x, y in zip(db, da):
            if x == y:
                continue
            if x == old and y == new:
                renamed += 1
            else:
                dep_other += 1
    print(f"\n  VERIFY over {len(before)} sentences:")
    print(f"    heads differing:                 {head_diff}")
    print(f"    deprels differing, `{old}`->`{new}`: {renamed}")
    print(f"    deprels differing, ANYTHING else:  {dep_other}")
    if head_diff or dep_other:
        raise SystemExit("VERIFICATION FAILED -- the model is not equivalent. Restore the backup.")
    print("    => the renamed model is parse-identical, modulo exactly the rename.")


if __name__ == "__main__":
    main()
