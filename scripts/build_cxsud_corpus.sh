#!/usr/bin/env bash
# Build the SUD corpus cxSUD converts and scores against (cxSUD `corpus/conllu-sud-md/`), from this
# project's English EWT+GUM arm. ONE recorded pipeline, every split through the same steps.
#
#   bash scripts/build_cxsud_corpus.sh OUT_DIR [CXSUD_DIR]
#
# WHY THIS EXISTS (2026-10-25). The corpus cxSUD had been using could only be reconstructed by
# forensics, and the reconstruction found its splits did not share a lineage:
#   * train/dev were cat(en_ewt.sud, en_gum.sud) -- [EWT][EWT][GUM], since `en_gum` already holds EWT --
#     then resolve_udep over the doubled file, an OLDER normalize_sud.py, an unrecorded drop of 6
#     sentences, and dedup_sud.py to undo the doubling;
#   * test was later overwritten with this project's `.relabeled_ext.udep_resolved.conllu`, which
#     skipped the hoist and the normalisation: no SudSubject in FEATS, no Reported in MISC;
#   * train alone then had 256 partitive `of` hand-edited from `unk` to `udep`.
#   * six sentences had been dropped by a step nobody recorded; they are back (user decision,
#     2026-10-25): train weblog-juancole...20030911_085700-0014, newsgroup...bears...-0044,
#     answers-20111107193044AAvUYBv_ans-0010, answers-20111108102445AA5mfrq_ans-0019,
#     GUM_academic_huh-38; dev GUM_academic_exposure-8.
# This script replaces all of that. The resolution happens ONCE, here, in `.udep_resolved`.
#
# Steps, per split:
#   1. assets/en_ewtgum-sud-<s>.relabeled_ext.reported.udep_resolved.conllu  (EWT + non-NC GUM,
#      udep relabelled by relabel_ext.py, residue resolved by
#        resolve_udep.py <.relabeled_ext.reported> --ruled <split's .udep_ruled> \
#          --table-from <TRAIN .relabeled_ext> <TRAIN .udep_ruled>
#      i.e. ONE table, from train, for every split; noun-level `of` and DET-headed partitive `of`
#      keep the treebank's `udep` -- see restore_noun_of.py and resolve_udep.kept_udep)
#   2. hoist_sud_gold.py --keys Subject Reported Shared, then SudShared -> Shared in FEATS (the name
#      the converter reads; dedup_sud.py used to do this rename)
#   3. cxSUD's normalize_sud.py (subjects to the outermost auxiliary; 'a little X' degree modifiers)
#   4. cxSUD's cxnew/mkmarkdown.py (GUM <hi> emphasis -> * / ** / *** PUNCT tokens)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT=${1:?usage: build_cxsud_corpus.sh OUT_DIR [CXSUD_DIR]}
CX=${2:-../cxSUD}
PY=.venv/bin/python
for f in "$CX/cxsud/scripts/normalize_sud.py" "$CX/cxsud/cxnew/mkmarkdown.py"; do
  [ -s "$f" ] || { echo "FATAL: missing $f (pass the cxSUD checkout as CXSUD_DIR)" >&2; exit 1; }
done
W=$(mktemp -d "${TMPDIR:-/tmp}/cxsud_corpus.XXXXXX"); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/pre" "$OUT"
for s in train dev test; do
  src=assets/en_ewtgum-sud-$s.relabeled_ext.reported.udep_resolved.conllu
  [ -s "$src" ] || { echo "FATAL: missing $src" >&2; exit 1; }
  $PY -I scripts/hoist_sud_gold.py "$src" "$W/$s.hoist.conllu" --keys Subject Reported Shared >/dev/null
  $PY -I - "$W/$s.hoist.conllu" "$W/$s.shared.conllu" <<'PYEOF'
import sys
out = []
for line in open(sys.argv[1], encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) == 10 and "SudShared=" in f[5]:
        f[5] = "|".join("Shared=" + kv.split("=", 1)[1] if kv.startswith("SudShared=") else kv
                        for kv in f[5].split("|"))
        line = "\t".join(f) + "\n"
    out.append(line)
open(sys.argv[2], "w", encoding="utf-8").write("".join(out))
PYEOF
  $PY -I "$CX/cxsud/scripts/normalize_sud.py" "$W/$s.shared.conllu" --out "$W/pre/en_ewtgum-sud-$s.conllu"
done
$PY -I "$CX/cxsud/cxnew/mkmarkdown.py" "$W/pre" "$OUT"
