#!/usr/bin/env bash
# Train the TRADITIONAL character segmenter the released zh_sud_gsd tokenises with.
#
# zh went traditional-only at 0.2.0 (one script inside, either script outside -- zh_script.py), but
# `models/zh_seg_jbdec` is trained on GSDSimp: 3,336 Han characters of which exactly ONE (乾, an era
# name) is traditional-only. A traditional arm carrying it meets OOV at nearly every character.
# `models/` is gitignored, so this driver is the only record of how the shipped segmenter is built.
#
# TWO THINGS THAT ARE NOT OBVIOUS, both measured:
#
#  1. THE LEXICON ONLY WORKS JACKKNIFED. A word list harvested from train covers 100 % of train and
#     ~88 % of test, so without --jackknife the model learns a reliability the feature will not have
#     and never develops a fallback. Per-fold coverage lands at 86.9 % here, matching the simplified
#     arm's ~87.0 %.
#
#  2. jieba IS ASKED ABOUT THE t2s RENDERING (--jieba-t2s). jieba's dictionary is simplified: its
#     boundary decisions score F 0.8920 on traditional text and 0.9223 on the t2s conversion, which
#     is what the simplified arm was built on (P 0.9730 / R 0.8793) -- so the whole gap is
#     vocabulary, not the language. Codes are per character and t2s preserves length (500/500 test
#     sentences), so the answer transfers by position. The flag is written into the segmenter's
#     vocab.json and read back at load time, because a channel asked a different question at
#     inference than at training is the `reads_spaces` trap again.
#
# Result: strict whole-token F 0.9242 on the traditional test, against 0.9210 for the simplified
# segmenter on its own. NB model init is unseeded, so expect a spread of a couple of tenths.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_zh/SUD_Chinese-GSD
SRC=zh_gsd-sud
SUF=relabeled_ext
D=data_seg_zh_trad
LEX=models/zh_lex_corpus_trad.txt
OUT=models/zh_seg_jbdec_trad

mkdir -p "$D"
for s in train dev test; do
  f="$A/${SRC}-${s}.${SUF}.conllu"
  [ -f "$f" ] || { echo "  missing $f"; exit 1; }
  $PY scripts/make_seg_pairs.py "$f" "$D/$s.jsonl" --min-chunk 1 2>&1 | sed "s/^/  $s /"
done

# The lexicon is the training split's FORM inventory -- the traditional counterpart of
# models/zh_lex_corpus.txt, which is exactly GSDSimp's.
$PY - "$A/${SRC}-train.${SUF}.conllu" "$LEX" <<'PYEOF'
import pathlib, sys
src, out = sys.argv[1:3]
forms = set()
for line in pathlib.Path(src).read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        f = line.split("\t")
        if len(f) > 3 and "-" not in f[0] and "." not in f[0]:
            forms.add(f[1])
pathlib.Path(out).write_text("\n".join(sorted(forms)), encoding="utf-8")
print(f"  lexicon {out}: {len(forms)} types")
PYEOF

# Source 0 = the corpus word list, source 1 = jieba's BMES decision. Both channels are 4-valued, so
# the architecture is the same size as the simplified arm's and the two are comparable.
$PY -u scripts/sa_presegment_lex.py "$D/train.jsonl" "$D/dev.jsonl" "$OUT" \
    --lexicon "$LEX" "$LEX" --min-lens 1 1 \
    --jieba-source 1 --jieba-t2s --jackknife 5 \
    --width 64 --depth 6 --epochs 30 2>&1 | grep -vE "prefix dict|Loading model|Prefix dict|pkg_resources|UserWarning|import pkg" | tail -12

# STRICT WHOLE-TOKEN F is the metric, not character accuracy or split-location F: a chunk with one
# wrong boundary loses two tokens here and one boundary there. eval_zh_seg reads the jieba settings
# off the model, so it cannot score it with a different channel than it was trained with.
$PY scripts/eval_zh_seg.py "$OUT" "$D/test.jsonl" --lexicon "$LEX" "$LEX" --min-lens 1 1 \
    --compare-jieba 2>&1 | grep -E "strict token|jieba alone|chunks"

echo "next: bash scripts/package_sud.sh zh   (add_zh_script.py wires this in and verifies the reload)"
