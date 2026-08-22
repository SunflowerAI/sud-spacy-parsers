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
#  2. jieba READS A TRADITIONAL DICTIONARY (--jieba-dict), not a converted rendering of the text.
#     jieba's own dict.txt is simplified, so a traditional arm that asks it directly gets boundary
#     F 0.8931 -- the defect. The first fix converted the whole chunk (--jieba-t2s, F 0.9236) and
#     kept the per-character answer, which recovered the vocabulary but left the channel answering
#     about a DIFFERENT STRING: t2s is many-to-one, so 乾/幹/干 all reach jieba as 干.
#     build_jieba_trad_dict.py converts jieba's DICTIONARY instead (s2tw -- the same conversion
#     zh_script applies to incoming simplified input, and GSD's own orthography), so the lookup
#     reads the traditional text itself. Only the OOV HMM still consults the t2s rendering, because
#     its emission probabilities are per character and were estimated on simplified text: that one
#     component is the entire remaining gap (F 0.9203 without it, 0.9237 with). The regime is
#     written into the segmenter's vocab.json and the dictionary is copied in beside the weights,
#     because a channel asked a different question at inference than at training is the
#     `reads_spaces` trap again -- and this one would come back on the wrong VOCABULARY silently.
#
# Result: the two jieba regimes are a WASH end to end -- ten runs each, mean strict whole-token F
# 0.9209 (traditional dictionary) against 0.9203 (t2s), sd 0.003, 6/10 seeds favouring the
# dictionary -- so it is chosen for reading the script the model works in, not for its score.
# ⚠ MODEL INIT IS UNSEEDED AND THE SPREAD IS WIDER THAN THE EFFECT: 0.9167-0.9268 over twenty
# traditional-dictionary runs. Select on DEV and quote the mean, never the best test draw. The
# dev-selected run (dev strict token F 0.9253, unchanged when the pool went from 10 candidates to
# 20) scores 0.9196 on test; the SUPERSEDED t2s arm the wheel shipped at 0.2.0 scored 0.9242, which
# is the top of the same spread rather than a better arm.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_zh/SUD_Chinese-GSD
SRC=zh_gsd-sud
SUF=relabeled_ext
D=data_seg_zh_trad
LEX=models/zh_lex_corpus_trad.txt
OUT=models/zh_seg_jbdict_trad
JBDICT=models/jieba_dict_trad.txt

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

# jieba's dictionary in the script the model works in. Derived from jieba's own, so the channel
# stays EXTERNAL and needs no jackknifing -- a traditional word list harvested from GSD train would
# reintroduce exactly the leak --jackknife exists to remove.
[ -f "$JBDICT" ] || $PY scripts/build_jieba_trad_dict.py -o "$JBDICT"

# Source 0 = the corpus word list, source 1 = jieba's BMES decision. Both channels are 4-valued, so
# the architecture is the same size as the simplified arm's and the two are comparable.
$PY -u scripts/sa_presegment_lex.py "$D/train.jsonl" "$D/dev.jsonl" "$OUT" \
    --lexicon "$LEX" "$LEX" --min-lens 1 1 \
    --jieba-source 1 --jieba-dict "$JBDICT" --jackknife 5 \
    --width 64 --depth 6 --epochs 30 2>&1 | grep -vE "prefix dict|Loading model|Prefix dict|pkg_resources|UserWarning|import pkg" | tail -12

# STRICT WHOLE-TOKEN F is the metric, not character accuracy or split-location F: a chunk with one
# wrong boundary loses two tokens here and one boundary there. eval_zh_seg reads the jieba settings
# off the model -- regime and dictionary both -- so it cannot score it with a different channel
# than it was trained with, and the `jieba alone` line is that same channel rather than a stock
# jieba on the raw text (which under-read it by 5 token F).
$PY scripts/eval_zh_seg.py "$OUT" "$D/test.jsonl" --lexicon "$LEX" "$LEX" --min-lens 1 1 \
    --compare-jieba 2>&1 | grep -E "strict token|jieba alone|chunks"

echo "next: bash scripts/package_sud.sh zh   (add_zh_script.py wires this in and verifies the reload)"
