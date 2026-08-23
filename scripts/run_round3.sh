#!/usr/bin/env bash
# Three retrains, run in sequence so they do not contend for CPU.
#
#  1. ko   the eojeol arm's treebank changed: extending the ext scope to NOUN/ADJ/ADV heads moved
#          udep 1022 -> 694 (comp:obl +25, mod +303), so base/morph/lemma are all stale.
#  2. id   the segmenter is retrained on pairs that finally CONTAIN the enclitic junction. The first
#          one learned punctuation splitting only, because MWT sub-tokens carry no `SpaceAfter=No`
#          (the range line does) so `penghuni`/`nya` read as separate whitespace words. 1002 `-nya`
#          junctions are now in the data. The parser arm then needs the segmenter loaded into its
#          tokenizer POST-HOC — the registered factory builds one with no model, and a tokenizer
#          silently falling back to whitespace cost TAG 76.13 / LAS 48.40 / SENT F 0.95 last time.
#  3. sa   UFAL raised to parity with Vedic inside the syntactic half (x612 against Vedic x5).
#          170 sentences repeated 612 times is heavy overfitting risk; the point is to find out
#          whether classical LAS moves, and the UFAL test set is held out from all of it.
set -euo pipefail

cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python
CODE=scripts/seg_code.py

echo "########## 1. Korean: rebuild corpus from the relabelled treebank ##########"
K=assets_ko/SUD_Korean-GSD
for s in train dev test; do
  $PY -m spacy convert $K/ko_gsd-sud-$s.relabeled_ext.conllu corpus_ko_eojeol/ \
      --converter conllu -n 10 2>&1 | grep -o "([0-9]* documents)" | sed "s/^/  $s /"
done
$PY -m spacy train configs/config_ko_eojeol_seg.cfg --output training_ko_eojeol_seg/ \
    --code $CODE 2>&1 | tail -2
$PY scripts/make_morph_config.py configs/config_ko_eojeol_seg.cfg \
    training_ko_eojeol_seg/model-best --out configs/config_ko_eojeol_morph.cfg
$PY -m spacy train configs/config_ko_eojeol_morph.cfg --output training_ko_eojeol_morph/ \
    --code $CODE 2>&1 | tail -2
$PY scripts/make_lemma_config.py configs/config_ko_eojeol_morph.cfg \
    training_ko_eojeol_morph/model-best --out configs/config_ko_eojeol_lemma.cfg
$PY -m spacy train configs/config_ko_eojeol_lemma.cfg --output training_ko_eojeol_lemma/ \
    --code $CODE 2>&1 | tail -2
$PY -m spacy evaluate training_ko_eojeol_lemma/model-best \
    corpus_ko_eojeol/ko_gsd-sud-test.relabeled_ext.spacy --code $CODE \
    --output metrics/ko/metrics_ko_eojeol_v2_raw.json 2>&1 | grep -E "TOK|TAG|POS|LEMMA|UAS|LAS|SENT"

echo "########## 2. Indonesian: retrain the segmenter, then the arm ##########"
$PY scripts/train_samhita.py data_seg_id/train.jsonl data_seg_id/dev.jsonl \
    models/id_seg_char2 --width 64 --depth 6 --epochs 30 2>&1 | tail -3
$PY - <<'PY'
import json, sys, pathlib
sys.path.insert(0, "scripts")
from sa_presegment import Presegmenter
seg = Presegmenter.from_disk("models/id_seg_char2")
for w in ("penghuninya", "rumahnya", "bukunya", "adalah", "salah", "dialah"):
    p = seg.predict([w])[0]
    out, cur = [], ""
    for ch, lb in zip(w, p):
        cur += ch
        if lb == "= ": out.append(cur); cur = ""
    if cur: out.append(cur)
    print(f"    {w:<14} -> {' / '.join(out)}")
PY
$PY -m spacy train configs/config_id_split_seg.cfg --output training_id_split_seg/ \
    --code $CODE 2>&1 | tail -2
$PY scripts/make_morph_config.py configs/config_id_split_seg.cfg \
    training_id_split_seg/model-best --out configs/config_id_split_morph.cfg
$PY -m spacy train configs/config_id_split_morph.cfg --output training_id_split_morph/ \
    --code $CODE 2>&1 | tail -2
$PY scripts/make_lemma_config.py configs/config_id_split_morph.cfg \
    training_id_split_morph/model-best --out configs/config_id_split_lemma.cfg
$PY -m spacy train configs/config_id_split_lemma.cfg --output training_id_split_lemma/ \
    --code $CODE 2>&1 | tail -2
$PY - <<'PY'
import sys; sys.path.insert(0, "scripts")
import char_seg_tokenizer, sud_affix_embed, gold_tok_corpus  # noqa
import spacy
nlp = spacy.load("training_id_split_lemma/model-best")
nlp.tokenizer.load_segmenter("models/id_seg_char2")   # the factory builds one with NO model
nlp.to_disk("build_id_charseg")
print("  segmenter loaded into build_id_charseg")
PY
$PY -m spacy evaluate build_id_charseg \
    corpus_id_split/id_gsd-sud-test.relabeled_ext.spacy --code $CODE \
    --output metrics/id/metrics_id_v2_raw.json 2>&1 | grep -E "TOK|TAG|POS|LEMMA|UAS|LAS|SENT"

echo "########## 3. Sanskrit: UFAL at parity with Vedic ##########"
$PY -m spacy train configs/config_sa_ufal_up.cfg --output training_sa_ufal_up/ \
    --code $CODE 2>&1 | tail -2
for c in vedic_test ufal_test; do
  $PY scripts/eval_sa_compound.py training_sa_ufal_up/model-best corpus_sa_split/$c.spacy \
      --out metrics/sa/metrics_sa_ufalup_$c.json >/dev/null 2>&1
done
$PY - <<'PY'
import json
a = json.load(open("metrics/sa/metrics_sa_ufalup_vedic_test.json"))
b = json.load(open("metrics/sa/metrics_sa_ufalup_ufal_test.json"))
print(f"  Vedic  LAS {a['dep_las']:.4f}  UAS {a['dep_uas']:.4f}   (multitask 0.5140, +5x upsample 0.5415)")
print(f"  UFAL   LAS {b['dep_las']:.4f}  UAS {b['dep_uas']:.4f}   (multitask 0.4163, +5x upsample 0.4032)")
PY
echo "########## done ##########"
