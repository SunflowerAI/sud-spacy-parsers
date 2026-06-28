#!/bin/bash
# Rebuild the id (coarsened) and ko (morpheme functional-head) RELABELED corpora that the released
# id_sud_gsd / ko_sud_gsd models train on (the corpus_* dirs were cleaned up). Deterministic
# transforms over the surviving *.relabeled.conllu — no LLM calls. Mirrors the id/ko corpus steps
# of scripts/relabel_retrain_retok.sh, without the (non-seg) trainings. Then retrain_seg.sh id ko.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
export MECAB_PATH=/opt/homebrew/lib/libmecab.dylib
PY=.venv/bin/python

echo "### id: coarsen enclitics on the relabeled file -> corpus_id_coarse_rl"
mkdir -p assets_id_coarse_rl corpus_id_coarse_rl
for s in train dev test; do
  $PY scripts/coarsen_id.py assets_id/SUD_Indonesian-GSD/id_gsd-sud-$s.relabeled.conllu \
    assets_id_coarse_rl/id_gsd-coarse-$s.relabeled.conllu
  $PY -m spacy convert assets_id_coarse_rl/id_gsd-coarse-$s.relabeled.conllu \
    corpus_id_coarse_rl/ --converter conllu -n 10 >/dev/null 2>&1
done

echo "### ko: morpheme retokenise the relabeled file -> corpus_ko_retok_rl"
mkdir -p assets_ko_retok_rl corpus_ko_retok_rl
for s in train dev test; do
  $PY scripts/retokenize.py --lang ko assets_ko/SUD_Korean-GSD/ko_gsd-sud-$s.relabeled.conllu \
    assets_ko_retok_rl/ko_gsd-retok-$s.relabeled.conllu
  $PY -m spacy convert assets_ko_retok_rl/ko_gsd-retok-$s.relabeled.conllu \
    corpus_ko_retok_rl/ --converter conllu -n 10 >/dev/null 2>&1
done

echo "### done. corpora:"
ls corpus_id_coarse_rl corpus_ko_retok_rl
