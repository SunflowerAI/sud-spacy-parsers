#!/bin/bash
# Train the three Cantonese arms (base / verb-rl / ext) into training_yue{,_rl,_ext}.
# config_yue.cfg initialises the tok2vec from the dual-script Mandarin model (init_tok2vec =
# zh_both_tok2vec.bin, extracted from training_zh_both/model-best) and fine-tunes; tagger+parser
# train fresh (label sets differ). This Mandarin-init is the DEFAULT — it lifts TAG/UAS (and
# baseline LAS) over from-scratch on this 804-sentence treebank; comp:obl F is unchanged within
# the 100-sentence test noise. Prereq: the .spacy corpora (corpus_yue*, built by the relabel
# pipeline) and zh_both_tok2vec.bin must exist.
cd /Users/sivakalyan/Linguistics/Tools/SUD-spaCy || exit 1
PY=.venv/bin/python
CFG=configs/config_yue.cfg
CODE="--code scripts/yue_tokenizer.py"

# corpus dir per arm, and the per-split filename SUFFIX (after the split: yue_hk-sud-<split><suf>.spacy)
declare -A CORP=( [base]=corpus_yue [rl]=corpus_yue_rl       [ext]=corpus_yue_ext )
declare -A SUF=(  [base]=""         [rl]=.relabeled          [ext]=.relabeled_ext )

for arm in base rl ext; do
  corp=${CORP[$arm]}; suf=${SUF[$arm]}
  out=training_yue_init_${arm}; met=metrics_yue_init_${arm}.json
  echo "############## TRAIN yue_init_$arm ##############"
  $PY -m spacy train $CFG $CODE --output $out/ \
    --paths.init_tok2vec zh_both_tok2vec.bin \
    --paths.train $corp/yue_hk-sud-train$suf.spacy \
    --paths.dev   $corp/yue_hk-sud-dev$suf.spacy > train_yue_init_$arm.log 2>&1
  if [ ! -d $out/model-best ]; then echo "!! $arm FAILED"; tail -10 train_yue_init_$arm.log; continue; fi
  $PY -m spacy evaluate $out/model-best $corp/yue_hk-sud-test$suf.spacy $CODE \
    --gold-preproc --output $met 2>&1 | grep -E 'TAG|UAS|LAS' | head -3
done
echo "YUE INIT DONE"
