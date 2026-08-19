#!/bin/bash
# Latin parsed off LEMMA SEMANTICS and DECOMPOSED MORPHOLOGY rather than surface forms alone.
#
# Two channels, added together to the released arm's embed and measured against a control that has
# their parameters and none of their information (scripts/make_la_lemvec_config.py):
#
#   lemma vectors   PPMI+SVD over the training treebank's own 529 809 LEMMA tokens, PCA'd to 96d
#                   (build_ppmi_vectors.py -> build_lemma_vectors.py). NOT fastText's Latin table:
#                   that is CC BY-SA 3.0 and the la wheel is CC BY-NC-SA, the same conflict that
#                   keeps Morpheus out of la_macronise (scripts/fetch_fasttext.sh records it).
#                   Distributional rather than orthographic -- `ignis` -> calor, calefactio, aer,
#                   fumus; `frater` -> soror, Iacob, Philippus -- which is what the parser cannot
#                   already read off PREFIX/SUFFIX/SHAPE and its character window.
#   feature hashes  one hash table per morphological category instead of one hash of the whole
#                   FEATS bundle, so `Case=Nom|Number=Sing` and `Case=Nom|Number=Plur` share a case.
#
# The bundle-hash version of the second channel HAS BEEN MEASURED AND WAS WORTHLESS: morphfirst LAS
# 0.7256 against its capacity control's 0.7255 (metrics_la_morphfirst_all_plain.json). That is the
# result being revisited, and it is why the control here is tight rather than nominal.
#
# Phases (run all, or name one): vectors | labels | base | ctl | eval
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
P=la_ittbproiel-sud
CODE="--code scripts/seg_code.py"
TRAIN=corpus_la_ext_macron/$P-train.relabeled_ext.macron.spacy
DEV=corpus_la_ext_union/dev
S=corpus_la_eval_slices

phase=${1:-all}

do_vectors() {
  echo "### VECTORS: PPMI+SVD over the treebank's own lemmas"
  # Train lemmas ONLY. A lemma is an ANNOTATION, not something readable off raw text, so building
  # the table over dev/test lemmas would be leakage of a kind the form-vector tables do not have.
  $PY - <<'PYEOF'
from pathlib import Path
out, sent = [], []
for line in Path("assets_la/la_ittbproiel-sud-train.relabeled_ext.macron.conllu").open(encoding="utf8"):
    line = line.rstrip("\n")
    if not line:
        if sent: out.append(" ".join(sent)); sent = []
        continue
    if line.startswith("#"): continue
    f = line.split("\t")
    if not f[0].isdigit() or f[3] == "PUNCT": continue
    sent.append(f[2] if f[2] != "_" else f[1].lower())
if sent: out.append(" ".join(sent))
Path("corpus_la_lemmas.txt").write_text("\n".join(out) + "\n", encoding="utf8")
print(f"  {len(out)} sentences, {sum(len(s.split()) for s in out)} lemma tokens")
PYEOF
  $PY scripts/build_ppmi_vectors.py --corpus corpus_la_lemmas.txt --dim 300 \
      --out vectors_la_lemma_ppmi.vec
  # --report is the gate, not decoration: the PCA keeps 45 % of the variance, so the neighbourhoods
  # have to be looked at rather than the variance number trusted.
  $PY scripts/build_lemma_vectors.py --in vectors_la_lemma_ppmi.vec --dim 96 \
      --out scripts/la_lemmavec_96.npz --report \
      --probe 'gladius,rex,pater,dico,ambulo,ciuitas,ignis,timeo,frater,sapientia'
}

do_labels() {
  echo "### LABELS: collect the tagger and parser label sets for this arm"
  $PY scripts/make_la_lemvec_config.py --out configs/config_la_lemvec.cfg
  $PY scripts/make_la_lemvec_config.py --out configs/config_la_lemvec_ctl.cfg --control
  $PY scripts/init_aug_labels.py configs/config_la_lemvec.cfg labels_la_lemvec $CODE --passes 2 \
      --paths.train "$TRAIN" --paths.dev "$DEV"
}

train_arm() {  # $1=suffix  $2=config
  echo "### $1"
  $PY -u -m spacy train "$2" $CODE --output training_la_$1/ \
    --paths.train "$TRAIN" --paths.dev "$DEV" > train_la_$1.log 2>&1
  [ -d training_la_$1/model-best ] || { echo "!! FAILED"; tail -20 train_la_$1.log; exit 1; }
  grep -E '^[[:space:]]*[0-9]' train_la_$1.log | tail -1
}

do_eval() {
  echo "### EVAL: released arm vs lemma-vector arm vs its capacity control"
  # Perseus reported apart: it is out-of-domain classical verse (LAS ~53 against ITTB+PROIEL's ~76),
  # and a lexical channel built on a corpus 80 % of which is scholastic prose could easily help one
  # and hurt the other. A combined figure would average that away.
  # model-LAST as well as model-best, and not for completeness. `score_weights` gives tag_acc 0.5
  # against dep_las 0.25, so `model-best` in this parsing experiment is selected half on the
  # TAGGER -- and the two disagree (LAS peaked at step 17 800, TAG at 18 400). The weights are left
  # alone because changing them would desynchronise selection from training_la_aug and confound the
  # comparison this arm exists for; reporting both checkpoints costs one eval and says how much the
  # mis-aimed selector is worth.
  for arm in aug lemvec lemvec_ctl; do
   for ck in model-best model-last; do
    d=training_la_$arm/$ck
    [ -d "$d" ] || { echo "== $arm/$ck: MISSING ($d) -- skip"; continue; }
    echo "== $arm/$ck"
    for sl in all itp perseus; do
      case "$sl" in
        all) t=corpus_la_ext/$P-test.relabeled_ext.spacy ;;
        *)   t=$S/$sl-test.relabeled_ext.spacy ;;
      esac
      [ -f "$t" ] || { printf "   %-8s MISSING\n" "$sl"; continue; }
      printf "   %-8s " "$sl"
      $PY -m spacy evaluate "$d" "$t" --gold-preproc $CODE \
          --output metrics_la_${arm}_${sl}_${ck}.json 2>/dev/null \
        | grep -E '^(TAG|UAS|LAS) ' | tr -s ' \n' ' '
      echo
    done
   done
  done
}

case "$phase" in
  vectors) do_vectors ;;
  labels)  do_labels ;;
  base)    train_arm lemvec configs/config_la_lemvec.cfg ;;
  ctl)     train_arm lemvec_ctl configs/config_la_lemvec_ctl.cfg ;;
  eval)    do_eval ;;
  all)     do_vectors; do_labels; train_arm lemvec configs/config_la_lemvec.cfg; \
           train_arm lemvec_ctl configs/config_la_lemvec_ctl.cfg; do_eval ;;
  *) echo "unknown phase: $phase (vectors|labels|base|ctl|eval)"; exit 1 ;;
esac
echo "DONE: $phase"
