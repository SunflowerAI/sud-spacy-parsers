#!/bin/bash
# Compare, on TEST, the three XPOS arms of a language:
#
#   released      the shipping tagger -- a LISTENER on the base arm's shared encoder, trained
#                 beside the parser and blind to UPOS/FEATS
#   _xposdown_ctl the capacity control -- same dedicated encoder as below, WITHOUT POS/MORPH
#   _xposdown     the tagger moved behind the morphologiser, reading POS + the hashed MORPH
#                 bundle as well as its own token embedding
#   _xposfeat     the same, but with ONE EMBEDDING TABLE PER FEATURE (sud.MultiHashEmbedFeats.v1)
#                 instead of the single hashed bundle
#   _xpostop_ctl  the tightest control there is: the RELEASED tagger's own Tok2VecListener on the
#                 frozen shared encoder, head retrained, no conditioning. It should reproduce the
#                 released row, and does (within 0.13) -- which is what licenses reading the next
#   _xpostop      the same listener, with the per-feature morphology concatenated UNDER THE SOFTMAX
#                 rather than convolved in at the bottom (sud.Tok2VecPlusFeats.v1)
#   _xposwarm     as _xpostop, but WARM-STARTED: the released tagger's head is copied in and the
#                 side channel's columns zeroed, so training begins AS the released tagger. This is
#                 also the only variant that covers la and en_gum, whose released taggers carry
#                 their own HashEmbedCNN (copied too) rather than a listener.
#   _xposwarm_ctl the same warm start with NO side channel: released tagger, fine-tuned. It should
#                 return to the released row, and does.
#
# The control is the row that matters: released -> ctl is the ARCHITECTURE change (listener to a
# dedicated encoder), ctl -> xposdown is the CONDITIONING. Reading released -> xposdown alone
# would credit the feature with both.
#
# Everything but en is evaluated --gold-preproc (CLAUDE.md: a tokenisation mismatch collapses
# alignment); en's spacing matches, so it does not need it.
#
#   bash scripts/eval_xpos.sh ar en fa ja id ko la zh yue lzh sa
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY=.venv/bin/python
export MECAB_PATH=${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}

released_arm() { case "$1" in
  en) echo training_en_sud ;;         en_gum) echo training_en_gum_sud_xpos ;;
  ar) echo training_ar_sud ;;         fa) echo training_fa_sud ;;
  id) echo training_id_sud ;;         yue) echo training_yue_sud ;;
  la) echo training_la_aug_sud_xpos ;; sa) echo training_sa_multitask ;;
  ko) echo training_ko_eojeol_lemma ;; lzh) echo training_lzh_trad_sud ;;
  zh) echo training_zh_trad_lemma ;;  ja) echo training_ja_lemma ;;
esac; }

test_corpus() { case "$1" in
  en) echo corpus_en_ewt_ext/en_ewt-sud-test.relabeled_ext.spacy ;;
  en_gum) echo corpus_en_gum_ext/en_ewtgum-sud-test.relabeled_ext.spacy ;;
  ar) echo corpus_ar_ext/ar_padt-sud-test.relabeled_ext.spacy ;;
  fa) echo corpus_fa_ext/fa_perdt-sud-test.relabeled_ext.spacy ;;
  ja) echo corpus_ja_ext/ja_gsd-sud-test.relabeled_ext.spacy ;;
  id) echo corpus_id_split/id_gsd-sud-test.relabeled_ext.spacy ;;
  ko) echo corpus_ko_eojeol/ko_gsd-sud-test.relabeled_ext.spacy ;;
  la) echo corpus_la_ext/la_ittbproiel-sud-test.relabeled_ext.spacy ;;
  en_gum) : ;;
  zh) echo corpus_zh_trad/zh_gsd-sud-test.relabeled_ext.spacy ;;
  yue) echo corpus_yue_ext/yue_hk-sud-test.relabeled_ext.spacy ;;
  lzh) echo corpus_lzh_trad/lzh_kyoto-sud-test.relabeled_ext.udep_ruled.punct.rulemerged.spacy ;;
  sa) echo corpus_sa_multitask/test.spacy ;;
esac; }

printf "%-7s %-16s %8s %8s %8s\n" lang arm TAG POS MORPH
for lang in "$@"; do
  t=$(test_corpus "$lang"); GP=--gold-preproc; [ "$lang" = en ] && GP=""
  [ -f "$t" ] || { echo "$lang: test corpus $t missing -- skip"; continue; }
  # XPOS_ARMS restricts the comparison to a subset of suffixes, so a later variant can be scored
  # without re-running every earlier one: XPOS_ARMS="_xposwarm_ctl _xposwarm"
  arms=()
  for suf in ${XPOS_ARMS:-_xposdown_ctl _xposdown _xposfeat _xpostop_ctl _xpostop _xposwarm_ctl _xposwarm}; do
    arms+=("training_${lang}${suf}")
  done
  for arm in "$(released_arm "$lang")" "${arms[@]}"; do
    d=$arm/model-best
    [ -d "$d" ] || { printf "%-7s %-16s %8s\n" "$lang" "$(basename $arm)" "MISSING"; continue; }
    out=metrics_${lang}_$(basename $arm).json
    $PY -m spacy evaluate "$d" "$t" $GP --code scripts/seg_code.py --output "$out" >/dev/null 2>&1
    $PY -c "
import json;p=json.load(open('$out'))
f=lambda k: f'{100*p[k]:8.2f}' if isinstance(p.get(k),(int,float)) else f'{\"-\":>8s}'
print(f'{\"$lang\":<7s} {\"$(basename $arm)\":<16s} {f(\"tag_acc\")} {f(\"pos_acc\")} {f(\"morph_acc\")}')" 2>/dev/null \
      || printf "%-7s %-16s %8s\n" "$lang" "$(basename $arm)" "EVAL-FAIL"
  done
done
