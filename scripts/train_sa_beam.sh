#!/usr/bin/env bash
# Beam-TRAINED parser for the morph-first sa arm (configs/config_sa_mp2_beam.cfg).
#
# Lives in scripts/ rather than the scratchpad because the scratchpad is cleared between sessions —
# a scheduled run written there was lost twice. A long training job needs a durable driver.
#
# WHY BEAM TRAINING AND NOT BEAM DECODING. Decoding with a beam on a GREEDILY trained model is
# useless: measured, rank-0 sits 13 LAS below greedy (0.4206 vs 0.5475), because beam_update_prob=0
# means the action scores were never calibrated as SEQUENCE scores. beam_width=1 reproduces greedy
# on 99.6 % of heads, so the machinery is fine — the scores are not.
#
# WHY IT MIGHT HELP HERE SPECIFICALLY. Pseudo-projective encoding makes non-projectivity a locally
# costly, globally correct choice: `mod||subj` is ~30x rarer than `mod` and only pays once the
# lifted arc is de-projectivised. Greedy decoding can never take that bet. The two cheaper levers
# left the decision procedure alone and did nothing (min_action_freq 30->5: Vedic -0.06 despite
# 45->132 composite moves; non-projective upsampling 3x: recall 0.119->0.237 but precision
# 0.466->0.431, net +0.43).
#
# COST: ~2.4x slower to decode at width 8 (measured: 25 809 -> 10 756 tok/s on the parser alone),
# and 6-12 h to train. Evaluate at widths 1/4/8 before shipping — width is a decode-time choice,
# so an arm trained at 8 can be served at 4 or 1.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
V=corpus_sa_mwt_rl2_norm/sa_vedic-sud-test.relabeled_ext.csl_mwt.spacy
H=corpus_sa_ufal_holdout_norm/sa_ufal_test.relabeled_ext.csl_mwt.spacy
$PY -u -m spacy train configs/config_sa_mp2_beam.cfg --output training_sa_beam_s1/ \
    --code scripts/seg_code.py --system.seed 1 \
    --paths.train corpus_sa_multitask_rl2/train.spacy \
    --paths.dev corpus_sa_multitask_rl2/dev.spacy > train_sa_beam_s1.log 2>&1
$PY scripts/eval_sa_compound.py training_sa_beam_s1/model-best "$V" --reader norm \
    --out metrics_sa_beam_s1_Vedic.json > /dev/null 2>&1
$PY scripts/eval_sa_compound.py training_sa_beam_s1/model-best "$H" --reader norm \
    --out metrics_sa_beam_s1_UFAL.json > /dev/null 2>&1
echo "BEAM_DONE $(date '+%Y-%m-%d %H:%M')"
