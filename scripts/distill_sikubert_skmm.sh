#!/bin/bash
# Launch (or RESUME) the SikuBERT -> 4L-312D general distillation on skmm. bash 3.2 safe (no wait -n).
# Run ON skmm:   bash ~/distill_siku/scripts/distill_sikubert_skmm.sh
# Or from here:  ssh skmm 'nohup bash ~/distill_siku/scripts/distill_sikubert_skmm.sh > /dev/null 2>&1 & disown'
# Re-running the same command resumes from runs/$RUN/ckpt.pt (model+optimiser+step+data position).
# Env overrides: RUN TOTAL_TOKENS THREADS BATCH_TOKENS LR WARMUP
set -u
ROOT="${ROOT:-$HOME/distill_siku}"
RUN="${RUN:-siku_4l312_v1}"
PY="${PY:-$HOME/miniconda3/envs/sudspacy/bin/python}"
cd "$ROOT" || exit 1
mkdir -p "runs/$RUN"
if pgrep -f "distill_sikubert.py.*--out runs/$RUN" > /dev/null; then
  echo "already running for $RUN" >&2; exit 1
fi
PYTHONUNBUFFERED=1 exec "$PY" -u scripts/distill_sikubert.py \
  --teacher teacher \
  --corpus data/corpus_lzh_kanripo_leakfree.txt \
  --out "runs/$RUN" \
  --total-tokens "${TOTAL_TOKENS:-200e6}" \
  --threads "${THREADS:-8}" \
  --batch-tokens "${BATCH_TOKENS:-8192}" \
  --lr "${LR:-5e-4}" --warmup "${WARMUP:-1000}" \
  >> "runs/$RUN/stdout.log" 2>&1
