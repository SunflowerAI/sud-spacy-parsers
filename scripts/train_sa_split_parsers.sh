#!/usr/bin/env bash
# Stages 3-5 of `train_sa_split_arms.sh`, split out so the width-96 joint encoder (stage 1, ~2.5 h)
# is not retrained after the stage-2 export crashed on a missing registry import.
#
# Stage 1 and 2 are already done: `training_sa_split_joint_w96/` holds the trained joint arm and
# `configs/config_sa_split_w96_init.cfg` + `sa_joint_tok2vec_w96.bin` were written from it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CODE=scripts/seg_code.py

echo "=== 3/5  parser arms ========================================================="
for arm in w96 w64 w64_init w96_init; do
  echo "--- $arm ---"
  $PY -m spacy train "configs/config_sa_split_${arm}.cfg" \
      --output "training_sa_split_${arm}/" --code $CODE 2>&1 | tail -3
done

echo "=== 4/5  evaluate on held-out UFAL and Vedic ================================="
$PY - <<'PY'
import json, subprocess
rows = []
for arm in ("w96", "w64", "w64_init", "w96_init"):
    r = {"arm": arm}
    for name, corpus in (("UFAL", "corpus_sa_split/ufal_test.spacy"),
                         ("Vedic", "corpus_sa_split/vedic_test.spacy")):
        subprocess.run([".venv/bin/python", "scripts/eval_sa_compound.py",
                        f"training_sa_split_{arm}/model-best", corpus, "--out", "/tmp/m.json"],
                       check=True, capture_output=True)
        m = json.load(open("/tmp/m.json"))
        r[name] = (m["tag_acc"], m["dep_uas"], m["dep_las"])
    rows.append(r)
print(f"\n  {'arm':<10}  {'UFAL (held out, 60 sents)':<26}  {'Vedic test':<26}")
print(f"  {'':<10}  {'tag':>7} {'UAS':>7} {'LAS':>7}     {'tag':>7} {'UAS':>7} {'LAS':>7}")
for r in rows:
    u, v = r["UFAL"], r["Vedic"]
    print(f"  {r['arm']:<10}  {u[0]:7.4f} {u[1]:7.4f} {u[2]:7.4f}     "
          f"{v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f}")
json.dump(rows, open("metrics_sa_split_arms.json", "w"), indent=2)
print("\n  -> metrics_sa_split_arms.json")
PY

echo "=== 5/5  done ================================================================"
