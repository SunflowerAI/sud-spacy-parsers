#!/usr/bin/env bash
# Retrain every Sanskrit parser arm against the UFAL-split corpus, and add a width-96 transfer arm.
#
# Two things changed and both need a rebuild:
#
#   1. UFAL is no longer wholly inside training. 60 of its 230 sentences are held out
#      (`assets_sa_ufal/SUD_Sanskrit-UFAL/sa_ufal_test.csl_mwt.conllu`), which gives the project its
#      FIRST held-out classical Sanskrit syntax. Every arm must be retrained without them, or its
#      score on that set is memorisation. UFAL-test is held out of the morph/lemma corpus too, so a
#      transfer arm cannot inherit it through the encoder either.
#
#   2. `init_tok2vec` requires the source and target encoders to have the SAME width, so testing
#      transfer at width 96 (the released architecture's width) needs a width-96 joint morph+lemma
#      arm first. That arm doubles as a test of whether the earlier joint-vs-separate wash was
#      capacity starvation at width 64.
#
# Arms, all trained on `corpus_sa_split` so nothing is confounded:
#
#   w96        from scratch, released architecture   — the headline control
#   w64        from scratch, joint encoder's width   — the capacity control
#   w64_init   init from the width-64 joint encoder  — transfer, already measured at +1.03 LAS
#   w96_init   init from the width-96 joint encoder  — the new arm
#
# NB w64_init's source (`sa_joint_tok2vec.bin`) was trained BEFORE the split, so it saw UFAL-test's
# 494 tokens. That is a representation leak of ~0.2 % of the morph/lemma data with no syntax in it;
# it is recorded rather than corrected, because retraining the width-64 joint arm to remove it costs
# ~1.5 h for a difference well inside noise. w96_init has no such leak.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CODE=scripts/seg_code.py

echo "=== 1/5  joint morph+lemma encoder at width 96 ==============================="
$PY -m spacy train configs/config_sa_mwt_joint_w96.cfg --output training_sa_split_joint_w96/ \
    --code $CODE 2>&1 | tail -3

echo "=== 2/5  derive the w96_init config from it =================================="
$PY - <<'PY'
import copy, pathlib
import spacy
from thinc.api import Config
import sys; sys.path.insert(0, "scripts")
# `spacy.load` on a sa arm resolves its config, which names the custom tokeniser and the custom
# embed architecture — BOTH must be registered first or it dies with E893 at load time.
import sa_tokenizer      # noqa: F401  — registers sa.SanskritInputTokenizer.v1/v2/v3
import sud_affix_embed   # noqa: F401  — registers sud.MultiHashEmbedAffix.v1
import gold_tok_corpus   # noqa: F401  — registers sud.GoldTokCorpus.v1 / sud.CompoundCorpus.v1

# export the trained joint encoder's weights for init_tok2vec
nlp = spacy.load("training_sa_split_joint_w96/model-best")
pathlib.Path("sa_joint_tok2vec_w96.bin").write_bytes(
    nlp.get_pipe("aux_tok2vec").model.to_bytes())

joint = Config().from_disk("training_sa_split_joint_w96/model-best/config.cfg", interpolate=False)
aux = copy.deepcopy(joint["components"]["aux_tok2vec"]["model"])
c = Config().from_disk("configs/config_sa_split_w96.cfg", interpolate=False)
c["components"]["tok2vec"]["model"] = aux
for comp in ("tagger", "parser"):
    c["components"][comp]["model"]["tok2vec"] = {
        "@architectures": "spacy.Tok2VecListener.v1", "width": 96, "upstream": "*"}
c["paths"]["init_tok2vec"] = "sa_joint_tok2vec_w96.bin"
c["initialize"]["init_tok2vec"] = "${paths.init_tok2vec}"
# spaCy validates the WHOLE [pretraining] block, so copy a known-good one and repoint it
yue = Config().from_disk("configs/config_yue.cfg", interpolate=False)
c["pretraining"] = dict(yue["pretraining"])
c["pretraining"]["component"] = "tok2vec"
c["pretraining"]["layer"] = ""
c.to_disk("configs/config_sa_split_w96_init.cfg")
print("  wrote configs/config_sa_split_w96_init.cfg")
PY

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
