#!/usr/bin/env bash
# Train treebank-consistent word segmenters, reusing the Sanskrit character tagger.
#
# Waits for the Sanskrit multi-task run to finish before starting, so the two do not contend for
# CPU. Pass a pid to wait on, or nothing to start immediately.
#
#   zh  the real target. pkuseg scores strict token F 0.837 against GSDSimp — about one token in
#       six is wrong on raw input, and the errors are broad, not just the demonstrative+classifier
#       merges (这/本书 for 这/本/书) that prompted this. A tagger trained ON the treebank is
#       consistent with it by construction. Caveat: GSD is ~4 k sentences, where the Sanskrit
#       presegmenter scored 93.3 on 20 k and 97.9 on 193 k, so the ceiling here is uncertain.
#
#   id  a different job: the chunks are already whitespace words and 0.5 % of them split, so the
#       model only has to find the `-nya`/`-lah` enclitic boundaries that `coarsen_id.py` currently
#       merges away. Trading a deterministic 0.989 for a statistical model that recovers real
#       syntax — run second, and judge on whether it holds the token F.
#
# ko and ja are deliberately absent: ko is already at strict token F 1.0000 (mecab is deterministic
# and the treebank was retokenised to match it) and ja at 99.4 via SudachiPy. There is nothing to
# win in either, and a learned model could only move them off those numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID (Sanskrit multi-task) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "    it finished; starting"
fi

echo "=== zh: train the character segmenter ======================================="
$PY scripts/train_samhita.py data_seg_zh/train.jsonl data_seg_zh/dev.jsonl \
    models/zh_seg_char --width 64 --depth 6 --epochs 30 2>&1 | tail -6

echo "=== zh: score against pkuseg on the SAME test set ==========================="
$PY - <<'PY'
import json, pathlib, sys
sys.path.insert(0, "scripts")
from sa_presegment import Presegmenter

rows = [json.loads(l) for l in open("data_seg_zh/test.jsonl", encoding="utf-8")]
seg = Presegmenter.from_disk("models/zh_seg_char")
preds = seg.predict([r["samhita"] for r in rows])

def to_tokens(text, labels):
    out, cur = [], ""
    for ch, lb in zip(text, labels):
        cur += ch
        if lb == "= ":
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out

def spans(toks):
    out, i = set(), 0
    for t in toks:
        out.add((i, i + len(t))); i += len(t)
    return out

def prf(pred_fn):
    tp = fp = fn = 0
    for r, p in zip(rows, preds):
        g = spans(r["csl"].split())
        q = spans(pred_fn(r, p))
        tp += len(g & q); fp += len(q - g); fn += len(g - q)
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)

ours = prf(lambda r, p: to_tokens(r["samhita"], p))
try:
    import spacy_pkuseg as pk
    k = pk.pkuseg(model_name="models/zh_gsdboth_pkuseg")
    base = prf(lambda r, p: k.cut(r["samhita"]))
except Exception as e:
    base = None
    print(f"  (pkuseg unavailable: {e})")

print(f"\n  zh test, strict whole-token span match:")
if base:
    print(f"    pkuseg (shipped)   P {base[0]:.4f}  R {base[1]:.4f}  F {base[2]:.4f}")
print(f"    char tagger (ours) P {ours[0]:.4f}  R {ours[1]:.4f}  F {ours[2]:.4f}")
if base:
    print(f"    delta F            {ours[2] - base[2]:+.4f}")
json.dump({"pkuseg_f": base[2] if base else None, "ours_f": ours[2]},
          open("metrics/zh/metrics_zh_seg_pilot.json", "w"), indent=2)
PY

echo "=== id: build pairs and train ==============================================="
ID=$(ls assets_id/SUD_Indonesian-GSD/id_gsd-sud-train.conllu 2>/dev/null || true)
if [ -n "$ID" ]; then
  D=$(dirname "$ID")
  for s in train dev test; do
    $PY scripts/make_seg_pairs.py "$D/id_gsd-sud-$s.conllu" "data_seg_id/$s.jsonl" --min-chunk 2 \
      2>&1 | sed 's/^/  id-/'
  done
  $PY scripts/train_samhita.py data_seg_id/train.jsonl data_seg_id/dev.jsonl \
      models/id_seg_char --width 64 --depth 6 --epochs 30 2>&1 | tail -6
else
  echo "  id treebank not found at assets_id/SUD_Indonesian-GSD -- skipped"
fi

echo "=== done ===================================================================="
