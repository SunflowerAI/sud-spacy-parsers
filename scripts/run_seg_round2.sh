#!/usr/bin/env bash
# Round 2 of the treebank-consistent segmenters: train Cantonese, then score zh / yue / id against
# whatever each language currently ships, on the SAME strict whole-token span match.
#
# "Strict" matters. CLAUDE.md's raw-eval token figures come from spaCy's Scorer, which aligns before
# scoring; the number below is the share of gold tokens reproduced exactly, which is what a caller
# actually receives. The two are not comparable and the strict one is always lower.
#
# Expectations, so the results are read honestly:
#   zh   156 k training characters -> already measured: 0.8385 pkuseg vs 0.8725 ours (+3.4)
#   yue  15 k characters, a TENTH of zh. The Sanskrit presegmenter scored 93.3 F on 20 k sentences
#        and 97.9 on 193 k, so this is far below where that model became good. A loss here is the
#        expected outcome, not a surprise, and pkuseg-for-yue is a strong incumbent (trained on the
#        same treebank, and CLAUDE.md records word-F1 0.947).
#   id   the enclitic split, not whole-word segmentation: 2.5 % of characters carry a boundary and
#        the dev split-location F was 99.79. The real question is end-to-end token F against the
#        current DETERMINISTIC 0.989, which is what the id block below measures.
set -euo pipefail
cd "$(dirname "$0")/.."

# metrics land in metrics/<lang>/. Several evals below send stderr to /dev/null, so a
# missing directory would fail SILENTLY and leave the driver reporting nothing.
mkdir -p metrics/{ar,en,fa,generic,id,ja,ko,la,lzh,misc,release,sa,ta,te,yue,zh}
PY=.venv/bin/python

echo "=== yue: train the character segmenter ======================================"
$PY scripts/train_samhita.py data_seg_yue/train.jsonl data_seg_yue/dev.jsonl \
    models/yue_seg_char --width 64 --depth 6 --epochs 40 2>&1 | tail -4

echo "=== strict whole-token span match, ours vs what ships ======================="
$PY - <<'PY'
import json, sys, pathlib
sys.path.insert(0, "scripts")
from sa_presegment import Presegmenter

def to_tokens(text, labels):
    out, cur = [], ""
    for ch, lb in zip(text, labels):
        cur += ch
        if lb == "= ":
            out.append(cur); cur = ""
    if cur: out.append(cur)
    return out

def spans(toks):
    out, i = set(), 0
    for t in toks:
        out.add((i, i + len(t))); i += len(t)
    return out

def prf(pairs):
    tp = fp = fn = 0
    for gold, pred in pairs:
        g, p = spans(gold), spans(pred)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)

results = {}
for lang, model, pkuseg_model in (("zh", "models/zh_seg_char", "models/zh_gsdboth_pkuseg"),
                                  ("yue", "models/yue_seg_char", "models/yue_pkuseg")):
    f = pathlib.Path(f"data_seg_{lang}/test.jsonl")
    if not f.exists():
        print(f"  {lang}: no test pairs"); continue
    rows = [json.loads(l) for l in f.open(encoding="utf-8")]
    seg = Presegmenter.from_disk(model)
    preds = seg.predict([r["samhita"] for r in rows])
    ours = prf([(r["csl"].split(), to_tokens(r["samhita"], p)) for r, p in zip(rows, preds)])
    base = None
    try:
        import spacy_pkuseg as pk
        import glob
        cand = pkuseg_model if pathlib.Path(pkuseg_model).exists() else \
               (sorted(glob.glob(f"models/{lang}_*pkuseg*")) or [None])[-1]
        if cand:
            k = pk.pkuseg(model_name=cand)
            base = prf([(r["csl"].split(), k.cut(r["samhita"])) for r in rows])
            base_name = cand
    except Exception as e:
        print(f"  {lang}: pkuseg unavailable ({e})")
    print(f"\n  {lang} test:")
    if base:
        print(f"    pkuseg ({base_name:<26})  P {base[0]:.4f} R {base[1]:.4f} F {base[2]:.4f}")
    print(f"    char tagger (ours)                    P {ours[0]:.4f} R {ours[1]:.4f} F {ours[2]:.4f}")
    if base:
        print(f"    delta F  {ours[2] - base[2]:+.4f}"
              + ("   OURS WINS" if ours[2] > base[2] else "   incumbent wins"))
    results[lang] = {"ours": ours[2], "base": base[2] if base else None}

json.dump(results, open("metrics/misc/metrics_seg_round2.json", "w"), indent=2)
print("\n  -> metrics/misc/metrics_seg_round2.json")
PY

echo "=== id: does splitting hold the token F that coarsening guarantees? ========="
$PY - <<'PY'
import json, sys, pathlib, glob
sys.path.insert(0, "scripts")
from sa_presegment import Presegmenter

f = pathlib.Path("data_seg_id/test.jsonl")
rows = [json.loads(l) for l in f.open(encoding="utf-8")]
seg = Presegmenter.from_disk("models/id_seg_char")
preds = seg.predict([r["samhita"] for r in rows])

def to_tokens(text, labels):
    out, cur = [], ""
    for ch, lb in zip(text, labels):
        cur += ch
        if lb == "= ": out.append(cur); cur = ""
    if cur: out.append(cur)
    return out

exact = sum(to_tokens(r["samhita"], p) == r["csl"].split() for r, p in zip(rows, preds))
tp = fp = fn = 0
for r, p in zip(rows, preds):
    g, q = r["csl"].split(), to_tokens(r["samhita"], p)
    gs = set(); i = 0
    for t in g: gs.add((i, i+len(t))); i += len(t)
    qs = set(); i = 0
    for t in q: qs.add((i, i+len(t))); i += len(t)
    tp += len(gs & qs); fp += len(qs - gs); fn += len(gs - qs)
P, R = tp/max(tp+fp,1), tp/max(tp+fn,1)
print(f"  id: {len(rows)} whitespace words, {exact} segmented exactly ({exact/len(rows):.2%})")
print(f"      token P {P:.4f}  R {R:.4f}  F {2*P*R/max(P+R,1e-9):.4f}")
print(f"      vs the current deterministic coarsen-and-merge: 0.989 (no enclitics recovered)")
PY
echo "=== done ===================================================================="
