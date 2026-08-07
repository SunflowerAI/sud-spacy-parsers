#!/usr/bin/env bash
# A REAL tokeniser for Classical Chinese, replacing "one Han character = one token".
#
# WHY. CLAUDE.md records lzh's tokeniser as one character per token, but the Kyoto treebank is not:
# 26,190 of its 920,780 tokens (2.84 %) are multi-character -- 君子 孔子 孟子 夫子 匈奴 契丹 五十 七十,
# i.e. honorific compounds, philosophers' names, ethnonyms and numerals, 5,069 types in all. So the
# released tokeniser splits 孔子 into 孔 + 子 and can never exceed ~97 % token F.
#
# This has been INVISIBLE in every published lzh figure, and that is worth understanding rather than
# just fixing: `gold_preproc` bypasses the tokeniser at evaluation, and `sud.GoldTokCorpus.v1` makes
# the parser segmenter-agnostic, so nothing in the metrics touches tokenisation. It would show up
# only in raw end-to-end token F, which does not appear to have been measured for lzh at all.
#
# The machinery already exists -- `sud.CharSegTokenizer.v1`, the treebank-trained character tagger
# that serves zh (0.8385 -> 0.9210 strict token F) and id. One rewrite label per character. lzh
# should be far easier than zh: the boundary is nearly always "split", and the exceptions are a
# largely CLOSED lexical set of names and titles.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
A=assets_lzh/SUD_Classical_Chinese-Kyoto-Both
SRC=lzh_kyotoboth-sud   # the punctuation-restored, rule-merged generation the release uses
SUF=relabeled_ext.udep_ruled.punct.rulemerged

mkdir -p data_seg_lzh
for s in train dev test; do
  f="$A/${SRC}-${s}.${SUF}.conllu"
  [ -f "$f" ] || { echo "  missing $f"; exit 1; }
  $PY scripts/make_seg_pairs.py "$f" "data_seg_lzh/$s.jsonl" --min-chunk 1 2>&1 | sed 's/^/  /'
done

# Same architecture as the zh/id segmenters. Han has a large character inventory, so the embedding
# does the work; depth 6 matches what id used.
$PY scripts/train_samhita.py data_seg_lzh/train.jsonl data_seg_lzh/dev.jsonl \
    models/lzh_seg_char --width 64 --depth 6 --epochs 30 2>&1 | tail -8

# The number that matters is STRICT TOKEN F against the treebank's own tokenisation -- not character
# accuracy, which is ~97 % for a model that never splits anything and would look like success.
$PY - <<'PYEOF'
import json, pathlib, sys
sys.path.insert(0, "scripts")
try:
    from sa_presegment import Presegmenter
except Exception as e:
    print(f"  scorer unavailable: {type(e).__name__}: {e}"); raise SystemExit
p = pathlib.Path("models/lzh_seg_char")
if not p.exists():
    print("  no model to score"); raise SystemExit
m = Presegmenter.from_disk(p)
gold_tok = gold_pred = hit = 0
for ln in open("data_seg_lzh/test.jsonl", encoding="utf-8"):
    ex = json.loads(ln)
    text, gold = ex.get("text") or ex.get("raw"), ex.get("words") or ex.get("tokens")
    if not text or not gold: continue
    pred = m.segment(text) if hasattr(m, "segment") else None
    if pred is None: continue
    def spans(ws):
        out, i = [], 0
        for w in ws: out.append((i, i + len(w))); i += len(w)
        return set(out)
    g, q = spans(gold), spans(pred)
    gold_tok += len(g); gold_pred += len(q); hit += len(g & q)
if gold_tok and gold_pred:
    p_ = hit / gold_pred; r = hit / gold_tok
    print(f"  strict token  P {p_:.4f}  R {r:.4f}  F {2*p_*r/max(p_+r,1e-9):.4f}")
    print("  baseline (split every character) would be ~0.97 by character accuracy but far lower here")
PYEOF
