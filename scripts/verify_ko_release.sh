#!/bin/bash
# Everything that has to be true of the ko analyser wheel before it goes anywhere.
#
# Each check below exists because its absence has already shipped something broken in this repo:
#
#   1  frozen components byte-identical      the freeze recipe's own claim, asserted with `cmp`
#   2  the arm SEGMENTS                      two sentences in, two sentences out. gold_preproc reads
#                                            99.70 SENT F for an arm that returns one (hazard 4)
#   3  the wheel INSTALLS AND LOADS CLEAN     `scripts/` off sys.path, no MECAB_PATH, no Homebrew —
#                                            the la lemma-vector arm raised E893 here and the build
#                                            itself could not see it
#   4  the installed model PARSES the same    identical parse to the training directory, so what was
#                                            measured is what ships
#   5  the channel is LIVE in the wheel       a model whose analyser silently reads nothing loads
#                                            perfectly and scores like its capacity control
#   6  the declared dependency is enough      python-mecab-ko alone, which is what a user gets
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
ARM=${ARM:-training_ko_anseg_xposwarm/model-best}
BASE=${BASE:-$(cat .ko_release_pick 2>/dev/null || echo training_ko_anseg_s0/model-best)}
WHEEL=$(find build_sud/ko -name 'ko_sud_gsd-*.whl' 2>/dev/null | head -1)
TARGET=${TARGET:-build_sud/ko_install_check}
MECAB_TARGET=${MECAB_TARGET:-build_sud/ko_mecab_check}

echo "### 1  frozen components byte-identical (the freeze recipe's own claim)"
# ⚠ The TAGGER is deliberately not in this list. Every other component travels up the chain
# unchanged — which is what lets the parser figures measured on the base stand for the wheel without
# re-verification — but the graft REPLACES the tagger, moving it behind the morphologiser and
# conditioning it on predicted UPOS+MORPH. Asserting it identical would be asserting the graft did
# not happen, so it is asserted DIFFERENT instead, and the pipeline order is checked directly.
for c in tok2vec parser; do
  cmp -s "$BASE/$c/model" "$ARM/$c/model" \
    && echo "    $c: identical to the base the parser was measured on" \
    || { echo "    $c: DIFFERS — the freeze recipe did not hold"; exit 1; }
done
for c in morphologizer; do
  cmp -s "training_ko_anseg_morph/model-best/$c/model" "$ARM/$c/model" \
    && echo "    $c: identical to the morph arm" \
    || { echo "    $c: DIFFERS against the morph arm"; exit 1; }
done
cmp -s "training_ko_anseg_lemma/model-best/lemmatizer/model" "$ARM/lemmatizer/model" \
  && echo "    lemmatizer: identical to the lemma arm" \
  || { echo "    lemmatizer: DIFFERS against the lemma arm"; exit 1; }
cmp -s "$BASE/tagger/model" "$ARM/tagger/model" \
  && { echo "    tagger: IDENTICAL to the base's — the graft did not happen"; exit 1; } \
  || echo "    tagger: differs from the base's, as the graft requires"
$PY - "$ARM" <<'EOF'
import json, pathlib, sys
pipe = json.loads((pathlib.Path(sys.argv[1]) / "meta.json").read_text())["pipeline"]
ok = pipe.index("morphologizer") < pipe.index("tagger")
print(f"    pipeline {pipe}")
sys.exit(0 if ok else 1)
EOF

echo "### 2  the arm segments (the check gold_preproc cannot make)"
MECAB_PATH="${MECAB_PATH:-/opt/homebrew/lib/libmecab.dylib}" $PY - "$ARM" <<'EOF'
import sys, pathlib
sys.path.insert(0, "scripts")
from spacy import util
util.import_file("cli_code", pathlib.Path("scripts/seg_code.py"))
import spacy
nlp = spacy.load(sys.argv[1])
doc = nlp("잡스는 워즈니악에게 도움을 청했다. 워즈니악은 게임을 설계했다.")
n, roots = len(list(doc.sents)), sum(1 for t in doc if t.head.i == t.i)
print(f"    two sentences in -> {n} out, {roots} self-headed roots")
sys.exit(0 if n == 2 and roots == 2 else 1)
EOF

[ -n "$WHEEL" ] || { echo "### no wheel in build_sud/ko — run build_ko_release.sh wheel first"; exit 1; }
echo "### 3  install the wheel into a clean target, with python-mecab-ko and NOTHING else"
rm -rf "$TARGET" "$MECAB_TARGET"
$PY -m pip install --quiet --target "$TARGET" "$WHEEL" >/dev/null
$PY -m pip install --quiet --target "$MECAB_TARGET" python-mecab-ko >/dev/null
echo "    installed $(basename "$WHEEL")"

echo "### 4-6  load it with scripts/ off sys.path, through the backend the wheel DECLARES"
# ⚠ `env -u MECAB_PATH` is NOT enough to make this test honest. natto-py is in this venv and finds
# libmecab without the variable, so an unpinned run silently exercises the development backend and
# proves nothing about what a user gets. `KO_ANALYSER_BACKEND=python-mecab-ko` pins it to the one
# declared in the wheel's requirements — the whole point of the check.
env -u MECAB_PATH KO_ANALYSER_BACKEND=python-mecab-ko \
  PYTHONPATH="$TARGET:$MECAB_TARGET" \
  $PY - "$ARM" <<'EOF'
import sys, pathlib
# ⚠ scripts/ is deliberately NOT on the path: that directory is what makes the TRAINING copy load,
# and a --code file missing from the wheel raises E893 only without it.
assert not any("scripts" in p for p in sys.path), sys.path
import ko_sud_gsd, spacy
nlp = ko_sud_gsd.load()
print(f"    loaded {nlp.meta['name']} {nlp.meta['version']}; pipeline {nlp.pipe_names}")

text = "잡스는 워즈니악에게 도움을 청했다. 워즈니악은 게임을 설계했다."
doc = nlp(text)
print(f"    parses: {len(list(doc.sents))} sentences, "
      f"{[(t.text, t.dep_, t.head.text, t.tag_, t.lemma_) for t in doc][:3]}")

# 5 — the channel must be LIVE, not silently reading nothing. The extractor is asked directly.
import ko_analyser
found = []
for pipe in ("parser", "tagger"):
    m = nlp.get_pipe(pipe).model
    found += [n for n in m.walk() if n.name in ("extract_ko_morph_ids", "extract_ko_tag_sets")]
tok2vec = nlp.get_pipe("tok2vec").model
found += [n for n in tok2vec.walk() if n.name in ("extract_ko_morph_ids", "extract_ko_tag_sets")]
assert found, "the analyser channel is NOT in the loaded model"
ids = [n for n in found if n.name == "extract_ko_morph_ids"][0]
print(f"    channel present, trained against {ids.attrs['ko_backend']!r}, "
      f"running on {ko_analyser.fingerprint()!r}")
out, _ = ids([nlp.make_doc("잡스는 잡스가 잡스를")], False)
import numpy
a = numpy.asarray(out[0])
assert len(set(a[:, 0].tolist())) == 1 and len(set(a[:, 1].tolist())) == 3, a
print("    three inflections of one stem -> 1 lexical key, 3 functional keys")

# 4 — the installed model must parse EXACTLY as the training directory does.
sys.path.insert(0, "scripts")
util = __import__("spacy").util
util.import_file("cli_code", pathlib.Path("scripts/seg_code.py"))
ref = spacy.load(sys.argv[1])
a, b = nlp(text), ref(text)
same = all((x.text, x.head.i, x.dep_, x.tag_, x.lemma_, str(x.morph)) ==
           (y.text, y.head.i, y.dep_, y.tag_, y.lemma_, str(y.morph)) for x, y in zip(a, b))
print(f"    installed parse identical to {sys.argv[1]}: {same}")
sys.exit(0 if same else 1)
EOF
echo "### ALL CHECKS PASSED"
