#!/usr/bin/env bash
# Train FLORET subword vectors and install them as a spaCy vectors model.
#
# Floret (Explosion's fastText fork) hashes CHARACTER N-GRAMS into a fixed bucket table, so EVERY
# string composes a vector and there is no OOV at all -- `ko_core_news_lg` ships this way and scores
# 100 % coverage on our eojeol tokenisation despite being built on a different segmentation.
# That property is what a fixed |V| lookup table cannot offer: `library` is simply absent from it.
#
# Worth most exactly where the corpus is small and inflection is heavy:
#   sa  5.69M tokens, 45.8 % hapax even on unsandhied forms
#   la  790k tokens, 51.5 % hapax -- and training on OUR OWN treebanks avoids fastText's
#       CC BY-SA 3.0, which is incompatible with the CC BY-NC-SA Latin wheel
#
#   bash scripts/train_floret.sh sa corpus_sa_unsandhied.txt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
lang=$1; corpus=$2
dim=${DIM:-300}; minn=${MINN:-4}; maxn=${MAXN:-5}
bucket=${BUCKET:-50000}; hashcount=${HASHCOUNT:-2}; minc=${MINCOUNT:-5}
echo "  training floret: $lang  dim=$dim  n-grams $minn-$maxn  bucket=$bucket  hashCount=$hashcount"
$PY - "$lang" "$corpus" "$dim" "$minn" "$maxn" "$bucket" "$hashcount" "$minc" <<'PYEOF'
import sys, floret
lang, corpus, dim, minn, maxn, bucket, hashcount, minc = sys.argv[1:9]
m = floret.train_unsupervised(
    corpus, model="cbow", dim=int(dim), minn=int(minn), maxn=int(maxn),
    mode="floret", hashCount=int(hashcount), bucket=int(bucket),
    minCount=int(minc), epoch=10, thread=6)
m.save_model(f"vectors_{lang}_floret.bin")
m.save_floret_vectors(f"vectors_{lang}_floret.vec")   # FLORET header (bucket dim minn maxn ...);
                                                       # save_vectors() writes a word2vec header and
                                                       # spacy init vectors --mode floret rejects it
print(f"    wrote vectors_{lang}_floret.{{bin,vec}}")
PYEOF
$PY -m spacy init vectors "$lang" "vectors_${lang}_floret.vec" "vectors_${lang}_floret" \
    --mode floret 2>&1 | tail -3
