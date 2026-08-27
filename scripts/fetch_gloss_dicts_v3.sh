#!/usr/bin/env bash
# Wiktionary lemma->English bags for the v3 TEST languages: the deployment fill, simulated.
#
# WHAT THESE ARE FOR. The v3 lexical channel is filled at deployment from an English gloss the user
# supplies. Only THREE of the twenty held-out languages carry a real `Gloss=` column (Chintang,
# Classical Armenian, Yoruba), so a headline resting on those alone would rest on three points.
# These stand in for the gloss a deployer with a dictionary would write, across sixteen.
#
# ⚠ THIS IS NOT A LEAK, AND THE REASON IS WORTH STATING. Wiktionary is an external resource; nothing
# here derives from a test treebank's annotation, so a headword's bag says nothing about any token's
# head or label. What it DOES touch is the lemma column, which the v2 contract does not declare as a
# user input -- so lookup by gold LEMMA is an upper bound, and lookup by surface FORM is the honest
# deployment number. Both are measured; neither is quoted for the other. `--follow-forms` is what
# makes the form route viable at all, by letting an inflected entry inherit its lemma's bag.
#
# ⚠ FOUR TEST LANGUAGES HAVE NO WIKTIONARY AT ALL: Bororo, Chintang, Komi-Zyrian, Xavante. Chintang
# is covered by its own gloss column; the other three cannot be scored on the gloss fill by any
# route, and that is a limit of the evaluation rather than of the channel. Say so in the write-up
# instead of quoting a macro over "the test languages" that quietly means seventeen.
#
# ⚠ kaikki KEEPS SPACES IN THE DIRECTORY AND STRIPS THEM FROM THE FILENAME. Ancient Greek, Old
# Armenian and K'iche' all 404 under the obvious URL and are three of the more useful extracts.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DICT=assets_vec/dict
mkdir -p "$DICT"

# lang : kaikki language name : kaikki lang_code
# aln (Gheg) has no extract of its own and falls back to Albanian, whose code is `sq`.
PAIRS=(
 "cop:Coptic:cop"           "el:Greek:el"              "eu:Basque:eu"
 "grc:Ancient Greek:grc"    "hu:Hungarian:hu"          "hy:Armenian:hy"
 "ka:Georgian:ka"           "lt:Lithuanian:lt"         "lv:Latvian:lv"
 "quc:K'iche':quc"          "th:Thai:th"               "vi:Vietnamese:vi"
 "wo:Wolof:wo"              "xcl:Old Armenian:xcl"     "yo:Yoruba:yo"
 "aln:Albanian:sq"
)

for p in "${PAIRS[@]}"; do
  lc="${p%%:*}"; rest="${p#*:}"; name="${rest%:*}"; code="${rest##*:}"
  out="$DICT/$lc-en.json"
  if [ -s "$out" ]; then echo "have  $lc ($(wc -c < "$out" | tr -d ' ') bytes)"; continue; fi
  dir=$(printf '%s' "$name" | sed "s/ /%20/g; s/'/%27/g")
  file=$(printf '%s' "$name" | sed "s/ //g; s/'//g")
  url="https://kaikki.org/dictionary/${dir}/kaikki.org-dictionary-${file}.jsonl"
  echo "=== $lc  <- $name ($code)"
  $PY scripts/kaikki_anchors.py --url "$url" --lang-code "$code" --out "$out" || echo "  !! $lc failed"
done
echo DONE
