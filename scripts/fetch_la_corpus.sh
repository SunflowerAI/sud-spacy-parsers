#!/usr/bin/env bash
# Fetch a large Latin corpus for homegrown vectors.
#
# Deliberately NOT Common Crawl. cc.la is what the fastText Latin vectors are already trained on and
# it covers only 52.0 % of our treebank types at @1 27.5 %; repeating that source would repeat the
# weakness. These four are the domains our treebanks actually come from -- classical (Perseus),
# late/medieval scholastic (ITTB is Aquinas), and the Vulgate (PROIEL) -- plus modern written Latin.
set -u
cd "$(dirname "$0")/.."
D=assets_vec/la_corpus
mkdir -p "$D"
get () { [ -s "$D/$2" ] && { echo "have $2"; return; }; echo "fetch $2"; curl -sSL --max-time 7200 "$1" -o "$D/$2.part" && mv "$D/$2.part" "$D/$2"; }

get https://dumps.wikimedia.org/lawikisource/latest/lawikisource-latest-pages-articles.xml.bz2 wikisource.xml.bz2
get https://dumps.wikimedia.org/lawiki/latest/lawiki-latest-pages-articles.xml.bz2            wikipedia.xml.bz2
get https://codeload.github.com/cltk/latin_text_latin_library/tar.gz/refs/heads/master        latin_library.tar.gz
get https://codeload.github.com/PerseusDL/canonical-latinLit/tar.gz/refs/heads/master         perseus.tar.gz
ls -la "$D"
echo DONE
