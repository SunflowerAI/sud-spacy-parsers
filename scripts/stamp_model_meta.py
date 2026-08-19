#!/usr/bin/env python
"""Stamp licence / author / attribution into a model's meta.json before `spacy package`.

**A model directory built by `spacy train` has an EMPTY `license` field.** The released wheels
carry one only because their meta was set at some point in their history — so an arm rebuilt from
training and packaged straight away ships with no licence at all, and nothing in the build warns.
That is not cosmetic: every model here is a derivative of CC BY-SA treebanks, and the lzh arm also
carries Kanseki Repository punctuation, so the attribution is an obligation rather than a courtesy.
It was caught only by diffing a script-built wheel against a hand-assembled one.

Licences differ per language and must not be blanket-applied: la is CC BY-**NC**-SA, forced by its
three NonCommercial treebanks, while the rest are kept free of NonCommercial sources deliberately.

A language can have more than one ARM with different SOURCES, so the registry key is `--arm`, not
`--lang`: `en` ships two wheels and only the EWT+GUM one owes GUM's attribution, and keying the
tables on the language code alone would give it to both. `--arm` defaults to `--lang`, so every
single-arm language is unaffected. `--license` overrides outright, for a one-off.

Usage:
    stamp_model_meta.py MODEL_DIR --lang lzh [--description TEXT]
    stamp_model_meta.py MODEL_DIR --lang en --arm en_gum      # the EWT+GUM English wheel
"""
import argparse
import json
import pathlib

AUTHOR = "Sunflower AI"
URL = "https://github.com/SunflowerAI/sud-spacy-parsers"
# Keyed by ARM, not language. la is NonCommercial (ITTB + PROIEL + Perseus are all CC BY-NC-SA).
# ar is NonCommercial too: SUD_Arabic-PADT has always been CC BY-NC-SA 3.0 (its LICENSE.txt:
# "distributed under the same license terms as PADT 1.0"), and the wheel had been falling through
# to the CC BY-SA 4.0 default since v0.1.0 -- a survey of every assets_*/ found ar to be the only
# arm where the declaration and the training data disagreed. Both are declared at 4.0 although
# their sources are 3.0 (and 2.5): ShareAlike permits licensing an adaptation under a later version.
#
# ⚠ en_gum WAS in this table and is NOT any more (2026-08-17). It ships EWT + the ten non-NC GUM
# genres, and the open question was whether GUM's ANNOTATIONS are NonCommercial whatever the
# document -- GUM's LICENSE.txt opens "The treebank is licensed under CC BY-NC-SA 4.0", and
# annotations are what a trained model absorbs, so this arm shipped NC regardless of the filter.
# Amir Zeldes (GUM's maintainer) answered the question directly by email: the annotations are
# produced at Georgetown under **CC BY**, and the NC comes only from the individual underlying
# documents, whose own licences must be respected -- "if you only use documents without the NC
# license, I don't see an issue in using the data for commercial purposes". The filter is therefore
# load-bearing after all, and this arm is CC BY-SA 4.0 like every other: the SA is forced by EWT
# (CC BY-SA 4.0) and by the ShareAlike text sources among the kept genres, not by GUM as a whole.
# What survives is the ATTRIBUTION, which CC BY makes an obligation -- see SOURCES below, which
# must keep naming the corpus, its website and its annotators.
LICENSE = {"la": "CC BY-NC-SA 4.0", "ar": "CC BY-NC-SA 4.0"}
DEFAULT_LICENSE = "CC BY-SA 4.0"

# Runtime imports a wheel needs beyond spaCy, declared here so `pip install` yields a model that
# LOADS. Getting this wrong is a known failure in this project: the ja wheel once required only
# `spacy` and hit an ImportError on every load. ja/yue/zh already declare theirs elsewhere (spaCy's
# own `ja` extras, bundle_yue_pkuseg, bundle_zh_charseg); ar had nothing, and its tokeniser raises
# at load time, so `pip install ar_sud_padt` produced a model that could not be opened.
#
# `camel_data` still has to be run by hand -- a data download is not expressible as a pip
# dependency, and the component says so in its own error. Declaring the LIBRARY at least reduces
# that from two missing pieces to one.
REQUIREMENTS = {
    "ar": ["camel-tools>=1.5.2"],
    # sa reads morphological CANDIDATE SETS off vidyut's kosha at RUNTIME
    # (`sud.AnalyserFeatsEmbed.v1`, runtime = true). The wheel bundles nothing of vidyut's: the
    # package is a dependency and the user fetches its ~77 MB data bundle once with
    #     python -c "import vidyut; vidyut.download_data('vidyut-data')"
    # pointing VIDYUT_DATA at it if it is not ./vidyut-data. That keeps the ~100 M-form table out
    # of a CC BY-SA wheel, which is a licence position as much as a size one. The layer REFUSES to
    # run without it rather than falling back to its silent bit, because a silent fallback loads
    # cleanly and merely parses worse.
    # indic-transliteration is NOT optional: `sa_tokenizer._normalise_body` imports it for the
    # DEVANAGARI input path, which is one of the two documented input scripts. It was never
    # declared, so the released wheel raises ModuleNotFoundError on Devanagari input unless
    # the user happens to have it. Caught by feeding the installed wheel श्रियं दिशतु वः.
    "sa": ["vidyut>=0.4.0", "indic-transliteration>=2.3.0"],
    # ko reads the morphemes an eojeol hides off mecab-ko at RUNTIME (`sud.KoAnalyserEmbed.v1`),
    # which is what lets it reach the stem inside a token it has never seen -- a third of the test
    # set (docs/korean.md). `python-mecab-ko` is the SHIPPABLE backend: it vendors the library and
    # pulls `python-mecab-ko-dic`, so a user needs neither Homebrew nor MECAB_PATH, and no data
    # download is left to do by hand. Both are permissive (BSD 3-Clause and Apache 2.0
    # respectively), so unlike ar's and sa's data this is a plain dependency with no licence
    # position attached -- the wheel still bundles none of it.
    #
    # ⚠ The arms were TRAINED through `natto-py` + Homebrew mecab-ko. That is a different binding on
    # the same dictionary, and `scripts/check_ko_backends.py` measured what the difference is worth
    # over all 31 532 distinct eojeol of the treebank: tag sequences identical on 100.00 %, lexical
    # keys on 99.99 %, and the shipping arm scores TAG/UAS/LAS identically to two decimals through
    # either. Hence the layer's guard compares the DICTIONARY, not the binding.
    "ko": ["python-mecab-ko>=1.3.7"],
}

SOURCES = {
    "ar": [
        {"name": "SUD_Arabic-PADT", "license": "CC BY-NC-SA 3.0",
         "url": "https://github.com/surfacesyntacticud/SUD_Arabic-PADT",
         "note": "Derived from the Prague Arabic Dependency Treebank 1.0, which is the source of "
                 "the NonCommercial term. Its Vform column is also the source of the vocalisation "
                 "table bundled with the ar_vocalise component."},
    ],
    # GUM's annotations are CC BY, so citing the corpus, pointing to its website and crediting the
    # annotators is an OBLIGATION, not a courtesy -- and it is the whole of what GUM asks of this
    # wheel now that the NonCommercial reading is retired (see LICENSE above). GUM's LICENSE
    # separately asks that the sources of the TEXTS be cited as their own sites require. Only the
    # ten non-NonCommercial genres are in the training data; their underlying sources carry three
    # different CC licences, and the Wikimedia ones are ShareAlike.
    "en_gum": [
        {"name": "SUD_English-EWT", "license": "CC BY-SA 4.0",
         "url": "https://github.com/surfacesyntacticud/SUD_English-EWT"},
        {"name": "SUD_English-GUM (academic, bio, conversation, court, interview, news, speech, "
                 "textbook, vlog, voyage; the NonCommercial genres essay/fiction/letter/podcast/"
                 "whow are excluded)",
         "license": "annotations CC BY 4.0; texts CC BY 4.0 / CC BY-SA 3.0 / CC BY 2.5 / public "
                    "domain per source",
         "url": "https://github.com/amir-zeldes/gum",
         "note": "Georgetown University Multilayer Corpus, Amir Zeldes and 300+ student "
                 "annotators, who are listed per document at https://gucorpling.org/gum/ -- "
                 "please cite the corpus and that site. Text sources include Wikinews, "
                 "Wikivoyage, OpenStax, the Santa Barbara Corpus (John Du Bois, UCSB), "
                 "Creative-Commons YouTube and public-domain political speeches."},
    ],
    "lzh": [
        {"name": "SUD_Classical_Chinese-Kyoto", "license": "CC BY-SA 4.0",
         "url": "https://github.com/surfacesyntacticud/SUD_Classical_Chinese-Kyoto"},
        # punctuation only — see NOTICE.md and scripts/align_kanripo_punct.py
        {"name": "Kanseki Repository (punctuation)", "license": "CC BY-SA 4.0",
         "url": "https://github.com/kanripo"},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--lang", required=True, help="the model's own language code (goes into meta)")
    ap.add_argument("--arm", help="registry key for LICENSE/SOURCES/REQUIREMENTS; defaults to "
                                  "--lang. Use it where one language ships more than one wheel "
                                  "(en vs en_gum), so the tables do not collide.")
    ap.add_argument("--license", dest="license_", help="override the licence outright")
    ap.add_argument("--description")
    args = ap.parse_args()
    arm = args.arm or args.lang

    p = pathlib.Path(args.model_dir) / "meta.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["license"] = args.license_ or LICENSE.get(arm, DEFAULT_LICENSE)
    m["author"] = AUTHOR
    m["url"] = URL
    if arm in SOURCES:
        m["sources"] = SOURCES[arm]
    if arm in REQUIREMENTS:
        # union, never replace: bundle_zh_charseg / bundle_yue_pkuseg write theirs the same way,
        # and this may run either side of them.
        m["requirements"] = sorted(set(m.get("requirements") or []) | set(REQUIREMENTS[arm]))
    if args.description:
        m["description"] = args.description
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.model_dir}: license={m['license']} sources={[s['name'] for s in m.get('sources', [])]}"
          f" requirements={m.get('requirements') or []}")


if __name__ == "__main__":
    main()
