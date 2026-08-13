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

A language can have more than one ARM with different licences, so the registry key is `--arm`, not
`--lang`: `en` ships two wheels, EWT-only under CC BY-SA and EWT+GUM under CC BY-NC-SA, and keying
the tables on the language code alone would flip both. `--arm` defaults to `--lang`, so every
single-arm language is unaffected. `--license` overrides outright, for a one-off.

Usage:
    stamp_model_meta.py MODEL_DIR --lang lzh [--description TEXT]
    stamp_model_meta.py MODEL_DIR --lang en --arm en_gum      # the NonCommercial English wheel
"""
import argparse
import json
import pathlib

AUTHOR = "Sunflower AI"
URL = "https://github.com/SunflowerAI/sud-spacy-parsers"
# Keyed by ARM, not language. la is NonCommercial (ITTB + PROIEL + Perseus are all CC BY-NC-SA).
# en_gum is the second English arm, EWT + the non-NonCommercial GUM genres: the five NC genres
# (essay/fiction/letter/podcast/whow) are filtered out, but GUM's LICENSE.md still opens "The
# treebank is licensed under CC BY-NC-SA 4.0" -- which, read strictly, offers the ANNOTATIONS under
# NC whatever the document, and annotations are what a trained model absorbs. So this wheel ships
# NC regardless of the filter, and plain `en` (EWT-only) stays commercially usable.
# ar joined this list on 2026-08-14, correcting a mis-declaration rather than changing anything
# about the model: SUD_Arabic-PADT has always been CC BY-NC-SA 3.0 (its LICENSE.txt: "distributed
# under the same license terms as PADT 1.0"), and the wheel had been falling through to the
# CC BY-SA 4.0 default since v0.1.0. The en_gum reasoning above applies unchanged -- annotations
# are what a trained model absorbs -- and a survey of every assets_*/ found ar to be the only arm
# where the declaration and the training data disagreed. Declared at 4.0 like la, whose sources are
# likewise BY-NC-SA 3.0: ShareAlike permits licensing an adaptation under the later version.
LICENSE = {"la": "CC BY-NC-SA 4.0", "en_gum": "CC BY-NC-SA 4.0", "ar": "CC BY-NC-SA 4.0"}
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
}

SOURCES = {
    "ar": [
        {"name": "SUD_Arabic-PADT", "license": "CC BY-NC-SA 3.0",
         "url": "https://github.com/surfacesyntacticud/SUD_Arabic-PADT",
         "note": "Derived from the Prague Arabic Dependency Treebank 1.0, which is the source of "
                 "the NonCommercial term. Its Vform column is also the source of the vocalisation "
                 "table bundled with the ar_vocalise component."},
    ],
    # GUM's LICENSE asks that the sources of the texts be cited and the annotators credited, so for
    # en_gum the attribution is an obligation, not a courtesy. Only the ten non-NonCommercial genres
    # are in the training data; their underlying sources carry three different CC licences, and two
    # of them (Wikipedia bios, Wikivoyage travel guides) are ShareAlike.
    "en_gum": [
        {"name": "SUD_English-EWT", "license": "CC BY-SA 4.0",
         "url": "https://github.com/surfacesyntacticud/SUD_English-EWT"},
        {"name": "SUD_English-GUM (academic, bio, conversation, court, interview, news, speech, "
                 "textbook, vlog, voyage; the NonCommercial genres essay/fiction/letter/podcast/"
                 "whow are excluded)",
         "license": "CC BY-NC-SA 4.0 (annotations CC BY 4.0; texts CC BY 4.0 / CC BY-SA 3.0 / "
                    "CC BY 2.5 per source)",
         "url": "https://github.com/amir-zeldes/gum",
         "note": "Georgetown University Multilayer Corpus, Amir Zeldes and 300+ student "
                 "annotators; text sources include Wikinews, Wikipedia, Wikivoyage, OpenStax, "
                 "the Santa Barbara Corpus and Creative-Commons YouTube."},
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
