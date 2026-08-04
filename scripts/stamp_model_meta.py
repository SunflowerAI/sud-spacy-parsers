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

Usage:
    stamp_model_meta.py MODEL_DIR --lang lzh [--description TEXT]
"""
import argparse
import json
import pathlib

AUTHOR = "Sunflower AI"
URL = "https://github.com/SunflowerAI/sud-spacy-parsers"
# la is NonCommercial (ITTB + PROIEL + Perseus are all CC BY-NC-SA); everything else stays
# commercially usable, which is why SUD_English-GUM is excluded from en.
LICENSE = {"la": "CC BY-NC-SA 4.0"}
DEFAULT_LICENSE = "CC BY-SA 4.0"

SOURCES = {
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
    ap.add_argument("--lang", required=True)
    ap.add_argument("--description")
    args = ap.parse_args()

    p = pathlib.Path(args.model_dir) / "meta.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["license"] = LICENSE.get(args.lang, DEFAULT_LICENSE)
    m["author"] = AUTHOR
    m["url"] = URL
    if args.lang in SOURCES:
        m["sources"] = SOURCES[args.lang]
    if args.description:
        m["description"] = args.description
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.model_dir}: license={m['license']} sources={[s['name'] for s in m.get('sources', [])]}")


if __name__ == "__main__":
    main()
