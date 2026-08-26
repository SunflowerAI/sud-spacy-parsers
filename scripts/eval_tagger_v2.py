#!/usr/bin/env python3
"""Score the generic tagging arm: UPOS, FEATS and lemma accuracy on a held-out language.

Reports the MAJORITY baseline for UPOS beside the model, because a tagger meeting an unseen script
emits its bias and that alone scores 15-35 % (`probe_tagger_transfer.py`).
"""
import argparse, collections, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spacy
from spacy.tokens import Doc
import generic_tag_code  # noqa: F401


def read_conllu(path, limit=0):
    sents, rows = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            if rows:
                sents.append(rows)
                if limit and len(sents) >= limit:
                    return sents
            rows = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 8 or "-" in f[0] or "." in f[0]:
            continue
        rows.append(f)
    if rows:
        sents.append(rows)
    return sents


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--conllu", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--give-pos", action="store_true",
                    help="supply gold UPOS as an input, as a hand-annotating user would")
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    sents = [s for p in a.conllu for s in read_conllu(p, a.limit)]
    n = pos = morph = lem = 0
    # The trivial baselines. FEATS: always predict nothing -- correct on every token the treebank
    # leaves empty. LEMMA: copy the wordform -- correct wherever the lemma IS the form, which for
    # an isolating language is nearly always. Thai's "100 % lemma accuracy" is this and nothing else.
    morph_base = lem_base = 0
    gold_pos = collections.Counter()
    for rows in sents:
        doc = Doc(nlp.vocab, words=[r[1] for r in rows])
        if a.give_pos:
            for tk, r in zip(doc, rows):
                if r[3] != "_":
                    tk.pos_ = r[3]
        doc._.tb_lang = a.lang
        doc = nlp(doc)
        for t, r in zip(doc, rows):
            if r[3] == "_":
                continue
            n += 1
            gold_pos[r[3]] += 1
            pos += int(t.pos_ == r[3])
            morph += int(str(t.morph) == ("" if r[5] == "_" else r[5]))
            gold_lem = r[1] if r[2] == "_" else r[2]
            lem += int(t.lemma_ == gold_lem)
            morph_base += int(r[5] == "_")
            lem_base += int(r[1] == gold_lem)
    base = max(gold_pos.values()) / max(n, 1)
    print(json.dumps({"lang": a.lang, "tokens": n, "pos_acc": pos / max(n, 1),
                      "morph_acc": morph / max(n, 1), "lemma_acc": lem / max(n, 1),
                      "pos_majority_baseline": base,
                      "morph_empty_baseline": morph_base / max(n, 1),
                      "lemma_identity_baseline": lem_base / max(n, 1)}))


if __name__ == "__main__":
    main()
