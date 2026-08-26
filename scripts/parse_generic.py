#!/usr/bin/env python3
"""Parse tagged CoNLL-U with the generic arm: heads and deprels in, HEAD/DEPREL columns out.

This is the arm's actual interface. It has no tokenizer worth running and no morphologiser, so it
does not take raw text: it takes a token sequence that already carries UPOS and (where the language
has them) FEATS, and fills in the two columns it predicts. For a low-resource language that means
the upstream can be anything -- a small tagger, a finite-state analyser, a lexicon, or hand
annotation -- and none of it has to be a spaCy pipeline.

    .venv/bin/python scripts/parse_generic.py training_generic_s0/model-best \\
        --lang te --in tagged.conllu --out parsed.conllu

⚠ `--lang` IS NOT OPTIONAL AND IS NOT A MODEL INPUT. It selects which of the thirteen row-sets the
aligned-vector lookup uses; the model has no parameter that varies with it. Passing the wrong one
does not raise -- the tokens simply miss the table and arrive as OOV, which is the shape of a silent
regression rather than an error, so `--report` prints the coverage it actually got and the tool
warns when it is implausibly low.

⚠ A LANGUAGE THE ARM WAS NOT TRAINED ON. `--lang` must name a language the VECTOR TABLE has rows
for, which is a different and larger set than the languages the parser saw in training: a
leave-one-language-out arm has rows for all thirteen and training data for twelve, and parsing the
held-out one is exactly the zero-shot condition. Parsing a language that is in NEITHER set needs its
own aligned asset first -- see `docs/aligned-vectors.md` for what fitting one costs (a rotation on
a gloss bag; no dictionary content is redistributed).

The input's own HEAD/DEPREL columns, if any, are ignored and overwritten. Everything else --
comments, MISC, XPOS, sentence boundaries -- is passed through unchanged, so the output diffs
cleanly against the input.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy                                                                    # noqa: E402
from spacy.tokens import Doc                                                    # noqa: E402

import generic_code                       # noqa: E402,F401  (registers the arm's layer and reader)


def read_conllu(path):
    """[(comments, [fields...])] -- word rows only, but MWT/empty rows are REMEMBERED in place.

    A multiword-token range row carries no HEAD of its own and must survive into the output
    untouched, or the file stops being valid CoNLL-U. They are kept in the row list and skipped when
    building the doc.
    """
    sents, comments, rows = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    sents.append((comments, rows))
                comments, rows = [], []
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            f = line.split("\t")
            while len(f) < 10:
                f.append("_")
            rows.append(f)
    if rows:
        sents.append((comments, rows))
    return sents


def is_word(f):
    return "-" not in f[0] and "." not in f[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--lang", required=True, help="which aligned row-set to look tokens up in")
    ap.add_argument("--in", dest="inp", required=True, help="tagged CoNLL-U (UPOS, and FEATS if any)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64, help="sentences per doc handed to the parser")
    ap.add_argument("--report", action="store_true", help="print vector coverage and label counts")
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    sents = read_conllu(a.inp)
    if not sents:
        sys.exit(f"no sentences in {a.inp}")

    # Coverage is checked against the table the MODEL is holding, so the number reported is the one
    # the parser actually saw -- not a re-derivation that could disagree with it.
    embed_table = None
    for _, proc in nlp.pipeline:
        model = getattr(proc, "model", None)
        if model is None:
            continue
        for node in model.walk():
            if node.name == "extract_aligned_vectors":
                embed_table = node.attrs.get("av_table")
                break
        if embed_table is not None:
            break
    if embed_table is not None and a.lang not in embed_table.langs:
        sys.exit(f"--lang {a.lang!r}: this model's table has no rows for it. Known: "
                 f"{' '.join(embed_table.langs)}")

    total = hit = 0
    labels = {}
    with open(a.out, "w", encoding="utf-8") as out:
        for start in range(0, len(sents), a.batch):
            chunk = sents[start:start + a.batch]
            words, spaces, owners = [], [], []
            for si, (_, rows) in enumerate(chunk):
                wrows = [f for f in rows if is_word(f)]
                for f in wrows:
                    words.append(f[1])
                    misc = f[9].split("|") if f[9] != "_" else []
                    spaces.append("SpaceAfter=No" not in misc)
                    owners.append((si, f))
            doc = Doc(nlp.vocab, words=words, spaces=spaces)
            doc._.tb_lang = a.lang
            for tok, (_, f) in zip(doc, owners):
                if f[3] != "_":
                    tok.pos_ = f[3]
                tok.set_morph(f[5] if f[5] != "_" else None)
                # The identity fallback prep_generic.py applies. sa's vectors are keyed by lemma,
                # so a literal `_` here would be an all-OOV language.
                tok.lemma_ = f[2] if f[2] != "_" else f[1]
                # Sentence boundaries come from the FILE, not from the parser: the caller has
                # already segmented, and letting the parser re-segment would silently move the
                # boundaries their annotation depends on.
                tok.is_sent_start = None
            offset = 0
            for si, (_, rows) in enumerate(chunk):
                n = sum(1 for f in rows if is_word(f))
                if n:
                    doc[offset].is_sent_start = True
                offset += n

            if embed_table is not None:
                for tok in doc:
                    total += 1
                    hit += embed_table.row(a.lang, tok) is not None

            for _, proc in nlp.pipeline:
                doc = proc(doc)

            # Map back. HEAD is 1-based within the SENTENCE, and a root is 0 -- spaCy marks a root
            # by `token.head is token`, which is a different convention and the usual place this
            # conversion goes wrong.
            idx = 0
            for si, (comments, rows) in enumerate(chunk):
                wrows = [f for f in rows if is_word(f)]
                base = idx
                pos_in_sent = {}
                for k, f in enumerate(wrows):
                    pos_in_sent[base + k] = k + 1
                for k, f in enumerate(wrows):
                    tok = doc[base + k]
                    if tok.head.i == tok.i:
                        f[6], f[7] = "0", "root"
                    else:
                        f[6] = str(pos_in_sent.get(tok.head.i, 0))
                        f[7] = tok.dep_ or "dep"
                        if f[6] == "0":
                            # The predicted head fell outside this sentence -- only possible if the
                            # parser crossed a boundary the file declared. Report it as a root
                            # rather than writing an out-of-range HEAD, which would be invalid.
                            f[7] = "root"
                    labels[f[7]] = labels.get(f[7], 0) + 1
                idx += len(wrows)

                for c in comments:
                    out.write(c + "\n")
                for f in rows:
                    out.write("\t".join(f) + "\n")
                out.write("\n")

    print(f"parsed {len(sents)} sentences -> {a.out}")
    if a.report and total:
        pct = 100 * hit / total
        print(f"aligned-vector coverage: {hit}/{total} tokens ({pct:.1f} %)")
        if pct < 40:
            print(f"⚠ {pct:.1f} % is far below what any language in the table scores (74-100 %). "
                  f"The usual cause is the wrong --lang, or text in a script or normalisation the "
                  f"asset was not built for. The parse will still be produced; it will be poor.")
        print("labels: " + "  ".join(f"{k}:{v}" for k, v in
                                     sorted(labels.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
