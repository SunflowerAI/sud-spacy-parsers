#!/usr/bin/env python3
"""Fit ONE language-embedding row for an unseen language on a small annotated sample.

The deployable form of a trainable per-language vector. The four-bit typological profile has a
deployment story -- a linguist states the bits, or annotates ~200 sentences and they are measured --
and this is the same annotation budget spent differently: instead of computing four bits from the
sample, fit a 128-d vector on it, with every other parameter in the model frozen.

    128 free parameters, ~3 400 tokens of supervision.

⚠ **EVERYTHING EXCEPT THE EMBEDDING TABLE IS FROZEN**, and that is enforced by wrapping the
optimizer rather than by hoping. If the encoder or the parser moved as well, this would be ordinary
fine-tuning on the target language and would answer a different question -- one where the comparison
against a 200-sentence profile is no longer like for like.

⚠ **THE SAMPLE MUST NOT BE PART OF THE SCORED TEST SET.** It is drawn from the language's `train`
split, which for a test language the experiment otherwise never touches. Five of the twenty test
languages are test-only treebanks and cannot take part.
"""
import argparse
import copy
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402
from spacy.training import Example  # noqa: E402
from thinc.api import Adam  # noqa: E402

import sud_generic_embed_v2  # noqa: E402,F401  (registers sud.GenericEmbed.v2)


def read_conllu(path):
    """Word rows only; MWT ranges (`3-4`) and empty nodes (`3.1`) dropped.

    Inlined rather than imported from `prep_generic` so this tool is self-contained: it ships
    INSIDE the model wheel, where the rest of the repo is not present.
    """
    sents, rows = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if rows:
                    sents.append(rows)
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


class OnlyTheseModels:
    """An optimizer that updates the given thinc node ids and silently drops every other gradient."""

    def __init__(self, inner, allow_ids):
        self.inner = inner
        self.allow = set(allow_ids)

    def __call__(self, key, weights, gradient, **kw):
        if key[0] not in self.allow:
            return weights, gradient * 0.0
        return self.inner(key, weights, gradient, **kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def find_nodes(nlp, name):
    out = []
    for _, proc in nlp.pipeline:
        mdl = getattr(proc, "model", None)
        if mdl is None:
            continue
        out += [n for n in mdl.walk() if n.name == name]
    return out


def docs_from_conllu(paths, n, seed, vocab, lang, no_feats=False):
    sents = [s for p in paths for s in read_conllu(p)]
    rng = random.Random(seed)
    if n and n < len(sents):
        sents = [sents[i] for i in sorted(rng.sample(range(len(sents)), n))]
    out = []
    for rows in sents:
        idx = {r[0]: i for i, r in enumerate(rows)}
        words = [r[1] for r in rows]
        heads, deps = [], []
        ok = True
        for i, r in enumerate(rows):
            h = r[6]
            if h == "0":
                heads.append(i)
                deps.append("root")
            elif h in idx:
                heads.append(idx[h])
                deps.append(r[7].split("@")[0].split("$")[0].split("/")[0])
            else:
                ok = False
                break
        if not ok or not words:
            continue
        ref = Doc(vocab, words=words, heads=heads, deps=deps,
                  sent_starts=[True] + [False] * (len(words) - 1))
        for tok, r in zip(ref, rows):
            if r[3] != "_":
                tok.pos_ = r[3]
            if r[5] != "_" and not no_feats:
                tok.set_morph(r[5])
        ref._.tb_lang = lang
        pred = Doc(vocab, words=words)
        for p, rr in zip(pred, rows):
            if rr[3] != "_":
                p.pos_ = rr[3]
            if rr[5] != "_" and not no_feats:
                p.set_morph(rr[5])
        pred._.tb_lang = lang
        out.append(Example(pred, ref))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--conllu", nargs="+", required=True, help="UNSCORED data for this language")
    ap.add_argument("--n-sents", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-feats", action="store_true",
                    help="fit without any morphology, i.e. the annotator supplies UPOS and syntax "
                         "only. Must be paired with --no-feats at evaluation.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    nlp = spacy.load(a.model)
    slot_nodes = find_nodes(nlp, "extract_lang_slot")
    embeds = [n for n in find_nodes(nlp, "embed") if n.has_param("E")]
    if not slot_nodes:
        sys.exit("this model has no lang_embed channel")
    if not embeds:
        sys.exit("could not find the embedding table")

    slots = dict(slot_nodes[0].attrs["ls_slots"])
    n_rows = int(embeds[0].get_param("E").shape[0])
    if a.lang in slots:
        print(f"note: {a.lang} already has slot {slots[a.lang]}; refitting it")
        row = slots[a.lang]
    else:
        used = set(slots.values())
        free = [i for i in range(n_rows) if i not in used]
        if not free:
            sys.exit(f"no spare embedding rows ({n_rows} used); rebuild with a larger --spare")
        row = free[0]
        print(f"assigning {a.lang} spare row {row} of {n_rows}")
    for node in slot_nodes:
        d = dict(node.attrs["ls_slots"])
        d[a.lang] = row
        node.attrs["ls_slots"] = d

    examples = docs_from_conllu(a.conllu, a.n_sents, a.seed, nlp.vocab, a.lang, a.no_feats)
    print(f"{len(examples)} training examples, "
          f"{sum(len(e.reference) for e in examples):,} tokens")

    allow = {n.id for n in embeds}
    before = copy.deepcopy(embeds[0].get_param("E"))
    frozen_ref = {n.id: {p: copy.deepcopy(n.get_param(p))
                         for p in n.param_names if n.has_param(p)}
                  for _, proc in nlp.pipeline if getattr(proc, "model", None) is not None
                  for n in proc.model.walk() if n.id not in allow}

    sgd = OnlyTheseModels(Adam(a.lr), allow)
    rng = random.Random(a.seed)
    for ep in range(a.epochs):
        rng.shuffle(examples)
        losses = {}
        for i in range(0, len(examples), 16):
            nlp.update(examples[i:i + 16], sgd=sgd, losses=losses)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  loss {sum(losses.values()):12.2f}")

    after = embeds[0].get_param("E")
    moved = float(abs(after[row] - before[row]).max())
    others = float(abs(after - before).max()) if n_rows > 1 else 0.0
    print(f"target row moved by {moved:.4f}")
    # The freeze is verified, not assumed: if any other parameter drifted, this is fine-tuning.
    drift = 0.0
    for _, proc in nlp.pipeline:
        if getattr(proc, "model", None) is None:
            continue
        for n in proc.model.walk():
            if n.id in allow or n.id not in frozen_ref:
                continue
            for pname, ref in frozen_ref[n.id].items():
                if n.has_param(pname):
                    drift = max(drift, float(abs(n.get_param(pname) - ref).max()))
    print(f"max drift in any FROZEN parameter: {drift:.3e}")
    if drift > 1e-6:
        sys.exit("FROZEN PARAMETERS MOVED -- this is fine-tuning, not embedding adaptation")
    if moved < 1e-6:
        sys.exit("the target row did not move; the optimizer filter is too aggressive")
    print(f"(rows other than the target moved at most {others:.4f})")

    nlp.to_disk(a.out)
    json.dump({"lang": a.lang, "row": row, "n_sents": len(examples),
               "tokens": sum(len(e.reference) for e in examples),
               "row_delta": moved, "frozen_drift": drift},
              open(pathlib.Path(a.out) / "adapt.json", "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
