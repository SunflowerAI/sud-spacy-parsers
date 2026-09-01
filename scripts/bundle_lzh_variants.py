#!/usr/bin/env python
"""Bundle the 異體字 map into a built lzh model's tokenizer, and verify what changed.

The map is applied at ORTH, inside `sud.CharSegTokenizer.v1`, so 无 reaches every encoder as 無.
NO RETRAIN IS INVOLVED and none is needed: `gold_preproc` + `sud.GoldTokCorpus.v1` make the parser
segmenter-agnostic, and this changes only what the tokeniser hands it. Every model weight must come
out BYTE-IDENTICAL, which `--verify` asserts.

⚠ `--verify` ALSO ASSERTS THAT TEXT WITHOUT A VARIANT IS UNTOUCHED. The map fires on 62.7 % of the
out-of-treebank character mass but must be a strict no-op on treebank orthography, or the released
metrics stop describing the wheel. It reproduces the token stream and the full parse digest of the
source arm on the treebank's own test text before it will write anything.

Usage:
    bundle_lzh_variants.py --src training_lzh_seg_sud_xw/model-best \
        --variants models/lzh_variant_norm.json --out build_sud/work_lzh.var --verify
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys


def load_code(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def digest(nlp, texts):
    h = hashlib.sha256()
    for t in texts:
        d = nlp(t)
        for tok in d:
            h.update(f"{tok.text}\t{tok.pos_}\t{tok.tag_}\t{tok.head.i}\t{tok.dep_}\n"
                     .encode("utf-8"))
    return h.hexdigest()


def sample_texts(conllu, n=200):
    out, cur = [], []
    for line in pathlib.Path(conllu).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                out.append("".join(cur))
                cur = []
            if len(out) >= n:
                break
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conllu", default="assets_lzh/SUD_Classical_Chinese-Kyoto/"
                                        "lzh_kyoto-sud-test.relabeled_ext.udep_ruled.punct."
                                        "rulemerged.conllu")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy

    probe = spacy.load(a.src)
    if not hasattr(probe.tokenizer, "load_variants"):
        sys.exit(f"{a.src}: its tokenizer is {type(probe.tokenizer).__name__}, "
                 f"which takes no variant map")

    if a.verify:
        texts = sample_texts(a.conllu)
        before_tokens = [[t.text for t in probe(x)] for x in texts]
        before_digest = digest(probe, texts)
        probe.tokenizer.load_variants(a.variants)
        after_tokens = [[t.text for t in probe(x)] for x in texts]
        if after_tokens != before_tokens:
            bad = next(i for i, (x, y) in enumerate(zip(before_tokens, after_tokens)) if x != y)
            sys.exit(f"  REFUSING: the map is not a no-op on treebank orthography — sample {bad} "
                     f"changed:\n    {''.join(before_tokens[bad])}\n    {''.join(after_tokens[bad])}")
        if digest(probe, texts) != before_digest:
            sys.exit("  REFUSING: the full-pipeline parse digest changed on treebank text")
        print(f"  verified: {len(texts)} test texts reproduce their token stream and parse digest")
    del probe

    # ⚠ WRITE FROM A FRESHLY LOADED MODEL THAT HAS PROCESSED NOTHING. Running text through a
    # pipeline interns strings, so `vocab/strings.json` grows — and the first version of this
    # script verified and wrote from the SAME object, which made the output differ from its source
    # in a file that has nothing to do with the change. The byte-identity check caught it, which is
    # the whole reason to compare every file rather than a chosen list of weights.
    nlp = spacy.load(a.src)
    n_before = len(nlp.tokenizer.variants)
    nlp.tokenizer.load_variants(a.variants)
    print(f"{a.src}: variant map {n_before} -> {len(nlp.tokenizer.variants)} entries")
    src_dir = pathlib.Path(a.src)
    nlp.to_disk(a.out)
    out = pathlib.Path(a.out)
    if a.verify:
        diff = []
        for f in sorted(src_dir.rglob("*")):
            # tokenizer/* IS the change. The top-level meta.json is excluded because spaCy
            # rewrites it on every `to_disk` for reasons of its own (it adds `"mode": "default"`
            # to the empty vectors block), not because of anything here. Everything else must
            # match — `vocab/strings.json` very much included (see the note above).
            if not f.is_file() or f.name in ("config.cfg", "README.md") \
                    or f.parent.name == "tokenizer" \
                    or (f.name == "meta.json" and f.parent == src_dir):
                continue
            g = out / f.relative_to(src_dir)
            if not g.exists():
                diff.append(f"MISSING {f.relative_to(src_dir)}")
            elif hashlib.sha256(f.read_bytes()).digest() != hashlib.sha256(g.read_bytes()).digest():
                diff.append(f"DIFFERS {f.relative_to(src_dir)}")
        if diff:
            sys.exit(f"  REFUSING: model files changed, this step must touch none: {diff}")
        print("  verified: every file outside tokenizer/ is byte-identical to the source arm")
    meta = json.loads((out / "tokenizer" / "meta.json").read_text(encoding="utf-8"))
    print(f"wrote {a.out}  (tokenizer/meta.json: {meta})")


if __name__ == "__main__":
    main()
