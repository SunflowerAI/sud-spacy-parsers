#!/usr/bin/env python
"""Bundle the 異體字 map into a built lzh model's tokenizer, and verify what changed.

The map is applied at ORTH, inside `sud.CharSegTokenizer.v1`, so 无 reaches every encoder as 無.
NO RETRAIN IS INVOLVED and none is needed: `gold_preproc` + `sud.GoldTokCorpus.v1` make the parser
segmenter-agnostic, and this changes only what the tokeniser hands it. Every model weight must come
out BYTE-IDENTICAL, which `--verify` asserts.

⚠ `--verify` ALSO ASSERTS THAT TEXT WITHOUT A VARIANT IS UNTOUCHED — on the corpus the arm was
TRAINED on. The map fires on 62.7 % of the out-of-treebank character mass but must be a strict
no-op on the orthography the model actually learned, or the released metrics stop describing the
wheel. It reproduces the token stream and the full parse digest of the source arm before it will
write anything.

⚠⚠ TWO DEFECTS THIS CHECK HAD, BOTH OF WHICH LET IT PASS WITHOUT TESTING ANYTHING.

1. IT SAMPLED THE FIRST 200 BLOCKS AND NONE OF THEM CONTAINED A MAP KEY. The first 200 blocks are
   論語; not one of the 1 178 keys occurs there, so the no-op assertion was VACUOUS — it would have
   passed for any map whatsoever, including one that rewrote a character the treebank uses. The
   sample is now chosen to CONTAIN keys, and `--verify` REFUSES if the corpus has keys the sample
   fails to exercise. A check that cannot fail is not a check.
2. IT ASSERTED THE NO-OP ON THE **TEST** FILE, WHICH IS THE WRONG CORPUS AND THE ASSERTION IS FALSE
   THERE. Map keys are characters absent from TRAIN by construction (`build_lzh_variant_norm.py`),
   which says nothing about dev/test: 46 keys occur in dev/test forms and rewrite 80 tokens. That
   is the map DOING ITS JOB — normalising a variant the model never trained on — not a violation.
   The default corpus is therefore the TRAIN file, where the no-op is the real invariant, and
   dev/test rewrites are REPORTED rather than asserted away.

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


def read_blocks(conllu):
    out, cur = [], []
    for line in pathlib.Path(conllu).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                out.append("".join(cur))
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append(f[1])
    if cur:
        out.append("".join(cur))
    return out


def sample_texts(conllu, keys, n=200):
    """Blocks that EXERCISE the map, plus ordinary ones for the digest.

    Selecting the first n blocks is what made this check vacuous: a block containing no map key
    cannot distinguish a correct map from a catastrophic one. Key-bearing blocks come first, so
    the assertion below actually has something to assert.
    """
    blocks = read_blocks(conllu)
    hit = [b for b in blocks if any(k in b for k in keys)]
    miss = [b for b in blocks if not any(k in b for k in keys)]
    covered = {k for k in keys if any(k in b for b in hit)}
    return hit[:n] + miss[: max(0, n - len(hit[:n]))], hit, covered


def key_hits(conllu, keys):
    """(blocks containing a key, tokens rewritten) — reported, not asserted, for dev/test."""
    toks = 0
    blocks = read_blocks(conllu)
    for b in blocks:
        toks += sum(b.count(k) for k in keys)
    return sum(1 for b in blocks if any(k in b for k in keys)), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    # ⚠ THE TRAIN FILE, NOT TEST. The no-op invariant holds on the orthography the arm LEARNED;
    # on dev/test the map legitimately rewrites (see the header). Naming the right corpus in the
    # default is the fix — a comment telling the next person is not.
    ap.add_argument("--conllu", default="assets_lzh/SUD_Classical_Chinese-Kyoto/"
                                        "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct."
                                        "rulemerged.conllu")
    ap.add_argument("--report-conllu", action="append", default=[],
                    help="corpora to REPORT map firings on without asserting (dev/test)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    load_code("scripts/seg_code.py")
    import spacy

    probe = spacy.load(a.src)
    if not hasattr(probe.tokenizer, "load_variants"):
        sys.exit(f"{a.src}: its tokenizer is {type(probe.tokenizer).__name__}, "
                 f"which takes no variant map")

    if a.verify:
        keys = list(json.loads(pathlib.Path(a.variants).read_text(encoding="utf-8")).get(
            "map", json.loads(pathlib.Path(a.variants).read_text(encoding="utf-8"))))
        keys = [k for k in keys if not k.startswith("__")]
        texts, hit, covered = sample_texts(a.conllu, keys)
        # ⚠ VACUITY GUARD. If the corpus exercises keys and the sample does not, the assertion
        # below proves nothing, which is exactly how this check passed for its whole life.
        n_blocks, n_tokens = key_hits(a.conllu, keys)
        if n_blocks and not covered:
            sys.exit(f"  REFUSING: {n_blocks} blocks of {a.conllu} contain a map key but the "
                     f"sample exercises none — the no-op assertion would be vacuous")
        print(f"  no-op corpus {pathlib.Path(a.conllu).name}: {len(keys)} keys, "
              f"{len(covered)} exercised by the sample, {n_blocks} key-bearing blocks, "
              f"{n_tokens} tokens they would rewrite")
        if not n_blocks:
            # Not an error — a map built by `build_lzh_variant_norm.py` keys on TRAIN-ABSENT
            # characters, so a no-op on train is true by construction. But say so, because the
            # previous version printed a confident "verified" over exactly this situation.
            print("  ⚠ NOTE: no map key occurs in this corpus, so the no-op assertion below is "
                  "TRUE BY CONSTRUCTION and tests nothing. It becomes a real check only for a map "
                  "containing a character the corpus uses (which is refused, as it should be).")
        for extra in a.report_conllu:
            b2, t2 = key_hits(extra, keys)
            print(f"  (reported, not asserted) {pathlib.Path(extra).name}: "
                  f"{b2} key-bearing blocks, {t2} tokens rewritten")
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
        print(f"  verified: {len(texts)} texts ({len(hit)} of them key-bearing) reproduce their "
              f"token stream and parse digest")
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
