#!/usr/bin/env python3
"""Distil SikuBERT into a static, PCA'd `.vec` table for lzh — a channel the encoder can read.

WHAT IT IS. `SIKU-BERT/sikubert` (BERT-base, Apache-2.0, pretrained on 四庫全書) is run over
leak-free kanripo text; every character's CONTEXTUAL last-hidden states are averaged over all its
occurrences, giving one type-level vector per character; PCA reduces that to `--dim`; a row is then
emitted for every character AND for every treebank type (a multi-character type gets the mean of
its characters' rows). The result is an ordinary spaCy static-vector table — no transformer, no
torch and no inference cost in the wheel.

WHY TYPE-LEVEL AVERAGING RATHER THAN THE INPUT EMBEDDING TABLE. The input table already separates
the variant characters cleanly (无 nearest 無 at cos 0.64, 隂 nearest 陰 at 0.64), but it is the
model's pre-contextual lexicon; averaging the states the model actually computes folds in how the
character behaves in Classical Chinese context. `--source embed` builds from the input table
instead, for comparison.

⚠ THE CORPUS MUST BE THE LEAK-FREE ONE. kanripo IS the source of the Kyoto treebank, so the stock
corpus contains every test sentence verbatim. `make_leakfree_lzh_corpus.py` removes dev/test.
(SikuBERT's own pretraining data cannot be cleaned that way and very likely contains the same texts;
that is a ceiling on every figure this table produces, and it is stated in docs/chinese-family.md.)

⚠ PRUNE BY DIMENSION, NEVER BY VOCABULARY. `--extra-types` emits a row for every treebank type even
when the leak-free corpus never shows it, exactly as `build_lzh_vectors.py` does — a table pruned to
a training vocabulary covers 0 % of the forms it exists to rescue. Characters absent from the
sampled text fall back to SikuBERT's input-embedding row rather than being dropped.

⚠ `--shuffle` WRITES THE CONTROL, and it is the only comparison worth reading.
`include_static_vectors` adds a `StaticVectors` projection and widens the Maxout, so arm-vs-baseline
confounds the information with the parameters. The shuffled table has the SAME rows with the
type-to-row correspondence destroyed: identical shapes, identical norms, identical parameter count,
zero information. Anything the shuffle also achieves was never the vectors.

Usage:
    build_lzh_sikubert_vectors.py --out vectors_lzh_siku96.vec --dim 96
    build_lzh_sikubert_vectors.py --out vectors_lzh_siku96_shuf.vec --dim 96 --shuffle
"""
import argparse
import collections
import pathlib
import sys

import numpy as np

CORPUS = "corpus_lzh_kanripo_leakfree.txt"
TRAIN = ("assets_lzh/SUD_Classical_Chinese-Kyoto/"
         "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")


def treebank_types(paths):
    t = collections.Counter()
    for p in paths:
        if not pathlib.Path(p).exists():
            continue
        for line in pathlib.Path(p).open(encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            t[f[1]] += 1
    return t


def read_vec(path):
    rows, dim = {}, None
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        _, d = fh.readline().split()
        dim = int(d)
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                continue
            rows[parts[0]] = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
    return rows, dim


def write_vec(path, rows, dim):
    with pathlib.Path(path).open("w", encoding="utf-8") as fh:
        fh.write(f"{len(rows)} {dim}\n")
        for k in sorted(rows):
            fh.write(k + " " + " ".join(f"{x:.5f}" for x in rows[k]) + "\n")


def shuffle_rows(rows, seed):
    """THE CONTROL: the same rows with the type-to-row correspondence destroyed. Keys stay sorted,
    so the two tables are identical in shape, order and norm distribution — only the information
    is gone."""
    rng = np.random.default_rng(seed)
    keys = sorted(rows)
    vals = [rows[k] for k in keys]
    perm = rng.permutation(len(vals))
    return {k: vals[perm[i]] for i, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SIKU-BERT/sikubert")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--chars", type=int, default=4_000_000,
                    help="characters of corpus text to encode")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--source", choices=("context", "embed"), default="context")
    ap.add_argument("--shuffle", action="store_true", help="write the matched CONTROL table")
    # The control is a PERMUTATION of the arm's own table, so it must be built FROM that table and
    # not by re-encoding: re-encoding would re-run the whole corpus for rows that are then thrown
    # into a different order, and (worse) any difference between the two runs would land in the
    # control rather than being controlled for.
    ap.add_argument("--from-vec", default=None,
                    help="permute this existing .vec instead of encoding anything (with --shuffle)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.from_vec:
        if not a.shuffle:
            raise SystemExit("--from-vec is only meaningful with --shuffle")
        rows, dim = read_vec(a.from_vec)
        write_vec(a.out, shuffle_rows(rows, a.seed), dim)
        print(f"wrote {a.out}: {len(rows)} keys x {dim}, permuted from {a.from_vec}")
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    tk = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    E = model.get_input_embeddings().weight.detach().cpu().numpy()
    unk = tk.unk_token_id

    types = treebank_types([a.train])
    print(f"treebank: {len(types)} types", flush=True)

    acc, cnt = {}, collections.Counter()
    if a.source == "context":
        # chunk the corpus into fixed-length character sequences
        seqs, buf, seen = [], "", 0
        with pathlib.Path(a.corpus).open(encoding="utf-8") as fh:
            for line in fh:
                buf += line.replace(" ", "").strip()
                while len(buf) >= a.seq:
                    seqs.append(buf[:a.seq])
                    buf = buf[a.seq:]
                    seen += a.seq
                if seen >= a.chars:
                    break
        print(f"encoding {len(seqs)} sequences x {a.seq} chars = {seen} characters", flush=True)
        for i in range(0, len(seqs), a.batch):
            chunk = [list(s) for s in seqs[i:i + a.batch]]
            enc = tk(chunk, is_split_into_words=True, return_tensors="pt",
                     padding=True, truncation=True, max_length=a.seq + 2)
            wids = [enc.word_ids(j) for j in range(len(chunk))]
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                h = model(**enc).last_hidden_state.float().cpu().numpy()
            for j, words in enumerate(chunk):
                for pos, w in enumerate(wids[j]):
                    if w is None or w >= len(words):
                        continue
                    c = words[w]
                    if c not in acc:
                        acc[c] = np.zeros(h.shape[-1], dtype=np.float64)
                    acc[c] += h[j, pos]
                    cnt[c] += 1
            if (i // a.batch) % 25 == 0:
                print(f"  {i:6d}/{len(seqs)} sequences, {len(acc)} character types", flush=True)
    print(f"characters with a contextual average: {len(acc)}", flush=True)

    # every character the model will ever be handed: corpus + treebank (prune by DIMENSION only)
    chars = set(acc) | {c for t in types for c in t}
    base = {}
    fallback = 0
    for c in sorted(chars):
        if c in acc and cnt[c] > 0:
            base[c] = (acc[c] / cnt[c]).astype(np.float32)
        else:
            i = tk.convert_tokens_to_ids(c)
            if i == unk:
                continue                      # no SikuBERT row at all: emit nothing, not a zero row
            base[c] = E[i].astype(np.float32)
            fallback += 1
    print(f"character rows: {len(base)}  ({fallback} from the input embedding table, "
          f"{len(chars)-len(base)} dropped as [UNK])", flush=True)

    # PCA, fitted on the character rows only, then applied to everything
    M = np.stack([base[c] for c in sorted(base)])
    mu = M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
    dim = min(a.dim, Vt.shape[0])
    var = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    P = Vt[:dim]
    print(f"PCA {M.shape[1]} -> {dim}: {var:.1%} of variance retained", flush=True)
    red = {c: ((base[c] - mu[0]) @ P.T).astype(np.float32) for c in base}

    # a row for every treebank type as well; a multi-character type is the mean of its characters
    rows = dict(red)
    for t in types:
        if t in rows:
            continue
        parts = [red[c] for c in t if c in red]
        if parts:
            rows[t] = np.mean(parts, axis=0).astype(np.float32)

    if a.shuffle:
        rows = shuffle_rows(rows, a.seed)
        print("wrote the SHUFFLED control table", flush=True)
    write_vec(a.out, rows, dim)
    print(f"wrote {a.out}: {len(rows)} keys x {dim}", flush=True)


if __name__ == "__main__":
    main()
