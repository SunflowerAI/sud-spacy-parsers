#!/usr/bin/env python3
"""Does a SikuBERT contextual encoder carry UPOS information the lzh morphologiser lacks?

The question is NOT "is the encoder informative" — an aggregate probe accuracy is exactly what
NEGATIVE-RESULTS.md says hides the answer. It is "is it informative WHERE THE SHIPPED ARM FAILS",
so every number here is reported on slices of the failure population:

  * forms UNSEEN in the treebank's training split,
  * MULTI-CHARACTER tokens (73.0 % of the treebank's are PROPN, so every segmenter merge lands on
    a PROPN prior),
  * tokens containing a character absent from the treebank altogether.

The baseline is the SHIPPED MORPHOLOGISER's own predictions on the same tokens, not a majority
class — a probe compared against nothing is unreadable.

⚠ SikuBERT is pretrained on 四庫全書 and the Kyoto treebank is drawn from the same tradition, so
the test text is very likely inside its pretraining corpus. That is the kanripo situation again
(NEGATIVE-RESULTS.md: "not label leakage, but the vectors would have been fitted to the very text
they were scored on") and it cannot be fixed by removing data from someone else's pretraining run.
Read every figure here as an UPPER BOUND.

Usage:
    probe_lzh_sikubert.py [--model SIKU-BERT/sikubert] [--train-tokens 120000] [--layer -1]
"""
import argparse
import collections
import sys

import numpy as np

D = "assets_lzh/SUD_Classical_Chinese-Kyoto"
SUFFIX = "relabeled_ext.udep_ruled.punct.rulemerged.conllu"


def blocks(split):
    cur = []
    for line in open(f"{D}/lzh_kyoto-sud-{split}.{SUFFIX}", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                yield cur
                cur = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        cur.append((f[1], f[3]))
    if cur:
        yield cur


def encode(bs, tok, model, device, layer, max_len=510, batch=16):
    """Mean-pool SikuBERT subtoken states over each treebank token. Returns one array per block.

    The treebank's tokens are runs of Han characters, and SikuBERT is a character-level WordPiece
    model for Chinese, so a token's subtokens are contiguous — but the mapping is built from
    `is_split_into_words=True` word_ids rather than assumed, because a token containing a character
    outside the vocabulary still has to land on the right row."""
    import torch
    out = []
    for i in range(0, len(bs), batch):
        chunk = [[w for w, _ in b][:max_len] for b in bs[i:i + batch]]
        enc = tok(chunk, is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        for j, words in enumerate(chunk):
            wid = tok(chunk, is_split_into_words=True, padding=True, truncation=True,
                      max_length=max_len + 2).word_ids(j)
            acc = np.zeros((len(words), hs.shape[-1]), dtype=np.float32)
            cnt = np.zeros(len(words), dtype=np.float32)
            h = hs[j].float().cpu().numpy()
            for pos, w in enumerate(wid):
                if w is None or w >= len(words):
                    continue
                acc[w] += h[pos]
                cnt[w] += 1
            cnt[cnt == 0] = 1.0
            out.append(acc / cnt[:, None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SIKU-BERT/sikubert")
    ap.add_argument("--train-tokens", type=int, default=120000)
    ap.add_argument("--layer", type=int, default=-1)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.linear_model import LogisticRegression

    tr_blocks, te_blocks = list(blocks("train")), list(blocks("test"))
    trforms = collections.Counter(w for b in tr_blocks for w, _ in b)
    trchars = set("".join(trforms))

    # cap the training side by TOKENS, keeping whole blocks (context is the point)
    keep, n = [], 0
    for b in tr_blocks:
        keep.append(b)
        n += len(b)
        if n >= a.train_tokens:
            break
    tr_blocks = keep
    print(f"train {sum(len(b) for b in tr_blocks)} tokens / {len(tr_blocks)} blocks;  "
          f"test {sum(len(b) for b in te_blocks)} tokens / {len(te_blocks)} blocks", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()

    unk = tok.unk_token_id
    allchars = {c for b in tr_blocks + te_blocks for w, _ in b for c in w}
    n_unk = sum(1 for c in allchars if tok.convert_tokens_to_ids(c) == unk)
    print(f"{a.model}: {n_unk} of {len(allchars)} treebank characters are [UNK]", flush=True)

    print("encoding train…", flush=True)
    Xtr = np.vstack(encode(tr_blocks, tok, model, device, a.layer))
    ytr = np.array([p for b in tr_blocks for _, p in b])
    print("encoding test…", flush=True)
    Xte = np.vstack(encode(te_blocks, tok, model, device, a.layer))
    yte = np.array([p for b in te_blocks for _, p in b])
    forms = np.array([w for b in te_blocks for w, _ in b], dtype=object)

    print(f"fitting the probe on {Xtr.shape}…", flush=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    np.save("/tmp/lzh_siku_pred.npy", pred)
    with open("/tmp/lzh_siku_forms.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(forms))
    report(yte, pred, forms, trforms, trchars, label="SikuBERT linear probe")


def report(yte, pred, forms, trforms, trchars, label):
    slices = {
        "ALL": np.ones(len(yte), bool),
        "form seen in train": np.array([f in trforms for f in forms]),
        "form UNSEEN in train": np.array([f not in trforms for f in forms]),
        "multi-character token": np.array([len(f) > 1 for f in forms]),
        "has a char absent from the treebank": np.array([any(c not in trchars for c in f) for f in forms]),
    }
    print(f"\n== {label} ==")
    print(f"{'slice':38s}{'n':>7}{'UPOS acc':>11}{'PROPN P':>10}{'PROPN R':>10}")
    for name, m in slices.items():
        if m.sum() == 0:
            continue
        acc = (yte[m] == pred[m]).mean()
        gp, pp = (yte[m] == "PROPN"), (pred[m] == "PROPN")
        tp = (gp & pp).sum()
        P = tp / pp.sum() if pp.sum() else float("nan")
        R = tp / gp.sum() if gp.sum() else float("nan")
        print(f"{name:38s}{m.sum():>7}{acc*100:>10.2f}%{P*100:>9.2f}%{R*100:>9.2f}%")


if __name__ == "__main__":
    main()
