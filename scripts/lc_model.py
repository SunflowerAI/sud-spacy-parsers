#!/usr/bin/env python
"""The generative incremental left-corner model, and its trainer.

THE FACTORISATION. A derivation is w_1, r_1, w_2, r_2, ..., r_{n-1}, w_n, where each w step chooses
a shift type (SHIFT or INSERT) and then a word, and each r step chooses a reduce name and then the
label of the arc it commits to. The model scores all four decisions, so it defines a proper joint
P(words, structure) and its word steps normalise over the whole vocabulary. That is the only reason
surprisal can be read off it at all.

WHAT THE STATE SEES. An LSTM over the EMITTED HISTORY, plus the top two right spines read fresh at
every step. Reduce steps enter the history in an id space disjoint from the words, so the recurrence
cannot confuse an action with a word. Nothing here reads the future, so the same featuriser runs
unchanged under beam decode -- see `lc_features.py`.

TIED OUTPUT EMBEDDINGS. The word softmax reuses the input word table. At this corpus size the
softmax would otherwise be most of the parameters, and the point of tying is not the parameter count
but that self-training later adds vocabulary to ONE table rather than two.

⚠ THE PERPLEXITIES PRINTED HERE ARE GOLD-PATH NUMBERS, not surprisal. They score the single oracle
derivation, so they are an UPPER bound on the model's own surprisal, which marginalises over the
beam. Use them to compare training runs, never as the psycholinguistic quantity; that comes out of
`lc_surprisal.py` and will be lower.
"""
import argparse
import math
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lc_features import Vocab, UNK, N_RESERVED

MAX_LAM, MAX_DEPTH = 3, 7
N_SHIFT, N_REDUCE = 2, 4


class CharDecoder(nn.Module):
    """P(spelling | parser state) for words outside the word vocabulary.

    This is what removes the `<unk>` CLASS. Before it, every out-of-vocabulary word shared one
    lumped probability, so the words that ought to carry the HIGHEST surprisal all carried the same
    surprisal -- which is fatal for a reading-time model specifically, whatever it does to
    perplexity. Now P(w) = P(<unk> | state) * P(spelling | state), and each rare word gets its own.

    ⚠ THE RESULT IS VERY SLIGHTLY IMPROPER. The character model can also spell words that ARE in
    the word vocabulary, so a little mass is double-counted and the total over all strings sums to
    marginally more than one. This is the standard hybrid open-vocabulary formulation and the leak
    is small, but it is a leak: do not describe the model as exactly normalised over English."""

    def __init__(self, n_chars, state_dim, emb=32, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb, padding_idx=0)
        self.bow = nn.Parameter(torch.zeros(emb))
        self.init_h = nn.Linear(state_dim, hidden)
        self.init_c = nn.Linear(state_dim, hidden)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_chars)

    def logprob(self, z, chars):
        """z: (N, state); chars: (N, T) zero-padded and <eow>-terminated. Returns (N,) log P."""
        n, t = chars.shape
        inp = torch.cat([self.bow.expand(n, 1, -1), self.emb(chars[:, :-1])], 1)
        h0 = torch.tanh(self.init_h(z)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(z)).unsqueeze(0)
        out, _ = self.lstm(inp, (h0, c0))
        logits = self.out(out)
        tgt = chars.masked_fill(chars == 0, -100)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                              ignore_index=-100, reduction="none").view(n, t)
        return -nll.sum(1)


class LCModel(nn.Module):
    def __init__(self, n_words, n_labels, n_chars=0, w_dim=128, hidden=256, state=256,
                 dropout=0.3):
        super().__init__()
        self.n_words, self.n_labels, self.n_chars = n_words, n_labels, n_chars
        self.word = nn.Embedding(n_words, w_dim, padding_idx=0)
        self.hist = nn.Embedding(n_words + N_REDUCE * n_labels, w_dim, padding_idx=0)
        self.flag = nn.Embedding(2, 8)
        self.lam = nn.Embedding(MAX_LAM + 1, 8)
        self.label = nn.Embedding(n_labels, 32)
        self.depth = nn.Embedding(MAX_DEPTH + 1, 16)
        self.lstm = nn.LSTM(w_dim, hidden, batch_first=True)
        feat = hidden + 4 * w_dim + 2 * (8 + 8 + 32) + 16
        self.mlp = nn.Sequential(nn.Linear(feat, state), nn.ReLU(), nn.Dropout(dropout))
        self.to_shift = nn.Linear(state, N_SHIFT)
        self.to_name = nn.Linear(state, N_REDUCE)
        self.to_label = nn.Linear(state, N_REDUCE * n_labels)
        self.to_word = nn.Linear(state, self.word.embedding_dim)
        self.word_bias = nn.Parameter(torch.zeros(n_words))
        self.chars = CharDecoder(n_chars, state) if n_chars else None
        self.drop = nn.Dropout(dropout)
        # Spelling of every word in the vocabulary, so the character model can be trained on
        # in-vocabulary words as an auxiliary task without storing a single extra byte in the
        # feature files -- the word ids already in the batch index straight into this.
        # ⚠ Registered ONLY when there is a character model, so a closed-vocabulary checkpoint
        # saved before this buffer existed still loads. A buffer added unconditionally makes every
        # older checkpoint fail on a missing key, which is a needless break of a comparison arm.
        if n_chars:
            self.register_buffer("char_table", torch.zeros(n_words, 1, dtype=torch.long))

    def set_char_table(self, vocab):
        rows = [vocab.cids(w) if not w.startswith("<") else [1] for w in vocab.words]
        width = max(len(r) for r in rows)
        t = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            t[i, :len(r)] = torch.tensor(r)
        self.char_table = t

    def encode_stack(self, stack, depth):
        """stack: (B, T, 10) -- five features for each of the top two spines."""
        s = stack
        parts = [self.word(s[..., 0]), self.word(s[..., 1]),
                 self.flag(s[..., 2]), self.lam(s[..., 3]), self.label(s[..., 4]),
                 self.word(s[..., 5]), self.word(s[..., 6]),
                 self.flag(s[..., 7]), self.lam(s[..., 8]), self.label(s[..., 9]),
                 self.depth(depth)]
        return torch.cat(parts, -1)

    def state_from(self, h, stack, depth):
        return self.mlp(torch.cat([self.drop(h), self.encode_stack(stack, depth)], -1))

    def forward(self, prev, stack, depth, hx=None):
        h, hx = self.lstm(self.drop(self.hist(prev)), hx)
        return self.state_from(h, stack, depth), hx

    def word_logits(self, z):
        return self.to_word(z) @ self.word.weight.t() + self.word_bias


def batches(data, size, shuffle=True, seed=0):
    idx = np.arange(len(data))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    order = sorted(idx, key=lambda i: len(data[i]["prev"]))     # length-bucketed, less padding
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    if shuffle:
        np.random.default_rng(seed + 1).shuffle(chunks)
    for ch in chunks:
        T = max(len(data[i]["prev"]) for i in ch)
        out = {}
        for key, dtype, pad in (("prev", torch.long, 0), ("stack", torch.long, 0),
                                ("depth", torch.long, 0), ("legal_ins", torch.bool, 0),
                                ("legal_comp", torch.bool, 0), ("is_shift", torch.bool, 0),
                                ("tgt_shift", torch.long, -100), ("tgt_word", torch.long, -100),
                                ("tgt_name", torch.long, -100), ("tgt_label", torch.long, -100)):
            rows = []
            for i in ch:
                a = data[i][key]
                w = ((0, T - len(a)), (0, 0)) if a.ndim == 2 else (0, T - len(a))
                rows.append(np.pad(a, w, constant_values=pad))
            out[key] = torch.as_tensor(np.stack(rows), dtype=dtype)
        out["mask"] = torch.as_tensor(
            np.stack([np.pad(np.ones(len(data[i]["prev"]), bool), (0, T - len(data[i]["prev"])))
                      for i in ch]))
        rows, steps, blocks = [], [], []
        for r, i in enumerate(ch):
            pos = data[i].get("oov_pos")
            if pos is None or not len(pos):
                continue
            rows += [r] * len(pos)
            steps += list(pos)
            blocks.append(data[i]["oov_chars"])
        if blocks:
            width = max(b.shape[1] for b in blocks)
            chars = np.concatenate([np.pad(b, ((0, 0), (0, width - b.shape[1]))) for b in blocks])
            out["oov_rows"] = torch.as_tensor(np.array(rows), dtype=torch.long)
            out["oov_steps"] = torch.as_tensor(np.array(steps), dtype=torch.long)
            out["oov_chars"] = torch.as_tensor(chars, dtype=torch.long)
        yield out


def losses(model, b, char_sample=0.0, gen=None):
    z, _ = model(b["prev"], b["stack"], b["depth"])
    m = b["mask"]
    # Shift steps: type (INSERT only where the top spine is incomplete), then the word.
    sl = model.to_shift(z).masked_fill(~b["legal_ins"].unsqueeze(-1) &
                                       torch.tensor([False, True]), -1e9)
    l_shift = F.cross_entropy(sl[m], b["tgt_shift"][m], ignore_index=-100, reduction="sum")
    l_word = F.cross_entropy(model.word_logits(z)[m], b["tgt_word"][m],
                             ignore_index=-100, reduction="sum")
    # Reduce steps: name (COMP only with an incomplete second spine), then the label given the name.
    nl = model.to_name(z).masked_fill(~b["legal_comp"].unsqueeze(-1) &
                                      torch.tensor([False, False, True, True]), -1e9)
    l_name = F.cross_entropy(nl[m], b["tgt_name"][m], ignore_index=-100, reduction="sum")
    ll = model.to_label(z)[m].view(-1, N_REDUCE, model.n_labels)
    pick = b["tgt_name"][m].clamp(min=0)
    ll = ll[torch.arange(len(pick)), pick]
    l_label = F.cross_entropy(ll, b["tgt_label"][m], ignore_index=-100, reduction="sum")
    # The word softmax has already charged -log P(<unk>) at these steps; the character model
    # charges the spelling given the same state. Their sum is -log P(the actual word).
    l_char = torch.zeros((), dtype=l_word.dtype)
    if model.chars is not None and "oov_rows" in b:
        z_oov = z[b["oov_rows"], b["oov_steps"]]
        l_char = -model.chars.logprob(z_oov, b["oov_chars"]).sum()

    # AUXILIARY: spell a sample of the IN-VOCABULARY words too. Without this the character model
    # sees only the out-of-vocabulary tail -- about 1 % of silver and 5 % of gold tokens -- and has
    # to learn English orthography from roughly 90 000 spellings an epoch. It is then a poor model
    # of spelling, and the open-vocabulary tail it exists to price comes out badly calibrated.
    # ⚠ THIS LOSS IS NOT PART OF P(data). It trains shared parameters on a related task, so it is
    # EXCLUDED from every perplexity and surprisal figure; only the out-of-vocabulary term above is
    # the model's probability of the corpus. Including it would silently double-charge in-vocabulary
    # words, which no diagnostic would report.
    l_aux = torch.zeros((), dtype=l_word.dtype)
    if model.chars is not None and char_sample > 0:
        # Ids 0-5 are the reserved slots (<pad> <unk> <bos> <dummy> <notok> <root>); they have no
        # spelling, and teaching the model to "spell" <root> as a bare end-of-word is noise.
        pick = b["is_shift"] & m & (b["tgt_word"] >= N_RESERVED)
        if pick.any():
            idx = pick.nonzero(as_tuple=False)
            keep = torch.rand(len(idx), generator=gen) < char_sample
            idx = idx[keep]
            if len(idx):
                z_aux = z[idx[:, 0], idx[:, 1]]
                wid = b["tgt_word"][idx[:, 0], idx[:, 1]]
                l_aux = -model.chars.logprob(z_aux, model.char_table[wid]).sum()
    n_words = (b["is_shift"] & m).sum()
    return l_shift, l_word, l_name, l_label, n_words, l_char, l_aux


def evaluate(model, data, batch_size):
    model.eval()
    tot = np.zeros(5)
    n = 0
    with torch.no_grad():
        for b in batches(data, batch_size, shuffle=False):
            ls = losses(model, b)          # char_sample=0: perplexity excludes the aux task
            tot += [float(x) for x in (ls[0], ls[1], ls[2], ls[3], ls[5])]
            n += int(ls[4])
    # ⚠ ppl_lex here INCLUDES the spelling term, so it is NOT comparable with a run that had no
    # character model: that run's `<unk>` was a single cheap outcome and this one has to spell the
    # word. The open-vocabulary model is charged for something the closed one got free.
    lex = tot[0] + tot[1] + tot[4]
    return dict(ppl_lex=math.exp(lex / n), ppl_joint=math.exp(tot.sum() / n),
                bits_struct=(tot[2] + tot[3]) / n / math.log(2),
                bits_char=tot[4] / n / math.log(2), words=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "silver"])
    ap.add_argument("--init-from", help="warm start from a checkpoint built on the SAME vocabulary")
    ap.add_argument("--char-sample", type=float, default=0.25,
                    help="fraction of in-vocabulary words the character model also learns to spell")
    ap.add_argument("--char-aux-weight", type=float, default=1.0)
    ap.add_argument("--lex-weight", type=float, default=1.0,
                    help="weight on the LEXICAL half of the objective (shift type, word, spelling) "
                         "relative to the structural half (reduce action, label). 1.0 is the joint "
                         "model; lower trades the language model for the parser. It changes the "
                         "training pressure only -- the model still normalises over the whole "
                         "vocabulary, so surprisal stays well defined at every setting.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    vocab = Vocab(d["words"], d["labels"], d.get("chars"))
    train, dev = d[args.split], d["dev"]
    model = LCModel(len(vocab.words), len(vocab.labels), len(vocab.chars))
    if vocab.chars and len(vocab.chars) > 3:
        model.set_char_table(vocab)
    gen = torch.Generator().manual_seed(args.seed + 7)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if (ck["words"] != vocab.words or ck["labels"] != vocab.labels
                or ck.get("chars") != vocab.chars):
            # Refuse rather than warn: a mismatched warm start does not fail, it trains from
            # scrambled weights and merely converges worse (CLAUDE.md hazard 7).
            raise SystemExit("--init-from was built on a different vocabulary; rebuild both stages "
                             "from one lc_features.py run")
        model.load_state_dict(ck["model"])
        print("warm started from %s (epoch %d, dev ppl_joint %.2f)"
              % (args.init_from, ck["epoch"], ck["dev"]["ppl_joint"]))
    print("split=%s  %d sentences" % (args.split, len(train)))
    print("%s  %.2f M parameters" % (vocab, sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)

    best, bad = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for b in batches(train, args.batch_size, seed=args.seed * 100 + ep):
            ls = losses(model, b, args.char_sample, gen)
            lex = args.lex_weight * (ls[0] + ls[1] + ls[5])     # shift type + word + spelling
            struct = ls[2] + ls[3]                              # reduce action + label
            loss = (lex + struct + args.char_aux_weight * ls[6]) / ls[4]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss) * int(ls[4])
            seen += int(ls[4])
        m = evaluate(model, dev, args.batch_size)
        sched.step(m["ppl_joint"])
        print("epoch %2d  %5.0fs  train/word %6.3f | dev ppl_lex %8.2f  ppl_joint %8.2f  "
              "struct %.3f  char %.3f bits/word"
              % (ep, time.time() - t0, run / seen, m["ppl_lex"], m["ppl_joint"],
                 m["bits_struct"], m["bits_char"]), flush=True)
        if m["ppl_joint"] < best - 0.05:
            best, bad = m["ppl_joint"], 0
            torch.save({"model": model.state_dict(),
                        "words": vocab.words, "labels": vocab.labels,
                        "chars": vocab.chars,
                        "dev": m, "epoch": ep}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop; best dev ppl_joint %.2f" % best)
                break
    print("wrote", args.out)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    sys.exit(main())
