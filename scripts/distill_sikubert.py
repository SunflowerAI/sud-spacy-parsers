#!/usr/bin/env python3
"""TinyBERT-style GENERAL distillation of SIKU-BERT/sikubert into a 4-layer x 312 student (CPU).

WHY. The English experiment found that a FROZEN TinyBERT-4L-312D plus a learned layer mix beats the
hash-embed baseline parser (`sud.TinyBertPlusHash.v1`, scripts/sud_tinybert_embed.py). No distilled
SikuBERT exists on the hub, so the same recipe for lzh needs one made here. The student is an
ordinary `transformers.BertModel`, so `sud_tinybert_embed.py` loads it unchanged, and it REUSES THE
TEACHER'S VOCAB (tokenizer files are saved beside the weights), so lzh characters tokenise identically.

STUDENT. BertConfig: 4 layers, hidden 312, 12 heads (26 per head), FFN 1200, vocab 29791, 512
positions (all taken from the teacher's config except the four sizes).

LOSS (all MSE, teacher in eval/no_grad, student layer i = 1..4 <-> teacher layer 3i):
  emb   : proj_emb(student embedding output)        vs teacher embedding output     (hidden_states[0])
  hid   : proj_hid(student hidden_states[i])        vs teacher hidden_states[3i]
  attn  : student scores of layer i                 vs teacher scores of layer 3i
  The two projections are learned 312 -> 768 linears (`proj_emb`, and `proj_hid` shared by the four
  layers, as in TinyBERT). They exist only for training; they are saved in `distill_heads.pt`, which
  BertModel.from_pretrained ignores.

ATTENTION TARGET = PRE-SOFTMAX SCORES (q.k / sqrt(d_head)), not post-softmax probabilities. Why: (1)
TinyBERT reports unnormalised scores distil better and converge faster; probabilities are tiny and
near-one-hot, so MSE on them is dominated by a handful of entries and gives vanishing gradients for
the rest; (2) it is shift-invariant under softmax, which is a nuisance MSE does not know about, but
scores are what both sides actually compute, so the match is exact in the teacher's own units. Both
sides have 12 heads, so head h maps to head h. HF's sdpa path never materialises scores, so they are
recomputed from forward hooks on `attention.self.query` / `.key` (one extra small matmul per layer).
Only (valid query, valid key) pairs count; padding never enters a loss.

EMBEDDING INIT. The student's word/position/token-type tables are the teacher's, projected on the
top-312 principal directions V of the teacher's word-embedding matrix:  word' = (E - mu) V,
pos' = P V, type' = T V + mu V  (the mean goes into the token-type row, which is always type 0), so
the student's pre-LayerNorm embedding sum IS the best rank-312 approximation of the teacher's sum.
`proj_emb` starts as the map x -> diag(gamma_teacher) V x with bias beta_teacher and `proj_hid` as x -> V x, so the
embedding loss starts low and, more importantly, the student starts with each character already
sitting near its teacher neighbours instead of at random -- the vocabulary is 29 791 types and a
42 M-token corpus gives most of them few occurrences, so random embeddings would be learned slowly
and badly. Everything above the embeddings starts from the standard BERT init.

DATA. `corpus_lzh_kanripo_leakfree.txt` (NOT the .train.txt/.ids.txt copies: those are an
IDS-decomposed variant, 613 MB, with ⿰⿱ strings that SikuBERT's vocab does not contain). One line per
sentence, tokens space-separated; spaces are removed so every token is a character. Consecutive lines
are concatenated in file order (kanripo is in text order) and cut into windows of up to
`--seq-len - 2` characters (+[CLS],[SEP]); `--short-prob` of the windows get a random shorter length
because the parser will feed this encoder single sentences (~25 tokens), not 128-char blocks.
Streaming and deterministic: the file is read in blocks of `--block-lines` lines, each block's
windows are shuffled with RNG(seed, epoch, block) and grouped into token-budget batches (windows of
similar length together, padding to the longest in the batch). The data position is (epoch, block,
batch-in-block), so a resumed run replays exactly the stream it would have seen. Characters are
mapped with the tokenizer's own output (cached per character), so tokenisation is the teacher's.

Usage:
  distill_sikubert.py --teacher DIR --corpus FILE --out DIR --total-tokens 200e6 --threads 12
  distill_sikubert.py --eval-only --teacher DIR --student DIR_OR_CKPT --eval-conllu dev.conllu
"""
import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig, BertModel

TEACHER_LAYER = {1: 3, 2: 6, 3: 9, 4: 12}  # student layer -> teacher layer


# ----------------------------------------------------------------------------- data
class CharMapper:
    """character -> vocab id, exactly as the teacher's tokenizer would produce it."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def __call__(self, ch):
        v = self.cache.get(ch)
        if v is None:
            ids = self.tok(ch, add_special_tokens=False)["input_ids"]
            v = ids[0] if len(ids) == 1 else self.tok.unk_token_id
            self.cache[ch] = v
        return v

    def encode(self, text):
        return np.fromiter((self(c) for c in text), dtype=np.int32, count=len(text))


def read_block(path, block, block_lines):
    out, start = [], block * block_lines
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if i >= start + block_lines:
                break
            out.append("".join(line.split()))
    return out


def n_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def block_batches(ids, rng, cfg, specials):
    """Windows of one block -> list of batches (each a list of int32 arrays, specials included)."""
    cls, sep = specials
    maxc = cfg["seq_len"] - 2
    wins, p, n = [], 0, len(ids)
    while p < n:
        L = maxc if rng.random() >= cfg["short_prob"] else rng.randint(6, maxc)
        w = ids[p:p + L]
        p += L
        if len(w) >= 4:
            wins.append(np.concatenate(([cls], w, [sep])).astype(np.int64))
    rng.shuffle(wins)
    # pools of sorted-by-length windows -> token-budget batches -> shuffled batch order
    batches, pool = [], 512
    for i in range(0, len(wins), pool):
        chunk = sorted(wins[i:i + pool], key=len)
        cur, curmax = [], 0
        for w in chunk:
            m = max(curmax, len(w))
            if cur and m * (len(cur) + 1) > cfg["batch_tokens"]:
                batches.append(cur)
                cur, m = [], len(w)
            cur.append(w)
            curmax = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def stream(cfg, tok, specials, state):
    """Yield (batch, (epoch, block, batch_idx_next)) forever, from `state`, deterministically."""
    mapper = CharMapper(tok)
    nl = cfg["n_lines"]
    nblocks = math.ceil(nl / cfg["block_lines"])
    ep, blk, bi = state
    while True:
        while blk < nblocks:
            lines = read_block(cfg["corpus"], blk, cfg["block_lines"])
            ids = mapper.encode("".join(lines))
            rng = random.Random(cfg["seed"] * 1000003 + ep * 10007 + blk)
            batches = block_batches(ids, rng, cfg, specials)
            for j in range(bi, len(batches)):
                yield batches[j], (ep, blk, j + 1)
            blk, bi = blk + 1, 0
        ep, blk, bi = ep + 1, 0, 0


def collate(batch):
    L = max(len(w) for w in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, w in enumerate(batch):
        ids[i, :len(w)] = torch.from_numpy(w)
        mask[i, :len(w)] = True
    return ids, mask


# ----------------------------------------------------------------------------- models
class ScoreTap:
    """Forward hooks that capture q and k of chosen layers and rebuild pre-softmax scores."""

    def __init__(self, bert, layers):
        self.q, self.k, self.nh, self.hs = {}, {}, bert.config.num_attention_heads, None
        self.hs = bert.config.hidden_size // self.nh
        for li in layers:  # li is the 0-based layer index
            sa = bert.encoder.layer[li].attention.self
            sa.query.register_forward_hook(lambda m, i, o, li=li: self.q.__setitem__(li, o))
            sa.key.register_forward_hook(lambda m, i, o, li=li: self.k.__setitem__(li, o))

    def scores(self, li):
        q, k = self.q[li], self.k[li]
        B, T, _ = q.shape
        q = q.view(B, T, self.nh, self.hs).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hs).transpose(1, 2)
        return q @ k.transpose(-1, -2) / math.sqrt(self.hs)  # B,H,T,T

    def clear(self):
        self.q.clear()
        self.k.clear()


class Distiller(torch.nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher, self.student = teacher, student
        ts, tt = student.config.hidden_size, teacher.config.hidden_size
        self.proj_emb = torch.nn.Linear(ts, tt)
        self.proj_hid = torch.nn.Linear(ts, tt)
        self.t_tap = ScoreTap(teacher, [TEACHER_LAYER[i] - 1 for i in TEACHER_LAYER])
        self.s_tap = ScoreTap(student, [i - 1 for i in TEACHER_LAYER])
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # the teacher is never in train mode
        return self

    def forward(self, ids, mask):
        with torch.no_grad():
            self.t_tap.clear()
            th = self.teacher(input_ids=ids, attention_mask=mask.long(),
                              output_hidden_states=True).hidden_states
            tsc = {i: self.t_tap.scores(TEACHER_LAYER[i] - 1) for i in TEACHER_LAYER}
        self.s_tap.clear()
        sh = self.student(input_ids=ids, attention_mask=mask.long(),
                          output_hidden_states=True).hidden_states
        m = mask.unsqueeze(-1).float()
        denom = m.sum() * th[0].shape[-1]

        def mse(a, b):
            return (((a - b) ** 2) * m).sum() / denom

        l_emb = mse(self.proj_emb(sh[0]), th[0])
        l_hid = sum(mse(self.proj_hid(sh[i]), th[TEACHER_LAYER[i]]) for i in TEACHER_LAYER) / 4
        pair = (mask[:, None, :, None] & mask[:, None, None, :]).float()
        pd = pair.sum() * self.student.config.num_attention_heads
        l_att = sum((((self.s_tap.scores(i - 1) - tsc[i]) ** 2) * pair).sum() / pd
                    for i in TEACHER_LAYER) / 4
        with torch.no_grad():
            cos = F.cosine_similarity(self.proj_hid(sh[4]), th[12], dim=-1)
            cos = (cos * mask).sum() / mask.sum()
        return l_emb, l_hid, l_att, cos


def build_student(teacher, args):
    tc = teacher.config
    cfg = BertConfig(vocab_size=tc.vocab_size, hidden_size=312, num_hidden_layers=4,
                     num_attention_heads=12, intermediate_size=1200,
                     max_position_embeddings=tc.max_position_embeddings,
                     type_vocab_size=tc.type_vocab_size, hidden_act=tc.hidden_act,
                     layer_norm_eps=tc.layer_norm_eps, pad_token_id=tc.pad_token_id,
                     hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout)
    return BertModel(cfg, add_pooling_layer=False)


@torch.no_grad()
def init_from_teacher(dist):
    t, s = dist.teacher.embeddings, dist.student.embeddings
    E = t.word_embeddings.weight.double()
    mu = E.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(E - mu, full_matrices=False)
    ds = s.word_embeddings.weight.shape[1]
    V = Vh[:ds].T  # 768 x 312
    var = (S[:ds] ** 2).sum() / (S ** 2).sum()
    s.word_embeddings.weight.copy_(((E - mu) @ V).float())
    s.position_embeddings.weight.copy_((t.position_embeddings.weight.double() @ V).float())
    s.token_type_embeddings.weight.copy_(
        (t.token_type_embeddings.weight.double() @ V + mu @ V).float())
    g, b = t.LayerNorm.weight.double(), t.LayerNorm.bias.double()
    dist.proj_emb.weight.copy_((g[:, None] * V).float())
    dist.proj_emb.bias.copy_(b.float())
    dist.proj_hid.weight.copy_(V.float())
    dist.proj_hid.bias.zero_()
    return float(var)


# ----------------------------------------------------------------------------- checkpoints
def save_hf(dist, tok, outdir, extra=None, teacher_dir=None):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dist.student.save_pretrained(outdir)
    tok.save_pretrained(outdir)  # tokenizer.json + config, loaded from the TEACHER
    if teacher_dir:  # the fast tokenizer does not write vocab.txt; copy the teacher's, byte for byte
        import shutil
        shutil.copyfile(pathlib.Path(teacher_dir) / "vocab.txt", outdir / "vocab.txt")
    torch.save({"proj_emb": dist.proj_emb.state_dict(), "proj_hid": dist.proj_hid.state_dict()},
               outdir / "distill_heads.pt")
    if extra:
        (outdir / "distill_info.json").write_text(json.dumps(extra, indent=1))


def save_ckpt(path, dist, opt, step, tokens, dstate, args):
    tmp = str(path) + ".tmp"
    torch.save({"student": dist.student.state_dict(), "proj_emb": dist.proj_emb.state_dict(),
                "proj_hid": dist.proj_hid.state_dict(), "opt": opt.state_dict(), "step": step,
                "tokens": tokens, "data_state": dstate, "args": vars(args)}, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-write cannot corrupt the last good checkpoint


def log(fh, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


# ----------------------------------------------------------------------------- eval
def conllu_sentences(path):
    out, cur = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            if cur:
                out.append("".join(cur))
            cur = []
        elif not line.startswith("#"):
            c = line.split("\t")
            if len(c) > 1 and "-" not in c[0] and "." not in c[0]:
                cur.append(c[1])
    if cur:
        out.append("".join(cur))
    return out


@torch.no_grad()
def evaluate(dist, tok, specials, sents, bs=64, maxlen=126):
    dist.eval()
    mapper = CharMapper(tok)
    seqs = [np.concatenate(([specials[0]], mapper.encode(s)[:maxlen], [specials[1]])).astype(np.int64)
            for s in sents if s]
    seqs.sort(key=len)
    tot = np.zeros(4)
    ntok = 0
    ctok = 0.0
    for i in range(0, len(seqs), bs):
        ids, mask = collate(seqs[i:i + bs])
        le, lh, la, cos = dist(ids, mask)
        n = int(mask.sum())
        tot += np.array([float(le), float(lh), float(la), float(cos)]) * n
        ntok += n
    r = tot / ntok
    return {"emb_mse": r[0], "hid_mse": r[1], "attn_mse": r[2], "cos_last": r[3],
            "sentences": len(seqs), "tokens": ntok}


def leak_check(corpus, conllus, min_len=6):
    """Count dev/test sentences found verbatim (line equality) or as a substring in the corpus."""
    held = set()
    for p in conllus:
        held.update(s for s in conllu_sentences(p) if len(s) >= min_len)
    held_s = sorted(held)
    lines = ["".join(l.split()) for l in open(corpus, encoding="utf-8")]
    lineset = set(lines)
    eq = sum(1 for s in held_s if s in lineset)
    blob = "\n".join(lines)
    sub = sum(1 for s in held_s if s in blob)
    return {"heldout_sentences(len>=%d)" % min_len: len(held_s), "equal_to_a_line": eq,
            "substring_of_corpus": sub}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="dir with SIKU-BERT/sikubert")
    ap.add_argument("--corpus", default="data/corpus_lzh_kanripo_leakfree.txt")
    ap.add_argument("--out", default="runs/siku_4l312")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--short-prob", type=float, default=0.2,
                    help="fraction of windows with a random shorter length (parser sees sentences)")
    ap.add_argument("--batch-tokens", type=int, default=8192, help="padded tokens per step")
    ap.add_argument("--block-lines", type=int, default=200000)
    ap.add_argument("--total-tokens", type=float, default=50e6, help="training budget (real tokens)")
    ap.add_argument("--max-steps", type=int, default=0, help="override: stop after this many steps")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-conllu", default="data/lzh_kyoto-sud-dev.relabeled_ext.udep_ruled.punct."
                    "rulemerged.conllu")
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--student", help="eval-only: HF dir of a saved student, or a ckpt .pt")
    ap.add_argument("--leak-check", nargs="*", help="conllu files; print overlap with the corpus, exit")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    if args.leak_check is not None:
        print(json.dumps(leak_check(args.corpus, args.leak_check), indent=1))
        return

    tok = AutoTokenizer.from_pretrained(args.teacher)
    specials = (tok.cls_token_id, tok.sep_token_id)
    teacher = BertModel.from_pretrained(args.teacher, add_pooling_layer=False,
                                        attn_implementation="sdpa").eval()
    student = build_student(teacher, args)
    dist = Distiller(teacher, student)
    var = init_from_teacher(dist)
    nparam = sum(p.numel() for p in student.parameters())
    log(logf, f"student params {nparam:,} (embeddings {student.embeddings.word_embeddings.weight.numel():,}); "
              f"PCA312 keeps {var:.3f} of word-embedding variance; threads {args.threads}")

    sents = conllu_sentences(args.eval_conllu)[:args.eval_n] if os.path.exists(args.eval_conllu) else []

    if args.eval_only:
        if args.student:
            p = pathlib.Path(args.student)
            if p.suffix == ".pt":
                ck = torch.load(p, map_location="cpu")
                dist.student.load_state_dict(ck["student"])
                dist.proj_emb.load_state_dict(ck["proj_emb"])
                dist.proj_hid.load_state_dict(ck["proj_hid"])
            else:
                dist.student = BertModel.from_pretrained(p, add_pooling_layer=False)
                heads = torch.load(p / "distill_heads.pt", map_location="cpu")
                dist.proj_emb.load_state_dict(heads["proj_emb"])
                dist.proj_hid.load_state_dict(heads["proj_hid"])
                dist.s_tap = ScoreTap(dist.student, [i - 1 for i in TEACHER_LAYER])
        r = evaluate(dist, tok, specials, sents)
        log(logf, "EVAL " + json.dumps(r))
        return

    cfg = dict(corpus=args.corpus, seq_len=args.seq_len, short_prob=args.short_prob,
               batch_tokens=args.batch_tokens, block_lines=args.block_lines, seed=args.seed)
    cfg["n_lines"] = n_lines(args.corpus)

    decay, no_decay = [], []
    for n, p in dist.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "LayerNorm" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr,
                            betas=(0.9, 0.999), eps=1e-6)

    # steps are estimated from the token budget; batches hold ~batch_tokens padded tokens, of which
    # ~`fill` are real. fill is re-estimated from the first batches and total_steps is fixed once.
    step, tokens, dstate = 0, 0, (0, 0, 0)
    ckpath = out / "ckpt.pt"
    if ckpath.exists():
        ck = torch.load(ckpath, map_location="cpu")
        dist.student.load_state_dict(ck["student"])
        dist.proj_emb.load_state_dict(ck["proj_emb"])
        dist.proj_hid.load_state_dict(ck["proj_hid"])
        opt.load_state_dict(ck["opt"])
        step, tokens, dstate = ck["step"], ck["tokens"], tuple(ck["data_state"])
        log(logf, f"RESUMED at step {step}, {tokens:,} tokens, data state {dstate}")
    fill = 0.97
    total_steps = args.max_steps or int(args.total_tokens / (args.batch_tokens * fill))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr * max(0.0, (total_steps - s) / max(1, total_steps - args.warmup))

    log(logf, f"total_steps {total_steps}  lr {args.lr}  warmup {args.warmup}  batch_tokens "
              f"{args.batch_tokens}  corpus lines {cfg['n_lines']:,}")
    if step == 0 and sents:
        log(logf, "EVAL@0 " + json.dumps(evaluate(dist, tok, specials, sents)))

    dist.train()
    t0, tw, tt = time.time(), time.time(), 0
    acc = np.zeros(4)
    nacc = 0
    gen = stream(cfg, tok, specials, dstate)
    while step < total_steps:
        batch, dstate = next(gen)
        ids, mask = collate(batch)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        le, lh, la, cos = dist(ids, mask)
        loss = le + lh + la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dist.parameters(), args.clip)
        opt.step()
        step += 1
        n = int(mask.sum())
        tokens += n
        tt += n
        acc += np.array([float(le), float(lh), float(la), float(cos)])
        nacc += 1
        if step % args.log_every == 0:
            now = time.time()
            a = acc / nacc
            log(logf, f"step {step}/{total_steps} tok {tokens/1e6:.2f}M lr {lr_at(step):.2e} "
                      f"emb {a[0]:.4f} hid {a[1]:.4f} attn {a[2]:.4f} cos {a[3]:.3f} "
                      f"| {tt/(now-tw):.0f} tok/s")
            acc[:] = 0
            nacc = 0
            tw, tt = now, 0
        if step % args.ckpt_every == 0:
            save_ckpt(ckpath, dist, opt, step, tokens, dstate, args)
        if sents and args.eval_every and step % args.eval_every == 0:
            log(logf, f"EVAL@{step} " + json.dumps(evaluate(dist, tok, specials, sents)))
            dist.train()
    save_ckpt(ckpath, dist, opt, step, tokens, dstate, args)
    info = {"steps": step, "tokens": tokens, "wall_s": time.time() - t0, "args": vars(args)}
    save_hf(dist, tok, out / "final", info, teacher_dir=args.teacher)
    if sents:
        log(logf, "EVAL@final " + json.dumps(evaluate(dist, tok, specials, sents)))
    log(logf, f"saved HF student to {out/'final'}")


if __name__ == "__main__":
    main()
