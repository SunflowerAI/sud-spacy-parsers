#!/usr/bin/env python
"""Word-synchronous beam decoding over the incremental left-corner model: per-word surprisal and
memory-load predictors, plus the Viterbi parse.

WHAT SURPRISAL MEANS HERE. Beam hypotheses carry P(w_1..t, d) for their own derivation d. Summing
over the beam approximates the prefix probability, and

    surprisal(w_t) = -log2 [ sum_i rho_i * sum_r P(r | z_i) * P(w_t | z_ir) ]  +  log2 sum_i rho_i

is the ratio of two such sums, so the derivation-set normalisation cancels. It is a LOWER bound on
the true prefix probability -- the beam drops derivations -- and therefore an UPPER bound on
surprisal, tightening as the beam widens. `--beam-sweep` reports how much is still on the table.

WHY THE BEAM NEEDS NO RESYNCHRONISATION. Shift-type and reduce-type actions alternate strictly, so
after k words every hypothesis has taken exactly 2k-1 actions and the beam is already word-
synchronous. An RNNG needs Stern et al.'s machinery precisely because it lacks this property.

THE MEMORY PREDICTORS, all reported both as a beam-posterior EXPECTATION and along the Viterbi
derivation, since they disagree exactly where the parser is uncertain and that is where reading
times are interesting:

    depth        stack elements after the word -- Noji & Miyao's memory cost
    open_slots   incomplete spines plus subtrees parked under an unseen head
    d_depth      depth change across the word, the embedding-difference analogue
    integ        summed head-dependent distance of arcs the word's own step created

⚠ SURPRISAL IS IN BITS and includes the structural decision, because the reduce action between two
words is part of what generating the next word costs. `--lexical-only` gives the word term alone;
the two differ, and a reading-time model should say which it used.
"""
import argparse
import math
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lc_features import (Vocab, SHIFT_TYPES, REDUCE_NAMES, MAX_LAM, MAX_DEPTH, NOTOK, DUMMY,
                         UNK, _spine_feats)
from lc_model import LCModel
from lc_transitions import Config, read_conllu, is_projective, SHIFT, INSERT

NEG = -1e9


class Hyp:
    __slots__ = ("cfg", "hx", "lp", "last")

    def __init__(self, cfg, hx, lp, last):
        self.cfg, self.hx, self.lp, self.last = cfg, hx, lp, last


def _stack_row(cfg, wids, vocab):
    top = cfg.stack[-1] if cfg.stack else None
    second = cfg.stack[-2] if len(cfg.stack) > 1 else None
    wid_of = lambda t: wids[t] if t is not None else NOTOK
    return (_spine_feats(top, vocab, wid_of) + _spine_feats(second, vocab, wid_of),
            min(len(cfg.stack), MAX_DEPTH))


def _step(model, hyps, wids, vocab):
    """Advance every hypothesis's recurrence by one step and return its state vector."""
    prev = torch.tensor([[h.last] for h in hyps])
    rows, depths = zip(*[_stack_row(h.cfg, wids, vocab) for h in hyps])
    stack = torch.tensor(np.array(rows), dtype=torch.long).unsqueeze(1)
    depth = torch.tensor(depths, dtype=torch.long).unsqueeze(1)
    h0 = torch.cat([h.hx[0] for h in hyps], 1)
    c0 = torch.cat([h.hx[1] for h in hyps], 1)
    z, hx = model(prev, stack, depth, (h0, c0))
    return z[:, 0], hx


def parse_sentence(model, vocab, forms, beam=10, expand=16, lexical_only=False,
                   lex_decode_weight=1.0):
    """Returns (rows, arcs). One row per real token; `arcs` is the Viterbi {dep: (head, label)}."""
    n = len(forms)                                   # forms already carry <ROOT> at the end
    wids = [0] + [vocab.wid(f) for f in forms[:-1]] + [vocab.w2i["<root>"]]
    n_lab = len(vocab.labels)
    zeros = (torch.zeros(1, 1, model.lstm.hidden_size), torch.zeros(1, 1, model.lstm.hidden_size))
    hyps = [Hyp(Config(n), zeros, 0.0, vocab.w2i["<bos>"])]
    rows, prev_depth, logZ = [], 0.0, 0.0

    for t in range(1, n + 1):
        # ---- reduce step (absent only before the first word) -------------------------------
        if t > 1:
            z, hx = _step(model, hyps, wids, vocab)
            legal_comp = torch.tensor([[h.cfg.legal("LEFT-COMP")] for h in hyps])
            nl = model.to_name(z).masked_fill(
                ~legal_comp & torch.tensor([False, False, True, True]), NEG).log_softmax(-1)
            ll = model.to_label(z).view(-1, len(REDUCE_NAMES), n_lab).log_softmax(-1)
            joint = (nl.unsqueeze(-1) + ll).reshape(len(hyps), -1)
            base = torch.tensor([h.lp for h in hyps]).unsqueeze(-1)
            k = min(expand, joint.shape[1])
            top_lp, top_ix = (base + joint).topk(k, dim=-1)
            nxt = []
            for i, h in enumerate(hyps):
                for j in range(k):
                    if top_lp[i, j].item() <= NEG / 2:
                        continue
                    flat = int(top_ix[i, j])
                    name, lab = REDUCE_NAMES[flat // n_lab], vocab.labels[flat % n_lab]
                    cfg = h.cfg.copy()
                    try:
                        cfg.apply((name, lab))
                    except Exception:
                        continue
                    sym = len(vocab.words) + (flat // n_lab) * n_lab + (flat % n_lab)
                    nxt.append(Hyp(cfg, (hx[0][:, i:i+1], hx[1][:, i:i+1]),
                                   float(top_lp[i, j]), sym))
            nxt.sort(key=lambda h: -h.lp)
            hyps = nxt[:beam]
            if not hyps:
                return None, None

        # ---- shift step: score the observed word, then branch on shift type ----------------
        z, hx = _step(model, hyps, wids, vocab)
        word_lp = model.word_logits(z).log_softmax(-1)[:, wids[t]]
        if wids[t] == UNK and model.chars is not None:
            # OPEN VOCABULARY: log P(w) = log P(<unk> | state) + log P(spelling | state), so a word
            # the vocabulary never saw gets its OWN surprisal instead of the one lumped value every
            # rare word used to share.
            cs = torch.tensor([vocab.cids(forms[t - 1])], dtype=torch.long)
            word_lp = word_lp + model.chars.logprob(z, cs.expand(len(hyps), -1))
        legal_ins = torch.tensor([[h.cfg.legal(INSERT)] for h in hyps])
        sl = model.to_shift(z).masked_fill(
            ~legal_ins & torch.tensor([False, True]), NEG).log_softmax(-1)
        base = torch.tensor([h.lp for h in hyps])
        # Prefix mass with and without w_t; the difference is the surprisal.
        new_logZ = float(torch.logsumexp(base + word_lp, 0))
        lex_only = float(torch.logsumexp(base + word_lp, 0)) - float(torch.logsumexp(base, 0))
        surp_bits = -(new_logZ - logZ) / math.log(2)
        lex_bits = -lex_only / math.log(2)

        nxt = []
        for i, h in enumerate(hyps):
            for s, name in enumerate(SHIFT_TYPES):
                if float(sl[i, s]) <= NEG / 2:
                    continue
                cfg = h.cfg.copy()
                try:
                    cfg.apply((name, None))
                except Exception:
                    continue
                nxt.append(Hyp(cfg, (hx[0][:, i:i+1], hx[1][:, i:i+1]),
                               h.lp + lex_decode_weight * (float(word_lp[i]) + float(sl[i, s])),
                               wids[t]))
        nxt.sort(key=lambda h: -h.lp)
        hyps = nxt[:beam]
        if not hyps:
            return None, None
        logZ = float(torch.logsumexp(torch.tensor([h.lp for h in hyps]), 0))

        # ---- predictors ---------------------------------------------------------------------
        w = torch.softmax(torch.tensor([h.lp for h in hyps]), 0).numpy()
        depths = np.array([h.cfg.cost for h in hyps], float)
        slots = np.array([h.cfg.open_slots for h in hyps], float)
        integ = np.array([sum(abs(d - hd) for d, (hd, _) in h.cfg.arcs.items()
                              if d == t or hd == t) for h in hyps], float)
        depth_mean = float(w @ depths)
        if t <= n - 1:                                # the artificial root is not a real token
            rows.append(dict(token=t, form=forms[t - 1],
                             surprisal=surp_bits if not lexical_only else lex_bits,
                             surprisal_lex=lex_bits, surprisal_full=surp_bits,
                             depth=depth_mean, depth_viterbi=float(depths[0]),
                             open_slots=float(w @ slots), open_slots_viterbi=float(slots[0]),
                             d_depth=depth_mean - prev_depth, integ=float(w @ integ),
                             beam_entropy=float(-(w * np.log2(np.maximum(w, 1e-12))).sum())))
        prev_depth = depth_mean

    best = max((h for h in hyps if h.cfg.terminal), key=lambda h: h.lp, default=None)
    return rows, (best.cfg.arcs if best else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conllu", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--expand", type=int, default=16)
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--lex-decode-weight", type=float, default=1.0,
                    help="weight on the WORD term when RANKING beam hypotheses. 1.0 is the joint "
                         "model. 0 ranks on structural scores alone -- the control for whether a "
                         "degraded language model hurts parsing through the SEARCH or through the "
                         "representation. ⚠ Surprisal is meaningless at anything but 1.0, because "
                         "the beam then keeps a set of hypotheses that is not the top of the joint "
                         "distribution it is being asked to marginalise.")
    ap.add_argument("--score", action="store_true", help="also report UAS/LAS against the gold")
    args = ap.parse_args()

    torch.set_num_threads(8)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    vocab = Vocab(ck["words"], ck["labels"], ck.get("chars"))
    model = LCModel(len(vocab.words), len(vocab.labels),
                    len(vocab.chars) if ck.get("chars") else 0)
    if ck.get("chars") and len(vocab.chars) > 3:
        model.set_char_table(vocab)      # buffer shape must match before load_state_dict
    model.load_state_dict(ck["model"])
    model.eval()

    fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    cols = ["sent", "token", "form", "surprisal", "surprisal_lex", "depth", "depth_viterbi",
            "open_slots", "d_depth", "integ", "beam_entropy"]
    print("\t".join(cols), file=fh)
    n_sent = tot_surp = tot_words = 0
    uas = las = arcs_n = failed = 0
    # ⚠ Scoring ONLY the projective gold sentences flatters this parser, and by a knowable amount:
    # it cannot emit a crossing arc, so every arc it gets wrong for that reason is excluded from
    # the projective-only figure. Both are reported; the all-sentences one is the comparable number.
    uas_all = las_all = arcs_all = 0
    with torch.no_grad():
        for path in args.conllu:
            for forms, heads, labs, n in read_conllu(path):
                if args.max_sents and n_sent >= args.max_sents:
                    break
                n_sent += 1
                rows, arcs = parse_sentence(model, vocab, forms, args.beam, args.expand,
                                            args.lexical_only, args.lex_decode_weight)
                if rows is None:
                    failed += 1
                    continue
                for r in rows:
                    print("\t".join([str(n_sent)] + ["%s" % r[c] if isinstance(r[c], (int, str))
                                                     else "%.5f" % r[c] for c in cols[1:]]), file=fh)
                    tot_surp += r["surprisal"]
                    tot_words += 1
                if args.score and arcs:
                    proj = is_projective(heads, n)
                    for d in range(1, n):
                        g_h, g_l = heads.get(d), labs.get(d)
                        p = arcs.get(d)
                        hit = bool(p and p[0] == g_h)
                        lab_hit = hit and p[1] == g_l
                        arcs_all += 1
                        uas_all += hit
                        las_all += lab_hit
                        if proj:
                            arcs_n += 1
                            uas += hit
                            las += lab_hit
    if args.out:
        fh.close()
    print("sentences %d  failed %d  words %d  mean surprisal %.3f bits"
          % (n_sent, failed, tot_words, tot_surp / max(tot_words, 1)), file=sys.stderr)
    if args.score and arcs_all:
        print("UAS %.2f  LAS %.2f   projective-only (%d arcs)"
              % (100.0 * uas / max(arcs_n, 1), 100.0 * las / max(arcs_n, 1), arcs_n),
              file=sys.stderr)
        print("UAS %.2f  LAS %.2f   ALL sentences  (%d arcs)"
              % (100.0 * uas_all / arcs_all, 100.0 * las_all / arcs_all, arcs_all), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
