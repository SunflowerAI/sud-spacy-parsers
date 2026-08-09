#!/usr/bin/env python3
"""The CSLiser with Apte stem-lexicon membership as an INPUT FEATURE, not a decoding rule.

Two attempts to use the lexicon at inference both failed, and for the same reason: at ~70 % recall
over compound stems, membership is far too noisy to act on as a hard decision.

    beam + lexicon rescoring   -0.11 F / -0.40 PM in domain   (where the model is unsure, every
                                                               candidate segmentation is real words)
    greedy + lexicon repair    +-0.00 in domain, -0.80 PM unseen; fires 2x per 250 sentences
                                                              (absence usually means Apte's gap,
                                                               not the model's error)

The standard remedy for a noisy knowledge source is to hand it to the model as evidence and let it
learn the weight. That is exactly what made `Compound=Yes` worth +1.30 LAS for the sa parser: a
feature the model calibrates, rather than a rule imposed on it.

**The feature.** At each character position the question a splitter asks is "does a break belong
here?", so the evidence is whether a break here would yield attested stems on both sides. Per
position we compute a 2-bit code:

    bit 0   some attested stem ENDS at this character
    bit 1   some attested stem STARTS at the next character

giving 4 values. Value 3 — a stem ends here and another starts next — is the positive evidence for a
compound break; 0 is evidence against. Crucially the model sees the raw code and decides what it is
worth, so a 30 % false-negative rate costs calibration rather than correctness.

Only stems of length >= `min_len` count. One- and two-character strings are attested almost by
accident and would fire everywhere, drowning the signal — the same reason the repair pass used a
minimum length.

Architecture is the shipped one plus a second `Embed` table over the 4 codes, concatenated with the
character embedding before the encoder. `scripts/sa_presegment.py` is untouched, so every released
model keeps loading exactly as before.
"""
import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
from thinc.api import (Adam, Embed, Maxout, Softmax, chain, clone, concatenate, expand_window,
                       residual, with_array)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sa_presegment import PAD_CHAR, UNK_CHAR, KEEP, build_vocabs   # noqa: E402
# eval_samhita is needed only by the TRAINING loop below. Importing it at module scope drags a
# training-only dependency into every wheel that bundles this file for inference (the zh model
# does), where it is not present -- so it is imported inside main() instead.

N_LEX = 4
N_FREQ = 25          # 5 frequency bands for "word ends here" x 5 for "word starts next"


def _band(f):
    """0 = unattested, then decade buckets: 1-9, 10-99, 100-999, 1000+."""
    if f <= 0:
        return 0
    import math
    return min(4, 1 + int(math.log10(f)))


def freq_codes(text, freq, max_len=6):
    """Graded replacement for `lex_codes`, banding the FREQUENCY of the best word at each edge.

    Binary membership is nearly vacuous for Chinese: with single-character words attested, "a word
    ends here and one starts next" is true at 64 % of NON-break positions, giving 1.46x enrichment.
    Frequency separates what membership collapses -- for `这本书正是`, breaking after 正 needs `书正`
    (count 0) while breaking after 书 needs `正是` (count 2), and the binary code calls both 3.
    Banded, the extremes reach 30x enrichment at the top and 0.21x at the bottom.
    """
    n = len(text)
    best_end = [0] * n
    best_start = [0] * n
    for i in range(n):
        for L in range(1, max_len + 1):
            if i - L + 1 >= 0:
                f = freq.get(text[i - L + 1:i + 1], 0)
                if f > best_end[i]:
                    best_end[i] = f
            if i + L <= n:
                f = freq.get(text[i:i + L], 0)
                if f > best_start[i]:
                    best_start[i] = f
    return [_band(best_end[i]) * 5 + _band(best_start[i + 1] if i + 1 < n else 0)
            for i in range(n)]


def lex_codes(text, lex, min_len=3, max_len=24):
    """Per character: 1 if an entry ends here, +2 if one starts next."""
    n = len(text)
    ends = [False] * n
    starts = [False] * n
    for i in range(n):
        for L in range(min_len, min(max_len, n - i) + 1):
            if text[i:i + L] in lex:
                starts[i] = True
                ends[i + L - 1] = True
    return [(1 if ends[i] else 0) | (2 if (i + 1 < n and starts[i + 1]) else 0) for i in range(n)]


_FREQ = {"counts": None}


def enable_freq(counts, mode="count"):
    """Route source 0 through the frequency-banded extractor.

    mode="count" bands log10 of the raw count. That is unstable under jackknifing: removing a fifth
    of the data moves 14.4 % of positions across a decade boundary, against 1.8 % for binary
    membership, and the resulting train/inference mismatch cost 6.5 test F.

    mode="rank" bands by frequency RANK percentile instead. A word's rank barely moves when a fifth
    of the corpus is dropped, so the gradation survives jackknifing.
    """
    if mode == "rank" and counts:
        order = sorted(counts, key=lambda w: -counts[w])
        n = len(order)
        _FREQ["counts"] = {w: max(1, int(10 ** (4 * (1 - i / n)))) for i, w in enumerate(order)}
    else:
        _FREQ["counts"] = counts


_INFLECT = {"stems": None, "by_add": None, "rms": None}


def enable_inflect(stems_path, endings_path):
    """Route source 0 through the sandhi/inflection-aware extractor instead of set membership."""
    import sa_inflect_feature as inf
    _INFLECT["stems"], _INFLECT["by_add"], _INFLECT["rms"] = inf.build(stems_path, endings_path)
    _INFLECT["fn"] = inf.inflect_codes


_JIEBA = {"index": None, "fn": None, "tok": None, "t2s": False}


def _jieba_via_t2s(base_fn):
    """Ask jieba about the SIMPLIFIED rendering, and keep the answer for the original characters.

    jieba's dictionary is simplified, so a traditional segmenter that asks it directly gets a
    materially weaker channel. Measured on the traditional GSD test, jieba's boundary decisions
    score F 0.8920 (P 0.9287 / R 0.8580) on the traditional text and **F 0.9223** (P 0.9725 /
    R 0.8772) on its `t2s` conversion — the latter being the same channel quality the simplified
    model was built on (P 0.9730 / R 0.8793), so the whole loss is vocabulary, not the language.

    The codes are per character and `t2s` is a per-character mapping, so the answer transfers by
    position. That holds only while the conversion preserves length, which is checked rather than
    assumed: it does on 500/500 traditional GSD test sentences, and where it ever does not, the
    original text is used and the channel simply degrades to its traditional-text quality.
    """
    import opencc
    conv = opencc.OpenCC("t2s")

    def codes(text, tok):
        simp = conv.convert(text)
        return base_fn(simp if len(simp) == len(text) else text, tok)

    return codes


def enable_jieba(index, userdict=None, t2s=False):
    """Route lexicon source `index` through jieba's SEGMENTATION DECISION (BMES) instead.

    Imported lazily, and from a separate module, because `sa_presegment_lex` is bundled into every
    wheel that ships a character segmenter — the zh model has no reason to carry jieba's 5 MB
    dictionary at import time, and the sa model has no reason to carry jieba at all. Same rule that
    kept `eval_samhita` out of module scope here.

    `t2s` asks jieba about the simplified rendering (see `_jieba_via_t2s`); it is what a
    TRADITIONAL segmenter wants, and it is recorded in `vocab.json` so a loaded model cannot ask
    the question differently from the way it was trained.
    """
    import zh_jieba_feature as jf
    _JIEBA["index"] = index
    _JIEBA["t2s"] = bool(t2s)
    _JIEBA["fn"] = _jieba_via_t2s(jf.jieba_codes) if t2s else jf.jieba_codes
    _JIEBA["tok"] = jf.get_tokenizer(userdict)


def multi_codes(text, lexes, min_lens, max_len=24):
    """One independent code per lexicon SOURCE.

    Separate channels are the point, not a convenience. A source may be well aligned with the
    treebank (a word list harvested from the training split) or systematically misaligned with it
    (jieba writes `这个` as one word where SUD Chinese-GSD writes `这 / 个`). Concatenating them into
    one code would force the model to average that disagreement; giving each its own embedding lets
    it learn a different weight per source — including a negative one — so it can effectively ignore
    a source exactly where that source is wrong, and still use it where it is right.
    """
    out = []
    for k, (lx, ml) in enumerate(zip(lexes, min_lens)):
        if _JIEBA["index"] == k:
            out.append(_JIEBA["fn"](text, _JIEBA["tok"]))
        elif k == 0 and _FREQ.get("counts") is not None:
            out.append(freq_codes(text, lx if isinstance(lx, dict) else _FREQ["counts"]))
        elif k == 0 and _INFLECT.get("fn") and lx:
            out.append(_INFLECT["fn"](text, _INFLECT["stems"], _INFLECT["by_add"],
                                      _INFLECT["rms"]))
        else:
            out.append(lex_codes(text, lx, ml, max_len))
    return out


def build_lex_model(n_chars, n_labels, width=64, depth=6, window_size=1, maxout_pieces=3,
                    lex_width=8, n_sources=1, n_values=None):
    used = lex_width * n_sources
    parts = [Embed(nO=width - used, nV=n_chars, column=0)]
    nv = n_values or [N_LEX] * n_sources
    for k in range(n_sources):
        parts.append(Embed(nO=lex_width, nV=nv[k], column=1 + k))
    embed = concatenate(*parts)
    cnn = chain(
        expand_window(window_size=window_size),
        Maxout(nO=width, nI=width * (window_size * 2 + 1), nP=maxout_pieces,
               dropout=0.0, normalize=True),
    )
    encoder = clone(residual(cnn), depth)
    encoder.set_dim("nO", width)
    # `pad` is NOT optional, and leaving it off was a regression against `sa_presegment.build_model`
    # (which has always passed `pad=window_size * depth`, copying spacy.MaxoutWindowEncoder).
    # `with_array` receives a LIST of sequences and flattens it into ONE array, so without padding
    # between them the ±1 window at the first layer reads the PREVIOUS sequence's last character.
    # Measured on zh before the fix: |Δ| 0.81 on the first character of a row depending on what
    # preceded it in the call, ~0 by the third — enough to flip 60 of 529 sentence-initial splits.
    # Training batches 32 rows, so the model learned that first decision with a real neighbour and
    # then met zero padding at inference, where each text is alone in its call.
    # `depth` window-1 layers give a receptive field of `depth`, which is the padding needed.
    # NB the padding is inserted in the INT input here (one `with_array` around the whole chain),
    # not in embedding space as `build_model` does it, so the pad rows arrive as character id 0 =
    # PAD_CHAR and lexicon code 0. `build_vocabs` reserves index 0 for exactly this. Keeping the
    # single wrapper also keeps every existing checkpoint loadable: `pad` is an attr, not a
    # parameter, so the serialised structure is unchanged.
    return with_array(chain(embed, encoder, Softmax(nO=n_labels, nI=width)),
                      pad=window_size * depth)


class LexPresegmenter:
    def __init__(self, chars, labels, lex, model=None, width=64, depth=6, min_len=3):
        # `lex` may be a single set or a LIST of sets, one per lexicon source
        self.lexes = list(lex) if isinstance(lex, (list, tuple)) else [lex]
        self.min_lens = min_len if isinstance(min_len, (list, tuple)) else \
            [min_len] * len(self.lexes)
        self.chars, self.labels = list(chars), list(labels)
        self.lex, self.min_len = self.lexes[0], self.min_lens[0]
        self.char_id = {c: i for i, c in enumerate(self.chars)}
        self.width, self.depth = width, depth
        self._cache = {}
        self.n_values = [N_FREQ if (i == 0 and _FREQ.get("counts") is not None) else N_LEX
                         for i in range(len(self.lexes))]
        self.model = model if model is not None else build_lex_model(
            len(self.chars), len(self.labels), width, depth,
            n_sources=len(self.lexes), n_values=self.n_values)

    def encode(self, text, codes=None):
        unk = self.char_id[UNK_CHAR]
        explicit = codes is not None      # jackknifed codes are per-ROW, not per-text: the cache
        if codes is None:                 # is keyed by text and would leak one fold's into another
            codes = self._cache.get(text)
        if codes is None:
            codes = multi_codes(text, self.lexes, self.min_lens)
            if not explicit and len(self._cache) < 400_000:
                self._cache[text] = codes
        return self.model.ops.asarray2i(
            [[self.char_id.get(c, unk)] + [ch[i] for ch in codes]
             for i, c in enumerate(text)], dtype="i")

    def predict(self, texts):
        ne = [t for t in texts if t]
        if not ne:
            return [[] for _ in texts]
        scores = self.model.predict([self.encode(t) for t in ne])
        it = iter(scores); out = []
        for t in texts:
            if not t:
                out.append([]); continue
            s = next(it)
            out.append([self.labels[int(i)] for i in s.argmax(axis=1)])
        return out

    def to_disk(self, path):
        p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True)
        (p / "model.bin").write_bytes(self.model.to_bytes())
        (p / "vocab.json").write_text(json.dumps(
            {"chars": self.chars, "labels": self.labels, "width": self.width,
             "depth": self.depth, "min_len": self.min_lens,
             # n_sources decides how many Embed tables the architecture has; without it
             # from_disk rebuilds a 1-source model and thinc raises "mismatched structure"
             "n_sources": len(self.lexes), "n_values": self.n_values,
             # which source (if any) is jieba's segmentation decision rather than a word list. A
             # model trained with this channel and loaded without it runs with one input deleted and
             # nothing raises, so the marker travels with the weights.
             "jieba_source": _JIEBA["index"],
             # and whether that channel was asked about the SIMPLIFIED rendering. A traditional
             # segmenter trained with it and loaded without it is the reads_spaces trap again:
             # a quietly different input regime, nothing raising.
             "jieba_t2s": _JIEBA["t2s"]},
            ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_disk(cls, path, lex):
        p = pathlib.Path(path)
        v = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        n_src = v.get("n_sources", 1)
        m = build_lex_model(len(v["chars"]), len(v["labels"]), v["width"], v["depth"],
                            n_sources=n_src, n_values=v.get("n_values"))
        if not isinstance(lex, (list, tuple)):
            lex = [lex] * n_src
        obj = cls(v["chars"], v["labels"], lex, model=m, width=v["width"], depth=v["depth"],
                  min_len=v.get("min_len", 3))
        obj.model.initialize()
        obj.model.from_bytes((p / "model.bin").read_bytes())
        return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("dev"); ap.add_argument("out")
    ap.add_argument("--lexicon", nargs="+", default=["models/apte_stems.txt"],
                    help="one or more lexicon files; EACH gets its own embedding channel so the "
                         "model can weight (or ignore) them independently")
    ap.add_argument("--min-lens", nargs="+", type=int, default=None,
                    help="minimum entry length per lexicon (default 3 for all; use 1 for Chinese, "
                         "where single-character words are real and frequent)")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freq-mode", default="count", choices=("count", "rank"),
                    help="band raw counts, or frequency RANK (stable under jackknifing)")
    ap.add_argument("--freq-counts", action="store_true",
                    help="source 0 carries BANDED FREQUENCY (25 values) rather than binary "
                         "membership. Counts come from the corpus; with --jackknife they are "
                         "rebuilt per fold, so train-time frequencies are as sparse as test-time "
                         "ones (the leak that cost 1.37 F when the zh lexicon was naive).")
    ap.add_argument("--inflect-endings", default=None,
                    help="path to models/sa_endings.json; makes source 0 use the sandhi/inflection"
                         "-aware extractor (suffix-strip against the stem list) rather than plain "
                         "membership")
    ap.add_argument("--jieba-source", type=int, default=None, metavar="K",
                    help="route lexicon source K through jieba's SEGMENTATION DECISION (BMES, 4 "
                         "values) rather than membership in the file passed at that position. The "
                         "file still has to be given — pass an empty one — because the source list "
                         "is what sizes the architecture. jieba is EXTERNAL, so this channel needs "
                         "no jackknifing: its dictionary was not harvested from our training split, "
                         "and train-time reliability therefore already equals test-time.")
    ap.add_argument("--jieba-t2s", action="store_true",
                    help="ask jieba about the t2s (simplified) rendering of each chunk and keep "
                         "the per-character answer for the original text. For a TRADITIONAL "
                         "segmenter: jieba's dictionary is simplified, and this is worth "
                         "F 0.8920 -> 0.9223 on its boundary decisions.")
    ap.add_argument("--jieba-userdict", default=None,
                    help="word list to force-split (`del_word`) before segmenting. Harvest it with "
                         "zh_jieba_feature.force_split_dict — but note it is derived from GOLD, so "
                         "a train-harvested one leaks unless it is jackknifed like the corpus "
                         "lexicon.")
    ap.add_argument("--jackknife", type=int, default=0, metavar="K",
                    help="build the lexicon from the TRAINING DATA by K-fold jackknifing: each "
                         "training sentence sees a lexicon derived from the OTHER folds only. A "
                         "lexicon harvested from the whole training set fires on 100 %% of training "
                         "words but only 87.6 %% of test words, so the model learns a reliability "
                         "the feature will not have at inference and never develops a fallback. "
                         "Jackknifing makes train-time coverage match test-time coverage. Dev/test "
                         "always use the FULL lexicon, which is what deployment sees.")
    ap.add_argument("--no-lex", action="store_true",
                    help="CAPACITY CONTROL: identical architecture and parameter count, but every "
                         "lexicon code forced to 0, so the extra Embed table carries no "
                         "information. Any gain the real arm shows over this one is the FEATURE, "
                         "not the parameters — the same discipline as the w96 control that showed "
                         "the sa affix gain was information rather than capacity.")
    a = ap.parse_args()

    if a.jieba_source is not None and not a.no_lex:
        enable_jieba(a.jieba_source, a.jieba_userdict, t2s=a.jieba_t2s)
        print(f"  source {a.jieba_source} = jieba segmentation decision (BMES)"
              + (" via t2s" if a.jieba_t2s else "")
              + (f", force-split userdict {a.jieba_userdict}" if a.jieba_userdict else ""))
    if a.inflect_endings and not a.no_lex:
        enable_inflect(a.lexicon[0], a.inflect_endings)
        print(f"  source 0 = sandhi/inflection-aware ({a.inflect_endings})")
    mins = a.min_lens or [3] * len(a.lexicon)
    lex = []
    for path, ml in zip(a.lexicon, mins):
        entries = {w for w in pathlib.Path(path).read_text(encoding="utf-8").split("\n")
                   if len(w) >= ml}
        lex.append(set() if a.no_lex else entries)
        print(f"  lexicon {pathlib.Path(path).name}: {len(entries)} entries (len>={ml})"
              + ("  [DISABLED: capacity control]" if a.no_lex else ""))
    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    dev = [json.loads(l) for l in open(a.dev, encoding="utf-8")]
    if a.limit:
        rows, dev = rows[:a.limit], dev[:max(200, a.limit // 10)]
    jk_codes = None
    if a.jackknife and not a.no_lex:
        K = a.jackknife
        folds = [[] for _ in range(K)]
        for i, r in enumerate(rows):
            folds[i % K].append(i)
        # per-fold lexicon = word types from every OTHER fold
        per_fold = []
        for k in range(K):
            held = set()
            for j in range(K):
                if j == k:
                    continue
                for i in folds[j]:
                    held.update(rows[i]["csl"].split())
            per_fold.append(held)
        full = set()
        for r in rows:
            full.update(r["csl"].split())
        print(f"  jackknife K={K}: full lexicon {len(full)} types; "
              f"per-fold {min(len(x) for x in per_fold)}-{max(len(x) for x in per_fold)}")
        # coverage the model will actually see during training, vs the naive version
        import itertools
        seen = tot = naive = 0
        for k in range(K):
            for i in folds[k]:
                for w in rows[i]["csl"].split():
                    tot += 1; seen += w in per_fold[k]; naive += w in full
        print(f"    train-time coverage: jackknifed {seen/max(tot,1):.1%} "
              f"(naive would be {naive/max(tot,1):.1%})")
        jk_codes = {}
        jk_jieba = a.jieba_source is not None and a.jieba_userdict == "auto"
        if jk_jieba:
            # The force-split userdict is harvested from GOLD, so it must be jackknifed exactly like
            # the corpus lexicon — otherwise jieba arrives at training already corrected on the very
            # rows it is being scored against, and the model learns a reliability the channel will
            # not have at inference. This is the same leak that cost the naive corpus lexicon 1.37 F.
            import zh_jieba_feature as jf
            print(f"  jieba userdict jackknifed K={K} (harvested per fold from the other folds)")
        for k in range(K):
            if jk_jieba:
                other = [rows[i] for j in range(K) if j != k for i in folds[j]]
                _JIEBA["tok"] = jf.set_userdict(jf.force_split_dict(other))
            if a.freq_counts:
                import collections as _c
                cnt = _c.Counter()
                for j in range(K):
                    if j == k: continue
                    for i in folds[j]:
                        cnt.update(rows[i]["csl"].split())
                if a.freq_mode == "rank":
                    o=sorted(cnt,key=lambda w:-cnt[w]); N=len(o)
                    cnt={w: max(1,int(10**(4*(1-i/N)))) for i,w in enumerate(o)}
                lexes_k = [cnt] + lex[1:]          # consumed by freq_codes
            else:
                lexes_k = [per_fold[k]] + lex[1:]  # fold lexicon replaces the FIRST source
            for i in folds[k]:
                jk_codes[i] = multi_codes(rows[i]["samhita"], lexes_k, mins)
        lex = [full] + lex[1:]                     # dev/test see the full lexicon
        if jk_jieba:
            # dev/test/deployment see the userdict harvested from the WHOLE training split, which is
            # what a released model would ship; it is saved so evaluation can reproduce it exactly.
            words = jf.force_split_dict(rows)
            _JIEBA["tok"] = jf.set_userdict(words)
            out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
            (out / "jieba_force_split.txt").write_text("\n".join(sorted(words)), encoding="utf-8")
            print(f"    full-train force-split userdict: {len(words)} words -> {out.name}/"
                  f"jieba_force_split.txt")

    if a.freq_counts and not a.no_lex:
        import collections as _c
        counts = _c.Counter()
        for r in rows:
            counts.update(r["csl"].split())
        enable_freq(counts, a.freq_mode)
        print(f"  source 0 = banded frequency ({a.freq_mode}) over {len(counts)} word types")
    chars, labels = build_vocabs(rows)
    seg = LexPresegmenter(chars, labels, lex, width=a.width, depth=a.depth, min_len=mins)
    print(f"  {len(rows)} train / {len(dev)} dev, {len(chars)} chars, {len(labels)} labels")

    t0 = time.time()
    for i, r in enumerate(rows + dev):
        seg.encode(r["samhita"], jk_codes.get(i) if (jk_codes and i < len(rows)) else None)
        if i and i % 40000 == 0:
            print(f"    lexicon features {i}/{len(rows)+len(dev)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"    lexicon features cached in {time.time()-t0:.0f}s")

    # thinc allocates Embed parameters lazily; without this the first update raises
    # KeyError: Parameter 'E' ... has not been allocated yet
    seg.model.initialize(X=[seg.encode(r["samhita"]) for r in rows[:8] if r["samhita"]])

    from eval_samhita import score as eval_score   # training-only dependency

    idx = {lb: i for i, lb in enumerate(seg.labels)}
    optimizer = Adam(a.lr)
    rng = random.Random(a.seed)
    best, bad = -1.0, 0
    for ep in range(a.epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), a.batch_size):
            batch = [rows[i] for i in order[s:s + a.batch_size] if rows[i]["samhita"]]
            if not batch:
                continue
            _ = batch
            X = [seg.encode(rows[i]["samhita"],
                            jk_codes.get(i) if jk_codes else None)
                 for i in order[s:s + a.batch_size] if rows[i]["samhita"]]
            Y = [np.array([idx[lb] for lb in r["labels"]], dtype="i") for r in batch]
            guess, backprop = seg.model.begin_update(X)
            grads = []
            for g, y in zip(guess, Y):
                d = g.copy()
                d[np.arange(len(y)), y] -= 1.0
                d /= max(len(y), 1)
                grads.append(d)
                tot += float(-np.log(np.clip(g[np.arange(len(y)), y], 1e-9, None)).mean())
            backprop(grads)
            seg.model.finish_update(optimizer)
        preds = seg.predict([r["samhita"] for r in dev])
        sc = eval_score(dev, {r["sent_id"]: p for r, p in zip(dev, preds)})
        f = sc["split_location"][2] * 100          # (P, R, F) tuple, as eval_samhita returns
        print(f"    epoch {ep:3d}  loss {tot/max(len(rows),1):.4f}  dev split-loc F {f:.2f}",
              flush=True)
        if f > best:
            best, bad = f, 0
            seg.to_disk(a.out)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"    early stop"); break
    print(f"  best dev split-location F {best:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
