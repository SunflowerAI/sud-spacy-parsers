#!/usr/bin/env python3
"""jieba's SEGMENTATION DECISION as an input channel for the character segmenter.

Distinct from the jieba channel already tried and rejected. That one asked "is this substring in
jieba's 349 044-entry dictionary" — plain set membership, and it lost to the jackknifed corpus
lexicon (0.8859 vs 0.8902). This one asks jieba to actually SEGMENT the chunk and reads off its
verdict per character. That is the output of a DAG search over the dictionary plus an HMM over
unknown words, so it is a strictly stronger signal than a substring lookup, and it is the piece the
earlier experiment left on the table.

**Why it should help, measured before building it** (SUD_Chinese-GSD test, against the shipped
`models/zh_seg_jk`):

    jieba boundaries     P 0.9730   R 0.8793      -- very precise, and INCOMPLETE: it under-splits
    our model            P 0.9507   R 0.9556

    per character position:  both right 86.89 %   ours only 7.33 %
                             JIEBA ONLY RIGHT 4.19 %   both wrong 1.59 %

So jieba is right at **72.5 %** of the 1079 positions the shipped model gets wrong. That is the
headroom, and it is reachable only by a feature: 97 % of this segmenter's errors are CONFIDENT
errors (margin >= 0.10), which no decoder — beam, lexicon repair, transition model — can touch.
A crude hard union of the two break sets already scores 0.8938 against 0.8902 without any training.

**The code is BMES**, the standard Chinese-segmentation tagset, 4 values so the channel is exactly
the same width as a lexicon channel and the capacity control stays parameter-identical:

    0 B  first character of a jieba word     2 E  last character  (=> jieba breaks after this one)
    1 M  interior character                  3 S  single-character jieba word (also a break after)

Encoding the whole tag rather than just "break after here" costs nothing and tells the model where a
jieba word STARTS as well as where it ends — the two facts the model needs at a junction.

**jieba is EXTERNAL, so this channel needs no jackknifing.** Its dictionary was not harvested from
our training split, so train-time reliability equals test-time reliability, and the leak that made
the corpus lexicon useless without jackknifing cannot arise. A userdict harvested from our own train
split WOULD reintroduce it — see `force_split_dict`.
"""
import pathlib
import sys

B, M, E, S = 0, 1, 2, 3
N_JIEBA = 4

# The name the traditional dictionary travels under, beside the segmenter's own weights. It has to
# travel: a channel trained against a traditional dictionary and loaded against jieba's simplified
# one is the `reads_spaces` trap again -- a quietly different input regime, with nothing raising.
TRAD_DICT_FILE = "jieba_dict.txt"

_STATE = {"tok": None, "dict": None, "hmm_t2s": False}
_FINALSEG_CUT = None



def _import_jieba():
    """Import jieba, falling back to the copy vendored inside the wheel.

    The zh wheel no longer declares `jieba>=0.42.1`: it ships the ~6 MB the segmenter actually
    loads under `<model>/vendor/jieba`, because the pip distribution is 42 MB of which the rest is
    POS tagging, a neural model and keyword extraction that this channel never touches. That
    matters for a serverless target with a 250 MB unzipped budget.

    An installed jieba WINS if there is one, so a user who has their own copy keeps it and nothing
    about training changes; the vendored tree is only reached when the import fails. The two are
    interchangeable — the pruned tree is byte-identical files, verified to give the same BMES codes
    and the same full-parse digest on the shipped wheel.

    The vendor directory is found by walking up from this module: when bundled by `spacy package`
    this file sits at `<pkg>/zh_jieba_feature.py` and the model data dir `<pkg>/<name>-<version>/`
    holds `vendor/`, so the version string never has to be known. In a source checkout there is no
    vendor tree and the plain import is the only path.
    """
    try:
        import jieba
        return jieba
    except ImportError:
        pass
    # Search ONLY this module's own directory, never its parents. An earlier version walked up two
    # levels and globbed `*/vendor`, which bound an unrelated `vendor/jieba` belonging to a
    # different tree entirely -- a silent cross-package import that happened to work because the
    # contents matched. In the wheel this file is at `<pkg>/zh_jieba_feature.py` and the model data
    # dir is `<pkg>/<name>-<version>/`, so one glob level is exactly right and anything wider is a
    # bug waiting for a machine where the neighbour is not a pruned jieba.
    here = pathlib.Path(__file__).resolve().parent
    for cand in [here / "vendor", *sorted(here.glob("*/vendor"))]:
        if (cand / "jieba" / "__init__.py").is_file():
            sys.path.insert(0, str(cand))
            import jieba
            return jieba
    raise ImportError(
        "jieba is required for the zh segmenter's BMES channel and no vendored copy was found. "
        "Install it with `pip install jieba>=0.42.1`.")

def _hmm_via_t2s(finalseg):
    """jieba's OOV HMM, asked about the SIMPLIFIED rendering of each unknown run.

    A traditional dictionary fixes the LOOKUP but not this: `finalseg`'s emission probabilities are
    per CHARACTER and were estimated on simplified text, so on traditional input every unknown run
    is scored against characters the model never met. Measured on the traditional GSD test, that
    one component is the ENTIRE remaining gap — jieba's boundary decisions score F 0.9203 with the
    traditional dictionary and the raw text, and **F 0.9237** with the same dictionary and this
    wrapper, against 0.9236 for converting the whole chunk (`_jieba_via_t2s`).

    The HMM segments; the characters handed back are the ORIGINAL ones, sliced by the lengths it
    returned. That is sound only while the conversion preserves length, which is checked per run
    rather than assumed — where it does not, the run goes through untouched and the channel
    degrades to its raw-traditional quality for that run alone.
    """
    global _FINALSEG_CUT
    if _FINALSEG_CUT is None:
        _FINALSEG_CUT = finalseg.cut
    base = _FINALSEG_CUT
    import opencc
    conv = opencc.OpenCC("t2s")

    def cut(sentence):
        simp = conv.convert(sentence)
        if len(simp) != len(sentence):
            yield from base(sentence)
            return
        i = 0
        for w in base(simp):
            yield sentence[i:i + len(w)]
            i += len(w)

    return cut


def set_dictionary(path=None, hmm_t2s=False, words=()):
    """Reset jieba onto dictionary `path` (None = its own), then force-split every word in `words`.

    `path` is how the TRADITIONAL channel is selected: `build_jieba_trad_dict.py` writes the s2tw
    conversion of jieba's own dictionary, and pointing jieba at it makes the lookup read the
    traditional text itself instead of a `t2s` rendering of it that collapses 乾/幹/干 into one
    string. `hmm_t2s` keeps the OOV HMM on simplified, which is where its probabilities came from.

    Resetting FIRST matters for the same reason it does in `set_userdict`: jieba's `dt` is a global,
    and a dictionary or a force-split left over from a previous call would silently join this one.
    """
    jieba = _import_jieba()
    import jieba.finalseg
    jieba.dt.FREQ.clear()
    jieba.dt.initialized = False
    jieba.finalseg.Force_Split_Words.clear()
    if path:
        jieba.dt.set_dictionary(str(path))
    else:
        jieba.dt.__init__()                       # back to jieba's own dict.txt
    # Patch the module attribute, not a local: jieba's `__cut_DAG` reads `finalseg.cut` at CALL
    # time (`from . import finalseg`), so this reaches the tokenizer without touching jieba's code.
    jieba.finalseg.cut = _hmm_via_t2s(jieba.finalseg) if hmm_t2s else (
        _FINALSEG_CUT or jieba.finalseg.cut)
    jieba.initialize()
    for w in words:
        jieba.del_word(w)
    _STATE["tok"] = jieba.dt
    _STATE["dict"] = str(path) if path else None
    _STATE["hmm_t2s"] = bool(hmm_t2s)
    return jieba.dt


def set_userdict(words=()):
    """Reset jieba to its stock dictionary, then force-split every word in `words`.

    `del_word(w)` sets w's frequency to 0, which registers a hard force-split in `finalseg`; that is
    the lever that matters here, because jieba's error against GSD is under-splitting (`这个`,
    `有人`, `为什么` are single jieba words and two GSD tokens each). `add_word` is the wrong
    direction — measured, it costs 0.8 F.

    Resetting first is what makes the per-fold jackknifing below honest: without it each fold's
    force-splits would accumulate on top of the previous fold's, so fold 4 would be segmenting with
    a dictionary derived from all five folds including its own.
    """
    return set_dictionary(_STATE["dict"], _STATE["hmm_t2s"], words)


def get_tokenizer(userdict=None, dictionary=None, hmm_t2s=None):
    """jieba's global Tokenizer, on the requested dictionary regime and userdict.

    The cached tokenizer is returned only when it is already on the regime asked for. Asking for a
    dictionary jieba is not currently loaded with REBUILDS it rather than quietly handing back the
    other one — a loaded segmenter must not be able to run against a dictionary it was not trained
    against, which is the whole reason `jieba_dict` travels in `vocab.json`.
    """
    want_dict = _STATE["dict"] if dictionary is None else (str(dictionary) if dictionary else None)
    want_hmm = _STATE["hmm_t2s"] if hmm_t2s is None else bool(hmm_t2s)
    if (_STATE["tok"] is not None and not userdict
            and want_dict == _STATE["dict"] and want_hmm == _STATE["hmm_t2s"]):
        return _STATE["tok"]
    words = ()
    if userdict and userdict != "auto":
        words = [w for w in pathlib.Path(userdict).read_text(encoding="utf-8").split("\n") if w]
    return set_dictionary(want_dict, want_hmm, words)


def jieba_codes(text, tok=None):
    """Per character of `text`, its BMES tag under jieba's segmentation."""
    tok = tok or get_tokenizer()
    out = []
    for w in tok.cut(text, HMM=True):
        n = len(w)
        if n == 1:
            out.append(S)
        else:
            out.extend([B] + [M] * (n - 2) + [E])
    if len(out) != len(text):                 # jieba never drops characters, but never guess
        out = (out + [S] * len(text))[:len(text)]
    return out


def force_split_dict(train_rows, min_wrong=2):
    """Harvest the jieba words a treebank USUALLY splits, from `csl`/`samhita` training rows.

    Derived from gold, so a channel built on it is only honest under the same K-fold jackknifing the
    corpus lexicon needs. Measured standalone on GSD (harvested on train, scored on test), it lifts
    jieba itself from token F 0.7989 to 0.8570 and halves the positions where jieba is wrong and the
    model is right (1369 -> 958) while keeping every position where jieba rescues the model.
    """
    from collections import Counter
    tok = set_userdict()                      # harvest against UNPRIMED jieba, same dictionary
    hit, miss = Counter(), Counter()
    for r in train_rows:
        gold, i = set(), 0
        for w in r["csl"].split(" "):
            gold.add((i, i + len(w)))
            i += len(w)
        i = 0
        for w in tok.cut(r["samhita"], HMM=True):
            (hit if (i, i + len(w)) in gold else miss)[w] += 1
            i += len(w)
    return [w for w in set(hit) | set(miss)
            if len(w) > 1 and miss[w] >= min_wrong and miss[w] > hit[w]]
