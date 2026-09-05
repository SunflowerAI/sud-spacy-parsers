#!/usr/bin/env python
"""Arc-factored (biaffine + Chu-Liu/Edmonds) parser, over a FROZEN or JOINTLY-TRAINED encoder.

GENERALISED from `train_lzh_arcfactored.py`, which built and measured this architecture for lzh
first (see that file's history for the ablations this one inherits rather than re-runs: frozen
plateaus because the transition parser's encoder was fitted to a different objective; a fresh joint
encoder needs real dropout/LR-decay/batching or the comparison measures the TRAINER, not the
architecture; signed distance buckets are a trade, not a free win). This file keeps the same
`Biaffine` class, loss functions and training loop verbatim and parameterises what differs per
language: the source model, the corpus, the window -- AND, the point this file was rebuilt around,
EXACTLY which upstream annotation the parser reads before it ever sees a token vector.

⚠ THE ARC-FACTORED MODEL MUST READ THE SAME INPUT AS THE TRANSITION MODEL, OR THE COMPARISON IS NOT
HONEST. la's and sa's released parsers are not lzh's: neither is `tok2vec -> parser` alone.
  * sa (`training_sa_mp2_sub_s1`, pipeline `tok2vec, tagger, morphologizer, lemmatizer, parser`):
    the parser has its OWN DEDICATED tok2vec submodel (freeze recipe), separate from the shared
    listener that feeds tagger/morphologizer/lemmatizer, and its embed is `sud.AnalyserFeatsEmbed.v1`
    reading `attrs=[NORM,PREFIX,SUFFIX,SHAPE,MORPH]` -- i.e. it needs predicted MORPH, which only
    EXISTS after the morphologizer has already run. Extracting `nlp.get_pipe("tok2vec")` alone (what
    the first version of this file did) is a DIFFERENT, morph-blind computation that the parser
    never actually uses.
  * la (`training_la_lemvec_sud`, pipeline `morphologizer, lemmatizer, tok2vec, parser, tagger, ...`):
    the parser's tok2vec IS a listener onto the shared `tok2vec` pipe, but that pipe's embed is
    `sud.LemmaVecFeatsEmbed.v1`, reading a DISTRIBUTIONAL LEMMA VECTOR (sealed into the model, see
    `sud_lemmavec_embed.py`) plus twelve categorical FEATS (Case, Number, Gender, VerbForm, Mood,
    Tense, Voice, Person, PronType, Degree, InflClass, Aspect) -- both of which need the
    MORPHOLOGIZER and LEMMATIZER to have already run, which is exactly why they sit BEFORE tok2vec
    in this arm's pipeline. Extracting `nlp.get_pipe("tok2vec")` from a doc with no lemma/morph set
    silently reads the "no entry" zero-vector path on every token.
`encoder_and_upstream()` below resolves both cases generically, from the loaded pipeline itself
(never hand-listed): the true input to score with is whichever tok2vec submodel the PARSER's own
`.model.get_ref("tok2vec")` names -- the top-level `tok2vec` pipe if that ref is a
`Tok2VecListener` (la), or the parser's own dedicated copy if it is not (sa) -- and the pipes that
must run FIRST are exactly `nlp.pipe_names[:nlp.pipe_names.index("parser")]`, in order, whatever
they are. This also means la's TAGGER (which runs AFTER parser in its pipeline) is correctly never
fed to this decoder: the real parser at inference time does not have it either.

⚠ JOINT MODE CANNOT REPRODUCE THE SEALED CHANNELS EXACTLY, AND SAYS SO. A fresh jointly-trained
encoder cannot start from `sud.LemmaVecFeatsEmbed.v1`'s sealed distributional table (that table is
fitted, not random-initialised) without unsealing it and duplicating a fair amount of bespoke
architecture code for a differently-initialised run that would immediately diverge from it anyway.
Instead `--joint` embeds MORPH (sa) and LEMMA+MORPH (la) as ordinary hashed attributes via
`spacy.MultiHashEmbed.v2`, which both pipes still need annotated by the SAME frozen upstream chain
first -- so the joint encoder sees the same KIND of signal (predicted lemma identity, predicted
morphological features) but through a coarser hash-embedding channel than the trained arm's
distributional vectors / per-feature categorical embeddings. This is the honest approximation, not
the exact one; frozen mode is the one that reads the literal deployed computation.

WHY LATIN AND SANSKRIT, SPECIFICALLY.
  * Latin's non-projective headroom has survived three interventions on the TRANSITION side (more
    actions, upsampling, beam search) and `NEGATIVE-RESULTS.md` indicts the pseudo-projective
    REPRESENTATION itself: round-tripping gold Latin through projectivize/deprojectivize fails to
    return 395/54 897 heads, and 78 of 200 decorated label types occur exactly once.
  * Sanskrit was not previously measured for this. Measured here on `corpus_sa_mwt_rl2` train:
    9.49 % of non-root arcs are involved in a crossing (comparable to Latin's 11.22 %), and a window
    of k=50 covers 99.99 % of all arcs and 99.99 % of CROSSING arcs (max crossing distance 75, over
    docs averaging 75 tokens) -- so the window measured for Latin turns out to cover Sanskrit
    equally well, and both share k=50 rather than each inventing its own number.

⚠ WINDOWED at k=50 for both languages (see measurement above). This standalone trainer hand-rolls
its own windowed scorer/decoder loop rather than using `sud.BiaffineArcScorer.v1`/`sud_cle.mst`
directly (`Biaffine` needs raw gradients for --joint, which a thinc `Model` abstracts away), so the
window constant here must be kept in step with them by hand.

⚠ NODE 0 IS A VIRTUAL ROOT and every token may attach to it, so CLE returns a FOREST and sentence
boundaries survive -- both la and sa corpora here are `convert -n 10` multi-sentence docs (la mean
145.5 tokens/doc, sa mean 75.4).
"""
import argparse, json, pathlib, sys, time, zlib
import numpy as np

# ⚠ RELATIVE TO THIS FILE, NOT TO CWD. A bare `sys.path.insert(0, "scripts")` "worked" for this
# file's whole life as a CLI research script only because CWD is always the repo root there -- but
# the packaged sa arc-factored wheel ships this SAME file as a bundled submodule, and if a test (or
# a user) happens to run with CWD containing its own "scripts" directory, the relative form finds
# THAT instead, re-discovering the dev copies of every sud_* file and colliding with the ones the
# wheel already registered (E004: "a factory ... already exists"). Matches the absolute,
# file-relative form `sud_arcfactored_parser.py`/`sud_cle` already use.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    # Registers every custom architecture/tokenizer/factory this project's OWN configs might name
    # -- needed when this file runs as the CLI research script (`python scripts/train_arcfactored.py
    # ...`), never needed by the pure helper functions below (dist_buckets, agreement_buckets,
    # build_joint_embed_from_meta, ...) that `sud_arcfactored_parser.py` imports at INFERENCE time
    # from inside a packaged wheel, which shíps this file but not the (huge, every-language)
    # seg_code.py alongside it. Optional, not silently-wrong: nothing in this module references a
    # name seg_code would have registered, so a missing import here cannot mask a real E893 later
    # -- that still surfaces the moment something actually calls spacy.load() on an arm needing it.
    import seg_code  # noqa: F401
except ImportError:
    pass
import spacy
from spacy.tokens import DocBin, Doc
from thinc.api import Adam, NumpyOps
from sud_cle import mst
from sud_joint_biaffine import JointBiaffine

NEG = -1e4

# ⚠ SIGNED DISTANCE BUCKETS. Inherited from lzh's measurement (a CNN of window 1 depth 4 has
# receptive field +-4, so a per-bucket bias supplies a prior that helps the common case and cannot
# discriminate the rare one; see --no-dist below). Edges are geometric, not fitted per corpus.
_EDGES = (1, 2, 3, 4, 5, 7, 11, 19, 31)


def dist_buckets(n, window):
    """(n+1, n) int bucket ids over [virtual root | tokens] x dependents."""
    h = np.arange(n + 1)[:, None] - 1          # row 0 is the virtual root
    d = np.arange(n)[None, :]
    delta = h - d
    mag = np.abs(delta)
    b = np.zeros_like(delta)
    for i, e in enumerate(_EDGES):
        b = np.where(mag >= e, i + 1, b)
    b = b * 2 - (delta < 0).astype(int)        # separate the two directions
    b = np.maximum(b, 0) + 1                   # shift, leaving 0 free for the root row
    b[0, :] = 0                                # every arc FROM the virtual root shares one bucket
    return b


N_DIST_BINS = 2 * len(_EDGES) + 2

# ⚠ DIRECTION BIAS (--joint-label --direction). `dist_by_label` already splits every magnitude
# bucket by direction (dist_buckets' `b*2 - (delta<0)`), so a label's overall "head usually
# precedes/follows" preference is already representable there -- but split thin across many
# (direction, magnitude) cells, each seeing few training examples for a rare label. Pooling every
# magnitude into ONE bit per label gives that preference a single, well-estimated cell instead.
# Needs nothing from the document (token POSITIONS only), so -- unlike agreement -- it is computed
# exactly like `dist`: a buckets_fn called inside JointBiaffine.forward, never threaded in by hand.
N_DIR_BINS = 3   # 0 = root/self (no direction), 1 = head precedes dependent, 2 = head follows it


def direction_buckets(n, window):
    h = np.arange(n + 1)[:, None] - 1          # row 0 = virtual root -> -1
    d = np.arange(n)[None, :]
    delta = h - d
    b = np.where(delta < 0, 1, np.where(delta > 0, 2, 0))
    b[0, :] = 0                                # the virtual root row: no direction, same as self-loop
    return b


# ⚠ AGREEMENT BIAS (--joint-label --agreement). diagnose_la_deprel_errors.py found `mod`'s dominant
# error under joint-label scoring is picking a WRONG candidate at the SAME distance as the right one
# (err-delta ~0) -- a distance bias cannot discriminate that, by construction. Latin (and Sanskrit,
# whose --joint-label result never showed this regression) resolve exactly this ambiguity through
# morphological CONCORD: an adjective agrees with its head noun in Case, Number and Gender. This
# gives the arc scorer that signal directly, mirroring how the distance bias works but answering a
# different question ("do these two forms agree" vs "how far apart are they").
AGREE_FEATS = ("Case", "Number", "Gender")
N_AGREE_BINS = 2 + len(AGREE_FEATS)   # 0 = N/A (either side lacks a full set); 1..len+1 = match count + 1


def agreement_buckets(doc):
    """(n+1, n) int bucket ids over [virtual root | tokens] x dependents: how many of AGREE_FEATS
    match between candidate head and dependent, or 0 if either lacks the full set (the virtual
    root always does, so its whole row is 0 -- the same "no information" code a real token with
    missing FEATS gets, which is the right answer: nothing here should look like a rich agreement
    match just because it involves the root).

    ⚠ EXACT TUPLE MATCH, NOT SET OVERLAP. `token.morph.get(f)` returns a list (a feature CAN be
    multi-valued, e.g. syncretic `Case=Nom,Acc`), and comparing as tuples requires the SAME set of
    values on both sides -- a token specified as `Nom,Acc` and one specified as plain `Nom` count as
    a MISMATCH here even though a human reader would call them compatible. A known simplification,
    not a bug: getting this exactly right needs treating each feature as a set of possible readings
    and checking non-empty intersection, which is a larger change than this first pass warrants.
    """
    n = len(doc)
    case = [tuple(t.morph.get("Case")) for t in doc]
    num = [tuple(t.morph.get("Number")) for t in doc]
    gen = [tuple(t.morph.get("Gender")) for t in doc]
    has = [bool(case[i]) and bool(num[i]) and bool(gen[i]) for i in range(n)]
    b = np.zeros((n + 1, n), dtype=np.int64)          # row 0 (virtual root) stays all-0: N/A
    for d in range(n):
        if not has[d]:
            continue
        for hi in range(n):
            if hi == d or not has[hi]:
                continue
            m = int(case[hi] == case[d]) + int(num[hi] == num[d]) + int(gen[hi] == gen[d])
            b[hi + 1, d] = 1 + m
    return b


# ⚠ POS-COMPATIBILITY BIAS (--joint-label --pos). Checking actual la gold counts: `subj` heads are
# VERB/AUX 94 % of the time and `cc` dependents are CCONJ 99 % of the time -- both strongly
# POS-selected relations that neither `dist` nor `agree` touches (agreement only fires when BOTH
# sides carry a full Case/Number/Gender set, which most subj/cc candidates don't jointly have
# anyway). Threaded exactly like agreement (needs the doc's predicted UPOS, not just token
# positions), never like direction. A JOINT (head-UPOS, dep-UPOS) bucket, not two separate marginal
# tables, because the label preference genuinely depends on the PAIR: "VERB head + NOUN dep" is
# compatible with subj, comp:obj AND mod at once, so only the per-label table (not the marginals
# alone) can tell them apart.
UPOS_LIST = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
             "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
UPOS_INDEX = {p: i for i, p in enumerate(UPOS_LIST)}
N_UPOS = len(UPOS_LIST) + 1                        # +1: fallback bucket for "" / an unseen tag
N_POS_BINS = 1 + N_UPOS * N_UPOS                    # 0 = ROOT-as-head; else 1 + head_id*N_UPOS + dep_id

# ⚠ lzh's `.pos_` IS EMPTY PRE-PARSE, ALWAYS -- its `morphologizer` (which sets UPOS) sits AFTER
# `parser` in the deployed pipeline (`encoder_and_upstream()`'s upstream for lzh is just
# `['tok2vec', 'tagger']`, and `tagger` here is plain `spacy.Tagger.v2`, which only ever sets TAG,
# never POS). Reading `t.pos_` directly would collapse `pos_buckets` to ONE constant non-root
# bucket for every lzh doc, at both train and eval time alike -- not train/inference skew, just a
# completely uninformative feature that happens to run without error. lzh's tag_ IS available
# pre-parse though, and its first comma-separated field is a real, if coarse (only 4-way: n/p/s/v),
# category -- mapped onto the SAME UPOS_LIST so pos_buckets needs no separate code path.
#
# ⚠ PRONOUNS ARE A SUBTYPE OF 'n' (代名詞, "pronoun"), NOT their own first-field value -- checking
# the tagger's actual label set, all 5 pronoun tags are `n,代名詞,...`. Mapping the first field
# alone would fold every lzh pronoun into NOUN, which is silently fatal for `preverbal_buckets`
# (built PURELY to distinguish pronoun objects from full-NP objects): before this check caught it,
# `is_pron` would have been 0 for every lzh token, ALWAYS, making that whole bias a no-op that runs
# without error -- the exact same class of bug this module already hit once with `.pos_` itself.
_LZH_TAG_COARSE = {"v": "VERB", "p": "PART", "s": "PUNCT"}   # "n" handled specially, below


def upos_like(t):
    """`t.pos_` where it is genuinely predicted pre-parse (la, sa); otherwise (lzh) a coarse proxy
    from the (first, second) fields of the composite XPOS `tag_` string -- see the module note
    above for why the second field matters specifically for `n,代名詞,...` -> PRON."""
    if t.pos_:
        return t.pos_
    parts = t.tag_.split(",")
    first = parts[0] if parts else ""
    if first == "n" and len(parts) > 1 and parts[1] == "代名詞":
        return "PRON"
    return _LZH_TAG_COARSE.get(first, "NOUN" if first == "n" else "")


def pos_buckets(doc):
    """(n+1, n) int bucket ids over [virtual root | tokens] x dependents: a joint (head UPOS,
    dependent UPOS) bucket, or 0 whenever the candidate head is the virtual root (no UPOS there --
    the same "distinct, uninformative" treatment agreement_buckets gives the root row)."""
    n = len(doc)
    ids = np.array([UPOS_INDEX.get(upos_like(t), N_UPOS - 1) for t in doc], dtype=np.int64)
    b = np.zeros((n + 1, n), dtype=np.int64)
    b[1:, :] = 1 + ids[:, None] * N_UPOS + ids[None, :]
    return b


# ⚠ LEMMA-VECTOR BIAS (--joint-label --lemvec). Checking actual lzh gold counts: VERB-headed
# `comp:obj` is dominated by a small, closed set of matrix verbs (曰 "say" alone is 7.5% of 59722
# tokens; the top 10 -- 曰謂使有無爲如以得知, exactly the classic speech/causative/existential/
# copular/modal/epistemic complement-taking verbs -- cover 28.3%), which a coarse UPOS bias cannot
# single out (VERB-headed is shared by mod, subj, comp:obl and conj:coord alike -- see pos_buckets'
# own docstring for why THAT one still helps elsewhere). Reuses the SAME distributional table the
# project already ships/trains with per language (la: the PPMI lemma-vector table sealed into the
# released parser; lzh: the SikuBERT/kanripo table `build_lzh_sikubert_vectors.py` built for the
# tagger) rather than inventing a new one -- see LANGS[lang]['lemvec_table'].
_LEMVEC_TABLES = {}


def load_lemvec_table(path):
    """(lemma -> row index, (n_lemmas, dim) float32 matrix), cached by path."""
    if path not in _LEMVEC_TABLES:
        d = np.load(path, allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        _LEMVEC_TABLES[path] = ({k: i for i, k in enumerate(keys)}, d["vectors"].astype("float32"))
    return _LEMVEC_TABLES[path]


def lemma_vecs(doc, path):
    """(n+1, dim) float32: row 0 (virtual root) and any lemma absent from the table are the ZERO
    vector -- the same "no entry" capacity-control convention `sud_lemmavec_embed.py` uses, so an
    unseen lemma contributes nothing rather than a stale/misleading direction.

    ⚠ `t.lemma_ or t.text`, NOT `t.lemma_` alone: lzh's source model has NO LEMMATIZER PIPE AT ALL
    (unlike la/sa), so `t.lemma_` is empty pre-parse for every token, always -- the exact same
    silent-degenerate-feature trap `pos_buckets` hit (see `upos_like`'s note), just with a
    different, simpler fix here: Classical Chinese barely inflects, so FORM already IS the lemma
    for nearly every token, and the SikuBERT table (`build_lzh_sikubert_vectors.py`) is itself
    keyed by literal orthographic form, not a normalised lemma -- so falling back to `.text` is not
    an approximation for lzh, it is the correct key."""
    idx, V = load_lemvec_table(path)
    dim = V.shape[1]
    out = np.zeros((len(doc) + 1, dim), dtype="float32")
    for i, t in enumerate(doc):
        j = idx.get(t.lemma_ or t.text)
        if j is not None:
            out[i + 1] = V[j]
    return out


# ⚠ DEPENDENT-SIDE LEMMA VECTOR (--joint-label --lemvec-dep). The symmetric counterpart of
# `lemma_vecs` above -- (n, dim), NOT (n+1, dim), since a dependent is never the virtual root, so
# there is no root row to reserve. Built for la's mod/comp:obl trade surviving every purely
# STRUCTURAL bias (agreement/direction/pos/feat all reshuffle which of the two wins, none resolves
# it): `lemvec` (head-side) answers "does this governing word select the relation" (a valency
# question); this answers "does this DEPENDENT word, by its own distributional profile, look
# oblique-like or modifier-like" -- e.g. nouns that habitually appear as instrumental/locative
# arguments vs. ones that habitually appear as attributive modifiers may cluster differently in the
# same static vector space the head-side term already reads, EVEN THOUGH `lemvec`'s own neighbourhoods
# (`build_lemma_vectors.py --report`) turned out coarse/collocational for single characters -- worth
# testing on its own evidence, not assumed to work by analogy to the head-side result.
def lemma_vecs_dep(doc, path):
    idx, V = load_lemvec_table(path)
    dim = V.shape[1]
    out = np.zeros((len(doc), dim), dtype="float32")
    for i, t in enumerate(doc):
        j = idx.get(t.lemma_ or t.text)
        if j is not None:
            out[i] = V[j]
    return out


# ⚠ MORPH-HASH BIAS (--joint-label --morphhash). The DEPENDENT-side half of the mod/comp:obl fix
# lemvec is the head-side half of: diagnose_la_deprel_errors.py shows the two labels trading errors
# under every purely STRUCTURAL bias tried so far (agreement/direction/pos) -- mod's dominant
# label-only error is comp:obl and vice versa, because both attach heavily to VERB heads and neither
# distance, direction nor coarse POS can tell them apart. What DOES: Latin obliques are case-marked
# by the governing verb's VALENCY (comp:obl), while an attributive modifier's case is whatever its
# head noun's case happens to be (mod, via agreement, already covered). `agreement_buckets` encodes
# only the MATCH between head and dependent -- never the dependent's absolute Case/Number/... value
# on its own, which is exactly what a verb's valency selects for. Hashed, not a dense per-value
# table, so a rare or unseen morph BUNDLE degrades gracefully via collisions rather than needing its
# own dedicated row.
MORPHHASH_BUCKETS = 64
N_MORPHHASH_BINS = MORPHHASH_BUCKETS + 1            # +1: bucket 0 reserved for an EMPTY morph bundle


def morph_hash_buckets(doc, n_buckets=MORPHHASH_BUCKETS):
    """(n,) int bucket ids, one per token -- head-INDEPENDENT (see the module note), so this is a
    per-dependent vector, not the (n+1, n) grid dist/agree/pos/direction use.

    ⚠ `zlib.crc32`, NOT Python's `hash()`: string hashing is RANDOMISED PER PROCESS
    (`PYTHONHASHSEED`) unless explicitly disabled, which would silently scramble which bucket means
    what between the process that TRAINED this bias and the one that later LOADS the weights to
    evaluate or deploy -- the exact same class of trap `rename_deprel_label.py` exists for (a
    mapping that must be position-for-position identical across processes, not merely "the same
    set")."""
    ids = np.zeros(len(doc), dtype=np.int64)
    for i, t in enumerate(doc):
        s = str(t.morph)
        if s:
            ids[i] = 1 + (zlib.crc32(s.encode("utf-8")) % n_buckets)
    return ids


# ⚠ LEMMA-IDENTITY HASH BIAS (--joint-label --lemhash), HEAD-SIDE, DISCRETE. `--lemcase` (verb-lemma
# x dependent-Case bilinear) cannot apply to lzh -- lzh has no Case morphology at all (the same fact
# that ruled out --agreement for lzh), so lemcase_bkt would collapse to one dead bucket. This is the
# lexically-specific term that DOES fit lzh's grammar: a discrete, hashed bias on the HEAD's lemma
# IDENTITY -- not the continuous `lemvec` (a distributional vector, coarse/collocational for single
# characters per build_lemma_vectors.py --report) and not `lemcase` (needs Case) -- built for lzh's
# `comp:obj`/`parataxis`, dominated by a small closed set of matrix verbs (曰謂使有無爲如以得知 alone
# cover 28.3%% of comp:obj) that `lemvec`'s continuous readout only lifted modestly. Broadcasts over
# the dependent axis exactly like `lemvec` does (a per-HEAD bias, independent of which dependent),
# but via a hash table instead of a linear readout of a pretrained vector -- so it needs no external
# lemma-vector table and degrades gracefully on an unseen lemma via collision, the same contract
# `morph_hash_buckets` already has for the dependent side.
#
# ⚠ 512 BUCKETS WAS TOO FEW, MEASURED: unlike `morph_hash_buckets`'s DEPENDENT-only, small-cardinality
# MORPH-bundle space (a few dozen real values, so 64 buckets is near-1:1), `lemma_hash_buckets` hashes
# EVERY token's own identity, since every token is a candidate HEAD, not just a closed verb class --
# lzh's presegmented training corpus has 9,018 distinct (lemma_ or text) types, ~17.6 per bucket at
# 512, and the distribution is sharply Zipfian (`。` 31,915 tokens, `，` 24,483, `之` 12,763, ...). A
# handful of massively frequent types dominate whichever bucket they collide into, since this term
# BROADCASTS additively over every dependent of a given head -- corrupting the score for every arc
# that head could plausibly govern, not just the rare lemma sharing its bucket. First measured at 512:
# `lzh_arcfactored_lemhash` (direction+pos+lemvec+pron+lemhash) collapsed to LAS 61.13 against
# `lzh_arcfactored_pron`'s 75.31 on the identical recipe minus lemhash -- see NEGATIVE-RESULTS.md.
# 4096 cuts the average collision load to ~2.2 types/bucket.
LEMHASH_BUCKETS = 4096
N_LEMHASH_BINS = LEMHASH_BUCKETS + 1                # +1: bucket 0 reserved for an empty/unknown lemma


def lemma_hash_buckets(doc, n_buckets=LEMHASH_BUCKETS):
    """(n+1,) int bucket ids, one per HEAD position INCLUDING the virtual root (row 0, always
    bucket 0) -- unlike morph_hash_buckets' (n,) dependent-only shape, this is indexed by a head
    position exactly like `lemvec`'s (n+1, dim) table is. Same crc32 mechanism/caveat as
    morph_hash_buckets (NOT Python's hash(), which is randomised per process)."""
    ids = np.zeros(len(doc) + 1, dtype=np.int64)
    for i, t in enumerate(doc):
        s = t.lemma_ or t.text
        if s:
            ids[i + 1] = 1 + (zlib.crc32(s.encode("utf-8")) % n_buckets)
    return ids


# ⚠ DEPENDENT-LEMMA HASH BIAS (--joint-label --lemhash-dep), `lemma_hash_buckets`' mirror image on
# the DEPENDENT axis -- (n,)-shaped like `morph_hash_buckets`, not (n+1,)-shaped like the head-side
# version (a dependent is never the virtual root). Built after `--lemcase` (verb x dependent-Case)
# measurably FAILED to close la's mod/comp:obl trade: checking whether the premise held at all found
# 81.1% of the ambiguous mass has NO Case value at all (the dependent is an ADP/ADV/SCONJ heading a
# PP/adverbial/clause, not a bare case-marked noun -- Case lives one level down, on the PP's own
# object, which no la bias reads). Within that Case-less 81%, the DEPENDENT's own lemma identity
# predicts the label at 88.75% (per/si/secundum/non/sicut/nisi/sic sit at ~100% one label; in/ad/ex
# remain genuinely mixed), dwarfing the ~67.5% lemma-blind baseline -- `lemvec_dep` already asks this
# question but CONTINUOUSLY; this asks it DISCRETELY, since the top offenders are a largely closed,
# near-deterministic class that a hash table can fit more cleanly than a distributional vector space.
#
# ⚠ SIZE THE BUCKET COUNT AGAINST THE ACTUAL VOCABULARY, MEASURED, not copied from a smaller
# precedent's count -- `--lemhash` (this table's head-side sibling) first shipped at 512 buckets for
# lzh's 9,018 distinct types and collapsed LAS 14 points from hash collision (NEGATIVE-RESULTS.md);
# la's corpus has 16,501 distinct (lemma_ or text) types, so this starts at a bucket count sized for
# that from the outset rather than repeating the mistake.
LEMHASHDEP_BUCKETS = 16384
N_LEMHASHDEP_BINS = LEMHASHDEP_BUCKETS + 1          # +1: bucket 0 reserved for an empty/unknown lemma


def lemma_hash_buckets_dep(doc, n_buckets=LEMHASHDEP_BUCKETS):
    """(n,) int bucket ids, one per DEPENDENT -- head-INDEPENDENT (see the module note), so this is
    a per-dependent vector like `morph_hash_buckets`, not the (n+1,) head-indexed shape
    `lemma_hash_buckets` uses. Same crc32 mechanism/caveat as both of those."""
    ids = np.zeros(len(doc), dtype=np.int64)
    for i, t in enumerate(doc):
        s = t.lemma_ or t.text
        if s:
            ids[i] = 1 + (zlib.crc32(s.encode("utf-8")) % n_buckets)
    return ids


# ⚠ PER-FEATURE SPLIT OF morphhash (--joint-label --feat). Hashing the WHOLE morph bundle into one
# bucket mixed Case together with Number/Gender/..., which diluted exactly the distinction that
# would separate la's `subj` (nominative) from `comp:obj` (accusative) -- morphhash's own
# regression on `subj` (gap -5.09 on `la_arcfactored_lexical`) traced to this. This gives each
# configured feature its OWN clean bias table, keyed on that feature's small enumerated vocabulary
# (Case has ~7 values, Number 2, ...) rather than a 64-way hash shared by everything at once. Which
# features to model comes from LANGS[lang]['joint_embed']['kwargs']['feats'] -- the SAME list the
# project already uses for this language's per-feature categorical embedding channel, so there is
# only one place that names "the morphological features this language's arc-factored decoder
# reads", not two that could drift apart.
#
# ⚠ THE VOCABULARY IS DATA-DERIVED AND MUST BE SAVED. Unlike UPOS_LIST (a fixed, small, universal
# tagset), a feature's value inventory is corpus-specific and its ORDER is what `feat_buckets`
# indexes into -- exactly the same contract `labs` (the sorted deprel list) already has with
# `meta["labels"]`. `build_feat_vocab` fixes the vocabulary ONCE, from plain_tr (predicted MORPH,
# never gold -- the same train/inference-skew discipline `agreement_buckets` already documented);
# meta.json's `feat_vocab` must travel with the checkpoint, and `feat_buckets` must be called with
# vocabularies rebuilt from THAT saved list, never recomputed from a possibly-different split.
#
# ⚠ TWO DIFFERENT VALUE SOURCES SHARE THIS ONE MECHANISM, via `feat_getter`. la/sa name real UD
# MORPH features (Case, Number, ...); lzh has none (see `upos_like`'s note on why `.pos_` is empty
# pre-parse) but DOES have a composite XPOS `tag_` string with four comma-separated fields
# (`'v,動詞,行為,態度'` -- coarse class, then three levels of subclass) that `upos_like` only ever
# reads the first two of. Naming a channel `"xpos_f<N>"` routes it to field N of `tag_` instead of a
# MORPH lookup, so `--feat` on lzh becomes "one independent bias per tag field" for free, with
# meta.json still storing only plain strings (channel NAMES), never a lambda -- `feat_getter` is
# the single place that turns a name back into an extractor, called identically at train and
# eval/analyse time.
def feat_getter(name):
    if name.startswith("xpos_f"):
        idx = int(name[len("xpos_f"):])

        def get(t):
            parts = t.tag_.split(",")
            return (parts[idx],) if idx < len(parts) and parts[idx] and parts[idx] != "*" else ()
        return get

    def get(t):
        return tuple(t.morph.get(name))
    return get


def build_feat_vocab(docs, feat):
    """Sorted list of distinct non-empty value-tuples for `feat` across `docs`, via `feat_getter`."""
    getter = feat_getter(feat)
    vals = {getter(t) for d in docs for t in d} - {()}
    return sorted(vals)


def feat_buckets(doc, feat, vocab_index):
    """(n,) int bucket ids, one per token -- head-independent, exactly like morph_hash_buckets: 0 =
    no value for this feature (including a value OUT OF VOCABULARY, e.g. unseen at train time --
    both get the same "no information" code, never a crash); 1.. = 1 + index into vocab_index."""
    getter = feat_getter(feat)
    ids = np.zeros(len(doc), dtype=np.int64)
    for i, t in enumerate(doc):
        j = vocab_index.get(getter(t))
        if j is not None:
            ids[i] = 1 + j
    return ids


# ⚠ PREVERBAL-PRONOUN SOFT CONSTRAINT (--joint-label --pron). Classical Chinese fronts a PRONOUN
# object before its verb far more readily than a full NP object -- canonical order is verb-object,
# and a preverbal NP object is comparatively rare and typically topicalised, while a preverbal
# PRONOUN object is common (negated/interrogative clauses in particular). Neither `direction`
# (label-specific but PRONOUN-blind) nor `pos` (POS-pair-specific but DIRECTION-blind) captures
# this alone -- it needs their CONJUNCTION, direction x is-the-dependent-a-pronoun, as one joint
# per-label cell. A SOFT constraint, i.e. a LEARNED bias like every other term here, not a hard
# decode-time filter -- consistent with this whole architecture's approach (a bias the model can
# still override when other evidence disagrees, not a rule that silently forbids a reading).
N_PRON_BINS = 3 * 2   # direction_buckets' 3 values (root/self, precedes, follows) x is-PRON (0/1)


def preverbal_buckets(doc, window):
    """(n+1, n) int bucket ids over [virtual root | tokens] x dependents: direction_buckets' own
    bucket, doubled by whether the DEPENDENT's own category is PRON -- reuses direction_buckets()
    and upos_like() rather than recomputing either. A full (h, d)-grid term, NOT head-independent
    like feat/morphhash: direction is a property of the PAIR, not of the dependent alone."""
    n = len(doc)
    dbkt = direction_buckets(n, window)                            # (n+1, n), values in {0,1,2}
    is_pron = np.array([1 if upos_like(t) == "PRON" else 0 for t in doc], dtype=np.int64)
    return dbkt * 2 + is_pron[None, :]


# Per-language config. `src` MUST be the arm that actually ships/is-current -- not merely one with
# the right component names -- because `encoder_and_upstream()` reads the parser's *actual*
# architecture off it.
#
# `joint_embed` says how --joint builds its fresh embed layer. Where the project already has a
# richer, PURPOSE-BUILT channel for this language's morphology/lexis, USE IT rather than a coarse
# whole-string MORPH/LEMMA hash:
#   * la: `sud.LemmaVecFeatsEmbed.v1` -- the REAL distributional lemma-vector table
#     (`scripts/la_lemmavec_96.npz`, the same PCA'd PPMI table `training_la_lemvec_sud` seals into
#     its own bytes) plus per-FEATURE categorical embeddings (Case, Number, Gender, ... -- see
#     `sud_lemmavec_embed.py`), rather than hashing the whole MORPH bundle as one opaque symbol.
#     `diagnose_la_deprel_errors.py` found comp:obl/mod and subj/comp:pred confusions dominated by
#     LABEL-ONLY errors with the head already correct -- exactly what a de-composed case/mood/voice
#     signal (Latin's obliques are case-marked) should help the LABEL scorer resolve, where a single
#     hashed bundle drowns it in everything else the encoder is also trying to learn.
#   * sa: `sud.MultiHashEmbedFeats.v1` with the same `feats` list its real dedicated parser encoder
#     reads (`Case, Number, Gender, Person`) -- not the full `sud.AnalyserFeatsEmbed.v1`, which also
#     pulls in the Sanskrit lexicon lookup (`kosha`/vidyut-data), a runtime dependency this research
#     trainer has no reason to take on for a per-feature embedding table alone.
#   * lzh: plain `spacy.MultiHashEmbed.v2` -- its parser sits right after tok2vec with nothing
#     upstream reading morph/lemma first, so there is no richer channel to reuse.
LANGS = {
    "la": {
        "src": "training_la_lemvec_sud/model-best",       # LA_BASE in package_sud.sh
        "corpus": "corpus_la_ext/la_ittbproiel-sud-%s.relabeled_ext.spacy",
        "window": 50,
        "lemvec_table": "scripts/la_lemmavec_96.npz",     # same table --joint's own embed seals in
        # ⚠ --feat's channel list. Kept as its OWN key rather than reading joint_embed's own
        # `feats` (they happen to match today, but --feat is meaningful even under --no-joint or a
        # different embed architecture, so it should not be implicitly tied to the embed's choices).
        "feat_channels": ["Case", "Number", "Gender", "VerbForm", "Mood", "Tense", "Voice",
                          "Person", "PronType", "Degree", "InflClass", "Aspect"],
        "joint_embed": {
            "arch": "sud.LemmaVecFeatsEmbed.v1",
            "kwargs": {
                "width": 96, "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                "rows": [5000, 1000, 2500, 2500], "include_static_vectors": False,
                "vectors": "scripts/la_lemmavec_96.npz", "vector_dim": 96,
                "feats": ["Case", "Number", "Gender", "VerbForm", "Mood", "Tense", "Voice",
                          "Person", "PronType", "Degree", "InflClass", "Aspect"],
                "feat_rows": [64, 16, 32, 32, 16, 32, 16, 16, 64, 32, 64, 16],
            },
        },
    },
    "sa": {
        "src": "training_sa_mp2_sub_s1/model-best",       # SA_BASE in package_sud.sh
        "corpus": {
            "train": "corpus_sa_mwt_rl2/train.csl_mwt.spacy",
            "dev": "corpus_sa_mwt_rl2/sa_vedic-sud-dev.relabeled_ext.csl_mwt.spacy",
            "test": "corpus_sa_mwt_rl2/sa_vedic-sud-test.relabeled_ext.csl_mwt.spacy",
        },
        "window": 50,
        "lemvec_table": "scripts/sa_lemmavec_96.npz",
        "feat_channels": ["Case", "Number", "Gender", "Person"],
        "joint_embed": {
            "arch": "sud.MultiHashEmbedFeats.v1",
            "kwargs": {
                "width": 96, "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE", "MORPH"],
                "rows": [5000, 1000, 2500, 2500, 1024], "include_static_vectors": False,
                "feats": ["Case", "Number", "Gender", "Person"],
                "feat_rows": [32, 16, 16, 16],
            },
        },
    },
    "lzh": {
        "src": "training_lzh_depmorph_resplit/model-best",
        "corpus": ("corpus_lzh_resplit_ctl/lzh_kyoto-sud-%s."
                   "relabeled_ext.udep_ruled.punct.rulemerged.resplit.spacy"),
        "window": 30,
        # SikuBERT/kanripo vectors already used for the tagger (build_lzh_sikubert_vectors.py),
        # rotated through build_lemma_vectors.py's PCA(96) (a no-op on variance -- already 96-d --
        # kept only for a consistent npz{keys,vectors} shape with la/sa's tables).
        "lemvec_table": "scripts/lzh_lemmavec_96.npz",
        # lzh has no UD MORPH at all -- these route through feat_getter's "xpos_f<N>" branch
        # instead, one independent bias per comma-field of the composite XPOS tag_ ('v,動詞,行為,
        # 態度'): field 0 is upos_like's own coarse class (v/n/p/s), fields 1-3 are three levels of
        # subclass (12/46/84 distinct values respectively -- checked directly against the tagger's
        # own label set, not assumed).
        "feat_channels": ["xpos_f0", "xpos_f1", "xpos_f2", "xpos_f3"],
        "joint_embed": {
            "arch": "spacy.MultiHashEmbed.v2",
            "kwargs": {
                "width": 96, "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                "rows": [5000, 1000, 2500, 2500], "include_static_vectors": False,
            },
        },
    },
}


def build_joint_embed(cfg):
    """The --joint embed layer, from LANGS[lang]['joint_embed'] -- one place, so the trainer and
    every analysis/diagnostic script build the IDENTICAL architecture rather than each hand-rolling
    its own reconstruction and drifting out of sync with the others."""
    from spacy.util import registry
    spec = cfg["joint_embed"]
    return registry.architectures.get(spec["arch"])(**spec["kwargs"])


def build_joint_embed_from_meta(meta):
    """Reconstruct a saved checkpoint's embed from its OWN meta.json, not from the (possibly since
    -changed) LANGS table -- a checkpoint must stay loadable even if LANGS[lang]['joint_embed'] is
    later edited. Falls back to the legacy `joint_attrs`/`joint_rows` schema for checkpoints saved
    before this function existed."""
    from spacy.util import registry
    if meta.get("joint_embed"):
        spec = meta["joint_embed"]
        return registry.architectures.get(spec["arch"])(**spec["kwargs"])
    return registry.architectures.get("spacy.MultiHashEmbed.v2")(
        width=96, attrs=meta["joint_attrs"], rows=meta["joint_rows"], include_static_vectors=False)


def corpus_path(lang, split):
    c = LANGS[lang]["corpus"]
    return c[split] if isinstance(c, dict) else c % split


def load(lang, split, nlp, limit=None):
    docs = list(DocBin().from_disk(corpus_path(lang, split)).get_docs(nlp.vocab))
    return docs[:limit] if limit else docs


def explode_sentences(docs):
    """Split every multi-sentence `convert -n 10` doc into one gold SENTENCE per item.

    ⚠ WHY THIS EXISTS. `diagnose_la_longdist.py` found that a token's dependency accuracy grows
    monotonically WORSE with the number of intervening tokens the whole-doc BiLSTM has to carry
    (dist 1 +0.5 -> dist 7 +4.6 -> root +6.3, re-encoding the SAME trained weights per gold
    sentence instead of per whole doc) -- the encoder's limited capacity (width 96, depth 2) is
    partly spent carrying content from the OTHER ~9 sentences sharing the document, which a
    transition parser's fresh-per-sentence stack never has to do. `--presegment` trains on that
    fresh-per-sentence regime directly, rather than reusing whole-doc-trained weights post hoc.

    ⚠ THE TRADE. A presegmented arc-factored decoder no longer discovers sentence boundaries
    itself -- every item has exactly one root by construction, so the "arc-factored beats the
    transition parser at finding roots without gold hints" result (docs/-- la ANALYSE log, +3.09)
    stops being something this arm can even be tested on. It needs a real sentence boundary from
    somewhere at deployment. Comparisons against this checkpoint should therefore also run the
    transition baseline per gold sentence (`analyse_arcfactored.py`'s `whole_doc=False` path) --
    matched regimes, not a doc-level parser against a sentence-level one.

    `Span.as_doc()` remaps HEAD/DEP to the new doc's own indexing and is exact for this project's
    trees: SUD attachment never crosses a sentence boundary (parataxis-like phenomena live in the
    separate MISC layer, not raw head attachment), so no head is ever left dangling outside its
    sentence.
    """
    out = []
    for d in docs:
        for s in d.sents:
            out.append(s.as_doc())
    return out


def encoder_and_upstream(nlp):
    """The tok2vec submodel the PARSER actually reads, and the pipes that must run first.

    Resolved from the loaded pipeline itself, never hand-listed per language -- see the module
    docstring for why sa and la need genuinely different answers here.
    """
    idx = nlp.pipe_names.index("parser")
    upstream = nlp.pipe_names[:idx]
    parser = nlp.get_pipe("parser")
    t2v_ref = parser.model.get_ref("tok2vec")
    encoder = nlp.get_pipe("tok2vec").model if "listener" in t2v_ref.name else t2v_ref
    return encoder, upstream


def annotate_upstream(nlp, doc, upstream):
    """Run every pipe before the parser, IN ORDER, so predicted TAG/MORPH/LEMMA are on the doc
    exactly as the real pipeline would leave them -- predicted, never gold, at both train and
    eval time, so there is no train/inference skew."""
    for name in upstream:
        nlp.get_pipe(name)(doc)
    return doc


def per_doc(out, docs):
    """Normalise an encoder's `.predict()` output to one array per doc.

    ⚠ NOT EVERY tok2vec REF RETURNS `List[Floats2d]`. la's parser listens to the plain `tok2vec`
    pipe (`spacy.Tok2Vec.v2`'s own contract, List[Floats2d]) -- but sa's parser has its OWN
    dedicated, non-listener ref, and that ref's name ends `...>>list2array>>linear`: it is not just
    the embed+encode block, it is the parser's FULL "lower" projection (width 96 -> hidden_width
    128), which `spacy.TransitionBasedParser.v2` builds as `tok2vec -> list2array -> linear` and
    registers ALL of it under the ref name "tok2vec". `list2array` concatenates the batch into one
    array in doc order, so `.predict()` on a batch of sa docs returns a single (total_tokens, 128)
    ndarray, not a list -- and indexing `[0]` on it silently returns just the first TOKEN's vector,
    not the first DOC's matrix. This is, if anything, a MORE faithful read than stopping at the
    embed+encode block: it is the literal input to the parser's action classifier.
    """
    if isinstance(out, list):
        return out
    lens = [len(d) for d in docs]
    res, off = [], 0
    for n in lens:
        res.append(np.asarray(out[off:off + n]))
        off += n
    return res


def batched_predict(encoder, plain_docs, batch=64):
    out = []
    for i in range(0, len(plain_docs), batch):
        chunk = plain_docs[i:i + batch]
        out.extend(per_doc(encoder.predict(chunk), chunk))
    return out


def make_plain(nlp, docs, upstream):
    """Docs carrying the SAME upstream annotation `vectors()` relies on -- needed by --joint, whose
    fresh embed layer reads MORPH/LEMMA off these docs directly rather than through a frozen
    encoder's forward pass."""
    out = []
    for d in docs:
        pd = Doc(nlp.vocab, words=[t.text for t in d])
        annotate_upstream(nlp, pd, upstream)
        out.append(pd)
    return out


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    idx = np.arange(n)
    m[:, 1:] = np.abs(idx[:, None] - idx[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m


class Biaffine:
    """Scores head->dep over a window; labels scored only for the selected arc."""

    def __init__(self, w, h, nlab, seed=0):
        r = np.random.default_rng(seed)
        s = lambda *d: (r.normal(size=d) * (1.0 / np.sqrt(d[0]))).astype("float32")
        self.p = {"Wh": s(w, h), "bh": np.zeros(h, "float32"),
                  "Wd": s(w, h), "bd": np.zeros(h, "float32"),
                  "U": np.zeros((h, h), "float32"), "u": np.zeros(h, "float32"),
                  "dist": np.zeros(N_DIST_BINS, "float32"),
                  "Lh": s(w, h), "Ld": s(w, h),
                  "V": np.zeros((nlab, h, h), "float32"), "v": np.zeros((nlab, 2 * h), "float32"),
                  "cb": np.zeros(nlab, "float32")}
        self.h, self.nlab = h, nlab

    def backprop_inputs(self, dH, dD, dLH, dLD, H, D, LH, LD):
        """Gradient wrt the ENCODER's output -- needed only for --joint."""
        return ((dH * (H > 0)) @ self.p["Wh"].T + (dD * (D > 0)) @ self.p["Wd"].T
                + (dLH * (LH > 0)) @ self.p["Lh"].T + (dLD * (LD > 0)) @ self.p["Ld"].T)

    use_dist = True

    def arc_scores(self, X, k, drop=0.0, rng=None):
        n = X.shape[0]
        H = np.maximum(X @ self.p["Wh"] + self.p["bh"], 0)
        D = np.maximum(X @ self.p["Wd"] + self.p["bd"], 0)
        # inverted dropout on the projections, as in Dozat & Manning: the masks must be kept and
        # reapplied in backprop, or the gradient is for a different network than the forward pass.
        self.mh = self.md = None
        if drop > 0 and rng is not None:
            keep = 1.0 - drop
            self.mh = (rng.random(H.shape) < keep).astype("float32") / keep
            self.md = (rng.random(D.shape) < keep).astype("float32") / keep
            H = H * self.mh; D = D * self.md
        Hr = np.vstack([np.zeros((1, self.h), "float32"), H])
        S = (Hr @ self.p["U"]) @ D.T + (self.p["u"] @ D.T)[None, :]
        self.bkt = dist_buckets(n, k)
        if self.use_dist:
            S = S + self.p["dist"][self.bkt]
        S = np.where(window_mask(n, k).T, S, NEG)
        return S, H, D, Hr

    def label_scores(self, X, heads):
        n = X.shape[0]
        LH = np.maximum(X @ self.p["Lh"], 0); LD = np.maximum(X @ self.p["Ld"], 0)
        hv = np.where((heads > 0)[:, None], LH[np.maximum(heads - 1, 0)], 0.0)
        bil = np.einsum("nh,lhg,ng->nl", hv, self.p["V"], LD)
        lin = np.concatenate([hv, LD], 1) @ self.p["v"].T
        return bil + lin + self.p["cb"], LH, LD, hv


def softmax_ce(S, gold):
    """per-dependent CE over candidate heads; S is (n+1, n) with the virtual root in row 0"""
    Z = S.T                                     # (n, n+1)
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    n = Z.shape[0]
    loss = -np.log(np.maximum(P[np.arange(n), gold], 1e-9)).sum()
    dZ = P.copy(); dZ[np.arange(n), gold] -= 1.0
    return loss, dZ.T                           # (n+1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANGS))
    ap.add_argument("--src", default="", help="default: LANGS[lang]['src']")
    ap.add_argument("--window", type=int, default=0, help="default: LANGS[lang]['window']")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    # ⚠ THE FREEZE RECIPE IS WRONG FOR THIS DECODER, which is why --joint exists (measured on lzh:
    # frozen plateaus because the source encoder was fitted to a transition system's stack/buffer
    # features, not to a head-projection.dependent-projection dot product).
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--decay", type=float, default=1.0, help="multiply lr by this each epoch")
    ap.add_argument("--batch", type=int, default=1,
                    help="docs per optimiser step; 1 wastes BLAS throughput")
    ap.add_argument("--no-dist", action="store_true",
                    help="disable the signed-distance buckets")
    ap.add_argument("--bilstm", action="store_true",
                    help="MultiHashEmbed -> BiLSTM encoder instead of the CNN (--joint only)")
    ap.add_argument("--save", default="", help="directory to write the best model into")
    ap.add_argument("--joint", action="store_true",
                    help="train a fresh encoder jointly with the biaffine instead of freezing")
    # ⚠ NAME COLLISION WITH --joint, WHICH IS ABOUT THE ENCODER. This one is about the SCORER:
    # score every candidate (head, label) pair together (sud_joint_biaffine.JointBiaffine) instead
    # of scoring arcs first and labelling only the winner -- so the signed-distance bias can be
    # PER-LABEL. Motivated by sweep_la_distbias.py: the shared-bias arm is already near-optimal in
    # aggregate and CANNOT be tuned further by rescaling, because conj:coord wants the bias
    # weak-to-absent while det/mod want it strong, and a single scalar cannot serve both at once.
    ap.add_argument("--joint-label", action="store_true",
                    help="score (head, dependent, label) jointly, with a per-label distance bias, "
                         "instead of scoring arcs then labelling only the winner")
    ap.add_argument("--agreement", action="store_true",
                    help="--joint-label only: add a per-label Case/Number/Gender agreement bias "
                         "(agreement_buckets) -- built for la's `mod`, which picks a wrong "
                         "SAME-DISTANCE candidate that a distance bias cannot discriminate")
    ap.add_argument("--direction", action="store_true",
                    help="--joint-label only: add a per-label head-precedes/follows bias "
                         "(direction_buckets), pooled across all magnitudes -- a more robust "
                         "cousin of dist_by_label's per-magnitude direction split, for rare labels")
    ap.add_argument("--pos", action="store_true",
                    help="--joint-label only: add a per-label (head UPOS, dependent UPOS) "
                         "compatibility bias (pos_buckets) -- built for `subj` (head VERB/AUX 94%%) "
                         "and `cc` (dependent CCONJ 99%%), strongly POS-selected relations that "
                         "neither the distance nor the agreement bias reaches")
    ap.add_argument("--lemvec", action="store_true",
                    help="--joint-label only: add a per-label linear readout of the HEAD's "
                         "pretrained static lemma vector (LANGS[lang]['lemvec_table']) -- built for "
                         "lzh's `comp:obj`, dominated by a closed set of matrix verbs (曰謂使有無爲"
                         "如以得知 alone cover 28.3%%) that a coarse UPOS bias can't single out")
    ap.add_argument("--morphhash", action="store_true",
                    help="--joint-label only: add a per-label bias on a HASH of the DEPENDENT's own "
                         "MORPH bundle (morph_hash_buckets) -- built for la's mod/comp:obl trade "
                         "(they attach to VERB heads about equally often, so pos can't separate "
                         "them; comp:obl's Case is selected by the verb's VALENCY, independent of "
                         "any head agreement, which is all agreement_buckets encodes). SUPERSEDED "
                         "by --feat, which gives each feature its own clean table instead of "
                         "hashing them all into one -- kept only so an already-saved checkpoint "
                         "stays loadable and reproducible.")
    ap.add_argument("--feat", action="store_true",
                    help="--joint-label only: add an INDEPENDENT per-label bias for each of "
                         "LANGS[lang]['joint_embed']['kwargs']['feats'] (Case, Number, Gender, ...), "
                         "each keyed on that feature's own enumerated vocabulary rather than one "
                         "shared hash -- the fix for --morphhash's regression on la's `subj` "
                         "(diluted by mixing Case together with every other feature)")
    ap.add_argument("--pron", action="store_true",
                    help="--joint-label only: add a per-label bias on (direction, is-dependent-a-"
                         "PRONOUN) -- the lzh preverbal-object soft constraint: a fronted PRONOUN "
                         "object is common, a fronted full-NP object is comparatively rare and "
                         "typically topicalised, a distinction neither direction nor pos alone "
                         "captures (see preverbal_buckets)")
    ap.add_argument("--lemvec-dep", action="store_true", dest="lemvec_dep",
                    help="--joint-label only: add a per-label linear readout of the DEPENDENT's "
                         "pretrained static lemma vector -- the dependent-side counterpart of "
                         "--lemvec, built for la's mod/comp:obl trade (does this dependent word, by "
                         "its own distributional profile, look oblique-like or modifier-like)")
    ap.add_argument("--lemcase", action="store_true",
                    help="--joint-label only: add a BILINEAR per-label term, head lemma vector x "
                         "dependent Case bucket -- unlike every other bias here, genuinely "
                         "multiplicative, built because `dico` ('say') alone governs 2332 mod + "
                         "1682 comp:obl tokens (a ~58/42 split WITHIN ONE LEMMA): the choice is "
                         "verb + Case together, not verb alone (lemvec) or Case alone (feat), and "
                         "an additive sum of the two cannot express that interaction")
    ap.add_argument("--lemhash", action="store_true",
                    help="--joint-label only: add a per-label bias on a HASH of the HEAD's own "
                         "lemma IDENTITY (lemma_hash_buckets) -- the lexically-specific term for "
                         "languages with no Case morphology (lzh), where --lemcase cannot apply: a "
                         "discrete counterpart to --lemvec's continuous readout, built for lzh's "
                         "comp:obj/parataxis matrix-verb closed class")
    ap.add_argument("--lemhash-dep", action="store_true", dest="lemhashdep",
                    help="--joint-label only: add a per-label bias on a HASH of the DEPENDENT's own "
                         "lemma IDENTITY (lemma_hash_buckets_dep) -- built for la after --lemcase "
                         "failed: 81.1%% of la's mod/comp:obl ambiguous mass has no Case value at "
                         "all (the dependent is an ADP/ADV/SCONJ, not a bare case-marked noun), and "
                         "within that slice the DEPENDENT's own lemma (per/si/secundum/cum/... vs. "
                         "the mixed in/ad/ex) predicts the label far better than lemma-blind chance")
    ap.add_argument("--presegment", action="store_true",
                    help="explode whole docs into one gold SENTENCE per training/eval item, so the "
                         "encoder never carries state across a sentence boundary -- see "
                         "explode_sentences() for why (diagnose_la_longdist.py's finding). Gives up "
                         "self-discovered sentence roots: every item has exactly one root.")
    a = ap.parse_args()
    cfg = LANGS[a.lang]
    a.src = a.src or cfg["src"]
    a.window = a.window or cfg["window"]
    nlp = spacy.load(a.src)
    encoder, upstream = encoder_and_upstream(nlp)
    print(f"  [{a.lang}] src={a.src} window={a.window}", flush=True)
    print(f"  upstream pipes read before the parser's own input: {upstream}", flush=True)
    tr = load(a.lang, "train", nlp, a.limit or None)
    te = load(a.lang, "test", nlp, (a.limit // 4) if a.limit else None)
    if a.presegment:
        tr, te = explode_sentences(tr), explode_sentences(te)
        print(f"  PRESEGMENT: exploded into {len(tr)} train / {len(te)} test single-sentence items",
              flush=True)
    print(f"  train {len(tr)} docs, test {len(te)}", flush=True)
    labs = sorted({t.dep_ for d in tr for t in d})
    li = {l: i for i, l in enumerate(labs)}
    print(f"  {len(labs)} deprel labels; window {a.window}", flush=True)
    enc = None
    plain_tr, plain_te = make_plain(nlp, tr, upstream), make_plain(nlp, te, upstream)
    if a.joint:
        from spacy.util import registry
        from thinc.api import chain as _chain
        embed = build_joint_embed(cfg)
        print(f"  JOINT: fresh {cfg['joint_embed']['arch']} embed "
              f"({cfg['joint_embed']['kwargs'].get('attrs')}"
              f"{', feats=' + str(cfg['joint_embed']['kwargs']['feats']) if cfg['joint_embed']['kwargs'].get('feats') else ''})",
              flush=True)
        if a.bilstm:
            # ⚠ THINC'S NATIVE LSTM, NOT `spacy.TorchBiLSTMEncoder.v1` -- torch is 437 MB installed
            # against this project's 250 MB serverless budget (docs/packaging-and-release.md).
            from thinc.api import LSTM, with_padded
            enc = _chain(embed, with_padded(LSTM(96, 96, bi=True, depth=2)))
            enc.set_dim("nO", 96)
        else:
            enc = _chain(embed, registry.architectures.get("spacy.MaxoutWindowEncoder.v2")(
                width=96, depth=4, window_size=1, maxout_pieces=3))
        enc.initialize(X=plain_tr[:64])
        print(f"  JOINT: training a fresh {'BiLSTM (depth 2)' if a.bilstm else 'MaxoutWindowEncoder (depth 4)'}"
              f" width-96 encoder with the biaffine", flush=True)
        Xtr = Xte = None
        w = 96
    else:
        Xtr = batched_predict(encoder, plain_tr)
        Xte = batched_predict(encoder, plain_te)
        w = Xtr[0].shape[1]
    if a.joint_label:
        lemvec_dim = 0
        if a.lemvec or a.lemcase:
            _, _lv = load_lemvec_table(cfg["lemvec_table"])
            lemvec_dim = _lv.shape[1]
        lemvec_dep_dim = 0
        if a.lemvec_dep:
            _, _lvd = load_lemvec_table(cfg["lemvec_table"])
            lemvec_dep_dim = _lvd.shape[1]
        # ⚠ THE VOCABULARY MUST BE BUILT (and its order fixed) BEFORE the scorer is constructed --
        # n_bins per feature comes from it -- and from plain_tr, never gold `tr` (see feat_buckets'
        # module note).
        feat_names = cfg.get("feat_channels", []) if a.feat else []
        if a.feat and not feat_names:
            print(f"  --feat has no effect: LANGS['{a.lang}'] configures no feat_channels",
                  flush=True)
        feat_vocabs = {name: build_feat_vocab(plain_tr, name) for name in feat_names}
        feat_bins = {name: 1 + len(v) for name, v in feat_vocabs.items()}
        # `lemcase` reuses feat_buckets' own vocabulary-building machinery for its Case side --
        # it is not part of `--feat`'s own channel list (a.lemcase can be used without --feat),
        # so its vocabulary is built and stored separately even though the underlying function is
        # identical.
        lemcase_vocab = build_feat_vocab(plain_tr, "Case") if a.lemcase else []
        lemcase_bins = 1 + len(lemcase_vocab)
        m = JointBiaffine(w, a.hidden, len(labs), N_DIST_BINS, dist_buckets,
                           n_agree_bins=(N_AGREE_BINS if a.agreement else 0),
                           n_dir_bins=(N_DIR_BINS if a.direction else 0),
                           dir_buckets_fn=(direction_buckets if a.direction else None),
                           n_pos_bins=(N_POS_BINS if a.pos else 0),
                           n_lemvec_dim=lemvec_dim,
                           n_morphhash_bins=(N_MORPHHASH_BINS if a.morphhash else 0),
                           feat_bins=(feat_bins if feat_names else None),
                           n_pron_bins=(N_PRON_BINS if a.pron else 0),
                           n_lemvec_dep_dim=lemvec_dep_dim,
                           n_lemcase_dim=(lemvec_dim if a.lemcase else 0),
                           n_lemcase_bins=(lemcase_bins if a.lemcase else 0),
                           n_lemhash_bins=(N_LEMHASH_BINS if a.lemhash else 0),
                           n_lemhashdep_bins=(N_LEMHASHDEP_BINS if a.lemhashdep else 0))
        # ⚠ sud_joint_biaffine.py deliberately keeps float64 (precision for its OWN gradient
        # check); thinc's optimiser and the rest of this file are float32 throughout.
        for kk in m.p:
            m.p[kk] = m.p[kk].astype("float32")
        if a.no_dist:
            print("  --no-dist has no effect under --joint-label (not yet supported)", flush=True)
        print("  JOINT-LABEL: scoring (head, dependent, label) together, per-label distance bias"
              + (" + per-label Case/Number/Gender agreement bias" if a.agreement else "")
              + (" + per-label head-precedes/follows bias" if a.direction else "")
              + (" + per-label (head UPOS, dep UPOS) compatibility bias" if a.pos else "")
              + (" + per-label head-lemma-vector bias" if a.lemvec else "")
              + (" + per-label dependent-morph-hash bias" if a.morphhash else "")
              + (f" + per-label bias for each of {feat_names}" if feat_names else "")
              + (" + per-label preverbal-pronoun bias" if a.pron else "")
              + (" + per-label dependent-lemma-vector bias" if a.lemvec_dep else "")
              + (" + per-label head-lemma x dependent-Case BILINEAR bias" if a.lemcase else "")
              + (" + per-label head-lemma-identity-hash bias" if a.lemhash else "")
              + (" + per-label dependent-lemma-identity-hash bias" if a.lemhashdep else ""),
              flush=True)
    else:
        m = Biaffine(w, a.hidden, len(labs))
        m.use_dist = not a.no_dist
        if a.no_dist:
            print("  distance buckets DISABLED", flush=True)
    ops = NumpyOps(); opt = Adam(a.lr)
    gold_tr = [(np.array([0 if t.head.i == t.i else t.head.i + 1 for t in d]),
                np.array([li.get(t.dep_, 0) for t in d])) for d in tr]
    # ⚠ FROM plain_tr/plain_te (predicted MORPH, via the SAME upstream chain X came from), never
    # from gold `tr`/`te` -- training on gold agreement and meeting predicted agreement at
    # inference is exactly the train/inference skew this project has been bitten by before.
    agree_tr = agree_te = None
    if a.joint_label and a.agreement:
        agree_tr = [agreement_buckets(d) for d in plain_tr]
        agree_te = [agreement_buckets(d) for d in plain_te]
    pos_tr = pos_te = None
    if a.joint_label and a.pos:
        pos_tr = [pos_buckets(d) for d in plain_tr]
        pos_te = [pos_buckets(d) for d in plain_te]
    lemvec_tr = lemvec_te = None
    if a.joint_label and (a.lemvec or a.lemcase):
        # shared by BOTH the additive head-only `lemvec` bias and the bilinear `lemcase` term --
        # same table, same lookup, one array threaded to both.
        lemvec_tr = [lemma_vecs(d, cfg["lemvec_table"]) for d in plain_tr]
        lemvec_te = [lemma_vecs(d, cfg["lemvec_table"]) for d in plain_te]
    morph_tr = morph_te = None
    if a.joint_label and a.morphhash:
        morph_tr = [morph_hash_buckets(d) for d in plain_tr]
        morph_te = [morph_hash_buckets(d) for d in plain_te]
    feat_tr = feat_te = None
    if a.joint_label and feat_names:
        feat_vocab_idx = {name: {v: i for i, v in enumerate(vocab)}
                           for name, vocab in feat_vocabs.items()}
        feat_tr = [{name: feat_buckets(d, name, feat_vocab_idx[name]) for name in feat_names}
                   for d in plain_tr]
        feat_te = [{name: feat_buckets(d, name, feat_vocab_idx[name]) for name in feat_names}
                   for d in plain_te]
    pron_tr = pron_te = None
    if a.joint_label and a.pron:
        pron_tr = [preverbal_buckets(d, a.window) for d in plain_tr]
        pron_te = [preverbal_buckets(d, a.window) for d in plain_te]
    lemvec_dep_tr = lemvec_dep_te = None
    if a.joint_label and a.lemvec_dep:
        lemvec_dep_tr = [lemma_vecs_dep(d, cfg["lemvec_table"]) for d in plain_tr]
        lemvec_dep_te = [lemma_vecs_dep(d, cfg["lemvec_table"]) for d in plain_te]
    lemcase_tr = lemcase_te = None
    if a.joint_label and a.lemcase:
        lemcase_vocab_idx = {v: i for i, v in enumerate(lemcase_vocab)}
        lemcase_tr = [feat_buckets(d, "Case", lemcase_vocab_idx) for d in plain_tr]
        lemcase_te = [feat_buckets(d, "Case", lemcase_vocab_idx) for d in plain_te]
    lemhash_tr = lemhash_te = None
    if a.joint_label and a.lemhash:
        lemhash_tr = [lemma_hash_buckets(d) for d in plain_tr]
        lemhash_te = [lemma_hash_buckets(d) for d in plain_te]
    lemhashdep_tr = lemhashdep_te = None
    if a.joint_label and a.lemhashdep:
        lemhashdep_tr = [lemma_hash_buckets_dep(d) for d in plain_tr]
        lemhashdep_te = [lemma_hash_buckets_dep(d) for d in plain_te]
    drng = np.random.default_rng(1234)
    best = (-1.0, -1)
    for ep in range(a.epochs):
        opt.learn_rate = a.lr * (a.decay ** ep)
        order = np.random.default_rng(ep).permutation(len(tr))
        tot = 0.0; t0 = time.time()
        bi = 0
        while bi < len(order):
            chunk = order[bi:bi + a.batch]; bi += a.batch
            c = bi
            if enc is not None:
                Xs_b, bp_enc = enc([plain_tr[j] for j in chunk], is_train=True)
            else:
                Xs_b, bp_enc = [Xtr[j] for j in chunk], None
            gacc = {}; dX_b = []
            for bslot, di in enumerate(chunk):
              X = Xs_b[bslot]
              gh, gl = gold_tr[di]; n = X.shape[0]
              if a.joint_label:
                # JointBiaffine encapsulates the whole forward+backward -- no hand-unrolled chain
                # rule needed here, unlike the two-stage Biaffine below (see sud_joint_biaffine.py,
                # gradient-checked to 3.55e-4 worst relative error across 10 seeds).
                loss, g, dXin = m.loss_and_backward(
                    X, a.window, gh, gl, agree_tr[di] if agree_tr is not None else None,
                    pos_tr[di] if pos_tr is not None else None,
                    lemvec_tr[di] if lemvec_tr is not None else None,
                    morph_tr[di] if morph_tr is not None else None,
                    feat_tr[di] if feat_tr is not None else None,
                    pron_tr[di] if pron_tr is not None else None,
                    lemvec_dep_tr[di] if lemvec_dep_tr is not None else None,
                    lemcase_tr[di] if lemcase_tr is not None else None,
                    lemhash_tr[di] if lemhash_tr is not None else None,
                    lemhashdep_tr[di] if lemhashdep_tr is not None else None)
                tot += loss / max(n, 1)
                if bp_enc is not None:
                    dX_b.append(dXin.astype("float32"))
              else:
                S, H, D, Hr = m.arc_scores(X, a.window, a.dropout, drng)
                loss, dS = softmax_ce(S, gh)
                LS, LH, LD, hv = m.label_scores(X, gh)
                Z = LS - LS.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
                loss += -np.log(np.maximum(P[np.arange(n), gl], 1e-9)).sum()
                dL = P.copy(); dL[np.arange(n), gl] -= 1.0
                tot += loss / max(n, 1)
                g = {}
                HW = Hr @ m.p["U"]; dHW = dS @ D
                g["U"] = Hr.T @ dHW
                g["u"] = dS.sum(0) @ D
                g["dist"] = (np.bincount(m.bkt.ravel(), weights=dS.ravel(),
                                       minlength=N_DIST_BINS).astype("float32")
                           if m.use_dist else np.zeros(N_DIST_BINS, "float32"))
                dD = dS.T @ HW + np.outer(dS.sum(0), m.p["u"])
                dH = (dHW @ m.p["U"].T)[1:]
                # ⚠ REAPPLY THE DROPOUT MASKS. H and D here are the POST-mask activations, so
                # (H > 0) still selects the right units, but the mask scaling must be carried back.
                if m.mh is not None:
                  dH = dH * m.mh; dD = dD * m.md
                g["Wh"] = X.T @ (dH * (H > 0)); g["bh"] = (dH * (H > 0)).sum(0)
                g["Wd"] = X.T @ (dD * (D > 0)); g["bd"] = (dD * (D > 0)).sum(0)
                g["V"] = np.einsum("nl,nh,ng->lhg", dL, hv, LD)
                g["v"] = dL.T @ np.concatenate([hv, LD], 1)
                g["cb"] = dL.sum(0)
                dhv = np.einsum("nl,lhg,ng->nh", dL, m.p["V"], LD) + dL @ m.p["v"][:, :m.h]
                dLD = np.einsum("nl,lhg,nh->ng", dL, m.p["V"], hv) + dL @ m.p["v"][:, m.h:]
                dLH = np.zeros_like(LH)
                src = gh - 1; ok = gh > 0
                np.add.at(dLH, src[ok], dhv[ok])
                g["Lh"] = X.T @ (dLH * (LH > 0)); g["Ld"] = X.T @ (dLD * (LD > 0))
                if bp_enc is not None:
                  dXin = m.backprop_inputs(dH, dD, dLH, dLD, H, D, LH, LD)
                  dX_b.append(dXin.astype("float32"))
              for kk in g:
                gacc[kk] = gacc.get(kk, 0) + g[kk]
            # ⚠ ONE optimiser step PER BATCH, after accumulating every doc's gradient.
            if bp_enc is not None and dX_b:
                bp_enc(dX_b)
                enc.finish_update(opt)
            for kk in gacc:
                m.p[kk], _ = opt(("bi", kk), m.p[kk], (gacc[kk] / len(chunk)).astype("float32"))
            if c and c % 1500 == 0:
                print(f"    ep{ep} {c}/{len(tr)} loss {tot/(c+1):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        # ---- evaluate with CLE ----
        uas = las = ntok = 0
        Xte_ep = ([enc.predict([pd])[0] for pd in plain_te] if enc is not None else Xte)
        for te_i, (d, X) in enumerate(zip(te, Xte_ep)):
            n = X.shape[0]
            if a.joint_label:
                S, chosen = m.decode_scores(
                    X, a.window, agree_te[te_i] if agree_te is not None else None,
                    pos_te[te_i] if pos_te is not None else None,
                    lemvec_te[te_i] if lemvec_te is not None else None,
                    morph_te[te_i] if morph_te is not None else None,
                    feat_te[te_i] if feat_te is not None else None,
                    pron_te[te_i] if pron_te is not None else None,
                    lemvec_dep_te[te_i] if lemvec_dep_te is not None else None,
                    lemcase_te[te_i] if lemcase_te is not None else None,
                    lemhash_te[te_i] if lemhash_te is not None else None,
                    lemhashdep_te[te_i] if lemhashdep_te is not None else None)
            else:
                S, *_ = m.arc_scores(X, a.window)      # eval: no dropout
            # `mst` takes a SQUARE matrix over [virtual root | tokens]; arc_scores/decode_scores
            # emit (n+1 heads, n dependents), so pad a dependent column for the root, which may
            # never take a head.
            Sq = np.full((n + 1, n + 1), NEG, dtype="float64")
            Sq[:, 1:] = S
            heads = mst(Sq)[1:]
            if a.joint_label:
                pl = chosen[heads, np.arange(n)]           # the label CHOSEN FOR the winning arc
            else:
                LS, *_ = m.label_scores(X, heads)
                pl = LS.argmax(1)
            for i, t in enumerate(d):
                gh_ = 0 if t.head.i == t.i else t.head.i + 1
                ntok += 1
                if heads[i] == gh_:
                    uas += 1
                    if labs[pl[i]] == t.dep_: las += 1
        print(f"  epoch {ep}: loss {tot/len(tr):.4f}   UAS {uas*100/ntok:.2f}   LAS {las*100/ntok:.2f}"
              f"   (lr {opt.learn_rate:.2e})", flush=True)
        if a.save and las > best[0]:
            best = (las, ep)
            out = pathlib.Path(a.save); out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "biaffine.npz", **{k: v for k, v in m.p.items()})
            if enc is not None:
                (out / "encoder.bin").write_bytes(enc.to_bytes())
            (out / "meta.json").write_text(json.dumps(
                {"lang": a.lang, "src": a.src, "labels": labs, "window": a.window,
                 "hidden": a.hidden, "epoch": ep, "uas": uas*100/ntok, "las": las*100/ntok,
                 "joint": bool(a.joint), "bilstm": bool(a.bilstm), "presegment": bool(a.presegment),
                 "joint_label": bool(a.joint_label),
                 "agreement": bool(a.joint_label and a.agreement),
                 "direction": bool(a.joint_label and a.direction),
                 "pos": bool(a.joint_label and a.pos),
                 "lemvec": bool(a.joint_label and a.lemvec),
                 "lemvec_table": cfg.get("lemvec_table")
                                 if (a.joint_label and (a.lemvec or a.lemvec_dep or a.lemcase))
                                 else None,
                 "morphhash": bool(a.joint_label and a.morphhash),
                 "feat_names": feat_names if (a.joint_label and feat_names) else None,
                 # ⚠ MUST TRAVEL WITH THE CHECKPOINT: the vocabulary's ORDER is what feat_buckets
                 # indexes into (see build_feat_vocab's module note) -- json turns each tuple into
                 # a list, so eval/analyse code must turn it back into a tuple before using it as a
                 # dict key.
                 "feat_vocab": ({name: [list(v) for v in feat_vocabs[name]] for name in feat_names}
                                if (a.joint_label and feat_names) else None),
                 "pron": bool(a.joint_label and a.pron),
                 "lemvec_dep": bool(a.joint_label and a.lemvec_dep),
                 "lemcase": bool(a.joint_label and a.lemcase),
                 "lemcase_vocab": ([list(v) for v in lemcase_vocab]
                                   if (a.joint_label and a.lemcase) else None),
                 "lemhash": bool(a.joint_label and a.lemhash),
                 "lemhashdep": bool(a.joint_label and a.lemhashdep),
                 "joint_embed": cfg["joint_embed"] if a.joint else None,
                 "upstream": upstream}, ensure_ascii=False, indent=1))
            print(f"    saved -> {a.save} (best LAS {las*100/ntok:.2f})", flush=True)


if __name__ == "__main__":
    main()
