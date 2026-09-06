#!/usr/bin/env python
"""JointBiaffine -- arc AND label scored together, per candidate (head, dependent) pair, so the
signed-distance bias can be PER-LABEL rather than shared across every relation.

WHY. `sweep_la_distbias.py` showed the trained shared-bias arm is already near-optimal in
aggregate, and CANNOT be improved by rescaling: `conj:coord` wants the bias weak-to-absent
(distance is uninformative -- the correct head is whichever token started the coordination chain,
however far), while `det`/`mod` want it strong (the correct head is almost always adjacent). One
scalar cannot serve both. The old pipeline scores arcs FIRST (with the one shared bias) and labels
only the WINNING arc afterward -- so `conj:coord`'s preference for a far head never gets a vote in
which head wins in the first place.

THE FIX. Score every candidate (h, d) pair with its label folded in: for each label l,
    combined[h, d, l] = arc_raw[h, d] + dist_by_label[l, bucket(h, d)] + label_raw[h, d, l]
and let the ARC score used for decoding be `max_l combined[h, d, l]` -- so a distant head with a
label whose distance bias favours it (a high `dist_by_label[conj:coord, far]`) can win the arc
selection, something a single shared scalar structurally cannot represent.

TRAINING LOSS is a single joint softmax over (h, l) pairs per dependent, not two separate
cross-entropies -- the arc and the label are no longer decided in two stages even during learning.

⚠ H/D (the ARC-side projections) and LH/LD (the LABEL-side projections) remain COMPLETELY SEPARATE,
exactly as in the two-stage `Biaffine` -- there is no term mixing D with LH or H with LD. This keeps
the backward pass cleanly separable (each projection's gradient comes from exactly one forward path)
and is why this module could be written as an EXTENSION of the two-stage class's shapes rather than
a redesign: only the SCORE COMBINATION and the DISTANCE BIAS move from "after decoding" to "before".

⚠ GRADIENT-CHECKED BEFORE TRUSTING IT, the same discipline `sud_biaffine.py` used (that module's
history: "two bugs found and fixed before training... worst relative error 2.3e-04"). Run this
file directly to gradient-check against finite differences on tiny random dimensions; import
`JointBiaffine` from `train_arcfactored.py` for real use.
"""
import numpy as np
from sud_self_attention import attn_forward, attn_backward

NEG = -1e4


def window_mask(n, k):
    m = np.zeros((n, n + 1), dtype=bool)
    m[:, 0] = True
    idx = np.arange(n)
    m[:, 1:] = np.abs(idx[:, None] - idx[None, :]) <= k
    np.fill_diagonal(m[:, 1:], False)
    return m.T                      # (n+1, n): heads x dependents, matching arc_raw's convention


class JointBiaffine:
    def __init__(self, w, h, nlab, n_dist_bins, dist_buckets_fn, n_agree_bins=0,
                 n_dir_bins=0, dir_buckets_fn=None, n_pos_bins=0, n_lemvec_dim=0,
                 n_morphhash_bins=0, feat_bins=None, n_pron_bins=0, n_lemvec_dep_dim=0,
                 n_lemcase_dim=0, n_lemcase_bins=0, n_lemhash_bins=0, n_lemhashdep_bins=0,
                 n_sib_bins=0, n_grand_bins=0, use_attn_hd=False, seed=0):
        r = np.random.default_rng(seed)
        s = lambda *d: (r.normal(size=d) * (1.0 / np.sqrt(d[0]))).astype("float64")
        self.p = {"Wh": s(w, h), "bh": np.zeros(h), "Wd": s(w, h), "bd": np.zeros(h),
                  "U": np.zeros((h, h)), "u": np.zeros(h),
                  "Lh": s(w, h), "Ld": s(w, h),
                  "V": np.zeros((nlab, h, h)), "v": np.zeros((nlab, 2 * h)),
                  "cb": np.zeros(nlab),
                  "dist": np.zeros((nlab, n_dist_bins))}
        # ⚠ AGREEMENT IS OPTIONAL AND OFF BY DEFAULT (n_agree_bins=0, no "agree" param at all) --
        # a checkpoint saved without it must stay loadable by code that doesn't know about it, and
        # a language with no Case/Number/Gender (lzh) has nothing for this term to read.
        if n_agree_bins:
            self.p["agree"] = np.zeros((nlab, n_agree_bins))
        # ⚠ DIRECTION IS A COARSER, MORE ROBUST COUSIN OF THE DISTANCE BIAS, not a duplicate of it.
        # `dist_by_label` already separates the two directions AT EVERY MAGNITUDE (dist_buckets'
        # `b*2 - (delta<0)`), but that means a rare label's preference for, say, "head always
        # follows" is split thin across many (direction, magnitude) cells, each seeing few examples.
        # Pooling every magnitude into one "which side" bit per label gives that preference a single,
        # well-estimated cell -- unlike agreement, this needs nothing from the document (only token
        # POSITIONS), so it is computed the SAME way as `dist`: a buckets_fn called inside `forward`,
        # not data threaded in by the caller.
        if n_dir_bins:
            self.p["direction"] = np.zeros((nlab, n_dir_bins))
        # ⚠ POS-COMPATIBILITY IS A FOURTH, INDEPENDENT SCATTER TERM, threaded EXACTLY like agreement
        # (needs the doc's predicted UPOS, not just token positions -- unlike direction) and gated
        # the same way (`n_pos_bins=0` by default, no "pos" param at all, so an old checkpoint stays
        # loadable). Built because `subj` (head UPOS VERB/AUX 94% of the time) and `cc` (dependent
        # UPOS CCONJ 99% of the time) are strongly POS-selected relations that neither the distance
        # nor the agreement bias touches -- agreement only fires when BOTH sides have a full
        # Case/Number/Gender set, which most subj/cc candidates do not jointly satisfy anyway.
        if n_pos_bins:
            self.p["pos"] = np.zeros((nlab, n_pos_bins))
        # ⚠ LEMMA-VECTOR BIAS IS CONTINUOUS, not a discrete bucket lookup like the other three --
        # a per-label LINEAR READOUT of the HEAD's PRETRAINED static lemma vector (never X, the
        # trainable encoder's own contextual output: a genuinely separate, frozen signal source).
        # Built for lzh's `comp:obj`/`parataxis`: checking actual gold counts, VERB-headed comp:obj
        # is dominated by a closed, specific set of matrix verbs (曰 "say" alone is 7.5% of all
        # 59722 tokens; the top 10 lemmas -- 曰謂使有無爲如以得知, exactly the classic
        # speech/causative/existential/copular/modal/epistemic complement-taking verbs -- cover
        # 28.3%) that a coarse UPOS bias cannot single out (VERB-headed is shared by mod, subj,
        # comp:obl and conj:coord alike). Structurally this is exactly `lin1` (LHr @ v[:, :h].T)
        # with a DIFFERENT input matrix substituted in -- broadcasts over the dependent axis the
        # same way, so it shares that derivation almost verbatim; the only new plumbing is that the
        # "activation" here is a fixed lookup, not a function of X, so nothing backprops into it.
        if n_lemvec_dim:
            self.p["lemvec"] = np.zeros((nlab, n_lemvec_dim))
        # ⚠ MORPH-HASH IS THE DEPENDENT-SIDE HALF of the same idea `lemvec` is the head-side half
        # of. Built for la's `mod`/`comp:obl` PP-attachment confusion: `agree_buckets` only encodes
        # whether the dependent's Case MATCHES the head's -- exactly right for `mod` (adjectives
        # agree with their head noun) but uninformative for `comp:obl` (whose case is selected by
        # the governing VERB's valency -- ablative of means, dative of interest -- independent of
        # any head agreement). This term instead reads the dependent's OWN morphology directly, a
        # discrete bucket exactly like `pos` -- but head-INDEPENDENT (broadcasts over h the same
        # way `lin2`/`dlin2` already do), so its gradient reduction is `dlin2`'s, not a fresh scatter
        # over the full (n+1, n) grid.
        if n_morphhash_bins:
            self.p["morphhash"] = np.zeros((nlab, n_morphhash_bins))
        # ⚠ PER-FEATURE SPLIT OF morphhash. Hashing the WHOLE morph bundle into one bucket mixed
        # Case together with Number/Gender/etc., which diluted exactly the distinction that would
        # separate la's `subj` (nominative) from `comp:obj` (accusative) -- morphhash's own
        # regression on `subj` (gap -5.09) traced to this. `feat_bins` is a dict {feature_name ->
        # n_bins}, one INDEPENDENT bias table per morphological feature (Case, Number, Gender,
        # VerbForm, ... -- whatever LANGS[lang]['joint_embed']['kwargs']['feats'] already lists for
        # this language, so no separate feature inventory to maintain), each keyed on that feature's
        # OWN small enumerated vocabulary rather than a hash -- no collisions, and each feature gets
        # a clean, separately-learnable weight instead of sharing one crowded hash table. Still
        # head-INDEPENDENT (dlin2-shaped, like morphhash), which is what makes this a straight
        # generalisation rather than a redesign: N copies of the same mechanism, not a new one.
        self.feat_names = list((feat_bins or {}).keys())
        for name, nb in (feat_bins or {}).items():
            self.p[f"feat_{name}"] = np.zeros((nlab, nb))
        # ⚠ PREVERBAL-PRONOUN SOFT CONSTRAINT (lzh). Classical Chinese fronts a PRONOUN object
        # before its verb far more readily than a full NP object (canonical order is verb-object;
        # a preverbal NP object is comparatively rare, typically topicalised) -- a lexical fact
        # about lzh word order, not captured by `direction` (which is label-specific but PRONOUN-
        # blind) or `pos` (POS-pair-specific but DIRECTION-blind) alone: this needs their
        # CONJUNCTION, direction x is-the-dependent-a-pronoun, as one joint per-label cell. A
        # discrete (h, d)-pair scatter exactly like `pos` -- not head-independent, since direction
        # is a property of the PAIR, not of the dependent alone.
        if n_pron_bins:
            self.p["pron"] = np.zeros((nlab, n_pron_bins))
        # ⚠ DEPENDENT-SIDE LEMMA VECTOR -- the symmetric counterpart of `lemvec` (which reads only
        # the HEAD's static vector). `lemvec` answers "does this GOVERNING word select the
        # relation" (la's matrix-verb-style valency question); this answers "does this DEPENDENT
        # word, by its own distributional profile, look oblique-like or modifier-like" -- a
        # genuinely different question, built for la's mod/comp:obl trade surviving every
        # STRUCTURAL bias tried so far (agreement/direction/pos/feat all reshuffle which of the two
        # wins rather than resolving it). Head-independent (dlin2-shaped, like feat/morphhash), NOT
        # a duplicate of `lemvec`'s derivation (that one broadcasts over d, from dlin1) -- its own
        # claim to gradient-check, not assumed correct by the head-side analogy.
        if n_lemvec_dep_dim:
            self.p["lemvec_dep"] = np.zeros((nlab, n_lemvec_dep_dim))
        # ⚠ VERB-LEMMA x DEPENDENT-CASE INTERACTION -- every term above this one is PURELY
        # ADDITIVE (combined = ... + lemvec_term + feat_term + ...), and that is structurally why
        # none of them closed la's mod/comp:obl trade: checking actual counts, `dico` ("say") alone
        # governs 2332 mod + 1682 comp:obl tokens -- a ~58/42 split WITHIN ONE LEMMA, so no amount
        # of "which verb" signal (lemvec, additive) or "which case" signal (feat/morphhash,
        # additive) can express "dico takes mod when its dependent is ablative-of-means but obl
        # when accusative" -- an additive sum of the two never lets one MODULATE the other. This
        # term is genuinely bilinear: `lemvec[h] @ p["lemcase"][l] @ onehot(case[d])`, so the
        # dependent's Case bucket SELECTS which column of the head's lemma-vector readout applies,
        # per label. Reuses the SAME `lemvec` array the head-side term already reads (no new head-
        # side data) and a Case-only cousin of `feat_buckets` for the dependent side.
        if n_lemcase_dim and n_lemcase_bins:
            self.p["lemcase"] = np.zeros((nlab, n_lemcase_dim, n_lemcase_bins))
        # ⚠ DISCRETE HEAD-LEMMA IDENTITY -- the missing cell in this table. `lemvec` is head-side
        # but CONTINUOUS (a linear readout of a pretrained static vector); `morphhash`/`feat` are
        # DISCRETE but dependent-side. Built for lzh, where `lemcase` above cannot apply at all (no
        # Case morphology, same reason `--agreement` was excluded there) but the matrix-verb
        # closed-class signal (comp:obj: 曰謂使有無爲如以得知 alone cover 28.3%) is fundamentally
        # about WHICH VERB, and `lemvec`'s own nearest-neighbour check found the SikuBERT vector
        # space coarse/collocational rather than cleanly separating verb classes -- a discrete
        # identity hash sidesteps needing that vector space to have encoded valency at all.
        # Structurally: `lin1`/`lemvec`'s broadcast-over-d shape (dlin1-derived gradient), but a
        # HASHED BUCKET lookup instead of a continuous readout -- the same combination `morphhash`
        # is for the dependent side, crossed the other way.
        if n_lemhash_bins:
            self.p["lemhash"] = np.zeros((nlab, n_lemhash_bins))
        # ⚠ DISCRETE DEPENDENT-LEMMA IDENTITY -- `lemhash`'s mirror image, crossed the other way
        # again: `lemhash` is head-side discrete, this is DEPENDENT-side discrete (dlin2-shaped,
        # like morphhash/feat/lemvec_dep). Built after `lemcase` (verb x dependent-Case) measurably
        # FAILED to close la's mod/comp:obl trade -- checking whether the premise held at all found
        # 81.1% of the ambiguous mass has NO Case value (the dependent is an ADP/ADV/SCONJ, not a
        # bare case-marked noun -- Case lives one level down, on the PP's own object, which no la
        # bias yet reads). Within that Case-less 81%, the DEPENDENT's own lemma identity predicts
        # the label at 88.75% (per/si/secundum/non/sicut/nisi/sic are ~100% one label; cum/quia/ut
        # above 93%), dwarfing the ~67.5% lemma-blind baseline -- `lemvec_dep` already asks this
        # question but CONTINUOUSLY (a linear readout of a distributional vector); this asks it
        # DISCRETELY, the same swap `lemhash` made on the head side, motivated the same way (a
        # largely closed, near-deterministic class doesn't need a vector space to separate it).
        if n_lemhashdep_bins:
            self.p["lemhashdep"] = np.zeros((nlab, n_lemhashdep_bins))
        # ⚠ SECOND-ORDER SIBLING BIAS -- the FIRST term in this file whose bucket cannot be computed
        # from the doc alone (agreement/pos/feat/pron all need only token attributes and positions,
        # known before any decoding happens); this one needs a PREDICTED tree, since "who is this
        # candidate dependent's predicted sibling under this candidate head" is a property of a
        # DECODE, not of the input. The caller (train_arcfactored.py) runs the model's OWN forward
        # pass once WITHOUT this term, decodes it via CLE to get that first-order tree, computes
        # `sib_bkt[h, d]` from IT (bucket = 1 + the predicted immediate preceding same-head
        # dependent's label id, 0 if none), and only THEN calls forward/loss_and_backward AGAIN with
        # `sib_bkt` filled in -- the same "predicted, never gold" discipline this project already
        # applies to every other bucket (agreement_buckets/pos_buckets read PREDICTED morphology/
        # UPOS from upstream, never gold), extended to a bucket whose source is the model's OWN
        # first-order self, not an external tagger. Built after checking real gold counts:
        # `conj:coord -> punct` (a coordinated conjunct immediately followed by punctuation, same
        # head) at P=0.985 over 2,580 occurrences -- exactly the kind of joint constraint a
        # FIRST-ORDER score, which never looks at a head's OTHER dependents, cannot express.
        # A full (n+1, n) grid scatter exactly like `pron`'s (a property of the head x dependent
        # PAIR, via which head is picked), NOT head- or dependent-independent like feat/morphhash.
        if n_sib_bins:
            self.p["sib"] = np.zeros((nlab, n_sib_bins))
        # ⚠ SECOND-ORDER GRANDPARENT BIAS -- `sib`'s sibling axis, this one's the PARENT axis: the
        # LABEL of candidate head h's OWN incoming arc in a first-order pre-decode (the classic
        # McDonald & Pereira / Koo & Collins grandparent factor). Unlike `sib`, this does NOT vary
        # with the dependent d at all -- "how is h itself attached above" is a property of h alone,
        # so it is dlin1-shaped (broadcasts over d) exactly like `lemhash`, not a full (h, d) grid
        # like `sib`/`pron`/`pos`. Built for the SAME diagnosis `sib` was: `conj:coord` is the
        # single worst deprel (43.56 vs the transition parser's 57.79, a -14.23 gap) and the error
        # is concentrated in long chains -- a coordinated conjunct's OWN dependents (a further
        # conj:coord member continuing the chain, or a modifier of the coordinated phrase) plausibly
        # need different treatment depending on whether h ITSELF is already a conj:coord link versus
        # an ordinary subj/comp:obj/mod -- context no first-order score, and no SIBLING bias either
        # (which only looks at h's OTHER dependents, never at h's own attachment), can see. Computed
        # via the identical two-pass, predicted-never-gold discipline `sib` established: decode
        # first-order, read `labels0[h-1]` off that pre-decode via `grandparent_buckets()`, only
        # THEN rerun with the bias included -- reuses `sib`'s own first-order pass rather than a
        # separate one when both are enabled (see train_arcfactored.py).
        if n_grand_bins:
            self.p["grand"] = np.zeros((nlab, n_grand_bins))
        # ⚠ ARC-SIDE SELF-ATTENTION (--attn-hd), placed AFTER Wh/Wd's own per-token projection and
        # ReLU, NOT before them like the refuted `--attn` (applied to raw X -- see
        # NEGATIVE-RESULTS.md: a genuine regression, -0.87 vs --sibling, even below plain baseline).
        # Wh/Wd (and their ReLU) are exactly "the pieces that need INDIVIDUAL token information" --
        # each needs X's own clean, unmixed per-token content to decide THAT token's own arc-scoring
        # activation pattern; mixing X before those projections blurred the very identity signal
        # (agreement, lemma) the OTHER bias terms already read cleanly. This instead refines H and D
        # (the "am I a good candidate head/dependent" representations) with sentence-wide context
        # AFTER they are individually computed, via the SAME attn_forward/attn_backward math
        # (sud_self_attention.py) applied directly to H and D as (n, h) matrices. TWO INDEPENDENT
        # instances (head-side, dependent-side) since H and D live in different learned subspaces
        # (Wh vs Wd) -- no weight sharing assumed. Deliberately does NOT touch LH/LD (label-scoring):
        # diagnosis found label accuracy given a correct head already close to the transition
        # parser's (91-92%) -- labelling is not the diagnosed problem, long-distance ATTACHMENT is,
        # so only the arc-scoring pathway is touched. `Wo` zero-initialised for both instances, same
        # "starts as an exact no-op" discipline as every other additive term here.
        if use_attn_hd:
            self.p["attn_h_Wq"] = s(h, h); self.p["attn_h_Wk"] = s(h, h); self.p["attn_h_Wv"] = s(h, h)
            self.p["attn_h_Wo"] = np.zeros((h, h))
            self.p["attn_d_Wq"] = s(h, h); self.p["attn_d_Wk"] = s(h, h); self.p["attn_d_Wv"] = s(h, h)
            self.p["attn_d_Wo"] = np.zeros((h, h))
        self.use_attn_hd = use_attn_hd
        self.h, self.nlab = h, nlab
        self.dist_buckets_fn = dist_buckets_fn
        self.dir_buckets_fn = dir_buckets_fn

    def forward(self, X, k, agree_bkt=None, pos_bkt=None, lemvec=None, morph_bkt=None,
                feat_bkt=None, pron_bkt=None, lemvec_dep=None, lemcase_bkt=None,
                lemhash_bkt=None, lemhashdep_bkt=None, sib_bkt=None, grand_bkt=None):
        p = self.p
        n = X.shape[0]
        H0 = np.maximum(X @ p["Wh"] + p["bh"], 0)
        D0 = np.maximum(X @ p["Wd"] + p["bd"], 0)
        # ⚠ ATTENTION RUNS ON H0/D0 (post-Wh/Wd, post-ReLU -- each token's OWN clean arc-scoring
        # representation, already decided from unmixed X), never on X itself -- see __init__'s note
        # on why the refuted `--attn` placement (pre-projection, on X) regressed. H/D from here on
        # are the (possibly attention-refined) representations arc_raw actually uses; H0/D0 are kept
        # in the cache so backward can apply the ReLU gate at the right point (H0>0, not H>0 --
        # attention's own output has no reason to share the original ReLU's zero pattern).
        attn_h_cache = attn_d_cache = None
        if self.use_attn_hd:
            H, attn_h_cache = attn_forward(H0, p["attn_h_Wq"], p["attn_h_Wk"], p["attn_h_Wv"],
                                            p["attn_h_Wo"])
            D, attn_d_cache = attn_forward(D0, p["attn_d_Wq"], p["attn_d_Wk"], p["attn_d_Wv"],
                                            p["attn_d_Wo"])
        else:
            H, D = H0, D0
        # ⚠ DTYPE MUST FOLLOW X (float32 in production, float64 in this module's own gradient
        # check) -- a bare `np.zeros(...)` defaults to float64 and silently upcasts everything
        # vstacked with it, which thinc's optimiser then rejects (float32 buffer expected).
        Hr = np.vstack([np.zeros((1, self.h), dtype=H.dtype), H])
        LH = np.maximum(X @ p["Lh"], 0)
        LD = np.maximum(X @ p["Ld"], 0)
        LHr = np.vstack([np.zeros((1, self.h), dtype=LH.dtype), LH])

        arc_raw = (Hr @ p["U"]) @ D.T + (p["u"] @ D.T)[None, :]              # (n+1, n)

        bil = np.einsum("hg,lgk,dk->hdl", LHr, p["V"], LD, optimize=True)                  # (n+1, n, nlab)
        lin1 = LHr @ p["v"][:, :self.h].T                                    # (n+1, nlab)
        lin2 = LD @ p["v"][:, self.h:].T                                     # (n, nlab)
        label_raw = bil + lin1[:, None, :] + lin2[None, :, :] + p["cb"][None, None, :]

        bkt = self.dist_buckets_fn(n, k)                                     # (n+1, n)
        dist_term = p["dist"].T[bkt]                                         # (n+1, n, nlab)

        combined = arc_raw[:, :, None] + label_raw + dist_term
        # ⚠ SAME MECHANISM AS THE DISTANCE BIAS, A DIFFERENT QUESTION. dist_by_label answers "how
        # far", agree_by_label answers "do these two forms AGREE" (Case/Number/Gender concord) --
        # the signal `mod`'s errors showed the arc scorer needed and didn't have: its dominant
        # error is picking a same-DISTANCE wrong candidate, which no distance term can discriminate.
        if agree_bkt is not None and "agree" in p:
            combined = combined + p["agree"].T[agree_bkt]
        if pos_bkt is not None and "pos" in p:
            combined = combined + p["pos"].T[pos_bkt]
        if lemvec is not None and "lemvec" in p:
            lemvec_term = lemvec @ p["lemvec"].T                             # (n+1, nlab)
            combined = combined + lemvec_term[:, None, :]
        if morph_bkt is not None and "morphhash" in p:
            morph_term = p["morphhash"].T[morph_bkt]                        # (n, nlab)
            combined = combined + morph_term[None, :, :]
        if feat_bkt is not None:
            for name in self.feat_names:
                key = f"feat_{name}"
                if name in feat_bkt and key in p:
                    feat_term = p[key].T[feat_bkt[name]]                    # (n, nlab)
                    combined = combined + feat_term[None, :, :]
        if pron_bkt is not None and "pron" in p:
            combined = combined + p["pron"].T[pron_bkt]                    # (n+1, n, nlab)
        if lemvec_dep is not None and "lemvec_dep" in p:
            depvec_term = lemvec_dep @ p["lemvec_dep"].T                   # (n, nlab)
            combined = combined + depvec_term[None, :, :]
        if lemvec is not None and lemcase_bkt is not None and "lemcase" in p:
            # M[h, l, c] = sum_k lemvec[h, k] * p["lemcase"][l, k, c] -- every (head, label,
            # case-value) score, computed ONCE; then each dependent just GATHERS the column for
            # its own Case bucket. `optimize=True`: this is exactly the 3-tensor contraction that
            # cost ~100x slowdown elsewhere in this file before it was added.
            M = np.einsum("hk,lkc->hlc", lemvec, p["lemcase"], optimize=True)   # (n+1, nlab, n_case)
            lemcase_term = M[:, :, lemcase_bkt].transpose(0, 2, 1)              # (n+1, n, nlab)
            combined = combined + lemcase_term
        if lemhash_bkt is not None and "lemhash" in p:
            lemhash_term = p["lemhash"][:, lemhash_bkt].T                      # (n+1, nlab)
            combined = combined + lemhash_term[:, None, :]
        if lemhashdep_bkt is not None and "lemhashdep" in p:
            lemhashdep_term = p["lemhashdep"].T[lemhashdep_bkt]                # (n, nlab)
            combined = combined + lemhashdep_term[None, :, :]
        if sib_bkt is not None and "sib" in p:
            combined = combined + p["sib"].T[sib_bkt]                          # (n+1, n, nlab)
        if grand_bkt is not None and "grand" in p:
            grand_term = p["grand"][:, grand_bkt].T                            # (n+1, nlab)
            combined = combined + grand_term[:, None, :]
        dbkt = None
        if self.dir_buckets_fn is not None and "direction" in p:
            dbkt = self.dir_buckets_fn(n, k)                                  # (n+1, n)
            combined = combined + p["direction"].T[dbkt]
        mask = window_mask(n, k)                                             # (n+1, n)
        combined = np.where(mask[:, :, None], combined, NEG)
        cache = dict(X=X, H=H, D=D, H0=H0, D0=D0, attn_h_cache=attn_h_cache,
                     attn_d_cache=attn_d_cache, Hr=Hr, LH=LH, LD=LD, LHr=LHr, mask=mask, bkt=bkt,
                     agree_bkt=agree_bkt, dbkt=dbkt, n=n)
        return combined, cache

    def loss_and_backward(self, X, k, gold_h, gold_l, agree_bkt=None, pos_bkt=None, lemvec=None,
                           morph_bkt=None, feat_bkt=None, pron_bkt=None, lemvec_dep=None,
                           lemcase_bkt=None, lemhash_bkt=None, lemhashdep_bkt=None, sib_bkt=None,
                           grand_bkt=None):
        """gold_h: (n,) int in [0,n] (0 = virtual root). gold_l: (n,) int label id.
        Returns (loss, grads_dict, dX). dX is the gradient wrt the ENCODER's output (for --joint)."""
        combined, c = self.forward(X, k, agree_bkt, pos_bkt, lemvec, morph_bkt, feat_bkt, pron_bkt,
                                    lemvec_dep, lemcase_bkt, lemhash_bkt, lemhashdep_bkt, sib_bkt,
                                    grand_bkt)
        n, nlab = c["n"], self.nlab
        # joint softmax over (h, l) per dependent d
        Z = combined.transpose(1, 0, 2).reshape(n, -1)                       # (n, (n+1)*nlab)
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        gold_flat = gold_h * nlab + gold_l
        loss = -np.log(np.maximum(P[np.arange(n), gold_flat], 1e-12)).sum()
        dZ = P.copy(); dZ[np.arange(n), gold_flat] -= 1.0
        dCombined = dZ.reshape(n, n + 1, nlab).transpose(1, 0, 2)             # (n+1, n, nlab)
        dCombined = np.where(c["mask"][:, :, None], dCombined, 0.0)

        p = self.p
        H, D, Hr, LH, LD, LHr = c["H"], c["D"], c["Hr"], c["LH"], c["LD"], c["LHr"]
        H0, D0 = c["H0"], c["D0"]

        d_arc_raw = dCombined.sum(-1)                                        # (n+1, n)
        d_label_raw = dCombined                                              # (n+1, n, nlab)

        g = {}
        # -- dist_by_label: scatter over buckets, per label --
        g["dist"] = np.zeros_like(p["dist"])                                 # (nlab, n_bins)
        flat_bkt = c["bkt"].ravel()
        flat_dC = dCombined.reshape(-1, nlab)
        scat = np.zeros((p["dist"].shape[1], nlab), dtype=p["dist"].dtype)
        np.add.at(scat, flat_bkt, flat_dC)
        g["dist"] = scat.T

        # -- agree_by_label: identical scatter, only when the term was actually used --
        if agree_bkt is not None and "agree" in p:
            flat_abkt = agree_bkt.ravel()
            ascat = np.zeros((p["agree"].shape[1], nlab), dtype=p["agree"].dtype)
            np.add.at(ascat, flat_abkt, flat_dC)
            g["agree"] = ascat.T

        # -- direction_by_label: identical scatter again -- four independent bias tables, four
        # independent gates, deliberately not sharing a code path with each other.
        if c["dbkt"] is not None and "direction" in p:
            flat_dbkt = c["dbkt"].ravel()
            dscat = np.zeros((p["direction"].shape[1], nlab), dtype=p["direction"].dtype)
            np.add.at(dscat, flat_dbkt, flat_dC)
            g["direction"] = dscat.T

        # -- pos_by_label: identical scatter, only when the term was actually used --
        if pos_bkt is not None and "pos" in p:
            flat_pbkt = pos_bkt.ravel()
            pscat = np.zeros((p["pos"].shape[1], nlab), dtype=p["pos"].dtype)
            np.add.at(pscat, flat_pbkt, flat_dC)
            g["pos"] = pscat.T

        # -- arc_raw backward (identical shape to the two-stage arc_scores backward) --
        HW = Hr @ p["U"]
        dHW = d_arc_raw @ D
        g["U"] = Hr.T @ dHW
        g["u"] = d_arc_raw.sum(0) @ D
        dD = d_arc_raw.T @ HW + np.outer(d_arc_raw.sum(0), p["u"])
        dHr_arc = dHW @ p["U"].T
        dH = dHr_arc[1:]

        # -- label_raw backward --
        g["cb"] = d_label_raw.sum(axis=(0, 1))
        dlin1 = d_label_raw.sum(axis=1)                                      # (n+1, nlab)
        dlin2 = d_label_raw.sum(axis=0)                                      # (n, nlab)

        # -- lemvec_by_label: SAME reduction as dlin1 (broadcasts over d identically), a DIFFERENT
        # input matrix -- no gradient flows into `lemvec` itself, a fixed pretrained lookup.
        if lemvec is not None and "lemvec" in p:
            g["lemvec"] = dlin1.T @ lemvec                                   # (nlab, dim)

        # -- morphhash_by_label: SAME reduction as dlin2 (broadcasts over h identically, the
        # dependent-only side) -- a discrete scatter this time, not a linear readout, so it looks
        # like `dist`'s backward with dlin2 standing in for the usual `flat_dC`.
        if morph_bkt is not None and "morphhash" in p:
            mscat = np.zeros((p["morphhash"].shape[1], nlab), dtype=p["morphhash"].dtype)
            np.add.at(mscat, morph_bkt, dlin2)
            g["morphhash"] = mscat.T

        # -- feat_<name>_by_label: same reduction as morphhash's, once per configured feature --
        # independent tables, independent gates, exactly the pattern agree/direction/pos already
        # established for "N independent bias terms sharing a scatter-add shape".
        if feat_bkt is not None:
            for name in self.feat_names:
                key = f"feat_{name}"
                if name in feat_bkt and key in p:
                    fscat = np.zeros((p[key].shape[1], nlab), dtype=p[key].dtype)
                    np.add.at(fscat, feat_bkt[name], dlin2)
                    g[key] = fscat.T

        # -- lemvec_dep_by_label: SAME reduction as dlin2 (broadcasts over h identically) -- the
        # dependent-side counterpart of lemvec's dlin1-shaped gradient. No gradient flows into
        # `lemvec_dep` itself, a fixed pretrained lookup, same as `lemvec`.
        if lemvec_dep is not None and "lemvec_dep" in p:
            g["lemvec_dep"] = dlin2.T @ lemvec_dep                           # (nlab, dim)

        # -- lemcase_by_label: genuinely bilinear, so neither dlin1's nor dlin2's reduction alone
        # applies -- this term depends on h, l AND d (via d's Case bucket), so it needs the FULL
        # (n+1, n, nlab) gradient (d_label_raw), scatter-added by CASE BUCKET along the d axis
        # (grouping dependents that share a Case value) before contracting away the head axis.
        if lemvec is not None and lemcase_bkt is not None and "lemcase" in p:
            n_case = p["lemcase"].shape[2]
            dM = np.zeros((n + 1, nlab, n_case), dtype=p["lemcase"].dtype)   # (n+1, nlab, n_case)
            dC_t = d_label_raw.transpose(0, 2, 1)                            # (n+1, nlab, n)
            np.add.at(dM, (slice(None), slice(None), lemcase_bkt), dC_t)
            g["lemcase"] = np.einsum("hk,hlc->lkc", lemvec, dM, optimize=True)  # (nlab, dim, n_case)

        # -- lemhash_by_label: SAME reduction as lemvec's (dlin1, broadcasts over d identically) --
        # a discrete scatter this time, not a continuous readout, so it looks like morphhash's
        # backward with dlin1 standing in for dlin2 (head-side, not dependent-side).
        if lemhash_bkt is not None and "lemhash" in p:
            hscat = np.zeros((p["lemhash"].shape[1], nlab), dtype=p["lemhash"].dtype)
            np.add.at(hscat, lemhash_bkt, dlin1)
            g["lemhash"] = hscat.T

        # -- lemhashdep_by_label: SAME reduction as morphhash's (dlin2, broadcasts over h
        # identically) -- lemhash's mirror image again, this time on the dependent side.
        if lemhashdep_bkt is not None and "lemhashdep" in p:
            hdscat = np.zeros((p["lemhashdep"].shape[1], nlab), dtype=p["lemhashdep"].dtype)
            np.add.at(hdscat, lemhashdep_bkt, dlin2)
            g["lemhashdep"] = hdscat.T

        # -- sib_by_label: a FULL (n+1, n) grid scatter, same reduction as pron's -- the bucket
        # itself is externally computed from a first-order pre-decode (see the __init__ note), but
        # to THIS function it is just another opaque precomputed (n+1, n) grid, no different from
        # pron_bkt's own shape/reduction contract.
        if sib_bkt is not None and "sib" in p:
            flat_sibkt = sib_bkt.ravel()
            sibscat = np.zeros((p["sib"].shape[1], nlab), dtype=p["sib"].dtype)
            np.add.at(sibscat, flat_sibkt, flat_dC)
            g["sib"] = sibscat.T

        # -- grand_by_label: SAME reduction as lemhash's (dlin1, broadcasts over d identically) --
        # a discrete HEAD-side scatter like lemhash, but the bucket's SOURCE is a first-order
        # pre-decode (like sib's), not the doc alone.
        if grand_bkt is not None and "grand" in p:
            grandscat = np.zeros((p["grand"].shape[1], nlab), dtype=p["grand"].dtype)
            np.add.at(grandscat, grand_bkt, dlin1)
            g["grand"] = grandscat.T

        # -- pron_by_label: a FULL (n+1, n) grid scatter like pos/agree/direction, not a
        # head-independent one like feat/morphhash -- the preverbal-pronoun signal is a property
        # of the (head, dependent) PAIR (does the head follow this dependent), not of the
        # dependent alone.
        if pron_bkt is not None and "pron" in p:
            flat_prbkt = pron_bkt.ravel()
            prscat = np.zeros((p["pron"].shape[1], nlab), dtype=p["pron"].dtype)
            np.add.at(prscat, flat_prbkt, flat_dC)
            g["pron"] = prscat.T
        g["V"] = np.einsum("hdl,hg,dk->lgk", d_label_raw, LHr, LD, optimize=True)
        dLHr_bil = np.einsum("hdl,lgk,dk->hg", d_label_raw, p["V"], LD, optimize=True)
        dLD_bil = np.einsum("hdl,lgk,hg->dk", d_label_raw, p["V"], LHr, optimize=True)
        dv1 = dlin1.T @ LHr                                                   # (nlab, h)
        dv2 = dlin2.T @ LD                                                    # (nlab, h)
        g["v"] = np.concatenate([dv1, dv2], axis=1)
        dLHr_lin = dlin1 @ p["v"][:, :self.h]
        dLD_lin = dlin2 @ p["v"][:, self.h:]
        dLHr = dLHr_bil + dLHr_lin
        dLD = dLD_bil + dLD_lin
        dLH = dLHr[1:]

        # -- attn_h/attn_d: `dH`/`dD` above are the gradient wrt H/D as arc_raw actually used them --
        # the ATTENTION-REFINED representations when --attn-hd is on. Route through attn_backward
        # FIRST to get dH0/dD0 (gradient wrt the pre-attention, post-ReLU H0/D0), so the ReLU gate
        # just below is evaluated at the point the ReLU actually ran (H0/D0's own zero pattern, not
        # attention's output -- which has no reason to share it).
        if self.use_attn_hd:
            g_ah, dH0 = attn_backward(dH, c["attn_h_cache"], p["attn_h_Wq"], p["attn_h_Wk"],
                                       p["attn_h_Wv"], p["attn_h_Wo"])
            g["attn_h_Wq"], g["attn_h_Wk"] = g_ah["Wq"], g_ah["Wk"]
            g["attn_h_Wv"], g["attn_h_Wo"] = g_ah["Wv"], g_ah["Wo"]
            g_ad, dD0 = attn_backward(dD, c["attn_d_cache"], p["attn_d_Wq"], p["attn_d_Wk"],
                                       p["attn_d_Wv"], p["attn_d_Wo"])
            g["attn_d_Wq"], g["attn_d_Wk"] = g_ad["Wq"], g_ad["Wk"]
            g["attn_d_Wv"], g["attn_d_Wo"] = g_ad["Wv"], g_ad["Wo"]
        else:
            dH0, dD0 = dH, dD

        g["Wh"] = X.T @ (dH0 * (H0 > 0)); g["bh"] = (dH0 * (H0 > 0)).sum(0)
        g["Wd"] = X.T @ (dD0 * (D0 > 0)); g["bd"] = (dD0 * (D0 > 0)).sum(0)
        g["Lh"] = X.T @ (dLH * (LH > 0))
        g["Ld"] = X.T @ (dLD * (LD > 0))

        dX = ((dH0 * (H0 > 0)) @ p["Wh"].T + (dD0 * (D0 > 0)) @ p["Wd"].T
              + (dLH * (LH > 0)) @ p["Lh"].T + (dLD * (LD > 0)) @ p["Ld"].T)
        return loss, g, dX

    def decode_scores(self, X, k, agree_bkt=None, pos_bkt=None, lemvec=None, morph_bkt=None,
                       feat_bkt=None, pron_bkt=None, lemvec_dep=None, lemcase_bkt=None,
                       lemhash_bkt=None, lemhashdep_bkt=None, sib_bkt=None, grand_bkt=None):
        """For eval: returns (best_label_score[h,d], chosen_label[h,d]) -- best_label_score feeds
        `sud_cle.mst` exactly like the two-stage scorer's `S`; chosen_label is read off for
        whichever arc CLE actually selects."""
        combined, _ = self.forward(X, k, agree_bkt, pos_bkt, lemvec, morph_bkt, feat_bkt, pron_bkt,
                                    lemvec_dep, lemcase_bkt, lemhash_bkt, lemhashdep_bkt, sib_bkt,
                                    grand_bkt)
        return combined.max(-1), combined.argmax(-1)


def _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_feat=False, use_pron=False,
                         use_lemvec_dep=False, use_lemcase=False, use_lemhash=False,
                         use_lemhashdep=False, use_sib=False, use_grand=False, use_attn_hd=False,
                         seed=0):
    """Tiny random example, no windowing edge cases (k large enough to matter, small n) -- finite
    differences against every parameter AND against X (needed for --joint's encoder backprop).
    `use_agree=True` also exercises the agreement-bias path (a second, independent scatter-add
    term structurally identical to `dist`, but wired through its own cache key and its own
    `agree_bkt is not None and "agree" in p` gate -- worth checking on its own, not assumed correct
    by analogy). `use_dir=True` exercises the THIRD, direction-bias path the same way. `use_pos=True`
    exercises the FOURTH, POS-compatibility path -- threaded like agreement (an argument, not a
    buckets_fn), so it needs its own check rather than inheriting direction's. `use_lemvec=True`
    exercises the FIFTH path, the only CONTINUOUS one (a linear readout of a fixed input, not a
    bucket lookup) -- its gradient formula was DERIVED BY ANALOGY to dlin1's (both broadcast over
    the dependent axis identically), which is exactly the kind of claim this check exists to catch
    if wrong, not to assume. `use_morphhash=True` exercises the SIXTH path, a discrete scatter like
    `pos` but head-INDEPENDENT (derived by analogy to dlin2 the way lemvec was derived from dlin1) --
    its own, separate claim to verify. `use_feat=True` exercises the SEVENTH path -- TWO independent
    per-feature tables at once (not just one), to catch a bug that would only show up with more than
    a single feature configured (e.g. writing into the wrong table by name). `use_pron=True`
    exercises the EIGHTH path, a full (h, d)-grid scatter like `pos`'s, not head-independent like
    `feat`'s -- its own claim, since it looks like `pos` but isn't derived from it. `use_lemvec_dep`
    exercises the NINTH path, the dependent-side counterpart of lemvec -- looks like feat/morphhash
    (head-independent, dlin2-shaped) but is a CONTINUOUS linear readout like lemvec, not a discrete
    scatter, so it inherits neither derivation cleanly and needs its own check. `use_lemcase=True`
    exercises the TENTH path, the only genuinely BILINEAR term (head lemma vector x dependent Case
    bucket) -- its gradient needs the FULL (n+1,n,nlab) tensor, unlike every additive term above,
    so it shares no derivation with any of them and needs `lemvec` supplied even when
    `use_lemvec` (the separate, additive head-only term) is off. `use_lemhash=True` exercises the
    ELEVENTH path, a discrete HEAD-side scatter (dlin1-shaped, like lemvec's derivation) -- the
    missing cell crossing morphhash's "discrete" with lemvec's "head-side", built for lzh where
    `lemcase` cannot apply (no Case morphology) but a head-lemma-IDENTITY signal still can.
    `use_lemhashdep=True` exercises the TWELFTH path, `lemhash`'s mirror image on the dependent
    side (dlin2-shaped, like morphhash's derivation) -- built for la after `lemcase` failed to
    close the mod/comp:obl trade: the dependent's own lemma identity (which preposition/adverb/
    subordinator), not the verb's Case government, turned out to carry the real signal.
    `use_sib=True` exercises the THIRTEENTH path, the second-order sibling bias -- a full (h,d)-grid
    scatter like `pron`'s, so its OWN gradient derivation is identical to an already-verified one,
    but it is the FIRST bucket in this file whose real-world source is a first-order pre-decode
    rather than the doc alone -- worth its own check since nothing else exercises a bucket array
    this shape being handed in from outside without also passing agree/pos/pron's OWN buckets_fn
    convention. `use_grand=True` exercises the FOURTEENTH path, the second-order grandparent bias --
    dlin1-shaped (broadcasts over d) like `lemhash`'s derivation, but like `sib` its bucket comes
    from a first-order pre-decode, not the doc alone; worth its own check since it combines a
    derivation from one already-verified term (lemhash's) with a data source from another
    (sib's), and nothing guarantees that combination is bug-free just because both halves are.
    `use_attn_hd=True` exercises the FIFTEENTH path, arc-side self-attention on H/D -- unlike
    every path above, this needs NO new forward/loss_and_backward ARGUMENT at all (it is a pure
    construction-time architectural flag, not per-call data), so this check needs no new bucket
    array either -- the existing generic `for key in m.p` loop below already covers its new
    attn_h_*/attn_d_* keys automatically. Its own claim: that threading attn_forward/attn_backward
    at a DIFFERENT point (H/D, not X) than sud_self_attention.py's own standalone check exercises
    (raw X) is ALSO correct, including the (H0>0) ReLU-gate interaction with attention's output."""
    rng = np.random.default_rng(seed)
    n, w, h, nlab, nbins, k = 5, 6, 4, 3, 4, 10
    nabins, ndbins, npbins, nlvdim, nmbins, nprbins, nlvddim = 5, 3, 7, 6, 8, 4, 5
    nlcbins = 5   # lemcase reuses nlvdim for its head-vector dim, like the real usage does
    nhhbins = 9
    nhdbins = 11
    nsibbins = 4   # nlab + 1 in real usage; kept small here like every other tiny test dimension
    ngpbins = 6    # nlab + 1 in real usage, like nsibbins
    feat_bins = {"f1": 5, "f2": 3} if use_feat else None

    def buckets_fn(n_, k_):
        # a tiny stand-in distance-bucket function: bucket = min(|delta|, nbins-1), same for both
        # directions (doesn't need to match the real one -- only used to exercise the scatter code)
        hh = np.arange(n_ + 1)[:, None] - 1
        dd = np.arange(n_)[None, :]
        return np.minimum(np.abs(hh - dd), nbins - 1)

    def dir_buckets_fn(n_, k_):
        hh = np.arange(n_ + 1)[:, None] - 1
        dd = np.arange(n_)[None, :]
        delta = hh - dd
        b = np.where(delta < 0, 1, np.where(delta > 0, 2, 0))
        b[0, :] = 0
        return b

    m = JointBiaffine(w, h, nlab, nbins, buckets_fn, n_agree_bins=(nabins if use_agree else 0),
                       n_dir_bins=(ndbins if use_dir else 0),
                       dir_buckets_fn=(dir_buckets_fn if use_dir else None),
                       n_pos_bins=(npbins if use_pos else 0),
                       n_lemvec_dim=(nlvdim if use_lemvec else 0),
                       n_morphhash_bins=(nmbins if use_morphhash else 0),
                       feat_bins=feat_bins,
                       n_pron_bins=(nprbins if use_pron else 0),
                       n_lemvec_dep_dim=(nlvddim if use_lemvec_dep else 0),
                       n_lemcase_dim=(nlvdim if use_lemcase else 0),
                       n_lemcase_bins=(nlcbins if use_lemcase else 0),
                       n_lemhash_bins=(nhhbins if use_lemhash else 0),
                       n_lemhashdep_bins=(nhdbins if use_lemhashdep else 0),
                       n_sib_bins=(nsibbins if use_sib else 0),
                       n_grand_bins=(ngpbins if use_grand else 0),
                       use_attn_hd=use_attn_hd, seed=seed + 1)
    for key in m.p:
        m.p[key] = rng.normal(size=m.p[key].shape) * 0.5
    X = rng.normal(size=(n, w))
    # gold heads must be VALID candidates (no self-loop: head != dependent+1), or the "gold"
    # position is itself masked to NEG and the check degenerates -- not a real input.
    gold_h = np.array([rng.choice([h_ for h_ in range(n + 1) if h_ != d + 1]) for d in range(n)])
    gold_l = rng.integers(0, nlab, size=n)
    agree_bkt = rng.integers(0, nabins, size=(n + 1, n)) if use_agree else None
    pos_bkt = rng.integers(0, npbins, size=(n + 1, n)) if use_pos else None
    # `lemvec` is shared by BOTH the additive head-only term (use_lemvec) and the bilinear
    # lemcase term (use_lemcase) -- exactly as the real usage shares one lemma-vector lookup.
    lemvec = rng.normal(size=(n + 1, nlvdim)) if (use_lemvec or use_lemcase) else None
    morph_bkt = rng.integers(0, nmbins, size=n) if use_morphhash else None
    feat_bkt = ({name: rng.integers(0, nb, size=n) for name, nb in (feat_bins or {}).items()}
                if use_feat else None)
    pron_bkt = rng.integers(0, nprbins, size=(n + 1, n)) if use_pron else None
    lemvec_dep = rng.normal(size=(n, nlvddim)) if use_lemvec_dep else None
    lemcase_bkt = rng.integers(0, nlcbins, size=n) if use_lemcase else None
    lemhash_bkt = rng.integers(0, nhhbins, size=n + 1) if use_lemhash else None
    lemhashdep_bkt = rng.integers(0, nhdbins, size=n) if use_lemhashdep else None
    sib_bkt = rng.integers(0, nsibbins, size=(n + 1, n)) if use_sib else None
    grand_bkt = rng.integers(0, ngpbins, size=n + 1) if use_grand else None

    loss0, g, dX = m.loss_and_backward(X, k, gold_h, gold_l, agree_bkt, pos_bkt, lemvec, morph_bkt,
                                        feat_bkt, pron_bkt, lemvec_dep, lemcase_bkt, lemhash_bkt,
                                        lemhashdep_bkt, sib_bkt, grand_bkt)

    def loss_at(Xp):
        combined, _ = m.forward(Xp, k, agree_bkt, pos_bkt, lemvec, morph_bkt, feat_bkt, pron_bkt,
                                 lemvec_dep, lemcase_bkt, lemhash_bkt, lemhashdep_bkt, sib_bkt,
                                 grand_bkt)
        Z = combined.transpose(1, 0, 2).reshape(n, -1)
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        gold_flat = gold_h * nlab + gold_l
        return -np.log(np.maximum(P[np.arange(n), gold_flat], 1e-12)).sum()

    eps = 1e-5
    worst = 0.0
    worst_key = None; worst_idx = None
    for key in m.p:
        arr = m.p[key]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps
            lp = loss_at(X)
            arr[idx] = old - eps
            lm = loss_at(X)
            arr[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = g[key][idx]
            # ⚠ FLOOR RAISED 1e-6 -> 1e-5 after --attn-hd's own combined check (seed=2 in the loop
            # below) failed at 7.11e-4 -- dumped and inspected (GRADCHECK_DUMP) rather than assumed
            # benign: the worst entry was p["u"] with num=-7.1e-10, ana=2.4e-16, BOTH ~4-5 orders of
            # magnitude below even the OLD floor -- an entry whose true gradient is analytically
            # zero, where the "error" is pure float64 rounding noise turned into a large RELATIVE
            # number only because the fixed floor sat far above the noise it was meant to bound.
            # Not attn_hd-specific (V/agree/lemcase/lemhash/grand show the identical near-zero-floor
            # pattern in the same dump, just below the old floor already) -- stacking MORE terms
            # (now 14) simply raises the chance ANY one multi-seed draw lands a near-exact-zero
            # analytic gradient somewhere. A real bug would show a LARGE relative error on an entry
            # of NON-negligible size, which this floor change cannot mask (production training casts
            # to float32 throughout, whose own precision floor is already coarser than 1e-5).
            denom = max(abs(num), abs(ana), 1e-5)
            rel = abs(num - ana) / denom
            if rel > worst:
                worst = rel; worst_key = key; worst_idx = idx
    # also check dX
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + eps
        lp = loss_at(X)
        X[idx] = old - eps
        lm = loss_at(X)
        X[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = dX[idx]
        denom = max(abs(num), abs(ana), 1e-8)
        rel = abs(num - ana) / denom
        if rel > worst:
            worst = rel; worst_key = "dX"; worst_idx = idx
    # ⚠ WHICH ENTRY, NOT JUST HOW BAD -- opt-in via GRADCHECK_DEBUG=1, since a bare "worst relative
    # error: X.Xe-YY" told nothing about WHERE to look when this check once failed (see the floor
    # comment above): a genuine bug and a near-zero-gradient floor artifact are indistinguishable
    # from the summary number alone, but instantly distinguishable once you know which key/index.
    import os
    if os.environ.get("GRADCHECK_DEBUG"):
        print(f"    worst entry: key={worst_key} idx={worst_idx} rel={worst:.2e}")
    tag = ("+".join([n for n, on in [("agree", use_agree), ("dir", use_dir), ("pos", use_pos),
                                       ("lemvec", use_lemvec), ("morphhash", use_morphhash),
                                       ("feat", use_feat), ("pron", use_pron),
                                       ("lemvec_dep", use_lemvec_dep), ("lemcase", use_lemcase),
                                       ("lemhash", use_lemhash), ("lemhashdep", use_lemhashdep),
                                       ("sib", use_sib), ("grand", use_grand),
                                       ("attn_hd", use_attn_hd)]
                      if on]) or "plain")
    print(f"[{tag}] worst relative error: {worst:.2e}  (loss0={loss0:.4f})")
    # ⚠ 1e-4 is too strict for a ReLU network: across seeds the worst entry lands EITHER at
    # machine precision (~1e-7) OR at ~1.8e-4 with no in-between, the fingerprint of a finite
    # difference straddling a ReLU's non-smooth point (not a scaling bug, which would move with the
    # gradient's own magnitude). `sud_biaffine.py`'s own verified bar was 2.3e-04; match it.
    assert worst < 5e-4, f"gradient check FAILED ({tag})"
    print(f"[{tag}] gradient check PASSED")


if __name__ == "__main__":
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False)
    _numeric_grad_check(use_agree=True, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False)
    _numeric_grad_check(use_agree=False, use_dir=True, use_pos=False, use_lemvec=False,
                         use_morphhash=False)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=True, use_lemvec=False,
                         use_morphhash=False)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=True,
                         use_morphhash=False)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_feat=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_pron=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_lemvec_dep=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_lemcase=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_lemhash=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_lemhashdep=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_sib=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_grand=True)
    _numeric_grad_check(use_agree=False, use_dir=False, use_pos=False, use_lemvec=False,
                         use_morphhash=False, use_attn_hd=True)
    _numeric_grad_check(use_agree=True, use_dir=True, use_pos=True, use_lemvec=True,
                         use_morphhash=True, use_feat=True, use_pron=True, use_lemvec_dep=True,
                         use_lemcase=True, use_lemhash=True, use_lemhashdep=True, use_sib=True,
                         use_grand=True, use_attn_hd=True)
    for s in range(1, 6):
        _numeric_grad_check(use_agree=True, use_dir=True, use_pos=True, use_lemvec=True,
                             use_morphhash=True, use_feat=True, use_pron=True,
                             use_lemvec_dep=True, use_lemcase=True, use_lemhash=True,
                             use_lemhashdep=True, use_sib=True, use_grand=True, use_attn_hd=True,
                             seed=s)
