# The generic parser v3: a lexical channel you can fill with an English gloss

> **Status: three seeds, complete.** Deltas are computed per seed against that seed's OWN baseline,
> so baseline drift cannot leak into an arm. Ranges are reported beside every mean, and the ones
> that swallow their effect are named as such.

v2 reads UPOS + FEATS + a typological profile and no wordform at all. v3 adds back **one lexical
channel**, and changes what fills it.

|  | v1 (`docs/generic-parser-v1.md`) | v3 |
|---|---|---|
| key | surface FORM | **LEMMA** |
| filled at deployment from | the language's own vector table | **an English gloss the user writes** |
| a language with no fastText | channel is empty — dead | channel fills from English |

That last row is the whole idea. Aligned fastText vectors are rotated into the **English hub**, so an
English gloss and a source lemma are points in one space rather than two kinds of thing. Training
coverage therefore limits how well the channel is **learned**, not where it can be **used**:
Chintang, K'iche' and Xavante have no fastText at all and can still fill it.

## The substitution was measured before anything was built

The arm rests on "a gloss vector stands in for a lemma vector", which is an assumption about an
input regime — and this repo has paid 4.83 F once for asking a model at inference for a regime it
never met (`CLAUDE.md` hazard 11). Measured first, on the two treebanks carrying both a `Gloss=`
column and an aligned table:

| | cos(source, en-gloss) | shuffled | random-en | beat shuffled |
|---|---|---|---|---|
| ar PADT, 236 k pairs | **+0.444** | +0.109 | +0.012 | 94.0 % |
| lzh Kyoto, 382 k pairs | **+0.340** | +0.116 | +0.022 | 84.5 % |

Re-measured end to end through the **built** table (lemma-keyed, with the Arabic fold): **+0.4597**
against +0.1115 shuffled, 95.4 % beating their partner, 239 748 pairs.

Well above a permuted control, and nowhere near 1.0. **Both halves of that sentence turned out to
matter** — see the results.

## Scope: the aligned-44

fastText publishes 44 languages already rotated into the English space, so they need no rotation
fitting at all (`route = "pre"`). Of v2's 80 training languages **32 are in that set** — 45 % of
training tokens; of the 20 held-out languages, **6** are. Extending to the other 48 means a
Procrustes fit per language against a bilingual dictionary, whose quality v1's report shows varying
from 0.62 to 0.07 hit@1. That is a separate decision and has not been taken.

Mean training token coverage of the lemma column: **92.5 %**, worst case 79.1 % (Norwegian, whose
misses are `$.` and `$,` punctuation lemmas and are correctly OOV).

## Four key folds, and three languages that would have trained on an empty channel

| | before | after | why |
|---|---|---|---|
| ar | 41.2 % | **96.2 %** | PADT lemmas are fully vocalised citation forms; fastText Arabic is not |
| ko | 36.4 % | **83.8 %** | `청+하+고` — morpheme-segmented lemmas against orthographic words |
| et | 82.6 % | 88.4 % | `maa_ilm` compound-boundary marks nobody writes |
| fi | 84.7 % | 88.6 % | the same, spelled `#` (`yli#opisto`) |

**The absent fold is the dangerous case, not a wrong one.** A 41 % channel trains, converges and
reports an ordinary loss curve; it is simply worse, and no metric in the sweep names the cause.

Two had an obvious fix that measurement rejected. **Korean: joining the morphemes is 16.7 points
worse than taking the first** (67.1 % against 83.8 %) — joining rebuilds an inflected surface string
rather than a lexeme, and Korean is stem-initial so the head morpheme is the lexical one.
**Arabic: the general-purpose NFD-then-drop-combining costs 2.0 points** against an explicit harakat
class (94.2 % against 96.2 %), because it also takes the hamza off `أ` and `إ` and folds both onto
`ا`, which fastText keeps distinct.

⚠ **Type coverage, which is what `align_vectors.py` printed, would have found none of this.** German
reads 28.5 % of TYPES and covers 90.2 % of TOKENS — the gap is the hapax-compound tail. The reports
carry token coverage now.

## The evaluation, and why one macro is the wrong number

**The group split is the whole analysis.** Six of the twenty held-out languages are in the
aligned-44, so for them the `lemma` fill finds REAL vectors — that run is the channel's UPPER BOUND.
For the other fourteen the identical run is all-OOV, its FLOOR. Averaged together the two cancel and
the channel looks inert. `compare_generic_v3.py` refuses to print one number for both.

**Two gloss keys, and only one is the deployment number.** `--gloss-key lemma` is an upper bound —
the v2 contract declares UPOS and `tb_lang` as user inputs and says nothing about lemmas.
`--gloss-key form` is what a deployer has. Report both; never quote one for the other.

**Three languages cannot be scored on the gloss fill by any route** — Bororo, Komi-Zyrian and
Xavante have neither a Wiktionary extract nor a `Gloss=` column. A macro "over the test languages"
that silently means seventeen is the kind of number this repo has had to retract.

**The Wiktionary figure understates a real glossing user**, which is the direction worth being wrong
in: mean fillable is 51.3 % by lemma and 37.0 % by form, while the three held-out languages carrying
a real gloss column run 67–81 %. Yoruba is the instructive case — 67.4 % from its own column against
15.3 % from Wiktionary, so the shortfall is the dictionary, not the language.

## Results (three seeds)

The baseline reproduces v2's published `g2_base`: **54.45 / 53.82 / 54.45** against **54.24**.

Held-in dev LAS: base 72.05, capacity control 71.88, shuffled 72.31, **`g3_vec` 73.72**,
`g3_vec_aug` 73.50, `g3_vec_aug2` 73.44. Both null channels sit within ±0.3 of the baseline and
every informative arm sits +1.4 to +1.7 above it.

Zero-shot, **six languages with real aligned rows**, delta vs baseline:

| arm / fill | s0 | s1 | s2 | **mean** | range |
|---|---|---|---|---|---|
| `g3_vec_ctl` / lemma | −0.96 | +0.43 | +1.56 | +0.34 | 2.52 |
| `g3_vec_shuf` / lemma | +1.92 | +3.69 | +2.70 | **+2.77** | 1.77 |
| `g3_vec` / lemma | +4.51 | +5.51 | +6.31 | **+5.44** | 1.80 |
| `g3_vec` / gloss | −0.86 | +0.43 | +2.03 | +0.53 | 2.89 |
| `g3_vec_aug` / lemma | +3.90 | +4.68 | +5.79 | +4.79 | 1.89 |
| **`g3_vec_aug` / gloss** | +0.74 | +1.53 | +0.73 | **+1.00** | **0.80** |
| `g3_vec_aug2` / lemma | +2.58 | +4.09 | +4.72 | +3.80 | 2.14 |
| `g3_vec_aug2` / gloss | −1.57 | +1.40 | +2.08 | +0.64 | 3.65 |

**1. The channel works where it has rows: +5.44.** Every seed positive.

**2. ⚠ THE SHUFFLE CONTROL GAINS +2.77 AND SHOULD NOT.** It permutes rows WITHIN a doc, which
destroys token-level alignment but preserves the document's bag of lexical content. So only about
**+2.67 of the +5.44** is attributable to per-token correctness; the rest is available from
document-level content however arranged. A clean control must shuffle ACROSS documents. This is the
number in the sweep to trust least, and it is a design error rather than a finding.

**3. The gloss fill needs augmentation to be worth anything, and then it is worth about +1.00.**
Unaugmented it is +0.53 with a range of 2.89 — indistinguishable from zero. Augmented it is +1.00
with a range of 0.80, positive in all three seeds and the tightest row in the table. The
augmentation's contribution is **reliability**, not headroom.

**4. ⚠ MATCHING THE DEPLOYMENT DISTRIBUTION IS THE WRONG TARGET FOR THE AUGMENTATION.** `g3_vec_aug`
perturbs to cos 0.70 by accident; `g3_vec_aug2` perturbs to cos 0.4611 (sd 0.1840) against a real
gloss's 0.4597 (sd 0.1845) — correct in both moments, and WORSE: +0.64 against +1.00 on the gloss
fill, with a range of 3.65 against 0.80, and −0.99 on the lemma fill. Perturbing all the way to the
deployment geometry destroys enough of the training signal that the model learns less from the
channel than it loses. **The mis-calibration was load-bearing.** The right augmentation strength is
a bias-variance trade-off to be tuned, not a distribution to be matched.

**5. Nothing on the fourteen languages with no rows survives three seeds.** Every arm lands between
−1.16 and −0.19 with ranges of 1.5–2.4. The seed-0 reading that an empty channel costs 1.88 LAS, and
that augmentation repairs it, are both inside noise and are withdrawn.

Over all twenty languages on the lemma fill: `g3_vec_aug` **+1.22** (range 0.75) is both the best
and the tightest, ahead of `g3_vec` +1.09 and `g3_vec_aug2` +0.97.

## The shift is a structured displacement, and that is what fixes it

Characterised on 239 748 Arabic pairs, so the augmentation was built from a measurement rather than
guessed:

| | |
|---|---|
| cos(source, gloss) | +0.4597 (sd 0.1845) |
| ‖mean shift‖ / mean ‖shift‖ | **34.6 %** — one constant direction |
| residual, top-8 of 128 dims | **36.3 %** of variance (6.2 % if isotropic) |
| cos after removing the mean | +0.5017 |

So **an isotropic-Gaussian augmentation would model the angle and none of the structure.**

The free correction that follows — subtract the constant direction from gloss rows at inference, no
retraining — was implemented and **does not work**: −0.86 → −1.05 on the six, −2.23 → −2.30 on the
rest. Raising the cosine to 0.50 buys nothing, so the mean displacement is not what the model fails
on. The residual is, and it is 65 % of the shift.

`g3_vec_aug` therefore displaces each training lemma vector by a **real shift vector sampled from
the measured 20 000**. The training signal is still the language's own aligned lemma vector — the
displacement is applied TO it, not substituted for it — and **no gloss enters training**.

⚠ **It does so at cos 0.7042 (sd 0.0717), not at the real 0.4597 (sd 0.1845)** — adding a
displacement sampled from one pair to an unrelated vector under-perturbs, because a displacement is
correlated with its own source. `g3_vec_aug2` corrects that exactly (0.4611 / 0.1840) by sampling
the target cosine from the empirical distribution and taking only the direction from a real shift.
**It is worse.** See result 4: the accident was load-bearing.

## Traps

1. **`paths.vectors` is spaCy's own key.** `[initialize] vectors` interpolates it and tries to load
   a spaCy `Vectors` object (E884). A lookup table the layer reads is not vectors the vocab
   attaches; use a custom `paths.vec_table`, as typology does.
2. **`basis.npz` and `fit_report.json` are per-GENERATION.** A v3 run into v1's work directory would
   rebuild thirteen v1 assets against a 32-language basis, and they would load and retrieve normally
   — a basis is only wrong relative to the rows it was fitted with. `--work` is required with
   `--sources`.
3. **The fold must reach the ARTEFACT, not just the build.** `stage_emit` took `key_norm` from the
   fit report, so a fold added after the fit was applied at build and recorded nowhere; Arabic came
   out at 41.2 % with the correct fold present in the code. It now comes from the declaration.
4. **A guard written from the deployment story can make the arm untrainable.** Two here fired on
   legitimate input: the no-rows guard (48 of 80 training languages have none) and the empty-doc
   guard (with one sentence per doc, a short sentence has nothing glossable). Enforcement of "the
   caller supplied nothing" belongs with the caller, which is the only place with a corpus view.
5. **The `lemma` fill is not a floor for every language.** For the six in the aligned-44 it is the
   upper bound. Labelling a run by its flag rather than by what it does to each language produced a
   wrong conclusion once in this file's history.
6. **`set_vectors_fill` and `set_gloss_debias` return a count and refuse at zero.** A silent no-op
   would score the lemma fill on a held-out language — all-OOV — and report it under the gloss
   fill's name.
7. **The predicted doc must carry LEMMA for the lemma fill.** `generic_corpus` copies UPOS, FEATS
   and LEMMA during training; an eval that copies only the first two makes the channel silently
   all-OOV. Caught by the layer's own refusal, not by any score.

## Files

| file | what |
|---|---|
| `scripts/fetch_vec_aligned44.sh` | the 32 training + 6 test aligned spaces |
| `scripts/build_vec_manifest_v3.py` | derives the corpus map from the v2 manifest; the four folds |
| `scripts/align_vectors.py` | `--sources`, `--work`, and the basis/fold refusals |
| `scripts/aligned_vectors.py` | `KEY_NORM` — one definition of each fold, read by builder and layer |
| `scripts/build_generic_vectors_v3.py` | merges 38 assets into one table; three vocabularies |
| `scripts/sud_generic_embed_v3.py` | `sud.GenericEmbed.v3`, the fills, the controls, the refusals |
| `scripts/make_generic_config_v3.py` | the five arms; `g3_base` built by calling v2's `build()` |
| `scripts/eval_generic_v3.py` | scoring under a named fill regime |
| `scripts/compare_generic_v3.py` | the group split |
| `scripts/estimate_gloss_shift_v3.py` | the shift, its mean, and the sample used for augmentation |
| `scripts/report_gloss_coverage_v3.py` | what fraction of each held-out language is fillable |
| `scripts/fetch_gloss_dicts_v3.sh` | 16 Wiktionary lemma→English bags |
| `metrics/generic_v3/` | the results |
