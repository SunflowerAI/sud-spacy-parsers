# XPOS: one tagset per arm, and conditioning it on UPOS+FEATS

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

Three arms train on more than one treebank, and each needed a different answer.

**sa (Vedic + UFAL + DCS) needs nothing**: field 5 is `_` in all three sources and is filled from
UPOS downstream, so the convention is already uniform. One stray was found and fixed -- a single
token (`dharm'`) carried `Compound=Yes` in the XPOS column in two UFAL files, a shifted FEATS
value, and it had reached the RELEASED sa tagger as a label. `normalise_sa_xpos.py` sets it to
the UPOS like every other sa token -- a script rather than a hand edit because `assets_sa*/` is
gitignored, so a manual fix there is lost by the next rebuild with nothing to say it ever
happened. FEATS is deliberately left alone: sa READS `Compound` as an input feature and the
tokeniser supplies it at runtime, which is how the arm is designed. **sa was NOT retrained** for
one token, so the junk label sits unused in the shipped model until the arm is next rebuilt.

## Latin: PROIEL and Perseus re-rendered as Index Thomisticus codes

ITTB is by far the largest (390 787 train tokens against PROIEL's 177 558 and Perseus's 18 259),
so its conventions win and **its own rows are left byte-identical**. The composite tag splits into
a LEXICAL head and a MORPHOLOGICAL tail, which is what makes the other two derivable:

    C1 | grn1 casB gen2 | vgr1
    ^^   ^^^^^^^^^^^^^^   ^^^^ graphical variant of the surface form
    |    a restatement of FEATS
    LETTER = declension/conjugation (lexical)   DIGIT = which paradigm the form inflects by

**Splitting the head is worth 1.5 points and is not obvious**: the digit is morphological, not
lexical -- a verb takes `3` when finite or infinitive and `2` when participial -- so keying the
whole head on the lemma tops out at 94.5 % where letter (98.43 % dominant per (lemma, UPOS)) plus
digit (98.15 % per (UPOS, VerbForm, has-Case)) reaches 95.9-96.3 %.

**The lemma-suffix rung is what makes it usable on the other treebanks.** 17.8 % of PROIEL and
23.4 % of Perseus tokens have a lemma ITTB never saw. On ITTB's own unseen-lemma tokens a UPOS
majority gets the letter right 41 % / 45 % (dev / test); a suffix model gets **83 % / 90 %** --
declension IS a property of the stem ending, so this is the feature the majority baseline was
missing, not a lucky correlation.

**`vgr` must be keyed on the FOLDED form** (length marks stripped, ligatures expanded, j/v -> i/u):
keying it on the form rather than the lemma is worth 4.5 points (89.2 -> 93.7 exact), and folding
is compulsory because the released arm is the orthographically augmented one, which respells every
token each epoch -- `vitae`, `vītae` and `vītæ` must render ONE tag. Verified: the plain and macron
copies of every split now produce byte-identical XPOS columns.

Held out on ITTB itself, the map reproduces the treebank's own gold XPOS **93.74 / 93.98 %** exactly
(test / dev) from (form, lemma, UPOS, FEATS) alone -- the honest ceiling on the tags it manufactures
for the other two. 1 614 300 tokens re-rendered into 1 518 tag types; only **1.97 %** of train tokens
land on a tag ITTB never used. Idempotent, XPOS-column-only, and which treebank a sentence belongs to
is read off its **sent_id**, not off a sentence COUNT as the blanking script did.

**Results.** Only the tagger was retrained (freeze recipe, `make_tagger_config.py` + `graft_pipe.py`),
because nothing in the Latin pipeline reads TAG as an input -- so parser, morphologiser, lemmatiser
and the three `sud_*` pipes come out byte-identical and every published Latin figure stands.

| slice | old arm / old gold | new arm / new gold | |
|---|---|---|---|
| **ITTB** | **90.68** | **92.92** | the only like-for-like row -- ITTB's gold never moved |
| PROIEL | 89.93 | 82.40 | not comparable: 23 coarse codes -> ~2 340 composite ones |
| Perseus | 24.29 | 68.37 | not comparable: the gold was BLANK |
| combined, plain test | 77.61 | **86.16** | POS 92.68, LEMMA 90.90, UAS 78.72, LAS 71.72 -- identical to the decimal |

**+2.24 on ITTB is the finding**, and it is the same shape as Perseus improving ITTB+PROIEL LAS:
converting the smaller treebanks' tags helps tagging on the largest one. **Perseus's 24.29 shows
blanking never removed it from the metric** -- spaCy keeps CoNLL-U `_` as a LITERAL tag, so 10 964
test tokens were scored against a gold value of "_", which was most of the gap to the published
77.61. TAG across orthographies (old -> new): plain 77.61 -> 86.16, macron 77.23 -> 85.31, vj 77.66
-> 86.20, lig 77.54 -> 86.03, caps 77.61 -> 86.12, breve 68.50 -> 76.44 -- so robustness is carried,
not traded.

## English: `,` now means comma in both halves

EWT and GUM share one PTB tagset (49 tags against 46, GUM's a strict subset) and agree on every word
class: of 688 (form, UPOS) types frequent in both, **17 disagree**, and all but a few are genuine
ambiguity rather than convention (`know` VB/VBP, `her` PRP/PRP$, `got` VBD/VBN -- web text and edited
prose really do differ in how often each reading occurs). Punctuation is the exception and is a flat
conflict: PTB reserves `,` for the comma and gives dashes, semicolons and ellipses `:`. GUM follows
that without exception; **EWT tags `;` as `,` 101 times out of 101**, `--` 123, `...` 159, `/` 142.

⚠ **This one deliberately does NOT follow the largest-treebank rule** (user decision, 2026-08-10).
In the arm that ships -- the merge with GUM's five NonCommercial genres dropped -- **EWT is the
LARGER half, 204 578 tokens against 135 746**. It is overridden because the disagreement is not a
house style but a standard: `,` is *defined* as the comma tag, GUM is consistent with that, EWT is
the outlier. (An earlier draft of this decision quoted GUM at 200 223 and called the margin 2 % --
that is the UNFILTERED treebank, which the shipped arm does not train on.)

`normalise_en_punct_xpos.py` rewrites 1 004 train / 131 dev / 149 test EWT cells. Two things it
needs and one bug it had:

- **The dash needs SPACING, not just the form.** PTB tags `-` HYPH inside `well-known` and `:`
  between clauses, so a form-keyed rule answers with GUM's compound-internal majority (HYPH 886)
  for tokens that are sentence punctuation. Split by spacing GUM is decisive where it matters:
  glued `-` is HYPH 593/608, and a spaced em dash is `:` 266/267.
- **...but then a spacing-blind BACKOFF, or unanimous evidence falls under the count bar.** GUM
  writes `/` SYM 49 of 49 and `?` `.` 593 of 595 whatever the spacing, but the glued halves are
  only 14 and 15 examples. The backoff does not fire for the dash, whose spacing-blind key is HYPH
  at 86 % -- under the bar precisely because there the distinction is real.
- ⚠ **Harvest the table ONCE, from train.** Harvesting per file looked reasonable and was a bug:
  dev and test carry a fraction of GUM's evidence, so they fell under `--min-count` on forms train
  committed (`/` -> SYM in train, left `,` in dev/test) -- which would have left the gold
  inconsistent between train and test, the exact defect the script exists to remove.

13 idiosyncratic web-text forms (`=`, `,?`, `|`, `….`) are reported and left alone rather than
silently committed. EWT-only material (`!!`, `*`, `<<`, `:)`, `@`) is untouched -- GUM has no opinion
about it, so there is no conflict to resolve. **The EWT-only files are out of scope**: `en_sud_ewt`
trains on EWT alone, where its convention is internally consistent and its published metrics stand.

**Results.** Headline TAG **94.19 -> 94.20** -- flat, because this is 0.3 % of the corpus -- with
POS/LEMMA/UAS/LAS identical to the decimal. The measurement that matters is on the affected forms:
**72.47 % -> 82.98 %**, and the old arm's single largest error was the conflict itself (`;` gold `,`,
predicted `:`, 23 times), a class that is now gone. What remains is `-` HYPH vs `:`, which is real
linguistic ambiguity rather than a clash of conventions.

## Released, 2026-08-10 (v0.2.0, clobbered)

`la_sud_ittb_proiel_perseus` and `en_sud_ewt_gum` were re-packaged at the SAME version and
re-uploaded; `en_sud_ewt` is untouched, since the EWT-only arm was deliberately out of scope.
`package_sud.sh` now NAMES the normalised arms by default (`training_la_aug_sud_xpos`,
`training_en_gum_sud_xpos`) rather than relying on an env override -- a default that names the
right arm is the fix, which is the fourth time that lesson has been paid for here.

Diffed file by file against the DOWNLOADED previous assets: la 30 of 40 identical, en_gum 30 of
38, and **no weight file except `tagger/model` moved** in either. The la movers are the tagger
(model + cfg), `config.cfg` (the tagger went from a listener to its own encoder),
`vocab/strings.json` (the new tag strings), two `meta.json` and the packaging metadata.
⚠ Two diff entries that look alarming and are not: la's `la_macronise/lut.json.gz` differs in its
GZIP bytes while the decompressed content is byte-identical (it is the empty `--no-lut`
placeholder -- gzip stamps a timestamp), and the `README.md`/`METADATA` churn is the regenerated
performance table.

⚠ **`graft_pipe.py` used to ship the REPLACED pipe's score.** `nlp.to_disk` writes the
recipient's meta, so the first build of both wheels carried the old tagger's `tag_acc` in the one
field `spacy info` shows users -- la claiming 0.9028, measured on a 1 952-label tagset, while
shipping a 2 342-label one. It now carries the grafted pipe's own metrics across (`PIPE_METRICS`),
and only those: the donor's `dep_las` matched the recipient's to the digit, which is the frozen
parser proving itself. la's dev `tag_acc` is 0.9028 -> **0.8945**, and that FALL is expected --
dev is ITTB+PROIEL only, so it is dominated by PROIEL's task getting much harder (23 codes ->
2 342). The gain is on ITTB, where the gold never moved.

**Wheels grew: la 27.3 -> 33.5 MB, en_gum 16.7 -> 22.8 MB.** That is the dedicated tagger encoder
replacing a listener, which has almost no parameters of its own. It is sized at the BASE arm's
width 96 / depth 4 rather than the 64/3 the morphologiser and lemmatiser get, deliberately: those
predict a UPOS or an edit tree, this predicts one of ~2 340 composite codes, and under-sizing it
would have confounded the tagset change with a capacity cut.

⚠ Same version, so `pip install -U` will NOT replace an older copy; `--force-reinstall` will.
Verified by DOWNLOADING both published assets (sha256 identical to what was built) and loading
them from a clean `--target` install, not from `build_sud/`.
`metrics/release/metrics_release_la*.json` still holds the pre-normalisation TAG and is now stale on that one
field; every other figure in it is unchanged and still correct.

## XPOS conditioned on UPOS+FEATS — and WHERE the conditioning enters

**BUILT AND MEASURED, NOT RELEASED.** Every arm grew the same way: base pipeline
`[tok2vec, tagger, parser]`, morphologiser added later as a frozen layer. So the one component whose
target is largely a restatement of UPOS+FEATS was the only one that could not see them, purely
because of the order the layers were built in. Fixing that works, but **only if the conditioning
enters ABOVE the encoder**, and the two attempts differ by more than the whole size of the effect.

⚠ **Injecting at the BOTTOM — extra columns in the embed — LOSES 0.2–0.6 TAG** (NEGATIVE-RESULTS.md).
A `MaxoutWindowEncoder` of depth 4 then convolves the channels over a ±4-token window, so each
token's tag comes to depend on its NEIGHBOURS' predicted morphology, and the token representation is
rebuilt from scratch instead of reusing the co-trained shared encoder. Whether the bundle is hashed
whole (`MORPH`) or decomposed per feature makes almost no difference (≤ 0.10) — **the injection
point was worth ~0.7, the representation ~0.1.**

**`sud.Tok2VecPlusFeats.v1` (`scripts/sud_feats_embed.py`, `make_xpos_config.py --top`)** keeps the
released tagger's own `spacy.Tok2VecListener.v1` on the FROZEN shared encoder — so the token
representation is EXACTLY the shipping one and the experiment is single-variable — and concatenates
a morphology side channel (width 32) immediately below the softmax. A token's own morphology then
reaches that token's decision and nothing else. `--top --no-cond` is the tightest control the harness
can make: the released tagger with a retrained head, and it reproduces the released row to within
0.15, which is what licenses reading the conditioned one.

    test TAG    released  top-ctl   top   Δcond  |   released  top-ctl   top   Δcond
    ar             89.44    89.30  89.70  +0.40  | ja   95.09    95.21  95.35  +0.14
    zh             90.81    90.34  91.01  +0.67  | yue  93.74    93.66  93.81  +0.15
    en             93.09    93.08  93.38  +0.30  | fa   96.19    96.22  96.27  +0.05
    lzh            92.59    92.77  92.80  +0.03  | id   92.12    92.18  92.19  +0.01
    ko             72.92    72.61  72.67  +0.06  |

**Positive in 9 of 9 languages**, and the arm beats the released tagger in 8 of 9 (ko −0.25, the one
language where the retrained head does not match the released one). ar and en were replicated over
**three seeds with NON-OVERLAPPING ranges** (ar control 88.67–88.75 v conditioned 89.03–89.13;
en 93.00–93.00 v 93.23–93.26) — the spread is ≤ 0.001 precisely because the encoder is frozen and
only the head trains, which is what makes a +0.25 effect readable at all. The other seven are single
seed.

**The side channel's contents are DERIVED, not chosen** (`scripts/build_feats_inventory.py`): each
FEATS key is ranked by the information it carries about XPOS *once the form is already known*, which
is the only question that matters since the tagger reads the form anyway. Its most useful output is
that **zh, id and ko have no such key at all** — H(XPOS|form) is already 0.251 / 0.018 / 0.089 bits —
so their XPOS is a function of the spelling and their side channel carries POS+MORPH only. Where keys
do qualify the lists are sensible: ar Case/Number/Definite/Gender/AdpType/Mood (Case alone 0.444
bits), en Number/PronType/VerbForm/Person/Tense/Mood/Degree — exactly the PTB VBD/VBN/VBP/VBZ and
JJ/JJR/JJS distinctions. SUD's own FEATS-column keys are excluded, sourced from `sud_misc` so the two
cannot drift; NB `ExtPos` is NOT excluded and is the only key ja qualifies on.

⚠ **`build_feats_inventory.py`'s gold-feature ranking is a guide, not a prediction.** The
`scripts/xpos_headroom.py --model` measurement is the cautionary one: a majority-class map on GOLD
UPOS+FEATS beats the tagger on most arms, and the SAME map on PREDICTED features loses to it on all
ten, because morphology is predicted at 0.75–0.99 exact and its errors land on the tokens the tagger
also finds hard. The realised gains above are a fraction of the gold oracle, and that is expected.

**WARM START (`--warm-start`, `sud.WarmStartTagger.v1`) is the version to use, and it covers all
eleven arms.** Trained from scratch the head has to relearn what the released tagger already knew,
and where the released head was the better one that deficit ate the gain (ko −0.31, zh −0.47 on the
retrained-head control alone). So: copy the released tagger's output layer into the first W columns,
**zero the S new ones**, and copy the inner encoder when it has one. At step 0 the model then IS the
released tagger — `scripts/check_warm_start.py` confirms it token for token (ar 13 928/13 928,
la 4 333/4 333) — and the side channel has to earn every column it uses. Copying the inner encoder
is also what extends this to **la and en_gum**, whose shipping taggers carry a dedicated
`HashEmbedCNN` from the XPOS-normalisation work instead of a listener; `--warm-start` reads the
released tagger's architecture and reproduces it verbatim (24 tensors for la).

⚠ **LABEL ORDER, not just the label set.** `W` is indexed by label id, so copying it into a tagger
whose labels sit in a different order silently scrambles every class — the hazard
`rename_deprel_label.py` guards for the parser's action table. `--warm-start` writes the released
arm's label list to `labels_config_<arm>/tagger.json` and initialises from it, so the orders agree by
construction, and the callback REFUSES to copy unless they match position for position.

    test TAG   released  warm-ctl  warm   Δcond  |    released  warm-ctl  warm   Δcond
    zh            90.81     90.63  91.12  +0.49  | id    92.12     92.13  92.27  +0.14
    en            93.09     93.15  93.50  +0.35  | fa    96.19     96.13  96.23  +0.10
    en_gum        94.22     94.14  94.45  +0.31  | lzh   92.59     92.81  92.88  +0.07
    ar            89.44     89.46  89.71  +0.25  | ko    72.92     72.93  72.93  +0.00
    yue           93.74     93.50  93.66  +0.16  | la    86.16     86.10  86.10  +0.00
    ja            95.09     95.23  95.38  +0.15  |

**Δcond >= 0 in 11 of 11**, mean +0.18, and the arm beats the released tagger in 9 of 11 (yue −0.08,
la −0.06 — both inside the noise). The warm start is what fixed ko (−0.25 → +0.01). **la gains
exactly nothing**, which is the one result worth flagging as a surprise: its composite XPOS tail is a
restatement of FEATS by construction, but its morphologiser is the weakest of the family
(`morph_acc` 0.826) and its tagger already reads a heavily orthographic signal, so training never
improved on the warm start at all.

## Released, 2026-08-12 (v0.2.0, clobbered) — the XPOS-downstream taggers

All **twelve** arms re-packaged and re-uploaded with the warm-started conditioned tagger.
⚠ sa was nearly missed, and it turned out to be the BIGGEST WIN: v0.2.0 had only ever held SIX
wheels ("Six of the eleven", per its own release body), so sa existed only at v0.1.0, and the first
pass through this work skipped it as a joint multi-task arm with "XPOS = a copy of UPOS, nothing to
gain". The second half of that is true and the conclusion was backwards -- sa's XPOS IS its UPOS on
100.00 % of the multitask corpus, and its tagger was scoring **below its own morphologiser** at
predicting the identical label (dev tag_acc 0.8957 v pos_acc 0.9017). Conditioned on UPOS it
converges to 0.9016, i.e. within 0.0001 of the morphologiser it now reads, and gains **+1.52 TAG**
on the UFAL test set -- more than any other arm. A label that is a COPY of another is the strongest
case for this change, not a reason to skip it.
`scripts/graft_xpos_tagger.py` puts the donor tagger into each shipping arm and MOVES it behind the
morphologiser, verifying three things per arm: the shared components are byte-identical, the
reordered pipeline reproduces the recipient's PARSE token for token (heads and deprels — so every
published LAS/UAS figure stands), and the grafted tagger reproduces the donor's tags exactly. All
three held on all eleven; the parse check covered 8 227–26 164 tokens per arm.

    sa 72.06 -> 73.58   ja 95.09 -> 95.38   yue 93.74 -> 93.66     (sa on the UFAL test set,
    ar 89.44 -> 89.71   id 92.12 -> 92.27   la  86.16 -> 86.10      which is what the README and
    en 93.09 -> 93.50   fa 96.19 -> 96.23   ko  72.92 -> 72.93      metrics/release/metrics_release_sa.json report)
    en_gum 94.22->94.45 lzh 92.59 -> 92.88
    zh 90.81 -> 91.12

yue and la go marginally BACKWARDS (−0.08 / −0.06, inside seed noise); shipped anyway by user
decision, so the whole family has one tagger architecture. The sa wheel diffed against its v0.1.0
predecessor is 34 of 49 files identical with **no weight file except `tagger/model`** moved. `metrics/release/*.json` updated on
`tag_acc`/`tag_micro_*` only — every other field is unchanged, because every other component is
byte-identical. ⚠ `metrics/release/metrics_release_la_{ittbproiel,perseus}.json` hold the per-slice TAG and were
NOT re-measured; they are stale on that field.

Packaging notes worth keeping: `pkg()` now appends `scripts/sud_feats_embed.py` to EVERY wheel's
`--code` rather than to eleven separate lists — ko passes no `--code` at all, and a list that has to
be remembered is a list that gets missed. THREE surgery scripts still failed on the first run because
they carry their own module lists that predate the layer (`add_id_lemma_case_fix.py` and
`add_sa_frontend.py` import curated sets; `add_la_macronise.py`/`add_la_enclitic_tokenizer.py` take
explicit `--code`), and every failure was SILENT in the driver (`>/dev/null 2>&1`), surfacing only
as "SRC missing — skip" and a STALE 0.1.0 id wheel still sitting in `build_sud/`. `SUD_BASE` overrides the packaged arm per run.

⚠ Same version, so `pip install -U` will NOT replace an older copy; `--force-reinstall` will.
Verified by DOWNLOADING all twelve published assets (sha256 identical to what was built) and loading
each from a clean `--target` install — pipeline order `morphologizer` before `tagger` confirmed in
every wheel, and lzh's `clause_parser` correctly lands AFTER the tagger so its punctuation XPOS is
not overwritten.

Arms kept as `training_<lang>_xpos{down,feat,top,warm}{,_ctl}{,_s1,_s2}`; drivers `train_xpos.sh`
(`XPOS_WARM=1`, `XPOS_TOP=1`, `XPOS_CTL=1`, `XPOS_FEATS=1`) and `eval_xpos.sh` (`XPOS_ARMS=...`).

