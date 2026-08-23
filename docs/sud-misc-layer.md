# SUD's own MISC layer (`sud_misc.py`, `sud_idiom.py`, `sud_tagger.py`)

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

Output slot: **`Token._.sud_misc`** (a dict; `sud_misc.py` owns it with `set_misc`/`get_misc`/
`misc_string`/`feats_string`, `has_extension`-guarded). `token.morph` is deliberately **not** used as
the slot, so a predicted SUD feature never has to compete for room with a morphological one.

**Which CoNLL-U column a key belongs to is a property of the KEY.** `Idiom`/`InIdiom`/`Reported`/
`Subject` are MISC (column 10) features in every treebank here; **`Shared` is a FEATS (column 6)
one** — 10 178 tokens in SUD_English-EWT train, all in field 6, none in field 10. We follow the data
rather than the prose throughout (SUD's guidelines list `Subject` among the FEATS features and the
data does not), so the two groups are declared separately in `sud_misc.py` (`SUD_MISC_KEYS` /
`SUD_FEATS_KEYS`) and serialised by `misc_string` / `feats_string`. At runtime both live in the one
dict.

Gold transport is via `hoist_sud_gold.py` (see the `spacy convert` gotcha in `CLAUDE.md`), which
now reads **both** source columns: a key is looked for in MISC, then in FEATS, and one found in FEATS is
*consumed* — leaving `Shared` beside `SudShared` would make the reference carry the same gold twice.
Carrying an already-hoisted key forward is what keeps the script idempotent for the FEATS-sourced
keys; without it a second run finds no bare `Shared` to re-derive from and silently deletes the gold.
Side effect: the frozen morphologiser is then scored against gold FEATS carrying keys it never
learned, so **`morph_acc` in these arms' logs reads artificially low** — cosmetic, score weight 0.

## `Idiom`/`InIdiom` — exact, no training

SUD marks idioms with features, not a `fixed` relation: the head carries `Idiom=Yes` + an `ExtPos`,
other members `InIdiom=Yes`, unanalysable members attach by `unk`. Measured over train in all seven
treebanks that annotate idioms, that is an exact recipe:

    Idiom=Yes    <=> has ExtPos AND has an `unk` dependent                 P = R = 100 %
    InIdiom=Yes  <=> attaches by `unk`, and walking up through consecutive
                     `unk` links reaches a head with ExtPos                P = R = 100 % (la 99.9)

Both conjuncts are needed: `unk` alone gives `InIdiom` precision fa 6.5 / ar 53 / en 75 %; `ExtPos`
alone over-predicts `Idiom` in English (702 ExtPos vs 477 Idiom). Both inputs are already predicted,
so this needs **no training and no retrain** — appended at packaging time by `add_sud_idiom.py`.
End-to-end it is much lower, because it inherits the morphologiser's `ExtPos` and the parser's `unk`
errors; `eval_sud_idiom.py` reports both, and the gap is the honest measure. Test, gold trees →
end-to-end F: ja 100→96.8/95.7, en 100→84.6/82.1, sa 99.6→77.7/81.3, fa 100→72.7 (n=6), ar
100→67.3/68.4, lzh 100→66.0/68.8, **la 100→35.3/50.0** (la has only 489 train `ExtPos` in 586k
tokens, so `ExtPos` is almost never predicted). Precision holds (78–98 %); **recall is the limiter**.

## `Subject` — trained, but the rule wins in two languages

The **value** is determined by (deprel, head UPOS) at 100 % (zh 91 %) over 3–10 contexts per
language; the **presence** is genuinely lexical, and that is the hard part.

`sud_tagger` is a custom `TrainablePipe` because spaCy ships no generic token classifier (`Tagger`
hardcodes `doc.c[j].tag` and `get_aligned("TAG")`; a second `morphologizer` would wipe the first's
morph; `Token._.` is unreachable by `get_aligned`, E983). It subclasses `Tagger`, keeps
`spacy.Tagger.v2` unchanged, and overrides the output slot, the gold source and the scorer. **`O` is
an explicit negative class** — `Tagger` maps a `""` label to *missing* (no gradient), wrong for a
majority class that must be learned. `sud_subject_rule.py` is the lexical alternative: a
(head lemma, deprel, head UPOS) frame table from `build_sud_subject_frames.py`. Compared end-to-end
on test by `eval_sud_subject.py`:

    lang   trained F   rule F   ships     n(test)
    en       80.0       63.9    trained     266
    fa       89.5       71.6    trained      38
    la       67.0       52.4    trained     674   (augmented base; 66.3 / 53.0 on the union one)
    yue      66.7       36.4    trained       6   (not meaningful either way)
    lzh      66.2       80.0    RULE        174
    zh       27.7       31.6    NEITHER     302
    sa       10.5       12.5    NEITHER      14   (142 train instances)

The split is principled: Classical Chinese raising rides on a handful of verbs (可/能/欲), which a
7-entry table captures and a small neural encoder cannot beat; en/fa/la raising has a long lexical
tail where the table's recall is fine but its precision collapses (en rule P 51 % vs trained 82 %).
**zh ships no `Subject` layer** — an annotation wrong two times in three is worse than none — and
since zh annotates no idioms either, **the zh wheel carries no SUD MISC layer at all**.

## `Reported` — bootstrapped from scratch

(`sud_reported_gold.py` builds the gold.) `Reported=Yes` occurs **zero** times in every treebank here
(the deprel form is 8 Latin tokens), so the class is synthesised, as ja's `comp:obl` was. It supersedes an older `parataxis:obj` analysis,
and that history fixes the target: the paratactic analysis existed for **direct** speech, so
`Reported=Yes` marks a complement of a speech/writing verb quoted verbatim.

Two independent direct-speech signals, and which one fires is a property of the language's
**punctuation habits**, not of the phenomenon: **quotation marks** in the complement's subtree (the
only signal in ar, 712/2297 candidates, and fa, 99/1606); and **a `discourse` dependent** inside the
complement — not a quotative marker but the direct-vs-indirect discriminator, because only verbatim
speech can host the speaker's own interjections. That is what makes la and sa tractable at all (both
have **0** quoted candidates), and the markers found are exactly right: en `no`/`well`/`yes`, la
`autem`/`quidem`/`uero` under `dico`/`inquit`, sa `vai`/`eva`/`hi` plus the quotative **`iti`** (908).
Indirect evidence commits the negative: an overt complementiser (en `that`, fa که, ar أنّ) or Latin's
accusative-and-infinitive. NB the test is on the **complement token itself**, not its subtree — SUD
makes the subordinator the head of the clause it introduces, so a complementiser anywhere else
belongs to an embedded clause, which inside a verbatim quote proves nothing.

**Latin needs almost no LLM** (`la_finite_direct`): Latin reports statements indirectly with the
accusative-and-infinitive, and every finite indirect clause carries an overt subordinator, which
under the functional-head analysis IS the complement token. So a finite complement of a speech verb
that is not itself a subordinator **has no way to be indirect** (219 cases in train — `dicit ,
meditatus sum…`, `dixit , fiat lux`). The one exception, the indirect question, is finite and
subordinator-less but requires the SUBJUNCTIVE, so mood separates it: an indicative clause containing
`qui` has a relative pronoun (75 cases), a subjunctive with no interrogative is a jussive inside a
quote (6). Only subjunctive + interrogative (39) is withheld — withheld **to the model**, not
committed as indirect.

**Reported speech is a CLAUSE.** A speech verb also takes ordinary nominal and prepositional objects
(`dicit hoc`, `loquor de X`), which must never reach the model — in Latin they were 4427 of a
4724-case residue. Candidates carry a `clausal` flag (complement is VERB/AUX/SCONJ or has a
`VerbForm`); a non-clausal residue case is dropped unannotated. Rule commits / model residue after
all three refinements: sa 1321/39, ar 997/1350, la 285/346, en 204/394, fa 110/487 — 2616 queries,
down from ~11 240. Residue goes to `disambiguate_pp.query` (resumable
`caches/relabel_cache_reported_<lang>.jsonl`).

**The `--structural` encoder: input features matter more than architecture.** The first arms used the
standard added-layer encoder (±3 receptive field over NORM/PREFIX/SUFFIX/SHAPE) and scored F
0.12–0.40 — right for `Subject`, wrong here, where every cue is non-local (the governing verb can be
far from the clause head, quotation marks sit at the clause EDGES, and Latin's diagnostic is the
complement's own VerbForm/Mood plus the ABSENCE of a subordinator). `make_sud_config.py --structural`
swaps in explicit `MultiHashEmbed` + `MaxoutWindowEncoder` so the embed can read **`DEP`**, **`LEMMA`**
(collapses inflection, so a speech verb is one symbol across its paradigm — decisive for la/ar/sa),
**`POS`/`MORPH`** (the whole Latin finite-vs-infinitive diagnostic) and **`IS_QUOTE`**, at window 3 /
depth 4 (±12, reaching the clause edges). This **requires `annotating_components`**: the corpus
readers build the predicted doc from gold words only, so DEP/POS/MORPH/LEMMA would be absent in
training and appear from nowhere at inference (same reasoning as sa's `Compound` feature). Result:
ar test F 37.4 → 46.7 (recall 26.7 → 45.4); dev ar 0.36→0.51, fa 0.20→0.40, sa 0.40→0.56, en
0.36→0.32, la broke (0.0004).

**Which component wins is predicted by where the gold came from** — end-to-end on test
(`eval_sud_reported.py`), against the share of gold the RULES committed rather than the LLM:

    lang   plain   structural   rule    ships       gold rule-derived
    ar     37.4      46.7       73.5    rule            95 %   (997 / 1047)
    la      0.0       0.0       17.7    neither         91 %   (285 / 314)
    sa     39.6      58.0       68.8    rule            73 %   (1321 / 1814)
    en     27.6      35.0       66.7    rule            45 %   (204 / 456)
    fa     20.0      40.0       23.5    STRUCTURAL      13 %   (110 / 836)

A rule reproduces the rule-committed portion almost by definition and cannot touch the LLM-decided
remainder; a trained model can learn either. So the rule wins where the gold is mostly rule-derived
and loses in Persian, whose gold is 87 % LLM-decided (rule P 1.00, R 0.13 — it only fires on the 110
cases it committed itself). This is a property of how the class was built, not a fact about the
languages. **ar/sa/en ship the rule; fa ships the STRUCTURAL trained pipe; la ships no `Reported`
layer** — Latin needs a four-deep chain of predicted lemma/deprel/VerbForm/Mood that compounds too
badly. (fa's figures below predate the `annotating_components` fix; retrained it is F 46.15 at
P 54.55 against its rule's 23.53, which is why it now ships.) `add_sud_reported_rule.py` removes the trained
pipe when it adds the rule, so no dead weights ship. Lexicons live in `sud_reported_data.py`,
imported by BOTH the gold builder and the runtime component so they cannot drift.
**Read these numbers with care:** there is no independent gold for `Reported` — the target is itself
these rules plus an LLM pass — so they measure *reproducibility at inference*, not correctness.

## `Shared` — the one key the morphologiser was already predicting

`Shared=Yes|No` says whether a dependent of a conjunct is shared with the other conjuncts — in
`identifying and breaking up terror cells`, `up` is `Shared=No` (it belongs to the second conjunct)
and `cells` is `Shared=Yes` (the object of both). It is the **broadest** of the five keys: every
treebank here annotates it, so the language list is no longer the `Subject` list. Only ja is left out
(27 `Yes` in 168 333 tokens — the call made for sa's `Subject`).

**It differs from the other four in being a FEATS feature**, which means the released morphologisers
have been predicting it all along inside their FEATS bundles, and badly: en test P 0.68 / R 0.15,
with `Shared=Yes` correct **4 times out of 247**, and 253 of en's 572 morph labels contain the key
(la 2110 of 6170), so it roughly doubles the label inventory it is carried in. A pipe therefore has
to *beat* that rather than merely exist, and where one ships it takes the feature over —
`clear_morph` deletes `Shared` from `token.morph` so the wheel has one answer rather than two.

**The candidate mask is the whole design** (`sud_shared_data.py`, shared by the harvester, the rule
and the eval so they cannot drift). A token is a candidate iff its head is a conjunct, its own
relation is neither `cc` nor `conj`, and it lies **outside** the span between the first and last
conjunct — a dependent sitting between two conjuncts is inside its own conjunct's territory and SUD
does not mark it. On en train that reaches 92.9 % of gold `Shared` while cutting the field from
204 578 tokens to 15 499, of which 63 % carry the feature. It is a recall device, not a rule (39 % of
what it admits is unmarked), and `sud_tagger` takes it as a `mask`: outside it the gold is *missing*,
not `O`, so the model spends no capacity reproducing a constraint it is being given.

Test, end to end over gold tokens (`eval_sud_shared.py`; "mask" = the share of gold the mask reaches
on a **predicted** parse, a ceiling on rule and trained alike):

| lang | mask | morph | rule | trained | ships |
|---|---|---|---|---|---|
| fa | 80.2 | 27.1 | 58.3 | **67.7** | trained |
| en | 70.6 | 24.7 | 55.1 | **62.6** | trained |
| lzh | 65.5 | 41.3 | 52.7 | **58.8** | trained |
| ar | 60.2 | 37.8 | 52.6 | **54.6** | trained |
| id | 57.1 | 36.1 | 49.1 | **53.6** | trained |
| la | 48.8 | 10.2 | 36.8 | **38.1** | trained — on the AUGMENTED base, which is what ships; the superseded union base preferred the rule, 35.9 v 35.1 |
| ko | 37.6 | 11.3 | 28.6 | 32.5 | neither (P 40.1) |
| zh | 32.7 | **37.5** | 29.1 | 31.5 | neither — the MORPHOLOGISER wins, uniquely |
| yue | 28.4 | 6.7 | 16.0 | 21.5 | neither (P 27.7, n=74) |
| sa | 17.3 | 8.6 | 9.4 | 3.8 | neither |

**Two different tests, and conflating them is a mistake worth not repeating.** Whether to ship
*anything* is a precision question — an annotation wrong more often than right is worse than none,
which is what kept `Subject` out of the zh wheel. *Which arm*, once both clear that, is decided on
**F**, as every other choice in this layer is (lzh's `Subject` rule at 75.8 over 68.8; ar/sa/en's
`Reported` rules). An earlier draft used the precision floor as a tiebreaker and shipped la's
trained pipe over its higher-F rule; that was wrong. Where nothing ships the
morphologiser's FEATS value is left alone: for zh that is the best arm available, for ko/yue/sa it is
merely the status quo. **id and ko had no SUD layer at all before this**; id now has one.

**The mask column predicts the whole table, and it is a fact about the PARSER.** The mask is defined
over the coordination, so its quality is parse quality on exactly that structure — not on the
sentence at large. Sanskrit is the worked example: on GOLD trees the harvested table reaches dev
F 52, but on sa's own predicted trees (LAS ~0.51) the mask covers 17 % of its gold, the trained pipe
saw almost no positive example, and it learnt nothing (F 3.8). Read the mask line before either arm.

**Architecture, measured on en (dev F, `sud_tagger`'s own scorer).** The encoder is a property of the
FEATURE, not of the language, so `make_sud_config.py` takes `--encoder` and `--mask` **per feature**:
en trains `Subject` (local), `Reported` (structural) and `Shared` (tree) in one arm.

    default encoder, no mask   0.323        structural + mask   0.586
    structural, no mask        0.547        tree + mask         0.616   <- ships
    tree, no mask              0.609

**⚠ The SUD layer must be trained on the arm that SHIPS, and for lzh that is not the obvious one.**
lzh's released chain is `training_lzh_rm_morph` — punctuation-restored, rule-merged, and with NO
trained lemmatizer (`han_lemma_lut` replaces it at packaging). Its `Shared` pipe was first trained on
`training_lzh_lemma` instead, whose parse is a different model's, so its coordination mask was a
different mask; and the resulting wheel was published, silently reverting the punctuation arm,
`--keep-marks` and the lemma table. `train_sud.sh` now names the arm `training_lzh_rm_sud` after the
chain it belongs to, and `src_conllu` gives lzh the `.punct.rulemerged` files — the plain
`.relabeled_ext` ones carry no PUNCT tokens, so a corpus built from them would not even align under
`gold_preproc`. The conclusion survived the correction (trained 58.8 v rule 52.7 v morphologiser
41.3); the numbers moved. **The rule TABLE is generation-coupled too** — `build_sud_shared_frames.py`
harvests lzh from the same `.punct.rulemerged` files, since a table keyed on a tree with no
punctuation in it answers a different question.

**⚠ `model-best` in a multi-feature arm is picked on the WEIGHTED MEAN of its features' scores**, so
a pipe can be checkpointed at an epoch that suited its neighbours. Latin is the case that matters:
its `Shared` peaked at dev F 37.34 while the saved epoch holds 31.90, chosen for `Subject`'s sake —
and la is precisely where the trained arm lost to the rule. Retrained ALONE
(`SUD_FEATS=Shared SUD_SUFFIX=_shared`, which `eval_sud_shared.py` then prefers) it reaches dev 35.91
/ test 35.10, and **still** does not beat the table's 35.85, so la ships the rule on a fair
comparison rather than a handicapped one. **No other decision turns on this** — the same gap is
≤ 2.9 everywhere else (zh 2.27, yue 2.85, sa 2.79, lzh 1.88, en 1.01, ar 0.53, fa 0.06), and the
single-feature id/ko arms have none by construction. `graft_pipe.py` puts a solo-trained pipe back
into a multi-feature arm, checking first that the two share a base (it refuses when the frozen
components differ, so a pipe cannot be fed a different model's predictions).

**But a dev-F gap is not a test-F gain, and the one time it was checked it went the other way.**
After the lzh arm was rebuilt on the right chain its own gap fell to 0.09, leaving en the largest at
1.01. Trained solo, en's `Shared` reached dev 61.55 against the combined arm's 59.99 — and **test
62.23 against 62.62**. So the combined-arm checkpoint was the better model on held-out data, the
graft was not made, and no arm now carries a gap worth acting on (en 1.01, ar 0.53, lzh 0.09,
fa 0.06, id/ko 0 by construction). Retrain solo when a gap is large enough to change a DECISION, as
la's was; not to chase a point of dev F.

`sud.HeadDepsTagger.v1` wins because the evidence is not linear at any width: what matters is which
token is my head and what else hangs off it, which `[own | head | mean of dependents]` reads
directly. The rule (`sud_shared_rule.py` + `build_sud_shared_frames.py`, a backoff table over
(deprel, head UPOS, position)) is the comparison arm; its threshold defaults to a plain **majority**,
not the 0.90 dominance test `apply_udep_rules.py` uses — that script commits annotation to a
treebank, this one has to answer wherever the mask asks (en dev F 63.7 at 0.90 vs 75.7 at 0.50, and
zh/yue collapse to nothing at 0.90).

### The pooling is a SEGMENTED REDUCTION, not a loop over tokens

`HeadDeps` originally built the third slice with a Python loop — `D[i] = X[idx].mean(axis=0)` once
per token, over a list of per-token index arrays walked off `Token.children`. That loop was never
inherent to the computation, only to writing it against the token API, and it is what made the
whole arm look like a bad fit for a GPU.

**`pool="deps"` — what the shipped pipe uses — needs no tree walk at all.** "All immediate
dependents" is exactly the INVERSE of the heads array: the edge list is `src = arange(n)`,
`seg = heads`, minus the root's self-loop. A ragged mean over that is a segmented reduction — one
gather, one `scatter_add`, one divide by the counts — so the layer costs O(1) array ops per document
instead of O(n). The backward is the same shape, because the gradient of a mean splits evenly:
dividing the PARENT row once and gathering it to each child is the identical arithmetic the loop did
as `dY[i, 2w:] / len(idx)`. Heads themselves come from `doc.to_array(HEAD)` (relative offsets stored
unsigned — view as signed and add the position) rather than a comprehension over `t.head.i`.

The other three modes vectorise too, and one detail is easy to get wrong: under `closed2` the
original filtered the GRANDCHILD's UPOS but left the intermediate link unfiltered
(`for c in t.children for g in c.children if g.pos_ in CLOSED_CLASS`). Filtering the middle link as
well would be a different feature. Multiplicity is likewise preserved rather than deduplicated — the
two-level modes can reach a token twice and the loop counted it twice, so the mean must too.

**Measured, on real parsed docs: 4.8–5.0x** for the layer (synthetic 5.5x). The counts are
accumulated with `scatter_add` on a vector of ones rather than `xp.bincount`, so no second backend
op has to exist and agree, and the denominator is clamped at 1 so a leaf divides to zeros not NaN.

**`scripts/check_head_deps.py` is the equivalence proof, and its reference is
`git show <ref>:scripts/sud_tagger.py`** — taken from git, not transcribed, so the check cannot
drift from what was actually there. Both wrappers get the SAME stub encoder, so only the pooling is
under test. Forward is BIT-IDENTICAL on all five modes; backward is bit-identical except `deps2` and
`closed2`, which differ by 4.768e-07. That was chased rather than waved through: against an exact
float64 accumulation the two implementations are **equidistant** (4.172e-07 each) and differ from
each other by exactly one float32 ULP at that magnitude, i.e. summation order, since a token there
receives several pooled contributions and the loop summed them per-parent while the edge list sums
them in edge order. The checker therefore demands exactness for the single-level modes and allows a
data-scaled 4-ULP budget for the two-level ones — a bound in ULPs of the largest gradient, not a
magic constant.

**A pure speed change, and verified as one**: no parameter shape moves, so existing weights are
untouched and every published `Shared` figure reproduces to the decimal (en_gum 58.15, en 63.10,
fa 67.71, lzh 58.78, with the mask and rule rows unchanged). A 400-step run confirms the TRAINING
path, which the eval never exercises (`SUD_SHARED_F` 13.41 -> 55.24 on a real loss). ⚠ Released
wheels BUNDLE `sud_tagger.py`, so a wheel keeps the old layer until it is re-packaged — a code-only
re-release is what hands users the faster inference.

**It does not overturn the GPU verdict.** It removes the specific blocker (O(n) kernel launches, and
one host->device copy per token to ship each index array across), but these remain small CNNs at
width 64–96 where transfer dominates, so the dependable payoff is faster CPU training — which is
where the pipe actually ships. Treat "makes GPU viable" as a hypothesis needing a probe.

## ⚠ `annotating_components` was missing `tok2vec` — every structural arm was trained on noise

Found while building `Shared`, and it **fails silently**. The tagger/parser/morphologizer/lemmatizer
in these arms are listeners on the shared encoder, so running them without `tok2vec` feeds them a
stale buffer. Nothing raises. On a 298-token dev doc the predicted parse came out with three distinct
deprels (`ROOT`, `comp:obj`, `goeswith`) and **no `conj` at all**, against twelve and four once
`tok2vec` runs — so a pipe reading DEP/POS/MORPH was reading noise, and the `Shared` mask was EMPTY on
every training doc, its loss a flat 0.00.

Fixed in `make_sud_config.py`; all arms retrained. What it moved, end-to-end on test:

    Reported  fa 40.0 -> 46.15 (structural, the one shipped)   ar 46.7 -> 45.98   sa 58.0 -> 52.17
    Subject   lzh 59.0 -> 68.83   en 80.0 -> 82.01   fa 89.5 -> 90.67   la 66.3 -> 62.60

**Every ship decision survives** (lzh still prefers its `Subject` rule at 80.0 against 66.2; ar/sa/en
still prefer their `Reported` rules).
⚠ **lzh's `Subject` rule cannot be scored on the bare `training_lzh_rm_morph`** — it keys on the head
LEMMA and that arm has no lemma layer, since lzh's is attached at packaging (`han_lemma_lut`). Doing
so returns a flat 0.00, which reads as a finding and is an artefact. `eval_sud_subject.lzh_rule_arm`
builds the same lemma layer the wheel ships, on demand, and evaluates against that. The `Subject` moves are seed noise, not the fix — that pipe uses the default
encoder and reads nothing structural, and model init here is unseeded.

**fa now SHIPS its `Reported` layer**, reversing an earlier decision (user decision, 2026-08-05).
That decision rested on P 0.50 — "half of what it emits is wrong" — measured before this fix.
Retrained it is **F 46.15 at P 54.55**, against its own rule's 23.53. fa remains the one language
where the trained pipe beats the rule for `Reported`, because its gold is 87 % LLM-decided and a
rule can only reach the 13 % it committed itself.

## The MISC layer is COUPLED to the arm underneath it

Every component here reads the released pipeline's own predictions (`ExtPos`, `unk`, deprel,
VerbForm/Mood, lemma), so retraining a base arm silently moves the MISC layer with it. The idiom rule
is the most exposed, being a CONJUNCTION of two predictions — upstream errors multiply rather than
add. Measured when sa switched from the freeze recipe to the joint multi-task arm (end-to-end test):
Idiom F 77.7→55.1, InIdiom 81.3→58.6, Reported 68.8→57.4 — precision holds, **recall collapses**.
**Re-run `eval_sud_idiom.py` and `eval_sud_reported.py` after any base retrain**; the gold-trees mode
of the idiom eval does not use the model and stays 100 %. What did NOT go stale:
`sud_subject_frames.py` re-harvests byte-identically after the udep-residue commit (raising
complements are `comp:obj`/`comp:obl`/`comp:pred`, never `udep`).


## ⚠ sa `Reported`: the shipped rule was never re-measured after the base moved (found 2026-08-23)

`add_sud_reported_rule.py` documents the sa rule at **F 68.8**. Re-measured through one harness, with
the rule attached to the arm it actually ships on:

| rule attached to | scored against | P | R | **F** |
|---|---|---|---|---|
| `training_sa_lemma3_noannot` (superseded csl_rev chain) | csl_rev gold | 72.93 | 65.10 | **68.79** |
| `training_sa_mp2_sub_s1` (**released**, `SA_BASE`) | csl_mwt gold | 65.56 | 38.82 | **48.76** |

The first row reproduces the documented 68.8 exactly, so the harness is sound. The second is the
configuration that ships. **The rule loses 20 F on the generation it is actually packaged onto**,
and the loss is almost entirely RECALL (65.10 → 38.82) while precision holds.

**It is not the lexicon and not the inputs**, both of which were checked before concluding:

- The lexicons in `sud_reported_data.py` are curated and LEMMA-keyed, compiled into the source
  rather than harvested at build time. The head-lemma distribution that seeded them is the same in
  both representations — **top-40 VERB lemmas 40/40 shared**, and per-lemma counts differ by 1–6 in
  the hundreds-to-thousands, tracking the 494-token difference in corpus size.
- The quotative test reads `token.lemma_`, not the form. `iti` has 627 gold lemmas in both
  representations (though only 523 surface FORMs in `csl_mwt`, which keeps the sandhied surface),
  and **both arms' lemmatisers recover all 627 at 100 %**.

So the rule's inputs are intact and what changed is the TREE it reads. This is standing hazard 5 in
CLAUDE.md — *"re-measure the MISC layer after any base retrain; every pipe there reads the base's own
predictions"* — arriving through a base **replacement** rather than a retrain: sa moved from the
freeze-recipe csl_rev chain to the joint multi-task `mp2` arm, and the rule went with it unmeasured.

**RE-TUNED 2026-08-23: F 48.76 -> 52.33 on test.** Tuned on dev, and test was consulted three times
(baseline, stem, final) -- stated because that is three chances for a dev-fitted gain to look real.

| step | dev F | test F |
|---|---|---|
| baseline, released base | 54.04 | **48.76** |
| + speech-verb stem fallback | 57.07 | 48.85 |
| + **`iti` quotative anchor** | **58.72** | **52.33** |

Final: P 57.48 / R 48.03 against the baseline's P 65.56 / R 38.82 — the tune buys **+9.2 recall for
−8.1 precision**, which is a real F gain and a genuine trade, not a free one.

**1. The stem fallback** recovers `brū` from the augmented forms the base leaves unlemmatised
(`abravīt`, `abruvan`, `'bravīd`): 42 distinct surface forms over 275 tokens against exactly ONE
gold lemma the same test matches, so it recovers a verb already in `SPEECH_VERBS` rather than
widening the lexicon. `is_speech_lemma()` in `sud_reported_data.py`, shared by the gold builder and
the runtime rule so they cannot drift; on gold lemmas it is a **byte-identical no-op**, verified by
regenerating all three gold files and diffing. **On its own it bought +0.09 on test** (dev said
+3.03 — do not cite the dev figure). Kept because it is correct in kind and the anchor below needs
it, not because it earned its place empirically.

**2. The `iti` quotative anchor** is what actually paid. The main loop is COMPLEMENT-anchored, and
that anchor fails whenever the parser attaches the quote as something other than
`comp:*`/`parataxis` — 42.8 % of gold positives under `mp2` (`flat` 25, `mod` 22, `conj:coord` 4).
The second pass anchors on `iti` instead and climbs at most one step to the speech verb's dependent.

**THE CEILING IS MEASURED, AND IT IS WHY NOTHING BIGGER WORKS.** Under `mp2`'s trees, on dev
gold positives:

| reachable under the predicted tree | share |
|---|---|
| any speech verb in the sentence | 98.7 % |
| dominated at ANY depth by a speech verb | 89.0 % |
| **direct dependent of a speech verb + evidence in subtree** | **57.9 %** |
| dominated at any depth + evidence in subtree | 72.8 % |

The main loop already reached R 52.19 against that **57.9 % ceiling — 90 % of the most any
complement-anchored rule can get**. The extra 15 points live only in the any-depth band, and taking
them costs precision: a full evidence-anchored rewrite scored dev F 46.64 against the main loop's
57.07, and unrestricted climbing 45.99. One step up, keyed on `iti` alone, is the only slice of that
headroom that pays.

**Three levers measured and declined**, each on mechanism rather than on F alone:

- **`discourse` as evidence for the anchor** — admits 115 dev candidates and gets **0** right.
  sa uses no quotation characters either, so the anchor is `iti`-only.
- **Widening the accepted deprel set** — `mod` runs at 33 % precision; `subj`/`orphan` move F under
  a point on n = 3 and n = 1.
- **Requiring `clausal`** — which the code computes and never enforces, since
  `direct and not indirect and (clausal or direct)` reduces to `direct and not indirect`. Enforcing
  it collapses dev F to 18.91: sa's quoted clauses are mostly not tagged VERB/AUX, and a verbatim
  quote may be verbless. The existing design is right.

**Still open.** 52.33 is a real improvement on 48.76 but remains far below the 68.79 the rule scored
on the chain it was built for, and the residue is attachment, not lexicon — the rule's inputs are
provably intact (both arms recover all 627 `iti` lemmas at 100 %). Ship at 52.33 or drop sa's
`Reported`: still a judgement call, now a better-informed one.

**No collateral damage.** `en` 66.67 and `ar` 73.49 reproduce their documented 66.7 / 73.5 exactly;
neither has `SPEECH_STEMS` and the second pass is `sa`-only.

Reproduce (`--arm` and `--gold` were added for this):

    python scripts/eval_sud_reported.py --lang sa
    python scripts/eval_sud_reported.py --lang sa \
        --arm training_sa_lemma3_noannot/model-best \
        --gold 'assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.csl_rev.reported.conllu'
