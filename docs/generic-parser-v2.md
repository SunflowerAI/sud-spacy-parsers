# The generic parser v2: UPOS + FEATS + four typological features, no lexical channel

A single transition-based parser trained on eighty SUD treebanks at once and scored on twenty
languages whose **genus never appears in training**. Its entire view of a token is:

    UPOS        one of 17 universal categories
    FEATS       one hash table per morphological CATEGORY, not one per bundle
    typology    four 2-bit fields, constant across a document

No wordform, no affix, no shape, no script, no vector, no language identity. The question is what
that is worth on a language for which no annotated data exist.

**How this differs from v1** (`docs/generic-parser-v1.md`, branch `generic-parser`). v1 read a
cross-lingually aligned 128-d fastText vector as a fourth channel, over thirteen treebanks, and
tested by holding out one language at a time. That channel was worth +8.60 macro LAS and is gone
here, deliberately: an aligned vector needs a large monolingual corpus and a bilingual dictionary
for the target language, which a genuinely unresourced language does not have. v1's one live
finding — a graded word-order profile lifting zero-shot LAS by a mean of +12.74 — was never
reportable, because the deranged-profile control never ran and the profile was read off the held-out
language's own gold treebank. Both holes are closed here by construction.

## The four features

The user's specification: OV/VO; SV/VS; head-marking/dependent-marking (agreement vs case); and
sex-based noun classification, with the first three two-hot because their values are not mutually
exclusive.

Encoded as four 2-bit fields, `[OV VO SV VS HM DM SEX NOSEX]`:

| code | meaning |
|---|---|
| `11` | both attested — German OV+VO, Arabic SV+VS, Latin double-marking |
| `10` / `01` | one of them |
| `00` | **unknown** — no evidence, not "in the middle" |

`00` replaces v1's separate `known` flag and preserves the same distinction, which this repo has
already paid for once: an unmeasured value that renders like a measured one cost Sanskrit 6.8 LAS
through `set_morph("")`. The fourth field is a one-hot over two bits rather than a true two-hot,
since a language either has sex-based class assignment or does not; the second bit exists so that
"no sex-based gender" is distinguishable from "not known".

**The known overload:** a genuinely isolating language is neither head- nor dependent-marking and
also lands on `00`. The `g2_typ12` arm measures what that costs by adding one `measured` flag per
field.

**Why binary here when v1 argued for graded.** v1 binarised its nine graded parameters and got 12
distinct profiles out of 13 languages — a language identifier in disguise, and a language identifier
measured at −0.02 macro LAS, i.e. nothing. Eight bits over eighty languages collide heavily
(29 distinct profiles over 83 training languages, a distinctness of 0.35), so languages must share
profile rows. That sharing is the only mechanism by which the channel can reach a language it has
never seen, so the collision is the point rather than a defect. `check_generic_inputs_v2.py` fails
the build if distinctness rises above 0.6.

## Two profile sources that must never cross

| pool | source | why |
|---|---|---|
| training languages | their own treebank (`build_typology_v2.py`) | complete, uniform, and legitimate — the treebank is in training |
| test languages | Grambank, WALS, or the descriptive literature (`typology_external.py`) | **never the test treebank.** A profile read off the data being tested on is an oracle, and that is what left v1's +12.74 ungated |

Every bit carries its own `source` string, and check 8 of the go/no-go script refuses the build if
any test language carries a bit sourced from a treebank.

Grambank is the better source for three of the four fields because its features are natively binary
and non-exclusive — exactly the two-hot semantics required, where WALS's mutually exclusive value
sets have to be translated:

| field | source | coverage |
|---|---|---|
| HM | Grambank GB089–GB094 (S/A/P indexed by an affix or clitic on the verb) | ~2 310 languages |
| DM | Grambank GB070 + GB072 (morphological case, non-pronominal, core and oblique) | ~2 260 |
| SEX | Grambank **GB051**, "gender/noun class system where sex is a factor in class assignment" — the requested feature verbatim | 2 206 |
| OV/VO | WALS 83A, Grambank GB133 as fallback | 1 518 / 2 336 |
| SV/VS | WALS 82A, Grambank GB131 as fallback | 1 496 / 2 348 |

**GB071/GB073 (pronominal case) and GB408/409 (flagging) are deliberately excluded.** The treebank
predicate measures `Case` on NOUN/PROPN, so including the pronominal features would give the DM bit
one meaning in training and a broader one at test time. English is the case in point: no nominal
case, but `him`/`them` carry it.

### Traps in the join, each of which returned a wrong answer silently

- **Grambank's `ISO639P3code` column is empty on every row.** It is keyed by Glottocode alone, so an
  iso3 join returns *nothing at all* — not a few gaps. Before the Glottolog bridge, head/dependent
  marking coverage read 30 of 156 languages; after it, 102. Anything reading that first number would
  have concluded the databases do not cover the feature.
- **UD names macrolanguages where Glottolog names individuals.** `ara`, `fas`, `est`, `ori`, `uzb`,
  `yid` are not languoids, so the bridge cannot resolve them either; `ISO_ALIAS` maps each to the
  member the treebank actually represents.
- **iso3 → WALS code is many-to-one.** WALS carries twenty-one separate "Arabic (…)" rows. The join
  is by Glottocode and resolves to the entry with the most coverage, recording `wals_code` per
  language so it is auditable.

### Calibration: how far apart the two sources are

Both paths were run over every language and compared where both exist
(`compare_typology.py` → `assets_typ/typology_agreement.json`). This is the only error bar available
on the test-side profiles.

| field | identical | no external value |
|---|---|---|
| O/V | 82/116 = 0.71 | 40 |
| S/V | 76/116 = 0.66 | 40 |
| mark | 53/102 = 0.52 | 54 |
| gender | 69/102 = 0.68 | 54 |

**54 % of the disagreement is the treebank being unable to measure, not contradicting** — a small
treebank, or a feature the annotation does not record. The rest is mostly methodological: the
treebank sees both orders above threshold in a corpus (`11`) where a grammarian records a dominant
order (`10`), which is Latin exactly.

⚠ **Do not tune the thresholds to maximise agreement.** The treebank predicates answer "what does
this treebank annotate"; the databases answer "what do descriptive grammars say". Forcing them
together fits the instrument to the thing being measured. **English is the standing example**: UD
annotates `Person` on 87 % of finite verbs — 2 355 tokens of `Person=1`, where English has no overt
first-person affix at all — because it records agreement potential rather than overt marking. The
treebank predicate therefore calls English head-marking and WALS 23A does not. That is a finding
about UD's annotation policy, not a bug in either source.

### Three predicates that were wrong before they were right

Each was measured against the local treebanks and produced a confidently wrong profile.

- **The core-case guard.** Classical Chinese carries `Case` on 31 % of its nouns — `Case=Loc`
  37 529 and `Case=Tem` 7 688, and nothing else. Those are semantic labels on locative and temporal
  nouns, not morphological case. Without requiring the inventory to intersect {Nom, Acc, Erg, Abs},
  an isolating language is called dependent-marking.
- **The `VerbForm=Fin` fallback.** Sanskrit-Vedic annotates *zero* `VerbForm=Fin` while carrying
  Person on ~90 % of its verbs. Restricted to finite verbs the predicate finds no denominator and
  calls Sanskrit unmarked. Worse, an earlier version divided a count over all verbs by the finite
  ones and produced `p_Person = 184.07` — a proportion above 1, which was the tell.
- **The gender domain.** French GSD and Arabic PADT annotate `Gender` on 63–100 % of DET and ADJ and
  on **zero** nouns. A NOUN-only predicate calls French and Arabic genderless. The rate is taken as
  the maximum over NOUN, DET and ADJ — gender on a determiner is agreement *with* a noun — while
  PRON is excluded, because referential pronoun gender (English `he`/`she`) is not a classification
  of nouns. Then the rate must count *any* gender annotation rather than sex-marked tokens
  specifically: Tamil's noun genders are Neut 2 269 / Com 526 / Masc 1, a rational/irrational split
  in which sex is plainly a criterion but marks a fifth of the nouns.

## The corpus

SUD 2.18, all 352 corpora, one tarball. The thirteen treebanks this repo already carries are the
same release, so nothing straddles two generations.

**Sixteen corpora ship with no train/dev/test split at all** — one `.conllu` per text, the Grew
convention — and they are very nearly the whole SUD-native set: all four Hausa varieties, Naija,
Haitian Creole, Ika, Beja, Zaar, Northwest Gbaya, Pesh, Bokota, Nenets, French ParisStories and
Rhapsodie. Dropping them on a filename convention would have removed the least Eurasian, least
converted-from-UD data in the release. `split_unsplit_sud.py` carves them 80/10/10 by **document**
where there are ten or more (so a text is not split across train and test, since consecutive
sentences share speaker and topic) and by sentence below that, following `split_yue.py`.

**`SUD_French-GSD` chunks its train split across five files**, `fr_gsd-sud-train_A.conllu` …
`-train_E.conllu`. A suffix match on `-train.conllu` scored it at zero train tokens while its `-dev`
and `-test` files kept it looking like a healthy test-only corpus, and 354 647 tokens vanished —
French was then represented by a 39 k spoken corpus.

### Selection

`build_tb_inventory.py` records per corpus: token and sentence counts per split, UPOS/FEATS/lemma
fill, the `@`-stripped deprel inventory, `udep`/`comp:obl`/`unk` rates, MWT and empty-node presence,
licence, and family/genus/iso3 from UD's `codes_and_flags.yaml`. Exclusions run in a fixed order and
each is written to `excluded.json` with the rule that dropped it, because a predicate that
accidentally matches half the release shows up later as a hole in the sample rather than as an
error.

Eligibility is **two booleans, not one**: a corpus with no train split is still an excellent
zero-shot test set, and v1's blanket rule would have discarded exactly the low-resource languages
this arm exists for. One corpus per language is decided *before* any cell is computed — local, then
SUD-native, then largest train, then alphabetical — so the sampler cannot pick whichever treebank
lands in the cell it needs filled.

### Cells and the split

A **cell** is the first three fields only: OV-axis × SV-axis × marking-axis, 36 nominal cells of
which 20 are populated. Gender is a secondary stratifier *within* a cell, not a fourth axis: the
full cross-product is 256 cells against eighty languages, and gender is the field the databases
cover worst.

A language is celled by the profile it will **actually be conditioned on**, so test eligibility
requires a complete external profile. Training budget is equal **per cell**, via v1's `allocate()`
with its intra-group redistribution.

**The split is genus-disjoint, and a genus moves as a whole.** Two things this required:

- **Romance is locked out of the test pool because Latin is pinned to training.** The UD taxonomy
  calls Latin "Italic" and Spanish "Romance", so genus-disjointness passed on a technicality while
  eight of Latin's own daughters — Spanish, French, Portuguese, Catalan, Italian, Galician, Occitan,
  Sicilian — sat in the test set. Zero-shot would have been measuring descent. `GENUS_KIN` locks
  historically continuous genera together.
- **Every other language of a test genus leaves the training set too.** Choosing a genus for test
  because *some* of its languages are testable left the rest in training: Western Armenian was
  training data while Eastern and Classical Armenian were being scored as zero-shot, and likewise
  standard Albanian against Gheg, and Egyptian against Coptic. Those three are now dropped from both
  pools and named in the log.

### What came out

80 training languages / 1 019 709 tokens / 14 families. 20 test languages / 233 143 tokens /
12 families across five macroareas: Basque, Georgian, Coptic, Wolof, Yoruba, K'iche', Xavante,
Bororo, Thai, Vietnamese, Chhintange, Komi-Zyrian, Hungarian, Latvian, Lithuanian, Gheg Albanian,
Modern and Ancient Greek, Eastern and Classical Armenian.

Thirty deprels after coarsening, and no label appears in test that is unattested in training.

### What the balance actually achieved, and the ceiling it needed

Equal budget per cell neutralises **corpus size** completely: German has 2 799 069 training tokens
available and contributes 44 197 of them; Chinese has 197 228 and contributes 6 963; Romanian has
532 881 and contributes 50 005. Every language is capped at its cell's equal share, so a large
treebank contributes no more than a small one beside it.

It does **not** neutralise **cell sparsity**, and that turned out to matter more. At an uncapped
100 k per cell, seven languages sat alone in their cell and took **26.6 % of all training tokens** —
Norwegian, Dutch, Latin and Swedish alone were 24 % — because a cell with one treebank is weighted
the same as a cell with sixteen. Two of the seven were alone only because their treebank failed to
*measure* marking (`VO|SV+VS|?`), so the cell was an artefact of the `00` overload rather than a
typological class.

Two ceilings fix this, and they do different jobs.

**`--max-lang-tokens 40000`** caps any one language. Cells of three or more languages are untouched,
since their equal share is already below the ceiling; only the sparse ones shrink.

**`--min-cell-tokens 20000`** protects a thin cell from the family ceiling — see below; it is
applied first, and a protected language is exempt from the scaling.

**`--family-ceiling 0.40`** caps any one *family's* share of train tokens. This is the one that
matters, because the first ceiling barely moved the genetic skew: an over-represented family has its
per-language caps scaled down and the cell budgets re-allocated, and `allocate` then hands the freed
budget to the other languages in the same cell, which are by construction from other families. The
loop is damped and floored, because scaling a family to nothing in one step would empty every cell
it is alone in rather than merely shrink it.

| | uncapped | 40 k lang cap | + family ceiling | + cell floor |
|---|---|---|---|---|
| train tokens | 1 545 256 | 1 210 348 | 804 957 | **1 019 709** |
| largest single language | 100 103 | 40 273 | 40 273 | 40 273 |
| top-10 languages' share | 47.2 % | 33.1 % | — | — |
| languages alone in a cell | 26.7 % | 16.9 % | — | — |
| thinnest cell | 10 294 | 10 294 | **1 962** | **10 294** |
| **Indo-European by token** | **72.5 %** | **68.3 %** | **40.3 %** | **40.9 %** |

**The target is the test set's own distribution**, not an abstract notion of fairness. A model whose
prior is two-thirds one family is not the language-agnostic parser this arm claims to be, and the
aggregate zero-shot number would then be measuring how far IE-shaped knowledge reaches rather than
what typological conditioning buys.

| family | train tok | train langs | test tok | test langs |
|---|---|---|---|---|
| IE | **40.9 %** | 47 | **39.9 %** | 7 |
| Afro-Asiatic | 16.7 % | 7 | 4.4 % | 1 |
| Uralic | 10.5 % | 4 | 7.9 % | 2 |
| Turkic | 8.6 % | 5 | 0 % | 0 |
| Sino-Tibetan | 3.7 % | 4 | 6.3 % | 1 |
| Niger-Congo | 0.3 % | 1 | 8.0 % | 2 |
| Basque / Mayan / Austro-Asiatic | 0 % | 0 | 8.6 / 4.3 / 5.0 % | 1 each |

Indo-European is matched to within one point. The remaining rows are **not** defects: a test family
with no training presence is the zero-shot condition working as intended — Basque is an isolate, and
Mayan and Austro-Asiatic have no training representation at all. Niger-Congo at 0.3 % train against
8.0 % test is the sharpest such case, and Wolof and Yoruba should be read with that in mind.

### The cell floor, and why the ceiling needed one

An unguarded family ceiling attacks exactly the wrong cells. Six cells are a single Indo-European
language — Manx, Faroese, Latin, Swedish, Norwegian, Dutch — so scaling IE down cut the
configurations with only one representative to 1 962, 4 475, 7 528, 7 569, 7 625 and 7 659 tokens.
**Three of those cells have a test language scored against them**: Latin's cell is what Armenian is
tested on, and it had been reduced to 7 528 tokens.

`--min-cell-tokens 20000` exempts a language from the family scaling when its cell would otherwise
fall below the floor. The protection is **sticky and only ever restores a cap**, so the loop cannot
oscillate. Fourteen languages end up exempt: `bej de fa fo ga gd gv la myv nds nl no orv sv`.

It is close to free. The IE share moves from 40.3 % to **40.9 %** — still within a point of the test
set's 39.9 % — and it **recovers 215 000 tokens**, because the budget freed in over-full cells lands
in cells that were starving. Two cells remain under the floor, and both are at their language's
absolute limit rather than being held back: Erzya contributes all 10 294 tokens it has, Manx all
10 378.

### Alignment: the gap the four features do not cover

**The training set is effectively nominative-accusative and the test set is not.** `Case=Erg` and
`Case=Abs` appear on **458 of 1 019 709 training tokens — 0.045 %** — all of them from two tiny
Chibchan treebanks, Ika (424) and Pech (34). On the test side the same values appear on **7 636 of
233 143 tokens, 3.275 %**: a rate 73× higher.

| | ergative languages | Case=Erg/Abs tokens |
|---|---|---|
| train (80 langs) | 2 (Ika, Pech), both Chibchan | 458 = 0.045 % |
| test (20 langs) | Basque, Chhintange, Georgian, and K'iche' head-marking | 7 636 = 3.275 % |

Grambank agrees on the shape of it: among the languages it covers, 2 of 34 training languages have
ergative flagging against 3 of 16 test languages.

**This is a property of the treebank inventory, not a sampling mistake, and it is not fixable
here.** The only train-eligible ergative treebanks in all of SUD 2.18 are Ika, Pech, Basque,
Georgian and Chhintange — and the last three are *test* languages precisely because their genera are
absent from training. Moving one to the training side would remove it from the test set and take the
typological diversity with it. There is no fourth option in the release.

**Two consequences to carry into the results.**

1. `Case=Erg` and `Case=Abs` are FEATS values whose hash-embed rows are trained on 106 and 352
   tokens. For Basque, Chhintange and Georgian these are frequent and highly informative, and the
   model has effectively never seen them.
2. **Alignment is not one of the four typological features**, so nothing tells the model that a test
   language is ergative. It parses Basque with a prior learned from data that is 99.955 %
   nominative-accusative.

Adding alignment as a fifth field would not repair this: a feature attested by two training
languages at 0.045 % of tokens cannot be learned, let alone transferred. The honest handling is to
**report the ergative test languages separately** rather than let them sit inside a macro average,
and to treat any deficit on them as predicted rather than discovered.

### Valency morphology, and why FEATS measures annotation rather than language

`Voice=Cau` is on 142 training tokens (0.016 %) and 593 test tokens (0.254 %) — a 16× higher rate on
the side the model has never seen. Bororo alone carries three times more causative annotation than
the whole training corpus.

**But this is an ANNOTATION gap, not a typological one, and the difference matters.** The training
languages are full of causatives; their treebanks simply do not put them in `Voice`:

| language | FEATS fill | `Voice` values | where its causative actually is |
|---|---|---|---|
| Japanese | 4.1 % | none at all | a separate AUX token — させ/せる, 448 occurrences, attached `comp:aux` |
| Korean | 4.7 % | none at all | inside the eojeol, or periphrastic `-게 하다` |
| Telugu | 2.3 % | none at all | MTG annotates 115 FEATS tokens in 5 097 |
| Tamil | 87.7 % | `Act`, `Pass` only | richly annotated, but TTB's Voice inventory has no causative value |
| Turkish (Penn) | 67.6 % | `Pass`, **`Cau` 477**, `Rfl`, `CauPass`, `Rcp` | in `Voice` — which is why Turkish supplies most of our training causatives |

Japanese has not lost its causative: SudachiPy segments `-sase-` as its own auxiliary, so the parser
sees it structurally as an AUX dependent even though the `Voice` channel is blind to it.

⚠ **This generalises to the whole design. The FEATS channels measure annotation practice at least as
much as morphology**, exactly as English carrying `Person` on 87 % of finite verbs does. A corpus
balanced on typological features derived from FEATS is balanced over what treebanks RECORD, not over
what languages DO — and nothing in the pipeline can tell the two apart.

**Applicatives run the other way and are never exercised.** `Voice=Appl` is 126 training tokens
(Pech 73, Khoekhoe 53) and **zero** in test, with `ApplPass` 11, `ApplRfl` 5, `MidAppl` 3 and
`ApplMid` 2 likewise training-only.

**Five `Voice` values appear in test and never in training**, and because a FEATS value is hashed as
a whole string, a composite shares nothing with its parts — `CauRfl` is unrelated to the `Cau` the
model saw 142 times:

| value | test tokens | language |
|---|---|---|
| `Voice=Antip` | 33 | K'iche' |
| `Voice=AgFoc` | 22 | K'iche' |
| `Voice=CauRfl` | 7 | Chintang |
| `Voice=CauInc` | 1 | Bororo |
| `Voice=CauRcp` | 1 | Chintang |

Antipassive and agent-focus are the two central valency operations of the Mayan focus system, so
K'iche' meets the parser with its most characteristic morphology as unseen symbols. **This is an
argument that the FEATS decomposition does not go far enough**: v1 decomposed FEATS from bundle to
category, but a category's VALUES are still opaque strings.

### What the balance cost

- **34 % of the training tokens**, 1 545 256 → 1 019 709. That is the price and it was taken
  deliberately.
- **Indo-European is still 47 of 80 languages (59 %)**, and that is left alone deliberately. What
  the model sees is tokens, not language names; dropping IE languages would empty cells that only
  IE populates, and more languages at fewer tokens each is the safer direction for an arm whose
  whole risk is memorising one treebank's idiosyncrasies.
- The relabelled share of train tokens rose to **25.9 %**, since this repo's thirteen are mostly
  non-IE and so survive the ceiling. The `udep` skew moved with it and should not be over-read as a
  designed effect.

### Label coarsening, wider than v1's

v1 stripped `@` subtypes and got 27 labels from thirteen treebanks, on the reasoning that `:` is
core SUD. Across a hundred treebanks that is no longer true: the release carries `mod:periph$cond`,
`comp:obj$utter`, `compound:svc$purp`, `dislocated:obj`, `parataxis:parenth` and a long tail, about
a hundred labels in all, most attested in one treebank and some on a single token. With
`min_action_freq = 1` every one becomes a parser action, and a label a test language uses but
training never saw is unreachable by construction. `coarsen_deprel` strips `@`, `$` and `/` markers,
keeps a `:` component only for the eight relations SUD actually defines with one, and maps `appos`
to `conj:appos` — which was the single label appearing in test and never in training.

### Two more leaks and a tag

- **`NameType` (Giv/Sur/Geo/Nat) is now stripped**, alongside v1's `Shared`. It is on 39 205 tokens
  of Classical Chinese alone and it tells the parser that a token is a personal name — information
  about the *wordform*, in an arm whose whole claim is that it reads no wordform. `Typo`, `Style`,
  `Foreign`, `Abbr` and `Hyph` go with it as annotation practice rather than morphology.
- **Non-UD UPOS values are mapped to `X`.** SUD_Hausa writes `IDEO` for ideophones, and spaCy
  refuses the whole file with E1021 rather than the token — eight tokens in 1.88 M cost a whole
  split. UPOS is this arm's primary channel, so an out-of-inventory tag is genuinely unknown to it.

### The label-policy split, kept deliberately

This repo's thirteen treebanks carry an LLM pass splitting `udep` into `comp:obl` and `mod`; the
other sixty-seven and *every* test treebank are stock SUD. Keeping the relabelling was a deliberate
choice, and it is cheaper than feared: relabelled corpora are **25.9 % of training tokens**, and the
`udep` rate is **7.53 % in train against 7.37 % in test** after the family ceiling — a 0.16-point gap. It is nonetheless
reported rather than absorbed: `eval_generic_v2.py` prints a **collapsed-label LAS** with
`udep`/`comp:obl`/`mod` merged into one label, which is the figure immune to the confound, and
`g2_stock` prices the split directly.

## The arms

| arm | differs by | isolates | seeds |
|---|---|---|---|
| `g2_base` | no typology channel at all | UPOS + FEATS alone | 3 |
| **`g2_typ`** | + the 8-bit channel | **the headline** | 3 |
| `g2_typ_ctl` | the channel carrying zeros | what the **parameters** buy | 3 |
| `g2_typ_der` | the real profiles on the wrong languages | what the **right** profile buys — **the gate** | 3 |
| `g2_nofeats` | no FEATS | the floor for a FEATS-less test language | *deferred* |
| `g2_langid` | + a language embedding | what knowing the language buys (held-in only) | *deferred* |
| `g2_typ12` | + 4 `measured` flags | what the `00 = unknown` overload costs | *deferred* |
| `g2_feats_all` | `--min-langs 1` | what the FEATS channel cut cost | *deferred* |

The four diagnostics are **deferred until the headline is settled**. They answer secondary questions
and their configs are generated and checked, so they can be run later without rebuilding anything;
`g2_feats_all` alone is about half their cost, at 91 embed blocks against 34. Nothing in the
headline claim depends on them: that rests on `g2_typ` against `g2_typ_der` and the baseline bar.

`max_steps` is **20 000**, not 30 000. v1 reached its best dev score at step 13 200 of 20 000, and
patience is 4 000, so the cut mostly removes steps after the plateau. It has to be identical across
arms: an arm allowed to train longer than its own control can find a better checkpoint on dev, which
is a confound rather than a saving — the in-flight arm was restarted rather than left at 30 000.

`g2_base` and `g2_typ_ctl` are both needed and are not the same arm: the first has no typology block,
the second has the block with nothing in it. **The delta that gets quoted is against `g2_typ_ctl`**,
because adding a block adds `width × nP × width` Maxout weights and a gain from parameters is not a
gain from typology.

**`g2_langid` is a train-side control by construction.** v1 emitted an all-zero row for an
unrecognised language, which let a language-embedding arm run on a zero-shot language and be
reported beside the typology arm. v2 raises instead: a language embedding has no row for a language
it never saw, and that is the whole reason a typological profile might do better.

### FEATS channels: `--min-langs 5`, against v1's 1

| setting | channels | `n_blocks` | Maxout params |
|---|---|---|---|
| v1 (13 treebanks, `min_langs 1`) | 42 | 44 | 2.16 M |
| v2 `min_langs 1` | 89 | 91 | 4.47 M |
| **v2 `min_langs 5`** | **32** | **34** | **1.67 M** |

v1 kept every category present anywhere, on the grounds that a per-language category is cheap and
dropping it taxes exactly the low-resource languages the arm exists for. That held when all thirteen
languages were *also* test languages. Here the test languages were never seen in training, so a
category attested in one training treebank out of eighty cannot transfer by construction — it can
only ever fire on that treebank. `g2_feats_all` measures the cost of the cut rather than assuming
it.

## Baselines, measured before any arm

Macro LAS over the twenty test languages, all through the same scorer, the same punctuation
exclusion and the same gold sentence boundaries as the arms.

| baseline | macro LAS |
|---|---|
| attach left | 13.85 |
| attach right | 18.84 |
| **UPOS-pair majority** (table from training only) | **27.52** |
| typology-conditioned chain | 17.63 |
| *gold heads + majority label* (labelling **ceiling**, not a baseline) | *54.24* |

**The bar a trained arm has to clear is 27.52.**

⚠ **The typology-conditioned chain — the same eight bits used by two lines of code — scores 17.63,
below plain right-attach.** So the bits used trivially are worth nothing, and any gain `g2_typ` shows
over `g2_typ_der` cannot be dismissed as a head-direction prior expressible without a parser. That
is worth knowing *before* the neural numbers exist rather than after, which is why it is a pipeline
stage and not an afterthought.

The gold-head row is the other half of the picture: with perfect attachment, a majority label per
UPOS pair reaches only 54.24, so labelling is a hard problem here in its own right.

## Go/no-go

`check_generic_inputs_v2.py`, 23 assertions, run before any training. Two have already fired on this
build:

- **Check 4** caught seven test languages carrying `-train.conllu` files left behind by an *earlier
  split*. The reader discovers languages by which files exist, so a stale file silently
  un-holds-out a language and nothing downstream can tell. `prep_generic_v2.py` now clears the
  output directory first.
- **Check 3b** caught the Armenian/Albanian/Egyptian genus leak described above.

And one was a false positive worth recording: **check 2b matched the wordforms `Sudan`, `Sudeste`
and `Sudetenland`** when grepping lines for a `Sud`-prefixed FEATS key. It now parses the FEATS
column. A check that fails on the wrong thing costs the same attention as one that passes on the
wrong thing.

**Check 8 is the gate**: no test language may carry a bit whose source is `treebank`.

## Results

All figures are macro LAS over the twenty held-out languages, gold sentence boundaries, gold
UPOS/FEATS, the 30-label target, three seeds unless marked. The bar is the UPOS-pair baseline at
**27.39**.

### The typology channel fails, and its own gate says why

| arm | mean | spread |
|---|---|---|
| `g2_base` — no channel | 54.11 | 53.96–54.24 |
| `g2_typ_ctl` — channel carrying zeros | 54.57 | 54.11–55.29 |
| `g2_typ_der` — **wrong** profile (the gate) | 53.40 | 53.26–53.49 |
| `g2_typ` — Grambank/WALS profile | 53.28 | 52.58–53.87 |

`g2_typ` beats `g2_typ_der` by **−0.12**. The correct profile is worth nothing over a deliberately
deranged one, and both sit below an empty channel. **The channel costs about a point and costs the
same whether its content is right or wrong.**

### Profile accuracy is not the constraint — the channel is

Because the profiles are baked into the trained layer, they can be swapped at evaluation time
without retraining. Three sources, on the fifteen test languages that have unscored data to profile
from:

| profile source | field accuracy vs the treebank | macro LAS |
|---|---|---|
| Grambank/WALS | 0.62 | 54.62 |
| 50-sentence sample | 0.73 | 54.17 |
| **200-sentence sample** | **0.90** | **56.39** |
| 500-sentence sample | 0.93 | 56.37 |
| oracle (the full treebank) | 1.00 | 56.38 |
| *`g2_typ_ctl`, an empty channel* | *—* | *56.33* |

Two hundred hand-annotated sentences reproduce the full-treebank oracle to two decimal places, so
profile quality is obtainable cheaply. **But the empty channel scores 56.33.** A perfectly accurate
profile is worth **+0.06** over carrying nothing at all. The only reliable effect of the channel is
the ~1.8 LAS it *costs* when the profile is wrong, and Grambank/WALS is wrong enough to pay it in
full.

That also retroactively explains v1's +12.74: its profiles came from the held-out language's own
gold treebank.

### A trainable language embedding, fitted on a handful of sentences, is what works

`g2_langemb` replaces the four bits with a per-document lookup into a trainable table — no
typological input at all — with spare rows so a language met after training can be given one and
fitted while every other parameter stays frozen. The freeze is enforced by wrapping the optimizer
and **verified**: max drift in any frozen parameter, 0.000e+00, and no row but the target moves.

Basque, seed 0:

| condition | LAS |
|---|---|
| `g2_typ` — external profile | 44.74 |
| row fitted on the **wrong** language (200 Wolof sentences) | 45.35 |
| `g2_base` | 46.84 |
| `g2_typ_ctl` | 47.44 |
| spare row assigned but **not** fitted | 48.74 |
| **row fitted on 200 Basque sentences** | **53.18** |

Three separable effects: the architecture is worth +1.30 unfitted, fitting on the target language
+4.44, and fitting on the *wrong* language −3.39. The last is the same "wrong information hurts"
signature the typology channel showed, which is how we know the channel is being read.

### Ten sentences is enough

| | `g2_base` | `typ_ctl` | unfitted | N=10 | N=25 | N=50 | N=100 | best |
|---|---|---|---|---|---|---|---|---|
| Basque | 46.84 | 47.44 | 48.74 | 49.79 | 51.58 | 53.05 | 53.60 | 53.60 |
| Thai | 36.70 | 39.20 | 45.62 | **57.80** | 59.76 | 58.92 | 60.60 | 60.60 |
| Georgian | 64.71 | 64.90 | 60.58 | **67.20** | 68.03 | 68.22 | 69.04 | 69.24 |

129–188 tokens. Thai gains **+12.18 from ten sentences**; everything past 100 is flat, and 400 is
slightly worse than 100 in two of three.

⚠ **The sample and the test set come from the same treebank** — different sentences, same genre and
annotators. Part of this is domain adaptation rather than language adaptation, and Thai's +21.9 over
`g2_base` is large enough that a substantial share probably is. That is the realistic deployment
case, but it is not the claim "the model learned Thai".

⚠ **Georgian shows the channel is not free when unfitted**: 60.58 against `g2_base`'s 64.71, a
4-point cost that adaptation then more than recovers.

### The embedding cannot be predicted from Grambank, at either width

| | d=128 | d=8 |
|---|---|---|
| in-sample dev LAS | 80.03 | 79.41 |
| Grambank LOO cosine (mean baseline) | 0.226 (−0.061) | **0.415** (−0.232) |
| Basque, unfitted row | 48.74 | 48.96 |
| Basque, **Grambank-predicted** | 48.73 | **46.78** |
| Basque, 200-sentence fitted | 53.18 | 50.75 |

Compressing to eight dimensions nearly doubles predictability — a trained 8-d space is far better
organised than a 128-d one truncated to 8, where the top eight principal components hold only 51 %
of the variance and recover only half the adaptation gain. **But the prediction still does not
parse.** At d=8 it scores 2.18 points *below* doing nothing. The trade runs the wrong way: widening
makes the embedding useful and unpredictable, narrowing makes it predictable and less useful, and at
neither end does prediction beat an unfitted row.

The learned space explains why. Same-genus pairs sit at cosine +0.154 against −0.016 across genera,
and correlation with typological bit-distance is r = −0.303 — real but weak. Nearest neighbours are
suggestive rather than typological: English→Naija, **Japanese→Urdu**, Turkish→Tamil,
Hebrew→Middle French. The space is mostly not typology; it is whatever residual each treebank needs.

### What this line of work concludes

**Information about a language that is merely correlated with the truth costs more than it pays.**
That held for four typological bits from Grambank, for the same bits from a 50-sentence sample, for
a 128-d regression and for an 8-d one. The only thing that paid was fitting the language's own
vector on its own data — and ten sentences was enough.

## Files

| file | purpose |
|---|---|
| `scripts/fetch_sud_release.sh` | the SUD 2.18 tarball; refuses on a corpus count ≠ 352 |
| `scripts/split_unsplit_sud.py` | carves the 16 corpora the release ships unsplit |
| `scripts/build_tb_inventory.py` | per-corpus stats, exclusions with reasons, one corpus per language |
| `scripts/build_typology_v2.py` | treebank-derived profiles — **training languages only** |
| `scripts/typology_external.py` | Grambank + WALS + literature — the test-language path |
| `scripts/compare_typology.py` | agreement between the two, i.e. the error bar |
| `scripts/prep_generic_v2.py` | cells, the genus-disjoint split, the balanced sample |
| `scripts/sud_generic_embed_v2.py` | `sud.GenericEmbed.v2` |
| `scripts/make_generic_config_v2.py` | the eight arm configs |
| `scripts/baseline_generic.py` | the five trivial baselines |
| `scripts/check_generic_inputs_v2.py` | the 23 go/no-go assertions |
| `scripts/eval_generic_v2.py` | zero-shot scoring; refuses a headline without baselines |
| `scripts/train_generic_v2.sh` | the driver |
