# Classical Chinese tokenisation, and the Heart Sūtra gold set

Extracted from `CLAUDE.md` so the main guide stays short. Read this before touching lzh
tokenisation, and before reaching for any lexicon or gazetteer to improve it.

## The thing CLAUDE.md's wheel table gets wrong

**lzh's released tokeniser is "one Han character = one token", and the treebank is not.** 26 190 of
Kyoto's 920 780 tokens (**2.84 %**) are multi-character — 5 069 types: honorific compounds,
philosophers' names, ethnonyms, toponyms and numerals. So the shipped tokeniser splits 孔子 into
孔 + 子 and cannot exceed ~97 % token F by construction.

⚠ **This was invisible in every published lzh figure, and the reason generalises.** `gold_preproc`
bypasses the tokeniser at evaluation and `sud.GoldTokCorpus.v1` makes the parser
segmenter-agnostic, so nothing in the standard metrics touches tokenisation at all. It shows up only
in raw end-to-end token F, which had never been measured for lzh. This is standing hazard 4 in a new
place.

`scripts/train_lzh_charseg.sh` trains the replacement (`models/lzh_seg_char`, the same
`sa_presegment.Presegmenter` character tagger that serves zh and id). **In-domain it scores strict
token F 0.9835 — but multi-char P 0.7818 / R 0.7359**, and the multi-char row is the one that
matters, since single characters need no decision. It is NOT yet bundled into the wheel.

## The Heart Sūtra gold set

`assets_cbeta/heart_sutra_gold.txt` — 330 characters, **281 tokens, 25 multi-character (8.90 %)**,
three times Kyoto's in-domain rate. Built by `scripts/make_heart_sutra_gold.py`, scored by
`scripts/eval_heart_sutra.py`. Source is CBETA T08n0251 via `scripts/cbeta_text.py`.

⚠ **Only `<div type="jing">` is the scripture.** T08n0251 also carries two prefaces in
`<div type="xu">` (794 + 156 chars) which are not the sūtra; extracting the whole `<body>` silently
triples the text. Hence `cbeta_text.py --div-type jing`.

⚠ **CBETA mixes namespaces inside `<body>`: `<div>` is in the CBETA namespace, not TEI.** A
namespaced `body.iter(f"{TEI}div")` returns nothing, silently. Match on local name.

⚠ **CBETA supplies no supervision.** Across all 42 files of T08 the body contains **zero** `<w>`,
`<seg>`, `<term>`, `<persName>`, `<placeName>` or `<foreign>` elements. Every `<name>` in these files
is header metadata (publisher, contributors). It is running text and nothing else.

**The annotation is a hand gold set produced by an LLM and wants specialist review.** Its
conventions were read off Kyoto rather than assumed, which settled most calls: every native
disyllabic compound in the text is unmerged in Kyoto (一切, 諸法, 無明, 老死, 三世, 恐怖, 顛倒, 夢想,
究竟, 罣礙, 真實, 眼界, 意識 — all 0), so they are split; the Buddhist units are merged following
Kyoto's counts (菩薩 43, 波羅蜜 18, 阿耨多羅 34, 三藐 30, 三菩提 30, 般若 13, 菩提 3, 涅槃 2).
**anuttarā-samyaksaṃbodhi is THREE tokens** (阿耨多羅 + 三藐 + 三菩提) because that is Kyoto's
segmentation; a gold that merged it would score the model against a convention its training data
does not use. Four deliberate departures and three uncertain calls are listed in the script's
docstring; 觀自在 is the one most worth a second opinion.

## The result: total memorisation, zero generalisation

    strict token F                       0.9128   (in-domain: 0.9835)
    multi-char recall, ATTESTED in Kyoto  11/11 = 1.0000
    multi-char recall, UNATTESTED (0x)     0/14 = 0.0000

Every miss has Kyoto count 0; every hit is attested. The spurious predictions say the same thing
from the other side: 波羅蜜 ×5 (it merges the attested three-character form and strands 多), 菩提 ×1
(inside 菩提薩埵), 莎婆 ×1 (a partial merge of 莎婆訶, **not** a generalisation).

**Kyoto is not free of Buddhist vocabulary, and assuming so is a mistake worth not repeating.**
`KR6c0023` — the Diamond Sūtra — is one of only **8 source works** in Kyoto train: 535 sentences,
5 700 tokens, 須菩提 137×, 如來 88×, 世尊 52×. That is why the segmenter recovers prajñā, pāramitā and
anuttarā-samyaksaṃbodhi on out-of-domain text. It is also why it misses 舍利子: the Diamond Sūtra's
interlocutor is Subhūti, so Śāriputra occurs **zero** times.

## What the 14 gaps decompose into, and which lever reaches each

    person / deity names (舍利子, 觀自在)          3 tokens   gazetteer + syntax
    doctrinal terms (波羅蜜多, 菩提薩埵)            6 tokens   term list, or induction from CBETA
    mantra (揭帝 ×2, 般羅揭帝, 般羅僧揭帝, 莎婆訶)   5 tokens   NOTHING LISTABLE — phonology only

**The last row is why the phonological angle is not interchangeable with the lexical one.** A
mantra has no syntax to exploit and its units are hapax: over 1 223 328 characters of T08, 般羅揭帝
and 般羅僧揭帝 occur **once each**, 莎婆訶 3×, 菩提薩埵 4×. No gazetteer and no distributional
induction reaches them. The frequent doctrinal terms are a different matter — 波羅蜜多 occurs
**2 227×** in T08, so induction is limited by the scoring method, not by the data.

## Sub-character channels: radical and Qieyun

`scripts/probe_translit_channel.py` predicts, per adjacent character pair, whether the two share a
gold token. The slice that matters is **bigrams never merged anywhere in train** (29 569 test pairs,
192 true merges) — identity is useless there by construction, so only a backoff can fire. F on the
merge class:

    arm                    all    bigram SEEN merged   bigram NEVER merged
    null (bias only)     0.000          0.000                0.000
    identity             0.682          0.877                0.201
    radical              0.339          0.885                0.055
    qieyun               0.394          0.892                0.092
    radical+qieyun       0.528          0.883                0.128
    identity+rad+qy      0.699          0.877                0.227
    gazetteer            0.264          0.507                0.054
    identity+gaz         0.680          0.878                0.204
    identity+rad+qy+gaz  0.694          0.876                0.221

**Qieyun beats the radical here (0.092 vs 0.055), reversing the sub-character probe in
`NEGATIVE-RESULTS.md`** (radical 57.00 vs Qieyun 48.06). That probe predicted *lexical class*, where
phonology is nearly irrelevant; transliteration is phonological by definition, so the earlier
ordering does not transfer and should not be cited against this. The gain is real but small:
+0.026 F, bootstrap 95 % CI **[−0.003, +0.056]**, P(Δ>0) = 0.965 — suggestive, not established, on
192 positives. Note also that the channels add **exactly nothing** on the memorised slice
(0.877 either way), which is the same dissociation the Heart Sūtra shows.

## Gazetteers: DILA, and why filtering matters more than the list

`DILA-edu/Authority-Databases` is **CC BY-SA 3.0** — no NC clause, and compatible with lzh's existing
CC BY-SA 4.0. (CBETA is **CC BY-NC-SA 4.0** for Category A, which includes Taishō vols 1–85; using
CBETA *text* in a shipped model would force the wheel to NC. DILA does not.) The person authority
gives 49 259 records / **89 039 Han name strings** (`assets_dila/person_names.txt`).

Used naively it is far worse than the trained segmenter — strict F 0.9314 against 0.9835, multi-char
P **0.1834**. Filtering, all measured with **gold trees** so these are ceilings:

    filter                                   P        R
    gazetteer alone                       0.1834   0.2570
    + complete subtree                    0.3249   0.2570
    + head POS is NOUN/PROPN              0.3632   0.2570
    + AND subject of 曰                    0.6500   0.0458
    + AND followed by a title character   1.0000   0.0129   (n=11; Wilson 95 % lower bound ≈ 0.74)
    + AND (曰-subject OR title)            0.7042   0.0587

**Syntactic context nearly doubles precision over the lexical filters** (0.363 → 0.704), which is
the trade a bootstrapping loop wants: buy precision, spend recall, seed the confident cases.

### Four traps recorded here

1. ⚠ **Subtree-hood alone is a weak filter, and the reason generalises.** It removes 520 of 975
   false positives but **455 remain, because they ARE complete subtrees** — two adjacent tokens form
   a subtree whenever one heads the other, which is the common case. Do not expect a constituency
   constraint to do heavy lifting over short spans in a dependency tree.
2. ⚠ **A gazetteer's coverage is a hard recall ceiling no filter can lift.** Only **27.93 %** of
   Kyoto's gold multi-char test tokens appear in DILA at all — PROPN 36.17 %, NOUN 12.35 %, NUM
   **0 %**. DILA is a Buddhist onomasticon; Kyoto's multi-char tokens are largely classical figures,
   toponyms and numerals.
3. ⚠ **Ask what the TREEBANK merges, not what the LANGUAGE affixes.** 公 is a title suffix in
   Classical Chinese and Kyoto merges it **7** times against **1 170** standalone occurrences
   (ceiling 0.006); likewise 王 0.001, 侯 0.015, 君 0.016. Only 子 carries mass (2 324 vs 2 421,
   ceiling 0.490), and the top finals by volume are toponym elements (陽 561, 山 299, 陵 204) and
   numerals (十 416, 萬 189, 百 132) — not titles. A closed title list covers 14.12 % of multi-char
   PROPN. The corollary also fails: when a standalone title follows, the preceding token is already
   single-character PROPN 37.4 % of the time and multi-character only 9.1 %, so there is usually
   nothing to merge — the title is a *disambiguator*, not a merge site.
4. ⚠ **Soothill–Hodous is not machine-readable in the public domain.** The Internet Archive scan
   (`in.ernet.dli.2015.367373`) has a `_djvu.txt` containing **zero CJK characters** — the OCR kept
   only the Latin. No GitHub digitisation exists; DDB is login-gated. Do not spend an afternoon on
   this again.

Also note the 曰-subject cue is weaker than it sounds on its own: subjects of 曰 are **NOUN 51.0 % /
PROPN 41.9 %** (子曰 dominates and 子 is NOUN here), so "subject of 曰 → name" is ~0.42 precise. Its
value is multi-character-ness — 27.6 % of 曰-subjects are multi-char against a 2.84 % base rate, a
ten-fold enrichment.

**Every syntax figure above uses gold trees and is therefore a ceiling.** With predicted parses they
will fall, and the honest use is offline annotation bootstrapping, where a parse is affordable —
not runtime tokenisation, where no tree exists when the merge decision is made.

## Assets and how to regenerate

    scripts/cbeta_text.py                --div-type jing extracts the scripture proper
    scripts/make_heart_sutra_gold.py     writes assets_cbeta/heart_sutra_gold.txt (round-trip asserted)
    scripts/eval_heart_sutra.py          scores a segmenter, split by Kyoto attestation
    scripts/probe_translit_channel.py    character-pair merge probe; --gazetteer adds the lexicon arms
    scripts/probe_gazetteer_subtree.py   the subtree / head-POS ceiling
    scripts/train_lzh_charseg.sh         trains models/lzh_seg_char

Gitignored asset dirs: `assets_cbeta/` (CBETA T08, NC), `assets_dila/` (person authority, BY-SA),
`assets_unihan/` (kRSUnicode, Unicode licence), `assets_qieyun/` (guangyun.csv, CC0).

## The jackknife: how to measure generalisation without annotating anything

`scripts/make_seg_jackknife.py` splits a random half of the multi-char types shared by train and
test back into single characters **in train and dev only**, leaving test untouched. 158 held-out
types become **611 held-out multi-char test tokens**, against 1 093 retained as a memorisation
control — a 44× larger denominator than the hand gold, at zero annotation cost. It self-checks that
an empty hold-out reproduces the original labels byte for byte before writing.

    model                                 token F   retained   HELD-OUT
    full data (has seen the held-out types) 0.9835    0.6761     0.8429
    jackknifed (never saw them merged)      0.9730    0.6322     0.0458

Removing the types collapses recall on exactly those types from **84.3 % to 4.6 %** while retained
barely moves. That 4.6 % is the generalisation baseline any channel must beat.

⚠ **The denominator is 611; the effective sample is the ~30 tokens actually recovered.** Control
seeds recovered 28 / 30 / 45, a 2.78-point swing that is just Poisson noise on ~30 events. Any
future experiment on this slice needs more seeds or a larger hold-out fraction — three seeds was not
enough to resolve a sub-point effect.

## Three Buddhist works, one per split — and the contamination this nearly caused

    train  KR6c0023  金剛般若波羅蜜經          Diamond Sutra              535 sents / 5,700 tok
    dev    KR6f0082  佛說阿彌陀經              Amitabha Sutra             141 sents / 1,921 tok
    test   KR6c0127  摩訶般若波羅蜜大明呪經    Heart Sutra (Kumarajiva)    56 sents /   360 tok

⚠ **Hand-annotating the Amitabha Sutra from CBETA as a "held-out" test set was one step away from
happening, and it IS Kyoto's dev split.** The tell was that 30 of 36 of its roll-call names came
back attested, most at exactly 1×. Check `sent_id` prefixes against the splits before annotating.

⚠ **Count attestation on TRAIN ONLY.** Because the Buddhist works are split one per split,
treebank-wide counts badly overstate what a model saw: 舍利弗 counts 41× overall and **0×** in
train. `eval_heart_sutra.py` was fixed to do this.

**KR6c0127 is a free second gold set** (`scripts/make_kyoto_work_gold.py`): a different translation
of the same sutra as the hand gold, in TEST so no model trained on it, annotated by the treebank's
own annotators. It also **corrected the hand annotation** — Kyoto segments pāragate as 波羅 + 竭帝,
two tokens, where the first version of the hand gold merged 般羅揭帝 as one.

    gold set                      token F   attested   unattested
    Heart Sutra (Xuanzang, hand)   0.9097   11/11      0/16
    Heart Sutra (Kumarajiva)       0.9534   23/24      4/12

Generalisation is low but **not zero** — the Kumarajiva text recovers 摩訶, 波羅 and 波羅僧, plausibly
because 波羅 prefixes the attested 波羅蜜.

## The transliteration RUN cue — the best signal found, and nearly missed

An inventory of transliteration characters is **induced, not curated**: log-odds of each character's
frequency in CBETA T08 against 42 M characters of kanripo. It reproduces the curated lists (蜜 rank
6, 訶 8, 薩 16, 菩 18, 耨 29, 羅 38, 藐 41, 迦 49, 陀 90), so Julien / Soothill are not needed. Note
**帝 ranks 976 with log-odds −1.12**, because 帝 "emperor" is ordinary classical vocabulary — 揭帝 is
visible only as a RUN, never per character.

"Run of ≥3 inventory characters" predicting "inside a multi-char token":

    Heart Sutra (Xuanzang)     P 0.907   R 0.527
    Heart Sutra (Kumarajiva)   P 0.930   R 0.596

⚠ **The aggregate hides this completely and says the opposite.** On the Kyoto-wide character-pair
probe the run feature is flat (identity+runs 0.200 vs identity 0.201) and *hurts* in combination
(0.210 vs 0.227) — because only ~7 of that slice's 192 true merges are transliteration at all. A cue
aimed at 4 % of a population cannot be read off that population's average. This was one report away
from being filed as a negative.

Re-inducing the inventory with both test sutras removed gives **identical** numbers (they are 0.1 %
of the inducing corpus), so the leak is measured rather than assumed.

### Lexicon proposes, run gates — 202 correct, 3 wrong

A run gives a REGION but no boundaries (般若波羅蜜多 is one run spanning TWO tokens), so it cannot
propose merges; it gates a lexicon that can. `scripts/eval_lexicon_runs.py`, on the two gold sets:

    lexicon      gate      correct  wrong  precision (Xz / Km)
    wiktionary   none         6/24   11/12   0.353 / 0.667
    wiktionary   run>=3       3/18    0/0    1.000 / 1.000
    DILA         run>=3       7/10    8/10   0.467 / 0.500
    both         run>=3       3/19    2/3    0.600 / 0.864

Evaluated on FOUR gold sets — the two Heart Sūtras plus Kyoto's own Amitābha (`KR6f0082`, dev) and
Diamond (`KR6c0023`, train), **746 multi-char tokens**, with the inventory re-induced excluding
T08n0235 / n0250 / n0251:

    gold set                       gold multi   correct  wrong  precision  recall
    Heart Sūtra (Xuanzang, hand)          27         3      0     1.0000   0.111
    Heart Sūtra (Kumārajīva)              36        18      0     1.0000   0.500
    Amitābha (Kyoto dev)                 179         9      2     0.8182   0.050
    Diamond (Kyoto train)                504       172      1     0.9942   0.341
    ALL FOUR                             746       202      3     0.9854   0.271

Using Kyoto's train and dev texts as evaluation is legitimate **only because the heuristic never
trains on Kyoto** — it is a Wiktionary list plus a CBETA-induced inventory. Ungated Wiktionary over
the same four is P 0.704 / R 0.610, so the gate buys 0.70 → 0.985 precision for 0.61 → 0.27 recall.
Wilson 95 % lower bound on 202/205 ≈ 0.958.

That beats the gold-tree syntactic stack's 0.704, and unlike it needs **no parse**, so there is no
ceiling to fall from. **Adding DILA makes the union worse than Wiktionary alone** — its false
positives leak through the gate. DILA earns its place on classical names (27.93 % Kyoto-wide ceiling
vs Wiktionary's 3.05 %), not on Buddhist text.

⚠ **THE GATE CATCHES DOCTRINAL TERMS, NOT THE MANTRA — an earlier claim here said otherwise.** What
it caught on the Kumārajīva text was 般若 7, 波羅蜜 7, 摩訶 2, 涅槃 1, 菩提 1; what it missed was
竭帝 ×4, 舍利弗 ×3, 阿耨多羅, 三藐. The cause is the binary threshold, and it is diagnosable per
character: 帝 scores **−1.16**, 利 **+0.57**, 弗 **+1.66**, 多 **+1.68**, all below the 2.0 cut, so
竭帝 / 舍利弗 / 阿耨多羅三藐三菩提 each break on one ordinary-looking character. **The mantra is
still unsolved by everything tested here.**

⚠ Do not confuse two gates that sound alike: "every character scores ≥ 2.0" gives P 0.83, while
"every character sits in a RUN of ≥ 3 such characters" gives P 0.95–0.99. The run condition is the
one that works, and the two were conflated once during this work.

### Wiktionary as a source

`Category:Chinese_terms_derived_from_Sanskrit` + `..._borrowed_from_Sanskrit` via the MediaWiki API:
**468 pure-Han terms** (234 "borrowed"). **CC BY-SA 4.0**, so like DILA it does not force lzh to NC.
Use "borrowed" for transliteration labels — "derived" also holds calques (七寶, 三千大千世界) whose
characters are ordinary classical vocabulary.

    coverage                     Xuanzang gaps   Kumarajiva gaps   Kyoto-wide ceiling
    Wiktionary                   8/16 = 50.0%    5/12 = 41.7%      3.05%
    DILA person authority        2/16 = 12.5%    4/12 = 33.3%      27.93%

They are **complementary**, not competing. `Literary_Chinese_proper_nouns` is EMPTY and
`Chinese_proper_nouns` is modern-Mandarin-heavy (its first entries are 110, 119, 11区), so Wiktionary
does not help with classical names. Be polite to the API — a 500-per-request loop with a 0.15 s
sleep earns a 429.

## Syntactic cues for names: which ones carry signal, and the trap in reading them

Enrichment of each context for MULTI-CHAR tokens, over train+dev+test (base rate 2.95 %):

    context                                    n      multi   enrich   PROPN
    vocative                                 201     78.11%    26.5x   58.71%
    subj of 曰                              3,504     27.60%     9.3x   41.87%
    subj of any speech verb                5,563     22.40%     7.6x   38.13%
    possessor of 固定物,地形 (terrain)        222     18.92%     6.4x   38.74%
    possessor of 固定物,建造物 (buildings)     202     14.36%     4.9x   29.21%
    subj of a speech verb, NOT 曰           2,059     13.55%     4.6x   31.76%
    possessor of 人,関係 (kin)                409     12.96%     4.4x   32.76%
    ------------------------------------ baseline ------------------------------
    subj of ANY head                      55,204      8.02%     2.7x   20.50%
    possessor (comp:obj of 之), any        9,404      7.51%     2.5x   21.43%
    subj of a PSYCH verb                   2,878      7.85%     2.7x   22.69%
    possessor of an INALIENABLE (body...)    342      4.68%     1.6x   12.28%
    comp:obj of a speech verb              9,461      3.91%     1.3x    6.12%
    possessor of 名/字/號/諱 ("name")          56      1.79%     0.6x    7.14%

**The axis is not argument structure or discourse role but whether the construction REQUIRES A
SPECIFIC INDIVIDUAL as its anchor.** Vocative (an addressee), subject of 曰 (a speaker), kin (a
relatum) and terrain/buildings (an owner) all do; possessor-in-general, experiencer and patient all
admit generics and all sit at the 2.5-2.7x baseline. This predicts in advance which further cues are
worth testing, which is the only reason the table is worth keeping.

⚠ **Inalienable possession goes the WRONG WAY (1.6x, below the general possessor rate).** Body parts
are inalienable in the unhelpful sense -- everyone has one, so 其身 / 民之心 / 人之性 take generic
possessors. Kin terms are relational and anchor to somebody in particular. Do not reason from
"inalienable" to "personal".

⚠ **`mod@poss` HAS ZERO OCCURRENCES; possession is marked by 之.** An earlier pass concluded
"possessors don't exist in Kyoto" by grepping for the UD-style label instead of the Classical
Chinese construction. 之 is `SCONJ` 9,149x (attributive) against `PRON` 6,554x (object pronoun), and
heads its possessor as `comp:obj` -- 9,404 tokens. Ask what the treebank encodes, not what UD would
have called it. (This is the same failure as the 公 suffix trap above, from the other direction.)

Two lexical notes: 國 is at baseline (2.0x, n=156) because a polity name attaches directly (齊國)
rather than through 之, leaving X之國 to generics. And 室/邦/邑/友/朋/族 all have n <= 13 -- too small
to read, so they are neither confirmed nor refuted.

### As GATES on a gazetteer, the ranking changes completely

    gate (gold trees)                    correct  wrong  precision  recall
    subtree + head POS (baseline)            219    384     0.3632  0.2570
    speech-subj OR vocative OR title          65     27     0.7065  0.0763
    possessor of a KIN term                    2      5     0.2857  0.0023
    possessor of terrain/building              1      4     0.2000  0.0012
    XPOS says 人,姓氏 / 人,名                   85     86     0.4971  0.0998
    all syntactic cues                        68     36     0.6538  0.0798
    all cues + name XPOS                     130    117     0.5263  0.1526

⚠ **ENRICHMENT OVER THE CORPUS DOES NOT PREDICT PRECISION AS A FILTER.** Kin enriches 4.4x yet gates
at 0.2857 -- BELOW the unfiltered baseline -- and adding kin + terrain drops the stack from 0.7065 to
0.6538. Enrichment is measured over all tokens; a gate only ever sees the gazetteer's proposals,
which are differently distributed, and in kin-possessor position those proposals are mostly spurious.
**Measure a candidate filter on the candidate set it will actually filter.**

So the best gate remains **speech-verb subject OR vocative OR following title: P 0.7065 / R 0.0763**
-- broadening 曰 to all speech verbs buys ~30 % more recall (0.0587 -> 0.0763) at equal precision.
The XPOS name categories are a different operating point (P 0.497 / R 0.0998): more seeds, one error
in two, usable only with human review.

⚠ All of these use GOLD trees and are ceilings. The transliteration-run gate above reaches ~1.00
precision with **no parse at all**, so on Buddhist text it dominates this entire stack; the syntactic
cues are for classical narrative prose, where the run cue is inert.

## Graded scores, an HMM, and phonetically blurry matching

Three attempts to fix the binary gate's diagnosed failure (竭帝 / 舍利弗 / 阿耨多羅 broken by one
below-threshold character each). All measured on the same four gold sets, 746 multi-char tokens.

**1. Homophone expansion of the lexicon: +1 correct.** `augment_translit_lexicon.py` substitutes
Qieyun-homophonous characters into Wiktionary terms (453 → 602 strict, → 870 with same-initial+rime).
⚠ It cannot generate the variants that matter, **for the same reason the gate rejects them**:
substitutes are restricted to the induced inventory, and 帝 is not in it. The restriction that keeps
the expansion clean is the one that blocks the target.

**2. Phonetically BLURRY matching: mechanism confirmed, yield small.** Instead of expanding the
lexicon, match spans to entries allowing character substitutions that share a Qieyun reading:

    gate                 blur   correct  wrong  precision  recall
    run >= 3             no         207     10     0.9539   0.2775
    run >= 3             yes        208     10     0.9541   0.2788
    neighbourhood        no         279     67     0.8064   0.3740
    neighbourhood        yes        284     72     0.7978   0.3807

The blurry hits are exactly the intended ones — **揭帝~揭諦 ×2, 竭帝~揭諦 ×2, 莎婆訶~娑婆訶** — so
Qieyun **does** work as an EQUIVALENCE RELATION where it failed twice as a classification feature.
But it only fires under a gate loose enough to admit those spans, and the loosening costs more
precision than the blurring recovers.

**3. A two-state HMM (`translit_hmm.py`): best F1, but it cannot be precise.** Emissions from the
two corpora already in use, transitions a swept sticky prior, posterior decoding for a curve:

    min P(T)   correct  wrong  precision  recall     F1
    0.50           484    187     0.7213  0.6488   0.6831
    0.90           481    174     0.7344  0.6448   0.6867   <- best F1 anywhere
    0.99           412    155     0.7266  0.5523   0.6276
    0.999          297    174     0.6306  0.3981   0.4881
    binary run>=3  207     10     0.9539  0.2775   0.4302
    ungated        455    191     0.7040  0.6100   0.6535

It **fixes the diagnosed misses** — 舍利弗 41, 阿耨多羅 35, plus 如來 85 and 世尊 52 — and more than
doubles recall. It still loses where it matters: no threshold reaches useful precision, and the only
high-precision point is 14 tokens at R 0.019.

⚠ **The HMM's two states are BUDDHIST REGISTER vs CLASSICAL REGISTER, not transliteration vs
native.** Emissions contrast a Buddhist corpus with a classical one, so inside a sūtra the posterior
saturates near 1 and stops discriminating. The binary inventory demands characters that are
INDIVIDUALLY extreme, which is a strictly stronger condition than "this text is Buddhist" — which is
why it is more precise despite being cruder. Fixing this needs emissions contrasting transliteration
against Buddhist-native vocabulary, i.e. the labelled data that already failed to materialise.

⚠ A bug worth not repeating: in a greedy longest-match loop, TIGHTENING the gate can LOWER precision,
because a long span that fails the gate makes the loop fall back to a shorter span at the same
position rather than skipping — and shorter spans are likelier to be wrong. That is what makes the
posterior table non-monotonic below P(T) ≥ 0.99; read only the 0.5–0.99 region.

**Two operating points, for two jobs.** Binary run≥3 (P 0.954 / R 0.278) to seed silver annotation
where precision is everything; the HMM (P 0.734 / R 0.645) for coverage where a human reviews
anyway.

## Shipped (2026-08-17)

`package_sud.sh lzh` now bundles the trained segmenter. Measured on the TRADITIONAL test set — the
arm that actually ships, not the both-scripts one every probe above used:

    tokeniser                        strict token F   multi-char P / R   (852 gold multi-char)
    split every character (was)             0.9624       0.0000 / 0.0000
    trained segmenter (now)                 0.9825       0.7693 / 0.7124

⚠ **THE TWO SEGMENTERS ARE NOT INTERCHANGEABLE AND NOTHING IN THE WEIGHTS SAYS SO.**
`models/lzh_seg_char` is trained on the BOTH-SCRIPTS corpus (7,342 characters),
`models/lzh_seg_char_trad` on the traditional one (5,732). lzh ships traditional-only. A vocabulary
overlap test cannot tell them apart — 1.5 % vs 1.6 % of characters absent from the arm's own
StringStore. The fix is provenance, as with `reads_spaces` and `jieba_t2s`: `train_samhita.py` now
stamps `corpus` into `vocab.json`, and `bundle_lzh_charseg.py` REFUSES a segmenter whose stamp is
not `data_seg_lzh_trad`.

⚠ **A THIRD PRE-GRAFT DEFAULT.** `LZH_BASE` named `training_lzh_trad_sud`, whose tagger sits before
the morphologiser, so `pkg()`'s guard refused it and the shipped wheel was not rebuildable from this
script — the defect already recorded for `en_gum` and `la`. The default now names
`training_lzh_trad_sud_xw`, rebuilt by `graft_xpos_tagger.py` from the arm plus
`training_lzh_xposwarm`: parse unchanged 5628/5628, tags match donor 5628/5628, **TAG 0.8469 →
0.9254**. That is the released wheel's generation; the grafted directory had simply not been kept.

Verified from the INSTALLED wheel, not the build directory: tokenizer `CharSegTokenizer`, pipeline
`[tok2vec, parser, morphologizer, tagger, clause_parser, sud_shared, han_lemma_lut,
sud_subject_rule, sud_idiom]`, 孔子曰學而時習之 → 孔子 | 曰 | 學 | 而 | 時 | 習 | 之, and a
two-sentence input still yields two sentences.
