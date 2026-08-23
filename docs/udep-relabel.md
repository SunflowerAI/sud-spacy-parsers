# The `udep` relabel, and SUD relation conformance

Extracted from `CLAUDE.md` so the main guide stays short — the same reason
`NEGATIVE-RESULTS.md` exists. Read this before touching the area it covers.

## The `udep` relabel (the research contribution)

**Scripts.** `disambiguate_pp.py` is the foundation module imported everywhere (`parse_conllu`,
`descendants`, `render`, and `query` — the canonical Ollama call). `build_gold.py` (en) /
`lang_gold.py` (others) build the *confident* comp/mod benchmark from unambiguous `udep` cases:
COMPLEMENT = the verb lexically selects the adposition (curated `(verb, adp)` frames); MODIFIER =
temporal/causal adposition or temporal object (note the temporal-object override — a frame with a
year object is a modifier, "believe in 1999"). SUD's own committed labels are too sparse and noisy to
serve as gold, hence the rule build. `eval_prompts.py` / `lang_bench.py` benchmark prompt variants
(`eval_prompts.PREFIXES["fewshot12_def"]` is the canonical English prompt; `en_errors.py` is the
error analysis that drove the contrastive shots);
`zh_bench.py`/`id_bench.py`/`en_bench.py` hold curated same-adposition contrastive few-shot.
`relabel.py` / `lang_relabel.py` apply the chosen prompt (rule first, model only for the genuinely
ambiguous remainder), resumable via on-disk `caches/relabel_cache*.jsonl`. `relabel_ext.py` covers the
**extended scope** into separate `*.relabeled_ext.conllu`. `udep_audit.py` / `udep_probe.py` /
`hard_examples.py` are the analysis behind the scope decisions.

**Prompts are static prefix (definitions + few-shot) + short variable suffix (the sentence)** so
Ollama reuses the cached prefix KV (~4× speedup). Keep them that way.
Block-based rewriters preserve the file byte-for-byte except target DEPREL cells — verify round-trip
before long runs.

**Extended scope** adds: ADP dependents of NOUN/PROPN/ADJ heads; clausal verb PPs; participial complex
prepositions (`according/based/following` → mod); a Korean case-suffix rule; zh 的/之, lzh 之, ja の
associative PART → mod; ko ADV-of-VERB → mod. **Partitives (NUM/DET/PRON heads) stay `udep`** —
SUD's documented default, by user decision.

**Results, `comp:obl` F, base → verb-rl → ext** (LAS within ~1 throughout):

| | base | verb-rl | ext | |
|---|---|---|---|---|
| id | 0.463 | 0.565 | **0.703** | prepositional, genuinely ambiguous |
| fa | 0.705 | 0.815 | 0.794 | ext dilutes an already-strong verb set |
| ja | 0.000 | 0.720 | 0.688 | GSD commits **no** `comp:obl` — the class is synthesised from scratch |
| ar | 0.617 | 0.659 | 0.634 | |
| la | 0.678 | 0.691 | 0.684 | |
| lzh | 0.716 | 0.659 | 0.701 | ext = the coverb rule below |
| en | — | 0.740 | 0.730 | large well-disambiguated verb set already |
| zh | 0.190 | 0.307 | 0.356 | |
| ko | 0.169 | 0.247 | 0.386 | at eojeol tokenisation this result is not yet reproduced |
| sa | 0.404 | — | 0.352 | case-based; stays un-relabelled |
| yue | 0.308 | 0.261 | 0.348 | |

**Findings.**
- Relabelling lowers headline **LAS by ~1–2** everywhere (the binary is harder than the noncommittal
  label) while **UAS is unchanged** — only labels change. The metric that reflects disambiguation
  quality is per-label **`comp:obl` F**. Caveat throughout: each relabel rewrites the *test* gold
  too, so `comp:obl` F has a moving denominator.
- Value scales with how genuinely ambiguous the adpositional system is: high for prepositional
  systems (en/id/fa/ja/ar/la), near-vacuous where the `udep` adpositions are ~all circumstantial
  (lzh's plain-`udep` residue) or where the system is case-based (sa).
- **Korean is not near-vacuous** once the case suffix on noun dependents is used — the verb-ADP view
  simply missed where its signal lives. Same lesson as lzh's locative complements.
- **Two `udep` families**: prepositional (fa/ar/la/lzh/ja/zh/yue/en/id — the adposition is the ADP
  head of the NP) use the verb-frame gold; case-based (sa, ko) use the dependent's morphological
  Case or case particle.
- Per-language relabel model: fa/sa/lzh/en/id/zh/ko/yue → qwen3:8b, ar/la → gemma4, **ja → qwen3 with
  a native-Japanese prompt**.

**Language-specific rules worth knowing.**
- **lzh coverbs.** The bulk of the signal is not on plain `udep` but on the annotators' **subtyped**
  `udep@lmod` (locative, ~3029) and `udep@tmod` (temporal, ~105), which the plain scope never
  reached. Decided from the annotators' own category + the head verb's class (XPOS field 3):
  **@tmod → mod**; **@lmod → comp:obl** only under a locus-selecting verb class (移動 motion / 姿勢
  posture / 設置 placement / 存在 existence / 生物 birth-death), else mod. This commits ~815 test
  coverbs and nearly doubles the comp:obl class (182→355) at precision 0.72 with LAS flat. Object
  FEATS `Case=Tem`/`Case=Loc` is the same signal for plain-`udep` coverbs.
- **lzh 於 routing.** After Loc/Tem, the residue splits ~evenly person 958 / non-person 912. The
  treebank commits **0 comp:obl and 0 mod on 於+person** (recipient-dative vs comparison vs
  passive-agent — maximally ambiguous, and inherently unvalidatable since there is no gold), so only
  the LLM can adjudicate it; 於+non-person IS committed (84:54) so a loose frame rule fits
  (`COMP_FRAMES["lzh"]` derived at minc=2/thresh=0.70 → ~15 frames: 至於/達於/在於/異於/甚於/長於/怒於…;
  the default minc=8/thresh=0.85 yields none). The rule intercepts cases *before* the cache, so no
  re-querying is needed.
- **Sanskrit case rule.** Recipients are **dative** (confirmed in-treebank: dā/prayam+Dat), not
  locative — the locative-of-locus is the Vedic ritual `hu` "offer into fire-LOC", which SUD leaves
  `udep`/mod. So Loc/Abl/Voc/Nom → mod, recipient datives → comp via (verb, Case) frames; blanket
  Dat → comp is avoided (the dative-of-purpose is adjunctival).

### `udep` beyond comp/mod: derived rules commit 10 730

`relabel_ext.py` asks one question, so anything that is not an adpositional or case-marked oblique
stays `udep` — 32 415 tokens over nine treebanks, dominated by material where no oblique/modifier
choice is being deferred: Persian's relativiser که (5060), English `'s` and infinitival `to` (950),
Japanese adnominal/copular た/だ (355).

**`udep_residue_audit.py`** answers "what SHOULD this be?" from the treebank's own committed
decisions — for each residual token, the DEPRELs annotators used for the same (head UPOS, dep UPOS,
dep lemma) signature. **`apply_udep_rules.py`** commits what is dominated past 90 % on ≥ 20 committed
examples, writing `*.udep_ruled.conllu` (DEPREL column only). Rules are DERIVED, never hardcoded.

    fa 7156  (NOUN<-SCONJ که -> mod, 98 % of 375)      lzh 1834 (VERB<-NOUN 今/後/初 -> comp:obj)
    ja  802  (NOUN<-AUX た/だ -> mod, 99 % of 1392)     en 526   ar 311   zh 54   id 33   ko 12   yue 2

Japanese is the clearest set: NOUN<-AUX た is a relative clause (the tense auxiliary heads it), だ is
the adnominal copula な, VERB<-AUX だ the adverbial に — the same copula in both non-finite guises,
recovered independently by the evidence. fa/lzh/ja retrained and re-released
(`retrain_udep_ruled.sh`; fa LAS 87.18, lzh 79.01, ja 88.21). The point is OUTPUT CORRECTNESS, not
accuracy: on 40 test sentences the old fa model emitted `udep` on 34 and `mod` on 4, the new one
`mod` on 35. en/ar (0.22 %/0.10 %) and id/ko/zh/yue (≤ 0.05 %) were skipped. Pre-rule treebanks kept
as `*.pre_ruled`. **fa also needs its `_sud` arm rebuilt** — it ships from a Subject layer stacked
above the lemma arm, which the base chain alone would miss.

An LLM pass over the remaining residue was built and **abandoned** — see NEGATIVE-RESULTS.

### Korean eojeol relabel (committed 2026-08-04)

424 DEPREL cells across `assets_ko/SUD_Korean-GSD/ko_gsd-sud-{train,dev,test}.relabeled_ext.conllu`
(313/53/58), all `udep` → `mod` (392) or `comp:obl` (32); DEPREL is the only column touched. This is
the extended relabel rebuilt at the eojeol granularity the released arm now uses, backed by +138
entries in `caches/relabel_cache_ext_ko.jsonl` (112 modifier / 26 complement). **None of it reached the
v0.1.0 wheels** — the shipped `ko_sud_gsd` predates it, so its `comp:obl` F 38.6 does not include it.

## SUD relation conformance (`normalise_reparandum.py`)

Audited against the guidelines: `conj` is correctly **chained** (each conjunct → the previous, `cc` →
the conjunct it precedes) in every treebank and no transform disturbs it; `appos` is never emitted
bare (apposition is the sanctioned `conj:appos`, 46 260). The UD relation **`reparandum`** survived
un-converted in a few upstream SUD releases and is rewritten to SUD's **`conj:dicto`** — 696 across
all derived files, distinct instances la 32 / yue 165 / zh 2, **DEPREL column only** (`reparandum` is
also a Latin gerundive word form, so FORM/LEMMA must be untouched). A pure label rename; la/yue/zh
bases were retrained so released models emit it.

⚠ **`en` was missed by that pass and shipped the UD relation until 2026-08-08.** Found while
building the EWT+GUM arm. The released `en_sud_ewt` had `reparandum` in its parser's label
inventory and no `conj:dicto`, and emitted it on real input — 18 predictions over the 44 sentences
whose gold carries the relation. 36/9/4 tokens in train/dev/test, 147 cells over the nine tracked
derived files. Confirmed against the **downloaded v0.2.0 asset**, not a local directory.

**Fixed WITHOUT retraining the parser, and that is the point.** `reparandum` → `conj:dicto` is a
pure label rename, so retraining on renamed data yields the same model up to RNG; renaming the
action inside the trained parser is the exact analogue and keeps every weight — and therefore every
published metric — byte-identical, so a clobbered wheel differs in the one thing that was wrong.
`scripts/rename_deprel_label.py` does it. **The hazard it guards is real**: spaCy orders actions by
`(frequency, label_string)` DESCENDING (`TransitionSystem.initialize_actions`), so the label string
is a tiebreak and a rename can renumber the actions, silently misaligning weights that are indexed
by action. en is one string away — `reparandum` and `comp:aux@pass` both have frequency 31 in
LEFT-ARC, and the order survives only because `conj:dicto` also sorts above `comp:aux@pass`. The
script refuses unless the full (action, label) sequence is unchanged position for position, and
`--verify-parses` re-parses a corpus with both models (0 heads, 0 deprels differing over all 2077
test sentences).

**Only `sud_shared` needed retraining**, because `sud_shared_data._is_conj` counts `conj:dicto` as a
conjunct and `reparandum` not, so the coordination mask moves (75 candidates across the splits, and
the repair token itself is now excluded as a conjunct — which drops `mod ADJ after` from the rule
table under `--min-count 20`). `sud_subject` reads only NORM/PREFIX/SUFFIX/SHAPE with gold from
MISC, so the released pipe was copied back in rather than re-initialised; it re-evaluates to exactly
its published 82.01. The rebuilt wheel is **29 of 38 files byte-identical** to the shipped one — the
movers are `parser/moves`, `sud_shared/model`, `vocab/strings.json` and metadata. Test: Shared
62.6 → 63.10, Subject 82.01 and Idiom 84.62/82.14 unchanged, every ship decision intact.

Other non-official UD carry-overs (`mod@poss`,
`@unmarked/@desc/@predet/@preconj`, `compound@prt`) were left as-is by user decision; `@lmod/@tmod`
and the other language-specific semantic subtypes are legitimate SUD conventions the pipeline relies
on.

## The annotators' `@subtype` was being erased on write

`relabel_ext.py` decided comp-vs-mod correctly for a subtyped `udep` and then wrote the answer flat:

    cols[7] = "comp:obl" if lab == "complement" else "mod"

so a resolved `udep@tmod` became plain `mod`, never `mod@tmod`. The effect was not a lost nuance but
a **misleading record**: every arm showed `mod@tmod` and `comp:obl@lmod` at ZERO while the residue
still carried `udep@tmod`, so it read as though the subtyped ones had never been relabelled at all.
They had been; the evidence was being destroyed at the write site. `--keep-subtype` carries it.

It is OFF BY DEFAULT, which is the opposite of this project's usual "the default is the fix" rule,
and deliberately so: the arms ON DISK were built without it — lzh 4 167 tokens, sa 8 249, yue 108
(zh, id and en have none in scope) — so flipping the default would make the script stop reproducing
their data while saying nothing. lzh has adopted it. **sa and yue have not, and should when next
rebuilt.**

`lzh` also lost its `UPOS == ADP` restriction, which had confined the rule to coverbs and left 2 188
`udep@tmod` untouched on train — 1 753 of them a bare temporal NOUN under a VERB (今日, 昔) and 173 a
NOUN under an AUX, not coverbs at all but the same WHEN adjunct. The yue branch has never had a UPOS
restriction, for exactly this reason. Residue fell 5 772 → 2 071.

⚠ **1 269 tokens changed hands between two rules.** Those had been committed as `comp:obj` by
`apply_udep_rules.py`'s derived rules (a ≥90 % type-level frequency over (upos, head-upos, form));
with the wider scope the coverb rule reaches them first and calls them `mod@tmod`. The precedence is
right — an annotator's per-instance subtype should beat a type-level frequency — but it is a
substantive relabelling, not a renaming, and it is why the derived-rule commit count on lzh drops
from 1 834 to 202.

**Transferring the change onto shipped data (`transfer_deprel_lzh.py`).** The chain ends in two
steps that do far more than relabel — `align_kanripo_punct.py` inserts 100 193 marks, and
`cross_unit_rules.py` merges 句讀 units and re-heads the merged root — so the delta is carried onto
the final file rather than the chain re-run. Two traps, both caught by refusals rather than by
inspection:

- the alignment check came back off by exactly **5**, the number of PUNCT tokens Kyoto is documented
  as not having but does (5 in 374 560). PUNCT has to be excluded from BOTH sides;
- a naive delta would have **reverted 15 394 cross-unit relations** — the merge re-heads unit roots
  (`root` → `comp:obj` 8 759, `mod` 2 957, `parataxis` 2 415, `conj:coord` 1 263), and those still
  read `root` upstream. The transfer never writes `root` back.

The `--old-is-target` path exists because assets are gitignored and the on-disk `.relabeled_ext`
accumulates later in-place passes, so re-running the previous code reproduces `relabel_ext` but not
that accumulated state. It was validated against a true-delta transfer on train: byte-identical.
