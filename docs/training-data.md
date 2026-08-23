# Training data

Sentences and tokens in the CoNLL-U each released model is trained and evaluated on — the SUD
treebank after the `udep` relabel, which changes DEPREL only and so leaves every count untouched.
Range lines (multiword tokens) and empty nodes are not counted as tokens.

| Model | Train (sent / tok) | Dev (sent / tok) | Test (sent / tok) |
|-------|-------------------:|-----------------:|------------------:|
| `en_sud_ewt` | 12,544 / 204,578 | 2,001 / 25,148 | 2,077 / 25,094 |
| `en_sud_ewt_gum` | 20,273 / 340,324 | 3,023 / 43,580 | 3,014 / 43,403 |
| `zh_sud_gsd` | 3,997 / 98,614 | 500 / 12,665 | 500 / 12,010 |
| `ja_sud_gsd` | 7,050 / 168,333 | 507 / 12,287 | 543 / 13,034 |
| `ko_sud_gsd` | 4,400 / 56,687 | 950 / 11,958 | 989 / 11,677 |
| `la_sud_ittb_proiel_perseus` | 40,305 / 586,604 ¤ | 3,334 / 43,805 | 4,300 / 54,897 |
| `lzh_sud_kyoto` | 59,215 / 460,390 ‖ | 5,111 / 38,739 | 4,567 / 34,233 |
| `ar_sud_padt` | 6,075 / 223,881 | 909 / 30,239 | 680 / 28,264 |
| `fa_sud_perdt` | 26,196 / 452,496 | 1,456 / 25,147 | 1,455 / 24,133 |
| `id_sud_gsd` | 4,482 / 97,602 | 559 / 12,661 | 557 / 11,756 |
| `sa_sud_vedic_ufal_dcs` | 21,647 / 163,308 ∴ | 2,996 / 23,862 | 230 / 1,843 ∴ ◊◊ |
| `ta_sud_ttb_mwtt` | 828 / 8,409 ◊ | 133 / 1,521 | 173 / 2,235 |
| `te_sud_mtg` | 1,051 / 5,097 | 131 / 666 | 146 / 722 |
| `yue_sud_hk` | 804 / 11,158 ◊ | 100 / 1,499 | 100 / 1,261 |

The two smallest rows are worth reading against the largest: `te_sud_mtg` trains on **5,097 tokens**
where `la_sud_ittb_proiel_perseus` has 586,604, and every figure reported for ta and te should be
read as resting on that.

---

¤ **Latin trains on one copy of the macronised treebank, resampled into a fresh edition style every
epoch** (`scripts/la_augment.py`) — not on two fixed spellings. Each pass rewrites the FORM column
along five axes printed Latin genuinely varies on: macrons, breves, `u`/`v`, `i`/`j`, `æ`/`œ`, and
sentence-initial capitals. The trees never move, so the token count is the treebank's own; what
changes is which spelling the model meets on any given epoch. Macron-stripping is exact
(586,604/586,604 tokens reproduce the plain FORM), so the plain spelling is *derived* rather than
stored and the macronised treebank is a strict superset. Dev and test are the plain half, so the
results table understates what the arm is for — see [`results-notes.md`](results-notes.md).

‖ Classical Chinese trains on a **punctuation-restored** Kyoto: the treebank carries no punctuation
of its own, so the marks are aligned in from the Kanseki Repository editions it was built from. The
counts above therefore include punctuation tokens that the bare treebank does not have.

∴ **The Sanskrit row is the parser's data**, which is the whole of it for the UAS/LAS reported.
DCS is much larger — 244,481 sentences / 1,732,852 tokens — but it carries **no dependency
annotation**, so it trains the **morphologiser and lemmatiser** only (and the tagger, whose XPOS is
a copy of UPOS on 100 % of tokens here, so it is predicting the same labels). Its docs are built
with no heads or deps at all, which is the only representation spaCy reads as genuinely missing —
blanking the columns to `_` would teach the parser a literal `_` label — so the parser takes no
gradient from them whatever. Read the DCS figure against `pos_acc`, `morph_acc` and `lemma_acc`,
never against LAS.

◊◊ The Sanskrit test row is the held-out **UFAL** set the model is reported on, not the Vedic test
the dev split comes from. See [`sanskrit.md`](sanskrit.md).

◊ **Test-only treebanks, carved deterministically.** SUD_Cantonese-HK and SUD_Tamil-MWTT each ship a
test split alone, so both are carved 80/10/10 round-robin (`scripts/split_yue.py`,
`scripts/prep_ta.py`). Tamil's row is TTB's own train/dev/test plus that MWTT carve; the two
treebanks disagree about annotation rather than merely tagset, so `scripts/train_ta.sh` reports the
TTB slice separately. See [`dravidian.md`](dravidian.md).

## Getting the data

The raw treebanks are not committed — re-download them from <https://grew.fr/download/>. The
relabelled derivatives and each `LICENSE.txt` are force-added (`git add -f`); everything else under
`assets*/` is gitignored, and `.gitignore` names each one with the command that rebuilds it.

```bash
cd assets
curl -sSLO https://grew.fr/download/SUD_2.18/SUD_English-EWT.tgz
tar xzf SUD_English-EWT.tgz && cd ..
```
