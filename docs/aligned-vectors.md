# Cross-lingually aligned vectors, as side assets

Thirteen `.npz` tables, one per language, in **one shared 128-dimensional space**: a word in any of
them can be compared to a word in any other by a plain dot product. They exist to **align parse
trees across parallel translations** — given a node in one tree, find the node in another language's
tree that means the same thing.

They are **not** a parser input, and nothing here should be read as a route to better LAS. Static
vectors as a parser feature were measured twice and rejected both times: fastText `md` on yue/id/ko
gave +0.2–0.9 LAS inside seed noise while costing 9–16× the model size, and the kanripo arm gave
**+0.04 LAS over three seeds** (`NEGATIVE-RESULTS.md`). This is a different job with a different
metric.

They are **side assets, never bundled in a wheel**, for two independent reasons. A table is only
useful when you hold two of them at once, so a copy inside each wheel would be thirteen copies of
something no single wheel can use. And the licences diverge: fastText is CC BY-SA 3.0, which could
not go inside the CC BY-NC-SA la/ta/te wheels at all.

## What is in an asset

```python
from scripts.aligned_vectors import AlignedVectors
en = AlignedVectors.load("release_vectors/sud_vec_en_128d.npz")
sa = AlignedVectors.load("release_vectors/sud_vec_sa_128d.npz")
sa.nearest(en["water"], k=3)     # [('ambhas', 0.68), ('jala', 0.67), ('toya', 0.67)]
```

| array | shape | what it is |
|---|---|---|
| `vectors` | N × 128 | unit-length rows, so cosine **is** the dot product |
| `keys` | N | UTF-8, newline-joined |
| `basis` | 300 × 128 | the shared truncation basis — **byte-identical in all thirteen assets** |
| `mean` | 300 | the joint mean, likewise identical |
| `rotation` | 300 × 300 | per-language; places that language's source space in the hub space |
| `meta` | json | route, source, `key_attr`, `lookup`, held-out scores, coverage |

`basis`, `mean` and `rotation` are there so a caller can project a vector the table was cut before
reaching: `v_shared = (normalise(v_source) @ rotation - mean) @ basis`, which is what
`AlignedVectors.project` does.

## The three decisions, and what they were measured against

**128 dimensions.** Retrieval P@1 into English saturates there and falls off a cliff below 48:

| dims | 16 | 32 | 48 | 64 | 96 | **128** | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| en←id P@1 | 14.0 % | 44.8 % | 61.3 % | 67.8 % | 73.6 % | **77.4 %** | 78.9 % | 77.5 % |

Fewer dimensions to buy more keys is the wrong trade, because token coverage is *already* 89.6 % at
20 000 keys and only 95.8 % at 100 000. At every byte budget from 5 MB to 40 MB per language, the
product coverage × P@1 is maximised at 128d — 96d and 200d are both worse, and 32d is barely half.

**One shared basis, computed on the joint space.** This is the decision that a reader is most likely
to undo by accident, so it is worth the control. Same anchors, same 128 dimensions, en←id, 1 500
pairs over 50 000 candidates:

    300d, no truncation        @1 64.3 %   @5 79.7 %
    128d, ONE SHARED basis     @1 63.8 %   @5 77.7 %
    128d, per-language PCA     @1  0.0 %   @5  0.0 %     <- the control

**A per-language PCA does not degrade the alignment, it destroys it.** Each language gets a
different rotation, so the spaces stop being commensurable at all. The truncation itself is nearly
free — half a point for 57 % of the bytes.

**Unit-length rows**, so no caller has to know how to preprocess before taking a cosine.

## Where each language's vectors come from, and how well it worked

English is the hub. Seven languages are published already aligned to it and take the identity
rotation; the rest are fitted by orthogonal Procrustes, which is a **rotation** — it can place a
space, never distort it.

`@1`/`@5` are held-out **set-membership** hits: is the nearest English word among the acceptable
ones? Splits are over source **words**, not pairs.

| lang | source | route | anchor words | @1 | @5 | treebank types covered | MB |
|---|---|---|---|---|---|---|---|
| en | fastText aligned | hub | — | — | — | 84.8 % | 25.4 |
| id | fastText aligned | pre | 83 367 | 62.3 % | 78.9 % | 86.4 % | 25.3 |
| ar | fastText aligned | pre | 26 953 | 53.4 % | 71.1 % | 83.0 % | 26.4 |
| ja | fastText CC | MUSE dict | 20 047 | 43.3 % | 62.6 % | 87.4 % | 25.7 |
| ko | fastText aligned | pre | 15 537 | 43.0 % | 62.5 % | 58.5 % | 26.7 |
| fa | fastText aligned | pre | 40 873 | 36.4 % | 53.7 % | 75.3 % | 27.6 |
| zh | fastText aligned | pre | 31 541 | 35.6 % | 54.7 % | 64.9 % | 25.3 |
| **la** | **floret over 112.8M homegrown tokens** | Wiktionary gloss | 39 565 | **37.7 %** | **51.6 %** | **100 %** | 32.1 |
| ta | fastText aligned | pre | 23 731 | 32.4 % | 48.5 % | 77.7 % | 24.4 |
| te | fastText CC | Wiktionary gloss | 5 597 | 22.6 % | 40.5 % | 76.0 % | 24.7 |
| sa | **floret over DCS lemmas** | **Apte** gloss | 27 414 | 11.5 % | 21.5 % | 100 % | 25.1 |
| yue | fastText wiki | Wiktionary gloss | 22 501 | 10.7 % | 21.0 % | 73.9 % | 24.1 |
| lzh | **floret over kanripo** | Wiktionary gloss | 8 353 | 6.8 % | 15.3 % | 100 % | 8.4 |

**Read the bottom three rows as weak, not as broken.** The score is dominated by rare words, and the
frequent vocabulary — which is where the tokens are — aligns visibly better than the aggregate says:

    en 'horse'  -> sa  aśva aśvakhura hastin      lzh 馬 騏 轡       la equus equi currus
    en 'war'    -> sa  saṃgrāma yodha yudh        lzh 兵 戰 戎       la proeliatus bellico bello
    en 'city'   -> sa  pura dāśapura purī         lzh 城 方城 都      la urbs oppidum capuae
    en 'write'  -> sa  vad paṭh adhī              lzh 此 述 論       la prologo scripsero scribo

## Latin is homegrown, and the orthography is the whole story

Latin was the worst-covered asset in the first build: cc.la reached **52.0 %** of treebank types at
@1 27.5 %. Replacing it with a homegrown corpus took it to **100 % coverage at @1 37.7 %**, ahead of
zh, fa and ta — but almost none of that came from corpus size alone.

**The corpus** is 112.8M tokens, deliberately NOT Common Crawl (which is what cc.la already is):

    la.wikisource        ~84.0M   classical and medieval texts in full
    latin_text_latin_library  12.1M
    la.wikipedia          10.0M
    PerseusDL/canonical-latinLit  6.2M   ⚠ `-latN.xml` ONLY -- the same repo ships `-engN.xml`
                                         English translations, which would train English vectors
                                         under Latin keys
    our own la treebanks   0.5M

**The orthography fold is what actually paid.** Our treebanks are **u-dominant** — of 586 604
training tokens, 2.2 % contain a `v` and **none** contain a `j`, a macron or a ligature — while
Wikisource, the Latin Library, Perseus and Wiktionary all spell with `v` and `j` freely. Folding
everything onto one spelling (lowercase, length marks off, `æ`/`œ` expanded, `v`→`u`, `j`→`i`):

    treebank types present in the corpus        52.0 %  ->  99.2 %
    Wiktionary headwords matching a vector      19 170  ->  39 565 anchor words

The fold has to be applied at **lookup** as well, because the released la arm is
orthography-augmented and will hand you any of the four spellings. That is why `_norm_la` lives in
`aligned_vectors.py` — the file a user gets — is imported by the build scripts rather than the other
way round, and is named in the asset as `key_norm`. Two consequences worth stating:

- **Merge, do not first-wins.** The fold collapses `vita`/`vīta` onto one row, and the first version
  of `anchors_gloss` kept whichever headword arrived first and dropped the other's glosses. Merging
  the bags instead was worth +0.8 to +1.4 on every gloss language (sa 10.7 → 11.5, lzh 5.6 → 6.8,
  yue 9.4 → 10.7).
- **`la` is the only asset with a `key_norm`.** Everything else is either lowercased or literal.

## Traps

**1. Sharing a graph is not sharing a distribution.** lzh and yue were built first on the obvious
route — anchor on the strings they share with already-aligned zh — and it is the *worse* route for
both, scored on the same held-out glosses: **lzh 1.3 % against 6.4 %, yue 7.1 % against 10.0 %**.
The reason is structural: a single character is a **word** in Literary Chinese and a **bound
morpheme** in modern Chinese, so zh's vector for it is built from the rare contexts where it stands
alone. Even the 500 most frequent lzh characters retrieve their own zh row only **10.0 %** of the
time (mean cosine 0.34). Modern Chinese is not a usable pivot for Literary Chinese, and no amount of
graph overlap changes that. The identity rotations are kept as `W_{lzh,yue}_identity.npy`.

**2. The IDS-expanded lzh table is the wrong table for this.** It exists to make *graphically*
similar characters similar, which is what a parser facing unseen characters needs and the opposite
of what alignment needs. A plain distributional floret over the same corpus replaced it. (This
turned out not to be the binding constraint — trap 1 was — but the plain model is still the right
one to ship.)

**3. Case folding is worth 31 points and cannot be guessed.** The published aligned vectors are
lowercased throughout; the CC ones are not. English treebank type coverage is **53.9 %** if you look
words up as written and **84.8 %** if you fold — so each asset records `lookup` in its meta and
`AlignedVectors.__getitem__` obeys it. Do not case-fold by hand.

**4. sa is keyed by LEMMA, everything else by surface FORM.** Apte — the only Sanskrit-English
resource we hold — is keyed by stems, and Sanskrit inflection makes a form-keyed table mostly hapax.
`meta["key_attr"]` says which, and `AlignedVectors.key_for(token)` reads the right attribute off a
spaCy `Token`. This is the same class of mistake as CLAUDE.md hazard 10: record the regime in the
artefact instead of assuming it downstream.

**5. Fitting on frequent words only — the standard MUSE recipe — HURTS here, and by a lot.**
Restricting the Procrustes fit to source rank < 20 000 cost la 30.0 → 21.9, te 23.5 → 16.3, ja
42.6 → 40.0, sa 9.7 → 8.9. The recipe assumes a clean 1:1 dictionary of a few thousand pairs; a
gloss bag is noisier but there is far more of it, and quantity wins. `--fit-max-rank` is kept, with
the restriction **off** by default.

**6. A definition dictionary cannot be scored by exact match, and a translation dictionary must not
be scored against one gold word.** MUSE dictionaries are many-to-many — `and` alone has three
Chinese translations — and scoring against a single gold word reads zh as **26.4 %** where the
honest number is **35.6 %**. Both routes are scored by set membership, and held-out splits are taken
over source words so a word's other translations cannot leak from train into test.

**7. zh's gap is tokenisation, not orthography.** Only 64.9 % of zh treebank types are in the vector
vocabulary, and it is tempting to blame traditional-vs-simplified. Measured: exact 63.1 %,
exact-or-t2s 63.8 %, and converting the vector keys with s2t makes it **worse** (43.8 %). The misses
are GSD's segmentation against fastText's, and no script conversion touches them.

## Rebuilding

```bash
scripts/fetch_vec_sources.sh                      # ~5 GB, truncated to the top 200k lines as it downloads
scripts/fetch_la_corpus.sh                        # ~390 MB of Latin dumps and tarballs
.venv/bin/python scripts/build_la_corpus.py       # -> assets_vec/la_corpus.txt, 112.8M tokens
DIM=300 MINN=3 MAXN=5 BUCKET=100000 EPOCH=5 THREAD=10 \
  bash scripts/train_floret.sh la_hg assets_vec/la_corpus.txt
.venv/bin/python scripts/build_dcs_corpus.py --key lemma --out assets_vec/dcs_lemma.txt
DIM=300 BUCKET=50000 MINCOUNT=3 bash scripts/train_floret.sh sa_lemma assets_vec/dcs_lemma.txt
DIM=300 MINN=1 MAXN=3 MINCOUNT=5 bash scripts/train_floret.sh lzh_plain corpus_lzh_kanripo_leakfree.txt
.venv/bin/python scripts/apte_anchors.py                                   # sa-en.json
.venv/bin/python scripts/kaikki_anchors.py --lang-code la --url … --out assets_vec/dict/la-en.json
.venv/bin/python scripts/align_vectors.py --stage fit                      # rotations + held-out scores
.venv/bin/python scripts/align_vectors.py --stage basis                    # the one shared basis
.venv/bin/python scripts/align_vectors.py --stage emit                     # release_vectors/*.npz
.venv/bin/python scripts/aligned_vectors.py --dir release_vectors --query water king horse
```

⚠ `fetch_vec_sources.sh` is the one script in the repo that must **not** run under `pipefail`: curl
exits 56 on SIGPIPE every time `head` closes the pipe, which is the *successful* path here.

## Licensing — unresolved for two of the thirteen

Eleven assets derive from fastText, **CC BY-SA 3.0**, relicensable to CC BY-SA 4.0 under 3.0's
later-version clause, and requiring attribution to Facebook AI Research.

**sa (DCS) and lzh (kanripo) derive from corpora that declare no licence at all** — neither
`OliverHellwig/sanskrit` nor the `kanripo/KR*` repositories carry a LICENSE file, and the kanripo
text headers carry no rights metadata. Settle this before publishing those two.

No dictionary content is redistributed by any asset. Apte and Wiktionary are used only to **fit a
rotation**; what ships is a 300 × 300 matrix of floats and a table of vectors.
