# Licences & attribution

This repository combines components under different licences.

## Source code — MIT
`scripts/`, `webapp/`, `configs/`, and the documentation are licensed under the MIT Licence
(see `LICENSE`), © 2026 Sunflower AI.

## Treebank-derived data and released models — CC BY-SA 4.0

The relabelled treebanks committed here (`*.relabeled*.conllu`), the per-language gold sets, and
the **released model wheels** are derivative works of Surface-Syntactic Universal Dependencies
(SUD) treebanks, which are themselves derived from Universal Dependencies. They are distributed
under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, the licence of
the underlying treebanks. You must give attribution and share derivatives alike.

| Component | Source treebank | Licence |
|-----------|-----------------|---------|
| `en_sud_ewt` model, `assets/en_ewt-sud-*.conllu` | SUD_English-EWT | CC BY-SA 4.0 |
| `en_sud_ewt_gum` model, `assets/en_ewtgum-sud-*.conllu` | SUD_English-EWT + SUD_English-GUM (ten non-NonCommercial genres) | CC BY-SA 4.0 (see GUM below) |
| `zh_sud_gsd_simp_trad` model, `assets_zh/.../*.relabeled*.conllu` | SUD_Chinese-GSD + SUD_Chinese-GSDSimp | CC BY-SA 4.0 |
| `ko_sud_gsd` model, `assets_ko/.../*.relabeled*.conllu` | SUD_Korean-GSD | CC BY-SA 4.0 |
| `id_sud_gsd` model, `assets_id/.../*.relabeled*.conllu` | SUD_Indonesian-GSD | CC BY-SA 4.0 |
| `la_sud_ittb_proiel_perseus` model, `assets_la/la_ittbproiel-sud-*.conllu` | SUD_Latin-ITTB + SUD_Latin-PROIEL + SUD_Latin-Perseus | **CC BY-NC-SA** (NonCommercial — see below) |

Each source treebank's own `LICENSE.txt` is retained alongside its data.

## Kanseki Repository 漢籍リポジトリ — CC BY-SA 4.0

`lzh_sud_kyoto` is trained on a punctuated form of SUD_Classical_Chinese-Kyoto. The Kyoto treebank
deliberately carries no punctuation ("in Classical Chinese we had no spaces or punctuations between
words or sentences, so we did not include any"), so the marks are restored by aligning each work
against its edition in the **Kanseki Repository**, coordinated by Christian Wittern at the Institute
for Research in Humanities, Kyoto University — the same source the treebank itself was built from.

- Repository: https://github.com/kanripo (per-work, e.g. `KR1h0004` 論語, `KR1d0052` 禮記)
- Licence: **CC BY-SA 4.0**, compatible with the treebank's own, so the released model stays
  CC BY-SA 4.0 and the ShareAlike obligation is satisfied by this notice.
- Only the **punctuation** is taken: `scripts/align_kanripo_punct.py` inserts the marks into the
  existing trees and changes nothing else. Verified by round-trip — dropping the inserted tokens and
  renumbering reproduces the source treebank byte-identically in fields 2–8.
- Witness branches used where the default differs from Kyoto's base text: `KR2e0003` 戰國策 from the
  `tls` branch. 十八史略 `KR2b0041` and `KR4h0169` have no usable kanripo counterpart and are left
  unpunctuated.

## Digital Corpus of Sanskrit (DCS) — CC BY 4.0

The Sanskrit **CSLiser** (`models/sa_presegment*`, the saṃhitā → CSL pre-tokeniser) is trained on
classical and epic text from the **Digital Corpus of Sanskrit**, Oliver Hellwig et al.,
<https://github.com/OliverHellwig/sanskrit> (`dcs/data/conllu`), licensed **CC BY 4.0**.

| Component | Source | Licence |
|-----------|--------|---------|
| `models/sa_presegment_dcs`, `data_samhita/dcs.jsonl`, `models/sa_lexicon` | DCS — Rāmāyaṇa, Mahābhārata, Kathāsaritsāgara, Bhāgavatapurāṇa, Bṛhatkathāślokasaṃgraha, Hitopadeśa, Daśakumāracarita, Harṣacarita, Kumārasaṃbhava | CC BY 4.0 |

CC BY 4.0 requires attribution but not share-alike, so it composes with the CC BY-SA 4.0 above
(the combined work stays CC BY-SA 4.0). The DCS checkout itself (`assets_dcs/`) is a sparse clone
and is not redistributed here. Cite DCS as: Oliver Hellwig, *The Digital Corpus of Sanskrit (DCS)*,
2010–2026.

## NonCommercial — the Latin model (`la_sud_ittb_proiel_perseus`)

The Latin model is trained on the union of three SUD Latin treebanks, **all NonCommercial**:
**ITTB** (CC BY-NC-SA 3.0), **PROIEL** (CC BY-NC-SA), and **Perseus** (CC BY-NC-SA 2.5). Unlike
the other models — which are kept free of NonCommercial sources to stay commercially usable — the
Latin model and its derived data (`assets_la/la_ittbproiel-sud-*.conllu` and the released wheel)
are therefore licensed **CC BY-NC-SA (NonCommercial)**. Use it for non-commercial purposes only.

## Latin macronisation — builder shipped, lookup table NOT shipped

`scripts/la_macronise.py` restores vowel-length macrons (`token._.macron` / `doc._.macron`) by
looking each word up in a table keyed on the form plus this project's own predicted UPOS/FEATS.
**That table is never committed and never included in a wheel.** Its vowel-length data originates
in **Morpheus** (Perseus Project, **CC BY-SA 3.0 US**), reached via Johan Winge's
**latin-macronizer** (**GPL-3.0**).

CC BY-SA permits commercial use but forbids imposing further restrictions on the work. The Latin
model is CC BY-**NC**-SA (forced by its three NonCommercial treebanks, above), so bundling
Morpheus-derived content into that wheel would add precisely the restriction BY-SA rules out.
Rather than resolve that tension by guesswork, the repository ships no table at all — only code.

**There are now two ways to supply the data, and the easy one is also the better one.**

```bash
# 1. FETCH Morpheus (recommended; one 4 MB download, 249,659 wordforms, nothing else needed)
python -c 'import sys; sys.path.insert(0,"scripts"); import la_macronise; la_macronise.fetch_morpheus()'

# 2. HARVEST a table from your own treebank (optional, and worth having IN ADDITION)
bash scripts/build_la_macron.sh     # macronise locally -> harvest table -> attach to a local model
```

Fetching is not redistributing, and that is what makes (1) available at all. Johan Winge commits
Morpheus's output in **latin-macronizer** as `latin_macronizer/macrons.txt`; **GPL-3.0 restricts
distribution, not use**, so a file the user's own machine downloads from upstream — and that never
enters a build of this package — is not ours to license. Nothing here ships it, and nothing should.

The two tables cascade rather than compete, and each is better than the other somewhere. Measured
against Alatius on the held-out ITTB+PROIEL test split, gold morphology:

| | harvest has the word (92.1 %) | it does not (7.9 %) | whole-token |
|---|---|---|---|
| harvested alone | 98.23 % | 52.46 % (its suffix levels) | 94.42 % |
| Morpheus alone | 93.98 % | 90.42 % | 93.71 % |
| **cascaded** | 98.23 % | 90.42 % | **97.63 %** |

and on Perseus (classical poetry, out of the harvest's domain, where its out-of-vocabulary share
rises to 23.8 %): harvested alone 87.02 %, Morpheus alone 95.75 %, **cascaded 97.33 %**.

The released wheel **does** ship the component itself — the code is ours (MIT) and travels via
`spacy package --code`, and the pipe is in the shipped pipeline — but it ships **no table**:

```python
nlp = spacy.load("la_sud_ittb_proiel_perseus")          # 6 pipes, macroniser last
nlp("Gallia est omnis divisa in partes tres.")._.macron
# with a Morpheus cache present:
# -> 'Gallia est omnis dīvīsa in partēs trēs.'
# with none: -> 'Gallia est omnis divisa in partes tres.'   (unchanged, one RuntimeWarning)
```

Shipping the pipe with no data is the only arrangement the licences allow, and it is better than
shipping neither: the model macronises as soon as the user runs `fetch_morpheus()`, with no
`add_pipe` and no config, and `doc._.macron` is readable either way. **With no data the component
passes text through unchanged and warns once** rather than raising — it is in the default pipeline
now, so raising would break every ordinary `nlp(text)` for the users who never wanted macrons. Pass
`config={"require_data": True}` to get the hard failure instead.

`scripts/la_macron_lut.json.gz` and `build_la_macron/` are gitignored, as is the fetched Morpheus
cache (`~/.cache/sud-spacy/la_macron_morpheus.json.gz`, or `$LA_MORPHEUS_TABLE`). A model you build
by ATTACHING a harvested table contains Morpheus-derived data — **keep it local and do not
redistribute it**. Fetching leaves the model alone: the cache sits outside it, so the released
wheel, which carries the component and no table, is still clean.

The macroniser's own tagger, **RFTagger** (Schmid & Laws), is licensed for education, research and
other **non-commercial** use only. It is used solely to label your treebank offline; the component
uses this project's morphologiser at inference, so RFTagger is not a runtime dependency. Route (1)
above does not touch it at all — `macrons.txt` is Morpheus's own output, tagged by Morpheus, so
fetching skips RFTagger, Docker and the Morpheus compile together.

## Georgetown University Multilayer Corpus (GUM) — `en_sud_ewt_gum`

English ships **two** wheels. `en_sud_ewt` is EWT-only and contains no GUM-derived data at all.
`en_sud_ewt_gum` adds **ten of GUM's fifteen genres** — academic, bio, conversation, court,
interview, news, speech, textbook, vlog, voyage — and is **CC BY-SA 4.0**, like every other model
here except Latin and Arabic.

**Why that is not NonCommercial, although GUM's `LICENSE.txt` opens "The treebank is licensed under
CC BY-NC-SA 4.0".** Read strictly, that line offers the *annotations* under NC whatever the
document, and annotations are what a trained model absorbs — so this wheel shipped NC at v0.2.0
while the question was open. Amir Zeldes, GUM's maintainer, resolved it directly (email,
2026-08-17): the annotations are produced at Georgetown under a **CC BY** licence, and the NC comes
**only from the individual underlying documents**, whose own licences must be respected. Using only
documents without an NC licence is therefore fine commercially. The five NonCommercial genres —
**essay, fiction, letter, podcast, whow** (wikiHow and fiction texts, and personal-blog material) —
are excluded from this wheel's training data by `scripts/build_en_ewt_gum.sh`, which is what earns
the licence. `NC_GENRES` in that script is a licence boundary, not a conservative default.

The **ShareAlike** term comes from SUD_English-EWT (CC BY-SA 4.0) and from the Wikimedia sources
among the kept genres (Wikivoyage and the Wikimedia bios, CC BY-SA 3.0) — not from GUM as a whole.

**Attribution is an obligation here, not a courtesy**, because CC BY makes it one:

- Cite the corpus and link the project website — **https://gucorpling.org/gum/** — which lists the
  annotators and links the original data for each document.
- Credit the **GUM annotators**: Amir Zeldes and 300+ Georgetown student annotators.
- Cite the **text sources** as their own sites require: Wikinews and Wikivoyage (Wikimedia),
  OpenStax open-access textbooks, the Santa Barbara Corpus of Spoken American English (John Du
  Bois, UCSB), Creative-Commons YouTube vlogs, and public-domain political speeches.

The same attribution travels inside the wheel, in `meta.json`'s `sources` (see
`scripts/stamp_model_meta.py`).

## Attribution

- Surface-Syntactic Universal Dependencies — https://surfacesyntacticud.github.io/
- Universal Dependencies — https://universaldependencies.org/
- Georgetown University Multilayer Corpus (GUM), Amir Zeldes et al. — https://gucorpling.org/gum/
- Morpheus (Perseus Project, CC BY-SA 3.0 US) — https://github.com/PerseusDL/morpheus
- latin-macronizer, Johan Winge (GPL-3.0) — https://github.com/Alatius/latin-macronizer
- RFTagger, Helmut Schmid & Florian Laws (non-commercial) —
  https://www.cis.uni-muenchen.de/~schmid/tools/RFTagger/
- Please cite the individual UD/SUD treebanks when using these models; their authors are credited
  in each treebank's `LICENSE.txt`.
