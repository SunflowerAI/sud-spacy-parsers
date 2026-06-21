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
| `zh_sud_gsdsimp` model, `assets_zh/.../*.relabeled*.conllu` | SUD_Chinese-GSDSimp | CC BY-SA 4.0 |
| `ko_sud_gsd` model, `assets_ko/.../*.relabeled*.conllu` | SUD_Korean-GSD | CC BY-SA 4.0 |
| `id_sud_gsd` model, `assets_id/.../*.relabeled*.conllu` | SUD_Indonesian-GSD | CC BY-SA 4.0 |

Each source treebank's own `LICENSE.txt` is retained alongside its data.

## NonCommercial exclusion — SUD_English-GUM

The English EWT+GUM development setup used **SUD_English-GUM**, which is **CC BY-NC-SA 4.0
(NonCommercial)**. To keep the published English model and data free of the NonCommercial
restriction, **GUM is excluded entirely**: the shipped `en_sud_ewt` model is retrained on
EWT only, and no GUM-derived sentences, gold, or metrics are committed. The development-time
EWT+GUM figures quoted in the docs are reported for method context only.

## Attribution

- Surface-Syntactic Universal Dependencies — https://surfacesyntacticud.github.io/
- Universal Dependencies — https://universaldependencies.org/
- Please cite the individual UD/SUD treebanks when using these models; their authors are credited
  in each treebank's `LICENSE.txt`.
