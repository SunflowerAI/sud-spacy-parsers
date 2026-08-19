#!/usr/bin/env python3
"""Single ``--code`` entry point for the sentence-segmentation retraining.

``spacy train --code`` accepts only ONE file (unlike ``spacy package``, which splits on commas),
so this module imports everything the per-language seg configs need: the gold-token multi-sentence
reader (required — segmentation training depends on it) plus every custom tokenizer/factory the
configs reference. The custom tokenizers are loaded best-effort so a language whose optional deps
are absent doesn't block the others; the reader is imported directly so its absence fails loudly.
"""
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gold_tok_corpus  # noqa: E402,F401  (registers sud.GoldTokCorpus.v1 — required)


def _load(fname):
    path = _HERE / fname
    if not path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # a language without its extra deps still loads its own tokenizer
        print(f"seg_code: skipped {fname}: {type(e).__name__}: {e}")


# vocal_augment registers the ar/fa vocalisation augmenters (and pulls in warm_start_tagger and
# sud_feats_embed). Added here rather than to each driver's own --code, because `spacy train` takes
# ONE file and a list that has to be remembered per driver is a list that gets missed -- the reason
# this module exists at all.
# The released vocalisers and the fa boundary normaliser: their factories are named in the ar/fa
# wheels' configs, so anything loading those models through this file needs them registered.
for _f in ("vocal_augment.py", "ar_vocalise.py", "fa_vocalise.py", "fa_align.py",
           "fa_normalise.py", "ar_tokenizer.py", "yue_tokenizer.py", "lzh_tokenizer.py",
           "sa_tokenizer.py", "clause_parser.py", "sud_unsandhi.py", "sa_devanagari.py",
           # registers sud.LatinEncliticTokenizer.v1. Its absence here meant every script that
           # loads a released la model through this file died with E893 -- found only once
           # something finally tried to OPEN the la wheel's own arm.
           "la_tokenizer.py", "la_enclitics.py",
           # registers sud.MultiHashEmbedAffix.v1 (per-component affix windows; sa morph/lemma)
           "sud_affix_embed.py",
           # registers sud.MultiHashEmbedFeats.v1 (one embedding table per morphological FEATURE,
           # rather than one hash of the whole FEATS bundle) -- the XPOS-downstream arms
           "sud_feats_embed.py",
           # registers sud.LexFieldEmbed.v1 (one table per comma-separated XPOS FIELD, read from a
           # shipped per-form lexicon). The lzh parser runs BEFORE any tagger, so there is no
           # predicted TAG for it to read; the lexicon is what supplies the channel at inference,
           # and it travels inside the model's own bytes.
           "sud_lex_embed.py",
           # registers sud.AnalyserFeatsEmbed.v1 (morphological CANDIDATE SETS from vidyut n
           # Heritage, multi-hot, looked up by token.norm_ — a constraint rather than a
           # prediction, so it cannot be confidently wrong the way a morphologiser can)
           "sud_analyser_embed.py",
           # registers sud.LemmaVecEmbed.v1 (distributional lemma vectors as a block)
           "sud_lemmavec_embed.py",
           # registers sud.WarmStartTagger.v1 (start a conditioned tagger AS the released one)
           "warm_start_tagger.py",
           # registers sud.CharSegTokenizer.v1 — the treebank-trained character segmenter used as
           # the TOKENIZER for zh (pkuseg 0.8385 -> 0.9202, the last +3 from jieba's segmentation
           # decision as an input channel) and id (enclitic split, 0.9985).
           # Must come after sa_presegment's dependencies; it imports that module lazily.
           "char_seg_tokenizer.py",
           # registers sud.SamplingCorpus.v1 — rebalance by SAMPLING rather than
           # duplicating docs, which inflates the parser's workload 10x
           "sampling_corpus.py",
           # registers sud.la_orth_variants.v1 — the Latin orthography augmenter, which replaces
           # the plain+macron union corpus with one copy resampled into a new edition style each
           # epoch (macrons, breves, j/v, æ/œ, sentence-initial capitals).
           "la_augment.py",
           # registers sud.dravidian_order_variants.v1 -- the ta/te word-order augmenter. Dravidian
           # is rigidly head-final (the side of the head is read off the data, never assigned); what
           # it re-linearises is the order of siblings in the preverbal field, which the treebanks
           # show is genuinely free (26 % OSV in ta, 23 % in te).
           "dravidian_augment.py",
           # registers sud.sa_case_variants.v1 — teaches that a capital carries no
           # syntax, which the tokeniser's case RESTORATION made necessary (14.62 %
           # of tokens change analysis when a sentence opens with a capital)
           "sa_augment.py",
           # sud_misc first: sud_tagger imports it. sud_shared_data holds the coordination
           # candidate mask, which sud_tagger looks up by name. (sud_idiom is packaging-time only
           # and needs no registration here, as with id_lemma_case_fix.)
           "sud_misc.py", "sud_shared_data.py", "sud_tagger.py",
           # the RULE variants: la ships sud_shared_rule and lzh sud_subject_rule instead of the
           # trained pipes, so a released arm cannot be opened without them.
           "sud_shared_rule.py", "sud_subject_rule.py", "sud_idiom.py",
           "sud_reported_rule.py", "sud_reported_data.py",
           # la_macronise sits in the released la PIPELINE, so the la wheel cannot be opened
           # without it -- the third such gap.
           "la_macronise.py", "id_lemma_case_fix.py",
           # registers sud.ZhTradTokenizer.v1 and the zh_script/lzh_script components. FOURTH
           # registration gap of the week -- each one broke loading a RELEASED arm, and each
           # surfaced only when something finally tried to OPEN that arm in-process.
           "zh_script.py",
           # han_lemma_lut IS needed here, unlike the other packaging-time components: it sits in
           # the lzh pipeline, so every later packaging step (add_clause_parser, the subject rule,
           # sud_idiom) has to be able to LOAD a model that already carries it, and each of those
           # loads through this file. Omitting it made add_clause_parser die with E002 while
           # package_sud.sh's `>/dev/null 2>&1` swallowed the error and shipped the old pipeline.
           "han_lemma_lut.py",
           # registers sud.SplitTok2Vec.v1 — the part-learned/part-frozen encoder
           "sud_split_tok2vec.py"):
    _load(_f)
