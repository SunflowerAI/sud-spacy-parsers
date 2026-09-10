"""Single --code entry point for the Sanskrit pretraining experiment.

spaCy's `--code` takes ONE file and this config needs three registrations across three modules:
the sa tokeniser, `sud.CompoundCorpus.v1` (the reader that copies ONLY the Compound feat from the
reference — sa's tokeniser-set input feature), and `sud.MultiHashEmbedAffix.v1`. Miss any one and
`spacy train` dies with E893 BEFORE training starts, which a driver that greps its output for
scores turns into a silently empty arm.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gold_tok_corpus   # noqa: F401  sud.CompoundCorpus.v1, sud.GoldTokCorpus.v1
import sa_tokenizer      # noqa: F401  sa.SanskritInputTokenizer.v2/v3
import sud_affix_embed   # noqa: F401  sud.MultiHashEmbedAffix.v1
import sud_analyser_embed  # noqa: F401  sud.AnalyserFeatsEmbed.v1 -- the sa BASE arm's own embed
import sud_feats_embed   # noqa: F401  sud.MultiHashEmbedFeats.v1 -- the ARC-FACTORED joint embed
