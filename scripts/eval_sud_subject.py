#!/usr/bin/env python3
"""Compare the trained `sud_tagger` against the `sud_subject_rule` frame table, end to end.

Neither approach dominates across languages, so the choice has to be measured per language --
and measured on the same footing, which is the point of this script. The two are NOT comparable
on gold trees: the rule reads the parser's deprel and the head's predicted UPOS, so its accuracy
falls with parse quality, while the trained pipe reads only surface forms through its own encoder.
Both are therefore run over gold TOKENS with everything else predicted.

    scripts/eval_sud_subject.py --lang lzh
    scripts/eval_sud_subject.py --lang en --split dev

Reports P/R/F for each arm over tokens that carry `Subject=` in the treebank.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

import stamp_ja_inflection  # noqa: E402,F401
import seg_code  # noqa: E402,F401  (custom tokenisers + sud_tagger)
import sud_subject_rule  # noqa: E402,F401  (registers the rule factory)
from sud_misc import get_misc  # noqa: E402

TEST = {
    "en":  "assets/en_ewt-sud-{split}.relabeled_ext.conllu",
    # en_gum: the second English arm (EWT + the ten non-NonCommercial GUM genres).
    # A separate key, not a replacement -- `en` must keep pointing at the EWT-only files
    # that the released CC BY-SA en_sud_ewt wheel was built from.
    "en_gum": "assets/en_ewtgum-sud-{split}.relabeled_ext.conllu",
    "zh":  "assets_zh/SUD_Chinese-GSDBoth/zh_gsdboth-sud-{split}.relabeled_ext.conllu",
    "yue": "assets_yue/SUD_Cantonese-HK/yue_hk-sud-{split}.relabeled_ext.conllu",
    # lzh: the punctuation-restored, rule-merged generation its released arm trains on.
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-{split}.relabeled_ext.udep_ruled.punct.rulemerged.conllu",
    "fa":  "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-{split}.relabeled_ext.conllu",
    "la":  "assets_la/la_ittbproiel-sud-{split}.relabeled_ext.conllu",
    # csl_mwt, not csl_rev: the DCS/MWT generation. `corpus_sa_csl_rev` is the SUPERSEDED pausa-normalised representation (CLAUDE.md lists `rebuild_sa_csl_rev.sh` under "Superseded but kept"); its FORMs and tokenisation differ from the arm that ships, and it is UNRELABELLED (`udep` 7.89 % of tokens against 0.00 %). See the BUILD PROVENANCE table in docs/sanskrit.md.
    "sa":  "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.relabeled_ext.csl_mwt.conllu",
    # ta: TTB + the MWTT 80/10/10 split. te: the MWT-SPLIT MTG, not MTG as shipped -- the arm
    # under this layer trains on the split words (scripts/split_te_mwt.py).
    "ta":  "assets_ta/ta_ttb_mwtt-sud-{split}.conllu",
    "te":  "assets_te/te_mtg-sud-{split}.conllu",
}
LEMMA_ARM = {"sa": "training_sa_lemma3_noannot/model-best",
             "la": "training_la_aug_lemma/model-best"}
# Where the TRAINED pipe lives, when it is not training_<lang>_sud.
SUD_ARM = {"lzh": "training_lzh_rm_sud/model-best",
           # la ships the orthographically augmented chain (see package_sud.sh).
           "la": "training_la_aug_sud/model-best"}

# ⚠ lzh's rule arm cannot be `training_lzh_rm_morph` as it stands. `sud_subject_rule` keys on the
# HEAD LEMMA, and that arm has no lemma layer at all -- lzh replaced its trained lemmatizer with
# `han_lemma_lut`, which is attached at PACKAGING time. Evaluated on the bare arm the rule matches
# no frame and scores a flat 0.00, which looks like a finding and is an artefact. So the eval builds
# the same lemma layer the wheel ships, from the same treebank generation, and evaluates on that.
LZH_BASE = "training_lzh_rm_morph/model-best"
LZH_LUT_ARM = "build_lzh_eval_lut"
LZH_LUT_CONLLU = ("assets_lzh/SUD_Classical_Chinese-Kyoto-Both/"
                  "lzh_kyotoboth-sud-train.relabeled_ext.udep_ruled.punct.rulemerged.conllu")


def lzh_rule_arm():
    """The lzh arm WITH its lemma layer, built on demand so this needs no manual step."""
    out = pathlib.Path(LZH_LUT_ARM)
    if not out.exists():
        import subprocess
        print(f"  (building {LZH_LUT_ARM} -- the lemma layer the lzh wheel ships)")
        subprocess.run([sys.executable, "scripts/han_lemma_lut.py", "--build", LZH_BASE,
                        str(out), "--conllu", LZH_LUT_CONLLU], check=True,
                       stdout=subprocess.DEVNULL)
    return str(out)


def sentences(path):
    block = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if block:
                yield block
            block = []
            continue
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        block.append(f)
    if block:
        yield block


def gold_subject(fields):
    for item in fields[9].split("|"):
        if item.startswith("Subject="):
            return item.split("=", 1)[1]
    return None



# ⚠ Any harness that builds a doc from GOLD WORDS must put back what the tokeniser would have
# supplied. `Doc(vocab, words=[...])` yields tag == 0 and MORPH unset, so an arm whose encoder
# reads tokeniser-set channels (ja: XPOS + Inflection) runs with those inputs DELETED and every
# number still prints. Measured on ja's idiom layer: the parser's `unk` F fell 0.948 -> 0.786 and
# the rule lost ~6-8 F, which looked exactly like a base regression and was not one.
# `spaces=` matters for the same reason: without it doc.text gets a space after every token, and
# re-running the tokeniser over that analyses a string that never occurs at inference -- which
# measured WORSE than supplying no channels at all.
_CHAN_CACHE = {}


def _channel_nlp(nlp):
    """The tokeniser whose channels this arm reads, or None. Behavioural, so no language table:
    ja pre-sets 3/3 tags on an ASCII probe, en/ko/zh 0/3. Cached -- building it is not free."""
    key = id(nlp)
    if key not in _CHAN_CACHE:
        _CHAN_CACHE[key] = (stamp_ja_inflection.build_tokenizer()
                            if stamp_ja_inflection.needs_channels(nlp) else None)
    return _CHAN_CACHE[key]


def _gold_doc(nlp, rows):
    doc = Doc(nlp.vocab, words=[f[1] or "_" for f in rows],
              spaces=["SpaceAfter=No" not in (f[9] if len(f) > 9 else "") for f in rows])
    chan = _channel_nlp(nlp)
    if chan is not None:
        stamp_ja_inflection.apply_channels(doc, chan)
    return doc


def score(nlp, rows_iter):
    tp = fp = fn = skipped = 0
    for rows in rows_iter:
        doc = _gold_doc(nlp, rows)
        doc = nlp(doc)
        if len(doc) != len(rows):
            skipped += 1
            continue
        for tok, f in zip(doc, rows):
            pred = get_misc(tok, "Subject")
            gold = gold_subject(f)
            if pred and gold and pred == gold:
                tp += 1
            else:
                fp += bool(pred)
                fn += bool(gold)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp + fn, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(TEST))
    ap.add_argument("--split", default="test")
    # ⚠ SUD_ARM / LEMMA_ARM name FIXED directories, several of which are superseded generations
    # (lzh's entries are the BOTH-SCRIPTS arm while the wheel ships traditional-only). Standing
    # hazard 5 says re-measure this layer after any base change -- which is impossible if the arm
    # cannot be pointed at the thing that changed. --model does that.
    ap.add_argument("--model", default=None,
                    help="evaluate THIS pipeline instead of the hardcoded arm (e.g. the built wheel)")
    args = ap.parse_args()

    path = TEST[args.lang].format(split=args.split)
    if not pathlib.Path(path).exists():
        sys.exit(f"missing {path}")
    rows = list(sentences(path))

    print(f"{args.lang} {args.split}   (gold tokens, everything else predicted)")

    trained_dir = args.model or SUD_ARM.get(args.lang, f"training_{args.lang}_sud/model-best")
    if pathlib.Path(trained_dir).exists():
        nlp = spacy.load(trained_dir)
        P, R, F, n, sk = score(nlp, rows)
        print(f"  sud_tagger (trained)  gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}"
              + (f"  [{sk} sents skipped]" if sk else ""))
    else:
        print(f"  sud_tagger (trained)  -- {trained_dir} missing")

    lemma = (lzh_rule_arm() if args.lang == "lzh"
             else args.model or LEMMA_ARM.get(args.lang, f"training_{args.lang}_lemma/model-best"))
    if pathlib.Path(lemma).exists():
        nlp = spacy.load(lemma)
        if "sud_subject_rule" in nlp.pipe_names:
            nlp.remove_pipe("sud_subject_rule")
        nlp.add_pipe("sud_subject_rule", last=True, config={"lang": args.lang})
        P, R, F, n, sk = score(nlp, rows)
        print(f"  sud_subject_rule      gold={n:5}  P={P:7.2%} R={R:7.2%} F={F:7.2%}"
              + (f"  [{sk} sents skipped]" if sk else ""))
    else:
        print(f"  sud_subject_rule      -- {lemma} missing")


if __name__ == "__main__":
    main()
