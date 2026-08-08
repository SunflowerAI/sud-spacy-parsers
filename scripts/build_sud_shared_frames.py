#!/usr/bin/env python3
"""Harvest the `Shared` decision table that `sud_shared_rule` reads, and write it as a module.

Derived from the TRAIN split only, so evaluating the rule on dev/test is honest. Within the
candidate mask of `sud_shared_data` the outcome is three-way -- `Shared=Yes`, `Shared=No`, or no
feature at all -- and all three are counted, because the mask deliberately over-generates (it
admits 1.6 tokens for every one the gold marks) and a rule that cannot decline has no way to give
that back. `O` is therefore a real outcome in the table, exactly as it is a real class in
`sud_tagger`.

Keys are the backoff ladder of `sud_shared_data.backoff_keys`, and the most specific level that
survives wins at lookup, so a sparse (deprel, head UPOS, position) context falls back to the
position alone rather than to nothing.

`--threshold` DEFAULTS TO A PLAIN MAJORITY, not to the 0.90 dominance test `apply_udep_rules.py`
uses, and the difference is deliberate: that script commits annotation to a treebank, where a wrong
label is worse than no label, whereas this table is a decision rule being compared against a
trained pipe and has to answer everywhere the mask asks. Dominance costs most of the recall for
precision nobody is buying -- en dev F 63.7 at 0.90 against 75.7 at 0.50, and zh and yue collapse
to nothing at all because no key of theirs is 90 % dominant. Raise it if you want the conservative
table instead.

    build_sud_shared_frames.py --out scripts/sud_shared_frames.py
    build_sud_shared_frames.py --report --split dev      # gold-tree accuracy, no model involved
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sud_shared_data import backoff_keys, candidates  # noqa: E402

# The split each released arm actually trains on. Every treebank here annotates `Shared`; only ja
# is left out, with 27 `Yes` in 168 333 tokens -- too sparse to derive anything from, the same
# call made for sa's `Subject`.
TRAIN = {
    "en":  "assets/en_ewt-sud-{split}.relabeled_ext.conllu",
    # en_gum: the second English arm (EWT + the ten non-NonCommercial GUM genres).
    # A separate key, not a replacement -- `en` must keep pointing at the EWT-only files
    # that the released CC BY-SA en_sud_ewt wheel was built from.
    "en_gum": "assets/en_ewtgum-sud-{split}.relabeled_ext.conllu",
    "zh":  "assets_zh/SUD_Chinese-GSDBoth/zh_gsdboth-sud-{split}.relabeled_ext.conllu",
    "yue": "assets_yue/SUD_Cantonese-HK/yue_hk-sud-{split}.relabeled_ext.conllu",
    # lzh: the PUNCTUATION-RESTORED, rule-merged generation its released arm is trained on. A
    # table harvested from the plain files would be keyed on a tree with no PUNCT tokens in it.
    "lzh": "assets_lzh/SUD_Classical_Chinese-Kyoto-Both/lzh_kyotoboth-sud-{split}.relabeled_ext.udep_ruled.punct.rulemerged.conllu",
    "fa":  "assets_fa/SUD_Persian-PerDT/fa_perdt-sud-{split}.relabeled_ext.conllu",
    "ar":  "assets_ar/SUD_Arabic-PADT/ar_padt-sud-{split}.relabeled_ext.conllu",
    "la":  "assets_la/la_ittbproiel-sud-{split}.relabeled_ext.conllu",
    "id":  "assets_id/SUD_Indonesian-GSD/id_gsd-sud-{split}.relabeled_ext.conllu",
    "ko":  "assets_ko/SUD_Korean-GSD/ko_gsd-sud-{split}.relabeled_ext.conllu",
    "sa":  "assets_sa/SUD_Sanskrit-Vedic/sa_vedic-sud-{split}.csl_rev.conllu",
}
# sa's arm trains on Vedic-train + UFAL combined, which lives outside the treebank directory.
TRAIN_OVERRIDE = {("sa", "train"): "corpus_sa_csl_rev/train.csl_rev.conllu"}

NEG = "O"


def path_for(lang, split):
    return TRAIN_OVERRIDE.get((lang, split), TRAIN[lang].format(split=split))


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


def col_get(col, key):
    for item in col.split("|"):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return None


def gold_shared(fields):
    """The gold value. `Shared` is a FEATS feature in every treebank here -- field 6, not 10 --
    but MISC is checked too so a hoisted or re-emitted file reads the same."""
    return col_get(fields[5], "Shared") or col_get(fields[9], "Shared")


def rows_to_tree(rows):
    """0-based heads (root points at itself) and relations, as `sud_shared_data` wants them."""
    index = {f[0]: i for i, f in enumerate(rows)}
    heads = []
    for i, f in enumerate(rows):
        h = index.get(f[6])
        heads.append(i if h is None else h)
    return heads, [f[7] for f in rows]


def observations(path):
    """Yield `(keys, gold)` for every candidate token in the file."""
    for rows in sentences(path):
        heads, deprels = rows_to_tree(rows)
        for i, position in candidates(heads, deprels):
            head_pos = rows[heads[i]][3]
            yield backoff_keys(deprels[i], head_pos, position), gold_shared(rows[i]) or NEG


def harvest(path, threshold, min_count):
    counts = collections.defaultdict(collections.Counter)
    for keys, gold in observations(path):
        for key in keys:
            counts[key][gold] += 1
    table = {}
    for key, counter in counts.items():
        total = sum(counter.values())
        value, hits = counter.most_common(1)[0]
        if total >= min_count and hits / total >= threshold:
            table[key] = value
    return table


def lookup(table, keys):
    for key in keys:
        if key in table:
            return table[key]
    return NEG


def report(table, path):
    """P/R/F against the gold on this split, over GOLD trees -- an upper bound on the component,
    which reads a predicted parse. `eval_sud_shared.py` is the end-to-end number."""
    tp = fp = fn = 0
    marked = set()
    for rows in sentences(path):
        heads, deprels = rows_to_tree(rows)
        cand = dict(candidates(heads, deprels))
        for i, f in enumerate(rows):
            gold = gold_shared(f)
            pred = NEG
            if i in cand:
                pred = lookup(table, backoff_keys(deprels[i], rows[heads[i]][3], cand[i]))
            if pred != NEG and gold == pred:
                tp += 1
            else:
                fp += pred != NEG
                fn += bool(gold)
            if gold:
                marked.add(i)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/sud_shared_frames.py")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--report", action="store_true",
                    help="score the harvested table on --split instead of writing the module")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--langs", nargs="+", default=sorted(TRAIN))
    args = ap.parse_args()

    tables = {}
    for lang in args.langs:
        train = path_for(lang, "train")
        if not pathlib.Path(train).exists():
            print(f"  {lang}: missing {train} -- skip")
            continue
        table = harvest(train, args.threshold, args.min_count)
        tables[lang] = table
        if args.report:
            held = path_for(lang, args.split)
            if not pathlib.Path(held).exists():
                print(f"  {lang}: {len(table):4d} keys   (no {args.split} split)")
                continue
            p, r, f, n = report(table, held)
            print(f"  {lang:4s} {len(table):4d} keys   {args.split} gold={n:6d}  "
                  f"P={p:6.2%} R={r:6.2%} F={f:6.2%}")
        else:
            print(f"  {lang}: {len(table)} keys")

    if args.report:
        return
    # The output is the WHOLE table module, not a per-language file, so writing it from a subset
    # silently deletes every language not in `--langs`. Caught the hard way: a `--langs lzh` run
    # to refresh one table left the module with one entry in it.
    missing = sorted(set(TRAIN) - set(tables))
    if missing:
        sys.exit(f"refusing to write {args.out}: it would drop {', '.join(missing)}. "
                 f"Re-run without --langs (use --report to inspect a subset).")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write('"""`Shared` decision table for `sud_shared_rule`. GENERATED by '
                 'scripts/build_sud_shared_frames.py -- do not edit by hand.\n\n'
                 'TABLE[lang]: a `sud_shared_data.backoff_keys` key -> "Yes", "No", or "O"\n'
                 '(no feature). Only keys whose majority outcome is dominant in the training\n'
                 'split are kept; lookup takes the most specific level present.\n"""\n')
        fh.write("TABLE = " + json.dumps(tables, ensure_ascii=False, indent=1,
                                         sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
