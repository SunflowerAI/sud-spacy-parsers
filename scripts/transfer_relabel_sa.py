#!/usr/bin/env python3
"""Carry the sa udep relabel onto the DCS-MWT representation, then commit the residue by rule.

WHY IT IS NEEDED. The comp/mod relabel — the project's research contribution — was run on the
pre-MWT Sanskrit representation and never carried across. `corpus_sa_csl_mwt` therefore ships every
`udep` noncommittal, 12 926 of 163 802 tokens, while `corpus_sa_ext` holds 11 937 decisions nobody
is using. This is the zh situation one language over, and `transfer_relabel_gsd.py` is the
precedent: an expensive LLM pass is an asset attached to a TREEBANK, not to a representation of it.

WHY A TRANSFER IS SOUND HERE. The MWT rebuild changed the SURFACE (which characters are in which
token) and not the trees. Checked, not assumed: per split the relabelled file and the csl_mwt file
have identical `sent_id` sets and identical token counts (train 21 477 sentences / 161 959 tokens),
and this script additionally refuses unless the HEAD column agrees token for token. If the trees
ever diverge, the transfer stops rather than silently pasting labels onto a different structure.

THE RULE PASS. Held out (fit on train, applied to dev), a table over (udep SUBTYPE × Case) reproduces
**90.2 %** of the LLM's decisions from 72 cells at 99.7 % coverage; subtype alone gives 84.8 % from
11; adding the head lemma looks better in sample (0.960) and is WORSE held out (0.839) because 5 048
cells over 11 937 decisions is mostly singletons. So the table is (subtype × Case) and no wider.
It is fitted on TRAIN decisions only and used only to commit what the LLM pass left noncommittal.

⚠ The rules REPRODUCE the earlier pass, they do not validate it: whatever that pass got wrong is
inherited. And as `docs/udep-relabel.md` records, a relabel rewrites the TEST gold too, so
`comp:obl` F has a moving denominator and figures are not comparable across relabel generations.

    transfer_relabel_sa.py --src assets_sa/SUD_Sanskrit-Vedic --rules
"""
import argparse
import collections
import pathlib


def read(path):
    """-> [(comment_lines, [fields...])] per sentence, plus sent_id index."""
    sents, cur, com = [], [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur or com:
                sents.append((com, cur))
            cur, com = [], []
        elif line.startswith("#"):
            com.append(line)
        else:
            cur.append(line.split("\t"))
    if cur or com:
        sents.append((com, cur))
    return sents


def sent_id(com):
    for c in com:
        if c.startswith("# sent_id"):
            return c.split("=", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets_sa/SUD_Sanskrit-Vedic")
    ap.add_argument("--rules", action="store_true", help="commit the residue with derived rules")
    ap.add_argument("--also", nargs="*", default=[],
                    help="further CoNLL-U files that get the RULE pass only — no transfer, because "
                         "no relabelled counterpart exists. UFAL's 125 udep are the case: the LLM "
                         "pass was Vedic-only, and leaving them noncommittal would put two "
                         "labelling schemes inside one training corpus, which is worse than "
                         "committing them with a table fitted on the other treebank.")
    a = ap.parse_args()
    src = pathlib.Path(a.src)

    decisions = []          # (subtype, Case, label) from TRAIN, for fitting the rule table
    stats = collections.Counter()
    out_sents = {}

    for split in ("train", "dev", "test"):
        rel = read(src / f"sa_vedic-sud-{split}.relabeled_ext.conllu")
        mwt = read(src / f"sa_vedic-sud-{split}.csl_mwt.conllu")
        rel_by = {sent_id(c): t for c, t in rel if sent_id(c)}
        got = []
        for com, toks in mwt:
            sid = sent_id(com)
            ref = rel_by.get(sid)
            words = [t for t in toks if "-" not in t[0] and "." not in t[0]]
            if ref is None:
                stats[f"{split}: sentence absent from the relabelled file"] += 1
            else:
                if len(ref) != len(words):
                    raise SystemExit(f"{split} {sid}: {len(words)} tokens vs {len(ref)} relabelled")
                for w, r in zip(words, ref):
                    if w[6] != r[6]:
                        raise SystemExit(
                            f"{split} {sid}: HEAD differs ({w[6]} vs {r[6]}) — the two "
                            f"representations are NOT the same tree; refusing to transfer.")
                    if w[7].startswith("udep") and r[7] != w[7]:
                        if split == "train":
                            case = dict(kv.split("=") for kv in w[5].split("|") if "=" in kv).get("Case", "-")
                            decisions.append((w[7], case, r[7]))
                        w[7] = r[7]
                        stats[f"{split}: transferred"] += 1
                    elif w[7].startswith("udep"):
                        stats[f"{split}: still udep after transfer"] += 1
            got.append((com, toks))
        out_sents[split] = got

    for extra in a.also:
        out_sents[f"extra:{extra}"] = read(extra)
    rule = {}
    if a.rules:
        tab = collections.defaultdict(collections.Counter)
        for sub, case, lab in decisions:
            tab[(sub, case)][lab] += 1
        rule = {k: c.most_common(1)[0][0] for k, c in tab.items()}
        fallback = collections.Counter(l for _, _, l in decisions).most_common(1)[0][0]
        for split, sents in out_sents.items():
            for com, toks in sents:
                for w in toks:
                    if "-" in w[0] or "." in w[0] or not w[7].startswith("udep"):
                        continue
                    case = dict(kv.split("=") for kv in w[5].split("|") if "=" in kv).get("Case", "-")
                    hit = rule.get((w[7], case))
                    w[7] = hit if hit else fallback
                    stats[f"{split}: committed by rule ({'cell' if hit else 'fallback'})"] += 1

    for split, sents in out_sents.items():
        out = (pathlib.Path(split[6:]).with_suffix("").with_suffix(".relabeled_ext.csl_mwt.conllu")
               if split.startswith("extra:")
               else src / f"sa_vedic-sud-{split}.relabeled_ext.csl_mwt.conllu")
        with out.open("w", encoding="utf-8") as f:
            for com, toks in sents:
                for c in com:
                    f.write(c + "\n")
                for t in toks:
                    f.write("\t".join(t) + "\n")
                f.write("\n")
        print(f"wrote {out}")
    print(f"\nrule cells fitted from train decisions: {len(rule)}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<48}{v}")


if __name__ == "__main__":
    main()
