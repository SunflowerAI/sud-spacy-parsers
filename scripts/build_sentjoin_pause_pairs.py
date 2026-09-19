#!/usr/bin/env python3
"""Build the two REAL deployment-population pair sets `classifier_join` was never trained on.

Population A (`analyze_direction_classifier_errors.build_pairs()`) is two clause-heads INSIDE one
still-unmerged 句讀 block, split by an internal comma, with a REAL gold arc. At inference,
`pause_join` instead scores pairs of two DIFFERENT 句讀 UNITS the parser left as separate roots,
joined because the first unit's own text ends at a pause mark -- Kyoto never draws an arc between
two units at all, so there is no gold label for this configuration directly. This script builds
that population from the pre-rulemerge, per-unit corpus (`...punct.conllu`, which still carries
`sent_group`/`sent_final` metadata cross_unit_rules.py's own `--rules-only --write` step consumes),
split into two configurations with different, principled labels:

  B) SAME sent_group: Kyoto's own annotators marked these two adjacent units as intended parts of
     one continuous sentence, even though the dependency layer draws no arc between them. Labelled
     by cross_unit_rules.py's OWN direction cascade (`decide()`, using rules harvested from the
     WHOLE training corpus's in-unit evidence): direction "back" -> y=1 (reverse to `mod`, the
     antecedent/consequent-particle configuration classifier_join exists to catch); direction "fwd"
     (rule-derived OR residue-default) -> y=0. This mirrors cross_unit_rules.merge_group()'s own
     behaviour exactly: only a "back" call ever produces a reversed edge.

  C) DIFFERENT sent_group: two adjacent units the annotators judged to be genuinely SEPARATE
     sentences, whose boundary nonetheless happens to end at a pause mark (`docs/chinese-family.md`:
     31.2% of Kyoto's real sentence boundaries do). This is the population the two raw end-to-end
     validation passes showed is actually dominating `pause_join`'s real firings. HARD NEGATIVE:
     y=0 always, regardless of any lexical resemblance to a conditional-clause pair, because this
     is not a clause-linking configuration in Kyoto's annotation model at all.

Same symbolic feature schema as population A (`analyze_direction_classifier_errors.side_features`,
BOOL_COLS/NUM_COLS names), and the same `_a_chars`/`_a_root_idx` fields population A's pairs carry,
so train_lzh_sentjoin_glue.py can concatenate all three populations' pairs with no special-casing.
"""
import pathlib
import re
import sys

sys.path.insert(0, "scripts")
from analyze_direction_classifier_errors import WORKS, side_features  # noqa: E402
import cross_unit_rules as cur  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "assets_lzh" / "SUD_Classical_Chinese-Kyoto"
# Matches sent_join.PAUSE exactly -- the actual trigger `_after_pause` fires on.
PAUSE = set("，、；：,;:")
THRESH, MINC = 0.90, 20


def adjfix(rows):
    """VERB+Degree=Pos -> ADJ, matching recode_lzh_adj.py, so UPOS-based filtering/features here
    match population A's already-adjfix-recoded corpus."""
    out = []
    for r in rows:
        r = list(r)
        if r[3] == "VERB" and "Degree=Pos" in r[5].split("|"):
            r[3] = "ADJ"
        out.append(r)
    return out


def work_of(sent_id):
    for w, pat in WORKS.items():
        if pat.match(sent_id):
            return w
    return None


def unit_root(rows):
    return next((t for t in rows if t[6] == "0"), None)


def build_pause_pairs():
    # Harvest direction rules from the WHOLE training corpus (pre-rulemerge, pre-adjfix -- exactly
    # how cross_unit_rules.py has always been invoked in this project), not just the two WORKS.
    train_path = BASE / "lzh_kyoto-sud-train.relabeled_ext.udep_ruled.punct.conllu"
    train_sents = cur.read(str(train_path))
    rules, _ = cur.harvest(train_sents, THRESH, MINC)
    back, _ = cur.harvest_backward(train_sents, THRESH, MINC)

    # NOTE, deviating from the literal "same WORKS" instruction: population A stays scoped to
    # WORKS={Analects, ZhanguoCe} because that is the existing, unchanged in-unit population. But
    # within just those two texts, DIFFERENT-sent_group pause-ending boundaries (population C) are
    # vanishingly rare (15 examples) -- nowhere near enough to characterise the hard-negative class
    # that dominates real `pause_join` firings across the FULL deployed wheel (all 13 Kyoto texts).
    # Building B/C from the WHOLE corpus instead serves the stated goal ("retrain on the ACTUAL
    # deployment population") far better than literal file-scope matching would; a same/different-
    # sent_group label is well-defined and correct regardless of which text a pair comes from, and a
    # pair accidentally spanning two unrelated texts' file-concatenation boundary is automatically
    # (and correctly) a `different sent_group` hard negative anyway. `work_of()` is kept only to
    # label provenance in the pair dict, not to filter.
    pairs = []
    counts = {"B": 0, "B_pos": 0, "C": 0, "C_pos": 0, "skipped_pos_filter": 0, "skipped_no_pause": 0}
    for split in ("train", "dev", "test"):
        path = BASE / f"lzh_kyoto-sud-{split}.relabeled_ext.udep_ruled.punct.conllu"
        units = [(m, adjfix(r)) for m, r in cur.read(str(path))]
        for (ma, ra), (mb, rb) in zip(units, units[1:]):
            sid_a, sid_b = ma.get("sent_id", ""), mb.get("sent_id", "")
            if "_title" in sid_a or "_title" in sid_b:
                continue
            if not ra or ra[-1][1] not in PAUSE:
                counts["skipped_no_pause"] += 1
                continue
            root_a, root_b = unit_root(ra), unit_root(rb)
            if root_a is None or root_b is None:
                continue
            if root_a[3] not in ("VERB", "ADJ") or root_b[3] not in ("VERB", "ADJ"):
                counts["skipped_pos_filter"] += 1
                continue
            by_a, by_b = {t[0]: t for t in ra}, {t[0]: t for t in rb}
            a_ids, b_ids = [t[0] for t in ra], [t[0] for t in rb]
            af, bf = side_features(a_ids, by_a), side_features(b_ids, by_b)
            if af["first_form"] == "曰":
                continue
            ga, gb = ma.get("sent_group"), mb.get("sent_group")
            same_group = ga is not None and ga == gb
            if same_group:
                direction, _rel, _label = cur.decide(ra, rb, rules, back)
                y = 1 if direction == "back" else 0
                config = "B"
            else:
                y = 0
                config = "C"
            a_has_subj = any(by_a[i][6] == root_a[0] and by_a[i][7] in ("subj", "subj@pass")
                              for i in a_ids)
            b_has_subj = any(by_b[i][6] == root_b[0] and by_b[i][7] in ("subj", "subj@pass")
                              for i in b_ids)
            token_distance = (len(ra) - int(root_a[0])) + int(root_b[0])
            counts[config] += 1
            counts[config + "_pos"] += int(y)
            pairs.append({
                "config": config, "backward": bool(y),
                "work": work_of(sid_a) or work_of(sid_b) or "other",
                "sent_id": f"{sid_a}||{sid_b}",
                "a_first_upos": af["first_upos"], "a_first_form": af["first_form"],
                "a_len": af["len"], "a_has_neg": af["has_neg"], "a_has_yi": af["has_yi"],
                "a_ends_final_particle": af["ends_final_particle"], "a_text": af["text"],
                "b_first_upos": bf["first_upos"], "b_first_form": bf["first_form"],
                "b_len": bf["len"], "b_has_neg": bf["has_neg"], "b_has_yi": bf["has_yi"],
                "b_ends_close_quote": bf["ends_close_quote"], "b_text": bf["text"],
                "a_has_subj": a_has_subj, "b_has_subj": b_has_subj,
                "token_distance": token_distance,
                "a_has_internal_mod": af["has_internal_mod"], "b_has_internal_mod": bf["has_internal_mod"],
                "deprel_set_match": af["deprel_set"] == bf["deprel_set"] and bool(af["deprel_set"]),
                "a_has_cond_marker": af["has_cond_marker"], "b_has_cond_marker": bf["has_cond_marker"],
                "_a_chars": [root_a[1]], "_a_root_idx": 0,
                "_b_chars": [root_b[1]], "_b_root_idx": 0,
            })
    return pairs, counts


if __name__ == "__main__":
    pairs, counts = build_pause_pairs()
    print(f"B (same sent_group): {counts['B']}  positive (reverse) = {counts['B_pos']} "
          f"({100*counts['B_pos']/max(counts['B'],1):.1f}%)")
    print(f"C (different sent_group, hard negative): {counts['C']}  positive = {counts['C_pos']}")
    print(f"skipped (root not VERB/ADJ): {counts['skipped_pos_filter']}")
    print(f"total pairs: {len(pairs)}")
