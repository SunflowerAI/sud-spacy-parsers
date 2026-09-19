#!/usr/bin/env python3
"""Error analysis for the direction (forward/backward) classifier: rebuild the clean in-unit
training pairs with text, get out-of-fold CV predictions, and dump misclassified examples for
inspection. Companion to scripts/build_direction_classifier.py -- same features/pipeline, but
keeps a_text/b_text through to the output instead of discarding them after training."""
import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "scripts")
import cross_unit_rules as cur

ROOT = pathlib.Path(__file__).resolve().parent
ROOT = ROOT.parent
WORKS = {"Analects": re.compile(r"^KR1h0004"), "ZhanguoCe": re.compile(r"^KR2e0003")}
FINAL_PART = {"也", "矣", "焉", "乎", "哉", "與", "邪"}
NEG = {"不", "弗", "未", "毋", "勿", "非", "無", "莫"}
INTRA = {"flat@vv", "comp:aux", "compound@redup", "compound"}
# Declared conditional/concessive/consequential markers (matches cross_unit_rules.py's
# DECLARED_ANTECEDENT={若,苟,縱,雖} + DECLARED_CONSEQUENT={故}, plus 則 -- harvested there rather
# than declared, but the strongest single signal in this whole investigation). Checked 2026-09-18:
# "both sides' internal mod-particle is one of these" -> 97.8% backward (n=90); "both sides' mod-
# particle is some OTHER (neutral) adverb" -> 49.9%, i.e. near chance (n=1325). The crude boolean
# has_internal_mod blends these two very different populations; this set separates them.
COND_MARKERS = set("則必雖苟縱若故")

CAT_COLS = ["b_first_upos", "b_first_form"]
BOOL_COLS = ["a_has_subj", "a_has_neg", "a_has_yi", "a_ends_final_particle",
             "b_has_subj", "b_has_neg", "b_has_yi", "b_ends_close_quote",
             "a_has_internal_mod", "b_has_internal_mod", "deprel_set_match",
             "a_has_cond_marker", "b_has_cond_marker"]
NUM_COLS = ["a_len", "b_len", "token_distance"]
SK_COLS = [f"sk{i}" for i in range(768)]


def subtree_ids(rows, root_id):
    children = {}
    for r in rows:
        children.setdefault(r[6], []).append(r[0])
    span, stack = set(), [root_id]
    while stack:
        t = stack.pop()
        if t in span:
            continue
        span.add(t)
        stack.extend(children.get(t, []))
    return sorted(span, key=lambda x: int(x))


def side_features(rows_span, by_id):
    toks = [by_id[i] for i in rows_span]
    real = [t for t in toks if t[3] != "PUNCT"]
    first = real[0] if real else toks[0]
    has_neg = any(t[1] in NEG for t in toks)
    has_yi = "以" in "".join(t[1] for t in toks) and first[1] != "以"
    ends_final = toks[-1][1] in FINAL_PART if toks else False
    ends_close_quote = toks[-1][1] == "」" if toks else False
    text = "".join(t[1] for t in toks)
    # internal (word-level) deprel composition, e.g. an adverbial marker (則/必/雖/既/不) attaching
    # `mod` to this span's own verb -- NOT the parataxis/mod relation BETWEEN A and B, a completely
    # different thing. "has_internal_mod" and the deprel SET are strong, cheap, application-time-
    # computable signals found 2026-09-18: neither side having an internal mod is a clean 0/481
    # rule for forward; both sides having one lifts backward to 64.7% (n=2310); an exact deprel-set
    # match (punct excluded) lifts it to 92.2% (n=153, dominated by {comp:obj, mod} -- the
    # antithetical conditional/concessive-couplet construction's signature shape).
    deprel_set = frozenset(t[7] for t in toks if t[6] != "0" and t[7] != "punct")
    has_internal_mod = "mod" in deprel_set
    # refinement, 2026-09-18: WHICH particle carries the internal mod matters far more than
    # whether one exists at all -- see COND_MARKERS' docstring above.
    mod_forms = [t[1] for t in toks if t[6] != "0" and t[7] == "mod"]
    has_cond_marker = any(m in COND_MARKERS for m in mod_forms)
    return {"first_upos": first[3], "first_form": first[1], "len": len(toks),
            "has_neg": has_neg, "has_yi": has_yi, "ends_final_particle": ends_final,
            "ends_close_quote": ends_close_quote, "text": text,
            "deprel_set": deprel_set, "has_internal_mod": has_internal_mod,
            "has_cond_marker": has_cond_marker}


def read_full(path):
    sents, cur_, meta = [], [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur_:
                sents.append((meta, cur_))
            cur_, meta = [], {}
        elif line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
        else:
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]:
                cur_.append(f)
    if cur_:
        sents.append((meta, cur_))
    return sents


def build_pairs():
    pairs = []
    for work, pat in WORKS.items():
        for split in ("train", "dev", "test"):
            path = ROOT / "assets_lzh/SUD_Classical_Chinese-Kyoto" / f"lzh_kyoto-sud-{split}.relabeled_ext.udep_ruled.punct.rulemerged.adjfix.conllu"
            for meta, rows in read_full(path):
                sid = meta.get("sent_id", "")
                if not pat.match(sid) or "_title" in sid:
                    continue
                by_id = {r[0]: r for r in rows}
                for r in rows:
                    tid, up, head, deprel = r[0], r[3], r[6], r[7]
                    if head == "0" or head not in by_id or deprel in INTRA:
                        continue
                    head_up = by_id[head][3]
                    if up not in ("VERB", "ADJ") or head_up not in ("VERB", "ADJ"):
                        continue
                    if deprel not in ("parataxis", "mod"):
                        continue
                    dep_span = subtree_ids(rows, tid)
                    # ⚠ DISJOINT SPANS, NOT RAW SUBTREES. head's own subtree structurally CONTAINS
                    # dep's subtree (dep is a descendant of head, by construction) -- the same fact
                    # behind the "same_len" bug found earlier this session. Using the raw subtree
                    # for head's span makes one side a literal prefix/suffix substring of the other
                    # whenever the nested dependent sits at the edge of head's own character
                    # sequence -- a configuration that CANNOT occur in genuine cross-unit
                    # application (real 句讀 units are never nested inside each other). Subtracting
                    # dep's span from head's makes training pairs shape-match real residue pairs,
                    # matching the "head span minus dependent subtree" fix the multi-class pass
                    # already used for cross-unit shape-matching.
                    head_span = [i for i in subtree_ids(rows, head) if i not in set(dep_span)]
                    backward = int(tid) < int(head)
                    if backward:
                        a_id, a_span, b_id, b_span = tid, dep_span, head, head_span
                    else:
                        a_id, a_span, b_id, b_span = head, head_span, tid, dep_span
                    af = side_features(a_span, by_id)
                    bf = side_features(b_span, by_id)
                    if af["first_form"] == "曰":
                        continue  # deterministic quotative rule, excluded as elsewhere this session
                    pairs.append({
                        "work": work, "sent_id": sid, "backward": backward,
                        "a_first_upos": af["first_upos"], "a_first_form": af["first_form"],
                        "a_len": af["len"], "a_has_neg": af["has_neg"], "a_has_yi": af["has_yi"],
                        "a_ends_final_particle": af["ends_final_particle"], "a_text": af["text"],
                        "b_first_upos": bf["first_upos"], "b_first_form": bf["first_form"],
                        "b_len": bf["len"], "b_has_neg": bf["has_neg"], "b_has_yi": bf["has_yi"],
                        "b_ends_close_quote": bf["ends_close_quote"], "b_text": bf["text"],
                        "a_has_subj": any(by_id[i][6] == a_id and by_id[i][7] in ("subj", "subj@pass") for i in a_span),
                        "b_has_subj": any(by_id[i][6] == b_id and by_id[i][7] in ("subj", "subj@pass") for i in b_span),
                        "token_distance": abs(int(a_id) - int(b_id)),
                        "a_has_internal_mod": af["has_internal_mod"], "b_has_internal_mod": bf["has_internal_mod"],
                        "deprel_set_match": af["deprel_set"] == bf["deprel_set"] and bool(af["deprel_set"]),
                        "a_has_cond_marker": af["has_cond_marker"], "b_has_cond_marker": bf["has_cond_marker"],
                        "_a_root_tid": a_id, "_b_root_tid": b_id, "_rows": rows,
                        # root's own position WITHIN its (now disjoint) span's character sequence --
                        # needed so the SikuBERT encoder can pick out the exact root token's hidden
                        # state, matching the validated single-root-token methodology, instead of
                        # falling back to mean-pooling (a confound, not a fix -- caught 2026-09-18).
                        "_a_root_idx": a_span.index(a_id), "_b_root_idx": b_span.index(b_id),
                        "_a_chars": [by_id[i][1] for i in a_span], "_b_chars": [by_id[i][1] for i in b_span],
                    })
    return pairs


def main():
    print("=== 1. building pairs ===")
    pairs = build_pairs()
    print(f"n = {len(pairs)}  backward = {sum(p['backward'] for p in pairs)}")

    print("=== 2. live SikuBERT contextual embeddings (pair-local: A tokens + B tokens) ===")
    tk = AutoTokenizer.from_pretrained("SIKU-BERT/sikubert")
    model = AutoModel.from_pretrained("SIKU-BERT/sikubert")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()

    BATCH = 32
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        char_lists = []
        for c in chunk:
            rows = c["_rows"]
            by_id = {r[0]: r for r in rows}
            a_span = subtree_ids(rows, c["_a_root_tid"])
            b_span = subtree_ids(rows, c["_b_root_tid"])
            a_chars = [by_id[i_][1] for i_ in a_span]
            b_chars = [by_id[i_][1] for i_ in b_span]
            c["_a_len_tok"] = len(a_chars)
            c["_a_root_idx"] = a_span.index(c["_a_root_tid"])
            c["_b_root_idx"] = b_span.index(c["_b_root_tid"])
            char_lists.append(a_chars + b_chars)
        enc = tk(char_lists, is_split_into_words=True, return_tensors="pt", padding=True,
                 truncation=True, max_length=256)
        wids = [enc.word_ids(j) for j in range(len(chunk))]
        enc_d = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc_d).last_hidden_state.float().cpu().numpy()
        for j, c in enumerate(chunk):
            a_wi, b_wi = c["_a_root_idx"], c["_a_len_tok"] + c["_b_root_idx"]
            word_to_subpos = {}
            for pos, w in enumerate(wids[j]):
                if w is not None and w not in word_to_subpos:
                    word_to_subpos[w] = pos
            ap, bp = word_to_subpos.get(a_wi), word_to_subpos.get(b_wi)
            if ap is None or bp is None:
                continue
            c["a_vec"] = h[j, ap].tolist()
            c["b_vec"] = h[j, bp].tolist()
        if i % (BATCH * 20) == 0:
            print(f"    {i}/{len(pairs)}", flush=True)
    before = len(pairs)
    pairs = [c for c in pairs if "a_vec" in c]
    print(f"  with vectors: {len(pairs)} (dropped {before - len(pairs)})")

    print("=== 3. CV predict ===")
    y = np.array([1 if c["backward"] else 0 for c in pairs])
    form_counts = Counter(c["b_first_form"] for c in pairs)
    def bucket(f):
        return f if form_counts.get(f, 0) >= 5 else "OTHER"
    rows_df = []
    for c in pairs:
        row = {"b_first_upos": c["b_first_upos"], "b_first_form": bucket(c["b_first_form"]),
               "a_len": c["a_len"], "b_len": c["b_len"], "token_distance": c["token_distance"],
               "a_has_subj": int(c["a_has_subj"]), "a_has_neg": int(c["a_has_neg"]),
               "a_has_yi": int(c["a_has_yi"]), "a_ends_final_particle": int(c["a_ends_final_particle"]),
               "b_has_subj": int(c["b_has_subj"]), "b_has_neg": int(c["b_has_neg"]),
               "b_has_yi": int(c["b_has_yi"]), "b_ends_close_quote": int(c["b_ends_close_quote"])}
        diff = np.array(c["b_vec"]) - np.array(c["a_vec"])
        for i, v in enumerate(diff):
            row[f"sk{i}"] = v
        rows_df.append(row)
    X = pd.DataFrame(rows_df)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("pca", Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=20, random_state=0))]), SK_COLS),
    ], remainder="passthrough")
    clf = Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=0, class_weight="balanced"))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X[CAT_COLS + BOOL_COLS + NUM_COLS + SK_COLS], y, cv=skf, scoring="accuracy")
    print(f"CV accuracy: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}  (majority {100*max(y.mean(),1-y.mean()):.2f}%)")

    proba = cross_val_predict(clf, X[CAT_COLS + BOOL_COLS + NUM_COLS + SK_COLS], y, cv=skf, method="predict_proba")[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    errors = []
    for c, p, yt, yp in zip(pairs, proba, y, y_pred):
        if yt != yp:
            errors.append({
                "work": c["work"], "sent_id": c["sent_id"], "true_backward": bool(yt),
                "pred_backward": bool(yp), "p_backward": float(p),
                "a_text": c["a_text"], "b_text": c["b_text"],
                "b_first_form": c["b_first_form"], "a_first_form": c["a_first_form"],
            })
    print(f"\n{len(errors)} errors / {len(pairs)} ({100*len(errors)/len(pairs):.1f}%)")

    fn = [e for e in errors if e["true_backward"] and not e["pred_backward"]]  # missed real backward
    fp = [e for e in errors if not e["true_backward"] and e["pred_backward"]]  # false backward call
    print(f"  missed real backward (false negative): {len(fn)}")
    print(f"  wrongly called backward (false positive): {len(fp)}")

    out = ROOT / "gold" / "lzh_direction_classifier_errors.json"
    out.write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")

    print("\n--- false negatives (true backward, missed), by confidence ---")
    for e in sorted(fn, key=lambda x: x["p_backward"])[:20]:
        print(f"  p={e['p_backward']:.3f}  {e['a_text']} || {e['b_text']}")
    print("\n--- false positives (true forward, wrongly called backward), by confidence ---")
    for e in sorted(fp, key=lambda x: -x["p_backward"])[:20]:
        print(f"  p={e['p_backward']:.3f}  {e['a_text']} || {e['b_text']}")


if __name__ == "__main__":
    main()
