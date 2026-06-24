#!/usr/bin/env python3
"""Disambiguate SUD `udep` prepositional dependents of verbs as comp vs mod.

Pulls candidates from a gold CoNLL-U file (reliable heads/subtrees): a preposition
(UPOS=ADP) attached to a verb (UPOS=VERB) with relation `udep` (the noncommittal SUD
label). For each, it reconstructs the phrase headed by the verb and the phrase headed by
the preposition, builds the prompt, and asks qwen3:8b (no thinking) via Ollama.
"""
import argparse, json, os, random, urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
# Default kept at qwen3:8b so the existing en/zh/ko/id caches stay valid; override with
# OLLAMA_MODEL (e.g. gemma4) for the new languages. Benchmarked per-language in Phase 3.
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


def parse_conllu(path):
    """Yield sentences as (sent_id, tokens). tokens: list of dicts (1-indexed by 'id')."""
    sent_id, toks = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                continue
            if not line:
                if toks:
                    yield sent_id, toks
                sent_id, toks = None, []
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:  # multiword-token / empty-node lines
                continue
            misc = cols[9]
            toks.append({
                "id": int(cols[0]), "form": cols[1], "lemma": cols[2], "upos": cols[3],
                "xpos": cols[4], "feats": cols[5], "head": int(cols[6]), "deprel": cols[7],
                "space_after": "SpaceAfter=No" not in misc,
            })
    if toks:
        yield sent_id, toks


def descendants(tokens, root_id):
    """All token ids in the subtree rooted at root_id (inclusive)."""
    children = {}
    for t in tokens:
        children.setdefault(t["head"], []).append(t["id"])
    out, stack = set(), [root_id]
    while stack:
        cur = stack.pop()
        out.add(cur)
        stack.extend(children.get(cur, []))
    return out


def render(tokens, ids):
    """Reconstruct surface text for token ids, honoring SpaceAfter and trimming
    leading/trailing punctuation (e.g. a comma the preposition's subtree pulls in)."""
    by_id = {t["id"]: t for t in tokens}
    seq = sorted(ids)
    while seq and by_id[seq[0]]["upos"] == "PUNCT":
        seq.pop(0)
    while seq and by_id[seq[-1]]["upos"] == "PUNCT":
        seq.pop()
    out = ""
    for i in seq:
        t = by_id[i]
        out += t["form"] + (" " if t["space_after"] else "")
    return out.strip()


def find_candidates(path):
    cands = []
    for sent_id, toks in parse_conllu(path):
        by_id = {t["id"]: t for t in toks}
        for t in toks:
            if t["upos"] != "ADP" or t["deprel"] != "udep" or t["head"] == 0:
                continue
            head = by_id[t["head"]]
            if head["upos"] != "VERB":
                continue
            prep_phrase = render(toks, descendants(toks, t["id"]))
            verb_phrase = render(toks, descendants(toks, head["id"]))
            if len(prep_phrase.split()) < 2:  # bare stranded prep, skip
                continue
            cands.append({
                "sent_id": sent_id, "verb": head["form"],
                "prep_phrase": prep_phrase, "verb_phrase": verb_phrase,
            })
    return cands


def build_prompt(c):
    # Criteria taken from the SUD guidelines: comp = argument, mod = modifier.
    # An argument is "an obligatory participant in the semantic description" of the
    # governor, and SUD is distributional — the governor subcategorizes (selects) its
    # complement, often dictating the preposition; a modifier attaches freely.
    return (
        f'In "{c["verb_phrase"]}", the phrase "{c["prep_phrase"]}" depends on the '
        f'verb "{c["verb"]}". In Surface-Syntactic Universal Dependencies (SUD), decide '
        f'whether it is a COMPLEMENT or a MODIFIER of "{c["verb"]}":\n'
        f'- COMPLEMENT (comp): an argument of the verb — an obligatory participant in '
        f'the meaning of the verb, subcategorized (selected) by the verb. The verb '
        f'requires or licenses it and typically fixes the choice of preposition '
        f'(e.g. depend ON, talk ABOUT, refer TO, consist OF, accuse OF). Only certain '
        f'verbs admit it.\n'
        f'- MODIFIER (mod): an optional attribute that is not part of the verb\'s '
        f'argument structure and can be added freely to a wide range of verbs '
        f'(e.g. circumstances of time, place, manner, reason, or accompaniment: '
        f'in the morning, at home, with a smile, for that reason).\n'
        f'Apply the omissibility / free-addition test, in this order. Note that many '
        f'complements are optional too, so the mere fact that the phrase CAN be dropped '
        f'does not make it a modifier. First check selection: if the verb selects this '
        f'kind of phrase — it requires or licenses it and fixes its preposition, and '
        f'only certain verbs admit it — label it a COMPLEMENT, even if it could be '
        f'omitted. Only if the phrase fails that test — it is a free circumstantial that '
        f'could attach equally to many unrelated verbs — label it a MODIFIER.\n'
        f'Answer with exactly one word: complement or modifier.'
    )


def query(prompt):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                         "think": False, "options": {"temperature": 0}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.load(r)["response"].strip()
    # normalize: first word, lowercased, no trailing punctuation
    return raw.split()[0].strip(".,!").lower() if raw else raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", default="assets/en_sud-train.conllu")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cands = find_candidates(args.conllu)
    print(f"Found {len(cands)} `udep` prepositional dependents of verbs in {args.conllu}.\n")
    random.seed(args.seed)
    sample = random.sample(cands, min(args.n, len(cands)))

    for i, c in enumerate(sample, 1):
        prompt = build_prompt(c)
        ans = query(prompt)
        print(f"[{i}] sent_id={c['sent_id']}")
        print(f"    verb phrase : {c['verb_phrase']}")
        print(f"    prep phrase : {c['prep_phrase']}")
        print(f"    head verb   : {c['verb']}")
        print(f"    -> qwen3:8b : {ans}\n")


if __name__ == "__main__":
    main()
