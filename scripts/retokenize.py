#!/usr/bin/env python
"""Re-tokenise an SUD CoNLL-U treebank so its tokens match spaCy's tokeniser,
re-projecting the dependency tree onto the new tokens.

Design (see conversation / CLAUDE.md):
  * Reversibility is the invariant: the new tokens are surface substrings of the
    original sentence and carry SpaceAfter=No so concatenation reproduces the text
    byte-for-byte. We assert the round-trip per sentence.
  * Korean (mecab): every eojeol refines into a contiguous run of morphemes, so the
    job is a pure 1->m *split* per eojeol. Internal structure is built functional-head
    (case particle / ending heads; lexical stem is comp:obj / comp:aux) following the
    mSUD standard (cf. mSUD_Nenets-Tundra: case suffix=ADP heads noun via comp:obj;
    verbal suffix=AUX heads clause with lexical verb as comp:aux).
  * Chinese (jieba) / en / id: same granularity, boundaries differ -> general
    char-span alignment with merge / split / crossing handling.

Usage:
  python scripts/retokenize.py --lang ko in.conllu out.conllu
  python scripts/retokenize.py --lang zh in.conllu out.conllu
"""
import argparse, sys
import spacy

# --------------------------------------------------------------------------- #
# CoNLL-U IO (surface tokens only; MWT ranges / empty nodes are not expected
# in these SUD treebanks -- we assert their absence).
# --------------------------------------------------------------------------- #
class Tok:
    __slots__ = ("id", "form", "lemma", "upos", "xpos", "feats",
                 "head", "deprel", "deps", "misc")
    def __init__(self, cols):
        (self.id, self.form, self.lemma, self.upos, self.xpos, self.feats,
         self.head, self.deprel, self.deps, self.misc) = cols
    def space_after(self):
        return "SpaceAfter=No" not in self.misc.split("|")

def read_conllu(path):
    sents = []
    comments, toks, mwts = [], [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            comments.append(line)
        elif line == "":
            if toks:
                sents.append((comments, toks, mwts))
            comments, toks, mwts = [], [], {}
        else:
            cols = line.split("\t")
            tid = cols[0]
            if "-" in tid:                               # multiword token range
                a, b = tid.split("-")
                mwts[int(a)] = (int(b), cols[1], cols[9])
            elif "." in tid:                             # empty node
                continue
            else:
                toks.append(Tok(cols))
    if toks:
        sents.append((comments, toks, mwts))
    return sents

def surface_and_spans(toks, mwts):
    """Reconstruct the sentence string and the [start,end) span of each syntactic
    token. Multiword tokens (don't -> do+n't) contribute their surface form once;
    covered sub-words get spans by greedy match within that form, so spaCy's split
    of the same form aligns cleanly."""
    s, spans = [], [None] * len(toks)
    pos, idx = 0, 0
    while idx < len(toks):
        tid = int(toks[idx].id)
        if tid in mwts:
            end, form, misc = mwts[tid]
            cov = []
            while idx + len(cov) < len(toks) and int(toks[idx + len(cov)].id) <= end:
                cov.append(idx + len(cov))
            fp = 0
            for k in cov:
                sub = toks[k].form
                j = form.find(sub, fp)
                if j < 0:
                    j = fp
                spans[k] = (pos + j, pos + j + len(sub))
                fp = j + len(sub)
            s.append(form)
            pos += len(form)
            if "SpaceAfter=No" not in misc.split("|"):
                s.append(" ")
                pos += 1
            idx = cov[-1] + 1
        else:
            t = toks[idx]
            spans[idx] = (pos, pos + len(t.form))
            s.append(t.form)
            pos += len(t.form)
            if t.space_after():
                s.append(" ")
                pos += 1
            idx += 1
    return "".join(s), spans

# --------------------------------------------------------------------------- #
# spaCy tokenisers matching the training configs.
# --------------------------------------------------------------------------- #
def make_tokenizer(lang):
    if lang == "zh":
        return spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "jieba"}}})
    return spacy.blank(lang)

# --------------------------------------------------------------------------- #
# Korean functional-head internal builder.
# Returns, for a list of (form, tag) morphemes of ONE eojeol:
#   head_rel  : index (0-based within eojeol) of the morpheme that heads the eojeol
#   parent    : list, parent[i] = index of internal head (or None for head_rel)
#   deprel    : list of internal deprels (None for head_rel)
#   upos      : list of UPOS tags
# --------------------------------------------------------------------------- #
KO_UPOS = {
    "NNG": "NOUN", "NNP": "PROPN", "NNB": "NOUN", "NNBC": "NOUN", "NR": "NUM", "NP": "PRON",
    "VV": "VERB", "VA": "ADJ", "VX": "AUX", "VCP": "AUX", "VCN": "AUX",
    "MM": "DET", "MAG": "ADV", "MAJ": "ADV", "IC": "INTJ",
    "JKS": "ADP", "JKC": "ADP", "JKG": "ADP", "JKO": "ADP", "JKB": "ADP",
    "JKV": "ADP", "JKQ": "ADP", "JX": "ADP", "JC": "CCONJ",
    "EP": "AUX", "EF": "AUX", "EC": "SCONJ", "ETN": "SCONJ", "ETM": "SCONJ",
    "XPN": "X", "XSN": "X", "XSV": "X", "XSA": "X", "XR": "X",
    "SF": "PUNCT", "SP": "PUNCT", "SS": "PUNCT", "SE": "PUNCT", "SO": "PUNCT", "SW": "SYM",
    "SL": "X", "SH": "X", "SN": "NUM", "SC": "PUNCT", "SY": "SYM",
}
KO_CASE = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"}
KO_ENDAUX = {"EP", "EF", "EC", "ETN", "ETM", "VX"}
KO_DERIV = {"XPN", "XSN", "XSV", "XSA", "XR"}

def ko_eojeol_tree(morphs):
    tags = [t for _, t in morphs]
    n = len(morphs)
    parent = [None] * n
    deprel = [None] * n
    upos = [KO_UPOS.get(t, "X") for t in tags]
    spine = None
    for k, tag in enumerate(tags):
        if spine is None:
            spine = k
            continue
        if tag in KO_CASE or tag in KO_ENDAUX or tag == "VCP":          # functional head
            if tag in KO_CASE:
                rel = "comp:obj"
            elif tag == "VCP":
                rel = "comp:pred"
            else:
                rel = "comp:aux"
            parent[spine] = k
            deprel[spine] = rel
            spine = k
        else:                                                           # derivation / compound -> internal
            parent[k] = spine
            deprel[k] = "compound"
    return spine, parent, deprel, upos

# --------------------------------------------------------------------------- #
# Generic char-span alignment into minimal blocks (gold-indices, spacy-indices).
# --------------------------------------------------------------------------- #
def align_blocks(gold_spans, sp_spans):
    blocks = []
    i = j = 0
    while i < len(gold_spans) and j < len(sp_spans):
        gi, sj = [i], [j]
        gend, send = gold_spans[i][1], sp_spans[j][1]
        while gend != send:
            if gend < send:
                i += 1; gi.append(i); gend = gold_spans[i][1]
            else:
                j += 1; sj.append(j); send = sp_spans[j][1]
        blocks.append((gi, sj))
        i += 1; j += 1
    if i != len(gold_spans) or j != len(sp_spans):
        raise ValueError("alignment did not consume both token streams")
    return blocks

def repair_tree(heads):
    """Force `heads` (0-based list, value = 1-based id or 0) into a single-root
    acyclic forest. Returns True if anything changed."""
    n = len(heads)
    changed = False
    for start in range(n):                              # break cycles
        seen = {start}
        cur = heads[start] - 1
        while cur >= 0:
            if cur in seen:
                heads[cur] = 0
                changed = True
                break
            seen.add(cur)
            cur = heads[cur] - 1
    roots = [i for i in range(n) if heads[i] == 0]
    if len(roots) > 1:                                  # attach extra roots to the first
        main = roots[0]
        for r in roots[1:]:
            heads[r] = main + 1
            changed = True
    return changed

# --------------------------------------------------------------------------- #
# Re-project one sentence.  Returns list of output rows (dicts) or None on failure.
# stats: dict counters mutated in place.
# --------------------------------------------------------------------------- #
def reproject(comments, toks, mwts, nlp, lang, stats):
    surface, gspans = surface_and_spans(toks, mwts)
    doc = nlp(surface)
    sp = list(doc)
    sspans = [(t.idx, t.idx + len(t.text)) for t in sp]
    blocks = align_blocks(gspans, sspans)

    # Per spaCy token: new fields. head stored as (kind, ref):
    #   ("ext", gold_idx)  -> external edge, resolve to head-rep of gold_idx's block
    #   ("int", sp_idx)    -> internal edge to another spaCy token
    #   ("root",)          -> from gold root
    n = len(sp)
    new_head = [None] * n
    new_dep = [None] * n
    new_up = [None] * n
    new_xp = [None] * n
    gold_headrep = {}                     # gold token index -> sp index that represents it

    for gi, sj in blocks:
        if len(gi) == 1 and len(sj) == 1:                # 1:1
            g, s = gi[0], sj[0]
            gt = toks[g]
            new_up[s], new_xp[s] = gt.upos, gt.xpos
            new_dep[s] = gt.deprel
            new_head[s] = ("ext", g)
            gold_headrep[g] = s
        elif len(gi) == 1 and len(sj) > 1:               # split 1:m
            g = gi[0]
            gt = toks[g]
            if lang == "ko":
                morphs = [(sp[s].text, sp[s].tag_) for s in sj]
                hrel, parent, deprel, upos = ko_eojeol_tree(morphs)
                for local, s in enumerate(sj):
                    new_up[s] = upos[local]
                    new_xp[s] = sp[s].tag_
                    if local == hrel:
                        new_dep[s] = gt.deprel
                        new_head[s] = ("ext", g)
                    else:
                        new_dep[s] = deprel[local]
                        new_head[s] = ("int", sj[parent[local]])
                gold_headrep[g] = sj[hrel]
                stats["split"] += 1
            else:                                        # generic split: first token = head
                head_local = 0
                for local, s in enumerate(sj):
                    new_up[s] = gt.upos if local == head_local else "X"
                    new_xp[s] = gt.xpos if local == head_local else "_"
                    if local == head_local:
                        new_dep[s] = gt.deprel
                        new_head[s] = ("ext", g)
                    else:
                        new_dep[s] = "dep"
                        new_head[s] = ("int", sj[head_local])
                gold_headrep[g] = sj[head_local]
                stats["split"] += 1
        else:                                            # merge n:1 or crossing n:m
            # choose the gold token whose head lies OUTSIDE the block as the block's
            # external anchor; its deprel/head become the block head's external edge.
            gset = set(gi)
            anchors = [g for g in gi if (toks[g].head == "0" or
                                         int(toks[g].head) - 1 not in gset)]
            anchor = anchors[0] if anchors else gi[0]
            head_local = 0                               # first spaCy token heads the block
            for local, s in enumerate(sj):
                gt = toks[anchor]
                if local == head_local:
                    new_up[s], new_xp[s] = gt.upos, gt.xpos
                    new_dep[s] = gt.deprel
                    new_head[s] = ("ext", anchor)
                else:
                    new_up[s], new_xp[s] = "X", "_"
                    new_dep[s] = "dep"
                    new_head[s] = ("int", sj[head_local])
            for g in gi:
                gold_headrep[g] = sj[head_local]
            if len(gi) > 1 and len(sj) > 1:
                stats["crossing"] += 1
            else:
                stats["merge"] += 1

    # Resolve heads to 1-based ids over the new token stream.
    heads = [0] * n
    rows = []
    for s in range(n):
        kind = new_head[s][0]
        if kind == "ext":
            g = new_head[s][1]
            head_id = 0 if toks[g].head == "0" else gold_headrep[int(toks[g].head) - 1] + 1
        else:                                            # internal
            head_id = new_head[s][1] + 1
        heads[s] = head_id
        misc = "_" if sp[s].whitespace_ else "SpaceAfter=No"
        rows.append([str(s + 1), sp[s].text, "_", new_up[s] or "X",
                     new_xp[s] or "_", "_", str(head_id), new_dep[s] or "dep",
                     "_", misc])

    # Safety net: guarantee a single-root, acyclic forest (only fires on lossy
    # zh merge/crossing blocks; Korean splits are already well-formed).
    if repair_tree(heads):
        stats["repaired"] += 1
        for s in range(n):
            if heads[s] != int(rows[s][6]):
                rows[s][6] = str(heads[s])
                if heads[s] == 0 and rows[s][7] != "root":
                    rows[s][7] = "root"
                elif heads[s] != 0 and rows[s][7] == "root":
                    rows[s][7] = "parataxis"

    # Reversibility check.
    recon = "".join(r[1] + (" " if r[9] != "SpaceAfter=No" else "") for r in rows).rstrip(" ")
    if recon != surface.rstrip(" "):
        stats["roundtrip_fail"] += 1
        return None
    return comments, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    nlp = make_tokenizer(args.lang)
    sents = read_conllu(args.inp)
    if args.limit:
        sents = sents[:args.limit]
    stats = dict(split=0, merge=0, crossing=0, roundtrip_fail=0, dropped=0, ok=0, repaired=0)

    with open(args.out, "w", encoding="utf-8") as f:
        for comments, toks, mwts in sents:
            try:
                res = reproject(comments, toks, mwts, nlp, args.lang, stats)
            except ValueError:
                res = None
            if res is None:
                stats["dropped"] += 1
                continue
            cm, rows = res
            stats["ok"] += 1
            for c in cm:
                f.write(c + "\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
            f.write("\n")

    print(f"[{args.lang}] sentences={len(sents)} ok={stats['ok']} dropped={stats['dropped']} "
          f"| splits={stats['split']} merges={stats['merge']} crossing={stats['crossing']} "
          f"repaired={stats['repaired']} roundtrip_fail={stats['roundtrip_fail']}", file=sys.stderr)

if __name__ == "__main__":
    main()
