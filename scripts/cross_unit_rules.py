#!/usr/bin/env python
"""Derive relations for links BETWEEN 句讀 units, from the annotators' own in-unit decisions.

Kyoto annotates each 句讀 unit as its own tree and never relates one unit to the next, so merging
units into punctuation-delimited sentences (see align_kanripo_punct.py) needs a relation that no
gold supplies. This harvests one from the only evidence there is: a 5.02-token unit still often
contains two clauses, and those links ARE annotated.

Method follows `udep_residue_audit.py` / `apply_udep_rules.py` — signatures are counted over the
treebank and a rule is emitted only where one relation dominates past `--thresh` on at least
`--minc` examples. Rules are DERIVED, never hardcoded; run with --show-evidence to see the support.

Three things about this evidence that the numbers do not say on their own:

1. **Serial-verb and auxiliary links are not clause linking.** `flat@vv`, `comp:aux` and
   `compound@redup` join verbs inside ONE clause, and including them (they are 21 % of verb-to-verb
   links) buries the distinction this is trying to draw. They are excluded.

2. **A cross-unit link has a DIRECTION, and both directions occur.** The obvious one attaches the
   following unit to the preceding (`comp:obj`, `conj:coord`). But a conditional or consecutive
   particle reverses it: in 學而不思則罔 the 則-clause is the main clause and what precedes it is a
   `mod` OF it. Harvesting post-head links alone therefore missed 則 almost entirely — its evidence
   lives in the pre-head configuration, where the preceding clause is `mod` 91.5 % of the time on
   213 examples. Both directions are harvested; a rule carries which way it points.

3. **The in-unit distribution is a BIASED prior for cross-unit links, so the majority baseline
   must not be transferred.** In-unit, `comp:obj` leads at 37 %, but a tight complement is exactly
   what an editor does NOT put a comma inside — the presence of a mark selects against it. The
   residue default is therefore left unset rather than filled with the in-unit majority.

A blind spot remains after that fix (see `--report`): 若 (n=8), 雖 (5), 苟 (2), 縱 (2) still have
essentially no in-unit evidence in either direction, because 句讀 segmentation puts a boundary
immediately before them, so the configuration the harvest needs — the particle INSIDE a unit — is
the one the corpus systematically lacks. Those are undecidable from this evidence and stay residue.

Usage:
    cross_unit_rules.py --train F.conllu [--apply F.punct.conllu] [--thresh 0.90] [--minc 20]
"""
import argparse
import collections

# verb-to-verb relations that join clauses; the rest are serial-verb / auxiliary constructions
INTRA = {"flat@vv", "comp:aux", "compound@redup", "compound"}
SPEECH = {"曰", "謂", "言", "云", "問", "對", "告", "報"}
OPEN_QUOTE, CLOSE_QUOTE = set("「『《〈“‘"), set("」』》〉”’")

# 曰 is admitted as a LEXICAL signature despite being an open-class VERB, which the closed-class
# guard would otherwise exclude. In the quotative frame (…, 曰: …) it is not an ordinary verb but
# the marker of a reporting frame, and it behaves like one: a unit opening with 曰 takes
# `parataxis` in 91.2 % of 137 in-unit examples. It is the single largest residue opener (961
# boundaries), and the carve-out is declared HERE rather than by lowering the threshold, so the
# closed-class principle stays intact and the exception stays auditable.
QUOTATIVE = {"曰"}

# DECLARED rules — from the grammar of Classical Chinese, not derived from the treebank, because
# 句讀 segmentation puts a unit boundary immediately before these particles and the in-unit
# evidence the harvest needs therefore does not exist (若 n=8, 雖 n=5, 苟 n=2, 縱 n=2).
#
# `故` opens a CONSEQUENT, like 則: what precedes it is the cause, so prev -mod-> next. The 8 in-unit
# examples that do exist agree at 88 %, below the bar but pointing the same way as the description.
DECLARED_CONSEQUENT = {"故": "consecutive 'therefore'; in-unit n=8 at 88 % agrees"}
# These open an ANTECEDENT — a conditional or concessive clause, which is the SUBORDINATE one. So
# the unit that opens with the particle is itself the modifier of what follows: this -mod-> next.
# Note this lands on the same relation and direction as the 則 rule, cued from the other side.
# 如 is deliberately EXCLUDED: it is conditional 'if' but also 'be like' / 'go to', and its in-unit
# behaviour reflects that ambiguity (comp:obj only 43.6 %), so it is not safely a conditional.
DECLARED_ANTECEDENT = {"若": "conditional 'if'", "苟": "conditional 'if only'",
                       "縱": "concessive 'even if'", "雖": "concessive 'although'"}


def read(path, want_groups=False):
    sents, cur, meta = [], [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append((meta, cur))
            cur, meta = [], {}
        elif line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
        else:
            f = line.split("\t")
            if "-" not in f[0] and "." not in f[0]:
                cur.append(f)
    if cur:
        sents.append((meta, cur))
    return sents


def subtree_first(tok, toks, kids):
    """Leftmost token of a token's own subtree — the word the clause actually opens with."""
    sub, frontier = [tok], [tok[0]]
    while frontier:
        nxt = []
        for k in frontier:
            for c in kids.get(k, []):
                sub.append(c)
                nxt.append(c[0])
        frontier = nxt
    return min(sub, key=lambda x: int(x[0]))


CLOSED = ("CCONJ", "SCONJ", "ADV", "PART", "PRON", "AUX", "DET")


def xclass(t):
    """Kyoto XPOS field 3 — the verb's semantic class (行為 action, 存在 existence, 可能 potential…)."""
    p = t[4].split(",")
    return p[2] if len(p) > 2 else "?"


def opener_key(first):
    return first[2] if len(first[2]) == 1 and first[3] in CLOSED else first[3]


def signatures(first, dep, head):
    """Signatures a clause licenses, most reliable first (ordered by held-out accuracy).

    A lexical signature is only trusted for a CLOSED class: `秦` opens a complement clause 63 times
    out of 63, but that is a state name, not a construction, and a rule keyed on it memorises the
    corpus. Open-class words go through their category instead, which is what generalises — an
    opening PROPN reaches 91.6 % where the individual names are noise.

    Measured on dev (derive on train, verify held out), the families rank:

        dependent clause's own head lemma      100.0 % accurate, 2.9 % coverage
        opener lemma                            99.5 %          17.4 %
        opener structure                        99.0 %          18.0 %
        opener + GOVERNOR CLASS                 98.1 %          21.4 %   <- best coverage
        governor lemma                          90.7 %           7.4 %   <- no margin, dropped
        governor class / dependent's class      no rules survive the 90 % bar at all

    So the heads DO carry information, but asymmetrically: the governing verb's class is worthless
    alone and useful only conjoined with the opener, while the dependent's own head is precise and
    almost never applicable. The opener dominates because Classical Chinese marks clause linkage at
    the LEFT EDGE — 而/則/其/是/此 encode the relation itself, where a head verb only says what the
    clause is about. Pairing the opener with the dependent's head lemma instead fragments the
    evidence (57 rules, coverage down to 9.6 %) and is not used.
    """
    sigs = []
    if (len(first[2]) == 1 and first[3] in CLOSED) or first[2] in QUOTATIVE:
        sigs.append(("lemma", first[2]))
    sigs.append(("dep-lemma", dep[2]))
    if head[2] in SPEECH:
        sigs.append(("head", "speech-verb"))
    sigs.append(("pair", opener_key(first), xclass(head)))
    sigs.append(("struct", f"{first[3]}/{'subj' if first[7] == 'subj' else 'other'}"))
    return sigs


def harvest_backward(sents, thresh, minc):
    """Rules pointing the other way: keyed on what the HEAD clause opens with, giving the relation
    the PRECEDING clause bears to it. This is where 則/故 live. Same closed-class restriction, so
    the polysemous open-class openers (爲 at 90.5 %) do not become rules."""
    ev = collections.defaultdict(collections.Counter)
    for _, toks in sents:
        idx = {t[0]: t for t in toks}
        for t in toks:
            if t[6] == "0" or t[7] in INTRA:
                continue
            h = idx.get(t[6])
            if h is None or t[3] not in ("VERB", "AUX") or h[3] not in ("VERB", "AUX"):
                continue
            if int(t[0]) > int(h[0]):
                continue                                  # post-head handled by harvest()
            span = [x for x in toks if int(t[0]) < int(x[0]) <= int(h[0])]
            if not span:
                continue
            opener = span[0]
            if len(opener[2]) == 1 and opener[3] in ("CCONJ", "SCONJ", "ADV", "PART", "PRON",
                                                     "AUX", "DET"):
                ev[("back-lemma", opener[2])][t[7]] += 1
    rules = {}
    for sig, c in ev.items():
        n = sum(c.values())
        rel, k = c.most_common(1)[0]
        if n >= minc and k / n >= thresh:
            rules[sig] = (rel, k / n, n)
    return rules, ev


def harvest(sents, thresh, minc):
    ev = collections.defaultdict(collections.Counter)
    for _, toks in sents:
        idx = {t[0]: t for t in toks}
        kids = collections.defaultdict(list)
        for t in toks:
            if t[6] != "0":
                kids[t[6]].append(t)
        for t in toks:
            if t[6] == "0" or t[7] in INTRA:
                continue
            h = idx.get(t[6])
            if h is None or t[3] not in ("VERB", "AUX") or h[3] not in ("VERB", "AUX"):
                continue
            if int(t[0]) < int(h[0]):
                continue
            first = subtree_first(t, toks, kids)
            for sig in signatures(first, t, h):
                ev[sig][t[7]] += 1
    rules = {}
    for sig, c in ev.items():
        n = sum(c.values())
        rel, k = c.most_common(1)[0]
        if n >= minc and k / n >= thresh:
            rules[sig] = (rel, k / n, n)
    return rules, ev


def decide(a, b, rules, back):
    """The cascade for one boundary. Returns (direction, relation, label); `label` is None for the
    residue, which the caller defaults to `parataxis` — the SUD relation for juxtaposed clauses,
    and what an unmarked boundary between two 句讀 units mostly IS. It is a default, not a decision:
    an LLM binary was benchmarked on the analogous in-unit links and scored 56.5 % against a 58.5 %
    majority baseline on the same sample, so there is nothing better to put here.
    """
    marks = [t[1] for t in a if t[3] == "PUNCT"] + [t[1] for t in b if t[3] == "PUNCT"]
    root, first, dep = unit_signature(a, b, marks)
    if root is None:
        return "fwd", "parataxis", None
    a_first = next((t for t in a if t[3] != "PUNCT"), None)
    # declared antecedent: THIS unit opens with a conditional / concessive particle, so it is the
    # subordinate clause and modifies what follows
    if a_first is not None and a_first[2] in DECLARED_ANTECEDENT:
        return "back", "mod", f"prev -mod-> next  (prev opens {a_first[2]}, declared)"
    if first[2] in DECLARED_CONSEQUENT:
        return "back", "mod", f"prev -mod-> next  (opens {first[2]}, declared)"
    # the quote rule is NOT harvestable — the treebank has no quotation marks at all — so it is
    # declared, with the speech-verb evidence as its warrant. The previous unit's root IS the
    # speech verb in the ordinary frame (子曰: 曰 is the root, 子 its subj), so no special target.
    if any(m in OPEN_QUOTE for m in marks) and any(t[2] in SPEECH for t in a):
        return "fwd", "comp:obj", "quote after speech verb -> comp:obj (declared)"
    # backward before forward: a consecutive particle opening the NEXT unit makes the PRECEDING one
    # a modifier of it, and that reading overrides any forward signature
    bsig = ("back-lemma", first[2])
    if bsig in back:
        return "back", back[bsig][0], f"prev -{back[bsig][0]}-> next  (opens {first[2]})"
    for sig in signatures(first, dep, root):
        if sig in rules:
            return "fwd", rules[sig][0], f"{sig[0]}:{':'.join(sig[1:])} -> {rules[sig][0]}"
    return "fwd", "parataxis", None


def merge_group(units, edges):
    """Merge a sent_group's units into one sentence, given `edges[i] = (direction, relation)` for
    the boundary between unit i and unit i+1.

    Each boundary contributes exactly ONE dependency edge, either forward (unit i+1's root becomes
    a dependent of unit i's root) or backward (unit i's root becomes a `mod` of unit i+1's root).
    n units and n-1 edges with at most one head each therefore give a tree — no cycle is possible,
    because edges only ever join ADJACENT units and a two-cycle would need both directions at the
    same boundary.

    Order matters, and getting it wrong is not a technicality. In 子曰：「學而不思則罔」 the quote
    rule attaches 學而不思 to 曰, and then 則罔 wants 學而不思 as its own modifier — which looks like a
    clash over one head. It is not: a subordinating edge changes WHICH UNIT HEADS THE SPAN, so the
    complement of 曰 was never 學而不思 but 罔, the main clause of the quoted sentence. Resolving the
    subordinating edges FIRST and then attaching each forward edge to the head of the span it points
    into gets that right, and removes the clash entirely rather than papering over it.

    So: pass 1 lays down every backward edge (only boundary i can subordinate unit i, so these never
    collide with each other); pass 2 attaches forward edges from `span_head(i+1)` — the end of the
    backward chain starting at i+1 — to unit i.

    No conflict survives, and that is a property of the construction rather than luck: for two
    forward boundaries i < j, the chain out of i+1 can only run past j if boundary j were itself
    backward, so `span_head(i+1) <= j < span_head(j+1)` and the two targets are always distinct.
    Cycles are impossible for the same reason they were before — backward edges point right, forward
    edges point strictly left, and no unit takes both.
    """
    sub = {}                                   # backward edges only; always point rightwards
    derived = {}                               # unit index -> was its incoming edge rule-derived?
    for i, (direction, rel, is_rule) in enumerate(edges):
        if direction == "back":
            sub[i] = (i + 1, rel)
            derived[i] = is_rule

    def span_head(u):
        while u in sub:
            u = sub[u][0]
        return u

    head_of, fallbacks = dict(sub), 0
    for i, (direction, rel, is_rule) in enumerate(edges):
        if direction != "back":
            src = span_head(i + 1)
            if src in head_of:                 # proved unreachable; counted so a proof bug shows up
                fallbacks += 1
                continue
            head_of[src] = (i, rel)
            derived[src] = is_rule

    offs, run = [], 0
    for u in units:
        offs.append(run)
        run += len(u)
    roots = [next((k for k, t in enumerate(u) if t[6] == "0"), 0) for u in units]

    out = []
    for i, u in enumerate(units):
        for k, t in enumerate(u):
            t = list(t)
            t[0] = str(offs[i] + k + 1)
            if t[6] == "0":
                if i in head_of:
                    hu, rel = head_of[i]
                    t[6] = str(offs[hu] + roots[hu] + 1)
                    t[7] = rel
                    # Provenance, in MISC, on the DEPENDENT of every cross-unit edge. Downstream
                    # needs to tell a derived relation from the `parataxis` default, because the
                    # two warrant completely different treatment in training and in scoring.
                    misc = [x for x in t[9].split("|") if x and x != "_"]
                    misc.append("CrossUnit=" + ("rule" if derived.get(i) else "default"))
                    t[9] = "|".join(misc)
                else:
                    t[6], t[7] = "0", "root"
            else:
                t[6] = str(offs[i] + int(t[6]))
            out.append(t)
    return out, fallbacks


def unit_signature(prev, nxt, marks):
    """Signature of a CROSS-UNIT boundary, in the same coordinate system as the harvest: the
    following unit's opening token AND its own root (the dependent), and the preceding unit's root
    (the governor). All three are available at inference — each unit is parsed on its own."""
    root = next((t for t in prev if t[6] == "0"), None)
    dep = next((t for t in nxt if t[6] == "0"), None)
    first = next((t for t in nxt if t[3] != "PUNCT"), None)
    if root is None or first is None or dep is None:
        return None, None, None
    return root, first, dep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--apply")
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--minc", type=int, default=20)
    ap.add_argument("--show-evidence", action="store_true")
    ap.add_argument("--write", metavar="SUFFIX",
                    help="merge each sent_group into ONE sentence, linking unit roots by the rules "
                         "and by `parataxis` where no rule fires, and write to *SUFFIX.conllu")
    ap.add_argument("--report", action="store_true", help="also report the in-unit blind spot")
    # Merge ONLY the rule-derived boundaries and leave the rest as sentence breaks. This is the
    # supported way to keep the `parataxis` default out of the parser's loss: spaCy cannot express
    # an unknown head in a reference — `get_aligned_parse` reads `token.head.i`, which for an unset
    # head is the token itself, so arc-eager sees a ROOT, and a corpus built that way collapsed the
    # parser to DEP_UAS 18. With no arc there is no loss to mask, and the unit root at an unmerged
    # boundary is a genuine root, so the supervision is correct rather than merely absent.
    ap.add_argument("--rules-only", action="store_true",
                    help="merge only rule-derived boundaries; break the sentence at the rest")
    args = ap.parse_args()

    train = read(args.train)
    rules, ev = harvest(train, args.thresh, args.minc)
    back, bev = harvest_backward(train, args.thresh, args.minc)
    print(f"DERIVED RULES  (dominance >= {args.thresh:.0%} on >= {args.minc} in-unit examples)\n")
    print(f"{'signature':<28} {'direction':<12} {'relation':<12} {'share':>7} {'support':>8}")
    for sig, (rel, share, n) in sorted(rules.items(), key=lambda kv: -kv[1][2]):
        print(f"{sig[0] + ':' + sig[1]:<28} {'next->prev':<12} {rel:<12} {share:>6.1%} {n:>8}")
    for sig, (rel, share, n) in sorted(back.items(), key=lambda kv: -kv[1][2]):
        print(f"{'lemma:' + sig[1]:<28} {'prev->next':<12} {rel:<12} {share:>6.1%} {n:>8}")

    if args.report:
        print("\nBLIND SPOT — clause-linking particles, in-unit support in BOTH directions:")
        for p in ("則", "若", "雖", "故", "苟", "縱", "而", "以", "其"):
            best = ("-", 0, "")
            for tbl, lab in ((ev.get(("lemma", p)), "next->prev"),
                             (bev.get(("back-lemma", p)), "prev->next")):
                n = sum(tbl.values()) if tbl else 0
                if n > best[1]:
                    rel, k = tbl.most_common(1)[0]
                    best = (f"{rel} {k/n:.0%}", n, lab)
            note = "" if best[1] >= args.minc else "   <- too rare in-unit to derive"
            print(f"   {p}  n={best[1]:<6} {best[0]:<18} {best[2]}{note}")
        print("   句讀 segmentation puts a boundary immediately BEFORE these, so the configuration"
              "\n   the harvest needs (particle inside a unit) is the one the corpus lacks.")

    if not args.apply:
        return

    # ---- apply to real cross-unit boundaries
    sents = read(args.apply)
    groups, cur, curmeta, gid = [], [], None, None
    for meta, toks in sents:
        g = meta.get("sent_group")
        if g != gid and cur:
            groups.append((curmeta, cur))
            cur = []
        if not cur:
            curmeta = meta
        gid = g
        cur.append(toks)
    if cur:
        groups.append((curmeta, cur))

    fired = collections.Counter()
    residue = collections.Counter()
    nb = 0
    merged, fallbacks = [], 0
    for meta, units in groups:
        edges = []
        for a, b in zip(units, units[1:]):
            nb += 1
            direction, rel, label = decide(a, b, rules, back)
            edges.append((direction, rel, label is not None))
            if label is None:
                # a unit can be punctuation only (a stray bracket between two units), in which
                # case there is no opener to key on — still counted, under its own bucket
                first = next((t for t in b if t[3] != "PUNCT"), None)
                residue["(punctuation-only unit)" if first is None else
                        f"{first[3]}/{first[2] if len(first[2]) == 1 else '·'}"] += 1
            else:
                fired[label] += 1
        if args.write:
            # --rules-only: split the group into runs at every DEFAULT boundary, then merge each
            # run on its own. A run's edges are all rule-derived, so nothing invented enters a tree.
            runs = [(units, edges)]
            if args.rules_only:
                runs, cu, ce = [], [units[0]], []
                for nxt, e in zip(units[1:], edges):
                    if e[2]:
                        cu.append(nxt); ce.append(e)
                    else:
                        runs.append((cu, ce)); cu, ce = [nxt], []
                runs.append((cu, ce))
            for r, (ru, re_) in enumerate(runs):
                toks, fb = merge_group(ru, re_)
                fallbacks += fb
                m = dict(meta)
                if len(runs) > 1:
                    m["sent_id"] = f"{meta.get('sent_id', '')}#{r}"
                merged.append((m, toks))

    print(f"\nAPPLIED to {nb} cross-unit boundaries in {args.apply.split('/')[-1]}\n")
    tot = sum(fired.values())
    for k, v in fired.most_common():
        print(f"   {v:>6}  {100*v/nb:5.1f} %   {k}")
    print(f"   {'-'*6}")
    print(f"   {tot:>6}  {100*tot/nb:5.1f} %   COMMITTED")
    print(f"   {nb-tot:>6}  {100*(nb-tot)/nb:5.1f} %   residue -> parataxis (default), top openers:")
    for k, v in residue.most_common(8):
        print(f"             {v:>6}  {k}")

    if args.write:
        out = args.apply.replace(".conllu", args.write + ".conllu")
        with open(out, "w", encoding="utf-8") as fh:
            for meta, toks in merged:
                sid = meta.get("sent_id", "")
                fh.write(f"# sent_id = {sid}\n")
                fh.write("# text = " + "".join(t[1] for t in toks) + "\n")
                for t in toks:
                    fh.write("\t".join(t) + "\n")
                fh.write("\n")
        note = ("no head conflicts, as the construction guarantees" if not fallbacks else
                f"*** {fallbacks} UNRESOLVED CONFLICTS — the span-head argument is wrong ***")
        print(f"\nwrote {out}: {len(merged)} sentences from {sum(1 for _ in sents)} units"
              f"  ({note})")


if __name__ == "__main__":
    main()
