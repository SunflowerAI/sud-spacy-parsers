#!/usr/bin/env python
"""Transfer the GSDSimp extended-scope relabel onto the real traditional GSD treebank.

SUD_Chinese-GSD (traditional) is the original; SUD_Chinese-GSDSimp is its simplified
auto-conversion — same sentences, sent_ids and (99.7%) the same trees. The udep->
comp:obl/mod relabel is a function of syntax + semantics, not script, so we overlay
each GSDSimp relabel change onto the aligned GSD token. The relabel only ever rewrites
udep, and we apply a change only where GSD's own deprel still equals the GSDSimp
*baseline* deprel (alignment guard), so the ~0.3% divergent tokens keep GSD's label.

Output: assets_zh/SUD_Chinese-GSD/zh_gsd-sud-{split}.relabeled_ext.conllu
"""
GSD = "assets_zh/SUD_Chinese-GSD/zh_gsd-sud"
SIMP = "assets_zh/SUD_Chinese-GSDSimp/zh_gsdsimp-sud"


def tok_lines(path):
    """Yield (sent_id, token_id, deprel, raw_cols_or_None) over content/MWT lines."""
    sid = None
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("# sent_id"):
            sid = ln.split("=", 1)[1].strip()
        elif ln.strip() and not ln.startswith("#"):
            c = ln.rstrip("\n").split("\t")
            yield sid, c[0], (c[7] if len(c) == 10 else None), c


def change_map(split):
    """(sent_id, tok_id) -> (orig_deprel, new_deprel) for every relabel change."""
    base = {(s, t): d for s, t, d, _ in tok_lines(f"{SIMP}-{split}.conllu")}
    out = {}
    for s, t, d, _ in tok_lines(f"{SIMP}-{split}.relabeled_ext.conllu"):
        o = base.get((s, t))
        if d is not None and o is not None and d != o:
            out[(s, t)] = (o, d)
    return out


def main():
    for split in ["train", "dev", "test"]:
        changes = change_map(split)
        applied = skipped = 0
        out = []
        sid = None
        for ln in open(f"{GSD}-{split}.conllu", encoding="utf-8"):
            if ln.startswith("# sent_id"):
                sid = ln.split("=", 1)[1].strip()
                out.append(ln)
                continue
            if not ln.strip() or ln.startswith("#"):
                out.append(ln)
                continue
            c = ln.rstrip("\n").split("\t")
            key = (sid, c[0])
            if key in changes and len(c) == 10:
                orig, new = changes[key]
                if c[7] == orig:
                    c[7] = new
                    applied += 1
                    out.append("\t".join(c) + "\n")
                    continue
                skipped += 1
            out.append(ln)
        dst = f"{GSD}-{split}.relabeled_ext.conllu"
        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(out)
        print(f"{split}: transferred {applied} relabels, skipped {skipped} "
              f"(divergent) of {len(changes)} GSDSimp changes -> {dst}")


if __name__ == "__main__":
    main()
