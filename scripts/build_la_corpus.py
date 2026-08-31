#!/usr/bin/env python3
"""Assemble a large, DOMAIN-MATCHED Latin corpus for homegrown vectors.

Four sources, chosen because they are the domains the la treebanks come from -- classical (Perseus),
late and medieval scholastic (ITTB is Aquinas), the Vulgate (PROIEL) -- plus modern written Latin:

    wikisource.xml.bz2   la.wikisource: classical and medieval texts in full
    wikipedia.xml.bz2    la.wikipedia
    latin_library.tar.gz cltk/latin_text_latin_library, public-domain classical texts
    perseus.tar.gz       PerseusDL/canonical-latinLit -- ⚠ ONLY the `-latN.xml` files. The same
                         repository ships ENGLISH TRANSLATIONS as `-engN.xml`, and folding those in
                         would train English vectors under Latin keys.

Common Crawl is deliberately absent: cc.la is exactly what the published fastText Latin vectors are
built on, and it reaches only 52.0 % of our treebank types.

Everything is folded onto ONE orthography by `aligned_vectors._norm_la` -- lowercase, macrons and
breves off, ae/oe ligatures expanded, v->u, j->i. Our treebanks are u-dominant (2.2 % of tokens
carry a `v`, none carry a `j`) while all four sources use `v` and `j` freely, so without the fold
the corpus and the treebank barely share a vocabulary. The loader applies the same function, which
is why it lives there and is imported here rather than the other way round.
"""
import argparse, bz2, io, pathlib, re, sys, tarfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from aligned_vectors import _norm_la

TEXT_OPEN = re.compile(r"<text[^>]*>")
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TABLE = re.compile(r"\{\|.*?\|\}", re.S)
FILELINK = re.compile(r"\[\[(?:File|Image|Fasciculus|Imago|Categoria|Category):[^\]]*\]\]", re.I)
PIPELINK = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
LINK = re.compile(r"\[\[([^\]]*)\]\]")
REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
TAG = re.compile(r"<[^>]+>")
XMLNOTE = re.compile(r"<note[^>]*>.*?</note>|<bibl[^>]*>.*?</bibl>|<head[^>]*>.*?</head>", re.S)
SENT = re.compile(r"[.!?;:]+")
WORD = re.compile(r"[a-z]+")

# A cheap Latin-ness gate. Wikisource and the Latin Library both carry editorial matter in other
# languages, and a vector table keyed by Latin words must not learn English from it.
LATIN_FN = {"et", "in", "non", "est", "ad", "cum", "qui", "quod", "ut", "sed", "de", "per", "ex",
            "autem", "esse", "enim", "quae", "sunt", "si", "hoc", "atque", "nec", "ab", "aut",
            "eius", "quam", "etiam", "iam", "sic", "tamen", "uel", "omnia", "erat", "se", "a"}


def strip_wiki(t):
    t = REF.sub(" ", t)
    t = TABLE.sub(" ", t)
    for _ in range(6):                      # templates nest; peel from the inside out
        t, n = TEMPLATE.subn(" ", t)
        if not n:
            break
    t = FILELINK.sub(" ", t)
    t = PIPELINK.sub(r"\1", t)
    t = LINK.sub(r"\1", t)
    t = TAG.sub(" ", t)
    t = t.replace("'''", " ").replace("''", " ")
    t = re.sub(r"^[=*#:;|!].*$", " ", t, flags=re.M)
    return t


def wiki_texts(path):
    """Stream <text>...</text> bodies out of a MediaWiki dump without holding it in memory."""
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        buf, inside = [], False
        for line in fh:
            if not inside:
                m = TEXT_OPEN.search(line)
                if m:
                    inside = True
                    line = line[m.end():]
                else:
                    continue
            if "</text>" in line:
                buf.append(line[: line.index("</text>")])
                yield "".join(buf); buf, inside = [], False
            else:
                buf.append(line)


def tar_texts(path, keep):
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not keep(m.name):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            yield fh.read().decode("utf-8", errors="replace")


def emit(raw, out, stats, kind):
    if kind == "wiki":
        raw = strip_wiki(raw)
    elif kind == "tei":
        raw = TAG.sub(" ", XMLNOTE.sub(" ", raw))
    for chunk in SENT.split(raw):
        toks = WORD.findall(_norm_la(chunk))
        if len(toks) < 5:
            continue
        if not (LATIN_FN & set(toks)):
            stats["dropped"] += 1
            continue
        out.write(" ".join(toks) + "\n")
        stats["sent"] += 1; stats["tok"] += len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="assets_vec/la_corpus")
    ap.add_argument("--treebanks", nargs="*", default=["assets_sud/la-train.sud.conllu"])
    ap.add_argument("--out", default="assets_vec/la_corpus.txt")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    stats = dict(sent=0, tok=0, dropped=0)
    with open(a.out, "w", encoding="utf-8") as out:
        for name, kind in (("wikisource.xml.bz2", "wiki"), ("wikipedia.xml.bz2", "wiki")):
            p = d / name
            if not p.exists():
                print(f"  (missing {name})"); continue
            before = stats["tok"]
            for n, t in enumerate(wiki_texts(p)):
                emit(t, out, stats, kind)
                if n % 20000 == 0 and n:
                    print(f"  {name}: {n} pages, {stats['tok']:,} tokens", flush=True)
            print(f"  {name}: +{stats['tok']-before:,} tokens")
        for name, keep in (("latin_library.tar.gz", lambda n: n.endswith(".txt")),
                           ("perseus.tar.gz", lambda n: re.search(r"-lat\d*\.xml$", n) is not None)):
            p = d / name
            if not p.exists():
                print(f"  (missing {name})"); continue
            before = stats["tok"]
            kind = "plain" if name.startswith("latin_library") else "tei"
            for t in tar_texts(p, keep):
                emit(t, out, stats, kind)
            print(f"  {name}: +{stats['tok']-before:,} tokens")
        before = stats["tok"]
        for tb in a.treebanks:                      # our own gold text, folded the same way
            p = pathlib.Path(tb)
            if not p.exists():
                continue
            sent = []
            for line in p.open(encoding="utf-8"):
                fs = line.rstrip("\n").split("\t")
                if not line.strip():
                    if sent:
                        out.write(" ".join(sent) + "\n")
                        stats["sent"] += 1; stats["tok"] += len(sent); sent = []
                elif len(fs) > 3 and fs[0].isdigit():
                    w = _norm_la(fs[1])
                    if WORD.fullmatch(w):
                        sent.append(w)
        print(f"  treebanks: +{stats['tok']-before:,} tokens")
    print(f"\n{a.out}: {stats['sent']:,} sentences, {stats['tok']:,} tokens "
          f"({stats['dropped']:,} chunks dropped by the Latin-ness gate)")


main()
