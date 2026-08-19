#!/usr/bin/env python3
"""Extract running body text from a CBETA TEI P5 file.

WHAT IS DROPPED AND WHY. `<note>` is editorial apparatus, not scripture; `<rdg>` are variant
readings from other witnesses and would interleave alternative characters into the stream, so only
`<lem>` (the adopted reading) is kept. `<lb/>` is a line beacon, not a token boundary. `<head>` is
kept separately because a title is not running prose and would skew any frequency count.

CBETA TEI marks NO entities in the body -- every `<name>` in these files is header metadata
(publisher, contributors). Do not mistake it for NER supervision.
"""
import argparse, pathlib, re, xml.etree.ElementTree as ET
NS = "{http://www.tei-c.org/ns/1.0}"
DROP = {f"{NS}note", f"{NS}rdg", f"{NS}teiHeader", f"{NS}app"}

def text_of(el, drop=DROP):
    out = []
    def walk(e):
        if e.tag in drop:
            if e.tag == f"{NS}app":                      # keep the adopted reading only
                for lem in e.findall(f"{NS}lem"):
                    walk_children(lem)
            return
        walk_children(e)
    def walk_children(e):
        if e.text: out.append(e.text)
        for c in list(e):
            walk(c)
            if c.tail: out.append(c.tail)
    walk(el)
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml"); ap.add_argument("--out")
    ap.add_argument("--include-head", action="store_true")
    ap.add_argument("--div-type", default=None,
                    help="keep only <div type=...> subtrees, e.g. 'jing' for the scripture proper "
                         "(T08n0251 wraps two prefaces in <div type=xu> that are NOT the sutra)")
    a = ap.parse_args()
    root = ET.parse(a.xml).getroot()
    body = root.find(f".//{NS}body")
    roots = [body]
    if a.div_type:
        # CBETA mixes namespaces inside <body> -- <div> is in the CBETA namespace, not TEI -- so
        # match on LOCAL NAME. A namespaced lookup silently returns nothing here.
        roots = [d for d in body.iter()
                 if d.tag.split("}")[-1] == "div" and d.get("type") == a.div_type]
        if not roots:
            raise SystemExit(f"no <div type={a.div_type}> in {a.xml}")
    chunks = []
    for p in (e for rt in roots for e in rt.iter()):
        if p.tag == f"{NS}p" or (a.include_head and p.tag == f"{NS}head"):
            s = text_of(p)
            s = re.sub(r"\s+", "", s)
            if s: chunks.append(s)
    txt = "\n".join(chunks)
    if a.out:
        pathlib.Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}: {len(chunks)} paragraphs, {len(txt.replace(chr(10),'')):,} characters")
    else:
        print(txt)

if __name__ == "__main__":
    main()
