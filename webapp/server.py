#!/usr/bin/env python3
"""Tiny dependency-free web server for testing the SUD spaCy parsers.

Run with the project venv (it has spaCy + spacy_pkuseg):

    .venv/bin/python webapp/server.py            # serves on http://127.0.0.1:8000

Korean needs mecab-ko; MECAB_PATH is set below before spaCy is imported so the
Korean pipeline loads without the caller having to export it.

Loads the released model wheels by package name (install them first, e.g.
`pip install <release-url>/zh_sud_gsdboth-0.1.0-py3-none-any.whl`), so the
browser can send plain text — each model is matched to its treebank tokenisation:

    en  -> en_sud_ewt         (default rules, EWT)
    zh  -> zh_sud_gsdboth     (pkuseg, GSDSimp + OpenCC traditional)
    ko  -> ko_sud_gsd         (mecab morphemes; needs mecab-ko)
    id  -> id_sud_gsd         (rule tokeniser, enclitics merged)
    fa  -> fa_sud_perdt       (rule tokeniser, PerDT)
    ar  -> ar_sud_padt        (rule tokeniser, PADT)
    la  -> la_sud_ittbproiel  (rule tokeniser, ITTB+PROIEL)
    sa  -> sa_sud_sandhi_csl  (accepts sandhied CSL text, de-sandhied to clean wordforms; case-based)
    lzh -> lzh_sud_kyoto      (custom one-char tokeniser bundled in the wheel, Kyoto)
    ja  -> ja_sud_gsd         (SudachiPy, GSD)

Only languages whose wheel is installed are advertised. These models predict SUD
relations with `udep` adpositional/case dependents disambiguated into `comp:obl`
(complement) vs `mod` (modifier).
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Must be set before spaCy / the Korean tokenizer is imported (see CLAUDE.md).
os.environ.setdefault("MECAB_PATH", "/opt/homebrew/lib/libmecab.dylib")

ROOT = Path(__file__).resolve().parent           # webapp/
PROJECT = ROOT.parent                            # repo root
INDEX = ROOT / "index.html"

# lang -> installed model wheel package name
MODELS = {
    "en": "en_sud_ewt",
    "zh": "zh_sud_gsdboth",
    "ko": "ko_sud_gsd",
    "id": "id_sud_gsd",
    "fa": "fa_sud_perdt",
    "ar": "ar_sud_padt",
    "la": "la_sud_ittbproiel",
    "sa": "sa_sud_sandhi_csl",
    "lzh": "lzh_sud_kyoto",
    "ja": "ja_sud_gsd",
}

LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ko": "Korean", "id": "Indonesian",
    "fa": "Persian", "ar": "Arabic", "la": "Latin", "sa": "Sanskrit",
    "lzh": "Classical Chinese", "ja": "Japanese",
}

# right-to-left scripts — the viewer lays these out right-to-left
RTL = {"fa", "ar"}

EXAMPLES = {
    "en": "I gave the book to John in the morning.",
    "zh": "我在北京学习中文。",
    "ko": "나는 아침에 도서관에서 책을 읽었다.",
    "id": "Saya membaca buku itu di perpustakaan pada pagi hari.",
    "fa": "وی در اواخر عمر به علت کار و مطالعهٔ زیاد نابینا شد.",
    "ar": "وفيما يلي اسماء الوزراء الجدد",
    "la": "perfectio autem operationis dependet ex quatuor.",
    "sa": "hāstidantaṃ badhnāti lomāni jatunā saṃdihya jātarūpeṇ' âpidhāpya",
    "lzh": "學而時習之",
    "ja": "また行きたい、そんな気持ちにさせてくれるお店です。",
}

_cache = {}  # (lang, variant) -> loaded nlp


def available_models():
    """Only advertise languages whose model wheel is installed (importable)."""
    import importlib.util
    out = {}
    for lang, pkg in MODELS.items():
        if importlib.util.find_spec(pkg) is not None:
            out[lang] = {
                "name": LANG_NAMES[lang],
                "package": pkg,
                "example": EXAMPLES.get(lang, ""),
                "rtl": lang in RTL,
            }
    return out


def get_nlp(lang):
    if lang not in _cache:
        import spacy
        pkg = MODELS[lang]
        sys.stderr.write(f"[load] {lang} <- {pkg}\n")
        _cache[lang] = spacy.load(pkg)
    return _cache[lang]


def parse_text(lang, text):
    nlp = get_nlp(lang)
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        toks = list(sent)
        base = toks[0].i
        sentences.append([
            {
                "id": t.i - base,
                "text": t.text,
                "pos": t.tag_ or t.pos_,   # tagger fills tag_ (XPOS); pos_ is empty
                "dep": t.dep_,
                "head": t.head.i - base,   # self-loop on the root
            }
            for t in toks
        ])
    return {"lang": lang, "package": MODELS[lang], "sentences": sentences}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # dev tool: never let the browser cache index.html / API responses, so edits show on reload
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, format, *args):  # quieter logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"index.html not found")
        elif self.path == "/api/models":
            self._json(200, available_models())
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/api/parse":
            self._send(404, b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            lang = req.get("lang")
            text = (req.get("text") or "").strip()
            if lang not in MODELS:
                self._json(400, {"error": f"unknown language {lang}"})
                return
            if not text:
                self._json(400, {"error": "empty text"})
                return
            self._json(200, parse_text(lang, text))
        except Exception as exc:  # surface parse/load errors to the browser
            sys.stderr.write(f"[error] {exc}\n")
            self._json(500, {"error": str(exc)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"SUD parser tester on http://127.0.0.1:{port}  (Ctrl-C to stop)")
    print("Available:", ", ".join(f"{l} ({m['package']})"
                                   for l, m in available_models().items()))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
