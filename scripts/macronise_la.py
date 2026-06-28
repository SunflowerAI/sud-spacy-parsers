#!/usr/bin/env python
"""Restore macrons (vowel-length marks) to the Latin SUD treebank.

We have gold tokens, lemmas and full morphology, so macronisation is far more
reliable than on raw text.  We use the gold-standard Alatius macroniser (Johan
Winge, ~98-99% vowel accuracy: RFTagger + a Morpheus-derived lexicon) running in
the `vedph2020/macronizer` Docker container, which exposes a Flask API:

    docker run -d --name macronizer --platform linux/amd64 \
        -p 51234:105 -e PYTHONUNBUFFERED=1 vedph2020/macronizer:0.1.3

The API keeps the orthography by default (no u->v / i->j), only adding macrons,
so the macronised forms differ from the originals *only* by combining/precomposed
macron marks.  We exploit that: macrons are transferred back onto the gold FORMs
by aligning on alphabetic characters alone (case and all non-letter characters of
the original are preserved verbatim).  This makes the transform:

  * 1:1 with the existing gold tokens (we feed the split sub-tokens, e.g.
    "nihil que", never the merged MWT surface), so every annotation column is
    untouched -- only FORM (and the regenerated `# text`) changes;
  * reversible -- stripping the macrons reproduces the original byte-for-byte
    (asserted per sentence; sentences that fail to align are left un-macronised).

Usage:
    .venv/bin/python scripts/macronise_la.py \
        assets_la/la_ittbproiel-sud-train.conllu \
        assets_la/la_ittbproiel-sud-train.macron.conllu
"""
import json
import sys
import time
import unicodedata
import urllib.request

API = "http://localhost:51234/macronize"
BATCH = 400  # sentences per request (newline-separated; API preserves newlines)

LONG = {"a": "ā", "e": "ē", "i": "ī", "o": "ō",
        "u": "ū", "y": "ȳ"}
LONG.update({k.upper(): v.upper() for k, v in list(LONG.items())})
LONG_SET = set(LONG.values())


def strip_macron(ch):
    """Return ch with any macron removed (handles precomposed and combining)."""
    n = unicodedata.normalize("NFD", ch)
    n = "".join(c for c in n if c != "̄")  # combining macron
    return unicodedata.normalize("NFC", n)


def macronize_batch(sentences):
    """POST newline-joined sentences; return one macronised string per sentence.

    The API preserves newlines, so line i corresponds to sentence i.  If the line
    count ever disagrees, fall back to one request per sentence."""
    payload = json.dumps({"text": "\n".join(sentences)}).encode("utf-8")
    req = urllib.request.Request(
        API, data=payload, headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["result"]
    lines = out.split("\n")
    if len(lines) == len(sentences):
        return lines
    return [macronize_batch([s])[0] for s in sentences]


def apply_macrons(forms, macronized_line):
    """Transfer macrons from the API output onto the gold FORMs.

    Alignment is on alphabetic characters only: the API adds macrons to vowels
    but otherwise leaves every letter (incl. Greek) and the letter order intact.
    For each original alphabetic char we keep its own case and add a macron iff
    the aligned output char carries one.  Non-letter chars are kept verbatim.

    Returns (new_forms, ok).  ok is False (and forms returned unchanged) if the
    character streams do not line up -- the caller then leaves the sentence as is.
    """
    out_letters = [unicodedata.normalize("NFC", c)
                   for c in macronized_line if c.isalpha()]
    in_positions = [(i, j) for i, f in enumerate(forms)
                    for j, c in enumerate(f) if c.isalpha()]
    if len(out_letters) != len(in_positions):
        return forms, False

    chars = [list(f) for f in forms]
    for (i, j), oc in zip(in_positions, out_letters):
        orig = chars[i][j]
        if strip_macron(oc).lower() != orig.lower():
            return forms, False  # letters disagree -> bail, leave untouched
        if oc in LONG_SET and orig in LONG:  # output is long; mark the original
            chars[i][j] = LONG[orig]
    return ["".join(c) for c in chars], True


def parse_blocks(text):
    return [b for b in text.split("\n\n") if b.strip()]


def is_token(line):
    return "\t" in line and line.split("\t", 1)[0].isdigit()


def rebuild_text(block_lines, form_by_id, range_forms):
    """Regenerate the `# text` comment from the (macronised) surface.

    MWT ranges contribute their merged surface (concatenation of the macronised
    sub-token forms); covered sub-tokens are skipped.  SpaceAfter=No is honoured."""
    covered = {}
    for (a, b), surf in range_forms.items():
        for k in range(a, b + 1):
            covered[k] = (a, b)
    pieces = []
    rows = [l for l in block_lines if "\t" in l]
    emitted_range = set()
    for line in rows:
        tid = line.split("\t", 1)[0]
        if "-" in tid or "." in tid:
            continue
        n = int(tid)
        misc = line.split("\t")[9]
        if n in covered:
            rng = covered[n]
            if rng in emitted_range:
                continue
            emitted_range.add(rng)
            surf = range_forms[rng]
            # SpaceAfter taken from the last sub-token of the range
            last = rng[1]
            last_misc = next(l.split("\t")[9] for l in rows
                             if l.split("\t", 1)[0] == str(last))
            nospace = "SpaceAfter=No" in last_misc
            pieces.append((surf, nospace))
        else:
            nospace = "SpaceAfter=No" in misc
            pieces.append((form_by_id[n], nospace))
    text = ""
    for k, (surf, nospace) in enumerate(pieces):
        text += surf
        if not nospace and k != len(pieces) - 1:
            text += " "
    return text


def process(in_path, out_path):
    blocks = parse_blocks(open(in_path, encoding="utf-8").read())

    # gather surface sentences (integer-id token forms only)
    sent_tokens = []  # list of (block_idx, [token ids], [forms])
    for bi, block in enumerate(blocks):
        ids, forms = [], []
        for line in block.splitlines():
            if is_token(line):
                cols = line.split("\t")
                ids.append(int(cols[0]))
                forms.append(cols[1])
        sent_tokens.append((bi, ids, forms))

    # macronise in batches
    new_forms = {}  # bi -> {id: macronised form}
    n_ok = n_fail = 0
    t0 = time.time()
    idx = 0
    while idx < len(sent_tokens):
        batch = sent_tokens[idx:idx + BATCH]
        lines = macronize_batch([" ".join(f) for _, _, f in batch])
        for (bi, ids, forms), line in zip(batch, lines):
            macd, ok = apply_macrons(forms, line)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                macd = forms
            new_forms[bi] = dict(zip(ids, macd))
        idx += BATCH
        sys.stderr.write(
            f"\r  {idx}/{len(sent_tokens)} sents  "
            f"(ok={n_ok} fail={n_fail})  {time.time()-t0:.0f}s")
        sys.stderr.flush()
    sys.stderr.write("\n")

    # rewrite blocks: FORM column + `# text`; everything else byte-identical
    out_blocks = []
    n_macron_tokens = 0
    for bi, block in enumerate(blocks):
        fb = new_forms.get(bi, {})
        lines = block.splitlines()
        # macronised MWT range surfaces = concat of macronised sub-token forms
        range_forms = {}
        for line in lines:
            if "\t" in line:
                tid = line.split("\t", 1)[0]
                if "-" in tid:
                    a, b = (int(x) for x in tid.split("-"))
                    range_forms[(a, b)] = "".join(fb.get(k, "") for k in range(a, b + 1))
        out_lines = []
        for line in lines:
            if not line.startswith("#") and "\t" in line:
                cols = line.split("\t")
                tid = cols[0]
                if tid.isdigit():
                    new = fb.get(int(tid), cols[1])
                    if new != cols[1]:
                        n_macron_tokens += 1
                    cols[1] = new
                elif "-" in tid and (int(tid.split("-")[0]), int(tid.split("-")[1])) in range_forms:
                    cols[1] = range_forms[(int(tid.split("-")[0]), int(tid.split("-")[1]))]
                out_lines.append("\t".join(cols))
            elif line.startswith("# text ="):
                form_by_id = fb if fb else {
                    int(l.split("\t")[0]): l.split("\t")[1]
                    for l in lines if is_token(l)}
                out_lines.append("# text = " + rebuild_text(lines, form_by_id, range_forms))
            else:
                out_lines.append(line)
        out_blocks.append("\n".join(out_lines))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out_blocks) + "\n\n")

    # reversibility check: stripping macrons must reproduce the original forms
    reverted = strip_macron_file(out_path)
    orig = [l for l in open(in_path, encoding="utf-8").read().splitlines()
            if is_token(l)]
    assert len(reverted) == len(orig), "token count changed!"
    bad = [(o, r) for o, r in zip(orig, reverted) if o != r]
    assert not bad, f"round-trip mismatch on {len(bad)} tokens, e.g. {bad[:3]}"

    print(f"{in_path} -> {out_path}")
    print(f"  sentences: {n_ok} macronised, {n_fail} left as-is "
          f"({100*n_ok/(n_ok+n_fail):.1f}% ok)")
    print(f"  tokens with macrons added: {n_macron_tokens}")
    print(f"  round-trip (strip macrons == original): OK")


def strip_macron_file(path):
    """Return the list of macron-stripped token lines (for the round-trip test)."""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if is_token(line):
            cols = line.split("\t")
            cols[1] = "".join(strip_macron(c) for c in cols[1])
            out.append("\t".join(cols))
    return out


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
