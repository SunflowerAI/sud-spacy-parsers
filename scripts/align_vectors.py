#!/usr/bin/env python3
"""Build the cross-lingually aligned side-asset vector tables, one per language.

WHAT THIS IS FOR. Aligning parse trees across parallel translations, not feeding the parser --
static vectors as a parser input were measured and do not pay (`NEGATIVE-RESULTS.md`, md vectors and
the kanripo arm). The operation here is RETRIEVAL: given a node in one language's tree, find the
node in another language's tree whose vector is nearest. Everything below is chosen for that.

THE THREE DECISIONS, and the measurements behind them.

1. **128 dimensions.** Retrieval P@1 into English, under a shared truncation, saturates at 128:
   16d 14.0 %, 32d 44.8, 48d 61.3, 64d 67.8, 96d 73.6, 128d 77.4, 200d 78.9, 300d 77.5 (en<-id,
   6 000 Procrustes anchors, 1 000 held-out pairs, 50 000 candidates). At every byte budget from
   5 MB to 40 MB per language, 128d maximises coverage x P@1 -- fewer dimensions buy vocabulary that
   is nearly worthless, because token coverage is already 89.6 % at 20 000 keys and only 95.8 % at
   100 000.

2. **ONE shared basis, computed on the joint space and applied to every language.** A per-language
   PCA rotates each language differently and destroys the alignment, which is the entire point of
   the asset. The basis and the joint mean are byte-identical in all thirteen assets; only
   `rotation` differs.

3. **Length-normalised rows**, so a cosine is a dot product and a caller needs no preprocessing.

THE HUB is English (`wiki.en.align.vec`). Seven languages are published already aligned to it and
need no rotation at all; the rest are fitted by orthogonal Procrustes on anchors, which is a
ROTATION -- it cannot distort within-language geometry, only place it.

ANCHOR ROUTES, worst-supported languages last:
    pre       en zh ko id fa ar ta   already in the hub space
    dict      ja                     MUSE en-ja translation pairs
    gloss     sa la te lzh yue       a dictionary's ENGLISH GLOSS BAG, averaged into one target
                                     vector (Apte for sa, Wiktionary for the rest)

⚠ The gloss route is weaker than a translation dictionary and the report says so per language:
`--stage fit` prints a held-out score for every language and no asset ships without that number in
its meta. Three of the sources are HOMEGROWN because no usable published vectors exist: sa (floret
over DCS lemmas), lzh (floret over kanripo) and la (floret over 112.8M tokens of Wikisource, the
Latin Library, Perseus and Wikipedia -- which took cc.la's 52.0 % treebank coverage at @1 27.5 % to
100 % at 37.7 %).

LICENCES DIVERGE FROM THE WHEELS, which is exactly why these are side assets and not bundled:
fastText vectors are CC BY-SA 3.0 and could not go inside the CC BY-NC-SA la/ta/te wheels. Nothing
from a dictionary is redistributed -- Apte and Wiktionary are used only to FIT a rotation.
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from aligned_vectors import KEY_NORM      # ONE definition of each key fold, shared with the loader

WORK = pathlib.Path("assets_vec/work")
SRC = pathlib.Path("assets_vec/src")
DICT = pathlib.Path("assets_vec/dict")

# treebank files whose FORM/LEMMA inventory each asset must cover
TREEBANK = {
    "en": ["assets_sud/en_ewt-sud-train.sud.conllu", "assets_sud/en_gum-train.sud.conllu"],
    "zh": ["assets_sud/zh-train.sud.conllu"], "yue": ["assets_sud/yue-train.sud.conllu"],
    "lzh": ["assets_sud/lzh-trad-train.sud.conllu"], "ja": ["assets_sud/ja-train.sud.conllu"],
    "ko": ["assets_sud/ko-train.sud.conllu"], "id": ["assets_sud/id-train.sud.conllu"],
    "fa": ["assets_sud/fa-train.sud.conllu"], "ar": ["assets_sud/ar-train.sud.conllu"],
    "la": ["assets_sud/la-train.sud.conllu"], "sa": ["assets_sud/sa-train.sud.conllu"],
    "ta": ["assets_sud/ta-train.sud.conllu"], "te": ["assets_sud/te-train.sud.conllu"],
}

SOURCES = {
    "en":  dict(vec=SRC / "align.en.vec", route="hub",      key="form",  src="fastText aligned (wiki.en.align)"),
    "zh":  dict(vec=SRC / "align.zh.vec", route="pre",      key="form",  src="fastText aligned (wiki.zh.align)"),
    "ko":  dict(vec=SRC / "align.ko.vec", route="pre",      key="form",  src="fastText aligned (wiki.ko.align)"),
    "id":  dict(vec=SRC / "align.id.vec", route="pre",      key="form",  src="fastText aligned (wiki.id.align)"),
    "fa":  dict(vec=SRC / "align.fa.vec", route="pre",      key="form",  src="fastText aligned (wiki.fa.align)"),
    "ar":  dict(vec=SRC / "align.ar.vec", route="pre",      key="form",  src="fastText aligned (wiki.ar.align)"),
    "ta":  dict(vec=SRC / "align.ta.vec", route="pre",      key="form",  src="fastText aligned (wiki.ta.align)"),
    "ja":  dict(vec=SRC / "cc.ja.vec",    route="dict",     key="form",  src="fastText CC (cc.ja.300)",
                pairs=DICT / "en-ja.txt"),
    # Homegrown, and the reason is coverage: cc.la reaches 52.0 % of our treebank types because it
    # is Common Crawl Latin and spells with v and j, while ITTB and PROIEL are u-dominant. A
    # 112.8M-token corpus of Wikisource + Wikipedia + the Latin Library + Perseus, folded onto one
    # orthography, reaches 99.2 %. `key_norm` is what makes the two vocabularies meet, and it must
    # be applied to the DICTIONARY headwords too -- Wiktionary spells `vita`, this table keys `uita`.
    "la":  dict(floret="vectors_la_hg_floret.bin", route="gloss", key="form", key_norm="la",
                src="floret over Wikisource+Wikipedia+Latin Library+Perseus (112.8M tokens)",
                gloss=DICT / "la-en.json", keysrc="assets_vec/la_corpus.txt"),
    "te":  dict(vec=SRC / "cc.te.vec",    route="gloss",    key="form",  src="fastText CC (cc.te.300)",
                gloss=DICT / "te-en.json"),
    "sa":  dict(floret="vectors_sa_lemma_floret.bin", route="gloss", key="lemma",
                src="floret over DCS lemmas (5.69M tokens)", gloss=DICT / "sa-en.json",
                keysrc="assets_vec/dcs_lemma.txt"),
    # ⚠ NOT the IDS-expanded table the parser experiments used. That one exists to make
    # GRAPHICALLY similar characters similar, which is the opposite of what alignment wants.
    #
    # Both of these were built on the identity route first -- anchor on the strings they share with
    # already-aligned zh -- and it is the WEAKER route for both, scored on the same held-out
    # Wiktionary glosses: lzh @1 1.3 % against 6.4 %, yue 7.1 % against 10.0 %. Sharing a graph is
    # not sharing a distribution. A single character is a WORD in Literary Chinese and a bound
    # morpheme in modern Chinese, so zh's vector for it comes from the rare contexts where it stands
    # alone; even the 500 most frequent lzh characters self-retrieve their own zh row only 10.0 % of
    # the time (cos 0.34). The identity rotations are kept as W_{lzh,yue}_identity.npy.
    "lzh": dict(floret="vectors_lzh_plain_floret.bin", route="gloss", key="form",
                src="floret over kanripo (plain, leak-free, 42M tokens)",
                gloss=DICT / "zhwikt-en.json", keysrc="corpus_lzh_kanripo_leakfree.txt"),
    "yue": dict(vec=SRC / "wiki.yue.vec", route="gloss", key="form",
                src="fastText wiki (wiki.zh_yue)", gloss=DICT / "zhwikt-en.json"),
}
HUB = "en"

# Carried into every asset's meta. Eleven are fastText derivatives and clean; sa and lzh derive from
# corpora whose upstream declares NO licence, and that is stated rather than smoothed over.
FT_LICENCE = "CC BY-SA 3.0 (fastText, Facebook AI Research); redistributed CC BY-SA 4.0 under 3.0 s4(b)"
LICENCE = {l: FT_LICENCE for l in SOURCES}
LICENCE["sa"] = ("derived from the Digital Corpus of Sanskrit (Oliver Hellwig). UPSTREAM DECLARES NO "
                 "LICENCE -- github.com/OliverHellwig/sanskrit carries no LICENSE file. Provenance "
                 "unresolved; released as a derived model, not a redistribution of the corpus.")
LICENCE["lzh"] = ("derived from the Kanseki Repository (kanripo). UPSTREAM DECLARES NO LICENCE -- the "
                  "kanripo/KR* repositories carry no LICENSE file and the text headers no rights "
                  "metadata. Provenance unresolved; released as a derived model, not a "
                  "redistribution of the corpus.")
ATTRIBUTION = ("Anchors were used only to FIT a rotation; no dictionary content is redistributed. "
               "Anchor sources: MUSE bilingual dictionaries (ja); Apte via CDSL (sa); "
               "Wiktionary/kaikki.org, CC BY-SA (la, te, lzh, yue).")


# ---------------------------------------------------------------- loading

def read_vec(path, limit=None):
    """Read a word2vec-style .vec. Returns (keys, X). Skips malformed rows rather than dying:
    the CC files contain a handful of keys with embedded spaces."""
    keys, rows = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        dim = int(header[1]) if len(header) >= 2 else None
        for line in f:
            p = line.rstrip("\n").rstrip().split(" ")
            if dim and len(p) != dim + 1:
                continue
            keys.append(p[0]); rows.append(p[1:])
            if limit and len(keys) >= limit:
                break
    X = np.asarray(rows, dtype=np.float32)
    return keys, X


def is_lowercased(keys, n=20000):
    """The published aligned vectors are lowercased throughout; the CC ones are not. This decides
    whether a caller must case-fold before lookup, and it is recorded in each asset rather than
    left for the caller to guess -- en treebank coverage is 53.9 % if you get it wrong and 84.8 %
    if you get it right."""
    head = keys[:n]
    return not any(k != k.lower() for k in head)


def treebank_types(lang, col):
    """FORM (col 1) or LEMMA (col 2) inventory of the treebank, so the asset covers what the
    released parser actually emits."""
    i = 1 if col == "form" else 2
    out = set()
    for f in TREEBANK.get(lang, []):
        p = pathlib.Path(f)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8", errors="replace"):
            fs = line.rstrip("\n").split("\t")
            if len(fs) > 3 and fs[0].isdigit() and fs[i] not in ("", "_"):
                out.add(fs[i])
    return out


def fold_for(s, keys):
    """The function mapping an arbitrary string onto this source's key convention. A source either
    declares an explicit fold (`key_norm`) or is judged lowercased-or-not from its own keys."""
    kn = s.get("key_norm")
    if kn:
        return KEY_NORM[kn]
    return str.lower if is_lowercased(keys) else (lambda w: w)


def load_source(lang, limit):
    """Every language ends up as (keys, X) in ITS OWN space, before any rotation."""
    s = SOURCES[lang]
    if "floret" in s:
        import floret
        m = floret.load_model(s["floret"])
        # floret has no vocabulary of its own to enumerate -- every string composes. Take the keys
        # from the corpus it was trained on, most frequent first, and add the treebank's own
        # inventory. Prune by DIMENSION, never by VOCABULARY (build_lzh_vectors.py records why).
        import collections
        c = collections.Counter()
        for line in open(s["keysrc"], encoding="utf-8"):
            c.update(line.split())
        keys = [w for w, _ in c.most_common(limit)]
        f = fold_for(s, keys)
        extra = sorted({f(w) for w in treebank_types(lang, s["key"])} - set(keys))
        keys += extra
        X = np.vstack([m.get_word_vector(w) for w in keys]).astype(np.float32)
        print(f"  {lang}: floret composed {len(keys)} rows ({len(extra)} from the treebank)")
        return keys, X
    keys, X = read_vec(s["vec"], limit)
    print(f"  {lang}: {len(keys)} keys x {X.shape[1]}d from {s['vec'].name}")
    return keys, X


def normalise(X):
    """MUSE's preprocessing: unit-length, mean-centre, unit-length again. Applied identically to
    every space so that Procrustes sees comparable geometry."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    X = X - X.mean(0)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def procrustes(Xs, Xt):
    """Orthogonal map W minimising ||Xs W - Xt||. Rotation only: it places a space, it cannot
    distort it."""
    U, _, Vt = np.linalg.svd(Xs.T @ Xt)
    return (U @ Vt).astype(np.float32)


# ---------------------------------------------------------------- anchors

def anchors_dict(lang, keys, hub_index, path):
    """MUSE translation pairs: 'english<TAB>foreign' (or space-separated). Returns index pairs."""
    src_index = {w: i for i, w in enumerate(keys)}
    pairs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) != 2:
            continue
        en, fw = p
        if en in hub_index and fw in src_index:
            pairs.append((src_index[fw], hub_index[en]))
    return pairs


def anchors_gloss(lang, keys, hub_index, hub_X, path, fold=None):
    """A definition dictionary gives PROSE, not a translation. Average the English vectors of a
    headword's gloss words (count-weighted) into one target, and keep the headwords whose bag has
    at least two English words that the hub actually holds -- a single-word bag is usually a
    metalanguage leak ('genitive'), not a gloss."""
    import collections
    bags = json.load(open(path, encoding="utf-8"))
    fold = fold or (lambda w: w)
    src_index = {w: i for i, w in enumerate(keys)}
    # The fold collapses spellings -- Wiktionary has `vita` and `vīta`, this table keys `uita` --
    # so MERGE the bags of every headword landing on one row rather than letting the first win.
    merged = collections.defaultdict(collections.Counter)
    for w, bag in bags.items():
        i = src_index.get(fold(w))
        if i is not None:
            merged[i].update(bag)
    rows, targets, kept_bags = [], [], []
    for i, bag in merged.items():
        vs, ws, present = [], [], []
        for g, n in bag.items():
            j = hub_index.get(g)
            if j is not None:
                vs.append(hub_X[j]); ws.append(float(n)); present.append(g)
        if len(vs) < 2:
            continue
        t = np.average(np.vstack(vs), axis=0, weights=ws)
        n = np.linalg.norm(t)
        if n < 1e-6:
            continue
        rows.append(i); targets.append(t / n); kept_bags.append(set(present))
    T = np.vstack(targets).astype(np.float32) if targets else np.zeros((0, hub_X.shape[1]), np.float32)
    return rows, T, kept_bags


def gloss_hit_rate(Q, cand, cand_keys, bags, k=5):
    """A definition dictionary has no single right answer, so exact-match P@1 cannot score it.
    Score instead the way a user would read the output: is the nearest English word ONE OF the
    headword's own gloss words? @1 and @k."""
    S = Q @ cand.T
    top = np.argpartition(-S, k, axis=1)[:, :k]
    hit1 = hitk = 0
    for r, bag in enumerate(bags):
        order = top[r][np.argsort(-S[r, top[r]])]
        words = [cand_keys[i] for i in order]
        hit1 += words[0] in bag
        hitk += any(w in bag for w in words)
    n = max(len(bags), 1)
    return hit1 / n, hitk / n


def anchors_identity(keys, pivot_keys, pivot_X, min_len=1):
    """Strings shared with an already-aligned pivot. For lzh and yue the pivot is zh, where a
    shared string is the SAME GRAPH -- a real anchor, though the senses have drifted."""
    piv = {w: i for i, w in enumerate(pivot_keys)}
    rows, targets = [], []
    for i, w in enumerate(keys):
        j = piv.get(w)
        if j is not None and len(w) >= min_len:
            rows.append(i); targets.append(pivot_X[j])
    return rows, (np.vstack(targets).astype(np.float32) if targets else np.zeros((0, pivot_X.shape[1]), np.float32))


def evaluate(Q, T, gold, k=10):
    """P@1 by nearest neighbour and by CSLS (which corrects the hubness that low-dimensional
    retrieval invites). Q and T must already be unit-length."""
    samp = T[np.random.default_rng(0).choice(len(T), min(5000, len(T)), replace=False)]
    rT = np.empty(len(T), dtype=np.float32)
    for i in range(0, len(T), 5000):
        blk = T[i:i + 5000] @ samp.T
        rT[i:i + 5000] = np.sort(blk, axis=1)[:, -k:].mean(1)
    S = Q @ T.T
    return float((S.argmax(1) == gold).mean()), float(((2 * S - rT[None, :]).argmax(1) == gold).mean())


# ---------------------------------------------------------------- stages

def stage_fit(a):
    """Place every language in the hub space and cache the result. Writes, per language,
    aligned_<lang>.npy + keys_<lang>.txt, and one row of fit_report.json.

    EVERY route is scored the same way and the score is a SET-MEMBERSHIP hit, not an exact match.
    Both parts matter. The MUSE dictionaries are many-to-many -- `and` alone has three Chinese
    translations -- so scoring against a single gold word undercounts a correct retrieval as a
    miss (zh reads 26.4 % that way and 44.2 % when the gold set is honoured). And a definition
    dictionary has no single right answer at all. Held-out splits are taken over SOURCE WORDS, not
    over pairs, or a word's other translations leak from train into test.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    print("hub:")
    hub_keys, hub_raw = load_source(HUB, a.limit)
    hub_X = normalise(hub_raw)
    hub_index = {w: i for i, w in enumerate(hub_keys)}
    np.save(WORK / "aligned_en.npy", hub_X)
    (WORK / "keys_en.txt").write_text("\n".join(hub_keys), encoding="utf-8")
    report = {}
    if (WORK / "fit_report.json").exists():
        report = json.load(open(WORK / "fit_report.json"))

    rng = np.random.default_rng(0)
    for lang in a.langs:
        s = SOURCES[lang]
        print(f"\n{lang} ({s['route']}):")
        if lang == HUB:
            keys, X = hub_keys, hub_X
            W = np.eye(X.shape[1], dtype=np.float32)
            fit_pairs, eval_bags = [], {}
        else:
            keys, raw = load_source(lang, a.limit)
            X = normalise(raw)
            fit_pairs, eval_bags = build_anchors(lang, s, keys, hub_keys, hub_index, hub_X,
                                                 fold_for(s, keys))
            print(f"  anchors: {len(fit_pairs)} pairs over {len(set(r for r, _ in fit_pairs))} source words")
            # Fit on FREQUENT source words only. A rotation fitted on rare words is fitted on
            # noise: in a 5.7M-token corpus a lemma seen three times has a vector that is mostly
            # its subwords. Everything is still PLACED by the resulting rotation -- the restriction
            # is on what gets a vote in choosing it. lzh is exempt because its table is written in
            # codepoint order, so a row index there is not a frequency rank.
            fit_ok = (lambda r: True) if not s.get("freq_ordered", True) else (lambda r: r < a.fit_max_rank)
            rows = sorted({r for r, _ in fit_pairs})
            rng.shuffle(rows)
            ntest = min(1000, len(rows) // 5)
            test_rows, train_rows = set(rows[:ntest]), set(rows[ntest:])
            if s["route"] == "pre":
                W = np.eye(X.shape[1], dtype=np.float32)
            else:
                tr = [(r, t) for r, t in fit_pairs if r in train_rows and fit_ok(r)]
                if len(tr) < 200:
                    print("  !! too few anchors to fit a rotation -- leaving this space UNPLACED")
                    W = np.eye(X.shape[1], dtype=np.float32)
                else:
                    print(f"  fitting on {len(tr)} pairs over {len(set(r for r, _ in tr))} words"
                          f" (rank < {a.fit_max_rank})" if s.get("freq_ordered", True) else
                          f"  fitting on {len(tr)} pairs")
                    W = procrustes(X[[r for r, _ in tr]], np.vstack([t for _, t in tr]))
            X = X @ W

        # ---- score held-out source words by whether the nearest English word is a gold one
        h1 = h5 = None
        nev = 0
        if lang != HUB and eval_bags:
            cand = hub_X[: a.eval_cand]
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
            cand_keys = hub_keys[: a.eval_cand]
            in_cand = set(cand_keys)
            ev = [(r, eval_bags[r] & in_cand) for r in sorted(test_rows) if eval_bags.get(r)]
            ev = [(r, b) for r, b in ev if b]
            if len(ev) >= 50:
                Q = X[[r for r, _ in ev]]
                Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
                h1, h5 = hit_rate(Q, cand, cand_keys, [b for _, b in ev])
                nev = len(ev)
                print(f"  held-out hit into en top-{a.eval_cand}: @1 {h1*100:.1f}%  @5 {h5*100:.1f}%  (n={nev})")
            else:
                print(f"  (only {len(ev)} scorable held-out words -- not measured)")

        np.save(WORK / f"aligned_{lang}.npy", X.astype(np.float32))
        (WORK / f"keys_{lang}.txt").write_text("\n".join(keys), encoding="utf-8")
        np.save(WORK / f"W_{lang}.npy", W)
        low = is_lowercased(keys)
        f = fold_for(s, keys)
        tb = treebank_types(lang, s["key"])
        ks = set(keys)
        cov = sum(1 for w in tb if f(w) in ks) / max(len(tb), 1)
        report[lang] = dict(route=s["route"], source=s["src"], key_attr=s["key"],
                            source_vocab=len(keys), anchor_words=len(set(r for r, _ in fit_pairs)),
                            hit_at_1=h1, hit_at_5=h5, eval_n=nev,
                            treebank_type_coverage=cov, lowercased=bool(low),
                            key_norm=s.get("key_norm"))
        print(f"  treebank {s['key']} types covered: {cov*100:.1f}%"
              f"{' (case-folded)' if low else ''}")
    json.dump(report, open(WORK / "fit_report.json", "w"), indent=2)
    print(f"\nwrote {WORK}/fit_report.json")


def build_anchors(lang, s, keys, hub_keys, hub_index, hub_X, fold=None):
    """-> (fit_pairs, eval_bags). fit_pairs are (source_row, target_vector); eval_bags maps a
    source row to the SET of acceptable English words for it."""
    import collections
    fold = fold or (lambda w: w)
    src_index = {w: i for i, w in enumerate(keys)}
    fit_pairs = []
    bags = collections.defaultdict(set)
    if s["route"] in ("pre", "dict"):
        path = s.get("pairs") or (DICT / f"en-{lang}.txt")
        if not pathlib.Path(path).exists():
            return [], {}
        for line in open(path, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) != 2:
                continue
            en, fw = p
            i, j = src_index.get(fold(fw)), hub_index.get(en)
            if i is not None and j is not None:
                fit_pairs.append((i, hub_X[j])); bags[i].add(en)
    elif s["route"] == "gloss":
        rows, targets, kept = anchors_gloss(lang, keys, hub_index, hub_X, s["gloss"], fold)
        for n, r in enumerate(rows):
            fit_pairs.append((r, targets[n])); bags[r] |= kept[n]
    elif s["route"] == "identity":
        pk = (WORK / f"keys_{s['pivot']}.txt").read_text(encoding="utf-8").split("\n")
        pX = np.load(WORK / f"aligned_{s['pivot']}.npy")
        rows, targets = anchors_identity(keys, pk, pX)
        for n, r in enumerate(rows):
            fit_pairs.append((r, targets[n]))
        # scoring borrows the PIVOT's dictionary: a shared string's English translations are the
        # only external check available for a language with no dictionary of its own
        d = DICT / f"en-{s['pivot']}.txt"
        if d.exists():
            piv_en = collections.defaultdict(set)
            for line in open(d, encoding="utf-8", errors="replace"):
                p = line.split()
                if len(p) == 2:
                    piv_en[p[1]].add(p[0])
            for r in {r for r, _ in fit_pairs}:
                w = keys[r]
                if w in piv_en:
                    bags[r] |= piv_en[w]
    return fit_pairs, dict(bags)


def hit_rate(Q, cand, cand_keys, bags, k=5):
    """Is the nearest English word one of the gold set? @1 and @k."""
    S = Q @ cand.T
    kk = min(k, S.shape[1] - 1)
    top = np.argpartition(-S, kk, axis=1)[:, :kk]
    h1 = hk = 0
    for r, bag in enumerate(bags):
        order = top[r][np.argsort(-S[r, top[r]])]
        words = [cand_keys[i] for i in order]
        h1 += words[0] in bag
        hk += any(w in bag for w in words)
    n = max(len(bags), 1)
    return h1 / n, hk / n


def stage_basis(a):
    """ONE basis for all languages. Computed on a balanced sample of the ALIGNED spaces, so no
    single large vocabulary decides the axes."""
    parts = []
    rng = np.random.default_rng(0)
    for lang in a.langs:
        X = np.load(WORK / f"aligned_{lang}.npy", mmap_mode="r")
        n = min(a.basis_sample, len(X))
        parts.append(np.asarray(X[rng.choice(len(X), n, replace=False)]))
        print(f"  {lang}: {n} rows")
    J = np.vstack(parts)
    mu = J.mean(0).astype(np.float32)
    _, s, Vt = np.linalg.svd(J - mu, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    P = Vt[: a.dims].T.astype(np.float32)
    np.savez(WORK / "basis.npz", basis=P, mean=mu,
             variance=np.float32(var[: a.dims].sum()), langs="\n".join(a.langs))
    print(f"joint basis: {J.shape[0]} rows -> {P.shape}, {var[:a.dims].sum()*100:.1f}% of variance")


def stage_emit(a):
    b = np.load(WORK / "basis.npz")
    P, mu = b["basis"], b["mean"]
    report = json.load(open(WORK / "fit_report.json"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for lang in a.langs:
        s = SOURCES[lang]
        keys = (WORK / f"keys_{lang}.txt").read_text(encoding="utf-8").split("\n")
        X = np.load(WORK / f"aligned_{lang}.npy", mmap_mode="r")
        low = report[lang].get("lowercased", False)
        f = fold_for(s, keys)
        tb = {f(w) for w in treebank_types(lang, s["key"])}
        # frequency-ordered head, plus every treebank type further down the list: the parser's own
        # inventory must not be pruned away by a frequency cut taken over a different corpus
        head = min(a.keys, len(keys))
        idx = list(range(head)) + [i for i in range(head, len(keys)) if keys[i] in tb]
        Y = (np.asarray(X[idx]) - mu) @ P
        Y = (Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        meta = dict(report[lang])
        meta.update(lang=lang, dims=int(P.shape[1]), rows=len(idx), hub=HUB,
                    lookup=(f"apply the '{s['key_norm']}' key fold before lookup "
                            f"(AlignedVectors does this for you)" if s.get("key_norm") else
                            "lowercase the token before lookup" if low else
                            "look the token up as written"),
                    normalised="unit length; cosine == dot product",
                    projection="v_shared = (normalise(v_source) @ rotation - mean) @ basis",
                    basis="shared across every language in this release",
                    licence=LICENCE[lang], attribution=ATTRIBUTION)
        np.savez_compressed(
            out / f"sud_vec_{lang}_{a.dims}d.npz",
            vectors=Y, keys="\n".join(keys[i] for i in idx).encode("utf-8"),
            basis=P, mean=mu, rotation=np.load(WORK / f"W_{lang}.npy"),
            meta=json.dumps(meta, ensure_ascii=False))
        f = out / f"sud_vec_{lang}_{a.dims}d.npz"
        print(f"  {f.name}: {len(idx)} keys x {P.shape[1]}d, {f.stat().st_size/1e6:.1f} MB")


def apply_sources(path):
    """Replace the hand-written SOURCES/TREEBANK with a GENERATED manifest (v3 and later).

    Thirteen languages fit in a literal; thirty-eight do not stay correct in one. The v3 arm derives
    its corpus map from the same manifest the parser was trained against, so the two cannot drift --
    the failure this repo has paid for four times is a default that names a superseded corpus, which
    loads, converts and trains exactly like a current one.

    Without `--sources` nothing here runs and v1's thirteen assets rebuild byte-for-byte as before.
    """
    global SOURCES, TREEBANK, LICENCE
    m = json.load(open(path, encoding="utf-8"))
    SOURCES, TREEBANK = {}, {}
    for lc, s in m["languages"].items():
        d = dict(route=s["route"], key=s["key"], src=s["src"])
        if s.get("vec"):
            d["vec"] = pathlib.Path(s["vec"])
        if s.get("key_norm"):
            d["key_norm"] = s["key_norm"]
        SOURCES[lc] = d
        TREEBANK[lc] = list(s.get("treebank") or [])
    LICENCE = {l: FT_LICENCE for l in SOURCES}      # every v3 source is a fastText derivative
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "basis", "emit"], required=True)
    ap.add_argument("--sources", help="generated source manifest (assets_vec/sources_v3.json). "
                                      "Omit to rebuild v1's thirteen from the literals above.")
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=200000, help="source vocabulary read per language")
    ap.add_argument("--keys", type=int, default=50000, help="frequency-ordered keys per asset")
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--basis-sample", type=int, default=20000)
    ap.add_argument("--fit-max-rank", type=int, default=20000,
                    help="only source words this frequent get a vote in fitting the rotation")
    ap.add_argument("--eval-cand", type=int, default=50000)
    ap.add_argument("--out", default="release_vectors")
    # ⚠ A SECOND GENERATION MUST NOT SHARE A WORK DIRECTORY WITH THE FIRST. `basis.npz` and
    # `fit_report.json` are per-GENERATION, not per-language: emit reads whichever basis is sitting
    # there and stamps it into every asset, so a v3 run into v1's directory would silently rebuild
    # thirteen v1 assets against a 32-language basis -- and they would load, retrieve and look
    # entirely normal, because a basis is only wrong relative to the rows it was fitted with.
    ap.add_argument("--work", default=None,
                    help="working directory (default assets_vec/work). Pass a NEW one for a new "
                         "generation; sharing one silently mixes bases across generations.")
    a = ap.parse_args()
    if a.work:
        global WORK
        WORK = pathlib.Path(a.work)
    if a.sources and not a.work:
        sys.exit("--sources names a new generation; give it its own --work directory (see --help)")
    meta = apply_sources(a.sources)["meta"] if a.sources else None
    if a.langs is None:
        a.langs = list(SOURCES)
    unknown = [l for l in a.langs if l not in SOURCES]
    if unknown:
        sys.exit(f"no source declared for {unknown}")

    # ⚠ THE BASIS MUST NOT SEE A TEST LANGUAGE. It is a PCA over the aligned spaces, so fitting it on
    # a held-out language's distribution is peeking -- no gold label is involved, which is exactly
    # why nothing downstream would ever raise. Test tables are PROJECTED through the train basis
    # instead (`AlignedVectors.project`). Refused here rather than left to the caller's memory.
    if meta and a.stage == "basis":
        allowed = set(meta.get("basis_langs") or meta["train"])
        leak = sorted(set(a.langs) - allowed)
        if leak:
            sys.exit(f"REFUSING to fit the basis on non-training languages: {leak}")

    if HUB not in a.langs:
        a.langs = [HUB] + [l for l in a.langs if l != HUB]
    # identity routes read their pivot's cached space, so the pivot must be fitted first
    a.langs.sort(key=lambda l: SOURCES[l]["route"] == "identity")
    {"fit": stage_fit, "basis": stage_basis, "emit": stage_emit}[a.stage](a)


main()
