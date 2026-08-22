#!/usr/bin/env python3
"""Wire script conversion into a traditional-only zh model: tokenizer in, `zh_script` component out.

The model trains on traditional GSD alone so 個 and 个 do not split one character's mass. Simplified
input is converted to traditional before segmentation and converted BACK afterwards, so a caller who
passes simplified text gets simplified text out.

⚠ ASSIGNING nlp.tokenizer DOES NOT UPDATE THE CONFIG. `to_disk` writes the config as it stands, so a
reloaded model rebuilds whatever the config names -- it loads, runs, converts nothing and says
nothing. `nlp.config["nlp"]["tokenizer"]` must be set too, and the checks below RELOAD FROM DISK
rather than trusting the in-memory object, which is correct in exactly the case the artefact is not.

⚠ THIS SCRIPT SHIPPED A MODEL THAT COULD NOT SEGMENT. It used to carry the trained segmenter over
from the input model's tokenizer by trying attribute names -- `("segmenter", "lexicon", "_seg",
"_lex")` -- and `CharSegTokenizer` holds its segmenter in **`seg`**, which is not one of them. So
`ZhTradTokenizer` came out with no segmenter, `CharSegTokenizer.to_disk` writes a `segmenter/`
directory only when it has one, and `from_disk` falls back silently when the directory is absent.
zh_sud_gsd 0.2.0 went to the release returning each input string as a SINGLE TOKEN. It loaded, it
parsed, and only a hash of the wheel's file list said otherwise.

The segmenter is therefore LOADED here, from named paths, and the reload check now segments a
sentence and refuses a model that does not. Copying state between objects by guessing attribute
names is what failed; asking for it explicitly is the fix.

    add_zh_script.py training_zh_trad_lemma/model-best build_zh_trad \
        --seg models/zh_seg_jbdict_trad --lexicon models/zh_lex_corpus_trad.txt
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code                                    # noqa: E402,F401
import spacy                                       # noqa: E402

# ⚠ The default names the arm the wheel actually ships, because a comment telling the next person
# is not the fix. `zh_seg_jbdec_trad` is the SUPERSEDED one: its jieba channel reads a t2s rendering
# of the text rather than a traditional dictionary (`train_zh_trad_charseg.sh`).
SEG = "models/zh_seg_jbdict_trad"
LEX = "models/zh_lex_corpus_trad.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model"); ap.add_argument("out_model")
    ap.add_argument("--seg", default=SEG, help="trained character segmenter directory")
    ap.add_argument("--lexicon", default=LEX,
                    help="the FULL training word list. Jackknifing applies only during training; "
                         "at inference the model expects the complete list it was evaluated with.")
    args = ap.parse_args()
    import zh_script                                # noqa: F401
    from char_seg_tokenizer import CharSegTokenizer  # noqa: F401

    nlp = spacy.load(args.in_model)
    tok = zh_script.ZhTradTokenizer(nlp.vocab)
    tok.load_segmenter(args.seg, lexicon=args.lexicon)
    if getattr(tok, "seg", None) is None:
        raise SystemExit(f"REFUSING: no segmenter loaded from {args.seg}")
    nlp.tokenizer = tok
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.ZhTradTokenizer.v1"}
    if "zh_script" not in nlp.component_names:
        nlp.add_pipe("zh_script", last=True)
    nlp.to_disk(args.out_model)

    out = pathlib.Path(args.out_model)
    # Both are runtime imports the model cannot open without: jieba feeds one of the segmenter's two
    # input channels, opencc does the script conversion at both boundaries. The ja wheel's
    # ImportError-on-every-load is the reason this is declared rather than assumed.
    mp = out / "meta.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["requirements"] = sorted(set(m.get("requirements") or [])
                               | {"jieba>=0.42.1", "opencc-python-reimplemented>=0.1.7"})
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- verify the RELOAD, which is the artefact users get -----------------------------------
    sd = out / "tokenizer" / "segmenter"
    if not (sd / "model.bin").exists():
        raise SystemExit(f"REFUSING: {sd}/model.bin was not written -- the wheel would not segment")
    meta = json.loads((sd / "vocab.json").read_text(encoding="utf-8"))
    if meta.get("jieba_source") is None:
        raise SystemExit("REFUSING: the saved segmenter has no jieba_source marker, so the wheel "
                         "would load without the channel it was trained with")
    # The dictionary is the channel's VOCABULARY, and a wheel missing it comes back on jieba's
    # simplified one: it loads, it segments, and it is wrong only where the two disagree -- which
    # is the whole reason the traditional dictionary exists. Same refusal as the marker above,
    # one level down.
    import zh_jieba_feature as jf
    if meta.get("jieba_dict") and not (sd / jf.TRAD_DICT_FILE).is_file():
        raise SystemExit(f"REFUSING: the segmenter records jieba_dict={meta['jieba_dict']!r} but "
                         f"{jf.TRAD_DICT_FILE} was not written beside its weights")

    rl = spacy.load(args.out_model)
    kind = type(rl.tokenizer).__name__
    print(f"  reloaded tokenizer: {kind}   pipeline: {rl.pipe_names}")
    print(f"  segmenter: n_sources={meta.get('n_sources')} jieba_source={meta.get('jieba_source')} "
          f"jieba_t2s={meta.get('jieba_t2s')} jieba_dict={meta.get('jieba_dict')}   "
          f"requirements={m['requirements']}")
    if kind != "ZhTradTokenizer":
        raise SystemExit(f"REFUSING: reloaded model rebuilt a {kind}")

    simp, trad = "我们在北京大学学习中文。", "我們在北京大學學習中文。"
    for t in (trad, simp):
        d = rl(t)
        print(f"  {t} -> {' '.join(x.text for x in d)}")
    # The check the old version did not make. A tokenizer with no segmenter returns the whole
    # string, which parses and says nothing; only the token count gives it away.
    for t in (trad, simp):
        if len(rl(t)) < 2:
            raise SystemExit(f"REFUSING: {t!r} came back as a single token -- no segmentation")
    if rl(simp).text == rl(trad).text:
        raise SystemExit("REFUSING: simplified input came back traditional -- zh_script did not "
                         "restore the caller's script")
    if rl(trad).text != trad:
        raise SystemExit(f"REFUSING: traditional input came back as {rl(trad).text!r} -- the "
                         "script detector read it as simplified")


if __name__ == "__main__":
    main()
