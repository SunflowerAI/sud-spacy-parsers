#!/usr/bin/env python3
"""Wire script conversion into a traditional-only zh model: tokenizer in, `zh_script` component out.

The model trains on traditional GSD alone so 個 and 个 do not split one character's mass. Simplified
input is converted to traditional before segmentation and converted BACK afterwards, so a caller who
passes simplified text gets simplified text out.

⚠ ASSIGNING nlp.tokenizer DOES NOT UPDATE THE CONFIG. `to_disk` writes the config as it stands, so a
reloaded model rebuilds whatever the config names -- it loads, runs, converts nothing and says
nothing. `nlp.config["nlp"]["tokenizer"]` must be set too, and the check below RELOADS FROM DISK
rather than trusting the in-memory object, which is correct in exactly the case the artefact is not.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import seg_code                                    # noqa: E402,F401
import spacy                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_model"); ap.add_argument("out_model")
    args = ap.parse_args()
    import zh_script                                # noqa: F401
    from char_seg_tokenizer import CharSegTokenizer  # noqa: F401

    nlp = spacy.load(args.in_model)
    old = nlp.tokenizer
    tok = zh_script.ZhTradTokenizer(nlp.vocab)
    for attr in ("segmenter", "lexicon", "_seg", "_lex"):        # carry the trained segmenter over
        if hasattr(old, attr):
            setattr(tok, attr, getattr(old, attr))
    nlp.tokenizer = tok
    nlp.config["nlp"]["tokenizer"] = {"@tokenizers": "sud.ZhTradTokenizer.v1"}
    if "zh_script" not in nlp.component_names:
        nlp.add_pipe("zh_script", last=True)
    nlp.to_disk(args.out_model)

    rl = spacy.load(args.out_model)
    kind = type(rl.tokenizer).__name__
    print(f"  reloaded tokenizer: {kind}   pipeline: {rl.pipe_names}")
    if kind != "ZhTradTokenizer":
        raise SystemExit(f"REFUSING: reloaded model rebuilt a {kind}")
    simp, trad = "我们在北京大学学习中文。", "我們在北京大學學習中文。"
    for t in (trad, simp):
        d = rl(t)
        print(f"  {t} -> {' '.join(x.text for x in d)}")
    if rl(simp).text == rl(trad).text:
        raise SystemExit("REFUSING: simplified input came back traditional -- zh_script did not "
                         "restore the caller's script")


if __name__ == "__main__":
    main()
