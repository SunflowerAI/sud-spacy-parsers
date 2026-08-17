#!/usr/bin/env python3
"""A two-state HMM over characters: TRANSLITERATION vs NATIVE.

WHY AN HMM RATHER THAN A THRESHOLD PLUS A RUN LENGTH. The binary gate ("every character scores >= 2.0
and sits in a run of >= 3") is two hand-set numbers doing the work of one model, and it fails in a
diagnosable way: 帝 scores −1.16, 利 +0.57, 弗 +1.66, 多 +1.68, so 竭帝 / 舍利弗 / 阿耨多羅三藐三菩提
are all excluded even though their neighbours are unmistakably transliteration. An HMM replaces the
threshold with a graded EMISSION (the character's likelihood under each corpus) and the run length
with a TRANSITION (transliteration is sticky), so a low-scoring character can be carried by its
context instead of breaking the span.

EMISSIONS come from the two corpora already in use: CBETA T08 (with every evaluation text excluded)
for the transliteration state, 42 M characters of kanripo for the native state. TRANSITIONS are a
sticky prior, swept rather than fitted -- there is no labelled T/N sequence data to fit them on, and
inventing one would beg the question.

⚠ The emission ratio is the SAME log-odds the binary inventory was built from. If the HMM wins it is
the graded treatment and the transition structure doing it, not new information.
"""
import argparse, collections, math, pathlib, subprocess

def corpora(excl):
    texts=[]
    for f in sorted(pathlib.Path("assets_cbeta/T08").glob("*.xml")):
        if f.name in excl: continue
        o=subprocess.run([".venv/bin/python","scripts/cbeta_text.py",str(f)],
                         capture_output=True,text=True)
        if o.returncode==0: texts.append(o.stdout)
    bud="".join(c for c in "".join(texts) if '一'<=c<='鿿')
    cls=pathlib.Path("corpus_lzh_kanripo_leakfree.txt").read_text(encoding="utf-8",errors="replace")
    cls="".join(c for c in cls if '一'<=c<='鿿')
    return bud, cls

class HMM:
    def __init__(self, bud, cls, stay_t=0.85, stay_n=0.995):
        cb=collections.Counter(bud); cc=collections.Counter(cls)
        V=len(set(cb)|set(cc))+1
        self.lb={c: math.log((cb.get(c,0)+1)/(len(bud)+V)) for c in set(cb)|set(cc)}
        self.lc={c: math.log((cc.get(c,0)+1)/(len(cls)+V)) for c in set(cb)|set(cc)}
        self.dflt_b=math.log(1/(len(bud)+V)); self.dflt_c=math.log(1/(len(cls)+V))
        self.a=[[math.log(stay_t), math.log(1-stay_t)],
                [math.log(1-stay_n), math.log(stay_n)]]     # 0 = T, 1 = N
        self.pi=[math.log(0.02), math.log(0.98)]
    def emit(self, ch):
        return (self.lb.get(ch, self.dflt_b), self.lc.get(ch, self.dflt_c))
    def viterbi(self, text):
        if not text: return []
        e=self.emit(text[0]); d=[self.pi[0]+e[0], self.pi[1]+e[1]]; bp=[]
        for ch in text[1:]:
            e=self.emit(ch); nd=[0.0,0.0]; b=[0,0]
            for j in (0,1):
                cand=[d[0]+self.a[0][j], d[1]+self.a[1][j]]
                b[j]=0 if cand[0]>=cand[1] else 1
                nd[j]=cand[b[j]]+e[j]
            bp.append(b); d=nd
        st=0 if d[0]>=d[1] else 1; out=[st]
        for b in reversed(bp):
            st=b[st]; out.append(st)
        return out[::-1]

    def posterior(self, text):
        """P(state=T | whole string) per character, by forward-backward.

        Viterbi gives ONE operating point; the posterior gives a curve, which is what a gate needs.
        Computed in log space -- these strings are thousands of characters and the naive product
        underflows well before that.
        """
        import math
        n=len(text)
        if not n: return []
        def lse(a,b):
            m=max(a,b)
            return m+math.log(math.exp(a-m)+math.exp(b-m)) if m>-math.inf else m
        E=[self.emit(c) for c in text]
        f=[[0.0,0.0] for _ in range(n)]
        f[0]=[self.pi[0]+E[0][0], self.pi[1]+E[0][1]]
        for t in range(1,n):
            for j in (0,1):
                f[t][j]=lse(f[t-1][0]+self.a[0][j], f[t-1][1]+self.a[1][j])+E[t][j]
        b=[[0.0,0.0] for _ in range(n)]
        for t in range(n-2,-1,-1):
            for i in (0,1):
                b[t][i]=lse(self.a[i][0]+E[t+1][0]+b[t+1][0],
                            self.a[i][1]+E[t+1][1]+b[t+1][1])
        out=[]
        for t in range(n):
            num=f[t][0]+b[t][0]; den=lse(f[t][0]+b[t][0], f[t][1]+b[t][1])
            out.append(math.exp(num-den))
        return out
