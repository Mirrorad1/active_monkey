# ============================================================
# Experiment 7 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01QL95KzyWY96i5K2fA6Euc4
#   description  : Context-depth control: n-gram bigram/trigram/4-gram on mirro and Q-to-A
#   recovered    : 2026-06-09 by recovery agent
#   transcription fix: 2026-06-09 — restored shell-consumed \" escapes to " (python -c artifact)
# ============================================================

import numpy as np, math
from active_loop.alphabet import V, encode, decode
# Context-depth CONTROL (count-based n-gram, NOT the AIF model): isolates how much
# context depth alone buys, to set the target the active-inference context model must match.
def ngram(text, n):
    idx=encode(text); ctx={}
    for i in range(len(idx)-1):
        c=tuple(idx[max(0,i-n+2):i+1]); ctx.setdefault(c,np.ones(V)*0.01)[idx[i+1]]+=1
    return {k:v/v.sum() for k,v in ctx.items()}
def surprise(model,text,n):
    idx=encode(text); tot=0; m=0
    for i in range(len(idx)-1):
        c=tuple(idx[max(0,i-n+2):i+1]); p=model.get(c, np.ones(V)/V)
        tot+=-math.log(p[idx[i+1]]+1e-12); m+=1
    return tot/max(m,1)/math.log(2)
def greedy(model,prefix,n,k):
    idx=encode(prefix); out=[]
    for _ in range(k):
        c=tuple(idx[max(0,len(idx)-n+1):]); p=model.get(c)
        nx=int(np.argmax(p)) if p is not None else encode(' ')[0]
        out.append(nx); idx.append(nx)
    return decode(out)
print('CONTEXT-DEPTH CONTROL (count-based n-gram):')
for n in (2,3,4):
    m=ngram('mirro '*30,n)
    print(f'  mirro n={n}: greedy from "m" -> {greedy(m,"m",n,11)!r}')
print()
for n in (2,3,4):
    m=ngram('name. mirro. '*30,n)
    print(f'  Q->A  n={n}: "name. " -> {greedy(m,"name. ",n,7)!r}')