# ============================================================
# Experiment 34 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01JnH3yoWUbeWa8sJj6ngS5W
#   description  : Exp 34: language bridge — query self-formed concepts in taught words
#   recovered    : 2026-06-09 by recovery agent
#   transcription fix: 2026-06-09 — restored shell-consumed \" escapes to " (python -c artifact)
# ============================================================

import numpy as np, math
rng=np.random.default_rng(0); F=3; words=['red','blue','green']
def raise_creature(predictable, steps=4000):
    P={f:(np.eye(F)[(f+1)%F] if f==predictable else np.ones(F)/F) for f in range(F)}
    counts=np.ones((F,F))*0.1; f=rng.integers(F)
    for _ in range(steps):
        nxt=rng.choice(F,p=P[f]); counts[f,nxt]+=1; f=nxt
    learned=counts/counts.sum(1,keepdims=True)
    ent=np.array([-np.sum(learned[g]*np.log(learned[g]+1e-12)) for g in range(F)])/math.log(2)
    value=np.exp(-3.0*ent); value/=value.sum()
    return value, ent
# TEACH word<->concept labels from a FEW examples (shared, taught): learn P(word|color)
def teach_words(n_examples=8):
    wc=np.ones((F,F))*0.1   # word x color
    for _ in range(n_examples):
        c=rng.integers(F); wc[c,c]+=1   # show color c, say its word -> learn association
    return wc/wc.sum(0,keepdims=True)
WC=teach_words()
def word_for(color): return words[int(np.argmax(WC[:,color]))]
def color_for(word): return int(np.argmax(WC[words.index(word),:]))
def ask_favorite(value): 
    fav=int(np.argmax(value)); return f'I like {word_for(fav)}'   # CONTENT self-formed, LABEL taught
def ask_about(value, ent, word):
    c=color_for(word)
    return (f'I like {word}' if value[c]>0.5 else f'{word} unsettles me') + f' (surprise {ent[c]:.1f} bits)'
vA,eA=raise_creature(0); vB,eB=raise_creature(2)
print('Exp34 — LANGUAGE BRIDGE: query SELF-FORMED concepts in TAUGHT words')
print(f'  taught word<->color map learned (P(word|color) argmax): {[word_for(c) for c in range(F)]} (correct={[word_for(c)==words[c] for c in range(F)]})')
print('  Q: what do you like?')
print(f'     creature-A: {ask_favorite(vA)}')
print(f'     creature-B: {ask_favorite(vB)}')
print('  Q: do you like green?')
print(f'     creature-A: {ask_about(vA,eA,"green")}')
print(f'     creature-B: {ask_about(vB,eB,"green")}')
print('  => same WORDED questions, DIFFERENT answers — content self-formed, labels taught (the bridge to talk-to-it)')