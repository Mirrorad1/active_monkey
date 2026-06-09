# ============================================================
# Experiment 40 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01J6XuG9JTYSC1VFQTEmm6iH
#   description  : Exp 40: opinions revisable by new experience
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math
F=4
rng=np.random.default_rng(0)
counts=np.ones((F,F))*0.1
def live(pred, steps, f0):
    f=f0
    def P(g): return np.eye(F)[(g+1)%F] if g==pred else np.ones(F)/F
    for _ in range(steps):
        nxt=rng.choice(F,p=P(f)); counts[f,nxt]+=1; f=nxt
    return f
def fav():
    learned=counts/counts.sum(1,keepdims=True)
    ent=np.array([-np.sum(learned[g]*np.log(learned[g]+1e-12)) for g in range(F)])/math.log(2)
    val=np.exp(-3.0*ent); return int(np.argmax(val))
f=live(pred=2, steps=4000, f0=0); a=fav()
print('Exp40 — are opinions REVISABLE by new experience?')
print(f'  phase1 (feature 2 comfortable): favorite = {a} (expect 2)')
# WORLD CHANGES: now feature 0 is the comfortable/predictable one. life continues.
f=live(pred=0, steps=8000, f0=f); bfav=fav()
print(f'  phase2 (world changed -> feature 0 comfortable): favorite = {bfav} (expect 0)')
print(f'  => opinion REVISED by new experience: {a==2 and bfav==0} (not frozen; updates as the world/evidence changes)')