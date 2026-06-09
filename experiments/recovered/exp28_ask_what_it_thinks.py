# ============================================================
# Experiment 28 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01Ybfp2S238U9aasW9w7rxoU
#   description  : Exp 28: ask-it-what-it-thinks query interface; answers reflect self-formed values
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math
rng=np.random.default_rng(0); F=3; names=['red','blue','green']
def world(predictable):
    P={}
    for f in range(F):
        P[f]=np.eye(F)[(f+1)%F] if f==predictable else np.ones(F)/F
    return P
def raise_creature(predictable, steps=4000):
    P=world(predictable); counts=np.ones((F,F))*0.1; f=rng.integers(F)
    for _ in range(steps):
        nxt=rng.choice(F,p=P[f]); counts[f,nxt]+=1; f=nxt
    learned=counts/counts.sum(1,keepdims=True)
    ent=np.array([-np.sum(learned[g]*np.log(learned[g]+1e-12)) for g in range(F)])/math.log(2)
    value=np.exp(-3.0*ent); value/=value.sum()   # self-formed value (Exp26): low free energy -> liked
    return value, ent, learned
def answer(value, ent, f):
    # map the creature's OWN learned value/free-energy for f into a verbal-ish self-report
    if value[f] > 0.5: return f'I like {names[f]} — it feels familiar and calm (low surprise {ent[f]:.1f} bits)'
    if value[f] > 0.2: return f'{names[f]} is okay (surprise {ent[f]:.1f} bits)'
    return f'{names[f]} unsettles me — it is unpredictable (surprise {ent[f]:.1f} bits)'
vA,eA,_=raise_creature(0)   # raised where 'red' was the predictable/comfortable one
vB,eB,_=raise_creature(2)   # raised where 'green' was comfortable
print('Exp28 — ASK IT WHAT IT THINKS (toy): two creatures, different upbringings, same questions')
for f in range(F):
    print(f'  Q: what do you think of {names[f]}?')
    print(f'     creature-A: {answer(vA,eA,f)}')
    print(f'     creature-B: {answer(vB,eB,f)}')
print(f'  favorite — A: {names[int(np.argmax(vA))]}   B: {names[int(np.argmax(vB))]}   (different, each from its own life)')
print('  => the ANSWER is the creature own, formed by lived experience, not pretrained')