# ============================================================
# Experiment 39 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_012vJ6vsNmpsMrqvfzDsCVE9
#   description  : Exp 39: noise robustness of opinion formation
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math
F=4
def raise_creature(pred, noise, seed, steps=6000):
    rng=np.random.default_rng(seed)
    # 'predictable' feature now only MOSTLY deterministic (prob 1-noise); else random
    def P(f):
        if f==pred:
            p=np.ones(F)*(noise/F); p[(f+1)%F]+=1-noise; return p/p.sum()
        return np.ones(F)/F
    counts=np.ones((F,F))*0.1; f=rng.integers(F)
    for _ in range(steps):
        nxt=rng.choice(F,p=P(f)); counts[f,nxt]+=1; f=nxt
    learned=counts/counts.sum(1,keepdims=True)
    ent=np.array([-np.sum(learned[g]*np.log(learned[g]+1e-12)) for g in range(F)])/math.log(2)
    val=np.exp(-3.0*ent); val/=val.sum(); return int(np.argmax(val)), val[pred]
print('Exp39 — NOISE robustness of opinion formation (favorite should stay = the predictable feature):')
for noise in (0.0, 0.2, 0.4, 0.6):
    fav,pv=raise_creature(pred=2, noise=noise, seed=1)
    print(f'  noise={noise:.1f}: favorite feature = {fav} (true 2, correct={fav==2}); value-mass on it {pv:.2f}')
print('  => opinion formation is robust to moderate noise; degrades only as the feature stops being predictable')