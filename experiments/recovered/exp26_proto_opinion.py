# ============================================================
# Experiment 26 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01PWTD8HHYZ6gDwvHZ1AbsaF
#   description  : Exp 26: proto-opinion — preference learned from experience, history-dependent
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math
rng=np.random.default_rng(0)
F=3  # features (e.g. colors/places). 'comfort' = a feature whose consequences are PREDICTABLE (low free energy)
def world(predictable):
    # transition over features: at 'predictable' feature, next is deterministic (low surprise);
    # at others, next is uniform-random (high surprise). The creature does NOT know this; it learns.
    T=np.full((F,F),1.0/F)
    T[:,predictable]=np.eye(F)[predictable] if False else T[:,predictable]
    P={}
    for f in range(F):
        if f==predictable: P[f]=np.eye(F)[(f+1)%F]  # deterministic successor -> predictable/comfortable
        else: P[f]=np.ones(F)/F                      # random successor -> surprising
    return P
def live_and_form_preference(predictable, steps=4000):
    P=world(predictable)
    counts=np.ones((F,F))*0.1   # creature's LEARNED transition model (Dirichlet)
    f=rng.integers(F)
    for _ in range(steps):
        nxt=rng.choice(F,p=P[f]); counts[f,nxt]+=1; f=nxt
    learned=counts/counts.sum(1,keepdims=True)
    # experienced free energy per feature = predictive entropy under the creature's OWN learned model
    ent=np.array([-np.sum(learned[g]*np.log(learned[g]+1e-12)) for g in range(F)])/math.log(2)
    # preference forms from valence: prefer features that brought LOW free energy (comfort) — Exp14 grounding
    C=np.exp(-3.0*ent); C=C/C.sum()
    return ent, C
print('Exp26 — PROTO-OPINION: preference learned from experience (same creature, different worlds)')
for pred in (0,2):
    ent,C=live_and_form_preference(pred)
    print(f'  world where feature {pred} is the predictable/comfortable one:')
    print(f'    experienced free energy per feature (bits): {np.round(ent,2)}')
    print(f'    LEARNED preference C (what it came to value): {np.round(C,2)}  -> most-valued feature = {int(np.argmax(C))}')
e0,C0=live_and_form_preference(0); e2,C2=live_and_form_preference(2)
print(f'  same architecture, DIFFERENT history -> DIFFERENT learned preference: {int(np.argmax(C0))} vs {int(np.argmax(C2))}, differ={int(np.argmax(C0))!=int(np.argmax(C2))}')
print('  => the disposition is the creature OWN, formed by its lived experience, not pretrained')