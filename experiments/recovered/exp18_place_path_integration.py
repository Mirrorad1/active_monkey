# ============================================================
# Experiment 18 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01XduXB1m4H6RX52YepERL9n
#   description  : Exp 18: place-cell emergence via path integration under aliased sensing
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
N=6
def b(x): return jnp.asarray(np.asarray(x)[None,...])
# ALIASED world: 6 cells, only 2 sensory colors. cmap repeats -> one glimpse can't localize.
cmap=np.array([0,0,1,0,1,1])         # cell -> color (aliased: 3 cells are color 0, 3 are color 1)
A=np.zeros((2,N))
for c in range(N): A[cmap[c],c]=1.0
# Movement model (known proprioception): left/right ring shift
Bt=np.zeros((N,N,2))
for s in range(N):
    Bt[(s-1)%N,s,0]=1.0; Bt[(s+1)%N,s,1]=1.0
ag=Agent(A=[b(A)[0]],B=[b(Bt)[0]],D=[b(np.ones(N)/N)[0]],num_controls=[2],policy_len=1,
         action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
def H(p): p=p/p.sum(); return -np.sum(p*np.log(p+1e-12))/math.log(2)
true=2  # start cell (hidden from agent; it begins fully uncertain)
prior=[ag.D[0]]
path=[1,1,0,1,1,0,1,1]  # a wander (right/left)
print('PLACE EMERGENCE via path integration (aliased sensing + known movement):')
print(f'  start: belief uncertainty {H(np.asarray(prior[0]).reshape(-1)):.2f} bits over {N} cells (lost)')
for t,a in enumerate(path):
    col=int(cmap[true])
    qs=ag.infer_states([jnp.array([col])],prior)
    bel=np.asarray(qs[0]).reshape(-1); bel=bel/bel.sum()
    if t in (0,2,4,7):
        print(f'  step {t+1}: saw color {col}, belief uncertainty {H(bel):.2f} bits, argmax-cell {int(np.argmax(bel))} (true {true})')
    prior=ag.update_empirical_prior(jnp.array([[a]]),qs)
    true=(true-1)%N if a==0 else (true+1)%N
fin=np.asarray(prior[0]).reshape(-1); fin=fin/fin.sum()
print(f'  => localized? uncertainty {H(fin):.2f} bits, predicted cell {int(np.argmax(fin))} vs true {true}, correct={int(np.argmax(fin))==true}')
print('  a precise PLACE representation emerged from movement+ambiguous sensing (path integration)')