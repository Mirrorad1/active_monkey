# ============================================================
# Experiment 36 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01LCyzteTwYAu5amSovMt7rA
#   description  : Exp 36: scale test — 6x6 place learning
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V  # unused; just confirms import path
G=6; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
rng=np.random.default_rng(0)
cmap=rng.integers(0,4,size=N)            # 4 aliased colors over 36 cells
NC=4
def mv(c,a):
    r,cc=divmod(c,G)
    if a==0:r=max(0,r-1)
    elif a==1:r=min(G-1,r+1)
    elif a==2:cc=max(0,cc-1)
    else:cc=min(G-1,cc+1)
    return r*G+cc
Bt=np.zeros((N,N,4))
for s in range(N):
    for a in range(4): Bt[mv(s,a),s,a]+=1.0
Bt=nrm(Bt,0)
A0=nrm(np.ones((NC,N))+0.05*rng.random((NC,N)),0)
ag=Agent(A=[b(A0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],pA=[b(np.full((NC,N),0.1))[0]],
         num_controls=[4],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
pos=0; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
for _ in range(2500):
    o=int(cmap[pos]); qs=ag.infer_states([jnp.array([o])],prior); qss.append(qs); obs.append(o)
    a=int(rng.integers(0,4)); act=jnp.array([[a]]); acts.append(act); prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a)
T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
Al=nrm(np.asarray(ag.A[0])[0],0)
tuning=np.array([int(np.argmax(Al[:,s])) for s in range(N)])
acc=(tuning==cmap).mean()
# functional localization with learned map from uniform start
def H(p): p=p/p.sum(); return -np.sum(p*np.log(p+1e-12))/math.log(2)
ag2=Agent(A=[b(Al)[0]],B=[b(Bt)[0]],D=[b(np.ones(N)/N)[0]],num_controls=[4],policy_len=1,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
true=0; prior=[ag2.D[0]]
for a in [3,1,3,1,3,1,2,1,3,0]:
    qs=ag2.infer_states([jnp.array([int(cmap[true])])],prior); prior=ag2.update_empirical_prior(jnp.array([[a]]),qs); true=mv(true,a)
fin=np.asarray(prior[0]).reshape(-1)
print(f'Exp36 — SCALE test 6x6 (36 cells, 4 aliased colors), learn place map from scratch:')
print(f'  learned sensory-map recovery accuracy: {acc:.2f} of {N} cells')
print(f'  localize with learned map: uncertainty {H(fin):.2f} bits, cell {int(np.argmax(fin))} vs true {true}, ok={int(np.argmax(fin))==true}')
print(f'  => recipe holds at ~4x scale' if acc>0.9 and H(fin)<1.0 else '  => degraded at this scale')