# ============================================================
# Experiment 20 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01NDcVEu8RgUf7rDb6U4zCLv
#   description  : Exp 20: continuous-belief fix — does a clean place map self-organize?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
N=6
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,0,1,0,1,1])
Bt=np.zeros((N,N,2))
for s in range(N): Bt[(s-1)%N,s,0]=1.0; Bt[(s+1)%N,s,1]=1.0
rng=np.random.default_rng(0)
A0=nrm(np.ones((2,N))+0.05*rng.random((2,N)),0)
ag=Agent(A=[b(A0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],pA=[b(np.full((2,N),0.1))],
         num_controls=[2],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
# FIX: one CONTINUOUS wander (belief carried, never reset). Start aligned state0=cell0.
pos=0; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
for _ in range(700):
    col=int(cmap[pos]); qs=ag.infer_states([jnp.array([col])],prior); qss.append(qs); obs.append(col)
    a=int(rng.integers(0,2)); act=jnp.array([[a]]); acts.append(act)
    prior=ag.update_empirical_prior(act,qs); pos=(pos-1)%N if a==0 else (pos+1)%N
T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
Alearn=nrm(np.asarray(ag.A[0])[0],0)
tuning=[int(np.argmax(Alearn[:,s])) for s in range(N)]
# structural: match true cmap up to cyclic rotation + reflection?
def rots(x):
    out=[]
    for r in range(N):
        out.append([x[(i+r)%N] for i in range(N)]); out.append([x[(r-i)%N] for i in range(N)])
    return out
struct = list(cmap) in rots(tuning) or tuning in rots(list(cmap))
print('Exp20 — CONTINUOUS belief (registration fixed):')
print(f'  learned per-state tuning: {tuning}   true: {list(cmap)}   structural-match(up to rot/reflect): {struct}')
def H(p): p=p/p.sum(); return -np.sum(p*np.log(p+1e-12))/math.log(2)
ag2=Agent(A=[b(Alearn)[0]],B=[b(Bt)[0]],D=[b(np.ones(N)/N)[0]],num_controls=[2],policy_len=1,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
true=2; prior=[ag2.D[0]]
for a in [1,1,0,1,1,0,1,1]:
    qs=ag2.infer_states([jnp.array([int(cmap[true])])],prior); prior=ag2.update_empirical_prior(jnp.array([[a]]),qs)
    true=(true-1)%N if a==0 else (true+1)%N
fin=np.asarray(prior[0]).reshape(-1)
print(f'  localize with LEARNED map: uncertainty {H(fin):.2f} bits, cell {int(np.argmax(fin))} vs true {true}, ok={int(np.argmax(fin))==true}')
print('  => clean place map self-organizes when registration is maintained' if struct and H(fin)<0.5 else '  => still imperfect')