# ============================================================
# Experiment 31 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01XSz4tkvaVkwgSDi8X95oEo
#   description  : Exp 31: learn both A and B from scratch — unique vs aliased sensing
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
N=5
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
Bt=np.zeros((N,N,2))
for s in range(N): Bt[(s-1)%N,s,0]=1.0; Bt[(s+1)%N,s,1]=1.0
def trial(aliased):
    rng=np.random.default_rng(0)
    if aliased: cmap=np.array([0,0,1,0,1]); NO=2
    else: cmap=np.arange(N); NO=N
    Atrue=np.zeros((NO,N))
    for s in range(N): Atrue[cmap[s],s]=1.0
    A0=nrm(np.ones((NO,N))+0.05*rng.random((NO,N)),0)
    B0=nrm(np.ones((N,N,2))+0.05*rng.random((N,N,2)),0)   # UNKNOWN connectivity
    ag=Agent(A=[b(A0)[0]],B=[b(B0)[0]],D=[b(np.eye(N)[0])[0]],
             pA=[b(np.full((NO,N),0.1))[0]],pB=[b(np.full((N,N,2),0.1))[0]],
             num_controls=[2],policy_len=1,action_selection='stochastic',sampling_mode='full',
             inference_algo='fpi',batch_size=1,learn_A=True,learn_B=True)
    pos=0; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
    for _ in range(1200):
        o=int(cmap[pos]); qs=ag.infer_states([jnp.array([o])],prior); qss.append(qs); obs.append(o)
        a=int(rng.integers(0,2)); act=jnp.array([[a]]); acts.append(act)
        prior=ag.update_empirical_prior(act,qs); pos=(pos-1)%N if a==0 else (pos+1)%N
    T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
    ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
    ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
    Bl=nrm(np.asarray(ag.B[0])[0],0)
    # how well did learned-B recover the ring? predicted next-state for 'right' from each state:
    predR=[int(np.argmax(Bl[:,s,1])) for s in range(N)]
    trueR=[(s+1)%N for s in range(N)]
    return predR, trueR, predR==trueR
for aliased in (False,True):
    pr,tr,ok=trial(aliased)
    print(f'  aliased={aliased}: learned right-step map {pr}  true {tr}  recovered-topology={ok}')
print('Exp31 — from-scratch TOPOLOGY (learn A AND B): unique sensing vs aliased')