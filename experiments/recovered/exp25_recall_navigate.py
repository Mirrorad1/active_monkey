# ============================================================
# Experiment 25 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01CcQN4HCSVz847fo3df8LUE
#   description  : Exp 25: recall a learned fact and navigate to it
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=3; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,1,2,1,2,0,2,0,1]); NC=3; obj_cell=4
def mv(cell,a):
    r,c=divmod(cell,G)
    if a==0:r=max(0,r-1)
    elif a==1:r=min(G-1,r+1)
    elif a==2:c=max(0,c-1)
    else:c=min(G-1,c+1)
    return r*G+c
Bt=np.zeros((N,N,4))
for s in range(N):
    for a in range(4): Bt[mv(s,a),s,a]+=1.0
Bt=nrm(Bt,0)
Acol=np.zeros((NC,N))
for s in range(N): Acol[cmap[s],s]=1.0
# PHASE 1: discover object@place by wandering (learn A_object), like Exp24
Aobj0=nrm(np.ones((2,N))+0.05*np.random.default_rng(0).random((2,N)),0)
agL=Agent(A=[b(Acol)[0],b(Aobj0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],
          pA=[b(np.full((NC,N),1e3))[0],b(np.full((2,N),0.1))[0]],num_controls=[4],policy_len=1,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1,learn_A=True)
rng=np.random.default_rng(1); pos=0; prior=[agL.D[0]]; qss=[]; oc=[]; oo=[]; acts=[]
for _ in range(800):
    qs=agL.infer_states([jnp.array([int(cmap[pos])]),jnp.array([1 if pos==obj_cell else 0])],prior)
    qss.append(qs); oc.append(int(cmap[pos])); oo.append(1 if pos==obj_cell else 0)
    a=int(rng.integers(0,4)); act=jnp.array([[a]]); acts.append(act); prior=agL.update_empirical_prior(act,qs); pos=mv(pos,a)
T=len(oc); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[oc[t] for t in range(T)]]),jnp.array([[oo[t] for t in range(T)]])]
aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
agL=agL.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
Aobj=nrm(np.asarray(agL.A[1])[0],0)
print(f'Exp25 — RECALL + NAVIGATE. learned object location = cell {int(np.argmax(Aobj[1,:]))} (true {obj_cell})')
# PHASE 2: WANT the object (C prefers object-present); navigate using the LEARNED object map + planning
C=[b(np.zeros(NC)), b(np.array([0.0,4.0]))]
agN=Agent(A=[b(Acol)[0],b(Aobj)[0]],B=[b(Bt)[0]],C=C,D=[b(np.eye(N)[0])[0]],num_controls=[4],policy_len=4,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
key=jax.random.PRNGKey(0); pos=0; prior=[agN.D[0]]; path=[0]
for t in range(10):
    qs=agN.infer_states([jnp.array([int(cmap[pos])]),jnp.array([1 if pos==obj_cell else 0])],prior)
    qpi,_=agN.infer_policies(qs); key,sub=jax.random.split(key)
    act=agN.sample_action(qpi,rng_key=jax.random.split(sub,1)); a=int(jnp.asarray(act).reshape(-1)[0])
    prior=agN.update_empirical_prior(act,qs); pos=mv(pos,a); path.append(pos)
    if pos==obj_cell: break
print(f'  navigated to its remembered object: reached cell {pos} in {len(path)-1} steps; path {path}; success={pos==obj_cell}')
print('  => LEARN a fact (object@place) by wandering, then RECALL it to guide goal-directed action')