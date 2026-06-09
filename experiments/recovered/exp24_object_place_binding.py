# ============================================================
# Experiment 24 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_017MBrRL4fJSu1zUeALeHm1E
#   description  : Exp 24: composite concept — bind object to place (relational learning)
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=3; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,1,2,1,2,0,2,0,1]); NC=3
obj_cell=4  # an object sits here (HIDDEN from agent; it must discover the relation object@place)
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
for s in range(N): Acol[cmap[s],s]=1.0          # known place-sense (from Exp21)
Aobj0=nrm(np.ones((2,N))+0.05*np.random.default_rng(0).random((2,N)),0)  # UNKNOWN object map
ag=Agent(A=[b(Acol)[0],b(Aobj0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],
         pA=[b(np.full((NC,N),1e3))[0], b(np.full((2,N),0.1))[0]],   # color fixed (huge prior), object learned
         num_controls=[4],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
rng=np.random.default_rng(1); pos=0; prior=[ag.D[0]]; qss=[]; ocol=[]; oobj=[]; acts=[]
for _ in range(900):
    qs=ag.infer_states([jnp.array([int(cmap[pos])]),jnp.array([1 if pos==obj_cell else 0])],prior)
    qss.append(qs); ocol.append(int(cmap[pos])); oobj.append(1 if pos==obj_cell else 0)
    a=int(rng.integers(0,4)); act=jnp.array([[a]]); acts.append(act)
    prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a)
T=len(ocol); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[ocol[t] for t in range(T)]]), jnp.array([[oobj[t] for t in range(T)]])]
aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
Aobj=nrm(np.asarray(ag.A[1])[0],0)              # learned object-presence map over places
learned_place=int(np.argmax(Aobj[1,:]))         # where the creature thinks the object is
print('Exp24 — COMPOSITE concept: bind OBJECT to PLACE from experience (a learned relation/fact)')
print(f'  learned P(object=present | place), per cell: {np.round(Aobj[1,:],2)}')
print(f'  creature infers object is at cell {learned_place}; true {obj_cell}; correct={learned_place==obj_cell}')
print('  => a place<->object RELATION self-organized from wandering: the seed of a proposition (\"the thing is there\")')