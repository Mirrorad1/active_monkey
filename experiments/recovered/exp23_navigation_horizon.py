# ============================================================
# Experiment 23 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_0136PvqZZUQJTK1uRREYUry1
#   description  : Exp 23: navigation with adequate planning horizon
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=3; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,1,2,1,2,0,2,0,1]); NC=3; goal=8
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
Acom=np.zeros((2,N)); Acom[0,:]=1.0; Acom[:,goal]=[0,1]
key=jax.random.PRNGKey(0)
def navigate(pl,maxsteps=12):
    global key
    C=[b(np.zeros(NC)), b(np.array([0.0,4.0]))]
    ag=Agent(A=[b(Acol)[0],b(Acom)[0]],B=[b(Bt)[0]],C=C,D=[b(np.eye(N)[0])[0]],
             num_controls=[4],policy_len=pl,action_selection='stochastic',sampling_mode='full',
             inference_algo='fpi',batch_size=1)
    pos=0; prior=[ag.D[0]]; path=[0]
    for t in range(maxsteps):
        qs=ag.infer_states([jnp.array([int(cmap[pos])]),jnp.array([1 if pos==goal else 0])],prior)
        qpi,_=ag.infer_policies(qs)
        key,sub=jax.random.split(key)
        act=ag.sample_action(qpi,rng_key=jax.random.split(sub,1))
        a=int(jnp.asarray(act).reshape(-1)[0]); prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a); path.append(pos)
        if pos==goal: return t+1,path
    return None,path
print('Exp23 — PLANNING DEPTH (does horizon>=distance enable navigation?) optimal=4, random~9.8')
for pl in (4,5):
    s,p=navigate(pl)
    print(f'  policy_len={pl}: reached goal in {s} steps  path={p}')