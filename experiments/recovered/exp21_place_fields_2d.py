# ============================================================
# Experiment 21 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01BeNtUsMK7isA9VkK42diLA
#   description  : Exp 21: scale to 2D grid — does place structure self-organize?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=3; N=G*G   # 3x3 grid, 9 cells
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
# aliased colors: 3 colors over 9 cells (each color 3x)
cmap=np.array([0,1,2, 1,2,0, 2,0,1])
NC=3
def mv(cell,a):
    r,c=divmod(cell,G)
    if a==0: r=max(0,r-1)
    elif a==1: r=min(G-1,r+1)
    elif a==2: c=max(0,c-1)
    else: c=min(G-1,c+1)
    return r*G+c
Bt=np.zeros((N,N,4))
for s in range(N):
    for a in range(4): Bt[mv(s,a),s,a]+=1.0
Bt=nrm(Bt,0)
rng=np.random.default_rng(0)
A0=nrm(np.ones((NC,N))+0.05*rng.random((NC,N)),0)
ag=Agent(A=[b(A0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],pA=[b(np.full((NC,N),0.1))],
         num_controls=[4],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
pos=0; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
for _ in range(900):
    col=int(cmap[pos]); qs=ag.infer_states([jnp.array([col])],prior); qss.append(qs); obs.append(col)
    a=int(rng.integers(0,4)); act=jnp.array([[a]]); acts.append(act)
    prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a)
T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
Alearn=nrm(np.asarray(ag.A[0])[0],0)
tuning=[int(np.argmax(Alearn[:,s])) for s in range(N)]
print('Exp21 — 2D grid (3x3), learn sensory map from scratch, continuous belief:')
print(f'  learned per-cell color tuning: {tuning}')
print(f'  true colormap                : {list(cmap)}   exact-match={tuning==list(cmap)}')
def H(p): p=p/p.sum(); return -np.sum(p*np.log(p+1e-12))/math.log(2)
ag2=Agent(A=[b(Alearn)[0]],B=[b(Bt)[0]],D=[b(np.ones(N)/N)[0]],num_controls=[4],policy_len=1,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
true=4; prior=[ag2.D[0]]
for a in [3,1,2,1,3,0,2,3]:
    qs=ag2.infer_states([jnp.array([int(cmap[true])])],prior); prior=ag2.update_empirical_prior(jnp.array([[a]]),qs); true=mv(true,a)
fin=np.asarray(prior[0]).reshape(-1)
print(f'  localize w/ LEARNED map in 2D: uncertainty {H(fin):.2f} bits, cell {int(np.argmax(fin))} vs true {true}, ok={int(np.argmax(fin))==true}')
print('  => recipe holds in 2D: place fields self-organize at larger scale' if tuning==list(cmap) and H(fin)<0.5 else '  => partial at this scale')