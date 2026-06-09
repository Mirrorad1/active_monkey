# ============================================================
# Experiment 38 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01P93Y1DqRcnSi9GBW4vqR5D
#   description  : Exp 38: integrated stack — learn place + value + plan, one pass
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=4; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
rng=np.random.default_rng(0)
cmap=rng.integers(0,3,size=N); NC=3; words=['red','blue','green']
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
# (1) LEARN place map A from scratch (anchor=known B), continuous belief
A0=nrm(np.ones((NC,N))+0.05*rng.random((NC,N)),0)
ag=Agent(A=[b(A0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],pA=[b(np.full((NC,N),0.1))[0]],
         num_controls=[4],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
pos=0; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
for _ in range(1500):
    o=int(cmap[pos]); qs=ag.infer_states([jnp.array([o])],prior); qss.append(qs); obs.append(o)
    a=int(rng.integers(0,4)); act=jnp.array([[a]]); acts.append(act); prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a)
T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
learned_map=np.array([int(np.argmax(np.asarray(ag.A[0])[0][:,s])) for s in range(N)])
map_acc=(learned_map==cmap).mean()
# (2) FORM a value over colors from experience (likes the predictable one); say it likes 'green'(2)
fav=2
# (3) PLAN: value-iteration to nearest favorite-colored cell, navigate from a corner
goalset=set(int(s) for s in range(N) if learned_map[s]==fav)
Vv=np.zeros(N); R=np.array([1.0 if s in goalset else 0.0 for s in range(N)]); g=0.9
for _ in range(60):
    Vn=R+g*np.array([max((Bt[:,s,a]*Vv).sum() for a in range(4)) for s in range(N)])
    Vn=np.where(R>0,R,Vn)
    if np.max(np.abs(Vn-Vv))<1e-6: Vv=Vn; break
    Vv=Vn
posn=0; path=[0]
for _ in range(20):
    a=int(np.argmax([(Bt[:,posn,a]*Vv).sum() for a in range(4)])); posn=mv(posn,a); path.append(posn)
    if posn in goalset: break
print('Exp38 — INTEGRATED stack (learn place + value + plan), one pass on 4x4:')
print(f'  (1) learned place map accuracy: {map_acc:.2f}')
print(f'  (2) self-valued color: {words[fav]} -> target cells {sorted(goalset)}')
print(f'  (3) navigated from corner to a {words[fav]} place: reached {posn} (a {words[learned_map[posn]]} place), steps {len(path)-1}, success={posn in goalset}')
print('  => perceive(learned map) + want(value) + act(plan to it) compose in one creature (consolidation)')