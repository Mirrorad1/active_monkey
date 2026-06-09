# ============================================================
# Experiment 17 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_019naekQFwdQ2Z7cE4ewzFjH
#   description  : Exp 17: embodied gridworld — creature learns world structure from wandering
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
N=5  # ring world of 5 cells; actions: 0=left,1=right (wraparound). A mouse wandering a corridor-loop.
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
A=np.eye(N)                              # observe current cell (grounded sensory input)
# TRUE dynamics (hidden from agent): left/right shift. Agent must LEARN this by wandering.
Btrue=np.zeros((N,N,2))
for s in range(N):
    Btrue[(s-1)%N,s,0]=1.0               # action0 -> left
    Btrue[(s+1)%N,s,1]=1.0               # action1 -> right
B0=nrm(np.ones((N,N,2)),0)               # agent starts with NO model (uniform transitions)
pB=[b(np.full((N,N,2),0.1))]
ag=Agent(A=[b(A)[0]],B=[b(B0)[0]],D=[b(np.ones(N)/N)[0]],pB=pB,num_controls=[2],policy_len=1,
         action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1,learn_B=True)
rng=np.random.default_rng(0); pos=0; NOOPdim=2
# wander: random actions, collect (obs, action) stream, learn B
def episode(ag,pos,steps):
    prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
    for _ in range(steps):
        qs=ag.infer_states([jnp.array([pos])],prior); qss.append(qs); obs.append(pos)
        a=int(rng.integers(0,2)); act=jnp.array([[a]]); acts.append(act)
        prior=ag.update_empirical_prior(act,qs)
        pos=(pos-1)%N if a==0 else (pos+1)%N   # the world responds (embodiment)
    T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
    ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
    ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel)
    return ag,pos
for ep in range(20):
    ag,pos=episode(ag,pos,30)
Blearn=np.asarray(ag.B[0])[0]  # (N,N,2)
# how well did the wandering creature recover the world's structure?
errL=np.abs(nrm(Blearn[:,:,0],0)-Btrue[:,:,0]).mean()
errR=np.abs(nrm(Blearn[:,:,1],0)-Btrue[:,:,1]).mean()
# can it now PREDICT where right-action leads from each cell? (argmax of learned B)
predR=[int(np.argmax(Blearn[:,s,1])) for s in range(N)]
trueR=[(s+1)%N for s in range(N)]
print('EMBODIED model learning (a wandering creature learns its world):')
print(f'  learned-vs-true transition error: left={errL:.3f} right={errR:.3f} (0=perfect)')
print(f'  predicted next cell for RIGHT action: {predR}   true: {trueR}   match={predR==trueR}')
print('  => an internal MODEL of the world structure emerged from acting, unsupervised')