# ============================================================
# Experiment 19 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01SUnnDX6sv1g5doVyqrZY94
#   description  : Exp 19: learn sensory map from scratch under aliasing; do place fields self-organize?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
from pymdp.agent import Agent
N=6
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,0,1,0,1,1])              # TRUE sensory map (hidden from agent), aliased
Bt=np.zeros((N,N,2))
for s in range(N): Bt[(s-1)%N,s,0]=1.0; Bt[(s+1)%N,s,1]=1.0   # innate movement model (known)
rng=np.random.default_rng(0)
A0=nrm(np.ones((2,N))+0.05*rng.random((2,N)),0)   # agent does NOT know the sensory map (~uniform)
ag=Agent(A=[b(A0)[0]],B=[b(Bt)[0]],D=[b(np.eye(N)[0])[0]],pA=[b(np.full((2,N),0.1))],
         num_controls=[2],policy_len=1,action_selection='stochastic',sampling_mode='full',
         inference_algo='fpi',batch_size=1,learn_A=True)
NOOP=2
# LEARN the sensory map by wandering: proprioception (B) carries the belief; A grounds each place's look
def wander_learn(ag,start,steps):
    pos=start; prior=[ag.D[0]]; qss=[]; obs=[]; acts=[]
    for _ in range(steps):
        col=int(cmap[pos]); qs=ag.infer_states([jnp.array([col])],prior); qss.append(qs); obs.append(col)
        a=int(rng.integers(0,2)); act=jnp.array([[a]]); acts.append(act)
        prior=ag.update_empirical_prior(act,qs); pos=(pos-1)%N if a==0 else (pos+1)%N
    T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
    ob=[jnp.array([[obs[t] for t in range(T)]])]; aa=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
    return ag.infer_parameters(beliefs_A=bel,observations=ob,actions=aa,beliefs_B=bel), pos
pos=0
for ep in range(25): ag,pos=wander_learn(ag,pos,30)
Alearn=np.asarray(ag.A[0])[0]
tuning=[int(np.argmax(Alearn[:,s])) for s in range(N)]   # each hidden state's learned color = place field
print('FROM-SCRATCH place fields (learned sensory map under aliasing):')
print(f'  learned per-state color tuning: {tuning}')
print(f'  true ring color pattern        : {list(cmap)}   match-up-to-nothing={tuning==list(cmap)}')
# functional test: localize from UNIFORM start using the LEARNED map (place inference)
def H(p): p=p/p.sum(); return -np.sum(p*np.log(p+1e-12))/math.log(2)
ag2=Agent(A=[b(nrm(Alearn,0))[0]],B=[b(Bt)[0]],D=[b(np.ones(N)/N)[0]],num_controls=[2],policy_len=1,
          action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
true=2; prior=[ag2.D[0]]
for a in [1,1,0,1,1,0,1]:
    qs=ag2.infer_states([jnp.array([int(cmap[true])])],prior); prior=ag2.update_empirical_prior(jnp.array([[a]]),qs)
    true=(true-1)%N if a==0 else (true+1)%N
fin=np.asarray(prior[0]).reshape(-1)
print(f'  localize with LEARNED map: uncertainty {H(fin):.2f} bits, cell {int(np.argmax(fin))} vs true {true}, ok={int(np.argmax(fin))==true}')
print('  => if it localizes with its OWN learned map, place fields self-organized functionally')