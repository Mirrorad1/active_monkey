# ============================================================
# Experiment 16 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01XtynLHq3srAe1K2XumM7MV
#   description  : 
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp, math
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode
T=2; ACT=jnp.array([[0,0]])
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def bat(x): return jnp.asarray(np.asarray(x)[None,...])

def char_trans(text):
    M=np.full((V,V),1e-3); idx=encode(text)
    for i in range(len(idx)-1): M[idx[i+1],idx[i]]+=1.0
    return nrm(M,0)
sky=char_trans('sky is blue. '*8); grass=char_trans('grass is green. '*8)

A=np.full((V,V),1e-3)
for c in range(V): A[c,c]=1.0
A=nrm(A,0)
B0=nrm(0.9*np.eye(T)+0.1,0)                 # topic persists, can switch a bit
# ASYMMETRIC WARM-START: topic0 = sky transitions, topic1 = grass transitions
B1=np.stack([sky,grass],axis=2)             # (V,V,T)

def build(B1init):
    return Agent(A=[bat(A)[0]], B=[bat(B0[:,:,None])[0], bat(B1init[...,None])[0]],
        D=[bat(np.ones(T)/T)[0], bat(np.ones(V)/V)[0]], A_dependencies=[[1]], B_dependencies=[[0],[1,0]],
        num_controls=[1,1], policy_len=1, action_selection='deterministic', sampling_mode='full',
        inference_algo='fpi', batch_size=1, learn_B=True,
        pB=[bat(np.full((T,T,1),0.05))[0], bat(np.full((V,V,T,1),0.05))[0]])

def run(ag,prefix,n_):
    Am=np.asarray(ag.A[0])[0]; B1m=np.asarray(ag.B[1])[0]  # (V,V,T,1)
    pr=[ag.D[0],ag.D[1]]
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],pr); pr=ag.update_empirical_prior(ACT,qs)
    tb=np.asarray(pr[0]).reshape(-1); cb=np.asarray(pr[1]).reshape(-1); out=[]
    for _ in range(n_):
        nb=np.zeros(V)
        for k in range(T): nb+=tb[k]*(B1m[:,:,k,0]@cb)
        nb=nrm(nb,0); c=int(np.argmax(Am@nb)); out.append(c); oh=np.zeros(V); oh[c]=1; cb=oh
    return decode(out), np.round(tb,2)

def train(ag,epochs):
    clauses=[('sky is blue. ',),('grass is green. ',)]
    for _ in range(epochs):
        for (txt,) in clauses:
            obs=encode(txt); prior=[ag.D[0],ag.D[1]]; bt=[[],[]]; acts=[]
            for o in obs:
                qs=ag.infer_states([jnp.array([o])],prior); bt[0].append(qs[0]); bt[1].append(qs[1])
                prior=ag.update_empirical_prior(ACT,qs); acts.append(ACT)
            Tn=len(obs); bel=[jnp.concatenate(bt[0],axis=1),jnp.concatenate(bt[1],axis=1)]
            ob=[jnp.array([[obs[t] for t in range(Tn)]])]; a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
            ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag

ag=build(B1)
s0,ts0=run(ag,'sky is ',6); g0,tg0=run(ag,'grass is ',6)
print('ASYMMETRIC WARM-START (symmetry-breaking test):')
print(f'  BEFORE unsup training: sky->{s0!r} t={ts0}   grass->{g0!r} t={tg0}   bound={s0!=g0}')
ag=train(ag,6)
s1,ts1=run(ag,'sky is ',6); g1,tg1=run(ag,'grass is ',6)
print(f'  AFTER 6 unsup epochs : sky->{s1!r} t={ts1}   grass->{g1!r} t={tg1}   bound={s1!=g1}')
print('  => persists = symmetry-breaking works; collapses = complexity term prunes the latent')
