# ============================================================
# Experiment 12 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01PM7UpSeSJ9RJjiX6JN9g8H
#   description  : Exp 12: full 2-factor unsupervised emergent-topic training + binding test
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode
T=2
def n(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
rng=np.random.default_rng(0)
A=np.full((V,V),1e-3)
for c in range(V): A[c,c]=1.0
A=n(A,0)
B0=n(0.85*np.eye(T)+0.15,0)                          # topic can switch occasionally
B1=n(np.full((V,V,T),1e-3)+0.02*rng.random((V,V,T)),0)  # asymmetric init breaks symmetry
ACT=jnp.array([[0,0]])
def build():
    return Agent(A=[b(A)[0]], B=[b(B0[:,:,None])[0], b(B1[...,None])[0]],
        D=[b(np.ones(T)/T)[0], b(np.ones(V)/V)[0]], A_dependencies=[[1]], B_dependencies=[[0],[1,0]],
        num_controls=[1,1], policy_len=1, action_selection='deterministic', sampling_mode='full',
        inference_algo='fpi', batch_size=1, learn_B=True,
        pB=[b(np.full((T,T,1),0.05))[0], b(np.full((V,V,T,1),0.05))[0]])
def train(ag,text,epochs):
    obs=encode(text)
    for _ in range(epochs):
        prior=[ag.D[0],ag.D[1]]; bt=[[],[]]; acts=[]
        for o in obs:
            qs=ag.infer_states([jnp.array([o])],prior)
            bt[0].append(qs[0]); bt[1].append(qs[1])
            prior=ag.update_empirical_prior(ACT,qs); acts.append(ACT)
        T_=len(obs)
        bel=[jnp.concatenate(bt[0],axis=1), jnp.concatenate(bt[1],axis=1)]
        ob=[jnp.array([[obs[t] for t in range(T_)]])]
        a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
        ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag
def run(ag,prefix,n_):
    Am=np.asarray(ag.A[0])[0]; B1m=np.asarray(ag.B[1])[0]  # (V,V,T,1)
    pr=[ag.D[0],ag.D[1]]
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],pr); pr=ag.update_empirical_prior(ACT,qs)
    tb=np.asarray(pr[0]).reshape(-1); cb=np.asarray(pr[1]).reshape(-1)
    out=[]
    for _ in range(n_):
        # predicted next char = sum_topic tb[k]*(B1[:,:,k]@cb)
        nb=np.zeros(V)
        for k in range(T): nb+=tb[k]*(B1m[:,:,k,0]@cb)
        nb=nb/nb.sum(); c=int(np.argmax(Am@nb)); out.append(c)
        oh=np.zeros(V); oh[c]=1; cb=oh
    return decode(out), np.round(tb,2)
ag=train(build(),'sky is blue. grass is green. '*16, epochs=10)
s,ts=run(ag,'sky is ',6); g,tg=run(ag,'grass is ',6)
print('UNSUPERVISED 2-factor result:')
print('  sky is   ->', repr(s), ' topic belief', ts)
print('  grass is ->', repr(g), ' topic belief', tg)
print('  bound differently?', s!=g)