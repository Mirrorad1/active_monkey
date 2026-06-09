# ============================================================
# Experiment 13 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_014WE5xW1qL988qFKWsmfAh6
#   description  : Exp 13: semi-supervised topic seeding then unsupervised binding test
#   recovered    : 2026-06-09 by recovery agent
#   notes        : Final run after batch-axis fix; two earlier runs failed due to PYTHONPATH and batch-axis bugs
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
B0=n(0.8*np.eye(T)+0.2,0)
B1=n(np.full((V,V,T),1e-3)+0.02*rng.random((V,V,T)),0)
def build():
    return Agent(A=[b(A)[0]], B=[b(B0[:,:,None])[0], b(B1[...,None])[0]],
        D=[b(np.ones(T)/T)[0], b(np.ones(V)/V)[0]], A_dependencies=[[1]], B_dependencies=[[0],[1,0]],
        num_controls=[1,1], policy_len=1, action_selection='deterministic', sampling_mode='full',
        inference_algo='fpi', batch_size=1, learn_B=True,
        pB=[b(np.full((T,T,1),0.05))[0], b(np.full((V,V,T,1),0.05))[0]])
ACT=jnp.array([[0,0]])
clauses=[('sky is blue. ',0),('grass is green. ',1)]
def train(ag,epochs,force=True):
    for _ in range(epochs):
        for txt,tk in clauses:
            obs=encode(txt); prior=[b(np.eye(T)[tk]) if force else ag.D[0], ag.D[1]]
            bt=[[],[]]; acts=[]
            for o in obs:
                qs=ag.infer_states([jnp.array([o])],prior)
                bt[0].append(qs[0]); bt[1].append(qs[1])
                prior=ag.update_empirical_prior(ACT,qs)
                if force: prior=[b(np.eye(T)[tk]), prior[1]]   # teacher-force topic each step
                acts.append(ACT)
            Tn=len(obs); bel=[jnp.concatenate(bt[0],axis=1),jnp.concatenate(bt[1],axis=1)]
            ob=[jnp.array([[obs[t] for t in range(Tn)]])]; a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
            ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag
def run(ag,prefix,n_):
    Am=np.asarray(ag.A[0])[0]; B1m=np.asarray(ag.B[1])[0]
    pr=[ag.D[0],ag.D[1]]   # UNSUPERVISED at test: topic uniform, must be inferred
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],pr); pr=ag.update_empirical_prior(ACT,qs)
    tb=np.asarray(pr[0]).reshape(-1); cb=np.asarray(pr[1]).reshape(-1); out=[]
    for _ in range(n_):
        nb=np.zeros(V)
        for k in range(T): nb+=tb[k]*(B1m[:,:,k,0]@cb)
        nb=n(nb,0); c=int(np.argmax(Am@nb)); out.append(c); oh=np.zeros(V); oh[c]=1; cb=oh
    return decode(out), np.round(tb,2)
ag=train(build(),8,force=True)
s,ts=run(ag,'sky is ',6); g,tg=run(ag,'grass is ',6)
print('SEMI-SUPERVISED (teacher-forced topic in training, UNSUPERVISED topic at test):')
print('  sky is   ->',repr(s),' inferred topic',ts)
print('  grass is ->',repr(g),' inferred topic',tg)
print('  bound differently?', s!=g)