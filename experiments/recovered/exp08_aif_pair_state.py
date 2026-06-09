# ============================================================
# Experiment 8 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01UVrtAndHd9y9ByAMDBUxYk
#   description  : AIF 2-char-context pair-state model: mirro and Q-to-A
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode
NOOP=jnp.array([[0]])
def bat(x): return jnp.asarray(np.asarray(x)[None,...])
K=V*V  # pair-state s=(prev,cur)=prev*V+cur  => 2-char memory inside active inference
# near-deterministic emission: state s emits its current char (s%V)
A=np.full((V,K),1e-3)
for s in range(K): A[s%V,s]=1.0
A=A/A.sum(0,keepdims=True)
# structured transition prior: from s=(p,c) only (c,*) are valid next states; learn which
B0=np.full((K,K,1),1e-4)
for s in range(K):
    c=s%V
    for n in range(V): B0[c*V+n,s,0]=1.0
B0=B0/B0.sum(0,keepdims=True)
def build():
    return Agent(A=[bat(A)[0]],B=[bat(B0)[0]],D=[bat(np.ones(K)/K)[0]],
                 pB=[bat(np.full((K,K,1),0.05))],num_controls=[1],policy_len=1,
                 action_selection='deterministic',sampling_mode='full',inference_algo='fpi',
                 batch_size=1,learn_B=True)
def train(ag,text,epochs):
    obs=encode(text)
    for _ in range(epochs):
        prior=[ag.D[0]]; qss=[]; acts=[]
        for o in obs:
            qs=ag.infer_states([jnp.array([o])],prior); qss.append(qs)
            prior=ag.update_empirical_prior(NOOP,qs); acts.append(NOOP)
        T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
        ob=[jnp.array([[obs[t] for t in range(T)]])]; a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
        ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag
def gen(ag,prefix,n):
    Am=np.asarray(ag.A[0])[0]; Bm=np.asarray(ag.B[0])[0,:,:,0]
    prior=[ag.D[0]]
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],prior); prior=ag.update_empirical_prior(NOOP,qs)
    st=np.asarray(prior[0]).reshape(-1); st=st/st.sum(); out=[]
    cur=np.arange(K)%V
    for _ in range(n):
        nxt=Bm@st                      # predicted next-state belief
        pchar=Am@nxt; pchar=pchar/pchar.sum()
        c=int(np.argmax(pchar)); out.append(c)
        m=(cur==c).astype(float); st=nxt*m
        if st.sum()<=0: st=nxt
        st=st/st.sum()
    return decode(out)
print('AIF 2-char-context (pair-state) model, greedy:')
ag=train(build(),'mirro '*18, epochs=10)
print('  taught mirro -> from \"m\":', repr(gen(ag,'m',11)))
ag2=train(build(),'name. mirro. '*18, epochs=12)
print('  Q->A: after \"name. \" ->', repr(gen(ag2,'name. ',8)))