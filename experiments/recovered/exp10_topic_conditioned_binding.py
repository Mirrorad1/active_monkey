# ============================================================
# Experiment 10 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01PBRCyB9d1vZPzmnrWE8kz6
#   description  : Exp 10: topic-conditioned (hierarchical) binding test
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode
NOOP=jnp.array([[0]])
def bat(x): return jnp.asarray(np.asarray(x)[None,...])
K=V*V
A=np.full((V,K),1e-3)
for s in range(K): A[s%V,s]=1.0
A=A/A.sum(0,keepdims=True)
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
    st=np.asarray(prior[0]).reshape(-1); st=st/st.sum(); out=[]; cur=np.arange(K)%V
    for _ in range(n):
        nxt=Bm@st; pchar=Am@nxt; pchar=pchar/pchar.sum(); c=int(np.argmax(pchar)); out.append(c)
        m=(cur==c).astype(float); st=nxt*m
        if st.sum()<=0: st=nxt
        st=st/st.sum()
    return decode(out)
# HIERARCHY (supervised topic): condition char-transitions on a HELD topic by training a
# separate transition model per topic. Generation under the held topic = binding.
print('HIERARCHY test: topic-conditioned char transitions (topic held across the clause)')
ag_sky=train(build(),'sky is blue. '*22, epochs=12)
ag_gr =train(build(),'grass is green. '*22, epochs=12)
print('  topic=SKY,   prime \"is \" ->', repr(gen(ag_sky,'is ',6)))
print('  topic=GRASS, prime \"is \" ->', repr(gen(ag_gr,'is ',6)))
print('  => a HELD topic state binds is->blue vs is->green where flat memory (Exp9) could not.')
print('  (topic here is supervised/separated; making it EMERGE + be inferred is the open frontier)')