# ============================================================
# Experiment 4 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01YJnNa5FqUGf4YCmvnPzGHF
#   description  : Run teaching experiments: memory-for-order and question-to-answer
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode

NOOP=jnp.array([[0]])
def norm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])

def make(K, seed=0):
    rng=np.random.default_rng(seed)
    A=b(norm(rng.random((V,K))+0.1,0)); B=b(norm(rng.random((K,K,1))+0.1,0)); D=b(np.ones(K)/K)
    pA=[b(np.ones((V,K))*0.1)]; pB=[b(np.ones((K,K,1))*0.1)]
    return Agent(A=[A[0]],B=[B[0]],D=[D[0]],pA=pA,pB=pB,num_controls=[1],policy_len=1,
                 action_selection='deterministic',sampling_mode='full',inference_algo='fpi',
                 batch_size=1,learn_A=True,learn_B=True)

def train(ag,text,epochs):
    obs=encode(text)
    for _ in range(epochs):
        prior=[ag.D[0]]; qs_seq=[]; acts=[]
        for o in obs:
            qs=ag.infer_states([jnp.array([o])],prior); qs_seq.append(qs)
            prior=ag.update_empirical_prior(NOOP,qs); acts.append(NOOP)
        T=len(obs)
        bel=[jnp.concatenate([qs_seq[t][0] for t in range(T)],axis=1)]
        ob=[jnp.array([[obs[t] for t in range(T)]])]
        a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
        ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag

def gen(ag,prefix,n,key):
    prior=[ag.D[0]]
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],prior); prior=ag.update_empirical_prior(NOOP,qs)
    Am=np.asarray(ag.A[0])[0]; Bm=np.asarray(ag.B[0])[0,:,:,0]; st=np.asarray(prior[0]).reshape(-1)
    out=[]
    for _ in range(n):
        p=Am@st; p=p/p.sum(); key,s=jax.random.split(key); c=int(jax.random.choice(s,V,p=jnp.asarray(p))); out.append(c)
        key,s=jax.random.split(key); k=int(jax.random.choice(s,len(st),p=jnp.asarray(st)))
        oh=np.zeros(len(st)); oh[k]=1; st=Bm@oh; st=st/st.sum()
    return decode(out)

key=jax.random.PRNGKey(0)
print('=== EXPERIMENT A: does MORE MEMORY (states K) let it say mirro in order? ===')
for K in (12,30,60):
    ag=make(K,seed=1); ag=train(ag,'mirro '*10, epochs=8)
    samples=[gen(ag,'',12,jax.random.split(key,2)[i]) for i in range(2)]
    print(f'  K={K:3d}: it says -> {samples}')
print()
print('=== EXPERIMENT B: can it learn question -> answer (what is your name -> mirro)? ===')
qa='what is your name. mirro. '
ag=make(40,seed=2); ag=train(ag,qa*12, epochs=10)
for i,pref in enumerate(['what is your name. ', 'what is your name. ']):
    print(f'  ask {pref!r} -> {gen(ag,pref,18,jax.random.split(key,4)[i])!r}')