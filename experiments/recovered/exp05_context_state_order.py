# ============================================================
# Experiment 5 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_0156SYpSQW1KfaphbZLqUnLn
#   description  : Test context model (state=last char) for sequence order
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode
NOOP=jnp.array([[0]])
def norm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])

# CONTEXT model: state == current character (K=V), A near-identity (state i emits char i),
# learn B = char->char transitions. This gives the model MEMORY of the last char (a bigram).
def make_ctx(seed=0):
    K=V
    A=b(norm(np.eye(V)*20+0.05,0))          # state i strongly emits char i
    B=b(norm(np.ones((K,K,1))+0.01,0))      # transitions learned
    D=b(np.ones(K)/K); pB=[b(np.ones((K,K,1))*0.05)]
    return Agent(A=[A[0]],B=[B[0]],D=[D[0]],pB=pB,num_controls=[1],policy_len=1,
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
def gen(ag,prefix,n,key):
    prior=[ag.D[0]]
    for o in encode(prefix):
        qs=ag.infer_states([jnp.array([o])],prior); prior=ag.update_empirical_prior(NOOP,qs)
    Am=np.asarray(ag.A[0])[0]; Bm=np.asarray(ag.B[0])[0,:,:,0]; st=np.asarray(prior[0]).reshape(-1); out=[]
    for _ in range(n):
        p=Am@st; p=p/p.sum(); key,s=jax.random.split(key); c=int(jax.random.choice(s,V,p=jnp.asarray(p))); out.append(c)
        key,s=jax.random.split(key); k=int(jax.random.choice(s,len(st),p=jnp.asarray(st)))
        oh=np.zeros(len(st)); oh[k]=1; st=Bm@oh; st=st/st.sum()
    return decode(out)
key=jax.random.PRNGKey(3)
print('CONTEXT model (state=last char => memory of the previous letter):')
ag=make_ctx(1); ag=train(ag,'mirro '*12, epochs=6)
print('  taught mirro ->', [gen(ag,'m',10,jax.random.split(key,3)[i]) for i in range(3)])
ag2=make_ctx(2); ag2=train(ag2,'the cat sat on the mat. '*12, epochs=6)
print('  taught the cat sat on the mat ->', repr(gen(ag2,'the ',24,key)))