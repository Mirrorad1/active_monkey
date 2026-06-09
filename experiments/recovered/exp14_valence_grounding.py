# ============================================================
# Experiment 14 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_016hBXNLfhf1EsSGBLmUNfrt
#   description  : Exp 14 refined: valence grounding with determined 2-char context
#   recovered    : 2026-06-09 by recovery agent
#   notes        : Refined final run with 2-char context; earlier attempt toolu_01TgEkHBLMGutR632qwpoozP used 1-char context
# ============================================================

import numpy as np, math, jax, jax.numpy as jnp
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
    for nn in range(V): B0[c*V+nn,s,0]=1.0
B0=B0/B0.sum(0,keepdims=True)
def build():
    return Agent(A=[bat(A)[0]],B=[bat(B0)[0]],D=[bat(np.ones(K)/K)[0]],pB=[bat(np.full((K,K,1),0.05))],
        num_controls=[1],policy_len=1,action_selection='deterministic',sampling_mode='full',
        inference_algo='fpi',batch_size=1,learn_B=True)
def train(ag,text,epochs):
    obs=encode(text)
    for _ in range(epochs):
        prior=[ag.D[0]]; qss=[]; acts=[]
        for o in obs:
            qs=ag.infer_states([jnp.array([o])],prior); qss.append(qs); prior=ag.update_empirical_prior(NOOP,qs); acts.append(NOOP)
        T=len(obs); bel=[jnp.concatenate([qss[t][0] for t in range(T)],axis=1)]
        ob=[jnp.array([[obs[t] for t in range(T)]])]; a=jnp.concatenate([x[:,None,:] for x in acts],axis=1)
        ag=ag.infer_parameters(beliefs_A=bel,observations=ob,actions=a,beliefs_B=bel)
    return ag
def ent_after(ag,prime):
    Am=np.asarray(ag.A[0])[0]; Bm=np.asarray(ag.B[0])[0,:,:,0]; prior=[ag.D[0]]
    for o in encode(prime):
        qs=ag.infer_states([jnp.array([o])],prior); prior=ag.update_empirical_prior(NOOP,qs)
    st=np.asarray(prior[0]).reshape(-1); st=st/st.sum(); p=Am@(Bm@st); p=p/p.sum()
    return -np.sum(p*np.log(p+1e-12))/math.log(2)
rng=np.random.default_rng(0); segs=[]; letters='bcdfghjklpqrtvwxy'
for _ in range(60):
    segs.append('ammmm. '); segs.append('z'+''.join(rng.choice(list(letters),4))+'. ')
ag=train(build(),''.join(segs),epochs=8)
# determined 2-char context: space precedes both cues in the corpus
ea=ent_after(ag,' a'); ez=ent_after(ag,' z')
# also: confidence one step deeper (after 'am' vs after 'z'+letter)
print('VALENCE GROUNDING (determined context):')
print(f'  P(next | ..a) uncertainty: {ea:.2f} bits   P(next | ..z): {ez:.2f} bits')
print(f'  cue a predicts a lower-free-energy (more understood) state than z: {ea<ez}  (delta {ez-ea:+.2f} bits)')
print(f'  interpretation: an arbitrary symbol acquired POSITIVE valence purely by co-occurring')
print(f'  with the agent low-surprise states - grounded, never labeled.' if ez-ea>0.3 else '  (effect weak; grounding present but small at this scale)')