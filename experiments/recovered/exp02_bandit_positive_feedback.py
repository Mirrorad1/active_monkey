# ============================================================
# Experiment 2 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01FqqQWY1CZgUfHoeLhhkppA
#   description  : Prototype: can an AIF agent learn to choose actions that earn positive feedback?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent

# Minimal affective bandit: states {start, positive, negative}; 2 responses.
# TRUTH (hidden from agent): response 0 -> positive, response 1 -> negative.
# Agent only has a PREFERENCE for the 'positive' observation (C). It must LEARN B.
def norm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(x[None,...])
S=3  # start,pos,neg
# A: identity-ish observation of state (3 obs)
A=b(norm(np.eye(3)*8+0.5,0))
# B: (S,S,2) start->? ; init near-uniform so agent is naive, learns from data
B0=norm(np.ones((S,S,2))+0.05*np.random.default_rng(0).random((S,S,2)),0)
B=b(B0)
D=b(np.array([1.0,0,0]))           # always start in 'start'
C=[b(np.array([0.0,3.0,-3.0]))]    # PREFER positive obs, disprefer negative
pB=[b(np.ones((S,S,2))*1.0)]       # learnable transitions
ag=Agent(A=[A[0]],B=[B[0]],C=C,D=[D[0]],pB=pB,num_controls=[2],policy_len=1,
         action_selection='stochastic',sampling_mode='full',inference_algo='fpi',
         batch_size=1,learn_B=True)
key=jax.random.PRNGKey(0)
def truth(resp):  # environment: resp0 -> positive(1), resp1 -> negative(2)
    return 1 if resp==0 else 2
pos_rate=[]; window=[]
for turn in range(60):
    prior=[ag.D[0]]
    qs=ag.infer_states([jnp.array([0])],prior)   # observe 'start'
    q_pi,neg_efe=ag.infer_policies(qs)
    key,sub=jax.random.split(key)
    act=ag.sample_action(q_pi,rng_key=jax.random.split(sub,1))
    resp=int(jnp.asarray(act).reshape(-1)[0])
    outcome=truth(resp)             # env returns positive/negative state
    window.append(1 if outcome==1 else 0)
    # learn B from this transition: start --resp--> outcome
    qs2=ag.infer_states([jnp.array([outcome])], ag.update_empirical_prior(act,qs))
    beliefs=[jnp.concatenate([qs[0],qs2[0]],axis=1)]   # (1,2,S)
    obs=[jnp.array([[0,outcome]])]
    actions=jnp.concatenate([act[:,None,:],act[:,None,:]],axis=1)
    ag=ag.infer_parameters(beliefs_A=beliefs,observations=obs,actions=actions,beliefs_B=beliefs)
    if len(window)>=10:
        pos_rate.append(np.mean(window[-10:]))
print('positive-feedback rate, early (turns 1-10):', round(np.mean(window[:10]),2))
print('positive-feedback rate, late  (turns 51-60):', round(np.mean(window[-10:]),2))
print('learned to seek positive:', np.mean(window[-10:]) > np.mean(window[:10]))