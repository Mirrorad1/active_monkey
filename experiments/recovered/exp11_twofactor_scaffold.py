# ============================================================
# Experiment 11 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01LkryVpr83PnGgZWoDCYrG5
#   description  : Probe: does pymdp 2-factor topic-conditioned model construct and infer?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V
T=2  # topics (slow factor)
def n(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
# A: 1 char modality, depends ONLY on char factor (factor 1)
A=np.full((V,V),1e-3); 
for c in range(V): A[c,c]=1.0
A=n(A,0)                                  # (V_obs, V_char)
# B0 topic: persists (near identity); B1 char: depends on [char, topic]
B0=n(np.eye(T)*0.98+0.01,0)               # (T,T)
B1=n(np.full((V,V,T),1e-3),0)             # (next_char, char, topic)
try:
    ag=Agent(A=[b(A)[0]],
             B=[b(B0[:,:,None])[0], b(B1[...,None])[0]],
             D=[b(np.ones(T)/T)[0], b(np.ones(V)/V)[0]],
             A_dependencies=[[1]], B_dependencies=[[0],[1,0]],
             num_controls=[1,1], policy_len=1, action_selection='deterministic',
             sampling_mode='full', inference_algo='fpi', batch_size=1, learn_B=True,
             pB=[b(np.full((T,T,1),0.05))[0], b(np.full((V,V,T,1),0.05))[0]])
    print('2-factor (topic+char, B1 depends on topic) CONSTRUCTED ok')
    prior=[ag.D[0], ag.D[1]]
    qs=ag.infer_states([jnp.array([5])], prior)
    print('infer_states ok; topic belief', np.round(np.asarray(qs[0]).reshape(-1),3), 'char-belief len', np.asarray(qs[1]).reshape(-1).shape)
except Exception as e:
    import traceback; print('SETUP ERROR (finding):', type(e).__name__, str(e)[:400])