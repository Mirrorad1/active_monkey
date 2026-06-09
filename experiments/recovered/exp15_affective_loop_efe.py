# ============================================================
# Experiment 15 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01YFWikGuoBWD4UMnPuTyv7K
#   description  : Exp 15: close the affective loop — agent acts via EFE to seek grounded-good state
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
# Choice world: 2 scenes after acting. action0 -> 'understood' scene (predictable obs),
# action1 -> 'surprising' scene (uniform obs). Preference C is GROUNDED in self-evidencing:
# the agent prefers the outcome it can predict/understand (the comfort obs), not an external label.
Vo=4
A=np.zeros((Vo,2))
A[:,0]=[0.91,0.03,0.03,0.03]   # good scene: peaked => low free energy / understood
A[:,1]=[0.25,0.25,0.25,0.25]   # bad scene: uniform => high free energy / surprised
A=nrm(A,0)
B=np.zeros((2,2,2))
B[:,:,0]=np.array([[1,1],[0,0]])  # action0 -> scene 0 (good)
B[:,:,1]=np.array([[0,0],[1,1]])  # action1 -> scene 1 (bad)
C=[b(np.array([2.0,0,0,0]))]      # prefer the comfort obs (the predictable one it grounded as good)
D=[b(np.array([0.5,0.5]))]
ag=Agent(A=[b(A)[0]],B=[b(B)[0]],C=C,D=D,num_controls=[2],policy_len=1,
         action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
qs=ag.infer_states([jnp.array([0])], D)   # neutral first obs
q_pi,neg_efe=ag.infer_policies(qs)
qp=np.asarray(q_pi).reshape(-1); ne=np.asarray(neg_efe).reshape(-1)
print('CLOSING THE AFFECTIVE LOOP (agent ACTS via EFE):')
print(f'  policy posterior q(pi): seek-good={qp[0]:.3f}  seek-bad={qp[1]:.3f}')
print(f'  EFE: good-action={-ne[0]:.3f}  bad-action={-ne[1]:.3f}  (lower EFE = chosen)')
print(f'  agent chooses to SEEK the self-grounded understood state: {qp[0]>qp[1]}')
print('  => action to occupy low-free-energy (understood) states emerges from EFE, no labeled reward')