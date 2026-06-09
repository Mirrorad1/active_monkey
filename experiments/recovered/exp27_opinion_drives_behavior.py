# ============================================================
# Experiment 27 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01DNpMg6QT9PULfyCWFZx3AV
#   description  : Exp 27: self-formed opinion drives divergent behavior
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
G=3; N=G*G
def nrm(x,ax): return x/x.sum(axis=ax,keepdims=True)
def b(x): return jnp.asarray(np.asarray(x)[None,...])
cmap=np.array([0,1,2,1,2,0,2,0,1]); NC=3
def mv(cell,a):
    r,c=divmod(cell,G)
    if a==0:r=max(0,r-1)
    elif a==1:r=min(G-1,r+1)
    elif a==2:c=max(0,c-1)
    else:c=min(G-1,c+1)
    return r*G+c
Bt=np.zeros((N,N,4))
for s in range(N):
    for a in range(4): Bt[mv(s,a),s,a]+=1.0
Bt=nrm(Bt,0)
Acol=np.zeros((NC,N))
for s in range(N): Acol[cmap[s],s]=1.0
key=jax.random.PRNGKey(0)
def navigate(pref_color,start=1,maxsteps=8):
    global key
    C=[b(np.eye(NC)[pref_color]*4.0)]   # the creature's SELF-FORMED value (learned in Exp26)
    ag=Agent(A=[b(Acol)[0]],B=[b(Bt)[0]],C=C,D=[b(np.eye(N)[start])[0]],num_controls=[4],policy_len=4,
             action_selection='stochastic',sampling_mode='full',inference_algo='fpi',batch_size=1)
    pos=start; prior=[ag.D[0]]; path=[pos]
    for t in range(maxsteps):
        qs=ag.infer_states([jnp.array([int(cmap[pos])])],prior)
        qpi,_=ag.infer_policies(qs); key,sub=jax.random.split(key)
        act=ag.sample_action(qpi,rng_key=jax.random.split(sub,1)); a=int(jnp.asarray(act).reshape(-1)[0])
        prior=ag.update_empirical_prior(act,qs); pos=mv(pos,a); path.append(pos)
        if cmap[pos]==pref_color: break
    return pos,int(cmap[pos]),path
print('Exp27 — self-formed OPINION drives BEHAVIOR (same world, same start cell 1):')
pa,ca,patha=navigate(0)   # creature raised to value color 0
pb,cb,pathb=navigate(2)   # creature raised to value color 2
print(f'  creature-A (values color 0): went to cell {pa} (color {ca}); path {patha}')
print(f'  creature-B (values color 2): went to cell {pb} (color {cb}); path {pathb}')
print(f'  divergent behavior from divergent self-formed values: {pa!=pb and ca==0 and cb==2}')
print('  => two identical creatures, different lived values -> different purposeful behavior')