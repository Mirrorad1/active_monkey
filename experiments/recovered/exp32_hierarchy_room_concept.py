# ============================================================
# Experiment 32 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_018aczVwQGTNCg2FHiRBc6tJ
#   description  : Exp 32: hierarchy — room concept recoverable from learned place map
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np
rng=np.random.default_rng(0)
# 2-room world: cells 0-3 (room A), 4-7 (room B); dense within room, single doorway 3<->4
N=8
adj={0:[1,2],1:[0,2,3],2:[0,1,3],3:[1,2,4],4:[3,5,6],5:[4,6,7],6:[4,5,7],7:[5,6]}
true_room=np.array([0,0,0,0,1,1,1,1])
# creature wanders (places observable = anchor, per Exp31 lesson) and LEARNS connectivity (counts)
counts=np.zeros((N,N)); pos=0
for _ in range(6000):
    nxt=rng.choice(adj[pos]); counts[pos,nxt]+=1; counts[nxt,pos]+=1; pos=nxt
W=counts/counts.max()
# extract higher-level ROOM concept from the SELF-LEARNED map via spectral clustering (community detection)
D=np.diag(W.sum(1)); L=D-W
Dm=np.diag(1.0/np.sqrt(W.sum(1))); Lsym=Dm@L@Dm
vals,vecs=np.linalg.eigh(Lsym)
fiedler=vecs[:,1]                      # 2nd-smallest eigenvector splits the graph into communities
pred=(fiedler>0).astype(int)
acc=max((pred==true_room).mean(),(pred!=true_room).mean())   # up to label swap
print('Exp32 — HIERARCHY: does a ROOM concept self-organize from the learned place map?')
print(f'  learned cluster assignment: {pred.tolist()}')
print(f'  true rooms                : {true_room.tolist()}   recovery accuracy={acc:.2f} (up to label swap)')
print(f'  spectral gap (room structure strength): lambda2={vals[1]:.3f}')
print('  => a higher-level ROOM concept is LATENT IN and RECOVERABLE FROM the self-learned structure' if acc==1.0 else '  => partial')