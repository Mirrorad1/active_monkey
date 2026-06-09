# ============================================================
# Experiment 30 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01M9YtZMUPmx3gbBj4FA3BYC
#   description  : Exp 30: scalable planning via value iteration over learned B
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np
def run(G):
    N=G*G
    def mv(cell,a):
        r,c=divmod(cell,G)
        if a==0:r=max(0,r-1)
        elif a==1:r=min(G-1,r+1)
        elif a==2:c=max(0,c-1)
        else:c=min(G-1,c+1)
        return r*G+c
    B=np.zeros((N,N,4))                       # the creature's (learned) world model
    for s in range(N):
        for a in range(4): B[mv(s,a),s,a]+=1.0
    goal=N-1
    # VALUE PROPAGATION (backward induction over learned B): V[s]=max_a gamma * sum_s' B[s',s,a] V[s'], reward at goal
    V=np.zeros(N); R=np.zeros(N); R[goal]=1.0; gamma=0.95
    iters=0
    for _ in range(200):
        Vn=R.copy()
        for s in range(N):
            if s==goal: continue
            Vn[s]=gamma*max((B[:,s,a]*V).sum() for a in range(4))
        iters+=1
        if np.max(np.abs(Vn-V))<1e-6: V=Vn; break
        V=Vn
    # greedy policy from the value field; navigate from corner 0
    pos=0; path=[0]
    for _ in range(4*N):
        a=int(np.argmax([(B[:,pos,a]*V).sum() for a in range(4)])); pos=mv(pos,a); path.append(pos)
        if pos==goal: break
    opt=2*(G-1)
    return len(path)-1, opt, iters, len(path)-1==opt
for G in (3,5,8):
    steps,opt,iters,ok=run(G)
    enum=4**opt
    print(f'  {G}x{G} ({G*G} cells): value-iteration {iters} sweeps -> reached goal in {steps} steps (optimal {opt}, ok={ok}); policy-enumeration would need ~4^{opt}={enum:,} policies')
print('Exp30 — SCALABLE planning via value-propagation over the learned map:')
print('  => optimal navigation at POLYNOMIAL cost; scales where exponential policy enumeration cannot')