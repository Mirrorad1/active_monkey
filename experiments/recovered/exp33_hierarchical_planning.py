# ============================================================
# Experiment 33 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_018G5qUguvTw2HyRvM6LPvV2
#   description  : Exp 33: hierarchical vs flat planning cost
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np
# corridor of K rooms, each 3x3=9 cells, linked linearly by doorways. goal in last room, start in first.
def build(K):
    RS=9; N=K*RS
    def rc(local): return divmod(local,3)
    adj={s:set() for s in range(N)}
    for k in range(K):
        base=k*RS
        for l in range(RS):
            r,c=rc(l)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<3 and 0<=nc<3: adj[base+l].add(base+nr*3+nc)
        if k<K-1:  # doorway: cell 5 (r1c2) of room k <-> cell 3 (r1c0) of room k+1
            a=base+5; b=base+RS+3; adj[a].add(b); adj[b].add(a)
    return N,adj,RS
def flat_cost(N,adj,goal):
    V=np.full(N,-1e9); V[goal]=0; updates=0; sweeps=0
    for _ in range(4*N):
        sweeps+=1; newV=V.copy()
        for s in range(N):
            if s==goal: continue
            newV[s]=max(V[n] for n in adj[s])-1; updates+=1
        if np.allclose(newV,V): break
        V=newV
    return updates,sweeps,V
def hier_cost(N,adj,RS,K,start,goal):
    # room graph (K nodes) connectivity is a line; plan room sequence (BFS over K)
    sr,gr=start//RS, goal//RS
    room_path=list(range(sr,gr+1)) if gr>=sr else list(range(sr,gr-1,-1))
    updates=K*len(room_path)  # tiny room-level VI cost
    # local VI only within rooms on the path
    for rm in room_path:
        base=rm*RS; local=[base+l for l in range(RS)]
        sub_goal = goal if rm==gr else base+5  # head to doorway else final goal
        V={s:-1e9 for s in local}; V[sub_goal]=0
        for _ in range(2*RS):
            nv=dict(V)
            for s in local:
                if s==sub_goal: continue
                nbrs=[n for n in adj[s] if n in V]
                if nbrs: nv[s]=max(V[n] for n in nbrs)-1; updates+=1
            if nv==V: break
            V=nv
    return updates,len(room_path)
print('Exp33 — HIERARCHICAL planning cost vs FLAT (corridor of rooms):')
for K in (4,8,16):
    N,adj,RS=build(K); start=0; goal=N-1
    fu,fs,_=flat_cost(N,adj,goal)
    hu,_=hier_cost(N,adj,RS,K,start,goal)
    print(f'  {K} rooms ({N} cells): flat VI state-updates={fu:,}  hierarchical={hu:,}  speedup x{fu/max(hu,1):.1f}')
print('  => abstraction (rooms) makes planning COARSE-TO-FINE: cost grows far slower than flat at scale')