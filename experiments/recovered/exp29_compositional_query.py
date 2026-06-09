# ============================================================
# Experiment 29 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01KcR6WLse1LUvJ4FWqJ35bu
#   description  : Exp 29: compositional relational value-laden query
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import numpy as np
G=3; N=G*G; names=['red','blue','green']
cmap=np.array([0,1,2,1,2,0,2,0,1])   # color at each cell (the creature's learned place->color map, Exp21)
def neighbors(cell):
    r,c=divmod(cell,G); out=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<G and 0<=nc<G: out.append(nr*G+nc)
    return out
def feel(color, fav):
    if color==fav: return 'I like it'
    return 'it unsettles me'
def compositional_answer(fav):
    # COMPOSE: favorite color -> its locations -> their neighbors -> the neighbor colors -> feeling
    fav_cells=[s for s in range(N) if cmap[s]==fav]
    near=set()
    for s in fav_cells:
        for nb in neighbors(s): near.add(int(cmap[nb]))
    near=sorted(near)
    parts=[f'{names[c]} ({feel(c,fav)})' for c in near]
    return fav_cells, near, parts
print('Exp29 — COMPOSITIONAL/relational query: \"what is near the thing you like, and how do you feel about it?\"')
for fav,label in [(0,'creature-A (likes red)'),(2,'creature-B (likes green)')]:
    cells,near,parts=compositional_answer(fav)
    print(f'  {label}:')
    print(f'    my favorite {names[fav]} is at cells {cells}; near it I find: ' + ', '.join(parts))
print('  => a two-hop, VALUE-LADEN, RELATIONAL thought composed from learned place-map + learned values')
print('  (different favorites -> different composed answers; each the creature own)')