"""Exp 43 — dissociation probe: held intent factor vs. flat context depth.

Corpus: two Q->A pairs whose questions share a 13-char suffix ' like green. ' and
differ only in prefix: A 'do you like green. ' -> 'it unsettles me. ';
B 'does anyone like green. ' -> 'no one does. '.
Hypothesis: a held intent factor binds question identity across a span that NO
in-window flat depth covers (shared suffix 13 chars >> depth 3).
Models: (1) flat depth-2 pymdp pair-state (Exp 41 machinery); (2) flat depth-3 --
identical generative math computed as a deterministic count-chain (dense pymdp at
K=28^3 is memory-infeasible; Dirichlet counts on deterministic states ARE transition
counts, equivalence established at depth 2 by Exp 8); (3) intent(2 blocks)+depth-2
(Exp 42 machinery), intent labels PROVIDED in training, intent INFERRED at test.
Prediction if TRUE: models 1 and 2 fail selection with identical continuations for
both questions; model 3 infers intent 2/2 and selects 2/2.
Predeclared selection metric: generated first-4 chars == correct answer's first-4
('it u' vs 'no o') and != the other answer's first-4. Secondary: exact match, intent mass.
Falsifier: model 3 selection <= models 1/2, OR models 1/2 selection succeeds (depth
suffices; intent factor unnecessary on this probe).
Seed: deterministic (greedy decode, no sampling).
"""

import numpy as np, jax, jax.numpy as jnp
from collections import defaultdict
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode, normalize

NOOP = jnp.array([[0]])

def bat(x): return jnp.asarray(np.asarray(x)[None,...])

# ── Corpus ──────────────────────────────────────────────────────────────────
PAIRS = [
    ("do you like green.", "it unsettles me."),
    ("does anyone like green.", "no one does."),
]
BLOCK = "".join(q + " " + a + " " for q, a in PAIRS)

# ── Exp 41 machinery (flat depth-2 pymdp) ───────────────────────────────────
K = V * V  # pair-state s=(prev,cur)=prev*V+cur

A = np.full((V, K), 1e-3)
for s in range(K): A[s % V, s] = 1.0
A = A / A.sum(0, keepdims=True)

B0 = np.full((K, K, 1), 1e-4)
for s in range(K):
    c = s % V
    for n in range(V): B0[c * V + n, s, 0] = 1.0
B0 = B0 / B0.sum(0, keepdims=True)

def build():
    return Agent(A=[bat(A)[0]], B=[bat(B0)[0]], D=[bat(np.ones(K) / K)[0]],
                 pB=[bat(np.full((K, K, 1), 0.05))], num_controls=[1], policy_len=1,
                 action_selection='deterministic', sampling_mode='full', inference_algo='fpi',
                 batch_size=1, learn_B=True)

def train(ag, text, epochs):
    obs = encode(text)
    for _ in range(epochs):
        prior = [ag.D[0]]; qss = []; acts = []
        for o in obs:
            qs = ag.infer_states([jnp.array([o])], prior); qss.append(qs)
            prior = ag.update_empirical_prior(NOOP, qs); acts.append(NOOP)
        T = len(obs); bel = [jnp.concatenate([qss[t][0] for t in range(T)], axis=1)]
        ob = [jnp.array([[obs[t] for t in range(T)]])]; a = jnp.concatenate([x[:, None, :] for x in acts], axis=1)
        ag = ag.infer_parameters(beliefs_A=bel, observations=ob, actions=a, beliefs_B=bel)
    return ag

def gen(ag, prefix, n):
    Am = np.asarray(ag.A[0])[0]; Bm = np.asarray(ag.B[0])[0, :, :, 0]
    prior = [ag.D[0]]
    for o in encode(prefix):
        qs = ag.infer_states([jnp.array([o])], prior); prior = ag.update_empirical_prior(NOOP, qs)
    st = np.asarray(prior[0]).reshape(-1); st = st / st.sum(); out = []
    cur = np.arange(K) % V
    for _ in range(n):
        pchar = Am @ st; pchar = pchar / pchar.sum()
        c = int(np.argmax(pchar)); out.append(c)
        m = (cur == c).astype(float); st2 = st * m
        if st2.sum() <= 0: st2 = st
        st = Bm @ (st2 / st2.sum())
        st = st / st.sum()
    return decode(out)

# ── Model 1: flat depth-2 ────────────────────────────────────────────────────
print("=== Model 1: flat depth-2 pymdp (BLOCK*4, epochs=8) ===")
ag1 = train(build(), BLOCK * 4, epochs=8)

m1_results = []
for i, (q, a) in enumerate(PAIRS):
    g = gen(ag1, q + " ", len(a))
    sel = (g[:4] == a[:4]) and (g[:4] != PAIRS[1-i][1][:4])
    exact = (g == a)
    m1_results.append((g, sel, exact))
    print(f"  pair {i}: generated={repr(g)}  target={repr(a)}  sel4={sel}  exact={exact}")

m1_sel = sum(r[1] for r in m1_results)
m1_exact = sum(r[2] for r in m1_results)
L1 = min(len(r[0]) for r in m1_results)
m1_ident = (m1_results[0][0][:L1] == m1_results[1][0][:L1])

# ── Model 2: flat depth-3 count-chain ───────────────────────────────────────
# counts don't need epochs; a single pass over the corpus is equivalent to
# N-pass training on a deterministic model (same counts, just scaled)
print()
print("=== Model 2: flat depth-3 count-chain (BLOCK*4, single pass) ===")

def train_ngram(text, d):
    counts = defaultdict(lambda: defaultdict(float))
    for i in range(len(text) - d):
        counts[text[i:i+d]][text[i+d]] += 1
    return counts

def gen_ngram(counts, prefix, n, d):
    ctx = prefix[-d:]; out = []
    for _ in range(n):
        f = counts.get(ctx) if isinstance(counts, dict) else counts[ctx]
        if not f: break
        c = max(sorted(f), key=lambda k: f[k])
        out.append(c); ctx = (ctx + c)[-d:]
    return "".join(out)

train_text2 = normalize(BLOCK * 4)
ng3 = train_ngram(train_text2, 3)

m2_results = []
for i, (q, a) in enumerate(PAIRS):
    pref = normalize(q + " ")
    tgt = normalize(a)
    g = gen_ngram(ng3, pref, len(tgt), 3)
    sel = (g[:4] == tgt[:4]) and (g[:4] != normalize(PAIRS[1-i][1])[:4])
    exact = (g == tgt)
    m2_results.append((g, sel, exact))
    print(f"  pair {i}: generated={repr(g)}  target={repr(tgt)}  sel4={sel}  exact={exact}")

m2_sel = sum(r[1] for r in m2_results)
m2_exact = sum(r[2] for r in m2_results)
L2 = min(len(r[0]) for r in m2_results)
m2_ident = (m2_results[0][0][:L2] == m2_results[1][0][:L2])

# ── Model 3: intent(2) + depth-2 joint (Exp 42 machinery) ───────────────────
print()
print("=== Model 3: intent(2)+depth-2 joint (per-intent agents, INTENTS=2) ===")
INTENTS = 2

blocks = []
for i, (q, a) in enumerate(PAIRS):
    corpus = (q + " " + a + " ") * 9
    ag_i = train(build(), corpus, epochs=8)
    Bi = np.asarray(ag_i.B[0])[0, :, :, 0]
    blocks.append(Bi)
    print(f"  intent {i}: trained on {repr(q + ' ' + a)}")

KJ = INTENTS * K  # 2 * 784 = 1568

AJ = np.full((V, KJ), 1e-3)
for s in range(KJ): AJ[s % K % V, s] = 1.0
AJ = AJ / AJ.sum(0, keepdims=True)

BJ = np.zeros((KJ, KJ))
for i in range(INTENTS):
    lo, hi = i * K, (i + 1) * K
    BJ[lo:hi, lo:hi] = blocks[i]

curchar_joint = (np.arange(KJ) % K) % V

def prime(belief, text):
    b = belief.copy()
    for o in encode(text):
        like = np.where(curchar_joint == o, 1.0, 1e-3)
        b = b * like; b = b / b.sum()
        b = BJ @ b; b = b / b.sum()
    return b

def gen_joint(belief, n):
    out = []; st = belief.copy()
    for _ in range(n):
        pchar = np.zeros(V)
        for v in range(V): pchar[v] = st[curchar_joint == v].sum()
        c = int(np.argmax(pchar)); out.append(c)
        m = (curchar_joint == c).astype(float); st2 = st * m
        if st2.sum() <= 0: st2 = st
        st = BJ @ (st2 / st2.sum()); st = st / st.sum()
    return decode(out)

belief0 = np.ones(KJ) / KJ
m3_results = []
for i, (q, a) in enumerate(PAIRS):
    b = prime(belief0, q + " ")
    intent_mass = np.array([b[j * K:(j + 1) * K].sum() for j in range(INTENTS)])
    argmax_intent = int(np.argmax(intent_mass))
    intent_ok = (argmax_intent == i)
    g = gen_joint(b, len(a))
    sel = (g[:4] == a[:4]) and (g[:4] != PAIRS[1-i][1][:4])
    exact = (g == a)
    m3_results.append((g, sel, exact, intent_ok, intent_mass))
    print(f"  pair {i}: q={repr(q)}")
    print(f"    intent_mass=[{intent_mass[0]:.3f}, {intent_mass[1]:.3f}]"
          f"  argmax={argmax_intent}  intent_ok={intent_ok}")
    print(f"    generated={repr(g)}  target={repr(a)}  sel4={sel}  exact={exact}")

m3_sel = sum(r[1] for r in m3_results)
m3_exact = sum(r[2] for r in m3_results)
m3_intent = sum(r[3] for r in m3_results)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"model1_flat_d2: selection {m1_sel}/2, exact {m1_exact}/2, identical_continuations={m1_ident}")
print(f"model2_flat_d3: selection {m2_sel}/2, exact {m2_exact}/2, identical_continuations={m2_ident}")
print(f"model3_intent_d2: intent {m3_intent}/2, selection {m3_sel}/2, exact {m3_exact}/2")

if m3_sel == 2 and m1_sel < 2 and m2_sel < 2:
    print("VERDICT: prediction CONFIRMED (intent binds beyond any tested depth)")
elif m1_sel == 2 or m2_sel == 2:
    print("VERDICT: falsifier HIT (flat depth suffices on this probe)")
elif m3_sel < 2:
    print("VERDICT: falsifier HIT (intent factor failed selection)")
else:
    print("VERDICT: MIXED")
