"""Exp 42 — slow held 'intent' factor above the pair-state stream (two timescales).

Hypothesis: a held upper intent state restores Q->A selection on the converse corpus
where the flat pair-state model failed (Exp 41: 0/3 exact, continuations identical).
Construction: joint state s=(intent, prev, cur), K=3*784=2352; B block-diagonal in
intent (intent NEVER transitions: a held slow factor by construction); each block's
transitions = the per-intent Dirichlet-learned pair-state model. Intent labels are
PROVIDED during training (each pair trained under its own intent block — taught, like
word labels). At test intent is NOT provided: prime the question from a uniform prior
and let ordinary state inference concentrate the intent marginal, then generate.
Prediction if TRUE: (a) intent-posterior argmax correct 3/3 after each question prime;
(b) generated answers 3/3 exact (baseline Exp 41: 0/3).
Falsifier: exact count not better than 0/3, or intent posterior at chance (~1/3 each).
Seed: deterministic (no sampling, greedy decode).
"""

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode

NOOP = jnp.array([[0]])

def bat(x): return jnp.asarray(np.asarray(x)[None,...])

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

# ── corpus ──────────────────────────────────────────────────────────────────
PAIRS = [
    ("what do you like.", "i like red."),
    ("do you like green.", "it unsettles me."),
    ("where are you.", "i am at a green place."),
]

# ── Train one flat pair-state agent per intent ───────────────────────────────
print("=== Training 3 intent-specific agents ===")
blocks = []
for i, (q, a) in enumerate(PAIRS):
    corpus = (q + " " + a + " ") * 9
    ag_i = train(build(), corpus, epochs=8)
    Bi = np.asarray(ag_i.B[0])[0, :, :, 0]   # shape (784, 784)
    blocks.append(Bi)
    print(f"  intent {i}: trained on {repr(q + ' ' + a)}")

# ── Assemble joint model ─────────────────────────────────────────────────────
KJ = 3 * K   # 2352

# Joint emission AJ (V x KJ): column s emits char (s % K) % V
AJ = np.full((V, KJ), 1e-3)
for s in range(KJ): AJ[s % K % V, s] = 1.0
AJ = AJ / AJ.sum(0, keepdims=True)

# Joint transition BJ (KJ x KJ): block-diagonal, intent held
BJ = np.zeros((KJ, KJ))
for i in range(3):
    lo, hi = i * K, (i + 1) * K
    BJ[lo:hi, lo:hi] = blocks[i]

# ── Bayes-filter helpers ──────────────────────────────────────────────────────
curchar_joint = (np.arange(KJ) % K) % V   # emission index for each joint state

def prime(belief, text):
    """Filter belief through text, then advance one step; returns belief over next char's state."""
    b = belief.copy()
    for o in encode(text):
        like = np.where(curchar_joint == o, 1.0, 1e-3)
        b = b * like
        b = b / b.sum()
        b = BJ @ b
        b = b / b.sum()
    return b

def gen_joint(belief, n):
    out = []
    st = belief.copy()
    for _ in range(n):
        pchar = np.zeros(V)
        for v in range(V): pchar[v] = st[curchar_joint == v].sum()
        c = int(np.argmax(pchar)); out.append(c)
        m = (curchar_joint == c).astype(float); st2 = st * m
        if st2.sum() <= 0: st2 = st
        st = BJ @ (st2 / st2.sum()); st = st / st.sum()
    return decode(out)

# ── Test: prime each question, report intent posterior + generated answer ─────
print()
print("=== Joint model: question priming ===")
belief0 = np.ones(KJ) / KJ

intent_correct_count = 0
exact_count = 0

for i, (q, a) in enumerate(PAIRS):
    b = prime(belief0, q + " ")
    # Intent posterior: mass per block
    intent_mass = np.array([b[j * K:(j + 1) * K].sum() for j in range(3)])
    argmax_intent = int(np.argmax(intent_mass))
    intent_ok = (argmax_intent == i)
    if intent_ok: intent_correct_count += 1

    g = gen_joint(b, len(a))
    exact = (g == a)
    if exact: exact_count += 1

    print(f"  pair {i}: q={repr(q)}")
    print(f"    intent_mass=[{intent_mass[0]:.3f}, {intent_mass[1]:.3f}, {intent_mass[2]:.3f}]"
          f"  argmax_intent={argmax_intent}  correct={intent_ok}")
    print(f"    generated={repr(g)}  target={repr(a)}  exact={exact}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"intent_correct={intent_correct_count}/3")
print(f"exact_count={exact_count}/3 (baseline exp41: 0/3)")
if intent_correct_count == 3 and exact_count == 3:
    print("VERDICT: prediction CONFIRMED (intent 3/3, answers 3/3)")
elif exact_count == 0 or all(
        np.array([b[j * K:(j + 1) * K].sum() for j in range(3)]).max() < 0.5
        for b in [prime(np.ones(KJ) / KJ, q + " ") for q, _ in PAIRS]):
    print("VERDICT: falsifier HIT")
else:
    print(f"VERDICT: MIXED (intent {intent_correct_count}/3, answers {exact_count}/3)")
