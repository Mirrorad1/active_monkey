"""Exp 41 — flat pair-state AIF over the converse vocabulary (multi-pair Q->A).

Hypothesis: the flat pair-state (2-char-context) AIF model (Exp 8 pattern) trained on a
3-pair converse Q->A corpus reproduces the taught question->answer mapping.
Prediction if TRUE: each question prime evokes its own taught answer exactly (3/3 exact).
Falsifier: <3/3 exact. Expected failure mode (Exp 9 mechanism): every question ends in the
identical pair-state ('.',' '), so the model cannot select among answers -- all three
continuations should come out IDENTICAL.
Control: single-pair training (Exp 8 redux on this vocab) must give exact recall,
isolating the failure (if any) to multi-pair binding, not mechanics.
Seed: fixed default numpy/jax determinism; greedy decode (no sampling).
Amendment (same iteration, before verdict): gen() emitted from L+2 (one-char skip, artifact also visible in recovered Exp 8 raw output); fixed to emit-then-advance and rescored. No other changes.
"""

import numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent
from active_loop.alphabet import V, encode, decode

NOOP = jnp.array([[0]])

def bat(x): return jnp.asarray(np.asarray(x)[None,...])

K = V * V  # pair-state s=(prev,cur)=prev*V+cur => 2-char memory inside active inference

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

# emit-then-advance: fixes the one-char skip present in the recovered Exp 8 generator (prior already holds next-state belief)
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

# ── corpus ──────────────────────────────────────────────────────────────────
PAIRS = [
    ("what do you like.", "i like red."),
    ("do you like green.", "it unsettles me."),
    ("where are you.", "i am at a green place."),
]
BLOCK = "".join(q + " " + a + " " for q, a in PAIRS)

# ── Part 1: control (single pair, Exp 8 redux) ──────────────────────────────
print("=== Part 1: Control (single pair) ===")
q0, a0 = PAIRS[0]
ctrl_corpus = (q0 + " " + a0 + " ") * 9
ag_ctrl = train(build(), ctrl_corpus, epochs=8)
ctrl_gen = gen(ag_ctrl, q0 + " ", len(a0))
ctrl_exact = (ctrl_gen == a0)
print(f"  generated : {repr(ctrl_gen)}")
print(f"  target    : {repr(a0)}")
print(f"  exact={ctrl_exact}")

# ── Part 2: multi-pair main experiment ──────────────────────────────────────
print()
print("=== Part 2: Multi-pair (3-pair BLOCK x3, epochs=8) ===")
ag_multi = train(build(), BLOCK * 3, epochs=8)

generated = []
for i, (q, a) in enumerate(PAIRS):
    g = gen(ag_multi, q + " ", len(a))
    exact = (g == a)
    generated.append(g)
    print(f"  pair {i+1}: generated={repr(g)}  target={repr(a)}  exact={exact}")

exact_count = sum(gen_s == a for gen_s, (_, a) in zip(generated, PAIRS))
L = min(len(g) for g in generated)
all_identical = (generated[0][:L] == generated[1][:L] == generated[2][:L])
print()
print(f"exact_count={exact_count}/3")
print(f"all_continuations_identical_truncated={all_identical}")
if exact_count == 3:
    print("VERDICT: prediction CONFIRMED (3/3)")
else:
    print("VERDICT: falsifier HIT (<3/3)")
