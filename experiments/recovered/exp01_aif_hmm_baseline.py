# ============================================================
# Experiment 1 — recovered from session transcript
# Provenance (added at recovery time, 2026-06-09):
#   session_id   : 72317201-ec87-49eb-88d2-beffa86bd7ec
#   transcript   : /Users/mirro/.claude/projects/-Users-mirro-Projects-pymdp/72317201-ec87-49eb-88d2-beffa86bd7ec.jsonl
#   tool_call_id : toolu_01A5bqD4UvR9AWHT45xYm8mm
#   description  : Prototype: does the AIF char HMM learn and beat uniform baseline?
#   recovered    : 2026-06-09 by recovery agent
# ============================================================

import time, numpy as np, jax, jax.numpy as jnp
from pymdp.agent import Agent

# tiny corpus with strong char structure
text = ('the cat sat on the mat. the dog ran in the fog. '
        'the cat and the dog sat on the mat. ') * 6
alphabet = sorted(set(text))
V = len(alphabet); idx = {c:i for i,c in enumerate(alphabet)}
print('V=', V, 'alphabet=', ''.join(alphabet), 'len text', len(text))
K = 12
rng = np.random.default_rng(0)

def norm(x, axis): return x / x.sum(axis=axis, keepdims=True)
def batch(x): return jnp.asarray(x[None, ...])

A = batch(norm(rng.random((V, K)) + 0.1, 0))
B = batch(norm(rng.random((K, K, 1)) + 0.1, 0))
D = batch(np.ones((K,))/K)
pA = [batch(np.ones((V, K)) * 0.1)]
pB = [batch(np.ones((K, K, 1)) * 0.1)]

ag = Agent(A=[A[0]], B=[B[0]], D=[D[0]], pA=pA, pB=pB, num_controls=[1],
           policy_len=1, action_selection='deterministic', sampling_mode='full',
           inference_algo='fpi', batch_size=1, learn_A=True, learn_B=True)

noop = jnp.array([[0]])  # (batch, num_factors)

def stream(ag, s, learn=True):
    prior = [ag.D[0]]
    qs_seq=[]; obs_seq=[]; act_seq=[]; surprise=0.0
    for ch in s:
        o = idx[ch]
        # predictive surprise BEFORE seeing: P(o) = sum_k A[:,k]*prior_k
        Amat = np.asarray(ag.A[0])[0]  # (V,K)
        pr = np.asarray(prior[0]).reshape(-1)
        pred = Amat @ pr; pred = pred/pred.sum()
        surprise += -np.log(pred[o] + 1e-12)
        qs = ag.infer_states([jnp.array([o])], prior)
        qs_seq.append(qs); obs_seq.append(o)
        prior = ag.update_empirical_prior(noop, qs)
        act_seq.append(noop)
    if learn:
        T=len(obs_seq)
        beliefs=[jnp.concatenate([qs_seq[t][0] for t in range(T)],axis=1)]
        observations=[jnp.array([[obs_seq[t] for t in range(T)]])]
        actions=jnp.concatenate([a[:,None,:] for a in act_seq],axis=1)
        ag=ag.infer_parameters(beliefs_A=beliefs, observations=observations, actions=actions, beliefs_B=beliefs)
    return ag, surprise/len(s)

split = int(len(text)*0.8)
train, held = text[:split], text[split:]
base = np.log(V)
print(f'uniform baseline: {base:.3f} nats/char = {base/np.log(2):.3f} bits/char')
t0=time.time()
_, held_s0 = stream(ag, held, learn=False)
print(f'held-out surprise BEFORE learning: {held_s0:.3f} nats = {held_s0/np.log(2):.3f} bits/char')
for epoch in range(8):
    ag, tr = stream(ag, train, learn=True)
_, held_s1 = stream(ag, held, learn=False)
print(f'held-out surprise AFTER 8 epochs: {held_s1:.3f} nats = {held_s1/np.log(2):.3f} bits/char')
print(f'beats uniform baseline: {held_s1 < base}; time {time.time()-t0:.1f}s')