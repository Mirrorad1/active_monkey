"""
Exp 138 — continuous-substrate rung 6: the amortized control (closed-form conjugate
vs a minimal ELBO-trained encoder vs the grid-exact posterior, one generative model).

Without this rung no closed-form-vs-amortized claim from rungs 1-5 is licensed.
Hypothesis (worked out before code): at toy scale, with training streams sampled
freely from the true generative model, the amortized encoder TIES the closed-form
agent on held-out predictive NLL (both near the grid-exact ceiling) — the real
difference is the training bill: the encoder needs >= 1000x more generative-model
evaluations than the conjugate agent's ZERO. Because the encoder trains on
prior-sampled states, between-concept (blend-like) states are IN-distribution for
it: blends are predicted to tie as well. The toy CANNOT resolve the at-scale fork
(rung 4 showed exponents invisible); this rung's job is to bound every claim.

One generative model for all engines: s ~ N(0, 4I), o_t | s iid ~ NORMALIZED
mixture p_k(s) = softmax_k(-||s - mu_k||^2 / (2*0.35^2)) over the 6 hexagon
centers (radius 1); streams of T = 50 words. Engines on IDENTICAL eval streams:
(a) closed-form: GaussianBelief precision accumulation with the unnormalized
    footprint (the ladder's declared conjugacy-buying mismatch); zero training.
(b) amortized: encoder counts/T (6-dim) -> MLP(6->32->32->4) -> (mu, log diag
    sigma^2) of q(s); trained by reparameterized ELBO (8 samples) on the TRUE
    normalized-mixture likelihood, Adam lr 1e-3, 2000 steps, batch 128 streams
    freshly sampled per step (s ~ prior); one encoder per seed.
(c) grid-exact: 241x241 grid on [-6,6]^2, exact posterior under the true model
    (the gold standard, also the predictive ceiling).
Held-out predictive NLL: mean over 50 fresh words from the same state of
-log E_q[p_mix(o | s)] (MC with 256 samples from q for (a) and (b); exact grid
integration for (c)). Localization: ||E_q[s] - s_true||. Eval: 64 fresh streams
with s_true ~ prior per seed (in-distribution), plus 32 BLEND streams (words
drawn 50/50 from two opposite concepts, the rung-2 stimulus) with target =
the midpoint; seeds 0..7.

Predictions (TRUE iff all):
- P1 apples-to-apples possible: all three engines evaluated on identical streams
  under one generative model with one metric — the card's FAIL clause does not
  trigger.
- P2 accuracy: in-distribution cell means: |NLL_amort - NLL_closed| <= 0.10;
  NLL_closed - NLL_exact <= 0.10; localization closed <= amort + 0.05.
- P3 cost: encoder training consumes >= 1000x more generative-model word-samples
  than closed-form's zero-training (report the absolute count: steps x batch x T);
  report inference wall-clock per stream for both (no predeclared winner).
- P4 blends in-distribution for the encoder: |NLL_amort - NLL_closed| <= 0.15 on
  blend streams.

Falsifier (any triggers NEGATIVE): amortized BEATS closed-form by > 0.10 nats
in-distribution (the conjugacy mismatch has real predictive cost — log as a
finding), OR amortized WORSE by > 0.30 (cannot fit at toy scale — log as such),
OR blend asymmetry > 0.30 either way, OR the comparison proves structurally
impossible (the card's FAIL — log exactly why). Three-way rule per PROTOCOL
step 3: POSITIVE requires EVERY conjunct; "not a falsifier" never counts toward
POSITIVE; otherwise MIXED with notes.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from active_loop.continuous import GaussianBelief

# ---------------------------------------------------------------------------
# Generative model constants
# ---------------------------------------------------------------------------
K = 6                   # number of mixture components (hexagon)
SIGMA_HEX = 0.35        # hexagon emission spread
S2 = SIGMA_HEX ** 2    # 0.1225
PRIOR_MU = np.zeros(2)
PRIOR_SIGMA = 4.0 * np.eye(2)  # s ~ N(0, 4I)
T_STREAM = 50           # words per stream
N_INDIST = 64           # in-distribution eval streams per seed
N_BLEND = 32            # blend eval streams per seed
N_SEEDS = 8

# Hexagon centers: mu_k = (cos(k*pi/3), sin(k*pi/3))
MU_HEX = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(K)])

# Grid for exact posterior
G_SIZE = 241
G_RANGE = 6.0
gx = np.linspace(-G_RANGE, G_RANGE, G_SIZE)
gy = np.linspace(-G_RANGE, G_RANGE, G_SIZE)
GX, GY = np.meshgrid(gx, gy)       # (G_SIZE, G_SIZE)
GRID_PTS = np.stack([GX.ravel(), GY.ravel()], axis=1)  # (G^2, 2)
G_TOTAL = G_SIZE * G_SIZE
G_STEP = (2 * G_RANGE) / (G_SIZE - 1)

# Training hyperparameters
TRAIN_STEPS = 2000
BATCH_SIZE = 128
ELBO_SAMPLES = 8
MC_SAMPLES = 256        # for predictive NLL
LR = 1e-3
B1, B2, ADAM_EPS = 0.9, 0.999, 1e-8

# Training cost
TRAIN_WORD_SAMPLES = TRAIN_STEPS * BATCH_SIZE * T_STREAM


# ---------------------------------------------------------------------------
# Generative-model helpers (numpy)
# ---------------------------------------------------------------------------

def log_emission_np(s: np.ndarray) -> np.ndarray:
    """log p(o=k | s) for all k; shape (K,). True normalized mixture."""
    # logits = -||s - mu_k||^2 / (2 sigma^2)
    diff = s[None, :] - MU_HEX          # (K, 2)
    logits = -np.sum(diff ** 2, axis=1) / (2 * S2)
    # log-softmax
    m = np.max(logits)
    log_probs = logits - m - np.log(np.sum(np.exp(logits - m)))
    return log_probs


def sample_word(s: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one word from the true emission distribution at state s."""
    probs = np.exp(log_emission_np(s))
    return int(rng.choice(K, p=probs))


def sample_stream(s: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """Sample T words iid from p(o|s). Returns int array shape (T,)."""
    probs = np.exp(log_emission_np(s))
    return rng.choice(K, size=T, p=probs)


def sample_prior(rng: np.random.Generator) -> np.ndarray:
    """Sample s ~ N(0, 4I)."""
    return rng.multivariate_normal(PRIOR_MU, PRIOR_SIGMA)


# ---------------------------------------------------------------------------
# Grid-exact posterior
# ---------------------------------------------------------------------------

def _precompute_grid_log_emission() -> np.ndarray:
    """(G^2, K) log p(o=k | grid_pt). Pre-computed once, reused."""
    # GRID_PTS: (G^2, 2), MU_HEX: (K, 2)
    diff = GRID_PTS[:, None, :] - MU_HEX[None, :, :]   # (G^2, K, 2)
    logits = -np.sum(diff ** 2, axis=2) / (2 * S2)      # (G^2, K)
    # log-softmax over K for each grid point
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return logits - lse   # (G^2, K)


GRID_LOG_EMIT = _precompute_grid_log_emission()   # (G^2, K) — fixed

# Prior log-density on grid: log N(s; 0, 4I)
def _grid_log_prior() -> np.ndarray:
    """(G^2,) log N(s; 0, 4I)."""
    d = 2
    log_norm = -0.5 * d * np.log(2 * np.pi * 4.0)
    log_p = log_norm - np.sum(GRID_PTS ** 2, axis=1) / (2 * 4.0)
    return log_p


GRID_LOG_PRIOR = _grid_log_prior()  # (G^2,)


def grid_posterior(words: np.ndarray) -> np.ndarray:
    """Return (G^2,) normalized log-posterior given word sequence."""
    # log p(words | grid) = sum_t GRID_LOG_EMIT[grid, word_t]
    log_liks = np.sum(GRID_LOG_EMIT[:, words], axis=1)   # (G^2,)
    log_post = GRID_LOG_PRIOR + log_liks
    # normalize
    m = log_post.max()
    log_post = log_post - m - np.log(np.exp(log_post - m).sum())
    return log_post   # (G^2,) log posterior


def grid_posterior_mean(log_post: np.ndarray) -> np.ndarray:
    """E[s] under log_post; shape (2,)."""
    w = np.exp(log_post)
    return (w[:, None] * GRID_PTS).sum(axis=0)


def grid_predictive_nll(log_post: np.ndarray, held_words: np.ndarray) -> float:
    """Mean -log p(o_t) over held-out words; exact grid integration."""
    w = np.exp(log_post)   # (G^2,)
    nll = 0.0
    for o in held_words:
        log_po = np.log(np.sum(w * np.exp(GRID_LOG_EMIT[:, o])) + 1e-300)
        nll -= log_po
    return nll / len(held_words)


# ---------------------------------------------------------------------------
# Closed-form agent (GaussianBelief with conjugacy mismatch)
# ---------------------------------------------------------------------------

# The closed-form agent uses the unnormalized Gaussian footprint:
# observe(mu_k, I / sigma^2). This is the declared mismatch.
WORD_MUS = [MU_HEX[k] for k in range(K)]
WORD_LAMBDA = np.eye(2) / S2   # precision = I/sigma^2 (same for all words)


def closed_form_posterior(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, Sigma) of GaussianBelief posterior after observing words."""
    belief = GaussianBelief(PRIOR_MU, PRIOR_SIGMA)
    for w in words:
        belief.observe(WORD_MUS[w], WORD_LAMBDA)
    return belief.mu, belief.Sigma


def mc_predictive_nll(mu: np.ndarray, Sigma: np.ndarray, held_words: np.ndarray,
                       rng: np.random.Generator) -> float:
    """MC estimate of mean -log p(o_t) using 256 samples from N(mu, Sigma)."""
    samples = rng.multivariate_normal(mu, Sigma, size=MC_SAMPLES)  # (256, 2)
    nll = 0.0
    for o in held_words:
        # p(o|s) for each sample
        diff = samples[:, None, :] - MU_HEX[None, :, :]  # (256, K, 2)
        logits = -np.sum(diff ** 2, axis=2) / (2 * S2)   # (256, K)
        m_ = logits.max(axis=1, keepdims=True)
        lse = m_ + np.log(np.exp(logits - m_).sum(axis=1, keepdims=True))
        log_probs_s = logits - lse   # (256, K)
        # E_s[p(o=k|s)] = mean over samples; log E[p]
        log_mean_p = np.log(np.exp(log_probs_s[:, o]).mean() + 1e-300)
        nll -= log_mean_p
    return nll / len(held_words)


# ---------------------------------------------------------------------------
# Amortized encoder (JAX)
# ---------------------------------------------------------------------------

def init_params(key: jax.Array) -> list:
    """MLP: 6 -> 32 -> 32 -> 4. Returns list of (W, b) tuples."""
    keys = jax.random.split(key, 6)
    # Xavier / He-like init
    def layer(k, fan_in, fan_out):
        scale = np.sqrt(2.0 / fan_in)
        W = jax.random.normal(k, (fan_out, fan_in)) * scale
        b = jnp.zeros(fan_out)
        return (W, b)
    return [
        layer(keys[0], 6, 32),
        layer(keys[1], 32, 32),
        layer(keys[2], 32, 4),
    ]


def mlp_forward(params: list, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Forward pass: x (6,) -> (mu (2,), logvar (2,))."""
    h = x
    for i, (W, b) in enumerate(params[:-1]):
        h = jnp.tanh(W @ h + b)
    W, b = params[-1]
    out = W @ h + b   # (4,)
    mu = out[:2]
    logvar = out[2:]
    return mu, logvar


def kl_diagonal(mu: jax.Array, logvar: jax.Array) -> jax.Array:
    """KL(N(mu, diag(exp(logvar))) || N(0, 4I))."""
    # KL = 0.5 * sum [ exp(logvar)/4 + mu^2/4 - 1 - logvar + log(4) ]
    return 0.5 * jnp.sum(jnp.exp(logvar) / 4.0 + mu ** 2 / 4.0 - 1.0 - logvar + jnp.log(4.0))


def log_emission_jax(s: jax.Array) -> jax.Array:
    """log softmax over K emission probs at state s; shape (K,)."""
    mu_hex_j = jnp.array(MU_HEX)
    diff = s[None, :] - mu_hex_j   # (K, 2)
    logits = -jnp.sum(diff ** 2, axis=1) / (2 * S2)
    return jax.nn.log_softmax(logits)


def elbo_loss_single(params: list, counts: jax.Array, eps: jax.Array) -> jax.Array:
    """ELBO loss for one stream (negative ELBO / T).

    counts: (K,) raw word counts
    eps: (ELBO_SAMPLES, 2) standard normal noise for reparameterization
    """
    mu_enc, logvar_enc = mlp_forward(params, counts / T_STREAM)
    std = jnp.exp(0.5 * logvar_enc)
    s_samples = mu_enc[None, :] + std[None, :] * eps   # (S, 2)

    # Reconstruction: mean over samples of sum_k c_k * log p(o=k | s_i)
    # = mean_i [ sum_k counts[k] * log_softmax_k(s_i) ]
    def recon_one(s):
        lp = log_emission_jax(s)   # (K,)
        return jnp.sum(counts * lp)

    recon = jnp.mean(vmap(recon_one)(s_samples))
    kl = kl_diagonal(mu_enc, logvar_enc)
    return -(recon - kl)


def elbo_loss_batch(params: list, counts_batch: jax.Array, eps_batch: jax.Array) -> jax.Array:
    """Mean ELBO loss over a batch."""
    losses = vmap(elbo_loss_single, in_axes=(None, 0, 0))(params, counts_batch, eps_batch)
    return jnp.mean(losses)


@jit
def update_step(params, opt_state, counts_batch, eps_batch):
    """One Adam step. Returns (new_params, new_opt_state, loss)."""
    loss, grads = jax.value_and_grad(elbo_loss_batch)(params, counts_batch, eps_batch)

    # Hand-written Adam update
    new_params = []
    new_opt_state = []
    for i, ((W, b), (mW, mb), (vW, vb), t) in enumerate(
        zip(params, opt_state['m'], opt_state['v'], opt_state['t'])
    ):
        t_new = t + 1
        # W
        gW = grads[i][0]
        mW_new = B1 * mW + (1 - B1) * gW
        vW_new = B2 * vW + (1 - B2) * gW ** 2
        mW_hat = mW_new / (1 - B1 ** t_new)
        vW_hat = vW_new / (1 - B2 ** t_new)
        W_new = W - LR * mW_hat / (jnp.sqrt(vW_hat) + ADAM_EPS)
        # b
        gb = grads[i][1]
        mb_new = B1 * mb + (1 - B1) * gb
        vb_new = B2 * vb + (1 - B2) * gb ** 2
        mb_hat = mb_new / (1 - B1 ** t_new)
        vb_hat = vb_new / (1 - B2 ** t_new)
        b_new = b - LR * mb_hat / (jnp.sqrt(vb_hat) + ADAM_EPS)
        new_params.append((W_new, b_new))
        new_opt_state.append((mW_new, mb_new, vW_new, vb_new, t_new))

    new_opt = {
        'm': [(s[0], s[1]) for s in new_opt_state],
        'v': [(s[2], s[3]) for s in new_opt_state],
        't': [s[4] for s in new_opt_state],
    }
    return new_params, new_opt, loss


def init_adam(params):
    """Initialize Adam optimizer state."""
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    t = [jnp.array(0) for _ in params]
    return {'m': m, 'v': v, 't': t}


def train_encoder(seed: int, np_rng: np.random.Generator) -> list:
    """Train the amortized encoder for TRAIN_STEPS steps."""
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    opt_state = init_adam(params)

    for step in range(TRAIN_STEPS):
        # Sample batch of streams from true generative model
        s_batch = np_rng.multivariate_normal(PRIOR_MU, PRIOR_SIGMA, size=BATCH_SIZE)  # (B, 2)
        counts_batch = np.zeros((BATCH_SIZE, K), dtype=np.float32)
        for b in range(BATCH_SIZE):
            words = sample_stream(s_batch[b], T_STREAM, np_rng)
            for w in words:
                counts_batch[b, w] += 1.0

        # Reparameterization noise
        key, eps_key = jax.random.split(key)
        eps_batch = jax.random.normal(eps_key, (BATCH_SIZE, ELBO_SAMPLES, 2))

        counts_j = jnp.array(counts_batch)
        params, opt_state, loss = update_step(params, opt_state, counts_j, eps_batch)

        if (step + 1) % 500 == 0:
            print(f"  [seed {seed}] step {step+1}/{TRAIN_STEPS}, ELBO loss: {float(loss):.4f}")

    return params


def amortized_posterior(params: list, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_enc, logvar_enc) for a stream of words."""
    counts = np.zeros(K, dtype=np.float32)
    for w in words:
        counts[w] += 1.0
    counts_j = jnp.array(counts)
    mu_j, logvar_j = mlp_forward(params, counts_j / T_STREAM)
    return np.array(mu_j), np.array(logvar_j)


def mc_predictive_nll_diag(mu: np.ndarray, logvar: np.ndarray, held_words: np.ndarray,
                             rng: np.random.Generator) -> float:
    """MC predictive NLL using diagonal Gaussian q(s)."""
    std = np.exp(0.5 * logvar)
    eps = rng.standard_normal((MC_SAMPLES, 2))
    samples = mu[None, :] + std[None, :] * eps   # (256, 2)
    nll = 0.0
    for o in held_words:
        diff = samples[:, None, :] - MU_HEX[None, :, :]  # (256, K, 2)
        logits = -np.sum(diff ** 2, axis=2) / (2 * S2)   # (256, K)
        m_ = logits.max(axis=1, keepdims=True)
        lse = m_ + np.log(np.exp(logits - m_).sum(axis=1, keepdims=True))
        log_probs_s = logits - lse   # (256, K)
        log_mean_p = np.log(np.exp(log_probs_s[:, o]).mean() + 1e-300)
        nll -= log_mean_p
    return nll / len(held_words)


# ---------------------------------------------------------------------------
# Stream generation for eval
# ---------------------------------------------------------------------------

def generate_indist_streams(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate n in-distribution (s ~ prior) eval streams."""
    streams = []
    for _ in range(n):
        s = sample_prior(rng)
        obs_words = sample_stream(s, T_STREAM, rng)
        held_words = sample_stream(s, T_STREAM, rng)
        streams.append({'s_true': s, 'obs_words': obs_words, 'held_words': held_words,
                         'blend': False})
    return streams


def generate_blend_streams(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate n blend streams (words 0 and 3, 50/50). Target = midpoint = (0,0)."""
    # word 0: mu = (1, 0), word 3: mu = (-1, 0), midpoint = (0, 0)
    s_target = np.array([0.0, 0.0])
    streams = []
    for _ in range(n):
        # Observation words: 50/50 from word 0 and word 3
        obs_words = np.array([0 if rng.random() < 0.5 else 3 for _ in range(T_STREAM)])
        held_words = np.array([0 if rng.random() < 0.5 else 3 for _ in range(T_STREAM)])
        streams.append({'s_true': s_target.copy(), 'obs_words': obs_words,
                         'held_words': held_words, 'blend': True})
    return streams


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_stream(stream: dict, enc_params: list,
                eval_rng: np.random.Generator) -> dict:
    """Evaluate all three engines on one stream. Returns dict of metrics."""
    obs_words = stream['obs_words']
    held_words = stream['held_words']
    s_true = stream['s_true']

    # ---- Grid-exact ----
    t0 = time.perf_counter()
    log_post_grid = grid_posterior(obs_words)
    mu_grid = grid_posterior_mean(log_post_grid)
    nll_exact = grid_predictive_nll(log_post_grid, held_words)
    loc_exact = float(np.linalg.norm(mu_grid - s_true))
    t_exact = time.perf_counter() - t0

    # ---- Closed-form ----
    t0 = time.perf_counter()
    mu_cf, Sigma_cf = closed_form_posterior(obs_words)
    nll_closed = mc_predictive_nll(mu_cf, Sigma_cf, held_words, eval_rng)
    loc_closed = float(np.linalg.norm(mu_cf - s_true))
    t_closed = time.perf_counter() - t0

    # ---- Amortized ----
    t0 = time.perf_counter()
    mu_enc, logvar_enc = amortized_posterior(enc_params, obs_words)
    nll_amort = mc_predictive_nll_diag(mu_enc, logvar_enc, held_words, eval_rng)
    loc_amort = float(np.linalg.norm(mu_enc - s_true))
    t_amort = time.perf_counter() - t0

    return {
        'nll_exact': nll_exact, 'loc_exact': loc_exact, 't_exact': t_exact,
        'nll_closed': nll_closed, 'loc_closed': loc_closed, 't_closed': t_closed,
        'nll_amort': nll_amort, 'loc_amort': loc_amort, 't_amort': t_amort,
        'blend': stream['blend'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_seed(seed: int) -> dict:
    """Full run for one seed: train encoder, eval on in-dist + blend streams."""
    print(f"\n=== Seed {seed} ===")
    # Master numpy rng — deterministic given seed
    np_rng = np.random.default_rng(seed)

    # Train encoder
    print(f"  Training encoder (seed {seed})...")
    t_train_start = time.perf_counter()
    enc_params = train_encoder(seed, np_rng)
    t_train = time.perf_counter() - t_train_start
    print(f"  Encoder trained in {t_train:.1f}s")

    # Generate eval streams (from same numpy rng, so deterministic)
    indist_streams = generate_indist_streams(N_INDIST, np_rng)
    blend_streams = generate_blend_streams(N_BLEND, np_rng)

    # Evaluate — use fresh sub-rng seeded from seed for reproducibility
    eval_rng = np.random.default_rng(seed + 10000)

    results_indist = []
    for i, s in enumerate(indist_streams):
        r = eval_stream(s, enc_params, eval_rng)
        results_indist.append(r)

    results_blend = []
    for i, s in enumerate(blend_streams):
        r = eval_stream(s, enc_params, eval_rng)
        results_blend.append(r)

    # Cell means
    def mean_field(results, field):
        return float(np.mean([r[field] for r in results]))

    nll_exact_indist = mean_field(results_indist, 'nll_exact')
    nll_closed_indist = mean_field(results_indist, 'nll_closed')
    nll_amort_indist = mean_field(results_indist, 'nll_amort')
    loc_exact_indist = mean_field(results_indist, 'loc_exact')
    loc_closed_indist = mean_field(results_indist, 'loc_closed')
    loc_amort_indist = mean_field(results_indist, 'loc_amort')

    nll_exact_blend = mean_field(results_blend, 'nll_exact')
    nll_closed_blend = mean_field(results_blend, 'nll_closed')
    nll_amort_blend = mean_field(results_blend, 'nll_amort')
    loc_exact_blend = mean_field(results_blend, 'loc_exact')
    loc_closed_blend = mean_field(results_blend, 'loc_closed')
    loc_amort_blend = mean_field(results_blend, 'loc_amort')

    # Timing medians
    t_closed_ms = float(np.median([r['t_closed'] for r in results_indist + results_blend]) * 1000)
    t_amort_ms = float(np.median([r['t_amort'] for r in results_indist + results_blend]) * 1000)

    gap_nll_indist = nll_amort_indist - nll_closed_indist
    gap_nll_exact_indist = nll_closed_indist - nll_exact_indist
    gap_nll_blend = nll_amort_blend - nll_closed_blend

    print(f"  In-dist  NLL: exact={nll_exact_indist:.4f} closed={nll_closed_indist:.4f} "
          f"amort={nll_amort_indist:.4f}  "
          f"gap(amort-closed)={gap_nll_indist:+.4f}  gap(closed-exact)={gap_nll_exact_indist:+.4f}")
    print(f"  Blend    NLL: exact={nll_exact_blend:.4f} closed={nll_closed_blend:.4f} "
          f"amort={nll_amort_blend:.4f}  gap(amort-closed)={gap_nll_blend:+.4f}")
    print(f"  Loc in-dist: exact={loc_exact_indist:.4f} closed={loc_closed_indist:.4f} "
          f"amort={loc_amort_indist:.4f}")
    print(f"  Timing: closed={t_closed_ms:.2f}ms amort={t_amort_ms:.2f}ms per stream")

    return {
        'seed': seed,
        'nll_exact_indist': nll_exact_indist,
        'nll_closed_indist': nll_closed_indist,
        'nll_amort_indist': nll_amort_indist,
        'loc_exact_indist': loc_exact_indist,
        'loc_closed_indist': loc_closed_indist,
        'loc_amort_indist': loc_amort_indist,
        'nll_exact_blend': nll_exact_blend,
        'nll_closed_blend': nll_closed_blend,
        'nll_amort_blend': nll_amort_blend,
        'loc_exact_blend': loc_exact_blend,
        'loc_closed_blend': loc_closed_blend,
        'loc_amort_blend': loc_amort_blend,
        'gap_nll_indist': gap_nll_indist,
        'gap_nll_exact_indist': gap_nll_exact_indist,
        'gap_nll_blend': gap_nll_blend,
        'closed_inf_ms': t_closed_ms,
        'amort_inf_ms': t_amort_ms,
        'train_word_samples': TRAIN_WORD_SAMPLES,
    }


def main():
    print("Exp 138 — continuous-substrate rung 6: amortized control")
    print(f"Training: {TRAIN_STEPS} steps x {BATCH_SIZE} batch x {T_STREAM} words = "
          f"{TRAIN_WORD_SAMPLES:,} word-samples per seed")
    print(f"Closed-form training: 0 word-samples")
    print(f"Cost ratio: inf (closed-form has zero training cost)")
    print(f"Absolute amortized cost meets >=1000x threshold: "
          f"{TRAIN_WORD_SAMPLES} >= {1000 * T_STREAM} => "
          f"{'PASS' if TRAIN_WORD_SAMPLES >= 1000 * T_STREAM else 'FAIL'}")
    print()

    all_results = []
    for seed in range(N_SEEDS):
        r = run_seed(seed)
        all_results.append(r)

    # Aggregate cell means over seeds
    def cmean(field):
        return float(np.mean([r[field] for r in all_results]))

    print("\n" + "=" * 80)
    print("CELL MEANS (over 8 seeds)")
    print("=" * 80)
    print(f"In-distribution NLL:")
    print(f"  Exact:   {cmean('nll_exact_indist'):.4f}")
    print(f"  Closed:  {cmean('nll_closed_indist'):.4f}")
    print(f"  Amort:   {cmean('nll_amort_indist'):.4f}")
    print(f"  gap(amort-closed): {cmean('gap_nll_indist'):+.4f}")
    print(f"  gap(closed-exact): {cmean('gap_nll_exact_indist'):+.4f}")
    print(f"Blend NLL:")
    print(f"  Exact:   {cmean('nll_exact_blend'):.4f}")
    print(f"  Closed:  {cmean('nll_closed_blend'):.4f}")
    print(f"  Amort:   {cmean('nll_amort_blend'):.4f}")
    print(f"  gap(amort-closed): {cmean('gap_nll_blend'):+.4f}")
    print(f"In-distribution Localization:")
    print(f"  Exact:   {cmean('loc_exact_indist'):.4f}")
    print(f"  Closed:  {cmean('loc_closed_indist'):.4f}")
    print(f"  Amort:   {cmean('loc_amort_indist'):.4f}")
    print(f"Timing (median per stream):")
    print(f"  Closed: {cmean('closed_inf_ms'):.2f}ms   Amort: {cmean('amort_inf_ms'):.2f}ms")

    # Per-seed table
    print("\n" + "=" * 80)
    print("PER-SEED TABLE")
    print("=" * 80)
    print(f"{'Seed':>4}  {'NLL_close':>9}  {'NLL_amort':>9}  {'NLL_exact':>9}  "
          f"{'gap(a-c)':>9}  {'gap(c-e)':>9}  {'gap_blend':>9}")
    for r in all_results:
        print(f"{r['seed']:>4}  {r['nll_closed_indist']:>9.4f}  {r['nll_amort_indist']:>9.4f}  "
              f"{r['nll_exact_indist']:>9.4f}  {r['gap_nll_indist']:>+9.4f}  "
              f"{r['gap_nll_exact_indist']:>+9.4f}  {r['gap_nll_blend']:>+9.4f}")

    # P-tallies
    print("\n" + "=" * 80)
    print("PREDICTION TALLIES")
    print("=" * 80)

    # P1: structural / apples-to-apples
    p1 = True
    print(f"P1 (apples-to-apples): all three engines on identical streams, one model, "
          f"one metric -> PASS (structural)")

    # P2: accuracy conjuncts
    abs_gap_amort_closed = abs(cmean('gap_nll_indist'))
    gap_closed_exact = cmean('gap_nll_exact_indist')
    loc_closed_vs_amort = cmean('loc_closed_indist') - cmean('loc_amort_indist')
    p2_c1 = abs_gap_amort_closed <= 0.10
    p2_c2 = gap_closed_exact <= 0.10
    p2_c3 = loc_closed_vs_amort <= 0.05
    p2 = p2_c1 and p2_c2 and p2_c3
    print(f"P2 (accuracy):")
    print(f"  |NLL_amort - NLL_closed| = {abs_gap_amort_closed:.4f} <= 0.10? {'PASS' if p2_c1 else 'FAIL'}")
    print(f"  NLL_closed - NLL_exact  = {gap_closed_exact:.4f} <= 0.10? {'PASS' if p2_c2 else 'FAIL'}")
    print(f"  loc_closed - loc_amort  = {loc_closed_vs_amort:+.4f} <= 0.05? {'PASS' if p2_c3 else 'FAIL'}")
    print(f"  P2 overall: {'PASS' if p2 else 'FAIL'}")

    # P3: cost
    train_count = TRAIN_WORD_SAMPLES
    threshold = 1000 * T_STREAM
    p3_c1 = train_count >= threshold
    closed_inf_med = cmean('closed_inf_ms')
    amort_inf_med = cmean('amort_inf_ms')
    p3 = p3_c1  # timing is report-only per spec
    print(f"P3 (cost):")
    print(f"  Amortized training word-samples: {train_count:,}")
    print(f"  Closed-form training word-samples: 0")
    print(f"  Cost ratio: inf (closed-form: zero training cost)")
    print(f"  1000x threshold: {train_count:,} >= {threshold:,}? {'PASS' if p3_c1 else 'FAIL'}")
    print(f"  Inference: closed={closed_inf_med:.2f}ms amort={amort_inf_med:.2f}ms (report-only)")
    print(f"  P3 overall: {'PASS' if p3 else 'FAIL'}")

    # P4: blend in-distribution
    abs_blend_gap = abs(cmean('gap_nll_blend'))
    p4 = abs_blend_gap <= 0.15
    print(f"P4 (blends in-distribution):")
    print(f"  |NLL_amort_blend - NLL_closed_blend| = {abs_blend_gap:.4f} <= 0.15? {'PASS' if p4 else 'FAIL'}")
    print(f"  P4 overall: {'PASS' if p4 else 'FAIL'}")

    # Falsifiers
    gap_amort_closed_signed = cmean('gap_nll_indist')  # amort - closed
    falsifier_amort_beats = gap_amort_closed_signed < -0.10   # amort better by > 0.10
    falsifier_amort_worse = gap_amort_closed_signed > 0.30    # amort worse by > 0.30
    falsifier_blend_asym = abs(cmean('gap_nll_blend')) > 0.30
    falsifier_structural = False  # P1 structural pass

    any_falsifier = falsifier_amort_beats or falsifier_amort_worse or falsifier_blend_asym or falsifier_structural

    print("\n" + "=" * 80)
    print("FALSIFIERS")
    print("=" * 80)
    print(f"  amort beats closed by > 0.10 nats in-dist: "
          f"{gap_amort_closed_signed:.4f} < -0.10 -> {'TRIGGERED' if falsifier_amort_beats else 'clear'}")
    if falsifier_amort_beats:
        print(f"    FINDING: conjugacy mismatch has real predictive cost")
    print(f"  amort worse than closed by > 0.30 nats:    "
          f"{gap_amort_closed_signed:.4f} > 0.30 -> {'TRIGGERED' if falsifier_amort_worse else 'clear'}")
    if falsifier_amort_worse:
        print(f"    FINDING: encoder cannot fit at toy scale")
    print(f"  blend asymmetry > 0.30 either way:         "
          f"|{cmean('gap_nll_blend'):.4f}| > 0.30 -> {'TRIGGERED' if falsifier_blend_asym else 'clear'}")
    print(f"  structural FAIL (comparison impossible):    "
          f"{'TRIGGERED' if falsifier_structural else 'clear'}")

    # Verdict (three-way rule)
    all_predictions = [p1, p2, p3, p4]
    if any_falsifier:
        verdict = "NEGATIVE"
        reason = []
        if falsifier_amort_beats:
            reason.append(f"amort beats closed by {-gap_amort_closed_signed:.4f} nats (conjugacy mismatch has predictive cost)")
        if falsifier_amort_worse:
            reason.append(f"amort worse than closed by {gap_amort_closed_signed:.4f} nats (cannot fit)")
        if falsifier_blend_asym:
            reason.append(f"blend asymmetry {abs(cmean('gap_nll_blend')):.4f} > 0.30")
        verdict_detail = "; ".join(reason)
    elif all(all_predictions):
        verdict = "POSITIVE"
        verdict_detail = "all conjuncts P1-P4 passed, no falsifiers triggered"
    else:
        verdict = "MIXED"
        failed = []
        if not p2: failed.append("P2")
        if not p3: failed.append("P3")
        if not p4: failed.append("P4")
        verdict_detail = f"failed: {', '.join(failed)}; no falsifiers triggered"

    print("\n" + "=" * 80)
    print(f"VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print("=" * 80)

    # JSON output
    os.makedirs("experiments/outputs", exist_ok=True)
    rows = []
    for r in all_results:
        for agent, nll_indist, nll_blend, loc_indist, loc_blend in [
            ('closed', r['nll_closed_indist'], r['nll_closed_blend'],
             r['loc_closed_indist'], r['loc_closed_blend']),
            ('amortized', r['nll_amort_indist'], r['nll_amort_blend'],
             r['loc_amort_indist'], r['loc_amort_blend']),
            ('exact', r['nll_exact_indist'], r['nll_exact_blend'],
             r['loc_exact_indist'], r['loc_exact_blend']),
        ]:
            rows.append({
                'seed': r['seed'],
                'agent': agent,
                'nll_indist': nll_indist,
                'nll_blend': nll_blend,
                'loc_indist': loc_indist,
                'loc_blend': loc_blend,
                'train_word_samples': r['train_word_samples'] if agent == 'amortized' else 0,
                'inference_ms': (r['amort_inf_ms'] if agent == 'amortized'
                                 else r['closed_inf_ms'] if agent == 'closed'
                                 else None),
            })
    with open("experiments/outputs/exp138_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON rows written to experiments/outputs/exp138_rows.json")


if __name__ == "__main__":
    main()
