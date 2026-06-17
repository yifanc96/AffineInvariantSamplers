"""Gravitational-wave population inference benchmark.

Hierarchical Bayesian inference on a synthetic LIGO/Virgo O3-like catalog.
The population-level log-likelihood is the importance-sampling form used
by `gwpopulation` (Talbot & Thrane 2018, Abbott+ GWTC-3 2023):

    log p(data | Λ) = Σ_i log [ (1/N_i) Σ_j p_pop(θ_{ij} | Λ) / π(θ_{ij}) ]
                      − N_events · log Vol_sel(Λ)          (selection term)

where ``Λ`` are the population hyper-parameters, ``θ_{ij}`` are single-event
posterior samples (MC draws that replace the hard per-event likelihood
integral), ``π`` is the per-event prior the samples were drawn under,
and the selection term accounts for the detection probability at fixed Λ.

For reproducibility we synthesise the per-event posterior samples from a
known ground-truth population; switching to real GWTC-3 posterior samples
(publicly released as H5 files) is a one-line swap in ``_make_synthetic``.

Population model: Power-Law + Peak (Talbot & Thrane 2018, the canonical
O3 primary-mass model) with ``α, β, m_min, m_max, μ_pp, σ_pp, λ_pp, δ_m``
— 8 hyperparameters, which puts the problem at d=8 but with strong
multimodality and long-tailed posteriors that break stretch.  Extensions
to spin (adds 2–4 dims) are in-scope and left as a comment.

References
----------
- Talbot & Thrane 2018 (arXiv:1801.02699) — Power-Law+Peak model.
- Abbott+ GWTC-3 population paper (arXiv:2111.03634) — likelihood form.
- `gwpopulation` (Romero-Shaw+ 2020) — reference implementation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# Ground-truth population (matches GWTC-3 Table II central values, roughly).
_TRUTH = dict(alpha=3.5, beta=1.1, m_min=5.0, m_max=87.0,
              mu_pp=34.0, sigma_pp=3.6, lambda_pp=0.04, delta_m=4.8)


# ─── Power-Law + Peak mass distribution ──────────────────────────────────────

def _smoothing(m, m_min, delta_m):
    """Talbot & Thrane smoothing window S(m | m_min, δ_m).  Eq (1) of
    arXiv:1801.02699.  Zero below m_min, 1 above m_min+δ, C^∞ between.

    Gradient-safe implementation — the boundary divisions blow up at the
    branch endpoints, which poisons JAX's reverse-mode gradient through
    the non-selected branch.  Fix: substitute a dummy-but-safe input in
    the non-ramp region so the inner exp() never sees a boundary value,
    then pick the correct output with `jnp.where` at the end (the "double
    where" trick — jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan)."""
    delta_safe = jnp.maximum(delta_m, 1e-6)
    in_ramp = (m > m_min) & (m < m_min + delta_safe)

    # Replace non-ramp inputs with a safe midpoint so autodiff never sees
    # a division by zero in the exp(·), even though that output will be
    # masked out below.
    m_safe = jnp.where(in_ramp, m, m_min + 0.5 * delta_safe)
    z = (m_safe - m_min) / delta_safe                      # ∈ (0, 1) strictly
    # f(z) = 1 / (1 + exp(−1/z − 1/(z−1)))  — Talbot & Thrane form.
    arg = -1.0 / z - 1.0 / (z - 1.0)
    f   = jax.nn.sigmoid(arg)
    out = jnp.where(m <= m_min, 0.0,
            jnp.where(m >= m_min + delta_safe, 1.0, f))
    return out


def _plp_mass_pdf(m1, m2, alpha, beta, m_min, m_max, mu_pp, sigma_pp,
                  lambda_pp, delta_m):
    """p(m1, m2 | Λ) = p(m1|Λ) · p(m2 | m1, β, m_min).  Not renormalised
    exactly — we absorb the partition function into the hyperprior via
    ``Vol_sel`` (population-level normalisation); for synthetic data with
    equal detection probability this factors out.

    Uses the "double where" pattern for gradient safety: the raw `m1**(-α)`
    and `m2**β` would see `log(m)` in the gradient, which can produce NaN
    if the non-selected branch gets a pathological value.  We substitute
    a known-safe input in those branches."""
    m1_ok = (m1 >= m_min) & (m1 <= m_max)
    m2_ok = (m2 >= m_min) & (m2 <= m1)

    m1_safe = jnp.where(m1_ok, m1, 10.0)     # dummy in [m_min, m_max]
    m2_safe = jnp.where(m2_ok, m2, 10.0)

    pl    = jnp.where(m1_ok, m1_safe ** (-alpha), 0.0)
    gauss = jnp.exp(-0.5 * ((m1 - mu_pp) / sigma_pp) ** 2) / (sigma_pp * jnp.sqrt(2 * jnp.pi))
    mix_m1 = (1.0 - lambda_pp) * pl + lambda_pp * gauss
    mix_m1 = mix_m1 * _smoothing(m1, m_min, delta_m)

    m2_pl = jnp.where(m2_ok, m2_safe ** beta, 0.0)
    m2_pl = m2_pl * _smoothing(m2, m_min, delta_m)

    return mix_m1 * m2_pl


def _make_synthetic(n_events: int = 70, n_samples_per_event: int = 2000,
                    seed: int = 0):
    """Draw per-event posterior samples from the true population.

    For each event, we draw ``n_samples_per_event`` (m1, m2) points from
    the true population distribution, then perturb by a small per-event
    measurement covariance to mimic the posterior spread of real GW
    events (O3 typical: ~10-20% fractional mass uncertainty).

    Returns an (n_events, n_samples, 2) array of (m1, m2) samples and
    a matching (n_events, n_samples) array of per-sample prior weights
    π(θ) — uniform on (m1, m2) in (5, 100)² for this toy.
    """
    rng = np.random.default_rng(seed)
    # Ancestral draws from true Λ via rejection sampling on a wide box.
    m_grid = np.linspace(5, 100, 500)
    M1, M2 = np.meshgrid(m_grid, m_grid, indexing="ij")
    valid  = M2 <= M1
    w      = np.asarray(_plp_mass_pdf(jnp.asarray(M1), jnp.asarray(M2),
                                       **_TRUTH)).copy()
    w[~valid] = 0.0
    w /= w.sum()
    flat_idx = rng.choice(w.size, size=n_events, p=w.ravel())
    m1_true = M1.ravel()[flat_idx]
    m2_true = M2.ravel()[flat_idx]

    # Synthetic per-event posterior: Gaussian around the true masses with
    # fractional width ≈ 10%.  (Real O3 posteriors are more structured, but
    # the MC-integral-over-samples likelihood form is identical.)
    sigma1 = 0.12 * m1_true
    sigma2 = 0.12 * m2_true
    s1 = m1_true[:, None] + sigma1[:, None] * rng.standard_normal(
        (n_events, n_samples_per_event))
    s2 = m2_true[:, None] + sigma2[:, None] * rng.standard_normal(
        (n_events, n_samples_per_event))
    # Clip to physical box (crude rejection substitute).
    s1 = np.clip(s1, 5.0, 100.0)
    s2 = np.clip(np.minimum(s2, s1), 3.0, 100.0)

    samples = np.stack([s1, s2], axis=-1).astype(np.float32)    # (E, S, 2)
    # Uniform "prior" on (5, 100) × (3, 100):  π = 1/(95 · 97) per sample.
    prior = np.full((n_events, n_samples_per_event), 1.0 / (95.0 * 97.0),
                     dtype=np.float32)
    return samples, prior


def make_gw_population(n_events: int = 70, n_samples_per_event: int = 2000,
                        seed: int = 0):
    """Hierarchical GW population log-posterior — 8-D.

    θ = (α, β, m_min, m_max, μ_pp, σ_pp, λ_pp, δ_m)

    with priors:

        α        ∈ [−4, 12]       (primary-mass power-law slope)
        β        ∈ [−4, 12]       (mass-ratio slope)
        m_min    ∈ [2, 10]        (M_☉)
        m_max    ∈ [50, 100]      (M_☉)
        μ_pp     ∈ [20, 50]       (Gaussian peak location)
        σ_pp     ∈ [1, 10]        (peak width)
        λ_pp     ∈ [0, 1]         (peak fraction)
        δ_m      ∈ [0, 10]        (low-mass smoothing)

    The per-event log-likelihood uses the MC-integral form
    log (1/S Σ_s p_pop(θ_s|Λ)/π(θ_s)).  For this synthetic benchmark the
    selection function is flat (detection probability = 1 across the box),
    so the ``log Vol_sel`` term is a Λ-independent constant and dropped.
    """
    samples_np, prior_np = _make_synthetic(n_events, n_samples_per_event, seed)
    samples = jnp.asarray(samples_np)          # (E, S, 2)
    prior   = jnp.asarray(prior_np)            # (E, S)
    log_prior = jnp.log(prior)                 # (E, S)

    LO = jnp.asarray([-4.0, -4.0,  2.0, 50.0, 20.0,  1.0, 0.0,  0.0])
    HI = jnp.asarray([12.0, 12.0, 10.0,100.0, 50.0, 10.0, 1.0, 10.0])

    def _one_lp(theta):
        alpha, beta, m_min, m_max, mu_pp, sigma_pp, lambda_pp, delta_m = theta
        m1 = samples[..., 0]
        m2 = samples[..., 1]
        p  = _plp_mass_pdf(m1, m2, alpha, beta, m_min, m_max,
                           mu_pp, sigma_pp, lambda_pp, delta_m)
        # Double-where: log only touches strictly positive values.  Zeros
        # still contribute −∞ to logsumexp (so the primal is unchanged),
        # but their gradients are masked to 0 to avoid the 0 · (1/0) NaN.
        pos = p > 0
        p_safe = jnp.where(pos, p, 1.0)
        log_p  = jnp.where(pos, jnp.log(p_safe), -jnp.inf)
        log_weights = log_p - log_prior                 # (E, S)
        per_event = jax.scipy.special.logsumexp(log_weights, axis=1) - jnp.log(samples.shape[1])
        return jnp.sum(per_event)

    def log_prob(theta):
        in_box = jnp.all((theta >= LO) & (theta <= HI), axis=1)
        m_max_gt_min = theta[:, 3] > theta[:, 2]
        valid = in_box & m_max_gt_min
        lp = jax.vmap(_one_lp)(theta)
        return jnp.where(valid, lp, -jnp.inf)

    info = dict(dim=8, n_data=n_events,
                n_samples_per_event=n_samples_per_event,
                param_names=["alpha", "beta", "m_min", "m_max",
                              "mu_pp", "sigma_pp", "lambda_pp", "delta_m"],
                prior_lo=np.asarray(LO), prior_hi=np.asarray(HI),
                truth={k: float(v) for k, v in _TRUTH.items()})
    return log_prob, info


def init_gw_population(key, n_chains: int):
    """Small ball around the GWTC-3-central truth values."""
    mean  = jnp.asarray([_TRUTH["alpha"], _TRUTH["beta"],
                         _TRUTH["m_min"], _TRUTH["m_max"],
                         _TRUTH["mu_pp"], _TRUTH["sigma_pp"],
                         _TRUTH["lambda_pp"], _TRUTH["delta_m"]])
    scale = jnp.asarray([0.5, 0.5, 0.5, 2.0, 1.0, 0.5, 0.01, 0.5])
    return mean + scale * jax.random.normal(key, (n_chains, 8))
