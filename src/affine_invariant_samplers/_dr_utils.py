"""Stable log probabilities shared by delayed-rejection samplers."""

import jax.numpy as jnp


def _safe_log1m_exp(log_alpha):
    """Return log(1 - exp(log_alpha)) for log_alpha <= 0.

    ``expm1`` preserves rejection probabilities when acceptance is close to
    one. Exact acceptance one has exact rejection zero, represented by -inf.
    """
    log_alpha = jnp.asarray(log_alpha)
    return jnp.where(
        log_alpha > -jnp.log(2.),
        jnp.log(-jnp.expm1(log_alpha)),
        jnp.log1p(-jnp.exp(log_alpha)),
    )


def _log_accept_ratio(log_num, log_den):
    """Return min(0, log_num - log_den), including zero path densities.

    A zero forward density describes an unreachable rejection history; its
    acceptance can be set to one. A zero reverse density on a reachable
    history instead requires rejection. Invalid (NaN) densities reject.
    """
    log_num, log_den = jnp.asarray(log_num), jnp.asarray(log_den)
    log_alpha = jnp.minimum(0., log_num - log_den)
    log_alpha = jnp.where(jnp.isneginf(log_den), 0., log_alpha)
    invalid = jnp.isnan(log_num) | jnp.isnan(log_den) | jnp.isnan(log_alpha)
    return jnp.where(invalid, -jnp.inf, log_alpha)
