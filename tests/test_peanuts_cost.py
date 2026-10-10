"""PEANUTS cost counts uncached gradients at both ends of every tree leaf."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers import sampler_peanuts


@pytest.mark.parametrize('move', ['h-walk', 'h-side'])
@pytest.mark.parametrize('warmup,depth', [(0, 1), (2, 2)])
def test_peanuts_counts_actual_gradient_calls_in_production_and_warmup(move, warmup, depth):
    gradient_calls = []

    def record(points):
        gradient_calls.append(len(points))

    def gradient(points):
        jax.debug.callback(record, points, ordered=True)
        return -points

    initial = jax.random.normal(jax.random.key(7), (8, 2))
    draws, thinning = 4, 2
    samples, info = sampler_peanuts(
        lambda x: -.5 * jnp.sum(x*x, axis=-1), initial, draws,
        warmup=warmup, step_size=.001, max_tree_depth=depth, thin_by=thinning,
        move=move, grad_log_prob_fn=gradient, adapt_step_size=False,
        find_init_step_size=False, verbose=False)
    samples.block_until_ready()
    jax.effects_barrier()
    # These short, tiny-step Gaussian trees reach the chosen depth on every chain.
    leaves = 2**depth - 1
    assert info['mean_num_steps'] == leaves
    production_calls = 2 * leaves * draws * thinning * len(initial)
    warmup_calls = 2 * leaves * warmup * len(initial)
    assert sum(gradient_calls) == production_calls + warmup_calls
    assert info['n_grad_evals_prod'] == production_calls
    assert info['n_grad_evals'] == production_calls + warmup_calls
    assert np.all(np.isfinite(samples))
