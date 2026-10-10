"""Benchmark diagnostics must not reward frozen walkers or omit gradient work."""
import json

import numpy as np
import pytest

pytest.importorskip("arviz")

from benchmark import metrics
from benchmark.distributions import make_gaussian
from benchmark.run import json_safe, production_cost, production_grads, run_pkg


@pytest.mark.parametrize("fast", [False, True])
def test_frozen_target_matching_ensemble_has_zero_ess(fast):
    samples = np.tile(np.array([-1., 1., -1., 1.])[None, :, None], (100, 1, 1))
    result = metrics.all_metrics(samples, {"mean": [0.], "std": [1.]}, fast=fast)
    assert result["mean_err_sigma_max"] == 0
    assert result["std_rel_err_max"] == 0
    assert result["ess_bulk_min"] == 0
    assert result["ess_ensemble_mean_min"] == 0
    assert result["ess_ensemble_second_moment_min"] == 0
    assert result["unestimable_mean_dimensions"] == [0]
    assert np.isinf(result["iat_max"])
    assert np.isnan(result["bias_z_max"])
    assert json_safe(result)["bias_z_max"] is None


def test_constant_and_nonfinite_observables_do_not_have_positive_ess():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(200, 4, 3))
    samples[:, :, 1] = 3
    samples[5, 2, 2] = np.inf
    ess, tau = metrics.ensemble_ess(samples)
    assert ess[0] > 0
    assert np.array_equal(ess[1:], [0., 0.])
    assert np.all(np.isinf(tau[1:]))


def test_cross_walker_time_dependence_is_visible_to_ensemble_mean():
    # Each walker is iid and walkers are independent at any fixed time, yet
    # walker two is walker one's previous value: grand mean variance is doubled.
    z = np.random.default_rng(27).normal(size=20001)
    samples = np.stack((z[1:], z[:-1]), axis=1)[:, :, None]
    result = metrics.all_metrics(samples, {"mean": [0.], "std": [1.]})
    assert 15000 < result["ess_ensemble_mean_min"] < 25000
    assert result["ess_rank_bulk_min"] > 1.5 * result["ess_ensemble_mean_min"]
    assert result["ess_method"] == "arviz_rank"
    assert "sokal" in result["ess_ensemble_method"]


@pytest.mark.parametrize("sampler", ["peaches", "peams", "chess", "pickles", "aldi"])
def test_production_counter_precedes_mean_length(sampler):
    assert production_grads(sampler, {"n_grad_evals": 120, "mean_L": 2}, 10, 4) == 120


def test_peanuts_uses_production_not_warmup_counter():
    assert production_grads("peanuts", {"n_grad_evals": 500, "n_grad_evals_prod": 80}, 10, 4) == 80
    assert production_grads("peanuts", {"n_grad_evals": 500}, 10, 4) is None
    assert production_grads("peanuts", {"n_grad_evals_prod": 0}, 10, 4) == 0


def test_gradient_free_cost_is_log_density_work():
    cost = production_cost("stretch", {}, 10, 4)
    assert cost["n_grad_evals_prod"] is None
    assert cost["n_log_prob_evals_prod"] == 40
    assert cost["n_evals_prod"] == 40
    assert cost["evaluation_unit"] == "log_density"


def test_malt_cost_counts_cached_gradient_integrator():
    assert production_grads("malt", {"L": 3, "benchmark_effective_settings": {"thin_by": 2}}, 10, 4) == 240


def test_malt_receives_single_point_density(monkeypatch):
    from benchmark import run
    import jax.numpy as jnp
    def fake_sampler(log_prob_fn, initial_state, num_samples, warmup, seed, verbose=False):
        assert log_prob_fn(initial_state[0]).shape == ()
        return jnp.tile(initial_state[None], (num_samples, 1, 1)), {"L": 3}
    monkeypatch.setitem(run.PKG_SAMPLERS, "malt", fake_sampler)
    lp, _, _ = make_gaussian(2, log_cond=0)
    samples, info, _ = run_pkg("malt", lp, jnp.ones((4, 2)), 10, 0, 0)
    assert samples.shape == (10, 4, 2)
    assert info["benchmark_effective_settings"]["warmup"] == 0


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_gaussian_condition_metadata_preserves_target(dim):
    _, truth, meta = make_gaussian(dim, log_cond=6, rotate=False)
    expected = np.exp(np.linspace(-.5, .5, dim) * 6)
    assert np.allclose(truth["eigen_std"], expected)
    assert np.isclose(meta["kappa_cov"], (expected.max() / expected.min()) ** 2)
    assert meta["kappa"] == meta["kappa_cov"]
    assert np.isclose(meta["kappa_std"], expected.max() / expected.min())


def test_json_is_strict_and_unavailable_values_are_null():
    value = json_safe({"ess": np.inf, "missing": np.nan, "count": np.int64(5)})
    assert json.loads(json.dumps(value, allow_nan=False)) == {"ess": None, "missing": None, "count": 5}


def test_gradient_free_thinning_cost_includes_all_iterations():
    cost = production_cost("walk", {"benchmark_effective_settings": {"thin_by": 3}}, 10, 4)
    assert cost["n_log_prob_evals_prod"] == 120


def test_runner_records_full_metadata_and_preparation(tmp_path, monkeypatch):
    from benchmark import run
    import sys
    monkeypatch.setattr(sys, "argv", ["benchmark.run", "--target", "gaussian", "--dim", "2",
                                    "--sampler", "stretch", "--n-chains", "8", "--warmup", "2",
                                    "--num-samples", "16", "--log-cond", "0", "--warm-start", "1",
                                    "--fast-ess", "--out", str(tmp_path)])
    run.main()
    cell = json.loads((tmp_path / "metrics.json").read_text())
    assert cell["target_metadata"]["kappa_cov"] == 1
    assert cell["requested_settings"]["target_seed"] == 0
    assert cell["effective_sampler_settings"]["thin_by"] == 1
    assert cell["source_metadata"]["git_revision"]
    assert len(cell["source_metadata"]["source_sha256"]) == 64
    assert cell["source_metadata"]["library_versions"]["jax"]
    assert cell["preparation"]["iterations"] == 2
    assert cell["preparation"]["log_prob_evals"] == 16
    assert cell["wall_s_with_preparation"] >= cell["wall_s"]
    assert cell["n_grad_evals_prod"] is None
    assert cell["n_log_prob_evals_prod"] == 128
    assert "ess_per_grad" not in cell
    assert "ess_per_log_prob" in cell
    assert cell["metrics_num_draws"] == 16
