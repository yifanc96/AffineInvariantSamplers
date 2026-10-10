"""Exercise notebook functions without executing its long plotting examples."""

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cholesky, solve_triangular


@pytest.fixture
def functions():
    path = (Path(__file__).parents[1] / 'other-samplers' / 'Delayed-rejection'
            / 'affine-invariant-samplers-delayed-rejection.ipynb')
    notebook = json.loads(path.read_text())
    namespace = dict(np=np, cholesky=cholesky, solve_triangular=solve_triangular)
    for index in (1, 3, 7):
        parsed = ast.parse(''.join(notebook['cells'][index]['source']))
        definitions = ast.Module(
            body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)],
            type_ignores=[])
        exec(compile(definitions, str(path), 'exec'), namespace)
    return namespace


def normal_density(y, x, sigma):
    return np.exp(-0.5 * ((y - x) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


@pytest.mark.parametrize('normal_std', [0.4, 1.0, 2.0])
def test_side_path_balance(functions, normal_std):
    x, y, z, s1, s2 = 0.4, 2.8, 0.9, 1.1, 0.45
    lp = lambda value: -0.5 * value**2
    alpha = functions['_side_dr_log_accept']

    def flux(start, end):
        la = alpha(lp(start), lp(y), lp(end), (y-start)/s1,
                   (end-start)/s2, s2/s1, normal_std)
        return (np.exp(lp(start)) * normal_density(y, start, s1*normal_std)
                * normal_density(end, start, s2*normal_std)
                * -np.expm1(min(0, lp(y)-lp(start))) * np.exp(la))

    forward, reverse = flux(x, z), flux(z, x)
    assert forward > 0
    assert forward == pytest.approx(reverse, rel=1e-12)


@pytest.mark.parametrize('dim', [1, 4])
def test_stretch_chat_example_path_balance(functions, dim):
    x, y, z, a1, a2 = 1.0, 1.8, 1.2, 2.0, 1.5
    lp = lambda radius: -radius**2
    alpha = functions['_stretch_dr_log_accept']

    def q(end, start, a):
        if not 1/a <= end/start <= a:
            return 0.0
        return 1/(2*(np.sqrt(a)-1/np.sqrt(a))*np.sqrt(start*end))

    def flux(start, end):
        first = min(0, lp(y)-lp(start)+(dim-1)*np.log(y/start))
        second = alpha(lp(start), lp(y), lp(end), y/start, end/start, dim, a1)
        return (np.exp(lp(start)) * start**(dim-1) * q(y, start, a1)
                * q(end, start, a2) * -np.expm1(first) * np.exp(second))

    forward, reverse = flux(x, z), flux(z, x)
    assert forward > 0
    assert forward == pytest.approx(reverse, rel=1e-12)


def test_exact_and_near_zero_rejections(functions):
    helper = functions['_safe_log1m_exp']
    assert np.isneginf(helper(0.0))
    assert helper(-np.inf) == 0.0
    assert helper(-1e-16) == pytest.approx(np.log(1e-16))
    ratio = functions['_log_accept_ratio']
    assert ratio(-np.inf, -np.inf) == 0.0  # unreachable forward path
    assert ratio(-np.inf, 0.0) == -np.inf
    assert ratio(0.0, -np.inf) == 0.0


def test_memoized_gn_near_zero_rejection(functions):
    hs = [1.0, 0.5]
    cache = {(0, 1, hs[0]): -1.0, (1, 0, hs[0]): -1.0-4e-13,
             (2, 1, hs[0]): -1.0, (1, 2, hs[0]): -1.0-2e-13,
             (0, 2, hs[1]): -1.0, (2, 0, hs[1]): -1.0}
    value = functions['compute_dr_acceptance_memoized'](
        0, 2, [1], [0., 0., 0.], [None]*3, [None]*3, [None]*3,
        hs, cache, {})
    assert value == pytest.approx(-np.log(2), abs=1e-3)


def test_memoized_gn_ordered_history_balance(functions):
    rng = np.random.default_rng(27)
    target = rng.dirichlet(np.ones(3))
    kernels = rng.dirichlet(np.ones(3), size=(4, 3))
    hs = [1., 2., 3., 4.]

    def transition(start, end, gradient, chol, step):
        return np.log(kernels[int(step)-1, int(start[0]), int(end[0])])

    functions['compute_transition_probability_chol'] = transition
    for depth in range(1, 5):
        for path in itertools.product(range(3), repeat=depth+1):
            def log_flux(sequence):
                pts = [np.array([point]) for point in sequence]
                lps = [np.log(target[point]) for point in sequence]
                cache, alphas = {}, {}
                weight = lps[0]
                for stage in range(1, depth+1):
                    weight += transition(pts[0], pts[stage], None, None, hs[stage-1])
                    alpha = functions['compute_dr_acceptance_memoized'](
                        0, stage, list(range(1, stage)), lps, pts,
                        [None]*len(pts), [None]*len(pts), hs, cache, alphas)
                    weight += (alpha if stage == depth
                               else functions['_safe_log1m_exp'](alpha))
                return np.exp(weight)

            paired = (path[-1], *path[1:-1], path[0])
            assert log_flux(path) == pytest.approx(log_flux(paired), abs=3e-17)


@pytest.mark.parametrize('name', ['side_move_with_twostage_dr',
                                'stretch_move_with_twostage_dr'])
def test_notebook_sampler_smoke(functions, name):
    np.random.seed(123)
    lp = lambda points: -0.5 * np.sum(points**2, axis=-1)
    kwargs = dict(normal_std=1.7) if name.startswith('side') else {}
    samples, acceptance, stages = functions[name](
        lp, np.array([0.5, -0.5]), 50, n_walkers=8, **kwargs)
    assert samples.shape == (8, 50, 2)
    assert np.all(np.isfinite(samples))
    assert np.all((acceptance >= 0) & (acceptance <= 1))
    assert np.all(stages['successes'] <= stages['attempts'])


@pytest.mark.parametrize('index', [3, 7])
def test_sampler_cells_run_without_other_sampler_definitions(index):
    path = (Path(__file__).parents[1] / 'other-samplers' / 'Delayed-rejection'
            / 'affine-invariant-samplers-delayed-rejection.ipynb')
    notebook = json.loads(path.read_text())
    parsed = ast.parse(''.join(notebook['cells'][index]['source']))
    definitions = ast.Module(
        body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[])
    namespace = dict(np=np, cholesky=cholesky, solve_triangular=solve_triangular)
    exec(compile(definitions, str(path), 'exec'), namespace)
    assert namespace['_safe_log1m_exp'](-1e-16) == pytest.approx(np.log(1e-16))
    if index == 3:
        actual = namespace['_stretch_dr_log_accept'](-1., -3.24, -1.44,
                                                     1.8, 1.2, 1, 2.)
    else:
        actual = namespace['compute_dr_acceptance_memoized'](
            0, 2, [1], [0., -2., -.5],
            [np.array([0.]), np.array([2.]), np.array([1.])],
            [np.zeros(1)] * 3, [np.eye(1)] * 3, [.5, .25], {}, {})
    assert np.isfinite(actual)
