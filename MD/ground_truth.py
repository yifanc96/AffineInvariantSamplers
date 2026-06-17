"""Load DEM ground-truth samples and expose them + their energies.

Origin: https://github.com/jarridrb/DEM/tree/main/data — .npy arrays of
shape (N, D).  DW-4 / LJ-13: 100 k train + 10 k test.  LJ-55: only
10 k test samples are published (``part1``).

The files use the same *equilibrium distance offset* and LJ ``r_m``
convention as our ``energies.py`` — verified by computing our
``E(x)`` on the arrays and confirming means/std match the paper's
reference (see ``MD/README.md`` table).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from MD.energies import dw4_energy, lj_energy

REF = Path(__file__).resolve().parent / "reference"

FILES = {
    "dw4":  dict(test="dw4_test.npy",  train="dw4_train.npy",   n_particles=4,  spatial_dim=2),
    "lj13": dict(test="lj13_test.npy", train="lj13_train.npy",  n_particles=13, spatial_dim=3,
                  pita_test="lj13_pita_test.npy"),
    "lj55": dict(test="lj55_test_part1.npy", train=None,         n_particles=55, spatial_dim=3,
                  pita_test="lj55_pita_test.npy"),
}


def load(problem: str, split: str = "test") -> np.ndarray:
    """Return (N, D) GT samples for ``problem`` ∈ {dw4, lj13, lj55}.

    ``split='pita_test'`` loads the numpyro-NUTS reference generated under
    pita's factor-of-2 formula at τ=1 (see ``MD/sample_pita_lj.py``).
    """
    fname = FILES[problem].get(split)
    if fname is None:
        raise FileNotFoundError(
            f"ground-truth {split!r} split not shipped for {problem}")
    return np.load(REF / fname)


def energies(problem: str, samples: np.ndarray) -> np.ndarray:
    """E(x) for each row of ``samples`` under our ``energies.py`` conventions."""
    x = jnp.asarray(samples)
    if problem == "dw4":
        return np.asarray(dw4_energy(x))
    return np.asarray(lj_energy(x, FILES[problem]["n_particles"]))


def pick_init(problem: str, n_chains: int, seed: int = 0,
              split: str = "train") -> jax.Array:
    """Pick ``n_chains`` ground-truth samples to seed an MCMC run.

    LJ-55 falls back to the test split (train isn't published).
    """
    try:
        arr = load(problem, split)
    except FileNotFoundError:
        arr = load(problem, "test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n_chains, replace=False)
    return jnp.asarray(arr[idx])
