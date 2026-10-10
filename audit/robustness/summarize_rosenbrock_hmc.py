"""Render the stored, paired Rosenbrock HMC checks without rerunning samplers."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METHODS = ("peaches", "pickles", "peanuts", "peams")
STUDIES = (
    ("Before guards", "rosenbrock_hmc_pre_numerical_fix"),
    ("After guards", "rosenbrock_hmc"),
    ("Longer run", "rosenbrock_hmc_roomy"),
)


def main():
    studies = [(label, json.loads((ROOT / "data" / directory / "results.json").read_text()))
               for label, directory in STUDIES]
    assert all(len(rows) == 16 for _, rows in studies), "Expected all paired cells"
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    for row_index, dimension in enumerate((2, 8)):
        for column, method in enumerate(METHODS):
            ax = axes[row_index, column]
            for seed in (0, 1):
                errors = [100 * next(r["variance_relative_error_max"] for r in rows
                                     if (r["dim"], r["sampler"], r["seed"])
                                     == (dimension, method, seed)) for _, rows in studies]
                ax.plot(range(3), errors, "o-", label=f"Seed {seed}")
            ax.set_title(f"{method.upper()}, {dimension}-D")
            ax.set_xticks(range(3), ["Before\nguards", "After\nguards", "Longer\nrun"])
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=.25)
            if column == 0:
                ax.set_ylabel("Maximum relative variance error (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Rosenbrock: numerical repairs and a larger sampling budget")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"rosenbrock_hmc.{extension}", dpi=160)
    plt.close(fig)

    text = """# Ensemble Hamiltonian samplers on Rosenbrock

The numerical guards removed zero-length trajectories in the observed failing
PEACHES/PICKLES cases. Every corrected run was finite and moved. Gaussian
accuracy does not imply equally fast sampling of the curved Rosenbrock valley:
the learned length, initialization, and sampling budget still matter.

## Protocol

The target is a product of independent pairs, with
`log pi(x)=-sum((x_even-1)^2+100*(x_odd-x_even^2)^2)`.
Equivalently, `X~N(1,.5)` and `Y=X^2+E`, `E~N(0,.005)` independent.
Coordinate means are `1/1.5`, variances `.5/2.505`. This is the repository's
paired Rosenbrock, rather than the chained optimization benchmark.

All cases used float32, 64 walkers, dimensions 2/8, sampler seeds 0/1,
and unit Gaussian starts at key `42+seed`. Each start was prepared by stretch
with 100 warmup plus 100 production iterations at that same sampler seed.
The resulting ensemble was passed identically to all four methods.
Preparation costs 12,864 log-density evaluations, including the initial one.
The older short-run files recorded 12,800, excluding that initial evaluation.
The before/after comparison used 750 warmup and 3,000 production iterations,
`max_L=20` or `max_tree_depth=5`. The longer study used 1,000 warmup and
10,000 production iterations, `max_L=128` or `max_tree_depth=8`.
It changes both warmup and available trajectory length, so it does not isolate
the effect of a larger production budget.

Stored metadata identifies the source hashes and settings for each run. These
studies predate the conditional full-ensemble warmup metric repair; that repair
does not trigger here because the complement has 32 walkers and D is 2 or 8.
The historical failures retain their original recorded trajectory/cost fields,
which were unreliable when nominal L became zero. Do not use those costs to
rank efficiency.

![Variance errors for every seed](rosenbrock_hmc.png)

The before/after comparison preserves all seeds. Three PEACHES cases reported
mean L=0; the D=2, seed=0 PICKLES case made no moves and had ESS=0 despite
acceptance=1. The actual-time tuning correction was already present in this
comparison; the new difference is the numerical guard repair.

## Accuracy and cost after the repair

Each cell below shows the range across both seeds. Variance error is the
largest relative coordinate error. ESS counts effective **ensemble averages**,
taking the minimum over coordinates, coordinate squares, banana residuals,
and squared residuals. Production gradient counters exclude preparation and
warmup. PEANUTS reports logical active-tree work; vectorization may execute
additional padded work, so its counter is not device FLOPs.

| Run | D | Sampler | Maximum variance error | Minimum observable ESS | Production gradients (millions) |
|---|---:|---|---:|---:|---:|
"""
    for label, rows in studies[1:]:
        for dimension in (2, 8):
            for method in METHODS:
                cells = [r for r in rows if r["dim"] == dimension and r["sampler"] == method]
                def span(key, scale=1., digits=1):
                    values = [r[key] * scale for r in cells]
                    return f"{min(values):.{digits}f}–{max(values):.{digits}f}"
                text += (f"| {label} | {dimension} | {method.upper()} | "
                         f"{span('variance_relative_error_max', 100)}% | "
                         f"{span('observable_ensemble_mean_ess_min')} | "
                         f"{span('production_gradient_evaluations', 1e-6, 2)} |\n")
    text += """
PEAMS gave the most consistent moments in the longer comparison: maximum
variance errors were 0.9–2.7%, with minimum observable ESS 152–273. PEACHES
reached 2.5–3.4% errors in 8-D; its 2-D seed 1 still learned short trajectories
and had only about 11 effective averages for the slowest observable. PICKLES
generally improved its moments, but one run learned mean L about 90 and spent
57.6 million gradients. Raising PEANUTS's depth from 5 to 8 was costly and did
not consistently improve accuracy or observable ESS. These results support
measuring a capacity increase, rather than assuming it always helps efficiency.

The [fixed-L check](rosenbrock_fixed_l/findings.md) compares a simple alternative
for PEACHES/PICKLES using the same prepared starts. It is an option to compare,
with mixed outcomes, rather than an applied default.

## Practical choices supported by these checks

For this stiff paired target, give warmup enough iterations and keep the
initial ensemble dispersed along the valley. Inspect realized mean L and cap
hit rate alongside movement, moments, and ESS of squared observables. A high
acceptance rate with very short trajectories can still mix slowly. If adaptation
chooses a poor length, compare a fixed L under the same gradient budget.
Compare independent ensemble seeds before trusting tail moments.

PEACHES/PICKLES/PEAMS dual averaging uses a harmonic mean of per-walker
acceptance probabilities, with default target .651. The reported fraction of
accepted moves and arithmetic mean acceptance are different statistics; they
need not match that target. The new harmonic-acceptance field helps interpret
whether step-size tuning is meeting its intended criterion.

PEAMS is a useful first comparison for this Rosenbrock protocol. PEACHES and
PICKLES also produced reasonable moments with the corrected warmup, but their
cost and slowest-observable mixing varied more between seeds. PEANUTS remains
an option whose tree depth should be justified by measured cost and mixing.
Affine invariance handles linear distortion; a curved narrow valley can remain
hard for these methods and for gradient-free affine samplers.

## Reproduce

From the repository root, use a new output directory to preserve stored results:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_rosenbrock_hmc.py --output-dir /tmp/rosenbrock-short
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_rosenbrock_hmc.py --warmup 1000 --production 10000 --max-L 128 --max-tree-depth 8 --output-dir /tmp/rosenbrock-long
python audit/robustness/summarize_rosenbrock_hmc.py
```

The sampler scripts use the source in the current checkout. The summary script
only reads the stored result files. Exact regeneration of a historical failed
run requires its recorded source snapshot; a newer checkout intentionally fixes
those paths. ESS uses a self-consistent ACF window with IAT capped at one, not
rank-normalized convergence diagnostics. Some traces fail the 50-times-IAT
length heuristic. Two seeds cannot establish a universal sampler ranking.
"""
    (ROOT / "rosenbrock_hmc.md").write_text(text)


if __name__ == "__main__":
    main()
