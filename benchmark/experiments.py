"""Define + launch the three benchmark experiments across GPUs.

Each experiment is a list of ``benchmark.run`` job specs (one subprocess per
cell, pinned to a GPU). Results land under ``benchmark/results/`` and are
consumed by ``plots.py``.

    python -m benchmark.experiments --exp 1 --gpus 0 1 2 3 4 5
    python -m benchmark.experiments --exp all --gpus 0 1 2 3

Experiments
-----------
1. Dimensional scaling: peaches vs stretch.
   - Gaussian D ∈ {10,50,100,200} (kappa=e^6): stretch degrades with D.
   - Rosenbrock D ∈ {2,4,8,16,32}: stretch cannot cross curvature.
   Shows burn-in bias (mean error in sigma) AND stationary IAT.
2. Affine vs non-affine: peaches vs chess (+ stretch), Gaussian D=50,
   kappa = e^{2,4,6,8}. peaches/stretch are kappa-invariant; chess degrades.
3. Warmup robustness: peaches vs NUTS, Gaussian D=100 + Rosenbrock D=8,
   warmup ∈ {50,100,200,500,1000} at fixed production length.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from collections import deque

SEEDS = [0, 1, 2]
N_PROD = 5000
WS = 200                  # stretch pre-warmup: equilibrate ensemble SHAPE (fair,
                          # gradient-free, no target knowledge). Ensemble analogue
                          # of NUTS mass-matrix warmup. Critical: an isotropic
                          # init leaves the empirical-covariance preconditioner
                          # ill-shaped and collapses the adapted step size.

# per-target sampler knobs
ROSEN_MAXL = 256          # Rosenbrock needs long trajectories (local curvature)
GAUSS_MAXL = 20           # default suffices for a Gaussian
MTD = 10                  # peanuts/NUTS max tree depth


def n_walkers(dim):
    """Ensemble methods need n_walkers > D for a full-rank empirical-covariance
    preconditioner (else ~D-n_walkers directions are unpreconditioned -> bias).
    Use ~4D, floor 128. stretch also needs walkers >~ 2D."""
    return max(128, 4 * dim)


def job(target, dim, sampler, seed, tag, warmup, **extra):
    nch = n_walkers(dim)
    ws = 0 if sampler == "nuts" else WS         # NUTS uses its own warmup
    spec = dict(target=target, dim=dim, sampler=sampler, seed=seed, tag=tag,
                warmup=warmup, num_samples=N_PROD, n_chains=nch, warm_start=ws)
    spec.update(extra)
    return spec


def exp1_jobs():
    """Dimensional scaling: peaches vs stretch (+ peanuts on Gaussian)."""
    jobs = []
    for D in (10, 50, 100, 200):
        for s in ("peaches", "stretch"):
            for sd in SEEDS:
                e = {"max_L": GAUSS_MAXL} if s == "peaches" else {}
                jobs.append(job("gaussian", D, s, sd, "exp1", 1000,
                                log_cond=6.0, **e))
    for D in (2, 4, 8, 16, 32):
        for s in ("peaches", "stretch"):
            for sd in SEEDS:
                e = {"max_L": ROSEN_MAXL} if s == "peaches" else {}
                jobs.append(job("rosenbrock", D, s, sd, "exp1", 2000, **e))
    return jobs


def exp2_jobs():
    """Affine invariance: peaches vs chess vs stretch, Gaussian D=50, vs kappa."""
    jobs = []
    for lc in (2.0, 4.0, 6.0, 8.0):
        for s in ("peaches", "chess", "stretch"):
            for sd in SEEDS:
                e = {"max_L": GAUSS_MAXL} if s in ("peaches", "chess") else {}
                jobs.append(job("gaussian", 50, s, sd, f"exp2_lc{int(lc)}",
                                1000, log_cond=lc, **e))
    return jobs


def exp4_jobs():
    """Dimensional scaling of ESS/grad on the ISOTROPIC Gaussian N(0,I_d).

    Tests the d^{-1/4} step-size law (Beskos et al. 2013): optimally-tuned
    HMC/kinetic samplers need eps ~ d^{-1/4}, so L ~ d^{1/4} and
    ESS/grad ~ d^{-1/4}. Isotropic target removes the conditioning confound;
    NUTS uses DIAGONAL mass (optimal & low-noise for isotropic). 4 seeds.
    """
    jobs = []
    SCALE_SEEDS = [0, 1, 2, 3]
    for D in (2, 4, 8, 16, 32, 64, 128):
        nch = max(128, 4 * D)                    # 4D: ensemble metric (and peanuts'
                                                 # affine U-turn) needs n_walkers>>D;
                                                 # 2D corrupts the U-turn at high D.
        for s in ("peaches", "peanuts", "nuts"):
            for sd in SCALE_SEEDS:
                spec = dict(target="gaussian", dim=D, sampler=s, seed=sd,
                            tag="exp4", warmup=1000, num_samples=3000,
                            n_chains=nch, warm_start=(0 if s == "nuts" else WS),
                            log_cond=0.0, no_rotate=True)
                if s == "peaches":
                    spec["max_L"] = 512           # don't bind: optimal L ~ d^{1/4}
                elif s == "peanuts":
                    spec["max_tree_depth"] = MTD
                else:                              # nuts
                    spec["nuts_mass"] = "diag"
                jobs.append(spec)
    return jobs


def exp4b_jobs():
    """CORRECTED dimensional scaling: n_walkers = 4d at EVERY d (no 128 floor).

    exp4 used n_walkers=max(128,4d), so d<=32 all ran at 128 walkers -- the
    preconditioner was not d-matched and the fitted eps/L-bar slopes were
    pre-asymptotically biased shallow. Here n_walkers=4d throughout and d is
    pushed to 256, so the slope can be fit on a clean tail and compared with
    the d^{-1/4} step-size law. Isotropic target; NUTS diagonal mass; 4 seeds.
    """
    jobs = []
    SCALE_SEEDS = [0, 1, 2, 3]
    for D in (8, 16, 32, 64, 128, 256):
        nch = 4 * D                              # d-matched at every d, no floor
        for s in ("peaches", "peanuts", "nuts"):
            for sd in SCALE_SEEDS:
                spec = dict(target="gaussian", dim=D, sampler=s, seed=sd,
                            tag="exp4b", warmup=1000, num_samples=3000,
                            n_chains=nch, warm_start=(0 if s == "nuts" else WS),
                            log_cond=0.0, no_rotate=True)
                if s == "peaches":
                    spec["max_L"] = 512
                elif s == "peanuts":
                    spec["max_tree_depth"] = MTD
                else:
                    spec["nuts_mass"] = "diag"
                jobs.append(spec)
    return jobs


def exp4c_jobs():
    """RECOVER the d^{-1/4} law: pin acceptance + converge the step-size adapter.

    exp4b showed peaches/peanuts step-size dual-averaging does NOT converge in
    warmup=1000 -- realized acceptance (0.83-0.95) sits far above target and
    drifts with d, so eps stays flat and never reaches the d^{-1/4} curve.
    Here we (i) pin target_accept=0.8 for all three samplers (the law holds at
    any FIXED acceptance) and (ii) lengthen warmup to 4000 so dual-averaging
    converges. With acceptance flat across d, theory predicts eps ~ d^{-1/4},
    L-bar ~ d^{+1/4}, ESS/grad ~ d^{-1/4}. n_walkers=4d throughout; 3 seeds.
    """
    jobs = []
    SCALE_SEEDS = [0, 1, 2]
    for D in (8, 16, 32, 64, 128, 256):
        nch = 4 * D
        for s in ("peaches", "peanuts", "nuts"):
            for sd in SCALE_SEEDS:
                spec = dict(target="gaussian", dim=D, sampler=s, seed=sd,
                            tag="exp4c", warmup=2000, num_samples=3000,
                            n_chains=nch, warm_start=(0 if s == "nuts" else WS),
                            log_cond=0.0, no_rotate=True, target_accept=0.8)
                if s == "peaches":
                    spec["max_L"] = 512
                elif s == "peanuts":
                    spec["max_tree_depth"] = MTD
                else:
                    spec["nuts_mass"] = "diag"
                jobs.append(spec)
    return jobs


def exp3_jobs():
    """Warmup robustness: peaches/peanuts vs NUTS, Gaussian D=100 + Rosenbrock D=8."""
    jobs = []
    for warm in (50, 100, 200, 500, 1000):
        for s in ("peaches", "peanuts", "nuts"):
            for sd in SEEDS:
                e = {"max_L": GAUSS_MAXL} if s == "peaches" else \
                    {"max_tree_depth": MTD} if s == "peanuts" else {}
                jobs.append(job("gaussian", 100, s, sd, f"exp3_w{warm}",
                                warm, log_cond=6.0, **e))
        for s in ("peaches", "peanuts", "nuts"):
            for sd in SEEDS:
                e = {"max_L": ROSEN_MAXL} if s == "peaches" else \
                    {"max_tree_depth": MTD} if s == "peanuts" else {}
                jobs.append(job("rosenbrock", 8, s, sd, f"exp3_w{warm}",
                                warm, **e))
    return jobs


def spec_to_cmd(spec):
    out = (f"benchmark/results/{spec['target']}_d{spec['dim']}/"
           f"{spec['sampler']}__{spec['tag']}/seed{spec['seed']}")
    cmd = [sys.executable, "-m", "benchmark.run",
           "--target", spec["target"], "--dim", str(spec["dim"]),
           "--sampler", spec["sampler"], "--seed", str(spec["seed"]),
           "--tag", spec["tag"], "--warmup", str(spec["warmup"]),
           "--num-samples", str(spec["num_samples"]),
           "--n-chains", str(spec["n_chains"]), "--out", out]
    if spec.get("warm_start"):
        cmd += ["--warm-start", str(spec["warm_start"])]
    if "log_cond" in spec:
        cmd += ["--log-cond", str(spec["log_cond"])]
    if spec.get("no_rotate"):
        cmd += ["--no-rotate"]
    if "max_L" in spec:
        cmd += ["--max-L", str(spec["max_L"])]
    if "max_tree_depth" in spec:
        cmd += ["--max-tree-depth", str(spec["max_tree_depth"])]
    if "nuts_mass" in spec:
        cmd += ["--nuts-mass", str(spec["nuts_mass"])]
    if "target_accept" in spec:
        cmd += ["--target-accept", str(spec["target_accept"])]
    return cmd, out


def run_pool(jobs, gpus):
    """Run jobs across GPUs, one subprocess per GPU at a time."""
    pending = deque(jobs)
    running = {}                      # gpu -> (proc, spec)
    done = 0
    total = len(jobs)
    free = deque(gpus)
    while pending or running:
        while pending and free:
            gpu = free.popleft()
            spec = pending.popleft()
            cmd, out = spec_to_cmd(spec)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "run.log"), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (proc, spec, log)
        # poll
        for gpu, (proc, spec, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                done += 1
                ok = "ok" if proc.returncode == 0 else f"FAIL({proc.returncode})"
                print(f"[{done}/{total}] gpu{gpu} {ok}  "
                      f"{spec['target']} D={spec['dim']} {spec['sampler']} "
                      f"{spec['tag']} s{spec['seed']}", flush=True)
                del running[gpu]
                free.append(gpu)
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["1", "2", "3", "4", "4b", "4c", "all"],
                    required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 5, 6],
                    help="GPU indices to use (default excludes GPU 1, which may "
                         "be shared with another user).")
    args = ap.parse_args()
    builders = {"1": exp1_jobs, "2": exp2_jobs, "3": exp3_jobs, "4": exp4_jobs,
                "4b": exp4b_jobs, "4c": exp4c_jobs}
    if args.exp == "all":
        jobs = exp1_jobs() + exp2_jobs() + exp3_jobs() + exp4_jobs()
    else:
        jobs = builders[args.exp]()
    print(f"exp {args.exp}: {len(jobs)} jobs over GPUs {args.gpus}", flush=True)
    run_pool(jobs, args.gpus)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
