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
WS = 100                  # stretch pre-warmup: equilibrate ensemble SHAPE (fair,
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


def scale2_jobs():
    """CLEAN dimensional scaling, ONE consistent protocol (tag ``scale2``).

    Fixes the low-d artefact in the old scaling figure. Old runs mixed several
    tags (scaleL/scaleLb/scaleLd/exp4c) with *different* warmup/tuning at the
    same walker count, and used uniform 4d walkers -- at which the empirical
    covariance is too noisy in low d for ChEES to find the (tiny) optimal
    L~d^{1/4}, so peaches over-integrates (mean_L~6 vs optimal ~1.5) and the
    per-gradient efficiency shows a spurious low-d hump.

    Here: isotropic N(0,I_d), n_walkers = max(128, 4d) (the SAME floor the
    conditioning/warmup experiments use -- abundant walkers in low d let ChEES
    resolve the short optimal trajectory), one fixed protocol, 5 seeds.
      - peaches: target_accept 0.8 (step-size law holds at fixed acceptance),
        max_L 512 (never bind), ws=100, warmup 4000 (dual-averaging converges).
      - stretch: only d<=64 (does not converge higher in a feasible run).
      - nuts:    diagonal mass (optimal, low-noise on an isotropic target).
    """
    jobs = []
    SC_SEEDS = [0, 1, 2, 3, 4]
    DIMS = (2, 4, 8, 16, 32, 64, 128, 256)
    for D in DIMS:
        nch = max(128, 4 * D)
        nprod = 3000 if D >= 256 else 5000        # d=256 sample array memory
        # peaches
        for sd in SC_SEEDS:
            jobs.append(dict(target="gaussian", dim=D, sampler="peaches", seed=sd,
                             tag="scale2", warmup=4000, num_samples=nprod,
                             n_chains=nch, warm_start=WS, log_cond=0.0,
                             no_rotate=True, target_accept=0.8, max_L=512))
        # stretch/emcee: only where it can converge
        if D <= 64:
            for sd in SC_SEEDS:
                jobs.append(dict(target="gaussian", dim=D, sampler="stretch",
                                 seed=sd, tag="scale2", warmup=3000,
                                 num_samples=12000, n_chains=nch, warm_start=0,
                                 log_cond=0.0, no_rotate=True))
    return jobs


def scale3_jobs():
    """SMOOTH dimensional scaling: constant walkers/dim = 8d (tag ``scale3``).

    scale2 used n_walkers=max(128,4d); at 4d walkers ChEES cannot resolve the
    short optimal trajectory (L~d^{1/4}<4) in low/moderate d, so it
    over-integrates and the per-eval efficiency has a spurious low-d hump plus a
    step at the floor->4d transition (d=16->32, where mean_L jumps 3->6). At a
    CONSTANT ratio of 8 walkers/dim the empirical-covariance (and the ChEES ESJD
    gradient) is equally well-resolved at every d, so mean_L grows smoothly as
    d^{1/4} and ESS/eval is a clean monotone d^{-1/4} decline. A small floor 64
    keeps d=2,4 well-populated. Scored with the fast ensemble-mean ESS
    (--fast-ess), which is O(D N log N) and makes even d=256 (2048 walkers)
    cheap. Isotropic N(0,I_d), 5 seeds.
    """
    jobs = []
    SC_SEEDS = [0, 1, 2, 3, 4]
    DIMS = (2, 4, 8, 16, 32, 64, 128, 256)
    for D in DIMS:
        nch = max(64, 8 * D)
        nprod = 3000 if D >= 128 else 5000
        for sd in SC_SEEDS:
            jobs.append(dict(target="gaussian", dim=D, sampler="peaches", seed=sd,
                             tag="scale3", warmup=4000, num_samples=nprod,
                             n_chains=nch, warm_start=WS, log_cond=0.0,
                             no_rotate=True, target_accept=0.8, max_L=512,
                             fast_ess=True))
        if D <= 64:
            for sd in SC_SEEDS:
                jobs.append(dict(target="gaussian", dim=D, sampler="stretch",
                                 seed=sd, tag="scale3", warmup=3000,
                                 num_samples=12000, n_chains=nch, warm_start=0,
                                 log_cond=0.0, no_rotate=True, fast_ess=True))
    return jobs


def condf_jobs():
    """Conditioning: ill-conditioned Gaussian d=50 vs kappa, ensemble ESS.

    8d walkers (=400 at d=50), matching the scaling sweep so peaches numbers are
    consistent across the note. peaches / emcee / NUTS-dense at tag condf_lc{k};
    NUTS-identity (the no-mass-adapt ablation) at condfid_lc{k}. 3 seeds,
    --fast-ess. The Rosenbrock curvature panel is produced by rosen8_jobs()."""
    jobs = []
    # (a) ill-conditioned Gaussian d=50 vs kappa
    for lc in (2, 4, 6, 8):
        for sd in SEEDS:
            jobs.append(dict(target="gaussian", dim=50, sampler="peaches", seed=sd,
                             tag=f"condf_lc{lc}", warmup=1000, num_samples=5000,
                             n_chains=400, warm_start=WS, log_cond=float(lc),
                             max_L=GAUSS_MAXL, fast_ess=True))
            jobs.append(dict(target="gaussian", dim=50, sampler="stretch", seed=sd,
                             tag=f"condf_lc{lc}", warmup=10000, num_samples=40000,
                             n_chains=400, warm_start=0, log_cond=float(lc),
                             fast_ess=True))
            jobs.append(dict(target="gaussian", dim=50, sampler="nuts", seed=sd,
                             tag=f"condf_lc{lc}", warmup=1000, num_samples=5000,
                             n_chains=400, warm_start=0, log_cond=float(lc),
                             nuts_mass="dense", fast_ess=True))
            jobs.append(dict(target="gaussian", dim=50, sampler="nuts", seed=sd,
                             tag=f"condfid_lc{lc}", warmup=1000, num_samples=2000,
                             n_chains=400, warm_start=0, log_cond=float(lc),
                             nuts_mass="identity", fast_ess=True))
    return jobs


def warmf_jobs():
    """Warmup robustness, re-scored with the ensemble ESS (--fast-ess).

    Gaussian d=100 (kappa=e^6, 400 walkers) + Rosenbrock d=8 (256 walkers),
    warmup in {50..1000}, peaches vs NUTS-dense, 8 seeds. Tags warmf_w{w} /
    warmfb_w{w}."""
    jobs = []
    W8 = [0, 1, 2, 3, 4, 5]
    for warm in (50, 100, 200, 500, 1000):
        for sd in W8:
            jobs.append(dict(target="gaussian", dim=100, sampler="peaches", seed=sd,
                             tag=f"warmf_w{warm}", warmup=warm, num_samples=5000,
                             n_chains=400, warm_start=200, log_cond=6.0,
                             max_L=GAUSS_MAXL, fast_ess=True))
            jobs.append(dict(target="gaussian", dim=100, sampler="nuts", seed=sd,
                             tag=f"warmf_w{warm}", warmup=warm, num_samples=5000,
                             n_chains=400, warm_start=0, log_cond=6.0,
                             nuts_mass="dense", fast_ess=True))
            jobs.append(dict(target="rosenbrock", dim=8, sampler="peaches", seed=sd,
                             tag=f"warmfb_w{warm}", warmup=warm, num_samples=5000,
                             n_chains=64, warm_start=200, max_L=ROSEN_MAXL,
                             fast_ess=True))
            jobs.append(dict(target="rosenbrock", dim=8, sampler="nuts", seed=sd,
                             tag=f"warmfb_w{warm}", warmup=warm, num_samples=5000,
                             n_chains=64, warm_start=0, nuts_mass="dense",
                             fast_ess=True))
    return jobs


def rosen8_jobs():
    """Rosenbrock at the constant 8d walker rule (floor 64), ensemble ESS.

    One unified set of Rosenbrock runs feeding BOTH the dimensional-scaling figure
    (d=2..32, peaches vs emcee) and the curvature panel of Exp.~2b (d=8,16,32,
    adding identity-mass NUTS). Lean 8d walkers land in the long (accurate)
    ChEES mode more often (Sec. bistability); 8 peaches seeds stabilise the median
    over the residual bimodality. Tags rz8 / rz8id."""
    jobs = []
    S8 = list(range(8))
    for D in (2, 4, 8, 16, 32):
        nch = max(64, 8 * D)
        for sd in S8:
            jobs.append(dict(target="rosenbrock", dim=D, sampler="peaches", seed=sd,
                             tag="rz8", warmup=8000, num_samples=8000,
                             n_chains=nch, warm_start=200, max_L=512,
                             target_accept=0.8, fast_ess=True))
        for sd in SEEDS:
            jobs.append(dict(target="rosenbrock", dim=D, sampler="stretch", seed=sd,
                             tag="rz8", warmup=20000, num_samples=60000,
                             n_chains=nch, warm_start=0, fast_ess=True))
        if D >= 8:      # curvature panel: identity-mass NUTS (mild global cond.)
            for sd in SEEDS:
                jobs.append(dict(target="rosenbrock", dim=D, sampler="nuts", seed=sd,
                                 tag="rz8id", warmup=1000, num_samples=2000,
                                 n_chains=nch, warm_start=0, nuts_mass="identity",
                                 fast_ess=True))
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
    if spec.get("fast_ess"):
        cmd += ["--fast-ess"]
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
    ap.add_argument("--exp", choices=["1", "2", "3", "4", "4b", "4c", "scale2",
                                      "scale3", "condf", "warmf", "rosen8",
                                      "notef", "all"], required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 5, 6],
                    help="GPU indices to use (default excludes GPU 1, which may "
                         "be shared with another user).")
    args = ap.parse_args()
    builders = {"1": exp1_jobs, "2": exp2_jobs, "3": exp3_jobs, "4": exp4_jobs,
                "4b": exp4b_jobs, "4c": exp4c_jobs, "scale2": scale2_jobs,
                "scale3": scale3_jobs, "condf": condf_jobs, "warmf": warmf_jobs,
                "rosen8": rosen8_jobs}
    if args.exp == "all":
        jobs = exp1_jobs() + exp2_jobs() + exp3_jobs() + exp4_jobs()
    elif args.exp == "notef":            # all three note re-scores, in one pool
        jobs = condf_jobs() + warmf_jobs() + rosen8_jobs()
    else:
        jobs = builders[args.exp]()
    print(f"exp {args.exp}: {len(jobs)} jobs over GPUs {args.gpus}", flush=True)
    run_pool(jobs, args.gpus)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
