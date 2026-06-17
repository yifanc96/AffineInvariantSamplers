"""Multi-GPU sweep runner for astro benchmarks.

Launches one ``astro.run_benchmark`` subprocess per GPU, cycling through a
user-supplied grid of (problem × sampler × extra-flags × seed) combos.

Usage
-----
    python -m astro.sweep \
        --problems des_y1 \
        --samplers stretch emcee peaches peams peanuts pickles chess malt aldi nuts \
        --seeds 0 1 2 \
        --gpus 0 1 2 3 4 5 6 7 \
        --n-chains 64 --warmup 500 --num-samples 1000

(The trimmed ``run_benchmark.py`` accepts ``--problem des_y1`` or
``des_y1_phys``; the all-problems driver is in ``archive/code/``.)

Each job inherits the common flags; per-sampler overrides can be added via
``--sampler-extras`` (JSON dict, see ``SAMPLER_DEFAULTS`` below).

Results live in ``astro/results/<problem>/<sampler>/seed<n>/metrics.json``;
``sweep.py`` also writes ``astro/results/sweep_summary.csv`` at the end.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


# Per-sampler default overrides for the runner CLI.  These capture the
# insights from the Hogg/transit debugging sessions: gradient-based
# samplers need a reasonable starting step and benefit from mode-finding.
# HMC-family samplers get 500 iters of `sampler_stretch` as pre-warmup to
# expand walkers to the posterior typical set *before* dual-averaging starts.
# This is standard emcee-to-HMC bootstrap practice (what bilby / numpyro's
# warmup does internally).  Without it, an artificially tight init ball
# makes dual-averaging learn a local step size that's too small once
# walkers spread out, and ε collapses.
SAMPLER_DEFAULTS = {
    "stretch":  {},
    "walk":     {},
    "emcee":    {},
    "chess":    {"warm_start": 500},
    "peaches":  {"step_size": 0.01, "warm_start": 500},
    "peanuts":  {"step_size": 0.01, "warm_start": 500},
    "peams":    {"step_size": 0.01, "warm_start": 500},
    "pickles":  {"step_size": 0.01, "warm_start": 500},
    "pickles_unadjusted": {"step_size": 0.001, "warm_start": 500},
    "nuts":     {"step_size": 0.01, "warm_start": 500},
    "malt":     {"step_size": 0.01, "warm_start": 500},
    "langevin_walk":      {"step_size": 0.01,  "warm_start": 500},
    "aldi":               {"step_size": 0.001, "warm_start": 500},
    "gndr":               {"step_size": 0.1,   "warm_start": 500},
}


def job_cmd(problem, sampler, seed, gpu, common):
    extras = SAMPLER_DEFAULTS.get(sampler, {})
    # Allow problem-specific overrides to go into a per-problem subdir for
    # dim sweeps.  Encode --dim in the out_dir so different dims don't collide.
    dim_tag = f"_d{common['dim']}" if "dim" in common else ""
    out_dir = Path(f"astro/results/{problem}{dim_tag}/{sampler}/seed{seed}")
    cmd = [sys.executable, "-m", "astro.run_benchmark",
           "--problem", problem, "--sampler", sampler,
           "--seed", str(seed), "--out", str(out_dir)]
    for k, v in common.items():
        if v is None:                     # omitted flag
            continue
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    for k, v in extras.items():
        if v is True:
            cmd += [f"--{k.replace('_', '-')}"]
        elif v is False or v is None:
            continue
        else:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"    # 8-way share on a GPU
    return cmd, env, out_dir


def worker(gpu_id, job_q, done_q):
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        problem, sampler, seed = job
        cmd, env, out_dir = job_cmd(problem, sampler, seed, gpu_id, WORKER_COMMON)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        status = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
        summary = None
        if p.returncode == 0 and (out_dir / "metrics.json").exists():
            summary = json.loads((out_dir / "metrics.json").read_text())
        done_q.put(dict(problem=problem, sampler=sampler, seed=seed, gpu=gpu_id,
                         status=status, wall_s=dt, metrics=summary,
                         stderr_tail=p.stderr[-800:] if p.returncode else None))


WORKER_COMMON: dict = {}     # set in main()


def main():
    global WORKER_COMMON

    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--samplers", nargs="+", required=True)
    ap.add_argument("--seeds",    nargs="+", type=int, default=[0])
    ap.add_argument("--gpus",     nargs="+", type=int, required=True)
    ap.add_argument("--n-chains",    type=int, default=64)
    ap.add_argument("--warmup",      type=int, default=500)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--dim",         type=int, default=None)
    ap.add_argument("--max-L",       type=int, default=None)
    ap.add_argument("--target-accept", type=float, default=None)
    ap.add_argument("--summary-csv",
                    default="astro/results/sweep_summary.csv")
    args = ap.parse_args()

    WORKER_COMMON = dict(n_chains=args.n_chains, warmup=args.warmup,
                         num_samples=args.num_samples)
    if args.dim is not None:           WORKER_COMMON["dim"] = args.dim
    if args.max_L is not None:         WORKER_COMMON["max_L"] = args.max_L
    if args.target_accept is not None: WORKER_COMMON["target_accept"] = args.target_accept

    jobs = list(itertools.product(args.problems, args.samplers, args.seeds))
    print(f"[sweep] {len(jobs)} jobs on {len(args.gpus)} GPUs")

    job_q:  queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()
    for j in jobs:
        job_q.put(j)

    threads = [threading.Thread(target=worker, args=(g, job_q, done_q),
                                 daemon=True) for g in args.gpus]
    for t in threads:
        t.start()

    results = []
    for i in range(len(jobs)):
        r = done_q.get()
        results.append(r)
        m = r.get("metrics") or {}
        info = (m.get("info") or {})
        ess_min = m.get("ess_min")
        ess_s   = m.get("ess_per_sec")
        ess_e   = m.get("ess_per_eval")
        kind    = (m.get("eval_kind") or "??")[:4]
        accept  = info.get("acceptance_rate")
        warn    = "[FAKE]" if m.get("fake_ess_warning") else ""
        fmt = lambda v, w=8, d=2: f"{v:{w}.{d}f}" if isinstance(v, (int, float)) else f"{str(v):>{w}s}"
        print(f"[{i+1}/{len(jobs)}] {r['status']:3s} gpu{r['gpu']} "
              f"{r['problem']:<9s} {r['sampler']:<12s} s{r['seed']}  "
              f"wall={fmt(r['wall_s'], 5, 1)}s  "
              f"ess_min={fmt(ess_min,  8, 1)}  "
              f"ess/s={fmt(ess_s,  8, 2)}  "
              f"ess/{kind}={fmt(ess_e, 9, 2)}  "
              f"acc={fmt(accept, 5, 3)}  {warn}")
        if r["status"] != "ok":
            print("    stderr:", (r.get("stderr_tail") or "").strip()[-300:])

    for t in threads:
        t.join(timeout=5)

    # CSV summary
    import csv
    cols = ["problem", "sampler", "seed", "gpu", "status", "wall_s",
            "ess_min", "ess_mean", "ess_per_sec", "ess_per_eval",
            "eval_kind", "n_target_evals",
            "acceptance_rate", "final_step_size", "fake_ess_warning"]
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            m = r.get("metrics") or {}
            info = m.get("info") or {}
            w.writerow([
                r["problem"], r["sampler"], r["seed"], r["gpu"], r["status"],
                r["wall_s"], m.get("ess_min"), m.get("ess_mean"),
                m.get("ess_per_sec"), m.get("ess_per_eval"),
                m.get("eval_kind"), m.get("n_target_evals"),
                info.get("acceptance_rate"), info.get("final_step_size"),
                m.get("fake_ess_warning"),
            ])
    print(f"\n[sweep] summary → {args.summary_csv}")


if __name__ == "__main__":
    main()
