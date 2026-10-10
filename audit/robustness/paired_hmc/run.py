"""Reproduce one paired PEACHES cell using this checkout and saved initial state.

Run from the repository root with PYTHONPATH=src and benchmark extras installed:
python -m audit.robustness.paired_hmc.run --target rosenbrock --walkers 32 --seed 0 --version original
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from audit.robustness._support import metadata, peaches_from_checkout
from audit.robustness.paired_hmc._metrics import all_metrics
from audit.robustness.paired_hmc._targets import make_target

HERE = Path(__file__).resolve().parent


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_cell(target, walkers, seed, version, warmup=1000, num_samples=1500):
    if version not in ("original", "corrected"):
        raise ValueError("version must be original or corrected")
    initial_record = json.loads((HERE / "initials.json").read_text())[
        f"{target}_w{walkers}_seed{seed}_ws100"]
    init_np = np.asarray(initial_record["values"], dtype=initial_record["dtype"])
    initial_hash = hashlib.sha256(init_np.tobytes()).hexdigest()
    if initial_hash != initial_record["sha256"]:
        raise ValueError("Saved initial ensemble does not match its hash")
    init = jnp.asarray(init_np)
    log_prob, truth, target_metadata = make_target(
        target, 2, **({"log_cond": 0} if target == "gaussian" else {}))
    sampler = peaches_from_checkout(pre_time_fix=(version == "original"))
    settings = dict(num_samples=num_samples, warmup=warmup, seed=seed,
                    target_accept=.8, max_L=20 if target == "gaussian" else 256,
                    verbose=False)
    effective_settings = {
        name: settings.get(name, param.default)
        for name, param in list(inspect.signature(sampler).parameters.items())[2:]
        if name in settings or param.default is not inspect.Parameter.empty}
    protocol = dict(target=target, dim=2, walkers=walkers, version=version,
                    settings=effective_settings, initial_sha256=initial_hash,
                    warm_start_requested=100, preparation_iterations=200,
                    initialization="saved common float64 stretch-prepared ensemble",
                    reference="historical halton*T expression" if version == "original"
                              else "actual integrated trajectory_time expression")
    provenance = metadata(protocol)
    t0 = time.perf_counter()
    samples, info = sampler(log_prob, init, **settings)
    jax.block_until_ready(samples)
    wall = time.perf_counter() - t0
    samples = np.asarray(samples)
    cost = int(info["n_grad_evals"])
    flat = samples.reshape(-1, 2)
    return json_safe(dict(protocol=protocol, source_metadata=provenance,
                          target_metadata=target_metadata, info=info,
                          n_grad_evals_prod=cost, wall_s=wall,
                          metrics=all_metrics(samples, truth, n_grad_evals=cost, fast=True),
                          mean=flat.mean(0), std=flat.std(0), second_moment=(flat**2).mean(0),
                          truth_mean=truth["mean"], truth_std=truth["std"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["gaussian", "rosenbrock"], required=True)
    parser.add_argument("--walkers", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--version", choices=["original", "corrected"], required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_cell(args.target, args.walkers, args.seed, args.version,
                      warmup=args.warmup, num_samples=args.num_samples)
    out = args.out or HERE / "rerun" / f"peaches_{args.target}_w{args.walkers}_seed{args.seed}_{args.version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"file": str(out), "info": result["info"], "metrics": result["metrics"]}, allow_nan=False))


if __name__ == "__main__":
    main()
