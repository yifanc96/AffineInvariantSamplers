"""Small references and provenance helpers for reproducible sampler studies."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import types

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Sampler source no longer matches the reviewed adaptation reference")
    return source.replace(old, new, 1)


def sampler_from_checkout(filename, function, adaptation, narrow_retry=False):
    """Load the checkout sampler, optionally changing only reviewed expressions.

    No file is edited. Original references retain current input guards and
    corrected DR acceptance, but restore log(a) warmup for valid initial a>1.
    The optional narrower retry changes only a2, preserving the acceptance law.
    """
    path = REPO_ROOT / "src/affine_invariant_samplers" / filename
    source = path.read_text()
    if adaptation == "original":
        source = _replace_once(source,
            "    log_a0 = jnp.log(a - 1.) if adapt_a else jnp.log(a)",
            "    log_a0 = jnp.log(a)")
        source = _replace_once(source,
            "        a_cur = (1. + jnp.maximum(jnp.exp(da.log_h), jnp.finfo(a.dtype).eps)\n"
            "                 if adapt_a else a)",
            "        a_cur = jnp.exp(da.log_h) if adapt_a else a")
        source = _replace_once(source,
            "    final_a = (1. + jnp.maximum(jnp.exp(da.log_h_bar), jnp.finfo(a.dtype).eps)\n"
            "               if adapt_a else a)",
            "    final_a = jnp.exp(da.log_h_bar) if adapt_a else a")
    elif adaptation != "positive":
        raise ValueError("adaptation must be original or positive")
    if narrow_retry:
        source = _replace_once(source, "        a2_cur = a_cur * shrink",
                                "        a2_cur = 1. + shrink * (a_cur - 1.)")
        source = _replace_once(source, "    final_a2 = final_a * shrink",
                                "    final_a2 = 1. + shrink * (final_a - 1.)")
    name = "affine_invariant_samplers._audit_" + filename[:-3] + "_" + adaptation
    if narrow_retry:
        name += "_narrow_retry"
    module = types.ModuleType(name)
    module.__package__ = "affine_invariant_samplers"
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return getattr(module, function)


def peaches_from_checkout(pre_time_fix=False, complement_tuner_reference=False):
    """Optional expression references for historical ChEES profiles."""
    path = REPO_ROOT / "src/affine_invariant_samplers/peaches.py"
    source = path.read_text()
    if pre_time_fix:
        source = _replace_once(source, "    g_m = trajectory_time * diff_sq * inner",
            "    g_m = state.halton * jnp.exp(state.log_T) * diff_sq * inner")
    if complement_tuner_reference:
        source = _replace_once(source,
            "        use_full_metric = (complement.shape[0] <= pos_cur.shape[1]\n"
            "                           and pos_cur.shape[0] + complement.shape[0] > pos_cur.shape[1])",
            "        use_full_metric = False")
    name = "affine_invariant_samplers._audit_peaches_" + ("pre_time_fix" if pre_time_fix else "current")
    if complement_tuner_reference:
        name += '_complement_reference'
    module = types.ModuleType(name)
    module.__package__ = "affine_invariant_samplers"
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.sampler_peaches


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def metadata(protocol):
    paths = [REPO_ROOT / "src/affine_invariant_samplers" / filename
             for filename in ("stretch.py", "ensemble_dr.py", "_dr_utils.py", "peaches.py",
                              "side.py", "walk.py", "_sampler_utils.py")]
    status = _git("status", "--porcelain")
    return dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        revision=_git("rev-parse", "HEAD"), branch=_git("branch", "--show-current"),
        worktree_dirty=bool(status) if status is not None else None,
        dirty_paths=status.splitlines() if status else [],
        source_sha256={str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths if path.exists()},
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
        backend=jax.default_backend(), jax_enable_x64=bool(jax.config.jax_enable_x64),
        protocol=protocol,
    )


def save_metadata(path, protocol):
    path.write_text(json.dumps(metadata(protocol), indent=2) + "\n")


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_results(stem, results):
    stem.with_suffix(".json").write_text(json.dumps(_finite_json(results), indent=2,
                                                  allow_nan=False) + "\n")
    fields = list(dict.fromkeys(key for row in results for key in row))
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
