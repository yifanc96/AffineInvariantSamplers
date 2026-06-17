"""Comprehensive audit producing the EXACT numbers used in the paper.

Reports for each sampler the median ess across all available seeds,
the seed range, the production gradient count, and ess/grad. Provides
clear annotations of which configurations are multi-seed (3 seeds) and
which are single-seed exploratory runs.

Run: python astro/audit_for_paper.py
"""
from __future__ import annotations
import json, glob
import numpy as np
from pathlib import Path

import sys; sys.path.insert(0, '.')
from affine_invariant_samplers.diagnostics import integrated_autocorr_time_1d


def ess_A_recompute(x):
    """Convention A: per-chain Sokal averaged."""
    N, C, D = x.shape
    taus = np.empty((C, D))
    for c in range(C):
        for d in range(D):
            taus[c, d], _, _ = integrated_autocorr_time_1d(x[:, c, d])
    return N * C / np.maximum(taus.mean(axis=0), 1.0)


def report(label, dirs, seed_filter=None):
    """Aggregate across seeds for a given config; returns rows."""
    rows = []
    for d in dirs:
        for sd in sorted(Path(d).glob('seed*/metrics.json')):
            try:
                m = json.load(open(sd))
                i = m.get('info') or {}
                ess_min = m['ess_min']
                n_grad = m['n_target_evals']
                wall = m['wall_s']
                # Acceptance (if available)
                acc = i.get('acceptance_rate', None)
                # Mean trajectory length / grad-per-sample
                if 'mean_L' in i:
                    mL = float(i['mean_L'])
                elif 'mean_num_steps' in i:
                    mL = float(i['mean_num_steps'])
                elif 'mean_tree_depth' in i:
                    mL = float(2 ** i['mean_tree_depth'])
                else:
                    mL = 1.0
                rows.append({
                    'seed': int(str(sd.parent).split('seed')[-1]),
                    'ess_min': ess_min,
                    'n_grad_prod': n_grad,
                    'wall_s': wall,
                    'mean_L': mL,
                    'acc': acc,
                    'div': i.get('n_divergences', 0),
                    'N': m['num_samples'],
                    'warmup': m['warmup'],
                    'n_chains': m['n_chains'],
                })
            except Exception as e:
                continue
    if not rows: return None

    n_seeds = len(rows)
    ess_vals = [r['ess_min'] for r in rows]
    epg_vals = [r['ess_min'] / r['n_grad_prod'] for r in rows]
    wall_vals = [r['wall_s'] / 3600 for r in rows]
    mL_vals = [r['mean_L'] for r in rows]
    acc_vals = [r['acc'] for r in rows if r['acc'] is not None]

    return {
        'label': label, 'n_seeds': n_seeds,
        'ess_med': float(np.median(ess_vals)),
        'ess_seeds': ess_vals,
        'epg_med': float(np.median(epg_vals)),
        'epg_seeds': epg_vals,
        'wall_med_h': float(np.median(wall_vals)),
        'mL_mean': float(np.mean(mL_vals)),
        'acc_mean': float(np.mean(acc_vals)) if acc_vals else None,
        'div_total': sum(r['div'] for r in rows if r['div']),
        'N': rows[0]['N'], 'warmup': rows[0]['warmup'], 'n_chains': rows[0]['n_chains'],
    }


# Run the audit for each section of both LaTeX files
def fmt(r):
    if r is None: return '   ---   no data ---'
    seeds_str = '/'.join(f'{int(v)}' for v in r['ess_seeds'])
    acc = f'{r["acc_mean"]:.2f}' if r['acc_mean'] is not None else ' --- '
    return (f'  n_seeds={r["n_seeds"]}  N={r["N"]} ch={r["n_chains"]}  '
            f'wall={r["wall_med_h"]:.1f}h  L̄={r["mL_mean"]:.1f}  acc={acc}  '
            f'ess_seeds=[{seeds_str}]  ess_med={r["ess_med"]:.0f}  '
            f'ess/grad={r["epg_med"]:.3e}  div={r["div_total"]}')

print('=' * 95)
print('FAIR-NO-WS multi-seed (init_scale=0.5, no stretch warmup, N=1000) — main paper table')
print('=' * 95)
configs = [
    ('peaches',                          ['astro/results/des_y1/peaches_init05_no_ws']),
    ('peams',                            ['astro/results/des_y1/peams_init05_no_ws']),
    ('peanuts (mtd=5)',                  ['astro/results/des_y1/peanuts_init05_no_ws']),
    ('PICKLES γ=2 default',              ['astro/results/des_y1/pickles_init05_no_ws']),
    ('PICKLES γ=0.5 max_L=20',           ['astro/results/des_y1/pickles_g05_maxL20_init05']),
    ('PICKLES γ=0.5 max_L=50',           ['astro/results/des_y1/pickles_g05_maxL50_init05']),
    ('PICKLES γ=0.5 max_L=128',          ['astro/results/des_y1/pickles_g05_init05_no_ws']),
    ('chess (tuned)',                    ['astro/results/des_y1/chess_init05_no_ws']),
    ('malt L=50 γ=0.5 (16 chains)',     ['astro/results/des_y1/malt_L50g05_init05_no_ws']),
    ('emcee/stretch',                   ['astro/results/des_y1/stretch']),
    ('ALDI',                             ['astro/results/des_y1/aldi_corrected']),
    ('langevin_walk (MALA)',            ['astro/results/des_y1/langevin_walk']),
    ('NUTS-diag',                        ['astro/results/des_y1/nuts_numpyro_diag']),
    ('NUTS-block',                       ['astro/results/des_y1/nuts_numpyro_block']),
    ('NUTS-dense',                       ['astro/results/des_y1/nuts_numpyro_dense']),
]
for label, dirs in configs:
    r = report(label, dirs)
    print(f'{label:<32s} {fmt(r)}')

print()
print('=' * 95)
print('Long production runs (N=5000, single seed) — table 2 in paper')
print('=' * 95)
long_configs = [
    ('PICKLES γ=0.5 max_L=20',  ['astro/results/des_y1/pickles_g05_init0.5_N5k_no_ws']),
    ('PICKLES γ=2 default',     ['astro/results/des_y1/pickles_init0.5_N5k_no_ws']),
    ('peaches',                 ['astro/results/des_y1/peaches_init0.5_N5k_no_ws']),
    ('peams',                   ['astro/results/des_y1/peams_init0.5_N5k_no_ws']),
    ('NUTS-dense (4 ch seq)',   ['astro/results/des_y1/nuts_numpyro_dense_N5k']),
]
for label, dirs in long_configs:
    r = report(label, dirs)
    print(f'{label:<32s} {fmt(r)}')

print()
print('=' * 95)
print('Warmup sweep (single seed, init=0.5, N=1000)')
print('=' * 95)
sweep_configs = [
    ('peaches warm=50',     ['astro/results/des_y1/peaches_init0.5_warm50_no_ws']),
    ('peaches warm=100',    ['astro/results/des_y1/peaches_init0.5_warm100_no_ws']),
    ('peaches warm=200',    ['astro/results/des_y1/peaches_init0.5_warm200_no_ws']),
    ('peams warm=50',       ['astro/results/des_y1/peams_init0.5_warm50_no_ws']),
    ('peams warm=100',      ['astro/results/des_y1/peams_init0.5_warm100_no_ws']),
    ('peams warm=200',      ['astro/results/des_y1/peams_init0.5_warm200_no_ws']),
    ('pickles γ=2 warm=50', ['astro/results/des_y1/pickles_init0.5_warm50_no_ws']),
    ('pickles γ=2 warm=100',['astro/results/des_y1/pickles_init0.5_warm100_no_ws']),
    ('pickles γ=2 warm=200',['astro/results/des_y1/pickles_init0.5_warm200_no_ws']),
    ('NUTS-dense warm=50',  ['astro/results/des_y1/nuts_numpyro_dense_warm50']),
    ('NUTS-dense warm=100', ['astro/results/des_y1/nuts_numpyro_dense_warm100']),
    ('NUTS-dense warm=200', ['astro/results/des_y1/nuts_numpyro_dense_warm200']),
    ('NUTS-block warm=50',  ['astro/results/des_y1/nuts_numpyro_block_warm50']),
    ('NUTS-block warm=100', ['astro/results/des_y1/nuts_numpyro_block_warm100']),
    ('NUTS-block warm=200', ['astro/results/des_y1/nuts_numpyro_block_warm200']),
]
for label, dirs in sweep_configs:
    r = report(label, dirs)
    print(f'{label:<32s} {fmt(r)}')

print()
print('=' * 95)
print('ALDI / pickles_unadjusted — long-budget runs (matched to Metropolized PICKLES)')
print('=' * 95)
unadj_configs = [
    ('ALDI N=1000',                     ['astro/results/des_y1/aldi_corrected']),
    ('ALDI N=20000 (long, init=0.5)',   ['astro/results/des_y1/aldi_long_init05']),
    ('ALDI ε=0.01',                     ['astro/results/des_y1/aldi_ss0.01_init05']),
    ('ALDI ε=0.05',                     ['astro/results/des_y1/aldi_ss0.05_init05']),
    ('ALDI ε=0.2',                      ['astro/results/des_y1/aldi_ss0.2_init05']),
    ('ALDI ε=0.5',                      ['astro/results/des_y1/aldi_ss0.5_init05']),
    ('pickles_unadjusted N=10000 long', ['astro/results/des_y1/pickles_unadjusted_long_init05']),
    ('pickles_unadjusted ε=0.1',        ['astro/results/des_y1/pickles_unadjusted_ss01_init05']),
    ('pickles_unadj init=0.1 ws=0 (existing)', ['astro/results/des_y1/pickles_unadj_init_init01_ws0']),
    ('pickles_unadj init=1.0 ws=0 (existing)', ['astro/results/des_y1/pickles_unadj_init_init1_ws0']),
]
for label, dirs in unadj_configs:
    r = report(label, dirs)
    print(f'{label:<40s} {fmt(r)}')
