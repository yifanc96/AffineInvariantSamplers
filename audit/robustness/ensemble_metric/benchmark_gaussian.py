import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ap=argparse.ArgumentParser()
ap.add_argument('--dim',type=int,required=True)
ap.add_argument('--walkers',type=int,required=True)
ap.add_argument('--seed',type=int,required=True)
ap.add_argument('--metric',choices=['complement','full'],required=True)
args=ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
import numpy as np
from benchmark_stretch import _mean_trace_ess

root=Path(__file__).resolve().parent
path=root/('full_metric_peaches.py' if args.metric=='full' else 'reference/peaches.py')
spec=importlib.util.spec_from_file_location('cell_sampler',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
max_initial=jax.random.normal(jax.random.key(42+args.seed),(4*args.dim,args.dim))
initial=max_initial[:args.walkers]
initial_np=np.asarray(initial)
kwargs=dict(warmup=200,seed=args.seed,verbose=False,
            chees_metric='affine-invariant-full' if args.metric=='full' else 'affine-invariant')
log_prob=lambda x:-.5*jnp.sum(x*x,axis=-1)
started=time.perf_counter()
samples,info=module.sampler_peaches(log_prob,initial,500,**kwargs)
samples.block_until_ready();wall=time.perf_counter()-started
x=np.asarray(samples,dtype=np.float64)
mean=x.mean(axis=(0,1));var=x.var(axis=(0,1))
traces=np.concatenate([x.mean(axis=1),(x*x).mean(axis=1)],axis=1)
stats=np.asarray([_mean_trace_ess(t) for t in traces.T])
rank_records=[]
for index in [0,100,200,300,400,499]:
    state=x[index];half=args.walkers//2
    for points in [state[:half],state[half:],state]:
        C=np.cov(points,rowvar=False);evals=np.linalg.eigvalsh(C)
        rank=np.linalg.matrix_rank(C)
        rank_records.append({'sample':index,'points':len(points),'rank':int(rank),
                             'condition':float(evals[-1]/max(evals[0],1e-300)),
                             'min_eigenvalue':float(evals[0])})
row=dict(dim=args.dim,walkers=args.walkers,seed=args.seed,metric=args.metric,
         initial_formula='jax.random.normal(key(42+seed),(4*D,D))[:N]',
         initial_sha256=hashlib.sha256(initial_np.tobytes()).hexdigest(),
         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         source_path=str(path),warmup=200,production=500,precision='float32',
         info=info,wall_seconds_including_jit=wall,
         mean_rms_error=float(np.sqrt(np.mean(mean**2))),
         variance_rms_error=float(np.sqrt(np.mean((var-1)**2))),
         variance_relative_error_max=float(np.max(np.abs(var-1))),
         coordinate_ensemble_mean_ess_min=float(stats[:args.dim,0].min()),
         squared_coordinate_ensemble_mean_ess_min=float(stats[args.dim:,0].min()),
         observable_ess_min=float(stats[:,0].min()),reliable_ess_fraction=float(stats[:,2].mean()),
         effective_draws_equivalent_per_million_gradients=float(args.walkers*stats[:,0].min()/info['n_grad_evals']*1e6),
         finite_samples=bool(np.isfinite(x).all()),rank_records=rank_records)
(root/'results').mkdir(exist_ok=True)
(root/'results'/f'gaussian_d{args.dim}_n{args.walkers}_seed{args.seed}_{args.metric}.json').write_text(json.dumps(row,indent=2))
print(json.dumps({key:row[key] for key in ['dim','walkers','seed','metric','mean_rms_error','variance_rms_error','observable_ess_min','effective_draws_equivalent_per_million_gradients','info']}),flush=True)
