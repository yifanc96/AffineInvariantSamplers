import os
from pathlib import Path
import subprocess
import sys
base=[sys.executable,str(Path(__file__).resolve().parent/'benchmark_gaussian.py')]
for dim in [32,64]:
    for seed in [0,1]:
        for walkers in [2*dim,2*dim+2,2*dim+4,4*dim]:
            for metric in (['complement','full'] if walkers==2*dim else ['complement']):
                cmd=base+['--dim',str(dim),'--walkers',str(walkers),'--seed',str(seed),'--metric',metric]
                result=subprocess.run(cmd,env={**os.environ,'JAX_PLATFORMS':'cpu'})
                if result.returncode:sys.exit(result.returncode)
