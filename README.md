# AffineInvariantSamplers

Paper: https://arxiv.org/abs/2505.02987

mcmc samplers are all in `samplers.py`

Run experiments (can first set a small number of iterations first, to see how the output looks like)
`experiments_Gaussian.py`
`experiments_ring.py`
`experiments_Allen-Cahn.py`


`test-experiments_xxxx.ipynb` These are for generating figures in the paper, based on obtained samples using the above .py experiments (in particular, the slurm version which is run on cpu clusters)


```
@article{chen2025new,
  title={New affine invariant ensemble samplers and their dimensional scaling},
  author={Chen, Yifan},
  journal={arXiv preprint arXiv:2505.02987},
  year={2025}
}
```
