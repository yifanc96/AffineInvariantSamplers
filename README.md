# AffineInvariantSamplers

Paper: https://arxiv.org/abs/2505.02987

MCMC samplers are all in `samplers.py`

The purpose of this project is to test the performance of affine-invariant, gradient aware Markov Chain Monte Carlo (MCMC) sampler algorithms.

While the standard ensemble samplers (such as the stretch move native to the emcee package) do well on many distributions, they struggle on high-dimensional distributions where the typical set is extremely small. This is often alleviated by using gradient aware "Hamiltonian Monte Carlo" (HMC) samplers, which (as the name suggests) uses the local gradient information and induced "momentum" to trace the typical sets, in analogy to Hamiltonian dynamics. By approximately tracing these typical sets, an HMC algorithm can perform better on higher-dimensional probability distributions where gradient unaware samplers fail. However, HMC samplers require a great deal of preconditioning to handle highly anisotropic distributions, thus negating the potential computational benefits of tracing the typical set.

By 


`test-experiments_xxxx.ipynb` These are for generating figures in the paper, based on obtained samples using the above .py experiments (in particular, the slurm version which is run on cpu clusters)




```
@article{chen2025new,
  title={New affine invariant ensemble samplers and their dimensional scaling},
  author={Chen, Yifan},
  journal={arXiv preprint arXiv:2505.02987},
  year={2025}
}
```
