"""
samplers
========

Single-chain (batched) HMC-family samplers, separated out from the main
``affine_invariant_samplers`` package.  The functions are re-exported by
``affine_invariant_samplers`` for backwards compatibility, so user code
that does ``from affine_invariant_samplers import sampler_nuts`` keeps
working unchanged.

For direct imports:

>>> from samplers import sampler_malt, sampler_mams, sampler_nuts
"""

from .malt  import sampler_malt
from .mams  import sampler_mams
from .nuts  import sampler_nuts
from .chees import sampler_chees

__all__ = ["sampler_malt", "sampler_mams", "sampler_nuts", "sampler_chees"]
