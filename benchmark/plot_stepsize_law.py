"""Paper figure: acceptance-optimal step-size law eps* ~ d^{-1/4}.

Reads the controlled-experiment result (/tmp/stepscaling_result.npz: the
acceptance-optimal eps* at fixed trajectory length and fixed target
acceptance, measured directly via an eps-grid so no adapter confounds it)
and produces a clean log-log figure with the d^{-1/4} reference and the
least-squares fitted slope.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path("benchmark/figs"); FIGS.mkdir(parents=True, exist_ok=True)
dat = np.load("/tmp/stepscaling_result.npz")
d, e = dat["d"], dat["eps_star"]
slope, intc = np.polyfit(np.log(d), np.log(e), 1)
slope8 = np.polyfit(np.log(d[d >= 8]), np.log(e[d >= 8]), 1)[0]

fig, ax = plt.subplots(figsize=(6, 4.6))
ax.plot(d, e, "o-", color="#c0392b", lw=2, ms=8,
        label=f"peaches (ensemble HMC), fit slope {slope:+.3f}")
ref = e[0] * (d / d[0]) ** (-0.25)
ax.plot(d, ref, "k--", lw=1.3, alpha=0.8, label="theory  $d^{-1/4}$")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("dimension $d$")
ax.set_ylabel(r"acceptance-optimal step size $\epsilon^\star$")
ax.set_title("Step-size scaling on isotropic Gaussian N(0, I_d)\n"
             f"(fixed L, optimal step at acc=0.8; fit {slope:+.3f}, "
             f"d>=8: {slope8:+.3f}; theory -1/4)")
ax.grid(alpha=0.3, which="both"); ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "exp4_stepsize_law.png", dpi=150, bbox_inches="tight")
print("wrote", FIGS / "exp4_stepsize_law.png")
print(f"fitted slope all-d {slope:+.3f}, d>=8 {slope8:+.3f}  (theory -0.25)")
