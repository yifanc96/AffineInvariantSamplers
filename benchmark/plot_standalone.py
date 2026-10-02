"""Three standalone figures (large fonts), peaches labelled "ensemble kinetic".

  fig1_conditioning_gaussian.png : ill-conditioned Gaussian d=50 vs kappa --
                                   ensemble kinetic vs emcee vs NUTS(no mass adapt.)
  fig2_scaling_gaussian.png      : dimensional scaling N(0,I_d) -- ensemble kinetic vs emcee
  fig3_warmup_rosenbrock.png     : warmup robustness on Rosenbrock d=8 -- ensemble kinetic vs NUTS

Written to benchmark/figs_standalone/. Median/band over CONVERGED seeds only.
"""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("benchmark/figs_standalone"); OUT.mkdir(parents=True, exist_ok=True)
RH = 1.03
PK = "ensemble kinetic"   # how peaches is captioned

plt.rcParams.update({
    "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 25,
    "legend.fontsize": 22, "xtick.labelsize": 22, "ytick.labelsize": 22,
    "lines.linewidth": 4.5, "lines.markersize": 15,
    "axes.linewidth": 2.4, "xtick.major.width": 2.4, "ytick.major.width": 2.4,
    "xtick.major.size": 9, "ytick.major.size": 9, "grid.linewidth": 1.3,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.borderpad": 0.4,
})


# ---------- shared readers (converged-median) ----------
def read_runs(pattern, tag, metric="ess_per_grad"):
    out = defaultdict(list)
    for f in glob.glob(pattern):
        r = json.load(open(f))
        if tag is not None and r.get("tag") != tag:
            continue
        v = r.get(metric)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        rh = r.get("rhat_max")
        out[r["dim"]].append((v, rh if (rh is not None and np.isfinite(rh)) else None))
    return out


def merge(*sources):
    m = {}
    for src in sources:
        for d, rec in src.items():
            if d not in m and rec:
                m[d] = rec
    return m


def to_arrays(m):
    ds = sorted(m)
    y, lo, hi, conv = [], [], [], []
    for d in ds:
        pairs = m[d]
        cv = [v for v, rh in pairs if rh is None or rh <= RH]
        use = cv if cv else [v for v, _ in pairs]
        y.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return np.array(ds, float), np.array(y), np.array(lo), np.array(hi), np.array(conv)


def draw(ax, d, y, lo, hi, conv, c, mk, lab, ls="-"):
    ax.plot(d, y, color=c, lw=4.5, ls=ls, zorder=3, label=lab)
    ax.fill_between(d, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(d[cm], y[cm], color=c, marker=mk, s=200, zorder=4)
    if (~cm).any():
        ax.scatter(d[~cm], y[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=200, zorder=4, linewidths=2.8)


def series(specs):
    xs, med, lo, hi, conv = [], [], [], [], []
    for x, tgt, samp, tag in specs:
        pairs = []
        for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}/seed*/metrics.json"):
            r = json.load(open(f)); eg = r.get("ess_per_grad"); rh = r.get("rhat_max")
            if eg is None or not np.isfinite(eg):
                continue
            pairs.append((eg, rh if (rh is not None and np.isfinite(rh)) else None))
        if not pairs:
            continue
        cv = [v for v, rh in pairs if rh is None or rh <= RH]
        use = cv if cv else [v for v, _ in pairs]
        nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
        xs.append(x); med.append(np.median(use)); lo.append(min(use)); hi.append(max(use))
        conv.append(nconv >= (len(pairs) + 1) // 2)
    return (np.array(xs, float), np.array(med), np.array(lo), np.array(hi), np.array(conv))


def agg_w(tgt, samp, tag, w):
    pairs = []
    for f in glob.glob(f"benchmark/results/{tgt}/{samp}__{tag}_w{w}/seed*/metrics.json"):
        r = json.load(open(f)); eg = r.get("ess_per_grad"); rh = r.get("rhat_max")
        if eg is None or not np.isfinite(eg):
            continue
        pairs.append((eg, rh if (rh is not None and np.isfinite(rh)) else None))
    if not pairs:
        return None
    cv = [v for v, rh in pairs if rh is None or rh <= RH]
    use = cv if cv else [v for v, _ in pairs]
    nconv = sum(1 for v, rh in pairs if rh is not None and rh <= RH)
    return (np.median(use), np.percentile(use, 25), np.percentile(use, 75),
            nconv >= (len(pairs) + 1) // 2)


def gauss_scale(samp, metric="ess_per_grad"):
    b = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleLb/seed*/metrics.json", "scaleLb", metric)
    sL = read_runs(f"benchmark/results/gaussian_d*/{samp}__scaleL/seed*/metrics.json", "scaleL", metric)
    e4 = read_runs(f"benchmark/results/gaussian_d*/{samp}__exp4c/seed*/metrics.json", "exp4c", metric)
    return to_arrays(merge(b, sL, e4))


def fixed_int(d, y, conv, slope, dmin):
    m = (d >= dmin) & conv.astype(bool)
    if m.sum() == 0:
        m = d >= dmin
    return np.mean(np.log(y[m]) - slope * np.log(d[m]))


def plot_s(ax, x, med, lo, hi, conv, c, mk, ls, lab):
    if len(x) == 0:
        return
    ax.plot(x, med, color=c, ls=ls, lw=4.5, zorder=3, label=lab)
    ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(x[cm], med[cm], color=c, marker=mk, s=200, zorder=4)
    if (~cm).any():
        ax.scatter(x[~cm], med[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=200, zorder=4, linewidths=2.8)


RED, GREEN, DARK, BLUE = "#c0392b", "#27ae60", "#2c3e50", "#2980b9"

# ===== Figure 1: ill-conditioned Gaussian, kinetic vs emcee vs NUTS(no mass) =====
fig, ax = plt.subplots(figsize=(8, 6.3))
KAP = [2, 4, 6, 8]
for samp, tag, c, mk, ls, lab in [
        ("peaches", "exp2_lc{}", RED, "o", "-", PK),
        ("stretch", "exp2_lc{}", GREEN, "D", "-", "emcee"),
        ("nuts", "idmass_lc{}", DARK, "v", "--", "NUTS (no mass adapt.)")]:
    x, med, lo, hi, conv = series([(k, "gaussian_d50", samp, tag.format(k)) for k in KAP])
    plot_s(ax, x, med, lo, hi, conv, c, mk, ls, lab)
ax.set_yscale("log"); ax.set_xticks(KAP)
ax.set_xlabel(r"$\log\kappa$   (condition number $\kappa=e^{\log\kappa}$)")
ax.set_ylabel("ESS / target evaluation")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, columnspacing=1.3,
          handlelength=1.7, fontsize=20)
ax.set_title(r"Ill-conditioned Gaussian ($d=50$)")
fig.savefig(OUT / "fig1_conditioning_gaussian.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig1_conditioning_gaussian.png")

# ===== Figure 2: dimensional scaling on Gaussian, kinetic vs emcee =====
fig, ax = plt.subplots(figsize=(8, 6.3))
dp, yp, lop, hip, cp = gauss_scale("peaches")
draw(ax, dp, yp, lop, hip, cp, RED, "o", PK)
de, ye, loe, hie, ce = gauss_scale("stretch")
draw(ax, de, ye, loe, hie, ce, DARK, "v", "emcee")
dd = np.array([dp.min(), dp.max()])
ax.plot(dd, np.exp(fixed_int(dp, yp, cp, -0.25, 64)) * dd ** -0.25, "k--", lw=3.4, alpha=0.65,
        label=r"theory $d^{-1/4}$")
ee = np.array([de.min(), de.max()])
ax.plot(ee, np.exp(fixed_int(de, ye, ce, -1.0, 16)) * ee ** -1.0, "k:", lw=3.4, alpha=0.65,
        label=r"theory $d^{-1}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("dimension $d$"); ax.set_ylabel("ESS / target evaluation")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, columnspacing=1.4,
          handlelength=1.8, fontsize=21)
ax.set_title(r"Dimensional scaling, $N(0,I_d)$")
fig.savefig(OUT / "fig2_scaling_gaussian.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig2_scaling_gaussian.png")

# ===== Figure 3: warmup robustness on Rosenbrock d=8, kinetic vs NUTS =====
fig, ax = plt.subplots(figsize=(8, 6.3))
WARMS = [50, 100, 200, 500, 1000]
for samp, c, mk, ls, lab in [("peaches", RED, "o", "-", PK), ("nuts", DARK, "v", "--", "NUTS-dense")]:
    xs, med, lo, hi, conv = [], [], [], [], []
    for w in WARMS:
        a = agg_w("rosenbrock_d8", samp, "exp3b", w)
        if a:
            xs.append(w); med.append(a[0]); lo.append(a[1]); hi.append(a[2]); conv.append(a[3])
    xs = np.array(xs, float); med = np.array(med); conv = np.array(conv)
    ax.plot(xs, med, color=c, ls=ls, lw=4.5, zorder=3, label=lab)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
    cm = conv.astype(bool)
    ax.scatter(xs[cm], med[cm], color=c, marker=mk, s=200, zorder=4)
    if (~cm).any():
        ax.scatter(xs[~cm], med[~cm], facecolors="white", edgecolors="gray", marker=mk,
                   s=200, zorder=4, linewidths=2.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(WARMS); ax.set_xticklabels(WARMS)
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("warmup iterations"); ax.set_ylabel("ESS / target evaluation")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, columnspacing=1.4,
          handlelength=1.8, fontsize=21)
ax.set_title(r"Warmup robustness, Rosenbrock ($d=8$)")
fig.savefig(OUT / "fig3_warmup_rosenbrock.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig3_warmup_rosenbrock.png")
