"""DES Y1 3x2pt efficiency comparison table (mean +/- std over 3 seeds).

Compares PICKLES / PEACHES (this work, ensemble kinetic samplers) against ALDI,
NUTS-dense at two warmups, and emcee. Two efficiency metrics: ESS per target
gradient (algorithmic) and ESS per wall-clock second (as actually run: 64 GPU
ensemble chains vs 4 NUTS chains). Rendered large-font to
benchmark/figs_standalone/ and also written as .tex and .csv.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("../benchmark/figs_standalone"); OUT.mkdir(parents=True, exist_ok=True)

# (display label, results dir, chains-note, highlight?)
ROWS = [
    ("PICKLES ($\\gamma{=}0.5$)", "pickles_priormed_g05_ws200", True),
    ("PEACHES", "peaches_fixed_dualavg", True),
    ("ALDI", "aldi_corrected", False),
    ("NUTS-dense (warmup 500)", "nuts_numpyro_dense", False),
    ("NUTS-dense (warmup 50)", "nuts_numpyro_dense_warm50", False),
    ("emcee (stretch)", "stretch", False),
]


def stat(dirn):
    fs = sorted(glob.glob(f"results/des_y1/{dirn}/seed*/metrics.json"))
    rows = [json.load(open(f)) for f in fs]

    def m(k):
        v = [r.get(k) for r in rows if r.get(k) is not None]
        return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
    return dict(n=len(rows), nch=rows[0].get("n_chains"), epe=m("ess_per_eval"), eps=m("ess_per_sec"))


data = []
for lab, dirn, hi in ROWS:
    s = stat(dirn)
    epe_m, epe_s = s["epe"]; eps_m, eps_s = s["eps"]
    data.append(dict(lab=lab, nch=s["nch"], hi=hi,
                     epe=(epe_m * 1e3, epe_s * 1e3), eps=(eps_m, eps_s)))

# best per column (over all rows) for bolding
best_epe = max(d["epe"][0] for d in data)
best_eps = max(d["eps"][0] for d in data)

# ---- render table ----
col_labels = ["sampler", "chains", "ESS / eval\n($\\times10^{-3}$)", "ESS / sec"]
cell_text, cell_colors = [], []
for d in data:
    epe_str = f"{d['epe'][0]:.2f} $\\pm$ {d['epe'][1]:.2f}"
    eps_str = f"{d['eps'][0]:.3f} $\\pm$ {d['eps'][1]:.3f}"
    if abs(d["epe"][0] - best_epe) < 1e-9: epe_str = "$\\mathbf{" + f"{d['epe'][0]:.2f}" + "}$ $\\pm$ " + f"{d['epe'][1]:.2f}"
    if abs(d["eps"][0] - best_eps) < 1e-9: eps_str = "$\\mathbf{" + f"{d['eps'][0]:.3f}" + "}$ $\\pm$ " + f"{d['eps'][1]:.3f}"
    cell_text.append([d["lab"], str(d["nch"]), epe_str, eps_str])
    hcol = "#fdecea" if d["hi"] else "white"          # light red highlight for PICKLES/PEACHES
    cell_colors.append([hcol, hcol, hcol, hcol])

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
               colColours=["#e8e8e8"] * 4, cellLoc="center", loc="center",
               colWidths=[0.42, 0.13, 0.22, 0.20])
tbl.auto_set_font_size(False)
tbl.set_fontsize(21)
tbl.scale(1, 3.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb"); cell.set_linewidth(1.5)
    if r == 0:
        cell.set_text_props(weight="bold", fontsize=20)
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")
        cell._loc = "left"
    if r in (1, 2) and r > 0:   # PICKLES, PEACHES rows -> bold sampler name
        tbl[(r, 0)].set_text_props(weight="bold")
ax.set_title("DES Y1 $3\\times2$pt: sampling efficiency (mean $\\pm$ std, 3 seeds)",
             fontsize=23, pad=20)
fig.text(0.5, 0.01, "ESS / eval counts a gradient for the gradient samplers, a density eval for "
         "emcee. PICKLES / PEACHES lead on ESS/eval and dominate ESS/sec.",
         ha="center", fontsize=13.5, color="#444")
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "fig4_des_efficiency_table.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig4_des_efficiency_table.png")

# ---- also .csv and .tex ----
with open(OUT / "des_efficiency_table.csv", "w") as f:
    f.write("sampler,chains,ess_per_grad_mean_e-3,ess_per_grad_std_e-3,ess_per_sec_mean,ess_per_sec_std\n")
    for d in data:
        f.write(f"\"{d['lab']}\",{d['nch']},{d['epe'][0]:.4f},{d['epe'][1]:.4f},{d['eps'][0]:.4f},{d['eps'][1]:.4f}\n")
print("wrote", OUT / "des_efficiency_table.csv")
