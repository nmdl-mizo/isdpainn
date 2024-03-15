#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from ck_edge_maker.dataset import CK

# load dataset
dataset = CK("./dataset", energies=(288,310,256), directional=True)

df = pd.read_pickle("../data/analyzed/time_mol_castep_vs_isdpainn.pkl")
x = df["calc_time"]
y = df["time_gpu"]
c = dataset.n_site.tolist()

# make figure
fig = plt.figure(figsize=(10, 5))

# make gridspecs
gs1 = gridspec.GridSpec(2, 2, figure=fig, left=0.1, right=0.48, width_ratios=[7, 2], height_ratios=[2, 7])
gs2 = gridspec.GridSpec(2, 1, figure=fig, left=0.55, right=0.95)

# left
ax_main = fig.add_subplot(gs1[1, 0])
ax_histx = fig.add_subplot(gs1[0, 0], sharex=ax_main)
ax_histy = fig.add_subplot(gs1[1, 1], sharey=ax_main)

# scatter plot
scatter = ax_main.scatter(x, y, c=c, alpha=0.25, marker="o", cmap="Spectral", edgecolors="gray")
formatter_x = ticker.ScalarFormatter(useMathText=True)
formatter_x.set_scientific(True)
formatter_x.set_powerlimits((0,0))
ax_main.xaxis.set_major_formatter(formatter_x)
formatter_y = ticker.ScalarFormatter(useMathText=True)
formatter_y.set_scientific(True)
formatter_y.set_powerlimits((0,0))
ax_main.yaxis.set_major_formatter(formatter_y)

ax_main.set_xlabel(r"$t_\mathrm{DFT}$ (s)")
ax_main.set_ylabel(r"$t_\mathrm{GNN}$ (s)")
#ax_main.set_xlim(0,2e4)

# histogram x
ax_histx.hist(x, bins=100, color="gray")
ax_histx.set_ylabel("count")
ax_histx.tick_params(labelbottom=False)

# histogram y
ax_histy.hist(y, bins=100, orientation='horizontal', color="gray")
ax_histy.set_xlabel("count")
ax_histy.tick_params(labelleft=False)

# colorbar
cbar = plt.colorbar(scatter, orientation="vertical", ax=[ax_main, ax_histx, ax_histy])
cbar.set_label(r'$N_\mathrm{ex}$')

# right
ax_calc = fig.add_subplot(gs2[0, :])
ax_pred = fig.add_subplot(gs2[1, :], sharex=ax_calc, sharey=ax_main)

ax_calc.scatter(dataset.natoms.tolist(), x, c=dataset.n_site.tolist(), alpha=0.25, cmap="Spectral", edgecolors="gray")
ax_pred.scatter(dataset.natoms.tolist(), y, c=dataset.n_site.tolist(), alpha=0.25, cmap="Spectral", edgecolors="gray")
ax_calc.yaxis.set_major_formatter(formatter_y)
ax_calc.set_ylim(0,)
#ax_pred.set_ylim(0, 0.005)
ax_calc.set_xlim(0)
ax_calc.set_ylabel(r"$t_\mathrm{DFT}$ (s)")
ax_calc.tick_params(labelbottom=False)
ax_pred.set_ylabel(r"$t_\mathrm{GNN}$ (s)")
ax_pred.set_xlabel(r"$N$")

# add labels to the subplots
text_param = {
    "fontsize": 16,
    #"fontweight": "bold",
    "va": "top",
    "ha": "right",
}
# left
ax_histx.text(-0.2, 1.25, 'a', transform=ax_histx.transAxes, **text_param)

# right
ax_calc.text(-0.1, 1.1, 'b', transform=ax_calc.transAxes, **text_param)
ax_pred.text(-0.1, 1.1, 'c', transform=ax_pred.transAxes, **text_param)

fig.tight_layout()
fig.savefig("../figures/time_comparison.png", dpi=300)
