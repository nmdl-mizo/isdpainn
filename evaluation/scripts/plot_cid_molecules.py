#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import torch
from analysis_utils import get_model
from castep_spectra_utils import get_true_pred_spectra
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")

# figure dir
figure_dir = Path("../figures")

# parameters for model
data_dir = Path("../data/model")
config_path=Path(data_dir, "config-defaults_random_split.yaml")
model_state_path=Path(data_dir, "model_state_random_split.pt")

# parameters for data
base_dir = Path(f"../data/nomad")
zip_path = Path(base_dir, "aromatic_amino_acids.zip")

# extract the zip file
calc_base_dir = Path(zip_path.parent, "aromatic_amino_acids_elnes")
if not calc_base_dir.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)

# parameters for data
sites_dir_name = ""
calc_dir_dict = {
    "phenylalanine": Path(calc_base_dir, "phenylalanine"),
    "tyrosine": Path(calc_base_dir, "tyrosine"),
    "tryptophan": Path(calc_base_dir, "tryptophan"),
    "histidine": Path(calc_base_dir, "histidine"),
}

# parameters for plot
fig_size = (8, 8)
nrows = 2
ncols = 2
energies_common = np.linspace(288, 310, 256)
# sp_mean of CK dataset for scaling prediction
sp_mean = 0.00011704213102348149

# get model
model = get_model(config_path=config_path, model_state_path=model_state_path)

# get spectra
y_true_mol, y_pred_mol = get_true_pred_spectra(
    list(calc_dir_dict.values()),
    model, energies_common, sp_mean, aggregate=True,
    sites_dir_name=sites_dir_name
)

# plot molecular spectra
fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=fig_size, sharex=True, sharey=True)
for i_mol, (ax, (label_mol, mol_data)) in enumerate(zip(axes.flatten(), calc_dir_dict.items())):
    for i_direction in range(3):
        c = "rgb"[i_direction]
        label = f"{'xyz'[i_direction]}"
        ax.plot(
            energies_common, 
            y_true_mol[i_mol, i_direction].T,
            label=f"{label}_calc",
            c=c, ls="--", alpha=0.2
        )
        #ax.scatter(
        #    energies_common, 
        #    y_true_mol[i_mol, i_direction].T,
        #    label=f"{label}_calc",
        #    c=c, alpha=0.1, marker="."
        #)
        ax.plot(
            energies_common, 
            y_pred_mol[i_mol, i_direction].T,
            label=f"{label}_pred",
            c=c, ls="-", alpha=0.8
        )
    ax.set_title(label_mol)
# set legend for the first axes
axes.flatten()[0].legend(loc="upper right")
# set xlabel and ylabel
for ax in axes[-1]:
    ax.set_xlabel("Energy (eV)")
for ax in axes.T[0]:
    ax.set_ylabel("Intensity (a.u.)")
# set xlim and ylim
ax.set_xlim(energies_common.min(), energies_common.max())
y_max = max(y_true_mol.max(), y_pred_mol.max())
ax.set_ylim(0, y_max * 1.1)

# save fig
fig.savefig(Path(figure_dir, f"mol_elnes_cid_molecules.png"), dpi=300)
fig.savefig(Path(figure_dir, f"mol_elnes_cid_molecules.svg"))
