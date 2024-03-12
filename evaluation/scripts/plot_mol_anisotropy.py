#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import torch
import seaborn as sns
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
upload_id = "loddw6wwSbKmyTgsHuX6PQ"
base_dir = Path(f"../data/nomad")
zip_path = Path(base_dir, f"{upload_id}.zip")

# extract the zip file
calc_base_dir = Path(zip_path.parent, "benzene_rotation_x")
if not calc_base_dir.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)

# parameters for plot
plot_direction = "x"#"all"
energies_common = np.linspace(288, 310, 256)
# sp_mean of CK dataset for scaling prediction
sp_mean = 0.00011704213102348149

# get model
model = get_model(config_path=config_path, model_state_path=model_state_path)

# get spectra
directions = "x"
spectra_true_dict = {}
spectra_pred_dict = {}
for direction in directions:
    rotation_dir = base_dir / f"benzene_rotation_{direction}"
    angle_dir_list = [p for p in Path(rotation_dir).glob("angle_*") if p.is_dir()]
    angle_list = np.loadtxt(rotation_dir / "angle_list.txt")
    y_true_mol, y_pred_mol = get_true_pred_spectra(angle_dir_list, model, energies_common, sp_mean, aggregate=True)
    spectra_true_dict[direction] = y_true_mol
    spectra_pred_dict[direction] = y_pred_mol


for direction in "xyz":
    rotation_dir = base_dir / f"rotation_{direction}"
    angle_dir_list = [p for p in Path(rotation_dir).glob("angle_*") if p.is_dir()]
    for angle_dir in angle_dir_list:
        err_list = [
            err
            for p in Path(angle_dir, "sites").glob("*")
            if p.is_dir() and p.name.isnumeric()
            for err in p.glob("*.err")
        ]
        if len(err_list) > 0:
            raise ValueError(f"Error file found: {err_list}")


skip = 1
color_values = angle_list / (np.max(angle_list[::skip]) + 18)
assert isinstance(color_values, np.ndarray)

cmap = sns.hls_palette(s=0.5, as_cmap=True)

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(6, 9), constrained_layout=True)
for i_direction, ax in enumerate(axes):
    y_pred_mol = spectra_pred_dict[plot_direction]
    y_true_mol = spectra_true_dict[plot_direction]
    for i, (s, s_pred) in enumerate(zip(y_true_mol[::skip], y_pred_mol[::skip])):
        c = cmap(color_values[i])
        if i == 0:
            label_calc = "0 deg (calc.)"
            label_pred = "0 deg (pred.)"
        else:
            label_calc = f"{angle_list[::skip][i]:.0f} deg (calc.)"
            label_pred = f"{angle_list[::skip][i]:.0f} deg (pred.)"
        ax.plot(energies_common, s[i_direction], color=c, ls="--", alpha=0.4, label=label_calc)
        ax.plot(energies_common, s_pred[i_direction], color=c, ls="-", alpha=1.0, label=label_pred)
    ax.set_xlim(energies_common[0], energies_common[-1])
    ax.set_ylabel("Intensity (a.u.)")
    #ax.legend(title="$\hat{n}\parallel " + f"{'xyz'[i_direction]}$", ncol=2, loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    order = [i for i in range(0, 12, 2)] + [i for i in range(1, 12, 2)]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        title=r"$\hat{{\mathrm{{n}}}} \parallel " + 'xyz'[i_direction] + "$",
        ncol=2,
        loc="upper right"
    )

# get max value of sp_list and y_pred for ylim
max_val = np.max(np.vstack([y_true_mol, y_pred_mol]))
ax.set_ylim(0., max_val*1.1)
ax.set_xlabel("Energy (eV)")

# save fig
fig.savefig(Path(figure_dir, f"rotated_benzene_{plot_direction}.png"), dpi=300)
fig.savefig(Path(figure_dir, f"rotated_benzene_{plot_direction}.svg"))
