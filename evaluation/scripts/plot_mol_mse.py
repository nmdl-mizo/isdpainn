#!/usr/bin/env python
from pathlib import Path
from tqdm import tqdm
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum
from train_utils import prepare_dataset
from wandb.sdk.lib.config_util import dict_from_config_file
from analysis_utils import get_model

# figure dir
figure_dir = Path("../figures")

# parameters for model
data_dir = Path("../data/model")
config_path=Path(data_dir, "config-defaults_random_split.yaml")
model_state_path=Path(data_dir, "model_state_random_split.pt")

batch_size = 128
config = dict_from_config_file(str(config_path))
assert config is not None

# parepare model
model = get_model(config_path=config_path, model_state_path=model_state_path)

# prepare dataset split
config["equiv_site"] = True
dataset_dict = prepare_dataset(config, as_dict=True)
assert isinstance(dataset_dict, dict)
dl = DataLoader(
    dataset_dict["test"],
    batch_size=batch_size,
    shuffle=False
)

# calculate mse
mse_list = []
directions = torch.eye(3, device=DEVICE)
with torch.no_grad():
    for data in tqdm(dl):
        #data = data.to(DEVICE)
        sp_calc = data.spectra

        # predict spectrum
        sp_pred = torch.stack(
            [
                model.forward(data, direction=direction)
                for direction in directions
            ]
        ).swapaxes(0, 1)

        # molecular spectrum
        sp_mol_calc = scatter_sum(
#            torch.einsum(
#                "ijk,i->ijk",
#                sp_calc, data.multiplicity
#            ),
            sp_calc,
            data.batch,
            dim=0
        ).cpu()
        sp_mol_pred = scatter_sum(
#            torch.einsum(
#                "ijk,i->ijk",
#                sp_pred,
#                data.multiplicity
#            ),
            sp_pred * data.node_mask.unsqueeze(-1).unsqueeze(-1),
            data.batch,
            dim=0
        ).cpu()

        # calc mse error between sp_mol_calc and sp_mol_pred
        mse = torch.mean((sp_mol_calc - sp_mol_pred)**2, dim=-1)

        mse_list.append(mse)
mse_list = torch.cat(mse_list)

# save mse_list to mse_mol.pt
torch.save(mse_list, Path("../data/analyzed/mse_mol.pt"))
# shape: (n_site_test, 3)

# get index for 0, 25, 50, 75, 100 percentiles of mse
percentiles = [0., 0.25, 0.50, 0.75, 1.]
typical_mol_index = torch.stack([
    torch.argsort(mse_list.flatten())[int((len(mse_list.flatten()) - 1) * p)]
    for p in percentiles
]) // 3
# save index list to typical_index_mol.pt
torch.save(typical_mol_index, Path("../data/analyzed/typical_mol_index.pt"))

# plot sorted_mse
from plot_utils import plot_sorted_mse
import matplotlib.pyplot as plt
fig, ax = plot_sorted_mse(
    {
        r"$\sum_nS_n(\mathcal{G}, \hat{\mathbf{n}})$": {"mse": mse_list.cpu().detach()}
    },
    alpha=0.25, marker=".", c="r"
)
fig.savefig(Path(figure_dir, "sorted_mol_mse.png"), dpi=300)
plt.close(fig)

## plot molecular spectra
#dataset = dataset_dict["test"]
#typical_id_mol = dataset[typical_mol_index].id_mol
#print(typical_id_mol)
#with_left_margin = True
#from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
#import numpy as np
#from draw_mol import process_mol, mol2svg, apply_inset_scale_shift, qm9_data2mol
#from svgutils.compose import SVG, Figure
#
#for id_mol in typical_id_mol:
#    index_mol = (dataset.id_mol == id_mol).nonzero().item()
#    # prepare mol data
#    data = Batch.from_data_list([dataset[index_mol],]).to(DEVICE)
#    directions = torch.eye(3).to(DEVICE)
#    n_site = data.node_mask.sum().item()
#
#    # prepare spectrum
#    sp_calc = data.spectra[data.node_mask]
#
#    # predict spectrum
#    with torch.no_grad():
#        sp_pred = torch.stack([
#            model.forward(data, direction=directions[i])
#            for i in range(3)
#        ]).swapaxes(0, 1)[data.node_mask]
#
#    # molecular spectrum
#    sp_mol_calc = sp_calc.sum(dim=0).cpu()
#    # must filter out node_mask for sp_pred
#    sp_mol_pred = sp_pred.sum(dim=0).cpu()
##    sp_mol_calc = torch.einsum(
##        "ijk,i->jk",
##        sp_calc.to(DEVICE),
##        data.multiplicity[data.node_mask]
##    ).detach().cpu().numpy()
##    sp_mol_pred = torch.einsum(
##        "ijk,i->jk",
##        sp_pred.to(DEVICE),
##        data.multiplicity[data.node_mask]
##    ).detach().cpu().numpy()
#
#    # convert to numpy
#    sp_calc = sp_calc.detach().cpu().numpy()
#    sp_pred = sp_pred.detach().cpu().numpy()
#
#    # plot
#    fig = plt.figure(figsize=(12, 4.5))
#    if with_left_margin:
#        # for with mol structure at the left
#        gs = GridSpec(1, 3, width_ratios=[0.5, 1, 1], figure=fig)
#        gs_index_start = 1
#    else:
#        gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
#        gs_index_start = 0
#    ax_0 = plt.subplot(gs[gs_index_start + 1])
#
#    # fig, axes = plt.subplots(n_site, figsize=(6, n_site))
#    energies = np.linspace(288, 310, 256)
#    gs_sub = GridSpecFromSubplotSpec(
#        n_site,
#        1,
#        subplot_spec=gs[gs_index_start],
#        wspace=0.4,
#        hspace=0.)
#    axes = [plt.subplot(s) for s in gs_sub]
#
#    color_dict = {
#        "x": "r",
#        "y": "g",
#        "z": "b"
#    }
#
#    # plot site spectrum
#    ymax = np.ceil(max(sp_calc.max(), sp_pred.max()))
#    for i_site, ax in enumerate(axes):
#        for i, (label, c) in enumerate(color_dict.items()):
#            ax.plot(energies, sp_calc[i_site, i], c="k",
#                    ls=["--", ":", "-."][i],
#                    alpha=0.5, label=f"{label} (calc)")
#            ax.plot(energies, sp_pred[i_site, i], c=c,
#                    alpha=0.5, label=f"{label} (pred)")
#        ax.tick_params(
#            labelbottom=False,
#            labelleft=True,
#            labelright=False,
#            labeltop=False)
#    #    ax.legend(title=f"mol_id: {data.name[0]}")
#        ax.set_ylim(0, ymax)
#        ax.set_xlim(288, 310)
#        ax.text(.975, .9, f"site {i_site + 1}",
#                transform=ax.transAxes, ha="right", va="top")
#    # set labels for the last ax
#    ax.set_xlabel("Energy (eV)")
#    ax.set_ylabel("Intensity (arb. unit)")
#    ax.yaxis.set_label_coords(-.075, n_site / 2)
#    ax.tick_params(
#        labelbottom=True,
#        labelleft=True,
#        labelright=False,
#        labeltop=False)
#
#    # set title for the first ax
#    axes[0].set_title("Site spectrum")
#
#    fig.suptitle(f"#{data.name[0].split('_')[-1]}")
#
#    # plot mol spectrum
#    ymax = np.ceil(max(sp_mol_calc.max().item(), sp_mol_pred.max().item()))
#    for i, (label, c) in enumerate(color_dict.items()):
#        ax_0.plot(
#            energies,
#            sp_mol_calc[i],
#            c="k",
#            ls=["--", ":", "-."][i],
#            alpha=0.5,
#            label=f"{label} (calc)")
#        ax_0.plot(
#            energies,
#            sp_mol_pred[i],
#            c=c,
#            ls="-",
#            alpha=0.5,
#            label=f"{label} (pred)")
#    ax_0.set_ylim(0, ymax)
#    ax_0.set_xlim(288, 310)
#    ax_0.legend()
#    ax_0.set_xlabel("Energy (eV)")
#    ax_0.set_ylabel("Intensity (arb. unit)")
#    ax_0.set_title("Molecular spectrum")
#
#    fig.tight_layout()
#    fig.savefig(Path(f"mol_{id_mol}.png"), dpi=300)
#
#    # make svg
#    base_svg = SVG(fig)
#
#    # plot molecule
#    mol = process_mol(qm9_data2mol(dataset, id_mol), coord_gen=True, with_note=True)
#    mol_svg = SVG(mol2svg(mol, size=(200, 200)))
#    apply_inset_scale_shift(ax_0, mol_svg, anchor="NW", scale=0.2, shift_x=125)
#
#    merged_svg = Figure(
#        base_svg.width,
#        base_svg.height,
#        base_svg,
#        mol_svg
#    )
#
#    merged_svg.save(Path(f"mol_{id_mol}_merged.svg"))
#