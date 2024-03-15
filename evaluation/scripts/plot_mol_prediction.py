#!/usr/bin/env python
import torch
from analysis_utils import get_model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import numpy as np
    from torch_geometric.loader import DataLoader
    from ck_edge_maker.dataset import CK
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "id_mol",
        type=int,
        nargs='+',
        help="id of molecule, can be multiple ids"
    )
    parser.add_argument(
        "--with_left_margin",
        action="store_true",
        help="with left margin for molecule structure"
    )
    parser.add_argument("-d", "--dir", type=str, help='data directory', default="../data/model")
    args = parser.parse_args()
    data_dir = args.dir
    split_type = "random_split"


    # figure dir
    figure_dir = Path("../figures")

    # load model
    config_path = Path(data_dir, f"config-defaults_{split_type}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"config file {config_path} does not exist")
    model_state_path = Path(data_dir, f"model_state_{split_type}.pt")
    if not model_state_path.exists():
        raise FileNotFoundError(f"model state file {model_state_path} does not exist")
    metric_dict_path = Path(data_dir, f"metric_dict_{split_type}.pt")
    if not metric_dict_path.exists():
        raise FileNotFoundError(f"metric dict file {metric_dict_path} does not exist")

    # prepare model

    model = get_model(config_path=config_path, model_state_path=model_state_path)
    model.to(DEVICE)

    # load dataset
    dataset = CK(root="./dataset", energies=(288, 310, 256), directional=True)

    for id_mol in args.id_mol:
        index_mol = (dataset.id_mol == id_mol).nonzero().item()
        # prepare mol data
        data = next(iter(DataLoader(
            dataset[index_mol:index_mol + 1],
            batch_size=1,
            shuffle=False
        ))).to(DEVICE)
        directions = torch.eye(3).to(DEVICE)
        n_site = data.node_mask.sum().item()

        # prepare spectrum
        sp_calc = data.spectra[data.node_mask]

        # predict spectrum
        with torch.no_grad():
            sp_pred = torch.stack([
                model.forward(data, direction=directions[i])
                for i in range(3)
            ]).swapaxes(0, 1)[data.node_mask]

        # molecular spectrum
        sp_mol_calc = torch.einsum(
            "ijk,i->jk",
            sp_calc.to(DEVICE),
            data.multiplicity[data.node_mask]
        ).detach().cpu().numpy()
        sp_mol_pred = torch.einsum(
            "ijk,i->jk",
            sp_pred.to(DEVICE),
            data.multiplicity[data.node_mask]
        ).detach().cpu().numpy()

        # convert to numpy
        sp_calc = sp_calc.detach().cpu().numpy()
        sp_pred = sp_pred.detach().cpu().numpy()

        # plot
        fig = plt.figure(figsize=(12, 4.5))
        if args.with_left_margin:
            # for with mol structure at the left
            gs = GridSpec(1, 3, width_ratios=[0.5, 1, 1], figure=fig)
            gs_index_start = 1
        else:
            gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
            gs_index_start = 0
        ax_0 = plt.subplot(gs[gs_index_start + 1])

        # fig, axes = plt.subplots(n_site, figsize=(6, n_site))
        energies = np.linspace(288, 310, 256)
        gs_sub = GridSpecFromSubplotSpec(
            n_site,
            1,
            subplot_spec=gs[gs_index_start],
            wspace=0.4,
            hspace=0.)
        axes = [plt.subplot(s) for s in gs_sub]

        color_dict = {
            "x": "r",
            "y": "g",
            "z": "b"
        }

        # plot site spectrum
        ymax = np.ceil(max(sp_calc.max(), sp_pred.max()))
        for i_site, ax in enumerate(axes):
            for i, (label, c) in enumerate(color_dict.items()):
                ax.plot(energies, sp_calc[i_site, i], c="k",
                        ls=["--", ":", "-."][i],
                        alpha=0.5, label=f"{label} (calc)")
                ax.plot(energies, sp_pred[i_site, i], c=c,
                        alpha=0.5, label=f"{label} (pred)")
            ax.tick_params(
                labelbottom=False,
                labelleft=True,
                labelright=False,
                labeltop=False)
        #    ax.legend(title=f"mol_id: {data.name[0]}")
            ax.set_ylim(0, ymax)
            ax.set_xlim(288, 310)
            ax.text(.975, .9, f"site {i_site + 1}",
                    transform=ax.transAxes, ha="right", va="top")
        # set labels for the last ax
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity (arb. unit)")
        ax.yaxis.set_label_coords(-.075, n_site / 2)
        ax.tick_params(
            labelbottom=True,
            labelleft=True,
            labelright=False,
            labeltop=False)

        # set title for the first ax
        axes[0].set_title("Site spectrum")

        fig.suptitle(f"#{data.name[0].split('_')[-1]}")

        # plot mol spectrum
        ymax = np.ceil(max(sp_mol_calc.max(), sp_mol_pred.max()))
        for i, (label, c) in enumerate(color_dict.items()):
            ax_0.plot(
                energies,
                sp_mol_calc[i],
                c="k",
                ls=["--", ":", "-."][i],
                alpha=0.5,
                label=f"{label} (calc)")
            ax_0.plot(
                energies,
                sp_mol_pred[i],
                c=c,
                ls="-",
                alpha=0.5,
                label=f"{label} (pred)")
        ax_0.set_ylim(0, ymax)
        ax_0.set_xlim(288, 310)
        ax_0.legend()
        ax_0.set_xlabel("Energy (eV)")
        ax_0.set_ylabel("Intensity (arb. unit)")
        ax_0.set_title("Molecular spectrum")

        fig.tight_layout()
        fig.savefig(Path(figure_dir, f"mol_{id_mol}.png"), dpi=300)
