#!/usr/bin/env python3
from matplotlib import figure
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from  wandb.sdk.lib.config_util import dict_from_config_file
# my modules
from train_utils import prepare_dataset
from analysis_utils import get_model
from plot_utils import plot_sorted_mse, plot_typical_spectra

# set device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")

def plot(split_type, config_path, model_state_path, metric_dict_path, percentile_positions, figure_dir):
    """
    Plot typical spectra for a given split type
    """

    # load config
    config = dict_from_config_file(config_path)

    # get model
    model = get_model(config_path=config_path, model_state_path=model_state_path)

    # prepare dataset split
    dataset_dict = prepare_dataset(config, as_dict=True)

    # print dataset info
    # number of data for train, val, test
    print(f"Number of data for train: {dataset_dict['train'].node_mask.sum().item()}")
    print(f"Number of data for val: {dataset_dict['val'].node_mask.sum().item()}")
    print(f"Number of data for test: {dataset_dict['test'].node_mask.sum().item()}")
    # number of spectra for train, val, test
    print(f"Number of spectra for train: {dataset_dict['train'].node_mask.sum().item() * 3}")
    print(f"Number of spectra for val: {dataset_dict['val'].node_mask.sum().item() * 3}")
    print(f"Number of spectra for test: {dataset_dict['test'].node_mask.sum().item() * 3}")

    # load metric dict
    metric_dict = torch.load(metric_dict_path)

    # plot typical spectra
    fig = plt.figure(figsize=(12, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 2], figure=fig)
    ax_0 = plt.subplot(gs[0])
    n_data = dataset_dict["test"].node_mask.sum().item() * 3

    index_filter = [
        int((n_data - 1)* i)
        for i in percentile_positions
    ]

    #ax_0_sec = ax_0.secondary_xaxis('top', functions=(lambda x: x/(n_data - 1), lambda x: x * (n_data - 1)))
    #ax_0.axvline(n_data * 0.95, color="k", alpha=0.5, linestyle="--")
    #ax_0.axhline(0.1, color="k", lw=1., alpha=0.5, linestyle="--")

    gs_sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], wspace=0.4, hspace=0.4)
    axes = np.array([
        plt.subplot(gs_sub[i, j])
        for i in range(2)
        for j in range(2)
    ]).reshape(2, 2)

    # plot data
    metric_name = config.get("loss_func", "mse")
    plot_sorted_mse(
        {"$S_n(\mathcal{G}, \hat{\mathbf{n}})$": metric_dict["test"]},
        ax=ax_0, alpha=0.25, marker=".", c="r", metric_name=metric_name
    )
    offsets = ax_0.collections[0].get_offsets()
    sorted_index, sorted_metric = offsets[:, 0], offsets[:, 1]
    #typical_indices  = np.linspace(0, len(offsets) - 1, 4, dtype=int)
    typical_indices = index_filter
    for typical_index in typical_indices:
        ax_0.annotate(
            f"{typical_index/(len(offsets) - 1):.0%}",
            xy=(sorted_index[typical_index], sorted_metric[typical_index]),
            xytext=(sorted_index[typical_index] + 8000, sorted_metric[typical_index]/1.4),
            ha="center",
            va="center",
            #fontsize=14,
            arrowprops=dict(arrowstyle="simple", color="k"),
        )
    ax_0.set_xticks(np.arange(0, 80000, 20000))

    plot_typical_spectra(
        model, metric_dict, dataset_dict, "test",
        n=4, mol_svg=False, col=2, row=2, axes=axes, indices_filter=typical_indices, legend=True
    )

    # add label (a, b, c, d, e) at the top left of each axis
    ax_0.text(-0.1, 1., "a", transform=ax_0.transAxes, fontsize=16)
    for ax, label in zip(axes.flatten(), "bcde"):
        ax.text(-0.2, 1., label, transform=ax.transAxes, fontsize=16)
        #ax.set_title("")
    fig.set_facecolor("white")
    fig.savefig(Path(figure_dir, f"grid_typical_spectra_{split_type}.png"), dpi=300)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("split_type", type=str, help='split type, "random_split" or "scaffold_split')
    parser.add_argument("-d", "--dir", type=str, help='data directory', default="../data/model")
    args = parser.parse_args()
    split_type = args.split_type
    data_dir = Path(args.dir)
    config_path = Path(data_dir, f"config-defaults_{split_type}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"config file {config_path} does not exist")
    model_state_path = Path(data_dir, f"model_state_{split_type}.pt")
    if not model_state_path.exists():
        raise FileNotFoundError(f"model state file {model_state_path} does not exist")
    metric_dict_path = Path(data_dir, f"metric_dict_{split_type}.pt")
    if not metric_dict_path.exists():
        raise FileNotFoundError(f"metric dict file {metric_dict_path} does not exist")
    percentile_positions = {
        "random_split": [0., 0.5, 0.95, 1.0],
        "scaffold_split": [0., 1./2., 3./4., 1.0]
    }[split_type]
    plot(split_type, config_path, model_state_path, metric_dict_path, percentile_positions, figure_dir=Path("../figures"))
