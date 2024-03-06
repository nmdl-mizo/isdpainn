import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from analysis_utils import get_typical_id_mol_site_from_metric


def plot_mse_hist(metric_dict, _range=(0., 0.2), bins=100, alpha=0.5, ax=None, metric_name="mse"):
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()
    for label, d in metric_dict.items():
        ax.hist(d[metric_name].reshape(-1).numpy(), bins=bins, range=_range, alpha=alpha, label=label)
    ax.legend()
    ax.set_xlabel("MSE")
    ax.set_ylabel("Frequency")
    return fig, ax


def plot_sorted_mse(metric_dict, alpha=0.5, ax=None, metric_name="mse", **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig = ax.get_figure()
    for label, d in metric_dict.items():
        metric_sorted = d[metric_name].reshape(-1).sort()[0].numpy()
        ax.scatter(torch.arange(len(metric_sorted)), metric_sorted, alpha=alpha, label=label, **kwargs)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("sort index")
    ax.set_ylabel("MSE")
    return fig, ax


def plot_typical_spectra(model, metric_dict, dataset_dict, label, n=12, indices_filter=None, ylim=(0., 20.), axes=None, col=4, row=3, figsize=(16, 9), mol_svg=False, legend=False, suptitle=False):
    device = next(model.parameters()).device
    id_mol_list, id_site_list, id_r_list, values_list = get_typical_id_mol_site_from_metric(
        *metric_dict[label].values(),
        n=n,
        indices_filter=indices_filter
    )
    id_mol_list = id_mol_list.to(device)
    id_site_list = id_site_list.to(device)
    id_r_list = id_r_list.to(device)
    values_list = values_list.to(device)
    dataset_filtered = dataset_dict[label][[
        (dataset_dict[label].id_mol.to(device) == id_mol).nonzero().item()
        for id_mol in id_mol_list
    ]]
    dl_filtered = DataLoader(dataset_filtered, batch_size=len(dataset_filtered))
    data_filtered = next(iter(dl_filtered)).to(device)
    with torch.no_grad():
        y_pred = torch.stack([model.forward(data_filtered, direction=r) for r in torch.eye(3).to(device)])
    y_pred_filtered = y_pred[id_r_list, data_filtered.ptr[:-1] + id_site_list]
    y_filtered = data_filtered.spectra[data_filtered.ptr[:-1] + id_site_list, id_r_list, ]

    if axes is None:
        fig, axes = plt.subplots(row, col, figsize=figsize)
    else:
        fig = axes.reshape(-1)[0].get_figure()
    energies = dataset_filtered.energies
    for i, (ax, y, y_pred) in enumerate(
        zip(
            axes.reshape(-1),
            y_filtered.cpu().detach().numpy(),
            y_pred_filtered.cpu().detach().numpy()
        )
    ):
        ax.plot(energies, y, c="gray", label="target")
        ax.plot(energies, y_pred, c="tab:red", label="pred")
        #ax.set_title(f'#{id_mol_list[i]}-{id_site_list[i]}-{"xyz"[id_r_list[i]]}, {values_list[i]:.1e}')
        ax.set_xlabel("Energy loss (eV)")
        ax.set_ylabel("Intensity (arb. unit)")
        ax.set_xlim(*energies[[0, -1]])
        ax.set_ylim(ylim)
        id_C_site = data_filtered.node_mask[data_filtered.batch == i].cumsum(0)[id_site_list[i]]
        if legend:
#            ax.legend(title=f"#{id_mol_list[i]}-{id_site_list[i] + 1}-${'xyz'[id_r_list[i]]}$\nMSE={values_list[i]:.1e}")
            ax.legend(title=f"#{id_mol_list[i]}-{id_C_site}-${'xyz'[id_r_list[i]]}$\nMSE={values_list[i]:.1e}")
    if suptitle:
        fig.suptitle(f"{label}")
    fig.tight_layout()
    if mol_svg:
        from draw_mol import qm9_data2mol, process_mol, mol2svg, apply_inset_scale_shift
        from svgutils.compose import SVG, Figure
        mol_svgs = [
            mol2svg(
                process_mol(mol, coord_gen=True),
                highlightAtoms=[id_site],
                size=(300, 300)
            )
            for mol, id_site in zip(
                qm9_data2mol(
                    dataset_dict[label],
                    id_mol_list.tolist()
                ),
                id_site_list.tolist()
            )
        ]
        inset_svg_list = [
            apply_inset_scale_shift(
                ax,
                SVG(mol_svg),
                scale=0.5,
                anchor="NW"
            )
            for ax, mol_svg in zip(axes.reshape(-1), mol_svgs)
        ]
        base_svg = SVG(fig)
        merged_svg = Figure(
            base_svg.width,
            base_svg.height,
            base_svg,
            *inset_svg_list
        )
        return merged_svg
    else:
        return fig, axes
