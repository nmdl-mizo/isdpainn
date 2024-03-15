#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# figure dir
figure_dir = Path("../figures")

# plot MSEs of ablation experiments
fig, ax = plt.subplots()
data_dir = "../data/model"
metric_dict = {
    "ISD-PaiNN": Path(data_dir, "metric_dict_random_split.pt"),
    "w/o DE": Path(data_dir, "metric_dict_wo_DE.pt"),
    "w/o SM": Path(data_dir, "metric_dict_wo_SM.pt"),
    "w/o DE-SM": Path(data_dir, "metric_dict_wo_DE-SM.pt")
}
for label, metric_path in metric_dict.items():
    metric_dict = torch.load(metric_path)
    value, index = torch.sort(metric_dict["test"]["mse"].reshape(-1))
    ax.scatter(
        torch.arange(len(value)),
        value.detach().cpu().numpy(),
        label=label,
        marker="o",
        s=5,
        alpha=0.5
    )
ax.legend(title=r"$S_n(\mathcal{G}, \hat{\mathbf{n}})$")
ax.set_xlabel("sort index")
ax.set_ylabel("MSE")
ax.set_yscale("log")
fig.savefig(Path(figure_dir, "ablation.png"), dpi=300)
