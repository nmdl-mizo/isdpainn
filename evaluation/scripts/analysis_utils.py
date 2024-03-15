from inspect import signature
from typing import Union
from pathlib import Path
from tqdm import tqdm
import torch
from wandb.sdk.lib.config_util import dict_from_config_file
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader
from isdpainn import ISDPaiNN
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(config_path: Union[str, Path], model_state_path: Union[str, Path]) -> ISDPaiNN:
    config = dict_from_config_file(str(config_path))
    if config is None:
        raise ValueError(f"config file not found: {config_path}")
    model = ISDPaiNN(**{
        key: value
        for key, value in config.items()
        if key in signature(ISDPaiNN).parameters.keys()
    })
    model_state_dict = torch.load(model_state_path, map_location=DEVICE)
    model.load_state_dict(
        model_state_dict["model_state_dict"]
    )
    return model


def make_rot_mat(rotvec):
    # rotvec = np.array([0., 0., np.pi]) # rotate about z axis by pi radian
    rot_mat = Rotation.from_rotvec(rotvec)
    return rot_mat


def apply_rot(data, rotvec):
    data_rot = data.clone()
    data_rot.apply(
        lambda p: torch.tensor(
            make_rot_mat(rotvec).apply(p.cpu().numpy()),
            dtype=data.pos.dtype,
            device=data.pos.device
        ),
        "pos"
    )
    return data_rot


def apply_si(data):
    data_si = data.clone()
    data_si.apply(lambda p: -p, "pos")
    return data_si


def apply_rot_both(rotvec, data, direction):
    rot_mat = make_rot_mat(rotvec)
    data_rot = data.clone()
    data_rot.apply(
        lambda p: torch.tensor(
            rot_mat.apply(p.cpu().numpy()),
            dtype=data.pos.dtype,
            device=data.pos.device
        ),
        "pos"
    )
    return data_rot, rot_mat.apply(direction)


# for mse only, will be deprecated
def get_mse_batch(model, data):
    data.to(next(model.parameters()).device)
    mse_batch = torch.vstack([
        (
            data.spectra[data.node_mask, i, :]
            - model.forward(data, r)[data.node_mask]
        ).cpu().detach().square().mean(-1)
        for i, r in enumerate(torch.eye(3))
    ])
    return mse_batch


def calc_dataset_mse(model, dataset_dict, batch_size):
    mse_dict = {}
    for label, dataset in dataset_dict.items():
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mse = torch.hstack([get_mse_batch(model, data) for data in tqdm(dl, desc=label)])
        id_mol_site = torch.tensor([
            [id_mol.item(), id_site]
            for data in dl
            for id_mol, id_site_list in zip(data.id_mol, data.id_site)
            for id_site in id_site_list
        ])
        mse_dict[label] = {'mse': mse, 'id_mol_site': id_mol_site}
    return mse_dict


def get_typical_id_mol_site_from_mse(mse, id_mol_site, n, indices_filter=None):
    values, indices = torch.sort(mse.reshape(-1))
    if indices_filter is None:
        indices_filter = torch.linspace(0, mse.shape[1] * 3 - 1, n, dtype=int)
    indices_filtered = indices.reshape(-1)[indices_filter]
    values_filtered = values.reshape(-1)[indices_filter]
    indices_filtered_mol_site, indices_filtered_r, = (
        indices_filtered % mse.shape[1],
        indices_filtered // mse.shape[1],
    )
    id_mol_filtered, id_site_filtered = id_mol_site[indices_filtered_mol_site].T
    return id_mol_filtered, id_site_filtered, indices_filtered_r, values_filtered


# for general metric
# general metric version of get_mse_batch
def get_metric_batch(model, data, metric_func):
    data.to(next(model.parameters()).device)
    metric_batch = torch.vstack([
        metric_func(
            data.spectra[data.node_mask, i, :],
            model.forward(data, r)[data.node_mask]
        ).cpu().detach()
        for i, r in enumerate(torch.eye(3))
    ])
    return metric_batch


# general metric version of calc_dataset_mse
def calc_dataset_metric(model, dataset_dict, batch_size, metric_func, metric_name):
    metric_dict = {}
    for label, dataset in dataset_dict.items():
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        metric_data = torch.hstack([
            get_metric_batch(model, data, metric_func)
            for data in tqdm(dl, desc=label)
        ])
        id_mol_site = torch.tensor([
            [id_mol.item(), id_site]
            for data in dl
            for id_mol, id_site_list in zip(data.id_mol, data.id_site)
            for id_site in id_site_list
        ])
        metric_dict[label] = {metric_name: metric_data, 'id_mol_site': id_mol_site}
    return metric_dict


# general metric version of get_typical_id_mol_site_from_mse
def get_typical_id_mol_site_from_metric(metric_data, id_mol_site, n, indices_filter=None):
    values, indices = torch.sort(metric_data.reshape(-1))
    if indices_filter is None:
        indices_filter = torch.linspace(0, metric_data.shape[1] * 3 - 1, n, dtype=int)
    indices_filtered = indices.reshape(-1)[indices_filter]
    values_filtered = values.reshape(-1)[indices_filter]
    indices_filtered_mol_site, indices_filtered_r, = (
        indices_filtered % metric_data.shape[1],
        indices_filtered // metric_data.shape[1],
    )
    id_mol_filtered, id_site_filtered = id_mol_site[indices_filtered_mol_site].T
    return id_mol_filtered, id_site_filtered, indices_filtered_r, values_filtered
