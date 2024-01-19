from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.loader import DataLoader
from ck_edge_maker.dataset import CK

class EarlyStopping:
    def __init__(self, model, optimizer, patience=5,
                 verbose=True, path='checkpoint_model.pth', loss_fmt='.3e'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = None
        self.path = path
        self.model = model
        self.optimizer = optimizer
        self.loss_fmt = loss_fmt

    def __call__(self, val_loss, optional_dict=None, nan_raise_error=False):
        if np.isnan(val_loss):
            if nan_raise_error:
                raise RuntimeError("val_loss = nan")
            print(f'val_loss = nan. Abort training.')
            self.early_stop = True
            return self.early_stop
        if self.val_loss_min is None:
            if self.verbose:
                print(f'Initial epoch finished ({val_loss:{self.loss_fmt}}).')
            self.val_loss_min = val_loss
            self.checkpoint(optional_dict=optional_dict)
        elif val_loss >= self.val_loss_min:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Reached patiance epoch ({self.patience}).')
                self.early_stop = True
        else:
            if self.verbose:
                print(
                    'Validation loss decreased '
                    f'({self.val_loss_min:{self.loss_fmt}} '
                    f'--> {val_loss:{self.loss_fmt}}).')
            self.val_loss_min = val_loss
            self.counter = 0
            self.checkpoint(optional_dict=optional_dict)
        return self.early_stop

    def checkpoint(self, optional_dict=None):
        chkpt_dict = {
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer is not None:
            chkpt_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if optional_dict is None:
            optional_dict = {}
        torch.save({**chkpt_dict, **optional_dict}, self.path)


def prepare_dataset(
        config,
        root="dataset",
        energies_default=(
            288,
            310,
            256
        ),
        as_dict=False,
        random_state_default=0,
        directional_default=True):
    shuffle = config.get("shuffle", True)
    random_state = config.get("random_state", random_state_default)
    # prepare Dataset split
    ck = CK(
        root=root,
        energies=config.get("energies", energies_default),
        directional=config.get("directional", directional_default)
    )
    if config.get("scaffold_split", False):
        print(
            "scaffold_split == True is specified."
            "Use deepchem.splits.ScaffoldSplitter."
        )
        smiles_ck = torch.load("smiles_ck.pt")
        import deepchem as dc
        splitter = dc.splits.ScaffoldSplitter()
        dataset = dc.data.DiskDataset.from_numpy(
            X=torch.arange(len(ck)),
            y=smiles_ck["id"],
            ids=smiles_ck["smiles"]
        )
        return {
            label: ck[data.X.tolist()]
            for label, data in zip(
                ["train", "val", "test"],
                splitter.train_valid_test_split(
                    dataset,
                    frac_train=1. - 2. * config.get("test_size", 0.1),
                    frac_valid=config.get("test_size", 0.1),
                    frac_test=config.get("test_size", 0.1),
                    seed=random_state
                )
            )
        }
    train_val_index, test_index = train_test_split(
        range(len(ck)), test_size=config["test_size"],
        shuffle=shuffle, random_state=random_state)
    train_index, val_index = train_test_split(
        range(len(train_val_index)),
        test_size=config["val_size"], shuffle=True, random_state=0)
    dataset_train = ck[np.array(train_val_index)[train_index]]
    dataset_val = ck[np.array(train_val_index)[val_index]]
    dataset_test = ck[test_index]
    if as_dict:
        return {
            "train": dataset_train,
            "val": dataset_val,
            "test": dataset_test}
    else:
        return dataset_train, dataset_val, dataset_test


# loss functions
def mean_squared_error(y, y_pred):
    return (y - y_pred).square().mean(-1)


def mean_absolute_error(y, y_pred):
    return (y - y_pred).abs().mean(-1)


def wasserstein_distance(y_true, y_pred, p=2, normalize=False, reduce="mean"):
    """Wasserstein distance

    Wasserstein distance should be implemented as optional metric for loss function
    defined as disscussed in [doi:10.26434/chemrxiv-2023-j1szt]
    W_p(alpha, beta) = (int_R |C_alpha(x) - C_beta(x)|^p dx)^{1/p},
    where C_alpha is the cumulative distribution function (CDF) of alpha(x)

    Args:
        y_true (torch.Tensor): (n_sample, n_dim)
        y_pred (torch.Tensor): (n_sample, n_dim)
        p (int, optional): index p. Defaults to 2.
        normalize (bool, optional): normalize inputs into probability distribution.
            Defaults to False.

    Returns:
        torch.Tensor: (n_sample,)
    """
    if normalize:
        y_true /= y_true.sum(dim=-1, keepdim=True)
        y_pred /= y_pred.sum(dim=-1, keepdim=True)
    y_true = torch.cumsum(y_true, dim=-1)
    y_pred = torch.cumsum(y_pred, dim=-1)
    if reduce == "mean":
        return (y_true - y_pred).abs().pow(p).sum(-1).pow(1. / p)
    elif reduce == "sum":
        return (y_true - y_pred).abs().pow(p).mean(-1).pow(1. / p)


loss_func_dict = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "wd": wasserstein_distance
}


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

