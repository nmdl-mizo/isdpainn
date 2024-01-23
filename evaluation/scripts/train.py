#!/usr/bin/env python
import os
from inspect import signature
import torch
import yaml
from torch_geometric.loader import DataLoader
from isdpainn import ISDPaiNN
from train_utils import EarlyStopping, prepare_dataset, loss_func_dict, calc_dataset_metric
import wandb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")
PROJECT_NAME = "isdpainn-ck"

def train(config_defaults=None,
          use_wandb=True,
          project=PROJECT_NAME,
          group=None):
    if use_wandb:
        wandb.init(config=config_defaults,
                   project=project,
                   group=group)
        config = wandb.config
        assert wandb.run is not None
        save_dir = wandb.run.dir
        assert isinstance(save_dir, str)
    else:
        wandb.init(mode="disabled")
        if config_defaults is None:
            from wandb.sdk.lib.config_util import dict_from_config_file
            config = dict_from_config_file("config-defaults.yaml")
        elif isinstance(config_defaults, dict):
            config = config_defaults
        else:
            raise TypeError("config_defaults must be dict or None")
        save_dir = "."
    assert config is not None
    # prepare dataset split
    dataset_dict = prepare_dataset(config, as_dict=True)
    assert isinstance(dataset_dict, dict)

    # parepare model
    torch.manual_seed(config.get("seed", 0))
    model = ISDPaiNN(**{
        key: value
        for key, value in config.items()
        if key in signature(ISDPaiNN).parameters.keys()
    })
    model.to(DEVICE)

    # watch model by wandb
    wandb.watch(model)

    # prepare training
    if config.get("optimizer", "Adam") == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    dl_train = DataLoader(
        dataset_dict["train"],
        batch_size=config["batch_size"],
        shuffle=True, drop_last=False
    )
    dl_val = DataLoader(
        dataset_dict["val"],
        batch_size=config["batch_size"],
        shuffle=True, drop_last=False
    )
    es = EarlyStopping(
        model=model,
        optimizer=optimizer,
        patience=config["patience"],
        verbose=config["verbose"],
        path=os.path.join(save_dir, config["model_state_path"])
    )
    loss_func = loss_func_dict[config.get("loss_func", "mse")]
    loss_dict = {"loss": [], "valloss": []}

    n_valid_spectra_train = dataset_dict["train"].node_mask.sum()
    n_valid_spectra_val = dataset_dict["val"].node_mask.sum()

    # learning epoch loop
    epoch_loss = 0.
    epoch_valloss = 0.
    for epoch in range(config["epochs"]):
        # train
        model.train()
        epoch_loss = 0.
        for data_train in dl_train:
            data_train.to(DEVICE)
            for i_direction, direction in enumerate(torch.eye(3)):
                optimizer.zero_grad()
                y_pred = model.forward(data_train, direction)
                assert y_pred is not None
                y_pred = y_pred[data_train.node_mask]
                y = data_train.spectra[data_train.node_mask, i_direction]
                loss = loss_func(y, y_pred).sum()
                epoch_loss += loss.item()
                loss /= data_train.node_mask.sum()
                loss.backward()
                optimizer.step()
        epoch_loss /= (n_valid_spectra_train * 3)
        loss_dict["loss"].append(epoch_loss)

        # validate
        model.eval()
        epoch_valloss = 0.
        with torch.no_grad():
            for data_val in dl_val:
                data_val.to(DEVICE)
                for i_direction, direction in enumerate(torch.eye(3)):
                    y_pred = model.forward(data_val, direction)
                    assert y_pred is not None
                    y_pred = y_pred[data_val.node_mask]
                    y = data_val.spectra[data_val.node_mask, i_direction]
                    loss = loss_func(y, y_pred).sum()
                    epoch_valloss += loss.item()
        epoch_valloss /= (n_valid_spectra_val * 3)
        loss_dict["valloss"].append(epoch_valloss)

        # log by wandb
        wandb.log(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "valloss": epoch_valloss,
                "valloss_min": es.val_loss_min
            }
        )

        # evaluate earlystopping and save checkpoint
        optional_dict = {
            "epoch": epoch,
        #    "config": config,
            "loss": loss_dict
        }

        # Early stopping
        if es(val_loss=epoch_valloss, optional_dict=optional_dict):
            print(
                f'loss:{epoch_loss:.3e}, val:{epoch_valloss:.3e}'
            )
            break

    # load the best model
    model_state_dict = torch.load(
        es.path,
        map_location=torch.device(DEVICE)
    )["model_state_dict"]
    # deal with the layer name change
    model.load_state_dict(model_state_dict)

    # calc metric
    metric_dict_filename = os.path.join(save_dir, "metric_dict.pt")
    metric_dict = calc_dataset_metric(
        model, dataset_dict, batch_size=config["batch_size"],
        metric_func=loss_func,
        metric_name=config.get("loss_func", "mse")
    )
    torch.save(metric_dict, metric_dict_filename)

    # send summary
    assert wandb.run is not None
    wandb.run.summary["val_loss_min_final"] = es.val_loss_min
    wandb.run.summary["epoch_loss"] = epoch_loss
    wandb.run.summary["epoch_valloss"] = epoch_valloss
    if isinstance(config.get("train_val_n_list", None), list):
        wandb.run.summary["train_val_n_min"] = min(config["train_val_n_list"])
        wandb.run.summary["train_val_n_max"] = max(config["train_val_n_list"])
    if isinstance(config.get("test_n_list", None), list):
        wandb.run.summary["test_n_min"] = min(config["test_n_list"])
        wandb.run.summary["test_n_max"] = max(config["test_n_list"])
    wandb.finish()
    return loss_dict


def sweep(sweep_id=None, config_yaml='config_sweep.yaml', project=PROJECT_NAME):
    if sweep_id is None:
        with open(config_yaml, 'r') as yml:
            sweep_config = yaml.safe_load(yml)
        sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, train)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local', action='store_true')
    parser.add_argument('-s', '--sweep', action='store_true')
    parser.add_argument('--project', default=PROJECT_NAME)
    parser.add_argument('--sweep_id')
    args = parser.parse_args()

    if args.sweep:
        sweep(sweep_id=args.sweep_id, project=args.project)
    else:
        train(use_wandb=not args.local, project=args.project)
