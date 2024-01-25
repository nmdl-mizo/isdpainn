# Description for model evaluation on C-K edge spectra dataset

## Dataset source and code for preprocessing dataset

The carbon K-edge spectra data were smoothed by a Gaussian filter at 0.5 eV for each site and include intensities in the x-, y-, and z-directions.
This data is published under the name "site_spectra_0.5eV.hdf5" as related data in the prior literature [1] at the following FigShare URL:
https://figshare.com/ndownloader/files/31947896

The structures of the molecules in the corresponding QM9 dataset [2, 3] were obtained using `torch_geometric.dataset.qm9.QM9` from the Python library [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) [4].

In the actual validation experiment, a vector of 256 lengths sampled at equal intervals from 288eV to 310eV was interpolated and used for training after integration with the structural data.
The class `ck_edge_maker.dataset.CK`, which inherits from torch_geometric's InmemoryDataset, is available in the Python library [ck-edge-maker](https://github.com/nmdl-mizo/ck_edge_maker) on GitHub[5].

## Model and training parameters

Model and training parameters are described in "scripts/config-defaults.yaml".

## Scripts for training

The code for training is "scripts/train.py".
To run the script, follow the procedure below:
1. Prepare Python environment with PyTorch Geometric and install [isdpainn](https://github.com/nmdl-mizo/ck_edge_maker) and [ck-edge-maker](https://github.com/nmdl-mizo/ck_edge_maker) installed. GPU environment is recommended.
1. Download the site-specific stectral dataset named "site_spectra_0.5eV.hdf5" from [FigShare](https://figshare.com/ndownloader/files/31947896) and place it under "./dataset/raw" directory.
1. Check the model and training parameters in "config-defaults.yaml" and modify it if you need.
1. Run `./train.py -l`.

## Trained model checkpoint

The weights of the trained model and MSE for random split and scaffold split discussed in the paper are available at https://doi.org/10.5281/zenodo.10547719.

The model weight can be loaded as follows:
```Python
from inspect import signature
import torch
from isdpainn import ISDPaiNN
from wandb.sdk.lib.config_util import dict_from_config_file
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = dict_from_config_file("config-defaults.yaml")
model = ISDPaiNN(**{
    key: value
    for key, value in config.items()
    if key in signature(ISDPaiNN).parameters.keys()
})
model_state_dict = torch.load("model_state.pth", map_location=DEVICE)
model.load_state_dict(
    model_state_dict["model_state_dict"]
)
```

## Time comparison between DFT and IDS-PaiNN

The directory "calculation_time_comaprison" contains data and script for the calculation/prediction time for the C-K edge in the dataset by CASTEP code and IDS-PaiNN.
"time_mol_castep_vs_isdpainn.pkl" contains the data of the calculation time, and "plot_calculation_time_comparison.py" is a script used for making a figure visualizing the dependency on molecular size.

## References
1. Shibata, K., Kikumasa, K., Kiyohara, S. et al. Simulated carbon K edge spectral database of organic molecules. Sci Data 9, 214 (2022). https://doi.org/10.1038/s41597-022-01303-8
2. Ruddigkeit, L., van Deursen, R., Blum, L. C. & Reymond, J.-L. Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17. Journal of Chemical Information and Modeling 52, 2864–2875, (2012). https://doi.org/10.1021/ci300415d
3. Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data 1, 140022, (2014). https://doi.org/10.1038/sdata.2014.22
4. Fey, Matthias and Lenssen, Jan E. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds (2019). https://github.com/pyg-team/pytorch_geometric/tree/master
5. K. Shibata, A dedicated Python script for making smeared C-K edge spectra from the hdf5 dataset of eigenvalues and dynamical structure factors (2024). https://github.com/nmdl-mizo/ck_edge_maker