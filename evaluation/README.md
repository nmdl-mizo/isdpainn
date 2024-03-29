# Description for model evaluation on C-K edge spectra dataset

This repository contains scripts for model evaluation on C-K edge spectra dataset including training and analysis.

## Environment for training and evaluation

All scripts should be run in an appropriate Python environment where CUDA 11.8 is available on a GPU.

For running scripts for evaluation, additional packages are required in addition to packages required by isdpainn.
To create the Python environment for evaluation we recommend to create a dedicated environment using conda.

You can use the package list files provided in `evaluation/conda_env` for easy construction: 
```bash
git clone git@github.com:nmdl-mizo/isdpainn.git
cd isdpainn/evaluation/conda_env
conda env create -f environment-eval.yaml
conda activate isdpainn-eval
pip install -r requirements-eval.txt
pip install deepchem # for scaffold split
```

Alternatively, run the following:

```bash
conda create -n isdpainn-eval python=3.10 pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 pyg=2.4.0 pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv matplotlib tqdm scikit-learn seaborn -c pytorch -c nvidia -c pyg
conda activate isdpainn-eval
pip install wandb pymatgen
pip install git+https://github.com/Open-Catalyst-Project/ocp.git@main#egg=ocp-models
pip install git+https://github.com/nmdl-mizo/isdpainn.git@main#egg=isdpainn
pip install "ck_edge_maker[pyg] @ git+https://github.com/nmdl-mizo/ck_edge_maker@main"
pip install git+https://github.com/nmdl-mizo/castep_elnes_parser@main#egg=ceparser
pip install deepchem # for scaffold split
```

## Dataset download

To run the scripts for evaluation, some data must be downloaded.

The files uploaded to zenodo and NOMAD can be downloaded under `evaluation/data` directory by running `evaluation/scripts/download.py`.
The script downloads following:
  - The trained weights, settings, and mean square error(MSE) for each of the random and scaffold splits from [zenodo](https://doi.org/10.5281/zenodo.10547719)
  - The trained weights, settings, and MSE for the ablation experiments from [zenodo](https://zenodo.org/doi/10.5281/zenodo.10566200).
  - Some of the raw CASTEP results files for analyzing the aromatic amino acids (Dataset id: [-wYS-_xcTce_8ufAVvJPZA](https://doi.org/10.17172/NOMAD/2024.01.23-1)) and the rotated benzene (Dataset id: [PFHr0r4-SDy-2otuTyk1Pw](https://doi.org/10.17172/NOMAD/2024.01.23-2)) from NOMAD.

Seperately, please download the site-specific C-K edge spectra dataset named "site_spectra_0.5eV.hdf5" from [FigShare](https://figshare.com/ndownloader/files/31947896) and place it under `evaluation/scripts/dataset/raw` directory.
This file contains the carbon K-edge spectra data smoothed by a Gaussian filter at 0.5 eV for each site including intensities corresponding to the x-, y-, and z-directions.
This data is published as related data in the prior literature [1].

### Code for preprocessing dataset

The structures of the molecules in the corresponding QM9 dataset [2, 3] were obtained using `torch_geometric.dataset.qm9.QM9` from the Python library [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) [4].

In the actual validation experiment, a vector of 256 lengths sampled at equal intervals from 288eV to 310eV was interpolated and used for training after integration with the structural data.
The class `ck_edge_maker.dataset.CK`, which inherits from torch_geometric's InmemoryDataset, is available in the Python library [ck-edge-maker](https://github.com/nmdl-mizo/ck_edge_maker) on GitHub[5].

## Run training

The default model and training parameters are stored in "evaluation/scripts/config-defaults.yaml".

The code for training is "scripts/train.py".
To run the script, follow the procedure below:
1. Prepare Python environment following [here](#environment-for-training-and-evaluation).
1. Change directory to `evaluation/scripts/`
1. Check the model and training parameters in `config-defaults.yaml` and modify it if you need.
1. Run `./train.py -l`.
1. Files will be output.
    - checkpoint file (`model_state.pt`)
    - MSE of site- (`metric_dict.pt`)

### Load trained model checkpoints

The trained model weight can be loaded as follows:
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
model_state_dict = torch.load("model_state.pt", map_location=DEVICE)
model.load_state_dict(
    model_state_dict["model_state_dict"]
)
```

## Analysis for model evaluation

The Python script files in `evaluation/scripts` can be used for the reproducing the analysis, and plots can be generated.

Before running the scripts, please prepare and activate a Python environment (see [here](#environment-for-training-and-evaluation)) and downloaded the dataset (see [here](#dataset-download)).

### Prediction errors for random split and scaffold split

The distribution of MSEs in the random and Scaffold splits was investigated, and spectra in typical positions were examined.

#### Random split
For MSEs of site spectra by the random split, run the following command and `evaluation/figures/grid_typical_spectra_random_split.png` will be generated.
```sh
./plot_typical_spectra_grid.py random_split
``` 

#### Scaffold split

For MSEs of site spectra by the scaffold split, run the following command and `evaluation/figures/grid_typical_spectra_scaffold_split.png` will be generated.
```sh
./plot_typical_spectra_grid.py scaffold_split
``` 

#### Molecular MSEs for random split

As a prediction accuracy per molecule rather than site, the sum of the MSEs of the site spectra for the valid carbon sites was taken as the MSE of the molecular spectra, and the distribution was examined for the model trained with the random split.
The `evaluation/figures/sorted_mol_mse.png` file can be generated for this result by executing the following command.
```sh
./plot_mol_mse.py
```

#### Molecular and site spectra of specific molecules

To plot the site and molecular spectra for a particular molecule for a model trained on random splits, run the following command for any integer value of QM9 molecule id.
```sh
./plot_mol_prediction.py [QM9 moelecule id]
```

### Molecular spectra of aromatic amino acids

To check extrapolation and generalization performance for large molecules not included in the training data, molecular spectra were predicted for four aromatic amino acids using weights trained with random splits and compared to those calculated with DFT.

Run the following command and `evaluation/figures/mol_elnes_cid_molecules.png` will be generated.
```sh
./plot_cid_moelcules.py
``` 

### Molecular spectra of a benzene molecule along a axis

The training data used intensities in the x, y, and z directions only, obtained from DFT calculations, but to test whether there is generalization performance for other general directions, molecular spectra were predicted for benzene molecules rotated about one axis using weights trained with random splits and compared to those calculated with DFT.

Run the following command and `evaluation/figures/rotated_benzene_x.png` will be generated.
```sh
./plot_mol_anisotropy
```

### Time comparison between DFT and IDS-PaiNN

`evaluation/data/analyzed/time_mol_castep_vs_isdpainn.pkl` and `evaluation/scripts/plot_calculation_time_comparison.py` are the data and script for the calculation/prediction time for the C-K edge in the dataset by CASTEP code and IDS-PaiNN.
To reproduce the figure visualizing the dependency on molecular size, run the following and `evaluation/figures/time_comparison.png` will be generated.
```sh
./plot_calculation_time_comparison.py
```

### Comparison between ablation experiments

To assess the contribution of individual components within our model, we conducted ablation experiments by selectively modifying components of ISD-PaiNN and comparing the resulting prediction accuracies.
Four distinct models were trained for the ablation experiments:

1. ISD-PaiNN
  The same model discussed and evaluated in the main text.

2. Without Symmetric Message Layer (w/o SM)
  This model omits the inversion symmetry and adopts the same message block as the original PaiNN.

3. Without Directional Embedding (w/o DE)
  In this model, node vector features are initialized with zero vectors, eliminating the directional embedding component present in the original PaiNN.

4. Without Symmetric Message Passing and Directional Embedding (w/o DE-SM)
  This model combines the modifications from both (2) and (3), removing both the symmetric message passing block and utilizing zero vector initialization for node vector features.
  This model is most similar to the original PaiNN model in terms of the message passing block and node vector feature initialization.
  The difference between this model and the original PaiNN model is the output block for predicting site-specific spectra.

The conversion from node-specific features to site-specific features is performed by the same output block as in ISD-PaiNN. Note that the original PaiNN model itself was not included in the ablation experiments, as its output is designed to predict energy or forces and is not tailored for predicting site-specific anisotropic spectra.
These ablation experiments were performed under random splitting conditions, utilizing the same training and test data as presented in the paper.

The configuration, weights, and MSEs of the models for the ablation experiments are available at https://doi.org/10.5281/zenodo.10566201.

To reproduce the scatter plot comparing the MSEs of these models, run the following:
```sh
./compare_ablation.py
```

## References
1. Shibata, K., Kikumasa, K., Kiyohara, S. et al. Simulated carbon K edge spectral database of organic molecules. Sci Data 9, 214 (2022). https://doi.org/10.1038/s41597-022-01303-8
2. Ruddigkeit, L., van Deursen, R., Blum, L. C. & Reymond, J.-L. Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17. Journal of Chemical Information and Modeling 52, 2864–2875, (2012). https://doi.org/10.1021/ci300415d
3. Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data 1, 140022, (2014). https://doi.org/10.1038/sdata.2014.22
4. Fey, Matthias and Lenssen, Jan E. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds (2019). https://github.com/pyg-team/pytorch_geometric/tree/master
5. K. Shibata, A dedicated Python script for making smeared C-K edge spectra from the hdf5 dataset of eigenvalues and dynamical structure factors (2024). https://github.com/nmdl-mizo/ck_edge_maker