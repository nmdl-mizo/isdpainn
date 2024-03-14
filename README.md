# Inversion Symmetry-aware Directional PaiNN (ISD-PaiNN)

The Inversion Symmetry-aware Directional PaiNN (ISD-PaiNN) is a machine learning model for predicting physical properties based on molecular graphs and their orientations. This model extends the Polarizable Atom Interaction Neural Network (PaiNN) by introducing an orientation vector as an additional input and making several other modifications to enhance the model’s ability to regress physical properties. 

This model is proposed in a paper
[Open Review: iSFsLFsGYX](https://openreview.net/forum?id=iSFsLFsGYX), which was accepted in [AI for 
Accelerated Materials Design(AI4Mat) - NeurIPS 2023](https://sites.google.com/view/ai4mat/home) as a spot-light talk.

This repository provides the implementation of ISD-PaiNN, along with installation instructions, testing scripts, and evaluation experiments.

### Model architecture
The model architecture is largely based on the Polarizable Atom Interaction Neural Network (PaiNN) proposed by Kristof T. Schütt, Oliver T. Unke, and Michael Gastegger in the paper:
"Equivariant message passing for the prediction of tensorial properties and molecular spectra"
[arXiv:2102.03150](https://arxiv.org/abs/2102.03150).

The main changes from the PaiNN model are as follows:
1. Input an orientation vector $\hat{\mathbf{n}}$ in addition to molecular graph $\mathcal{G}$.
1. Non-zero initialization of node-wise equivariant vector feature $\vec{\mathbf{v}}_i^0$.
1. Symmetrized message block to satisfy the resultant equivariance and invariance of the features $\vec{\mathbf{v}}_i^l$ and $\mathbf{s}_i^l$ including spatial inversion on either $\mathcal{G}$ or $\hat{\mathbf{n}}$.

These changes enable to regress physical properties not only on molecular graph but also orientation relative to the graph.

The code is developed based on the implementation of PaiNN in [`ocp-models.models.painn`](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/painn).

## Prerequisities
- A computer with a CPU capable of running Python 3.10 and PyTorch. Creating a dedicated Python environment using conda is recommended.
- A GPU is highly recommended for training and inference. The GPU should be compatible with CUDA 12.1 if you want to take advantage of PyTorch's GPU acceleration.

## Installation
1. Install `pytorch`, `torch_geometric`, and `ocp-models` in advance depending on your environment (GPU/CPU).
    - [pytorch](https://pytorch.org/)
    - [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    - [ocp-models](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md)
    - The packages above can be installed using conda as follows:
        ```
        conda create -n isdpainn-gpu python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        conda activate isdpainn-gpu
        conda install -qy pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg
        pip install -e git+https://github.com/Open-Catalyst-Project/ocp.git@main#egg=ocp-models
        ```
1. Clone this repository and run `pip install .` in the repository directory, or run `pip install https://github.com/nmdl-mizo/isdpainn.git@main#egg=isdpainn` to install directly from GitHub.

## Docker
Dockerfile is available in the repository.
You can build the docker image by running `docker build -t isdpainn .` in the repository directory.

## Test
A test script for checking invariance and equivariance of the model is available in `tests/test_model.py`.
You can run the test by `pytest tests/unit_test.py`.

## Evaluation experiment on simulated C-K edge dataset
Some scripts, weights and description is available for Evaluation experiment on simulated C-K edge dataset.
Please check [here](/evaluation/README.md).

## License
This code is released under the [MIT license](./LICENSE).

Some parts of this module include usage and modifications of the original code from `opcmodels.models.painn.painn.py`, which is licensed under the MIT license.
The original copyright notice and MIT license text are preserved in `isdpainn/model.py`.

## Citing ISD-PaiNN
If you use this code in your work, please consider citing the following for the time being:
```
@inproceedings{
shibata2023message,
title={Message Passing Neural Network for Predictig Dipole Moment Dependent Core Electron Excitation Spectra},
author={Kiyou Shibata and Teruyasu Mizoguchi},
booktitle={AI for Accelerated Materials Design - NeurIPS 2023 Workshop},
year={2023},
url={https://openreview.net/forum?id=iSFsLFsGYX}
}
```

## References
- PaiNN [schütt2021equivariant]
- ocp by Open Catalyst Project [ocp_dataset]
```
@misc{schütt2021equivariant,
      title={Equivariant message passing for the prediction of tensorial properties and molecular spectra}, 
      author={Kristof T. Schütt and Oliver T. Unke and Michael Gastegger},
      year={2021},
      eprint={2102.03150},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
