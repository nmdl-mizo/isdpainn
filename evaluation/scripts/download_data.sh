#!/bin/bash
DATA_CK_DIR="dataset/raw"
DATA_MODEL_DIR="data/model"
DATA_NOMAD_DIR="data/nomad"

## Download the C-K edge dataset
#mkdir -p $DATA_CK_DIR
## Site specific eigenvalues and dynamical structure factors
## doi: 10.6084/m9.figshare.c.5494395.v1
#curl -X GET "https://api.figshare.com/v2/file/download/31947962" -o $DATA_CK_DIR/site_spectra_0.5eV.hdf5

# Download the model data
mkdir -p $DATA_MODEL_DIR
# check if zenodo_get is installed, and if not abort script
if ! command -v zenodo_get &> /dev/null
then
    echo "zenodo_get could not be found, please install it by running 'pip install zenodo_get'"
    exit
fi
# Model weights for Inversion ISD-PaiNN on simulated C-K edge spectra dataset
# doi: 10.5281/zenodo.10547718
zenodo_get -o data 10.5281/zenodo.10547718
# Model weights of ablation experiments on simulated C-K edge spectra dataset for evaluating ISD-PaiNN
# doi: 10.5281/zenodo.10566200
zenodo_get -o data 10.5281/zenodo.10566200

# Download the NOMAD data
mkdir -p $DATA_NOMAD_DIR
# C-K edge of four aromatic amino acids
# dataset_id: -wYS-_xcTce_8ufAVvJPZA
# upload_id: T-aDo4HSTjufWxmV6vZXHQ
# doi: 10.17172/NOMAD/2024.01.23-1
curl -X 'GET' \
  'https://nomad-lab.eu/prod/v1/api/v1/uploads/T-aDo4HSTjufWxmV6vZXHQ/raw/.?offset=0&length=-1&decompress=false&ignore_mime_type=false&compress=true&re_pattern=.%2A.cell%24%7C.%2A.castep%24%7C.%2A.bands%24%7C.%2A.eels_mat%24' \
  -H 'accept: application/octet-stream' \
  -o $DATA_NOMAD_DIR/aromatic_amino_acids.zip
# Carbon K edge of a series of benzene molecules rotated around the x-axis
# dataset_id: PFHr0r4-SDy-2otuTyk1Pw
# upload_id: loddw6wwSbKmyTgsHuX6PQ
# doi: 10.17172/NOMAD/2024.01.23-2
curl -X 'GET' \
  'https://nomad-lab.eu/prod/v1/api/v1/uploads/loddw6wwSbKmyTgsHuX6PQ/raw/.?offset=0&length=-1&decompress=false&ignore_mime_type=false&compress=true&re_pattern=.%2A.txt%24%7C.%2A.cell%24%7C.%2A.castep%24%7C.%2A.bands%24%7C.%2A.eels_mat%24' \
  -H 'accept: application/octet-stream' \
  -o $DATA_NOMAD_DIR/rotated_benzene.zip
