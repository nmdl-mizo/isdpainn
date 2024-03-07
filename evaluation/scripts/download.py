#!/usr/bin/env python
"""
Download the model and NOMAD data
"""
import json
import requests
import hashlib
from pathlib import Path

def download_zenodo(record_id: int, save_dir: Path):
    """
    Download the data from Zenodo
    """
    url = f"https://zenodo.org/api/records/{record_id}/files"
    entries = json.loads(requests.get(url).text)["entries"]
    for entry in entries:
        print(f"download {entry['key']}")
        data = requests.get(entry["links"]["content"]).content
        if entry["checksum"].split(":")[1] == hashlib.md5(data).hexdigest():
            print("Checksum OK")
        else:
            raise Exception("Checksum NG")
        with open(Path(save_dir, entry["key"]), mode="wb") as f:
            f.write(data)

def download_nomad(upload_id: str, save_dir: Path):
    """
    Download the data from NOMAD
    """
    url = f"https://nomad-lab.eu/prod/v1/api/v1/uploads/{upload_id}/raw/.?offset=0&length=-1&decompress=false&ignore_mime_type=false&compress=true&re_pattern=.%2A.txt%24%7C.%2A.cell%24%7C.%2A.castep%24%7C.%2A.bands%24%7C.%2A.eels_mat%24"
    data = requests.get(url).content
    with open(Path(save_dir, f"{upload_id}.zip"), mode="wb") as f:
        f.write(data)

if __name__ == "__main__":
    # Download the model data
    model_dir = Path("../data/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    # Model weights for Inversion ISD-PaiNN on simulated C-K edge spectra dataset
    # doi: 10.5281/zenodo.10547718
    download_zenodo(10547719, model_dir)
    # Model weights of ablation experiments on simulated C-K edge spectra dataset for evaluating ISD-PaiNN
    # doi: 10.5281/zenodo.10566200
    download_zenodo(10566201, model_dir)

    # Download the NOMAD data
    nomad_dir = Path("../data/nomad")
    nomad_dir.mkdir(parents=True, exist_ok=True)
    # C-K edge of four aromatic amino acids
    # dataset_id: -wYS-_xcTce_8ufAVvJPZA
    # upload_id: T-aDo4HSTjufWxmV6vZXHQ
    # doi: 10.17172/NOMAD/2024.01.23-1
    download_nomad("T-aDo4HSTjufWxmV6vZXHQ", nomad_dir)
    # Carbon K edge of a series of benzene molecules rotated around the x-axis
    # dataset_id: PFHr0r4-SDy-2otuTyk1Pw
    # upload_id: loddw6wwSbKmyTgsHuX6PQ
    # doi: 10.17172/NOMAD/2024.01.23-2
    download_nomad("loddw6wwSbKmyTgsHuX6PQ", nomad_dir)
