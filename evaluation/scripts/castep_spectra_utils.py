from pathlib import Path
import numpy as np
from pymatgen.core import Element
import torch
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
import ceparser as cep
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use {DEVICE}")


def parse_castep_cell(file_path):
    data = {
        'LATTICE_CART': [],
        'POSITIONS_FRAC': [],
        'POSITIONS_ABS': [],
        'SPECIES': [],
        'KPOINT_MP_GRID': None,
        'SPECTRAL_KPOINT_MP_GRID': None,
        'ELNES_KPOINT_MP_GRID': None,
        'symmetry_generate': False,
        'SPECIES_POT': [],
        'SPECIES_LCAO_STATES': []
    }
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_block = None
        for line in lines:
            line = line.strip()
            if line.startswith('%BLOCK'):
                current_block = line.split()[1]
            elif line.startswith('%ENDBLOCK'):
                current_block = None
            elif current_block:
                if current_block == 'LATTICE_CART':
                    data[current_block].append(np.array(line.split(), dtype=float))
                elif current_block == 'POSITIONS_FRAC':
                    split_line = line.split()
                    data['SPECIES'].append(split_line[0])
                    data[current_block].append(list(map(float, split_line[1:])))
                elif current_block == 'POSITIONS_ABS':
                    split_line = line.split()
                    data['SPECIES'].append(split_line[0])
                    data[current_block].append(list(map(float, split_line[1:])))
                elif current_block in ['KPOINT_MP_GRID', 'SPECTRAL_KPOINT_MP_GRID', 'ELNES_KPOINT_MP_GRID']:
                    data[current_block] = list(map(int, line.split()))
                elif current_block == 'symmetry_generate' and line == 'symmetry_generate':
                    data[current_block] = True
                elif current_block in ['SPECIES_POT', 'SPECIES_LCAO_STATES']:
                    data[current_block].append(line.split())
        for key in ['LATTICE_CART', 'POSITIONS_ABS', 'POSITIONS_FRAC', 'SPECIES', 'SPECIES_POT', 'SPECIES_LCAO_STATES']:
            data[key] = np.array(data[key])
    return data



def get_spectrum(calc_dir, energies_common, sigma, ex_site_species="C:ex", ex_site_species_original="C",
                 seed_name_eels="case_EELS", seed_name_ex="case", seed_name_gs="case-gs",
                 as_data=True):
    en_dict = cep.get_energies(Path(calc_dir, f"{seed_name_gs}.castep"), Path(calc_dir, f"{seed_name_ex}.castep"))
    en_ex = en_dict["excitation_energy"]
    sp = cep.get_smeared_spectrum(
        energies=energies_common - en_ex,
        sigma=sigma,
        calc_dir=calc_dir,
        seed_name=seed_name_eels
    )
    cell_data = parse_castep_cell(Path(calc_dir, f"{seed_name_ex}.cell"))
    ex_index = cell_data["SPECIES"].tolist().index(ex_site_species)
    node_mask = cell_data["SPECIES"] == ex_site_species
    species = cell_data["SPECIES"]
    lattice = cell_data["LATTICE_CART"]
    z = [Element(s.replace(ex_site_species, ex_site_species_original)).Z for s in species]
    if cell_data["POSITIONS_ABS"].shape[0] != 0:
        pos = cell_data["POSITIONS_ABS"]
    else:
        pos_frac = cell_data["POSITIONS_FRAC"]
        pos = np.matmul(
            lattice,
            pos_frac.T
        ).T
    if as_data:
        return Data(
            pos=torch.Tensor(pos),
            z=torch.LongTensor(z),
            spectra=torch.Tensor(sp),
            natoms=len(species),
            node_mask=torch.BoolTensor(node_mask),
#            lattice=lattice,
        )
    else:
        return {
            "spectrum": sp,
            "excitation_energy": en_ex,
            "ex_index": ex_index,
            "calc_dir": calc_dir,
            "lattice": lattice,
            "species": species,
            "z": z,
            "node_mask": node_mask,
            "pos": pos
        }


def merge_mol_data_list(data_list, force_sort=True):
    natoms = torch.unique(torch.LongTensor([d.natoms for d in data_list]))
    if len(natoms) != 1:
        raise ValueError("Number of atoms in each data is not the same.")
    pos_all = torch.stack([
        d.pos
        for d in data_list
    ])
    z_all = torch.stack([
        d.z
        for d in data_list
    ])
    if z_all.unique(dim=0).shape[0] != 1 or pos_all.unique(dim=0).shape[0] != 1:
        if force_sort:
            print("Atomic positions are not the same. Trying to sort them.")
            data_list = sort_data_list(data_list)
        else:
            raise ValueError("Atomic positions are not the same.")
    sp = torch.vstack([
        d.spectra
        for d in data_list
    ])
    sp_merged = torch.zeros(natoms, 3, sp.shape[-1])
    node_mask = torch.stack([
        d.node_mask
        for d in data_list
    ])
    sp_merged[node_mask.nonzero()[:, 1]] = sp
    return Data(
        pos=pos_all[0],
        z=z_all[0],
        spectra=sp_merged,
        natoms=natoms,
        node_mask=node_mask.any(dim=0),
#        lattice=lattice,
    )


def get_mol_spectrum(mol_dir, energies_common, sigma=0.5, sites_dir_name="sites",
                     seed_name={"ex":"case", "gs":"case_gs", "eels":"case_elnes"}):
    calc_subdir_list = [
        p
        for p in Path(mol_dir, sites_dir_name).glob("*")
        if p.is_dir() and p.name.isnumeric()
    ]
    for calc_dir in calc_subdir_list:
        for err in calc_dir.glob("*.err"):
            raise ValueError(f"Error file found: {err}")
    data_list = [
        get_spectrum(
            calc_dir,
            energies_common=energies_common,
            sigma=sigma,
            seed_name_eels=seed_name["eels"],
            seed_name_gs=seed_name["gs"],
            seed_name_ex=seed_name["ex"],
        )
        for calc_dir in calc_subdir_list
    ]
    mol_data = merge_mol_data_list(data_list)
    return mol_data

def get_true_pred_spectra(mol_dir_list, model, energies_common, sp_mean, aggregate=True, sites_dir_name="sites"):
    mol_data_dict = {
        i: get_mol_spectrum(site_dir, energies_common, sites_dir_name=sites_dir_name)
        for i, site_dir in enumerate(mol_dir_list)
    }
    data_batch = Batch.from_data_list(list(mol_data_dict.values()))
    with torch.no_grad():
        directions = torch.eye(3).to(DEVICE)
        y_pred = torch.stack(
            [
                model.forward(data_batch, direction=direction)
                for direction in directions
            ]
        ).swapaxes(0, 1) * sp_mean
        y_pred = y_pred
    if aggregate:
        y_true = data_batch.spectra

        y_true_mol = scatter(y_true[data_batch.node_mask], data_batch.batch[data_batch.node_mask], dim=0, reduce="mean")
        y_pred_mol = scatter(y_pred[data_batch.node_mask], data_batch.batch[data_batch.node_mask], dim=0, reduce="mean")
        return y_true_mol, y_pred_mol
    else:
        return y_true, y_pred

def get_permutation(a, b):
    permutation = torch.empty(b.shape[0], dtype=torch.long)
    for i in range(a.shape[0]):
        matching_rows = torch.all(torch.eq(b, a[i]), dim=-1)
        permutation[i] = torch.where(matching_rows)[0]
    return permutation

def sort_data_list(data_list):
    pos_all = torch.stack([d.pos for d in data_list])
    permutation_info = [
        get_permutation(pos_all[0], p)
        for p in pos_all
    ]
    sorted_pos = torch.stack([p[permutation] for p, permutation in zip(pos_all, permutation_info)])
    assert torch.stack([sorted_pos[0] == p for p in sorted_pos]).all()
    data_list_sorted = [
        d.update({
            "z": d.z[permutation],
            "node_mask": d.node_mask[permutation],
            "pos": d.pos[permutation],
        })
        for d, permutation in zip(data_list, permutation_info)
    ]
    return data_list_sorted
