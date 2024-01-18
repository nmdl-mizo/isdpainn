from typing import Callable
import torch
from torch_geometric.data import Batch, Data


def rotation_matrix(n: torch.Tensor) -> torch.Tensor:
    """
    Generate a rotation matrix using Rodrigues' rotation formula.

    This function takes a tensor `n` representing the rotation axis and magnitude,
    and returns a rotation matrix. The rotation axis is given by the direction of `n`,
    and the rotation angle is given by the magnitude of `n`.

    Args:
        n (torch.Tensor): A tensor representing the rotation axis and magnitude.

    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """
    norm = torch.norm(n)

    # Check if the norm is zero (i.e., n is a zero vector)
    if norm == 0:
        raise ValueError("The input tensor n should not be a zero vector.")

    n = n / norm
    n_cross = torch.tensor([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    # Rodrigues' rotation formula
    rotation_matrix = torch.eye(3) + torch.sin(norm) * n_cross + (1 - torch.cos(norm)) * torch.matmul(n_cross, n_cross)
    return rotation_matrix

def make_data_batch(n: int, z_max: int, n_graph: int=1) -> Batch:
    """
    Create a batch of data for testing purposes.

    This function generates `n_graph` number of graphs, each with `n` random positions
    and atomic numbers, with the atomic numbers ranging from 1 to `z_max`. These are then used to
    create a list of Data objects which is converted into a Batch object and returned.

    Args:
        n (int): The number of random positions and atomic numbers to generate for each graph.
        z_max (int): The maximum atomic number for the random atomic numbers.
        n_graph (int, optional): The number of graphs to generate. Defaults to 1.

    Returns:
        Batch: A Batch object containing the generated data for `n_graph` number of graphs.
    """
    data_list = []
    for _ in range(n_graph):
        # Generate `n` random positions
        pos = torch.rand(n, 3)
        # Generate `n` random atomic numbers, ranging from 1 to `z_max`
        z = torch.randint(low=1, high=z_max, size=(n,), dtype=torch.long)
        # Create a Data object with the generated positions and atomic numbers
        data = Data(pos=pos, z=z, natoms=torch.tensor([n]))
        data_list.append(data)
    # Convert the Data object to a Batch object
    data_batch = Batch.from_data_list(data_list)
    return data_batch

def transform_data_batch(data_batch: Batch, transform_func: Callable) -> Batch:
    """
    Apply a transformation function to the position data in a DataBatch object.

    This function takes a DataBatch object and a transformation function, applies the transformation
    to the position data, and returns a new DataBatch object with the transformed position data.

    Args:
        data_batch (Batch): A Batch object containing the position data to be transformed.
        transform_func (Callable): A function that takes a tensor and returns a tensor of the same shape.

    Returns:
        Batch: A new Batch object with the transformed position data.
    """
    new_data_batch = data_batch.clone()
    new_data_batch.update({"pos": transform_func(data_batch.pos)})
    return new_data_batch

def rotate_data(data_batch: Batch, R: torch.Tensor) -> Batch:
    """
    Rotate the position data using a given rotation matrix.

    This function takes a data_batch object and a rotation matrix `R`, applies the rotation
    to the position data, and returns a new data_batch object with the rotated position data.

    Args:
        data_batch (Batch): A Batch object containing the position data to be rotated.
        R (torch.Tensor): A 3x3 rotation matrix.

    Returns:
        Batch: A new Batch object with the rotated position data.
    """
    rotate_func = lambda pos: torch.mm(pos, R)
    return transform_data_batch(data_batch, rotate_func)

def invert_data(data_batch: Batch) -> Batch:
    """
    Invert the position data.

    This function takes a data_batch object, inverts the position data,
    and returns a new data_batch object with the inverted position data.

    Args:
        data_batch (Batch): A Batch object containing the position data to be inverted.

    Returns:
        Batch: A new Batch object with the inverted position data.
    """
    invert_func = lambda pos: -pos
    return transform_data_batch(data_batch, invert_func)
