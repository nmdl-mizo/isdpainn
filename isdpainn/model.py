"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

MIT License

Copyright (c) 2021 www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Copyright (c) 2024 Kiyou Shibata

This source code includes usage and modifications of the original code
from opcmodels.models.painn.painn.py, which is licensed under the MIT license.
The original copyright notice and MIT license text are preserved above.

Defined new classes:
- ISDPaiNN based on PaiNN with different arguments, input, and transformation
- ISDPaiNNMessage based on PaiNNMessage with a modified symmetric message function
- ISDPaiNNUpdate based on PaiNNUpdate with a modified forward function
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, segment_coo

# from ocpmodels
from ocpmodels.models.base import BaseModel
from ocpmodels.models.gemnet.layers.base_layers import ScaledSiLU
from ocpmodels.models.gemnet.layers.embedding_block import AtomEmbedding
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from ocpmodels.modules.scaling import ScaleFactor
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
# from ocpmodels.painn
from ocpmodels.models.painn.utils import get_edge_id, repeat_blocks
from ocpmodels.models.painn.painn import PaiNNUpdate, PaiNNMessage, PaiNNOutput


@registry.register_model("isdpainn")
class ISDPaiNN(BaseModel):
    """
    Inversion Symmetry-aware Directional PaiNN model

    Args:
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        out_src (str): Source of output, "invariant" or "equivariant".
        num_layers (int): Number of message passing layers.
        num_out_layers (int): Number of output layers.
        num_rbf (int): Number of radial basis functions.
        cutoff (float): Cutoff distance for radial basis functions.
        max_neighbors (int): Maximum number of neighbors.
        rbf (dict): Dictionary containing radial basis function parameters.
        envelope (dict): Dictionary containing envelope parameters.
        use_pbc (bool): Whether to use periodic boundary conditions.
        otf_graph (bool): Whether to generate graph on-the-fly.
        num_elements (int): Number of elements.
        scale_file (str): Path to scale file.
        message_factor_normalize (bool): Whether to normalize the factor in message.
        symmetric_message (bool): Whether to use symmetric message passing., default: True
    """
    def __init__(
        self,
        hidden_channels: int=512,
        out_channels: int=256, # new
        out_src: str="invariant", # new, "invariant" or "equivariant
        num_layers: int=6,
        num_out_layers: int=2, # new
        num_rbf: int=128,
        cutoff: float=12.0,
        max_neighbors: int=50,
        # output_layer="mlp",
        rbf: Dict[str, str] = {"name": "gaussian"},
        envelope: Dict[str, Union[str, int]] = {
            "name": "polynomial",
            "exponent": 5,
        },
#        regress_forces=False, # not valid
#        direct_forces=True, # not valid
        use_pbc: bool=True,
        otf_graph: bool=True,
        num_elements: int=83,
        scale_file: Optional[str] = None,
        message_factor_normalize: bool=True, # new
        symmetric_message: bool=True, # new,
        zero_initialization: bool=False, # new
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.out_src = out_src
        self.num_layers = num_layers
        self.num_out_layers = num_out_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = False # not valid, but keep for compatibility
        self.direct_forces = False # not valid, but keep for compatibility
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.message_factor_normalize = message_factor_normalize
        self.zero_initialization = zero_initialization

        # Borrowed from GemNet.
        self.symmetric_edge_symmetrization = False

        #### Learnable parameters #############################################

        self.invariant_atom_emb = AtomEmbedding(hidden_channels, num_elements)
        if not self.zero_initialization:
            self.equivariant_atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(num_layers):
            if symmetric_message:
                self.message_layers.append(
                    ISDPaiNNMessage(hidden_channels, num_rbf, self.message_factor_normalize).jittable()
                )
            else:
                self.message_layers.append(
                    PaiNNMessage(hidden_channels, num_rbf).jittable()
                )
            self.update_layers.append(ISDPaiNNUpdate(hidden_channels))
            setattr(self, "upd_out_scalar_scale_%d" % i, ScaleFactor())

        if out_src == "feature_only":
            pass
        elif out_src == "invariant":
            self.out_invariant = nn.ModuleList()
            for i in range(self.num_out_layers):
                if i < self.num_out_layers - 1:
                    self.out_invariant.append(nn.Linear(hidden_channels, hidden_channels))
                    self.out_invariant.append(ScaledSiLU())
                else:
                    self.out_invariant.append(nn.Linear(hidden_channels, out_channels))
        elif out_src == "equivariant":
            self.out_equivariant = PaiNNOutput(hidden_channels)
        else:
            raise ValueError("Invalid out_src: %s" % out_src)
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

        load_scales_compat(self, scale_file)

    def reset_parameters(self):
        if hasattr(self, "out_invariant"):
            for layer in self.out_invariant[0::2]:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)


    # Borrowed from GemNet.
    def select_symmetric_edges(
        self, tensor, mask, reorder_idx, inverse_neg
    ) -> torch.Tensor:
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    # Borrowed from GemNet.
    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        batch_idx,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        If `symmetric_edge_symmetrization` is False,
        we only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        If `symmetric_edge_symmetrization` is True,
        we always use both directions.
        """
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat(
                [edge_index, edge_index.flip(0)],
                dim=1,
            )
            cell_offsets_bothdir = torch.cat(
                [cell_offsets, -cell_offsets],
                dim=0,
            )

            # Filter for unique edges
            edge_ids = get_edge_id(
                edge_index_bothdir, cell_offsets_bothdir, num_atoms
            )
            unique_ids, unique_inv = torch.unique(
                edge_ids, return_inverse=True
            )
            perm = torch.arange(
                unique_inv.size(0),
                dtype=unique_inv.dtype,
                device=unique_inv.device,
            )
            unique_idx = scatter(
                perm,
                unique_inv,
                dim=0,
                dim_size=unique_ids.shape[0],
                reduce="min",
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]

            # Order by target index
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]

            # Subindex remaining tensors
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]

            # Count edges per image
            # segment_coo assumes sorted edge_index_new[1] and batch_idx
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(
                ones, edge_index_new[1], dim_size=num_atoms
            )
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            # Generate mask
            mask_sep_atoms = edge_index[0] < edge_index[1]
            # Distinguish edges between the same (periodic) atom by ordering the cells
            cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | (
                    (cell_offsets[:, 0] == 0)
                    & (cell_offsets[:, 1] == 0)
                    & (cell_offsets[:, 2] < 0)
                )
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            # Mask out counter-edges
            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(
                2, -1
            )

            # Concatenate counter-edges after normal edges
            edge_index_cat = torch.cat(
                [edge_index_new, edge_index_new.flip(0)],
                dim=1,
            )

            # Count remaining edges per image
            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device),
                neighbors,
            )
            batch_edge = batch_edge[mask]
            # segment_coo assumes sorted batch_edge
            # Factor 2 since this is only one half of the edges
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(
                ones, batch_edge, dim_size=neighbors.size(0)
            )

            # Create indexing array
            edge_reorder_idx = repeat_blocks(
                torch.div(neighbors_per_image, 2, rounding_mode="floor"),
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            # Reorder everything so the edges of every image are consecutive
            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, False
                )
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(
                    tensor, mask, edge_reorder_idx, True
                )
                for tensor in reorder_tensors_invneg
            ]

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            edge_index_new.flip(0), -cell_offsets_new, num_atoms
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )

    def generate_graph_values(self, data):
        (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(edge_dist, torch.tensor(0.0), atol=1e-6)
        edge_dist[mask_zero] = 1.0e-6
        edge_vector = distance_vec / edge_dist[:, None]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        # Symmetrize edges for swapping in symmetric message passing
        (
            edge_index,
            cell_offsets,
            neighbors,
            [edge_dist],
            [edge_vector],
            id_swap,
        ) = self.symmetrize_edges(
            edge_index,
            cell_offsets,
            neighbors,
            data.batch,
            [edge_dist],
            [edge_vector],
        )

        return (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data, direction=None, feature_only=False, aggregate=None):
        """
        Forward pass.

        Args:
            data (torch_geometric.data.Data): Input data.
            direction (torch.Tensor, optional): Direction of the input data.
                Defaults to None. If None, data.direction is used.
            feature_only (bool, optional): Whether to only return the features.
                Defaults to False.
            aggregate (str, optional): Aggregation method for the output.
                Defaults to None. Can be "sum" or "mean".

        Returns:
            torch.Tensor: Output prediction or features
        """
        z = data.z.long()

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        # initialize node invariant and equivariant features
        x = self.invariant_atom_emb(z)  # (nodes, hidden_channels)
        if self.zero_initialization:
            # initialize equivariant features with zeros
            vec = torch.zeros(x.shape[0], 3, x.shape[1], device=x.device)
        else:
            # initialize equivariant features according to embedding of z and direction
            if direction is None and hasattr(data, "direction"):
                direction = data.direction
                direction /= direction.norm(dim=-1, keepdim=True)
                direction = direction.to(z.device)
                # (graphs, 3)
                vec = torch.einsum(
                    "nh,nd->ndh",
                    self.equivariant_atom_emb(z),
                    direction[data.batch]
                )  # (nodes, 3, hidden_channels)
            elif direction is not None:
                if not torch.is_tensor(direction):
                    direction = torch.tensor(direction, dtype=torch.float32)
                direction /= direction.norm(dim=-1, keepdim=True)
                direction = direction.to(z.device)
                vec = torch.einsum(
                    "nh,d->ndh",
                    self.equivariant_atom_emb(z),
                    direction
                )  # (nodes, 3, hidden_channels)
            else:
                raise ValueError("Direction is not specified, and data.direction is not available.")
        #vec = self.equivariant_atom_emb(z)
        ## vec should be normalized along hidden_channels
        #vec /= vec.sum(axis=-1).unsqueeze(-1)
        #vec = torch.einsum("nh,d->ndh", vec, direction)  # (nodes, 3, hidden_channels)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_rbf, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)
        #### Output block #####################################################
        if feature_only or self.out_src == "feature_only":
            return x, vec
        elif self.out_src == "invariant":
            for layer in self.out_invariant:
                x = layer(x)
            if aggregate == "sum":
                x = scatter(x, data.batch, dim=0, reduce="sum")
            elif aggregate == "mean":
                x = scatter(x, data.batch, dim=0, reduce="mean")
            return x
        elif self.out_src == "equivariant":
            vec = self.out_equivariant(x, vec)
            if aggregate == "sum":
                vec = scatter(vec, data.batch, dim=0, reduce="sum")
            elif aggregate == "mean":
                vec = scatter(vec, data.batch, dim=0, reduce="mean")
            return vec

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"max_neighbors={self.max_neighbors}, "
            f"cutoff={self.cutoff})"
        )


class ISDPaiNNMessage(MessagePassing):
    """
    Inversion Symmetry-aware Directional PaiNN message passing layer
    """
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        message_factor_normalize=True,
    ) -> None:
        super(ISDPaiNNMessage, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.message_factor_normalize = message_factor_normalize

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        # new fixed message
        # size of tensors:
        # xh_j: (batch, hidden_channels)
        # rbfh_ij: (batch, hidden_channels)
        # x: (batch, hidden_channels)
        # xh2: (batch, hidden_channels)
        # xh3: (batch, hidden_channels)
        # factor: (batch, hidden_channels)
        # vec_j: (batch, 3, hidden_channels)
        # r_ij: (batch, 3)
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        factor = torch.sum(vec_j * r_ij.unsqueeze(2), dim=1, keepdim=True) # batch, hidden_channels
        if self.message_factor_normalize:
            factor = factor.sign()
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * factor * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

#    def message_with_bug(self, xh_j, vec_j, rbfh_ij, r_ij):
#        # size of tensors:
#        # xh_j: (batch, hidden_channels)
#        # rbfh_ij: (batch, hidden_channels)
#        # x: (batch, hidden_channels)
#        # xh2: (batch, hidden_channels)
#        # xh3: (batch, hidden_channels)
#        # factor: (batch)
#        # vec_j: (batch, 3, hidden_channels)
#        # r_ij: (batch, 3)
#        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
#        xh2 = xh2 * self.inv_sqrt_3
#        factor = torch.sum(vec_j.mean(-1) * r_ij, dim=-1)
#        if self.message_factor_normalize:
#            factor = factor.sign()
#        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * (factor.unsqueeze(1) * r_ij).unsqueeze(2)
#        vec = vec * self.inv_sqrt_h
#
#        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class ISDPaiNNUpdate(PaiNNUpdate):
    """
    Inversion Symmetry-aware PaiNN update layer
    """
    def __init__(self, hidden_channels):
        super(ISDPaiNNUpdate, self).__init__(hidden_channels)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, vec2.norm(dim=-2)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec
