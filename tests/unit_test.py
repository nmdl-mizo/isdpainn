#!/usr/bin/env python
import unittest
import torch
from torch_geometric.data import Data, Batch
from isdpainn.model import ISDPaiNN, ISDPaiNNMessage, ISDPaiNNUpdate
from isdpainn.utils import rotation_matrix, make_data_batch, rotate_data, invert_data
from ocpmodels.models.painn.painn import PaiNNMessage


class TestISDPaiNN(unittest.TestCase):
    """Test ISDPaiNN."""
    def setUp(self) -> None:
        """Set up test."""
        # tolerance for numerical error
        # Typically atol, rtol=1e-4 is enough to check and is satisfied.
        # However, it is rarely not satisfied for models with many message and update blocks
        # and unphysically dense molecular graph,
        # possibly due to the accumulation of numerial error.
        self.atol = 1e-4 # tolerance for torch.testing.assert_close
        self.rtol = 1e-4 # tolerance for torch.testing.assert_close

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data parameters
        self.n = 6 # number of atoms
        self.z_max = 83 # maximum atomic number
        # comment out the two lines below to try random seed for test
        self.seed = 12345
        torch.random.manual_seed(self.seed)
        # model parameters
        self.model_params = {
            "hidden_channels": 512,
            "out_channels": 256,
            "num_elements": self.z_max,
            "num_layers": 6,
            "num_rbf": 128,
            "use_pbc": False,
            "symmetric_message": True
        }
        self.hidden_channels = self.model_params["hidden_channels"]
        self.normalize = True
        # prepare data
        self.prepare_data()

    def test_feature_symmetry(self):
        """Test feature symmetry."""
        model = ISDPaiNN(
            **self.model_params,
            out_src="feature_only",
            message_factor_normalize=self.normalize
        ).to(self.device)
        model.eval()
        # original
        s_pred, v_pred = model.forward(self.data, self.n_vec)
        # transformed before prediction
        s_inv_pred, v_inv_pred = model.forward(self.data_inv, self.n_vec_inv)
        s_rot_pred, v_rot_pred = model.forward(self.data_rot, self.n_vec_rot)
        s_invn_pred, v_invn_pred = model.forward(self.data, self.n_vec_inv)
        s_invg_pred, v_invg_pred = model.forward(self.data_inv, self.n_vec)
        # transformed after prediction
        v_pred_inv = -v_pred
        v_pred_rot = torch.mm(v_pred.swapaxes(1, 2).reshape(-1, 3), self.rot_mat).reshape(v_pred.shape[0], v_pred.shape[2], 3).swapaxes(1, 2)

        # s invarinance
        torch.testing.assert_close(s_pred, s_rot_pred, atol=self.atol, rtol=self.rtol) # s invariance about rotation
        torch.testing.assert_close(s_pred, s_inv_pred, atol=self.atol, rtol=self.rtol) # s invariance about inversion
        torch.testing.assert_close(s_pred, s_invn_pred, atol=self.atol, rtol=self.rtol) # s invariance about inversion on n
        torch.testing.assert_close(s_pred, s_invg_pred, atol=self.atol, rtol=self.rtol) # s invariance about inversion on G

        # v equivariance
        torch.testing.assert_close(v_rot_pred, v_pred_rot, atol=self.atol, rtol=self.rtol) # v equivariance about rotation
        torch.testing.assert_close(v_pred_inv, v_inv_pred, atol=self.atol, rtol=self.rtol) # v equivariance about inversion
        torch.testing.assert_close(v_pred_inv, v_invn_pred, atol=self.atol, rtol=self.rtol) # v equivariance about inversion on n
        torch.testing.assert_close(v_pred, v_invg_pred, atol=self.atol, rtol=self.rtol) # v invariance about inversion on G

    def test_invariant_symmetry(self):
        """Test invariant symmetry."""
        model = ISDPaiNN(
            **self.model_params,
            out_src="invariant",
            message_factor_normalize=self.normalize
        ).to(self.device)
        model.eval()
        # original
        y_pred = model.forward(self.data, self.n_vec)
        # transformed before prediction
        y_inv_pred = model.forward(self.data_inv, self.n_vec_inv)
        y_rot_pred = model.forward(self.data_rot, self.n_vec_rot)
        y_invn_pred = model.forward(self.data, self.n_vec_inv)
        y_invg_pred = model.forward(self.data_inv, self.n_vec)

        # invariant output
        torch.testing.assert_close(y_pred, y_rot_pred, atol=self.atol, rtol=self.rtol) # invariance about rotation
        torch.testing.assert_close(y_pred, y_inv_pred, atol=self.atol, rtol=self.rtol) # invariance about inversion
        torch.testing.assert_close(y_pred, y_invn_pred, atol=self.atol, rtol=self.rtol) # invariance about inversion on n
        torch.testing.assert_close(y_pred, y_invg_pred, atol=self.atol, rtol=self.rtol) # invariance about inversion on G

    def test_equivariant_symmetry(self):
        """Test equivariant symmetry."""
        model = ISDPaiNN(
            **self.model_params,
            out_src="equivariant",
            message_factor_normalize=self.normalize
        ).to(self.device)
        model.eval()
        # original
        y_pred = model.forward(self.data, self.n_vec)
        # transformed before prediction
        y_inv_pred = model.forward(self.data_inv, self.n_vec_inv)
        y_rot_pred = model.forward(self.data_rot, self.n_vec_rot)
        y_invn_pred = model.forward(self.data, self.n_vec_inv)
        y_invg_pred = model.forward(self.data_inv, self.n_vec)
        # transformed after prediction
        y_pred_inv = -y_pred
        y_pred_rot = torch.mm(y_pred, self.rot_mat)

        # equivariant output
        torch.testing.assert_close(y_pred_rot, y_rot_pred, atol=self.atol, rtol=self.rtol) # equivariance about rotation
        torch.testing.assert_close(y_pred_inv, y_inv_pred, atol=self.atol, rtol=self.rtol) # equivariance about inversion
        torch.testing.assert_close(y_pred_inv, y_invn_pred, atol=self.atol, rtol=self.rtol) # equivariance about inversion on n
        torch.testing.assert_close(y_pred, y_invg_pred, atol=self.atol, rtol=self.rtol) # invariance about inversion on G

    def test_message_layer(self):
        """Test message layer."""
        # prepare model
        message_block = ISDPaiNNMessage(
            hidden_channels=self.hidden_channels,
            num_rbf=self.model_params["num_rbf"],
            message_factor_normalize=self.normalize
        ).to(self.device)
        message_block.eval()
        # check invariance and equivariance of the message function input
        # generate graph values using the model methods
        model = ISDPaiNN(
            **self.model_params,
            out_src="equivariant"
        ).to(self.device)
        model.eval()
        # original
        (
            edge_index,
            _, #neighbors not used,
            edge_dist,
            edge_vector,
            _, #id_swap not used,
        ) = model.generate_graph_values(self.data)
        edge_rbf = model.radial_basis(edge_dist) 
        # rotation
        (
            edge_index_rot,
            _, # neighbors not used,
            edge_dist_rot,
            edge_vector_rot,
            _, # id_swap not used,
        ) = model.generate_graph_values(self.data_rot)
        edge_rbf_rot = model.radial_basis(edge_dist_rot) 
        # inversion
        (
            edge_index_inv,
            _, #neighbors not used,
            edge_dist_inv,
            edge_vector_inv,
            _, #id_swap not used,
        ) = model.generate_graph_values(self.data_inv)
        edge_rbf_inv = model.radial_basis(edge_dist_inv) 
        # check
        torch.testing.assert_close(edge_index, edge_index_rot)
        torch.testing.assert_close(edge_dist, edge_dist_rot)
        torch.testing.assert_close(torch.mm(edge_vector, self.rot_mat), edge_vector_rot) # rotated due to rotation
        torch.testing.assert_close(edge_index, edge_index_inv)
        torch.testing.assert_close(edge_dist, edge_dist_inv)
        torch.testing.assert_close(edge_vector, -edge_vector_inv) # opposite due to inversion

        # check invariance and equivariance of the message function output
        # original
        s_out, v_out = message_block.forward(self.s, self.v, edge_index, edge_rbf, edge_vector)
        # transformed before prediction
        s_inv_out, v_inv_out = message_block.forward(self.s, self.v_inv, edge_index_inv, edge_rbf_inv, edge_vector_inv)
        s_rot_out, v_rot_out = message_block.forward(self.s, self.v_rot, edge_index_rot, edge_rbf_rot, edge_vector_rot)
        s_invn_out, v_invn_out = message_block.forward(self.s, self.v_inv, edge_index, edge_rbf, edge_vector)
        s_invg_out, v_invg_out = message_block.forward(self.s, self.v, edge_index_inv, edge_rbf_inv, edge_vector_inv)
        # transformed after prediction
        v_out_inv = -v_out
        v_out_rot = torch.mm(v_out.swapaxes(1, 2).reshape(-1, 3), self.rot_mat).reshape(v_out.shape[0], v_out.shape[2], 3).swapaxes(1, 2)

        # s invarinance
        torch.testing.assert_close(s_out, s_rot_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_inv_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_invn_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_invg_out, atol=self.atol, rtol=self.rtol)
        # v equivariance and invariance
        torch.testing.assert_close(v_rot_out, v_out_rot, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(v_inv_out, v_out_inv, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(v_invn_out, v_out_inv, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(v_invg_out, v_out, atol=self.atol, rtol=self.rtol)
        print("\nv equivariance error about rotation in ISDPaiNNMessage:")
        self.print_error(v_rot_out, v_out_rot)


    def test_message_layer_painn(self):
        """Test message layer."""
        # prepare model
        message_block = PaiNNMessage(
            hidden_channels=self.hidden_channels,
            num_rbf=self.model_params["num_rbf"]
        ).to(self.device)
        message_block.eval()
        # check invariance and equivariance of the message function input
        # generate graph values using the model methods
        model = ISDPaiNN(
            **self.model_params,
            out_src="equivariant"
        ).to(self.device)
        model.eval()
        # original
        (
            edge_index,
            _, #neighbors not used,
            edge_dist,
            edge_vector,
            _, #id_swap not used,
        ) = model.generate_graph_values(self.data)
        edge_rbf = model.radial_basis(edge_dist) 
        # rotation
        (
            edge_index_rot,
            _, # neighbors not used,
            edge_dist_rot,
            edge_vector_rot,
            _, # id_swap not used,
        ) = model.generate_graph_values(self.data_rot)
        edge_rbf_rot = model.radial_basis(edge_dist_rot) 
        # inversion
        (
            edge_index_inv,
            _, #neighbors not used,
            edge_dist_inv,
            edge_vector_inv,
            _, #id_swap not used,
        ) = model.generate_graph_values(self.data_inv)
        edge_rbf_inv = model.radial_basis(edge_dist_inv) 
        # check
        torch.testing.assert_close(edge_index, edge_index_rot)
        torch.testing.assert_close(edge_dist, edge_dist_rot)
        torch.testing.assert_close(torch.mm(edge_vector, self.rot_mat), edge_vector_rot) # rotated due to rotation
        torch.testing.assert_close(edge_index, edge_index_inv)
        torch.testing.assert_close(edge_dist, edge_dist_inv)
        torch.testing.assert_close(edge_vector, -edge_vector_inv) # opposite due to inversion

        # check invariance and equivariance of the message function output
        # original
        s_out, v_out = message_block.forward(self.s, self.v, edge_index, edge_rbf, edge_vector)
        # transformed before prediction
        s_inv_out, v_inv_out = message_block.forward(self.s, self.v_inv, edge_index_inv, edge_rbf_inv, edge_vector_inv)
        s_rot_out, v_rot_out = message_block.forward(self.s, self.v_rot, edge_index_rot, edge_rbf_rot, edge_vector_rot)
        s_invn_out, v_invn_out = message_block.forward(self.s, self.v_inv, edge_index, edge_rbf, edge_vector)
        # transformed after prediction
        v_out_inv = -v_out
        v_out_rot = torch.mm(v_out.swapaxes(1, 2).reshape(-1, 3), self.rot_mat).reshape(v_out.shape[0], v_out.shape[2], 3).swapaxes(1, 2)

        # s invarinance
        torch.testing.assert_close(s_out, s_rot_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_inv_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_invn_out, atol=self.atol, rtol=self.rtol)

        # v equivariance
        torch.testing.assert_close(v_rot_out, v_out_rot, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(v_inv_out, v_out_inv, atol=self.atol, rtol=self.rtol)
#        torch.testing.assert_close(v_invn_out, v_out_inv, atol=self.atol, rtol=self.rtol)
        print("\nv equivariance error about rotation in PaiNNMessage:")
        self.print_error(v_rot_out, v_out_rot)


    def test_update_layer(self):
        """Test update layer."""
        # prepare model
        update_block = ISDPaiNNUpdate(hidden_channels=self.hidden_channels).to(self.device)
        update_block.eval()
        # original
        s_out, v_out = update_block.forward(self.s, self.v)
        # transformed before prediction
        s_rot_out, v_rot_out = update_block.forward(self.s, self.v_rot)
        s_inv_out, v_inv_out = update_block.forward(self.s, self.v_inv)
        # transformed after prediction
        v_out_inv = -v_out
        v_out_rot = torch.mm(v_out.swapaxes(1, 2).reshape(-1, 3), self.rot_mat).reshape(v_out.shape[0], v_out.shape[2], 3).swapaxes(1, 2)

        # s invarinance
        torch.testing.assert_close(s_out, s_rot_out, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(s_out, s_inv_out, atol=self.atol, rtol=self.rtol)
    
        # v equivariance
        torch.testing.assert_close(v_rot_out, v_out_rot, atol=self.atol, rtol=self.rtol)
        torch.testing.assert_close(v_inv_out, v_out_inv, atol=self.atol, rtol=self.rtol)

    def prepare_data(self):
        """Prepare data for test."""
        # prepare data
        # original input
        self.data = make_data_batch(self.n, self.z_max).to(self.device)
        self.data.update({'atomic_numbers': self.data.z})
        self.data.to(self.device)
        n_vec = torch.randn(3).to(self.device)
        self.n_vec = n_vec / torch.norm(n_vec)

        # symmetryic input
        # inversion
        self.data_inv = invert_data(self.data)
        self.n_vec_inv = -self.n_vec

        # rotation
        n_rot = torch.randn(3)
        self.rot_mat = rotation_matrix(n_rot).to(self.device)
        assert torch.allclose(
            torch.matmul(self.rot_mat, self.rot_mat.T),
            torch.eye(3).to(self.device),
            atol=self.atol,
        )
        self.data_rot = rotate_data(self.data, self.rot_mat)
        self.n_vec_rot = torch.mm(self.n_vec.unsqueeze(0), self.rot_mat).squeeze(0)

        # prepare random feature for Message and Update blocks
        self.s = torch.randn(self.n, self.hidden_channels).to(self.device)
        self.v = torch.randn(self.n, 3, self.hidden_channels).to(self.device)
        self.v_inv = -self.v
        self.v_rot = torch.mm(
            self.v.swapaxes(1, 2).reshape(-1, 3),
            self.rot_mat
        ).reshape(self.v.shape[0], self.v.shape[2], 3).swapaxes(1, 2)


    def print_error(self, a, b, digits=4):
        """Print absolute and relative error between tensors a and b."""
        print(f"absolute error: {torch.abs(a - b).max().item():.{digits}e}")
        print(f"relative error: {(torch.abs(a - b) / torch.abs(a)).max().item():.{digits}e}")


if __name__ == "__main__":
    unittest.main()
