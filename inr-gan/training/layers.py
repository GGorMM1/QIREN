from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from torch_utils import persistence
from torch_utils.ops import bias_act
from torch_utils import misc
import torchquantum as tq
import math


# ----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, type: str, channel_dim: int, w_dim: int, resolution: int):
        super().__init__()
        self.type = type

        if type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, resolution, resolution]))
            self.total_dim = channel_dim
        elif type == 'coords':
            self.input = CoordsInput(w_dim, resolution)
            self.total_dim = self.input.get_total_dim()
        else:
            raise NotImplementedError

    def forward(self, batch_size: int, w: Tensor = None, device=None, dtype=None, memory_format=None) -> Tensor:
        if self.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.type == 'coords':
            x = self.input(batch_size, w, device=device, dtype=dtype, memory_format=memory_format)
        else:
            raise NotImplementedError

        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class CoordsInput(nn.Module):
    def __init__(self, w_dim: int, resolution: int):
        super().__init__()
        self.resolution = resolution
        self.coord_fuser = CoordFuser(w_dim, resolution)

    def get_total_dim(self) -> int:
        return self.coord_fuser.total_dim

    def forward(self, batch_size: int, w: Optional[Tensor] = None, device='cpu', dtype=None,
                memory_format=None) -> Tensor:
        dummy_input = torch.empty(batch_size, 0, self.resolution, self.resolution)
        dummy_input = dummy_input.to(device, dtype=dtype, memory_format=memory_format)
        out = self.coord_fuser(dummy_input, w, dtype=dtype, memory_format=memory_format)

        return out


# ----------------------------------------------------------------------------

@persistence.persistent_class
class CoordFuser(nn.Module):
    """
    CoordFuser which concatenates coordinates across dim=1 (we assume channel_first format)
    """

    def __init__(self, w_dim: int, resolution: int):
        super().__init__()

        self.resolution = resolution
        self.log_emb_size = 0
        self.random_emb_size = 0
        self.shared_emb_size = 0
        self.predictable_emb_size = 0
        self.const_emb_size = 0
        self.fourier_scale = np.sqrt(10)
        self.use_cosine = False
        self.use_raw_coords = True
        self.init_dist = 'randn'
        self._fourier_embs_cache = None
        self._full_cache = None
        self.use_full_cache = False

        if self.log_emb_size > 0:
            self.register_buffer('log_basis', generate_logarithmic_basis(
                resolution, self.log_emb_size,
                use_diagonal=True))  # [log_emb_size, 2]

        if self.random_emb_size > 0:
            self.register_buffer('random_basis', self.sample_w_matrix((self.random_emb_size, 2), self.fourier_scale))

        if self.shared_emb_size > 0:
            self.shared_basis = nn.Parameter(self.sample_w_matrix((self.shared_emb_size, 2), self.fourier_scale))

        if self.predictable_emb_size > 0:
            self.W_size = self.predictable_emb_size * 2
            self.b_size = self.predictable_emb_size
            self.affine = FullyConnectedLayer(w_dim, self.W_size + self.b_size, bias_init=0)

        if self.const_emb_size > 0:
            self.const_embs = nn.Parameter(torch.randn(1, self.const_emb_size, resolution, resolution).contiguous())

        self.total_dim = self.get_total_dim()
        self.is_modulated = (self.predictable_emb_size > 0)

    def sample_w_matrix(self, shape: Tuple[int], scale: float):
        if self.init_dist == 'randn':
            return torch.randn(shape) * scale
        elif self.init_dist == 'rand':
            return (torch.rand(shape) * 2 - 1) * scale
        else:
            raise NotImplementedError(f"Unknown init dist: {self.init_dist}")

    def get_total_dim(self) -> int:

        total_dim = 0
        total_dim += (2 if self.use_raw_coords else 0)
        if self.log_emb_size > 0:
            total_dim += self.log_basis.shape[0] * (2 if self.use_cosine else 1)
        total_dim += self.random_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.shared_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.predictable_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.const_emb_size

        return total_dim

    def forward(self, x: Tensor, w: Tensor = None, dtype=None, memory_format=None) -> Tensor:
        """
        Dims:
            @arg x is [batch_size, in_channels, img_size, img_size]
            @arg w is [batch_size, w_dim]
            @return out is [batch_size, in_channels + fourier_dim + cips_dim, img_size, img_size]
        """
        assert memory_format is torch.contiguous_format

        if False:
            return x

        batch_size, in_channels, img_size = x.shape[:3]
        out = x

        if self.use_full_cache and (not self._full_cache is None) and (self._full_cache.device == x.device) and \
                (self._full_cache.shape == (batch_size, self.get_total_dim(), img_size, img_size)):
            return torch.cat([x, self._full_cache], dim=1)

        if (not self._fourier_embs_cache is None) and (self._fourier_embs_cache.device == x.device) and \
                (self._fourier_embs_cache.shape == (
                        batch_size, self.get_total_dim() - self.const_emb_size, img_size, img_size)):
            out = torch.cat([out, self._fourier_embs_cache], dim=1)
        else:
            raw_embs = []
            raw_coords = generate_coords(batch_size, img_size, x.device)  # [batch_size, coord_dim, img_size, img_size]

            if self.use_raw_coords:
                out = torch.cat([out, raw_coords.to(dtype=dtype, memory_format=memory_format)], dim=1)

            if self.log_emb_size > 0:
                log_bases = self.log_basis.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, log_emb_size, 2]
                raw_log_embs = torch.einsum('bdc,bcxy->bdxy', log_bases,
                                            raw_coords)  # [batch_size, log_emb_size, img_size, img_size]
                raw_embs.append(raw_log_embs)

            if self.random_emb_size > 0:
                random_bases = self.random_basis.unsqueeze(0).repeat(batch_size, 1,
                                                                     1)  # [batch_size, random_emb_size, 2]
                raw_random_embs = torch.einsum('bdc,bcxy->bdxy', random_bases,
                                               raw_coords)  # [batch_size, random_emb_size, img_size, img_size]
                raw_embs.append(raw_random_embs)

            if self.shared_emb_size > 0:
                shared_bases = self.shared_basis.unsqueeze(0).repeat(batch_size, 1,
                                                                     1)  # [batch_size, shared_emb_size, 2]
                raw_shared_embs = torch.einsum('bdc,bcxy->bdxy', shared_bases,
                                               raw_coords)  # [batch_size, shared_emb_size, img_size, img_size]
                raw_embs.append(raw_shared_embs)

            if self.predictable_emb_size > 0:
                misc.assert_shape(w, [batch_size, None])
                mod = self.affine(w)  # [batch_size, W_size + b_size]
                W = self.fourier_scale * mod[:, :self.W_size]  # [batch_size, W_size]
                W = W.view(batch_size, self.predictable_emb_size, 2)  # [batch_size, predictable_emb_size, coord_dim]
                bias = mod[:, self.W_size:].view(batch_size, self.predictable_emb_size, 1,
                                                 1)  # [batch_size, predictable_emb_size, 1]
                raw_predictable_embs = (torch.einsum('bdc,bcxy->bdxy', W,
                                                     raw_coords) + bias)  # [batch_size, predictable_emb_size, img_size, img_size]
                raw_embs.append(raw_predictable_embs)

            if len(raw_embs) > 0:
                raw_embs = torch.cat(raw_embs,
                                     dim=1)  # [batch_suze, log_emb_size + random_emb_size + predictable_emb_size, img_size, img_size]
                raw_embs = raw_embs.contiguous()  # [batch_suze, -1, img_size, img_size]
                out = torch.cat([out, raw_embs.sin().to(dtype=dtype, memory_format=memory_format)],
                                dim=1)  # [batch_size, -1, img_size, img_size]

                if self.use_cosine:
                    out = torch.cat([out, raw_embs.cos().to(dtype=dtype, memory_format=memory_format)],
                                    dim=1)  # [batch_size, -1, img_size, img_size]

        if self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._fourier_embs_cache = out[:, x.shape[1]:].detach()

        if self.const_emb_size > 0:
            const_embs = self.const_embs.repeat([batch_size, 1, 1, 1])
            const_embs = const_embs.to(dtype=dtype, memory_format=memory_format)
            out = torch.cat([out, const_embs], dim=1)  # [batch_size, total_dim, img_size, img_size]

        if self.use_full_cache and self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > \
                x.shape[1]:
            self._full_cache = out[:, x.shape[1]:].detach()

        return out


def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool = False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float()  # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1  # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1)  # [img_size, img_size]
    y_coords = x_coords.t().flip(dims=(0,))  # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2)  # [img_size, img_size, 2]
    coords = coords.view(-1, 2)  # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1,
                                                              1)  # [batch_size, 2, img_size, img_size]

    return coords


def generate_logarithmic_basis(
        resolution: int,
        max_num_feats: int = np.float('inf'),
        remove_lowest_freq: bool = False,
        use_diagonal: bool = True) -> Tensor:
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    # while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
    #     bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis


def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_wavefront_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1)  # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1)  # [num_feats, 1]
    result = basis * powers * period_coef  # [num_feats, 2]

    return result.float()


class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """

    def __init__(self, in_channels, out_channels, learnable_features=False):
        super(FourierFeatures, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels, in_channels),
                                        std=1.0)
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.einsum('oi,bihw->bohw', self.frequency_matrix.to(coordinates.device), coordinates)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, ansatz_layer, use_noise):
        super().__init__()
        self.n_wires = in_features
        self.spectrum_layer = spectrum_layer
        self.ansatz_layer = ansatz_layer

    def entangler(self, qdev):
        for i in range(self.n_wires - 1):
            qdev.cnot([i, i + 1])
        # qdev.cnot([self.n_wires - 1, 0])

    def forward(self, x, qstyles):
        b, hw, c = x.shape
        # qstyles = qstyles.reshape((b, -1)).unsqueeze(1).repeat(1, hw, 1)
        qstyles = qstyles.unsqueeze(1).repeat(1, hw, 1, 1, 1, 1)
        qstyles = qstyles.reshape((b * hw, self.spectrum_layer + 1, self.ansatz_layer, self.n_wires, 3))

        x = x.reshape((-1, c))
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=b * hw, device=x.device)

        for j in range(self.ansatz_layer):
            for k in range(self.n_wires):
                qdev.u3(params=qstyles[:, 0, j, k, :], wires=k)
            if self.n_wires > 1:
                self.entangler(qdev)

        for i in range(self.spectrum_layer):
            for k in range(self.n_wires):
                qdev.rz(params=x[:, k], wires=k)
            for j in range(self.ansatz_layer):
                for k in range(self.n_wires):
                    qdev.u3(params=qstyles[:, i + 1, j, k, :], wires=k)
                if self.n_wires > 1:
                    self.entangler(qdev)

        out = tq.measurement.expval_joint_analytical(qdev, observable="ZI")
        return out.reshape(b, hw)


class QuantumCircuitFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, learnable_features=False):
        super().__init__()
        self.learnable_features = learnable_features
        self.out_channels = out_channels
        self.qlayer_features = 2
        self.ansatz_layer = 2
        self.spectrum_layer = 2
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels * 2, in_channels),
                                        std=1.0)
        quantum_circuit_param = torch.normal(
            mean=torch.zeros(1, self.spectrum_layer + 1, self.ansatz_layer, out_channels * 2, 3), std=1.0)

        for idx in range(out_channels):
            qlayer = QuantumLayer(self.qlayer_features, self.spectrum_layer, self.ansatz_layer, False)
            setattr(self, f'qlayer{idx}', qlayer)

        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
            self.norm = nn.BatchNorm1d(out_channels * 2)
            self.quantum_circuit_param = nn.Parameter(quantum_circuit_param)
        else:
            self.register_buffer('frequency_matrix', frequency_matrix)
            self.register_buffer('quantum_circuit_param', quantum_circuit_param)
            # self.register_buffer('out', torch.empty(0))

    def forward(self, coordinates):
        b, c, h, w = coordinates.shape
        # if self.out.numel() > 0 and self.learnable_features == False:
        #    return self.out.repeat(b, 1, 1, 1)
        origin_dtype = coordinates.dtype
        prefeatures = torch.einsum('oi,bihw->bohw', self.frequency_matrix.to(coordinates.device), coordinates)
        prefeatures = prefeatures.permute(0, 2, 3, 1).reshape((b, h * w, -1))
        prefeatures = prefeatures[0].unsqueeze(0)
        prefeatures = self.norm(prefeatures.permute(0, 2, 1)).permute(0, 2, 1)
        out = torch.empty([1, h * w, self.out_channels]).to(prefeatures.device, dtype=prefeatures.dtype)

        for idx in range(self.out_channels):
            layer = getattr(self, f'qlayer{idx}')
            start = idx * self.qlayer_features
            end = (idx + 1) * self.qlayer_features
            out[:, :, idx] = layer(prefeatures[:, :, start:end],
                                   self.quantum_circuit_param[:, :, :, start:end, :])

        out = out.permute(0, 2, 1).reshape(1, -1, h, w).to(dtype=origin_dtype)
        # self.out = out
        return out.repeat(b, 1, 1, 1)
