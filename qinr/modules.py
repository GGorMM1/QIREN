import pennylane as qml
import torch.nn as nn
import torch
import math
import numpy as np
import qiskit.providers.aer.noise as noise


class FourierFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, learnable_features=False):
        super(FourierFeatures, self).__init__()
        frequency_matrix = torch.normal(mean=torch.zeros(out_channels, in_channels),
                                        std=1.0)
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        prefeatures = torch.einsum('oi,bli->blo', self.frequency_matrix.to(coordinates.device), coordinates)
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        return torch.cat((cos_features, sin_features), dim=2)


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3),
                        "weights2": (2, in_features, 3)}
        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        out = self.qnn(x)
        return out.reshape(orgin_shape)


class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)

    def forward(self, x):
        x1 = self.clayer(x)
        x1 = self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.qlayer(x1)
        return out


class Hybridren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, spectrum_layer, use_noise,
                 outermost_linear=True):
        super().__init__()

        self.net = []
        self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1))

        for i in range(hidden_layers):
            self.net.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
        else:
            final_linear = HybridLayer(hidden_features, out_features, spectrum_layer, use_noise)

        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


class FFNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, res=True):
        super().__init__()
        self.res = res
        self.clayer1 = nn.Linear(in_features, 2 * in_features, bias=bias)
        self.norm = nn.BatchNorm1d(2 * in_features)
        self.activ = nn.ReLU()
        self.clayer2 = nn.Linear(2 * in_features, out_features, bias=bias)

    def forward(self, x):
        x1 = self.clayer1(x)
        x1 = self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.clayer2(self.activ(x1))
        if self.res:
            out = out + x
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        self.net = []
        self.net.append(FFNLayer(in_features, hidden_features, res=False))

        for i in range(hidden_layers):
            self.net.append(FFNLayer(hidden_features, hidden_features))

        self.net.append(FFNLayer(hidden_features, out_features, res=False))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


class SineLayer_bn(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, activ='relu', omega_0=30):
        super().__init__()

        self.is_first = is_first
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'sine':
            self.activ = torch.sin

    def forward(self, input):
        x1 = self.linear(input)
        x1 = self.omega_0 * self.norm(x1.permute(0, 2, 1)).permute(0, 2, 1)
        return self.activ(x1)


class Siren_bn(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, activ='relu',
                 first_omega_0=30, hidden_omega_0=30, rff=False):
        super().__init__()

        self.net = []
        if rff:
            self.net.append(FourierFeatures(in_features, hidden_features // 2))
        else:
            self.net.append(
                SineLayer_bn(in_features, hidden_features, is_first=True, activ=activ, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer_bn(hidden_features, hidden_features, is_first=False, activ=activ, omega_0=hidden_omega_0))

        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(SineLayer_bn(hidden_features, out_features, is_first=False, activ=activ))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, idx=0):
        super().__init__()
        self.idx = idx
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = self.omega_0 * self.linear(input)
        return torch.sin(out)

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, idx=1))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
