import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ClassNet(nn.Module):
    def __init__(self, size=1):
        super().__init__()

        self.fc1 = nn.Linear(size, 16 * size)
        self.fc2 = nn.Linear(16 * size, size)
        # self.fc3 = nn.Linear(4 * size, 2 * size)
        # self.fc4 = nn.Linear(2 * size, size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        # out = self.fc4(self.relu(self.fc3(x)))
        return x


class QuantumNet(nn.Module):
    def __init__(self, n_qubits=1):
        super().__init__()
        self.layer = 6

        def _circuit(inputs, weights1):
            for k in range(self.layer):
                qml.Rot(weights1[k, 0], weights1[k, 1], weights1[k, 2], wires=0)
                qml.RX(inputs[0], wires=0)
            qml.Rot(weights1[-1, 0], weights1[-1, 1], weights1[-1, 2], wires=0)

            return qml.expval(qml.PauliZ(0))

        # weight_shape = {"weights1": (2, 2, 3), "weights2": (3, 2 * seq_len, 3), "weights3": (3, 2 * seq_len, 3)}
        weight_shape = {"weights1": (self.layer + 1, 3)}
        torch_device = qml.device('default.qubit', wires=n_qubits)
        self.qlayer = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")
        self.qnn = qml.qnn.TorchLayer(self.qlayer, weight_shape)

    def forward(self, x):
        out = self.qnn(x)
        return out.unsqueeze(1)


def target_fun(x):
    return 3 * torch.sin(x) + torch.sin(5 * x)


LR = 0.3
EPOCHS = 50
model = QuantumNet()
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
x = torch.from_numpy(np.linspace(-3.14, 3.14, 100, dtype=np.float32))
for epoch in range(EPOCHS):
    batch_index = np.random.randint(0, len(x), (20,))
    x_batch = x[batch_index]
    x_batch = x_batch.unsqueeze(1)
    predict = model(x_batch)
    y_batch = target_fun(x_batch)
    loss = mseloss(predict, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(loss.data.item())
plt.plot(x, target_fun(x), c='black')
plt.scatter(x, target_fun(x), facecolor='white', edgecolor='black')
plt.plot(x, model(x.unsqueeze(1)).detach().numpy(), c='blue')
plt.show()
