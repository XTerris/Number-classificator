import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST(root=".data/mnist")

samples = []
losses = []
epochs = []

n = 40000

for i in range(n):
    print(f"Progress of preparing: {int(i / (n - 1) * 100)}%.    ", end="\r")
    x = torch.Tensor(np.asarray(mnist[i][0]) / 255).reshape(-1, 1, 28, 28)
    y = torch.zeros(10).reshape(-1, 10)
    y[0][mnist[i][1]] = 1
    samples.append([x, y])

print()

model = nn.Sequential(
    nn.Conv2d(1, 6, 3),
    nn.ReLU(),
    nn.Conv2d(6, 8, 4),
    nn.ReLU(),
    nn.Conv2d(8, 2, 2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(968, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.ReLU()
)

# model.load_state_dict(torch.load("model"))

loss_fn = nn.MSELoss()
optim = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

e = 2
for n in range(e):
    i = 0
    c = 0
    last_losses = 0
    for x, y in samples:
        optim.zero_grad()
        out = model(x)
        loss = loss_fn(y, out)
        loss.backward()
        optim.step()
        print(f"Progress of training: {int(i / (len(samples) - 1) * 100)}%. Loss: {loss}   ", end="\r")
        if c == 49:
            losses.append(last_losses / (c + 1))
            last_losses = 0
            c = 0
        i += 1
        c += 1
        last_losses += loss
    torch.save(model.state_dict(), "model")
    print("\nModel saved to /model")

plt.plot(np.linspace(0, e * n * 50, len(losses)) + 1, losses, "-b")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.show()