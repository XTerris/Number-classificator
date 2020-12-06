import torch
import torch.nn as nn
import torchvision
import numpy as np

mnist = torchvision.datasets.MNIST(root=".data/mnist")

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

model.load_state_dict(torch.load("model"))

n = int(input("Count of pictures: "))

if __name__ == "__main__":
    result = 0
    indicies = np.random.choice(len(mnist), n)
    for i in indicies:
        x = torch.Tensor(np.asarray(mnist[i][0]) / 255).reshape(-1, 1, 28, 28)
        y = mnist[i][1]
        if torch.argmax(model(x)).item() == y:
            result += 1

    print(f"Accuracy: {int(result / max(n, 1) * 100)}%.")

input()