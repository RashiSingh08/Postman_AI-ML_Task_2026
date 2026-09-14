### Postman_AI-ML_Task_2026
This is my submission for Postman club AI/ML vertical. I have done Experiment 3 here. 

I chose it as it only required Numpy and PyTorch and the topic too was fun.
I learnt PyTorch yesterday at 1AM. It was a mistake as I heavily underestimated how it would be compared to Numpy or Pandas...

# AI Usage
AI usage was minimal, especially in 3.1 and 3.4. Done only to create boilerplate and to help with debugging.
it was a bit more prominently used in 3.3, but I understand the code well...
I have attached my AI chat for vetting, just in case.
https://gemini.google.com/share/d/1fpBY_jiP19KnZym1ETEjw0bBynC7hvtr?usp=sharing

# About the Project
I chose 3.1, 3.3, 3.4 and 3.6 as they were the ones manageable under a short time frame.
I tested the code using AI generated code, yes but the code I used for that is attached here and was the only Ai generated code in this.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x.var(dim=0, unbiased=True)
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_hat + self.beta


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = CustomBatchNorm1d(num_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten 28x28 images into 784-dim vectors
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 1. Training Pass
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx == 5:
            print(f"Train Step {batch_idx} Loss: {loss.item():.4f}")
            print("Tracked Running Mean (First 5 features):", model.bn1.running_mean[:5])
            break

    # 2. Evaluation Pass
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            accuracy = (preds == targets).float().mean()
            print(f"Eval Batch Accuracy: {accuracy.item() * 100:.2f}%")
            break
