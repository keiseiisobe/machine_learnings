'''
estimate linear function (y = 2x + 10) using linear regression.
'''

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train(input, label, model, loss_fn, optimizer):
    model.train()
    pred = model(input)
    loss = loss_fn(pred, label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"loss: {loss.item()}")

def test(input, label, model, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(input)
        loss = loss_fn(pred, label)
        print(f"average loss: {loss.item()}")
    return pred

train_input = np.arange(10, dtype=np.float32) #[0, 1, 2, ..., 9]
train_input = train_input.reshape(-1, 1)
train_label = 2 * train_input + 10
print(f"train input: {train_input}")
print(f"train label: {train_label}")
train_input = torch.from_numpy(train_input)
train_label = torch.from_numpy(train_label)

test_input = np.arange(10, 20, dtype=np.float32) #[10, 11, 12, ..., 19]
test_input = test_input.reshape(-1, 1)
test_label = 2 * test_input + 10
print(f"test input: {test_input}")
print(f"test label: {test_label}")
test_input = torch.from_numpy(test_input)
test_label = torch.from_numpy(test_label)

device = "cpu"
model = LinearRegression().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

epochs = 500
for epoch in range(epochs):
    print(f"Epoch {epoch+1}{'-'*50}")
    train(train_input, train_label, model, loss_fn, optimizer)
    pred = test(test_input, test_label, model, loss_fn)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(test_input, pred, label="pred", color="blue")
ax2.plot(test_input, test_label, label="true", color="orange")
ax1.set_xlim(xmin=0)
ax2.set_xlim(xmin=0)
ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0)
plt.show()
