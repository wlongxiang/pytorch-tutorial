import torch
from torch import nn

# L1 loss, please read the official doc: https://pytorch.org/docs/stable/nn.html#l1loss

# L1 loss is calculated as the mean of sum of absolute difference
l1loss = nn.L1Loss(reduction='mean')
x = torch.tensor([[1, 2], [3, 4]]).float()
y = torch.tensor([[0, 2], [1, 3]]).float()

# the expected output is (1+ 0 + 2 + 1) / 4 = 1.0

loss_output = l1loss(x, y)
print("the L1 loss is: ", loss_output.item())
