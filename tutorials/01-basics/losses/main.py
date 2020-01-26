import torch
from torch import nn

# set seed to preserve reproducibility
torch.manual_seed(0)

# L1 loss, please read the official doc: https://pytorch.org/docs/stable/nn.html#l1loss

# L1 loss is calculated as the mean of sum of absolute difference
l1loss = nn.L1Loss(reduction='mean')
x = torch.tensor([[1, 2], [3, 4]]).float()
y = torch.tensor([[0, 2], [1, 3]]).float()

# the expected output is (1+ 0 + 2 + 1) / 4 = 1.0

loss_output = l1loss(x, y)
print("the L1 loss is: ", loss_output.item())

# cross entropy loss
cross_entropy_loss = nn.CrossEntropyLoss()

# what is logits anyway?
#  Logit is a function that maps probabilities [0, 1] to [-inf, +inf].
# in pytorch or tensorflow, logit number is used to refer to unnormized number in [-inf, +inf]
batch_size = 4
num_classes = 3
logits = torch.randn(batch_size, num_classes)
print("unnormized logits: \n", logits)
# targets length must be same as batch_size
targets = torch.tensor([0, 0, 1, 2])
print("targets class indece: ", targets)
loss = cross_entropy_loss(input=logits, target=targets)
print("cross entropy loss: ", loss.item())

# manual verification
# step 1: apply log softmax to normize logits to [0, 1]
log_softmax_func = nn.LogSoftmax(dim=1)
log_softmax = log_softmax_func(input=logits)
print("log softmax: \n", log_softmax)
# step 2: apply negative log likelihood function with reduction "none"
# as you can see, with the targets [1, 0 , 2, 2], it is used as index to find the output candidate
negative_log_likelihood = nn.NLLLoss(reduction="none")
out = negative_log_likelihood(log_softmax, targets)
print("cross entropy loss without mean over batch size: ", out)
print("cross entropy loss with mean over batch size (default): ", out.mean().item())
