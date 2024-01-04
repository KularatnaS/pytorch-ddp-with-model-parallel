import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchnet.dataset import SplitDataset
from torch.utils.data import Dataset

import smdistributed.modelparallel.torch as smp


class DummyDataset(Dataset):
    def __init__(self, size=40):
        self.data = torch.ones(size, 16) * torch.arange(size).unsqueeze(1)
        self.target = torch.randn(size, 5)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


class GroupedNet(nn.Module):
    def __init__(self):
        super(GroupedNet, self).__init__()
        # define layers
        self.net1 = torch.nn.Linear(16, 16)
        self.net2 = torch.nn.Linear(16, 16)
        self.net3 = torch.nn.Linear(16, 16)
        self.net4 = torch.nn.Linear(16, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # define forward pass and return model outputs
        x = self.relu(self.net1(x))
        x = self.relu(self.net2(x))
        x = self.relu(self.net3(x))
        x = self.net4(x)
        return x


# smdistributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, target):
    output = model(data)
    loss = F.mse_loss(output, target, reduction="mean")
    model.backward(loss)
    return output, loss


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # smdistributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        print(f"Step: {batch_idx}, RDP Rank {smp.rdp_rank()}, training batch: {data}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Return value, loss_mb is a StepOutput object
        _, loss_mb = train_step(model, data, target)

        # smdistributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()

        optimizer.step()

# smdistributed: initialize the backend
smp.init()

# smdistributed: Set the device to the GPU ID used by the current process.
# Input tensors should be transferred to this device.
torch.cuda.set_device(smp.local_rank())
device = torch.device("cuda")

# smdistributed: Download only on a single process per instance.
# When this is not present, the file is corrupted by multiple processes trying
# to download and extract at the same time
dataset = DummyDataset()

# smdistributed: Shard the dataset based on data-parallel ranks
if smp.rdp_size() > 1:
    print(f"Sharding dataset for RDP size {smp.rdp_size()}")
    partitions_dict = {f"{i}": 1 / smp.rdp_size() for i in range(smp.rdp_size())}
    dataset = SplitDataset(dataset, partitions=partitions_dict)
    dataset.select(f"{smp.rdp_rank()}")

# smdistributed: Set drop_last=True to ensure that batch size is always divisible
# by the number of microbatches
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=True)

model = GroupedNet()
optimizer = optim.Adadelta(model.parameters(), lr=4.0)

# smdistributed: Use the DistributedModel container to provide the model
# to be partitioned across different ranks. For the rest of the script,
# the returned DistributedModel object should be used in place of
# the model provided for DistributedModel class instantiation.
model = smp.DistributedModel(model)
optimizer = smp.DistributedOptimizer(optimizer)

train(model, device, train_loader, optimizer)