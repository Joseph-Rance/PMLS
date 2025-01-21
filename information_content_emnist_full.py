"""MNIST classification."""

from math import ceil
import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import EMNIST

from cnn import CNN

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
TRAINING_EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

transform = transforms.Compose([
#    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = EMNIST("./data", split="balanced", train=True, transform=transform, download=True)
# skip val for now
test_dataset = EMNIST("./data", split="balanced", train=False, transform=transform, download=True)

# no need for dataloaders - we can just load the full dataset into GPU memory!
# usually dataloaders shuffle on every epoch which we are going to lose here but that doesn't really matter
# we can't augment the data every easily this way, but that also doesn't matter much

inputs = []
labels = []

for x, y in train_dataset:
    inputs.append(x)
    labels.append(y)

for x, y in test_dataset:
    inputs.append(x)
    labels.append(y)

inputs = torch.cat(inputs, dim=0).reshape((-1, 1, 28, 28))[:70_000]
labels = torch.tensor(labels)[:70_000]

model = CNN(num_outputs=47).to(DEVICE)
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

FINETUNE_STEP = 20
# MUTLIPLIER ^ FINETUNE_STEP - 70000 * MUTLIPLIER + 70000 - 1 = 0
MUTLIPLIER = 1.71817
finetune_sizes = [0] + [round(MUTLIPLIER**i) for i in range(FINETUNE_STEP)]
finetune_sizes[-1] = 70_000

@torch.no_grad()
def test(inputs, labels, i):

    model.eval()

    data_start, data_end = finetune_sizes[i], finetune_sizes[i+1]

    x = inputs[data_start:data_end]
    y = labels[data_start:data_end]

    z = model(x)
    loss = cross_entropy(z, y, reduction="sum")

    num_examples = x.size()[0]
    #assert num_examples == data_end - data_start

    top = z.topk(5, 1, sorted=True).indices
    top1 = (top[:, 0] == y).sum().item()
    top5 = (top == y.view(-1, 1)).sum().item()

    return loss, (top1, top5), num_examples


def train(inputs, labels, i):

    data_end = finetune_sizes[i]

    optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=TRAINING_EPOCHS*ceil(data_end/BATCH_SIZE))

    lrs, loss, top1, top5, norms = [], [], [], [], []

    model.train()

    for _epoch in trange(TRAINING_EPOCHS, leave=False):

        norm_epoch = loss_epoch = top1_epoch = top5_epoch = num_examples = 0
        train_loader_bar = trange(ceil(data_end / BATCH_SIZE), leave=False, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]",
                                                               postfix="lr: ?, loss: ?, acc@1: ?%")
        for idx in train_loader_bar:

            x = inputs[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]
            y = labels[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]

            # TODO: does this help?
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_batch = cross_entropy(z, y)

            loss_batch.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()

            with torch.no_grad():

                loss_epoch += loss_batch.item()
                norm_epoch += norm.cpu()

                top = z.topk(5, 1, sorted=True).indices
                top1_epoch += (top[:, 0] == y).sum().item()
                top5_epoch += (top == y.view(-1, 1)).sum().item()

                num_examples += x.size()[0]

                train_loader_bar.set_description_str(f"lr: .{int(optimiser.param_groups[0]['lr']*1000):0>3}, loss: {loss_batch.item():2.2f}, " \
                                                    f"acc@1: {top1_epoch * 100 / max(1, num_examples):3.1f}%")
                train_loader_bar.refresh()

        lrs.append(optimiser.param_groups[0]['lr'])
        loss.append(loss_epoch)
        top1.append(top1_epoch)
        top5.append(top5_epoch)
        norms.append(norm_epoch)
    
    return (lrs, loss, top1, top5, norms, num_examples)

if __name__ == "__main__":

    print("[training]")

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    losses = []
    full_res = []

    for i in range(FINETUNE_STEP):

        # since dataset lives on GPU we need to index it based on cg and bg rather than keeping index lists

        train_res = train(inputs, labels, i)

        loss, (top1, top5), num_examples = test(inputs, labels, i)

        full_res.append((train_res, (loss.cpu(), (top1, top5), num_examples)))
        losses.append(loss.cpu())
        np.save(f"outs/information_content_metrics.npy", np.array(full_res, dtype=object), allow_pickle=True)
        np.save(f"outs/losses.npy", losses)
        #torch.save(model.state_dict(), f"outs/models/model_{cg}_{bg}.pt")