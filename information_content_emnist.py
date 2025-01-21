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

SEED = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
TRAINING_EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = EMNIST("./data", split="balanced", train=True, transform=transform, download=True)
test_dataset = EMNIST("./data", split="balanced", train=False, transform=transform, download=True)

CLASS_STEP = 3

perm = np.random.permutation(47)  # to shuffle classes

inputs = [[] for __ in range(ceil(47/CLASS_STEP))]
labels = [[] for __ in range(ceil(47/CLASS_STEP))] 

for x, y in train_dataset:
    inputs[perm[y]//CLASS_STEP].append(x)
    labels[perm[y]//CLASS_STEP].append(y)

for x, y in test_dataset:
    inputs[perm[y]//CLASS_STEP].append(x)
    labels[perm[y]//CLASS_STEP].append(y)

# note that can't use all 47 classes because we dont want last epoch to have less than CLASS_STEP classes
# TODO: change if CLASS_STEP is changed
inputs = inputs[:-1]
labels = labels[:-1]

lengths = [len(i) for i in inputs]

inputs = torch.cat([torch.cat(i, dim=0) for i in inputs]).reshape((126_000, 1, 28, 28))
labels = torch.cat([torch.tensor(i) for i in labels])

model = CNN(num_outputs=47).to(DEVICE)  # works better than the pytorch 32x4d version
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

FINETUNE_STEP = 10
# MUTLIPLIER ^ FINETUNE_STEP - 2800 * CLASS_STEP * MUTLIPLIER + 2800 * CLASS_STEP - 1 = 0
MUTLIPLIER = 2.58481

def get_batch_idxs(cg, bg):

    num_pretrain = sum(lengths[:cg])

    finetune_sizes = [round(MUTLIPLIER**i) for i in range(bg+1)]

    start = num_pretrain + sum(finetune_sizes[:-1])
    length = min(finetune_sizes[-1], lengths[cg])

    return start, start+length

def train(inputs, labels, cg, bg):

    data_end, __ = get_batch_idxs(cg, bg)

    # we need all the data to be shuffled
    # this could be slightly quicker if we avoided re-shuffling the already shuffled data
    # but that makes a really minor difference
    perm = torch.randperm(data_end)
    # TODO: make sure this doesn't do anything weird with gradients
    inputs[:data_end] = inputs[perm]
    labels[:data_end] = labels[perm]

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

@torch.no_grad()
def test(inputs, labels, cg, bg):

    model.eval()

    data_start, data_end = get_batch_idxs(cg, bg)

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


if __name__ == "__main__":

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    print("[training]")

    losses = []
    full_res = []

    pretrain_data = []

    # since dataset lives on GPU we need to index it based on cg and bg rather than keeping index lists
    for cg in trange(len(lengths)):

        for bg in trange(FINETUNE_STEP, leave=False):

            train_res = train(inputs, labels, cg, bg)

            loss, (top1, top5), num_examples = test(inputs, labels, cg, bg)

            full_res.append((train_res, (loss.cpu(), (top1, top5), num_examples)))
            losses.append(loss.cpu())
            np.save(f"outs/information_content_metrics.npy", np.array(full_res, dtype=object), allow_pickle=True)
            np.save(f"outs/losses.npy", losses)
            #torch.save(model.state_dict(), f"outs/models/model_{cg}_{bg}.pt")