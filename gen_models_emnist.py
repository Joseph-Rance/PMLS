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

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = EMNIST("./data", split="balanced", train=True, transform=transform, download=True)

CLASS_STEP = 3

perm = np.random.permutation(47)  # to shuffle classes

inputs = [[] for __ in range(ceil(47/CLASS_STEP))]
labels = [[] for __ in range(ceil(47/CLASS_STEP))] 

for x, y in train_dataset:
    inputs[perm[y]//CLASS_STEP].append(x)
    labels[perm[y]//CLASS_STEP].append(y)

# note that can't use all 47 classes because we dont want last epoch to have less than CLASS_STEP classes
# TODO: change if CLASS_STEP is changed
inputs = inputs[:-1]
labels = labels[:-1]

lengths = [len(i) for i in inputs]

inputs = torch.cat([torch.cat(i, dim=0) for i in inputs]).reshape((-1, 1, 28, 28))
labels = torch.cat([torch.tensor(i) for i in labels])

model = CNN(num_outputs=47).to(DEVICE)  # works better than the pytorch 32x4d version
model = torch.compile(model)
torch.backends.cudnn.benchmark = True


def train(classes):

    data_end = sum(lengths[:classes+1])

    optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS*ceil(data_end/BATCH_SIZE))

    for _epoch in trange(MAX_EPOCHS, leave=False):

        model.train()

        for idx in trange(ceil(data_end/BATCH_SIZE), leave=False):

            x = inputs[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]
            y = labels[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]

            # TODO: is there any point in this?
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_train_batch = cross_entropy(z, y)

            loss_train_batch.backward()
            _norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()

'''
# since classes are unbalanced in MNIST, actually split up training sets rather than
# getting full accuracy every time as in CIFAR-100 implementation
@torch.no_grad()
def test(classes):

    data_end = sum(test_lengths[:classes+1])

    model.eval()
    loss = top1 = top5 = num_examples = 0

    for idx in range(ceil(data_end / BATCH_SIZE)):

        x = test_inputs[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]
        y = test_labels[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]

        z = model(x)
        loss += cross_entropy(z, y)

        num_examples += x.size()[0]

        top = z.topk(5, 1, sorted=True).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y.view(-1, 1)).sum().item()

    return loss, (top1, top5), num_examples
'''

if __name__ == "__main__":

    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    for classes in trange(10):
        train(classes)
        torch.save(model.state_dict(), f"outs_emnist/model_{classes}_forward.pt")

    # reset model
    model = CNN(num_outputs=47).to(DEVICE)
    model = torch.compile(model)

    for classes in tqdm(range(9, -1, -1)):
        train(classes)
        torch.save(model.state_dict(), f"outs_emnist/model_{classes}_backward.pt")