"""CIFAR-100 classification."""

import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import CIFAR100

from resnext import resnext50

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 200

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_val_dataset = CIFAR100("./data", train=True, transform=train_transform, download=True)
test_dataset = CIFAR100("./data", train=False, transform=test_transform, download=True)

train_dataset = train_val_dataset

CLASS_STEP = 10  # we retrain every CLASS_STEP classes
classes = [[] for __ in range(100//CLASS_STEP)]

for i, (_x, y) in enumerate(train_dataset):
    classes[y//CLASS_STEP].append(i)

train_loaders = []

acc_idxs = []
for i, c in enumerate(classes):
    acc_idxs += c
    dataset = Subset(train_dataset, [i for i in acc_idxs])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True)
    train_loaders.append((str(min(100, CLASS_STEP*(i+1))), loader))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
val_loader = test_loader

model = resnext50().to(DEVICE)  # works better than the pytorch 32x4d version
model = torch.compile(model)
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def test(loader, tqdm_subst=lambda x: x):

    model.eval()
    loss = top1 = top5 = num_examples = 0

    for x, y in tqdm_subst(loader):

        x, y = x.to(DEVICE), y.to(DEVICE)

        z = model(x)
        loss += cross_entropy(z, y)

        num_examples += x.size()[0]

        top = z.topk(5, 1, sorted=True).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y.view(-1, 1)).sum().item()

    return loss, (top1, top5), num_examples


def train(loader):

    optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(optimiser, factor=0.2, patience=5, threshold=0.25, threshold_mode="rel")

    _loss_val, (top1_val, _top5_val), num_val_examples = test(val_loader)

    for epoch in trange(MAX_EPOCHS, leave=False):

        model.train()
        loss_train = top1_train = top5_train = num_train_examples = norm_epoch = 0

        train_loader_bar = tqdm(loader, leave=False, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]",
                                                           postfix="lr: ?, loss: ?, train@1: ?%, val@1*: ?%")
        for x, y in train_loader_bar:

            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_train_batch = cross_entropy(z, y)

            loss_train_batch.backward()
            _norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()

            with torch.no_grad():

                loss_train += loss_train_batch.item()

                top = z.topk(5, 1, sorted=True).indices
                top1_train += (top[:, 0] == y).sum().item()

                num_train_examples += x.size()[0]

                train_loader_bar.set_description_str(f"lr: .{int(optimiser.param_groups[0]['lr']*1000):0>3}, loss: {loss_train_batch.item():2.2f}, " \
                                                     f"train@1: {top1_train * 100 / max(1, num_train_examples):3.1f}%, " \
                                                     f"val@1*: {top1_val * 100 / max(1, num_val_examples):3.1f}%")
                train_loader_bar.refresh()

        _loss_val, (top1_val, _top5_val), num_val_examples = test(val_loader)

        scheduler.step(loss_train)
        if optimiser.param_groups[0]['lr'] < 0.0005:
            break  # quit when we have converged


if __name__ == "__main__":

    for name, loader in tqdm(train_loaders):
        train(loader)
        torch.save(model.state_dict(), f"outs/model_{name}_forward.pt")

    print(f"before reset: {test(val_loader)}")

    # reset model
    model = resnext50().to(DEVICE)
    model = torch.compile(model)

    print(f"after reset: {test(val_loader)}")

    for name, loader in tqdm(train_loaders[::-1]):
        train(loader)
        torch.save(model.state_dict(), f"outs/model_{name}_backward.pt")