"""Test performance of CIFAR-100 models at different pruning levels"""

from copy import deepcopy
import random
from time import perf_counter
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
import torch.nn.utils.prune as prune
from torchvision import transforms
from torchvision.datasets import CIFAR100

from resnext import resnext50

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
FINETUNE_EPOCHS = 2

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

train_dataset = CIFAR100("./data", train=True, transform=train_transform, download=True)
test_dataset = CIFAR100("./data", train=False, transform=test_transform, download=True)

CLASS_STEP = 10  # new model every CLASS_STEP classes
classes = [[] for __ in range(100//CLASS_STEP)]
test_classes = [[] for __ in range(100//CLASS_STEP)]

for i, (_x, y) in enumerate(train_dataset):
    classes[y//CLASS_STEP].append(i)

for i, (_x, y) in enumerate(test_dataset):
    test_classes[y//CLASS_STEP].append(i)

train_loaders = []
test_loaders = []

acc_idxs = []
for i, c in enumerate(classes):
    acc_idxs += c
    dataset = Subset(train_dataset, [i for i in acc_idxs])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True)
    train_loaders.append((str(min(100, CLASS_STEP*(i+1))), loader))

test_acc_idxs = []
for i, c in enumerate(test_classes):
    test_acc_idxs += c
    dataset = Subset(test_dataset, [i for i in test_acc_idxs])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    test_loaders.append(loader)

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def test(model, loader):

    model.eval()
    _loss = top1 = _top5 = num_examples = 0

    for x, y in loader:

        x, y = x.to(DEVICE), y.to(DEVICE)

        z = model(x)
        #_loss += cross_entropy(z, y)

        num_examples += x.size()[0]

        top = z.topk(5, 1, sorted=True).indices
        top1 += (top[:, 0] == y).sum().item()
        #_top5 += (top == y.view(-1, 1)).sum().item()

    return top1 / num_examples

def finetune(model, loader, reg=5e-4):

    optimiser = SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=reg)
    scheduler = CosineAnnealingLR(optimiser, T_max=FINETUNE_EPOCHS*len(loader))

    for _epoch in trange(FINETUNE_EPOCHS, leave=False):

        model.train()

        for x, y in tqdm(loader, leave=False):

            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss = cross_entropy(z, y)

            loss.backward()
            _norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()

if __name__ == "__main__":

    outs = []

    data_splits = [(i+1)*10 for i in range(10)]

    # load pretrained models

    models = [
        (f"outs/model_{split}_{dir}.pt", resnext50())
            for split in data_splits for dir in ["forward", "backward"]
    ]

    for path, model in models:
        model.load_state_dict({k[10:]:v for k,v in torch.load(path, weights_only=True).items() if k.startswith("_orig_mod.")})

    # iterate over models (i.e. forward and backward for each number of classes)
    for i, (split, (__, model)) in enumerate(zip([j for i in data_splits for j in [i, i]], models)):
        print(f"\n== {split} CLASSES ==")

        model.to(DEVICE)

        PRUNE_AMOUNTS = [0.625, 0.650, 0.675]  # proportion of weights to prune

        for j in range(len(PRUNE_AMOUNTS)):

            # train the model for FINETUNE_EPOCHS with high regularisation to encourage sparsity

            initial_acc = test(model, test_loaders[i//2])

            finetune(model, train_loaders[i//2][1], reg=5e-3)

            sparse_acc = test(model, test_loaders[i//2])

            # prune the largest layers in the model an extra PRUNE_AMOUNT (ontop of what it is already pruned to)

            start_time = perf_counter()

            for k in range(6):
                for l in [0, 3, 6]:
                    prune.l1_unstructured(list(list(model.conv4)[k].split_transforms)[l], name="weight", amount=float(PRUNE_AMOUNTS[j]))
                    #prune.remove(list(list(model.conv4)[k].split_transforms)[l], 'weight')

            for k in range(3):
                for l in [0, 3, 6]:
                    prune.l1_unstructured(list(list(model.conv5)[k].split_transforms)[l], name="weight", amount=float(PRUNE_AMOUNTS[j]))
                    #prune.remove(list(list(model.conv5)[k].split_transforms)[l], 'weight')

            end_time = perf_counter()
            total_time = end_time - start_time

            prune_acc = test(model, test_loaders[i//2])

            # finetune again to clear up problems from pruning

            finetune(model, train_loaders[i//2][1], reg=5e-4)

            finetune_acc = test(model, test_loaders[i//2])

            print(f"{split} | accuracy: {initial_acc:.3f} -> {sparse_acc:.3f} -> {prune_acc:.3f} -> {finetune_acc:.3f}; prune fraction: {(j+1)*PRUNE_AMOUNT:.3f}; prune time: {total_time:.3f}")
            outs.append((initial_acc, sparse_acc, prune_acc, finetune_acc))
            np.save("outs/res.npy", outs)