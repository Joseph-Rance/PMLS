"""Test performance of EMNIST models at different pruning levels"""

from math import ceil
import random
from time import perf_counter
from tqdm import trange
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy
import torch.nn.utils.prune as prune
from torchvision import transforms
from torchvision.datasets import EMNIST

from cnn import CNN

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

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

#perm = np.random.permutation(47)  # to shuffle classes
perm = list(range(47))

train_inputs = [[] for __ in range(ceil(47/CLASS_STEP))]
train_labels = [[] for __ in range(ceil(47/CLASS_STEP))] 
test_inputs = [[] for __ in range(ceil(47/CLASS_STEP))]
test_labels = [[] for __ in range(ceil(47/CLASS_STEP))]

for x, y in train_dataset:
    train_inputs[perm[y]//CLASS_STEP].append(x)
    train_labels[perm[y]//CLASS_STEP].append(y)

for x, y in test_dataset:
    test_inputs[perm[y]//CLASS_STEP].append(x)
    test_labels[perm[y]//CLASS_STEP].append(y)

# note that can't use all 47 classes because we dont want last epoch to have fewer than CLASS_STEP classes
# TODO: change this if CLASS_STEP is changed!
train_inputs = train_inputs[:-1]
train_labels = train_labels[:-1]
test_inputs = test_inputs[:-1]
test_labels = test_labels[:-1]

train_lengths = [min(1000, len(i)) for i in train_inputs]
test_lengths = [min(1000, len(i)) for i in test_inputs]

train_inputs = torch.cat([torch.cat(i[:1000], dim=0) for i in train_inputs]).reshape((-1, 1, 28, 28))
train_labels = torch.cat([torch.tensor(i[:1000]) for i in train_labels])

test_inputs = torch.cat([torch.cat(i[:1000], dim=0) for i in test_inputs]).reshape((-1, 1, 28, 28))
test_labels = torch.cat([torch.tensor(i[:1000]) for i in test_labels])

# now inputs holds all the inputs for class 0, then all the inputs for class 1, and so on.
# since we want to access all classes up to some class, t, we can just train on slices of
# inputs and labels

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def test(model, inputs, labels, i):

    model.eval()

    data_start, data_end = 0, sum(test_lengths[:i+1])

    x = inputs[data_start:data_end]
    y = labels[data_start:data_end]

    z = model(x)
    _loss = cross_entropy(z, y, reduction="sum")

    num_examples = x.size()[0]
    #assert num_examples == data_end - data_start

    top = z.topk(5, 1, sorted=True).indices
    top1 = (top[:, 0] == y).sum().item()
    _top5 = (top == y.view(-1, 1)).sum().item()

    #return loss, (top1, top5), num_examples
    return top1 / num_examples

def finetune(model, inputs, labels, i, reg=5e-3, initial_lr=0.004, epochs=2):

    data_end = sum(train_lengths[:i+1])

    perm = np.random.permutation(data_end)

    inputs[0 : data_end] = inputs[perm]
    labels[0 : data_end] = labels[perm]

    optimiser = SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=reg)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs*ceil(data_end/BATCH_SIZE))

    for _epoch in trange(epochs, leave=False):

        model.train()

        for idx in trange(ceil(data_end / BATCH_SIZE), leave=False):

            x = inputs[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]
            y = labels[idx*BATCH_SIZE : min(data_end, (idx+1)*BATCH_SIZE)]

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

    data_splits = [(i+1)*10 for i in range(10, 15)]

    models = [
        (f"outs_emnist/model_{int(split/10-1)}_{dir}.pt", CNN(num_outputs=47))
            for split in data_splits for dir in ["forward"]
    ]

    # uncomment to use pretrained models (not necessary when training takes such little time)
    #for path, model in models:
    #    model.load_state_dict({k[10:]:v for k,v in torch.load(path, weights_only=True).items() if k.startswith("_orig_mod.")})

    train_inputs = train_inputs.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_inputs = test_inputs.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    for k, (split, (__, model)) in zip(range(10, 15), zip(data_splits, models)):

        print(f"\n== {int(split/10-1)} CLASSES ==")

        model.to(DEVICE)

        finetune(model, train_inputs, train_labels, int(split/10-1), reg=5e-3, initial_lr=0.1, epochs=4)

        PRUNE_AMOUNT = 0.1  # we prune for an extra 10%, ...
        PRUNE_STEPS = 10  # ... 10 times

        for j in range(PRUNE_STEPS):

            # train the model for FINETUNE_EPOCHS with high regularisation to encourage sparsity

            initial_acc = test(model, test_inputs, test_labels, int(split/10-1))

            finetune(model, train_inputs, train_labels, int(split/10-1), reg=5e-3)

            sparse_acc = test(model, test_inputs, test_labels, int(split/10-1))

            # prune the model an extra PRUNE_AMOUNT

            start_time = perf_counter()

            prune.l1_unstructured(model.fc1, name="weight", amount=float((j+1)*PRUNE_AMOUNT))
            prune.l1_unstructured(model.fc2, name="weight", amount=float((j+1)*PRUNE_AMOUNT))

            end_time = perf_counter()
            total_time = end_time - start_time

            prune_acc = test(model, test_inputs, test_labels, int(split/10-1))

            # finetune again to clear up problems from pruning

            finetune(model, train_inputs, train_labels, int(split/10-1), reg=5e-4)

            finetune_acc = test(model, test_inputs, test_labels, int(split/10-1))

            print(f"{int(split/10-1)} | accuracy: {initial_acc:.3f} -> {sparse_acc:.3f} -> {prune_acc:.3f} -> {finetune_acc:.3f}; prune fraction: {float((j+1)*PRUNE_AMOUNT):.3f}; prune time: {total_time}")
            outs.append((initial_acc, sparse_acc, prune_acc, finetune_acc))
            np.save("outs/res.npy", outs)