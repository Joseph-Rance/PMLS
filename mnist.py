"""MNIST classification."""

from math import ceil
import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import MNIST

from cnn import CNN

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
TRAINING_EPOCHS = 200

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

transform = transforms.Compose([
#    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST("./data", train=True, transform=transform, download=True)
# skip val for now
test_dataset = MNIST("./data", train=False, transform=transform, download=True)

# no need for dataloaders - we can just load the full dataset into GPU memory!
# usually dataloaders shuffle on every epoch which we are going to lose here but that doesn't really matter
# we can't augment the data every easily this way, but that also doesn't matter much

train_inputs, test_inputs = [], []
train_labels, test_labels = [], []

for x, y in train_dataset:
    train_inputs.append(x)
    train_labels.append(y)

for x, y in test_dataset:
    test_inputs.append(x)
    test_labels.append(y)

train_inputs = torch.cat(train_inputs, dim=0).reshape((60_000, 1, 28, 28))
train_labels = torch.tensor(train_labels)
test_inputs = torch.cat(test_inputs, dim=0).reshape((10_000, 1, 28, 28))
test_labels = torch.tensor(test_labels)

model = CNN().to(DEVICE)
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# note that this scheduler is called *per batch*
scheduler = ReduceLROnPlateau(optimiser, factor=0.2, patience=300, threshold=0.1, threshold_mode="rel")

@torch.no_grad()
def test(dataset, tqdm_subst=lambda x: x):

    model.eval()
    loss = top1 = top5 = num_examples = 0

    for idx in tqdm_subst(range(ceil(len(train_inputs) / BATCH_SIZE))):

        x = dataset[0][idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
        y = dataset[1][idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]

        z = model(x)
        loss += cross_entropy(z, y)

        num_examples += x.size()[0]

        top = z.topk(5, 1, sorted=True).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y.view(-1, 1)).sum().item()

    return loss, (top1, top5), num_examples


if __name__ == "__main__":

    print("[training]")

    loss_trains, loss_vals = [], []
    top1_trains, top1_vals = [], []
    top5_trains, top5_vals = [], []
    lrs, norms = [], []

    train_inputs = train_inputs.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    test_inputs = test_inputs.to(DEVICE)
    test_labels = test_labels.to(DEVICE)

    loss_val, (top1_val, top5_val), num_val_examples = test((test_inputs, test_labels))

    for epoch in trange(TRAINING_EPOCHS):

        model.train()
        loss_train = top1_train = top5_train = num_train_examples = norm_epoch = 0

        train_loader_bar = trange(ceil(len(train_inputs) / BATCH_SIZE), leave=False,
                                  bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]",
                                  postfix="lr: ?, loss: ?, train@1: ?%, val@1*: ?%")
        for idx in train_loader_bar:

            x = train_inputs[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]
            y = train_labels[idx*BATCH_SIZE : (idx+1)*BATCH_SIZE]

            # TODO: does this help??
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_train_batch = cross_entropy(z, y)

            loss_train_batch.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step(loss_train_batch)  # TODO: this has high variance

            with torch.no_grad():

                loss_train += loss_train_batch.item()
                norm_epoch += norm.cpu()

                top = z.topk(5, 1, sorted=True).indices
                top1_train += (top[:, 0] == y).sum().item()
                top5_train += (top == y.view(-1, 1)).sum().item()

                num_train_examples += x.size()[0]

                train_loader_bar.set_description_str(f"lr: .{int(optimiser.param_groups[0]['lr']*1000):0>3}, loss: {loss_train_batch.item():2.2f}, " \
                                                     f"train@1: {top1_train * 100 / max(1, num_train_examples):3.1f}%, " \
                                                     f"val@1*: {top1_val * 100 / max(1, num_val_examples):3.1f}%")
                train_loader_bar.refresh()

        loss_val, (top1_val, top5_val), num_val_examples = test((test_inputs, test_labels))

        loss_trains.append(loss_train)
        top1_trains.append(top1_train)
        top5_trains.append(top5_train)
        loss_vals.append(loss_val.item())
        top1_vals.append(top1_val)
        top5_vals.append(top5_val)
        lrs.append(optimiser.param_groups[0]['lr'])
        norms.append(norm_epoch)

        np.save(f"outs/train_metrics_{num_train_examples}_{num_val_examples}.npy",
                (lrs, loss_trains, top1_trains, top5_trains, loss_vals, top1_vals, top5_vals, norms))
        torch.save(model.state_dict(), "outs/model.pt")

        if optimiser.param_groups[0]['lr'] < 0.0005:
            break  # quit when we have converged

    print("[testing]")

    _loss, (top1_test, top5_test), num_test_examples = test((test_inputs, test_labels), tqdm_subst=tqdm)

    print(f"top1: {top1_test * 100 / num_test_examples:3.3f}; " \
          f"top5: {top5_test * 100 / num_test_examples:3.3f}")