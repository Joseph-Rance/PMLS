"""CIFAR-100 classification."""

import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import CIFAR100

from resnext import resnext50

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

aug_transform = transforms.Compose([  # we may only augment data from previous classes
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100("./data", train=True, transform=other_transform, download=True)
test_dataset = CIFAR100("./data", train=False, transform=other_transform, download=True)
combined_dataset = ConcatDataset([train_dataset, test_dataset])

model = resnext50().to(DEVICE)  # works better than the pytorch 32x4d version
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

FINETUNE_STEP = 20
# MUTLIPLIER ^ FINETUNE_STEP - 70000 * MUTLIPLIER + 70000 - 1 = 0
MUTLIPLIER = 1.71817
finetune_sizes = [0] + [round(MUTLIPLIER**i) for i in range(FINETUNE_STEP)]
finetune_sizes[-1] = 70_000

def train(group_dataset):

    group_loader = DataLoader(group_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True, pin_memory=True)

    optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimiser, factor=0.2, patience=5, threshold=0.1, threshold_mode="rel")

    lrs, loss, top1, top5, norms = [], [], [], [], []

    model.train()

    for _epoch in trange(MAX_EPOCHS, leave=False):

        norm_epoch = loss_epoch = top1_epoch = top5_epoch = num_examples = 0

        train_loader_bar = tqdm(group_loader, leave=False, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]",
                                                            postfix="lr: ?, loss: ?, acc@1: ?%")
        for x, y in train_loader_bar:

            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_batch = cross_entropy(z, y)

            loss_batch.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()

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

        scheduler.step(loss_epoch)
        if optimiser.param_groups[0]['lr'] < 0.0005:
            break  # quit when we have converged
    
    return (lrs, loss, top1, top5, norms, num_examples)

@torch.no_grad()
def test(dataset, tqdm_subst=lambda x: x):

    loader = DataLoader(dataset, batch_size=1024)

    model.eval()
    loss = top1 = top5 = num_examples = 0

    for x, y in tqdm_subst(loader):

        x, y = x.to(DEVICE), y.to(DEVICE)

        z = model(x)
        loss += cross_entropy(z, y, reduction="sum")

        num_examples += x.size()[0]

        top = z.topk(5, 1, sorted=True).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y.view(-1, 1)).sum().item()

    return loss, (top1, top5), num_examples


if __name__ == "__main__":

    print("[training]")

    losses = []
    full_res = []
    for i in range(FINETUNE_STEP):

        # don't reset the model!
        train_res = train(Subset(combined_dataset, range(finetune_sizes[i])))

        new_data = Subset(combined_dataset, range(finetune_sizes[i], finetune_sizes[i+1]))

        loss, (top1, top5), num_examples = test(new_data)

        full_res.append((train_res, (loss.cpu(), (top1, top5), num_examples)))
        losses.append(loss.cpu())
        np.save(f"outs/information_content_metrics.npy", np.array(full_res, dtype=object), allow_pickle=True)
        np.save(f"outs/losses.npy", losses)
        torch.save(model.state_dict(), f"outs/models/model_{i}.pt")