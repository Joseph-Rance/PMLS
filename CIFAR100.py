"""CIFAR-100 classification."""

import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import CIFAR100

from resnext import resnext50

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
WARMUP_EPOCHS = 1
TRAINING_EPOCHS = 200

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

train_transform = transforms.Compose([
    #transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
val_loader = test_loader

#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=100).to(DEVICE)

model = resnext50().to(DEVICE)  # works better than the pytorch 32x4d version
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

optimiser = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

WARMUP_ITERS = WARMUP_EPOCHS*len(train_loader)
warmup_scheduler = LinearLR(optimiser, start_factor=1/(WARMUP_ITERS+1), total_iters=WARMUP_ITERS)

#TRAINING_EPOCHS = 175
#END_LR_SCALE = 0.25
#restarts_scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=25, T_mult=2)
#decay_scheduler = ExponentialLR(optimiser, gamma=END_LR_SCALE**(1/TRAINING_EPOCHS))
#main_scheduler = ChainedScheduler([restarts_scheduler, decay_scheduler], optimizer=optimiser)

main_scheduler = MultiStepLR(optimiser, milestones=[60, 120, 180], gamma=0.2)


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


if __name__ == "__main__":

    print("[training]")

    loss_trains, loss_vals = [], []
    top1_trains, top1_vals = [], []
    top5_trains, top5_vals = [], []
    lrs, norms = [], []

    loss_val, (top1_val, top5_val), num_val_examples = test(val_loader)

    for epoch in trange(WARMUP_EPOCHS + TRAINING_EPOCHS):

        model.train()
        loss_train = top1_train = top5_train = num_train_examples = norm_epoch = 0

        train_loader_bar = tqdm(train_loader, leave=False, bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]",
                                                           postfix="lr: ?, loss: ?, train@1: ?%, val@1*: ?%")
        for x, y in train_loader_bar:

            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):

                z = model(x)
                loss_train_batch = cross_entropy(z, y)

            loss_train_batch.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()

            if epoch < WARMUP_EPOCHS:  # warmup per batch for the first epoch
                warmup_scheduler.step()

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

        loss_val, (top1_val, top5_val), num_val_examples = test(val_loader)

        loss_trains.append(loss_train)
        top1_trains.append(top1_train)
        top5_trains.append(top5_train)
        loss_vals.append(loss_val.item())
        top1_vals.append(top1_val)
        top5_vals.append(top5_val)
        lrs.append(optimiser.param_groups[0]['lr'])
        norms.append(norm_epoch)

        if epoch >= WARMUP_EPOCHS:  # skip warmup epoch
            main_scheduler.step()

        np.save(f"outs/train_metrics_{num_train_examples}_{num_val_examples}.npy",
                (lrs, loss_trains, top1_trains, top5_trains, loss_vals, top1_vals, top5_vals, norms))
        torch.save(model.state_dict(), "outs/model.pt")

    print("[testing]")

    _loss, (top1_test, top5_test), num_test_examples = test(test_loader, tqdm_subst=tqdm)

    print(f"top1: {top1_test * 100 / num_test_examples:3.3f}; " \
          f"top5: {top5_test * 100 / num_test_examples:3.3f}")