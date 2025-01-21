"""Compute the information content of each class in the CIFAR-100 task.

To compute the information content of the whole task, set
CLASS_STEP to 100, and adjust the BATCH_STEP accordingly

"""

import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.nn.functional import cross_entropy
from torchvision import transforms
from torchvision.datasets import CIFAR100

from resnext import resnext50

import warnings
warnings.filterwarnings("ignore", message="^.*epoch parameter in `scheduler.step()` was not necessary.*$")

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
MAX_EPOCHS = 100  # we quit training when loss stagnates, so rarely hit this value

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_float32_matmul_precision('high')

# we only want to augment data that is not on the current class we are learning, because augmentation
# introduces information about the current class that we do not account for in our calculation ...
aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# ... therefore we use this transform for the current class
other_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# we don't need a test dataset, because we only care about information at the moment, not generalisation
# therefore we may as well train on the test dataset as well
# don't transform data here because we don't know if we want to augment yet. This is done later.
train_dataset = CIFAR100("./data", train=True, transform=None, download=True)
test_dataset = CIFAR100("./data", train=False, transform=None, download=True)
combined_dataset = ConcatDataset([train_dataset, test_dataset])

CLASS_STEP = 10  # we compute information content every CLASS_STEP classes
assert 60_000 % CLASS_STEP == 0  # necessary for implementation but not in general
classes = [[] for __ in range(100//CLASS_STEP)]

for i, (_x, y) in enumerate(combined_dataset):
    classes[y//CLASS_STEP].append(i)

BATCH_STEP = 1000  # we retrain the model every BATCH_STEP data points
assert (60_000 // CLASS_STEP) % BATCH_STEP == 0  # necessary for implementation but not in general

# A. uncomment below for constant step size

for cg, __ in enumerate(classes):
    random.shuffle(classes[cg])
    classes[cg] = np.array(classes[cg]).reshape((-1, BATCH_STEP)).tolist()

# B. uncomment below for variable step size
#
#FINETUNE_STEP = 10
#MULTIPLIER = 1.87
#
## we finetune on smaller datasets initially because we want to train the model
## more frequently to improve accuracy as fast as possible
#finetune_sizes = [round(MULTIPLIER**i) for i in range(FINETUNE_STEP)]
#
#for cg, __ in enumerate(classes):
#    random.shuffle(classes[cg])
#    split = classes[cg]
#    classes[cg] = [[classes[cg][0]]]
#
#    for i in range(1, FINETUNE_STEP):
#        finetune_sizes[i] = min(600*CLASS_STEP, finetune_sizes[i] + finetune_sizes[i-1])
#        classes[cg].append(split[finetune_sizes[i-1]:finetune_sizes[i]])

model = resnext50().to(DEVICE)  # this implementation works better than the pytorch 32x4d version for some reason?
model = torch.compile(model)
torch.backends.cudnn.benchmark = True

# this allows us to index a subset of our dataset, while specifying which datapoints to augment and which not to
# this implementation is very similar to torch.utils.data.Subset
class SubsetAugment(Dataset):

    def __init__(self, dataset, aug_indices, other_indices, aug_transform, other_transform) -> None:
        self.dataset = dataset
        # save tuples of index and pointer to the transform it uses
        self.indices = [(i, aug_transform) for i in aug_indices] + [(i, other_transform) for i in other_indices]

        random.shuffle(self.indices)

    def __getitem__(self, idx):

        if isinstance(idx, list):
            return self.__getitems__(idx)

        i, transform = self.indices[idx]
        return transform(*self.dataset[i])

    def __getitems__(self, indices):

        if callable(getattr(self.dataset, "__getitems__", None)):
            vals = self.dataset.__getitems__([self.indices[idx][0] for idx in indices])
        else:
            vals = [self.dataset[self.indices[idx][0]] for idx in indices]

        transforms = [self.indices[idx][1] for idx in indices]
        for i, __ in enumerate(vals):
            vals[i] = transforms[i](*vals[i])
        return vals

    def __len__(self):
        return len(self.indices)

def train(pretrain_data, finetune_data):

    group_dataset = SubsetAugment(combined_dataset, pretrain_data, finetune_data, lambda x,y: (aug_transform(x), y), lambda x,y: (other_transform(x), y))
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
def test(data, tqdm_subst=lambda x: x):

    dataset = SubsetAugment(combined_dataset, [], data, None, lambda x,y: (other_transform(x), y))
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
    pretrain_data = []

    # we compute the information content of each class, given the previous classes already computed
    # this means we do not need to reset the model on each iteration
    for cg in trange(len(classes)):

        finetune_data = []  # indexes of data we have seen of the current class

        # we iterate over groups of indexes pointing to data in class cg
        for bg in trange(len(classes[cg]), leave=False):

            # don't reset the model!

            # first train the model using the data from previous classes (pretrain_data), and the
            # data we have seen so far from this class (finetune_data)
            train_res = train(pretrain_data, finetune_data)

            # then get the next group of data from this class. If we are using constant step size (A, above), these sets are always
            # the same size. If we are using variable step size (B), these get bigger as we see more data from the class
            new_data = classes[cg][bg]

            # test the information content (loss) of the new data
            loss, (top1, top5), num_examples = test(new_data)

            full_res.append((train_res, (loss.cpu(), (top1, top5), num_examples)))
            losses.append(loss.cpu())
            np.save(f"outs/information_content_metrics.npy", np.array(full_res, dtype=object), allow_pickle=True)
            np.save(f"outs/losses.npy", losses)
            torch.save(model.state_dict(), f"outs/models/model_{cg}_{bg}.pt")

            # update dataset with the new data
            finetune_data += new_data  # not super efficient but dataset is relatively small so this is probably ok

        # once we have seen a full class, add it to the pretrain data for the next class
        pretrain_data += finetune_data
