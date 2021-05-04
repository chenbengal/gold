import pandas as pd
from tqdm import tqdm
# import tensorflow as tf
from torch import optim as opt
import torch.nn as nn
import torch

# PyTorch itself

# Dataset - the base class to be inherited
from torch.utils.data import Dataset, DataLoader
# We will need DataLoader later for the training process

class CPUDataset(Dataset):
    def __init__(self, data: pd.DataFrame, size: int,
                 step: int = 1):
        self.chunks = torch.FloatTensor(data['Price']).unfold(0, size + 1, step)
        self.chunks = self.chunks.view(-1, 1, size + 1)

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, i):
        x = self.chunks[i, :, :-1]
        y = self.chunks[i, :, -1:].squeeze(1)
        return x, y



def conv_layer(in_feat, out_feat, kernel_size=3, stride=1,
               padding=1, relu=True):
    res = [
        nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_feat),
    ]
    if relu:
        res.append(nn.ReLU())
    return nn.Sequential(*res)


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat, self.out_feat = in_feat, out_feat
        self.conv1 = conv_layer(in_feat, out_feat)
        self.conv2 = conv_layer(out_feat, out_feat, relu=False)
        if self.apply_shortcut:
            self.shortcut = conv_layer(in_feat, out_feat,
                                       kernel_size=1, padding=0,
                                       relu=False)

    def forward(self, x):
        out = self.conv1(x)
        if self.apply_shortcut:
            x = self.shortcut(x)
        return x + self.conv2(out)

    @property
    def apply_shortcut(self):
        return self.in_feat != self.out_feat


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class CNN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.base = nn.Sequential(
            ResBlock(1, 8),  # shape = batch, 8, n_factors
            ResBlock(8, 8),
            ResBlock(8, 16),  # shape = batch, 16, n_factors
            ResBlock(16, 16),
            ResBlock(16, 32),  # shape = batch, 32, n_factors
            ResBlock(32, 32),
            ResBlock(32, 64),  # shape = batch, 64, n_factors
            ResBlock(64, 64),
        )
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),  # shape = batch, 128, 1
            nn.Flatten(),
            nn.Linear(128, out_size)
        )

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


def train_model(model: CNN, dataloaders: dict, optimizer: opt.Optimizer,
                scheduler, criterion, device: torch.device, epochs: int):
    losses_data = {'train': [], 'valid': []}
    model.to(device)

    # Loop over epochs
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch}/{epochs - 1}')

        # Training and validation phases
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_total = 0.

            # Loop over batches of data
            for idx, batch in tqdm(enumerate(dataloaders[phase]),
                                   total=len(dataloaders[phase]),
                                   leave=False
                                   ):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    loss = criterion(out, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)

            epoch_loss = running_loss / running_total
            print(f'{phase.capitalize()} Loss: {epoch_loss}')
            losses_data[phase].append(epoch_loss)
    return losses_data