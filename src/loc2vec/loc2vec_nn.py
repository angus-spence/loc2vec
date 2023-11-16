from loc2vec.data_loader import Data_Loader
from loc2vec.data_loader import Data_Loader
from loc2vec.config import Params

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

loader = Data_Loader(
    x_path=Params.X_PATH.value,
    x_pos_path=Params.X_POS_PATH.value,
    batch_size=60,
    shuffle=False
)

data = loader.load_from_dirs()

class Network(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(3, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)

class TripletLossFunction(nn.Module):
    """
    Evaluates triplet loss
    
    Parameters
    ----------
    margin: float
        margin for use in loss function
    """
    def __init__(self, margin=1.0):
        super(TripletLossFunction, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the euclidean distance between two vectors in the embedding space

        Parameters
        ----------
        x1: torch.Tensor
            Tensor 1
        x2: torch.Tensor
            Tensor 2
        
        
        """
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, anchor_pos: torch.Tensor, anchor_neg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for nn.Module

        Parameters
        ----------
        anchor: torch.Tensor
            Tensor for the x[i] sample
        anchor_pos: torch.Tensor
            Tensor for (+) anchor for x[i]
        anchor_neg: torch.Tensor
            Tensor for (-) anchor for x[i]
        """
        distance_to_pos = self.calc_euclidean(anchor, anchor_pos)
        distance_to_neg = self.calc_euclidean(anchor, anchor_neg)
        losses = torch.relu(distance_to_pos - distance_to_neg + self.margin)
        return losses.mean()
    
model = Network()
optimiser = torch.optim.Adam(model.parameters(), lr=Params.LEARNING_RATE.value)
criterion = TripletLossFunction()

for epoch in tqdm(range(Params.EPOCHS.value), desc='Epochs'):
    running_loss = []
    for step, (anchor, anchor_pos, anchor_neg) in enumerate(tqdm(data, desc="Training", leave=False)):
        x = anchor
        x_pos = anchor_pos
        x_neg = anchor_neg

        x_out = model(x)
        x_pos_out = model(x_pos)
        x_neg_out = model(x_neg)

        loss = criterion(x_out, x_pos_out, x_neg_out)
        loss.backward()
        optimiser.step()

        running_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch: {epoch+1}/{Params.EPOCHS.value} - Loss: {round(np.mean(running_loss), 5)}')