from loc2vec.config import Params

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

class Network(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(36, 64, 1, stride=1, padding=0),
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
        
        Return
        ------
        torch.Tensor
            A tensor representing the euclidean distance between two vectors
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
        
        Returns
        -------
        loss: torch.Tensor
            Tensor output from the Triplet Loss Funciton
        """
        distance_to_pos = self.calc_euclidean(anchor, anchor_pos)
        distance_to_neg = self.calc_euclidean(anchor, anchor_neg)
        losses = torch.relu(distance_to_pos - distance_to_neg + self.margin)
        return losses.mean()