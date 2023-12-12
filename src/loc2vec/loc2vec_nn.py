from loc2vec.config import Params

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from tqdm import tqdm

class Network(torch.nn.Module):
    def __init__(self, in_channels, debug=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if debug:
            self.debug = debug
            self.dropout = nn.Dropout2d(0.5)
            self.relu = nn.ReLU()
            self.leak_relu = nn.LeakyReLU()
            self.relu_p = nn.PReLU()
            self.pool = nn.MaxPool2d(2, stride=2, padding=0)
            self.conv1 = nn.Conv2d(in_channels, 64, 1, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 1, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 64, 3, stride=2, padding=1)
            self.conv5 = nn.Conv2d(64, 32, 3, stride=2, padding=1)
            self.conv6 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.fc1 = nn.Linear(1024, 16)
        else:
            self.model = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels, 64, 1, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.PReLU(),
            nn.Linear(8192, 16)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for network

        Parameters
        ----------
        x: torch.Tensor
            Tensor for model input
        
        Returns
        -------
        x: torch.Tensor
            Tensor for model output
        """
        if self.debug:
            x = self.dropout(x)
            x = self.conv1(x)
            x = self.leak_relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.pool(x)
            x = self.leak_relu(x)
            x = self.conv4(x)
            x = self.pool(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.leak_relu(x)
            x = self.conv6(x)
            x = self.relu_p(x)
            print(x.shape)
            x = x.view(x.size(0), -1)
            print(x.shape)
            x = self.fc1(x)
            return x
        else:
            return self.model(x)

# BOP IT

class Loc2vec(nn.Module):
    """
    Using pretrained ResNet50

    num_ftrs is taken from surya501's loc2vec implementation
    """
    def __init__(self):
        super(Loc2vec, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        num_ftrs = 18432
        self.model.fc = nn.Linear(num_ftrs, 16)

    def forward(self, x): 
        x = self.model(x)
        return x

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
        distance_a_pos = F.pairwise_distance(anchor, anchor_pos)
        distance_a_neg = F.pairwise_distance(anchor, anchor_neg)
        distance_pos_neg = F.pairwise_distance(anchor_pos, anchor_neg)
        distance_min_neg = torch.min(distance_a_neg, distance_pos_neg)
        losses = F.relu(distance_a_pos - distance_min_neg + self.margin)
        
        np_losses = losses.cpu().data.numpy()
        np_distance_a_pos = np.mean(distance_a_pos.cpu().data.numpy())
        np_distance_a_neg = np.mean(distance_a_neg.cpu().data.numpy())
        np_min_neg_dist = np.mean(distance_min_neg.cpu().data.numpy())

        loss_log = f'MAX LOSS: {round(float(np.max(np_losses)),3)} | MEAN LOSS: {round(float(np.mean(np_losses)),3)} | (o)/(+) DIST: {round(float(np.mean(np_distance_a_pos)),3)} | (o)/(-) DIST: {round(float(np.mean(np_distance_a_neg)),3)}'

        return losses.mean(), loss_log, np_distance_a_pos, np_distance_a_neg, np_min_neg_dist